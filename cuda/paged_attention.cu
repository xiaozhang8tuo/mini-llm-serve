// PagedAttention 的 CUDA kernel，照着 vLLM 论文的思路写的。
// 里面有两个 kernel:
//   1. store_kv_kernel —— 把新 token 的 K/V 按 block table 寻址写入分页 cache
//   2. paged_attention_kernel —— 遍历 block table 读 KV，做 online softmax attention
//
// 编译和运行说明见同目录下的 build_and_run.txt。
// 这文件完全独立，不依赖 vLLM 也不依赖任何其他库，只需要 CUDA toolkit。

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// ---- 常量，和 include/block.h 里的保持一致 ----
constexpr int BLOCK_SIZE         = 16;     // 每个物理块能装几个 token
constexpr int NUM_HEADS          = 32;
constexpr int HEAD_DIM           = 128;
constexpr int MAX_BLOCKS         = 1024;   // 物理块总共多少个
constexpr int MAX_BLOCKS_PER_SEQ = 128;    // 一个序列最多映射多少逻辑块

// 把四维坐标 (phy, head, tok, dim) 摊平成一维偏移。
// KV cache 在显存里是一维数组，但逻辑上是四维的:
//   key_cache[phy][head][tok][dim]
// = key_cache[phy * (H*B*D) + head * (B*D) + tok * D + dim]
//
// 这段代码在 GPU 上跑（__device__），用来算线程要读/写的具体地址。
__device__ inline int kv_offset(int phy, int head, int tok, int dim) {
    return phy * (NUM_HEADS * BLOCK_SIZE * HEAD_DIM)
         + head * (BLOCK_SIZE * HEAD_DIM)
         + tok  * HEAD_DIM
         + dim;
}

// ================================================================
//  Kernel 1: store_kv_kernel
//  ================================================================
//  把刚算出来的一批 K_new / V_new 存进 paged KV cache。
//
//  launch config: <<<num_seqs, 512>>>
//    - grid 里每个 block 处理一个序列的一个新 token
//    - 512 个线程用 stride loop 铺满 32 × 128 = 4096 个元素
//      （32 heads × 128 dims），每个线程处理 4096/512 = 8 个元素
//
//  关键步骤: token_idx → 逻辑块 → 页表翻译 → 物理块 → 写入。
//  ★ 最关键的地方就是 block_tables 的寻址——查页表，逻辑块转物理块。
//
//  参数:
//    k_new, v_new: 当前 token 所有 head 的 K/V，扁平成 [NUM_HEADS * HEAD_DIM]
//    key_cache, value_cache: 物理块池，一维数组，存在 HBM 上
//    block_tables: 页表，每个序列各占 MAX_BLOCKS_PER_SEQ 个 int
//    token_idx: 当前 token 在整个序列中的位置（从 0 数起）
__global__ void store_kv_kernel(
    const half* __restrict__ k_new,
    const half* __restrict__ v_new,
    half* __restrict__ key_cache,
    half* __restrict__ value_cache,
    const int* __restrict__ block_tables,
    int token_idx
) {
    int seq_id = blockIdx.x;  // 一个 block 处理一个序列

    // 算出这个 token 落在哪个逻辑块、块内偏移、对应哪个物理块
    int lb  = token_idx / BLOCK_SIZE;
    int off = token_idx % BLOCK_SIZE;
    int phy = block_tables[seq_id * MAX_BLOCKS_PER_SEQ + lb]; // ★ 页表查表

    // stride loop: 总共 NUM_HEADS * HEAD_DIM = 4096 个元素，
    // 512 个线程，每个线程负责 4096/512 = 8 个元素。
    // 相邻线程访问相邻元素 → coalesced memory access。
    int tid          = threadIdx.x;
    int total_elems  = NUM_HEADS * HEAD_DIM;

    for (int i = tid; i < total_elems; i += blockDim.x) {
        int h   = i / HEAD_DIM;   // 第几个 head
        int dim = i % HEAD_DIM;   // head 内的第几维

        int dst = kv_offset(phy, h, off, dim);

        key_cache[dst]   = k_new[i];
        value_cache[dst] = v_new[i];
    }
}

// ================================================================
//  Kernel 2: paged_attention_kernel
//  ================================================================
//  对每个 (seq_id, head_idx) 对做一次 attention。遍历该序列的 block table，
//  把所有 KV 读出来和 Q 做 scaled dot-product attention。
//
//  launch config: <<<(num_seqs, NUM_HEADS), 256>>>
//   - grid.x = 序列 id, grid.y = head 编号
//   - 每个 GPU block 的 256 个线程一起算一个 (序列, head) 的输出
//
//  算法: online softmax —— flash attention 里的那种，不存完整 score 矩阵。
//  流程:
//    for 每个逻辑块 (通过 block table 拿到物理块):
//      for 块内每个 token:
//        从 HBM 读 K[t] → 放到 shared memory
//        算 score = q · k[t] / sqrt(d)
//        online softmax: 更新 max, rescale 旧结果, 累加 weight * V[t]
//    out = acc / sum_w
//
//  性能方面的考虑:
//    - Q 放在寄存器里（每个线程一个 scalar of Q）
//    - weighted sum 的累加器放在 shared memory 里（s_acc），
//      避免反复读写 HBM
//    - 每个 token 的 K 和 V 只从 HBM 读一次，读的时候
//      相邻线程访问相邻地址 → coalesced
__global__ void paged_attention_kernel(
    half* __restrict__ out,
    const half* __restrict__ q,
    const half* __restrict__ key_cache,
    const half* __restrict__ value_cache,
    const int* __restrict__ block_tables,
    const int* __restrict__ context_lens,
    float sm_scale
) {
    int seq_id   = blockIdx.x;
    int head_idx = blockIdx.y;

    int ctx_len  = context_lens[seq_id];
    if (ctx_len <= 0) return;

    int n_blocks = (ctx_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int tid      = threadIdx.x;

    // 从 HBM 把 Q 的当前 head 读到寄存器（每个线程读自己负责的那一维）
    half q_reg = (tid < HEAD_DIM) ? q[seq_id * NUM_HEADS * HEAD_DIM
                                    + head_idx * HEAD_DIM
                                    + tid] : __float2half(0.0f);

    // shared memory 分配:
    //   s_max, s_sum: online softmax 的状态（scalar，只有第一个线程写）
    //   s_acc: 加权 V 的累加器，HEAD_DIM 个 float → 128×4 = 512 bytes
    //   s_scores: 暂存每个 token 的 score，BLOCK_SIZE 个 float（后面做 softmax 用不到这里，先留着）
    __shared__ float s_max;
    __shared__ float s_sum;
    __shared__ float s_acc[HEAD_DIM];
    __shared__ float s_scores[BLOCK_SIZE];

    if (tid == 0) { s_max = -INFINITY; s_sum = 0.0f; }
    if (tid < HEAD_DIM) { s_acc[tid] = 0.0f; }
    __syncthreads();

    // 遍历每个逻辑块，通过 block table 找到物理块，读里面的 token
    for (int b = 0; b < n_blocks; b++) {
        int phy = block_tables[seq_id * MAX_BLOCKS_PER_SEQ + b]; // ★ 页表翻译
        // 最后一块可能不满——最后一个 block 里的 token 数单独算
        int n_tok = (b < n_blocks - 1) ? BLOCK_SIZE
                                       : (ctx_len - b * BLOCK_SIZE);

        for (int t = 0; t < n_tok; t++) {
            // 从 key_cache 读 K[t] 的当前 head 的一维
            half k_val = __float2half(0.0f);
            if (tid < HEAD_DIM) {
                int k_off = kv_offset(phy, head_idx, t, tid);
                k_val = key_cache[k_off];
            }

            // q·k: 先每个线程乘自己那维，然后 tree reduction 求和
            __shared__ float s_partial[BLOCK_SIZE];
            s_partial[tid] = __half2float(q_reg) * __half2float(k_val);
            __syncthreads();

            // 树形归约: 把 HEAD_DIM 个线程的部分积加起来 = 点积
            for (int stride = HEAD_DIM / 2; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    s_partial[tid] += s_partial[tid + stride];
                }
                __syncthreads();
            }

            float score = s_partial[0] * sm_scale;  // scale by 1/√d
            s_scores[t] = score;
            __syncthreads();

            // online softmax 更新:
            // 如果当前 score 比之前见过的都大，需要对之前的累加结果 rescale
            if (score > s_max) {
                float rescale = expf(s_max - score);
                for (int d = tid; d < HEAD_DIM; d += blockDim.x) {
                    s_acc[d] *= rescale;
                }
                __syncthreads();
                s_sum *= rescale;
                s_max = score;
            }

            float weight = expf(score - s_max);
            if (tid == 0) {
                s_sum += weight;
            }

            // 读 V[t] 并累加 weight * V[t]
            half v_val = __float2half(0.0f);
            if (tid < HEAD_DIM) {
                int v_off = kv_offset(phy, head_idx, t, tid);
                v_val = value_cache[v_off];
            }

            if (tid < HEAD_DIM) {
                s_acc[tid] += weight * __half2float(v_val);
            }
            __syncthreads();
        }
    }

    // 归一化: out = acc / sum_w，写回 HBM
    if (tid < HEAD_DIM) {
        int out_off = seq_id * NUM_HEADS * HEAD_DIM
                    + head_idx * HEAD_DIM
                    + tid;
        out[out_off] = __float2half(s_acc[tid] / s_sum);
    }
}

// ================================================================
//  Host 端演示: 造 3 个序列的 mock 数据，手工搭 block table，
//  填初始 KV，调 store_kv 追加新 token，调 paged_attention 验证输出。
// ================================================================
int main() {
    const int num_seqs = 3;
    int context_lens[num_seqs] = {35, 20, 60};

    // ---- 分 device memory ----
    half *d_kv_cache, *d_q, *d_out;
    int  *d_block_tables, *d_context_lens;

    // key 和 value 放在同一块 device memory 里，前一半 key 后一半 value
    size_t kv_bytes = MAX_BLOCKS * NUM_HEADS * BLOCK_SIZE * HEAD_DIM * sizeof(half);
    cudaMalloc(&d_kv_cache, kv_bytes * 2);
    half* d_value_cache = d_kv_cache + (kv_bytes / sizeof(half));

    cudaMalloc(&d_q,            num_seqs * NUM_HEADS * HEAD_DIM * sizeof(half));
    cudaMalloc(&d_out,          num_seqs * NUM_HEADS * HEAD_DIM * sizeof(half));
    cudaMalloc(&d_block_tables,  num_seqs * MAX_BLOCKS_PER_SEQ * sizeof(int));
    cudaMalloc(&d_context_lens, num_seqs * sizeof(int));

    cudaMemset(d_kv_cache,     0, kv_bytes * 2);
    cudaMemset(d_q,            0, num_seqs * NUM_HEADS * HEAD_DIM * sizeof(half));
    cudaMemcpy(d_context_lens, context_lens, num_seqs * sizeof(int), cudaMemcpyHostToDevice);

    // ---- 构造 block tables（物理块号故意不连续，验证页表寻址） ----
    int h_block_tables[num_seqs * MAX_BLOCKS_PER_SEQ];
    memset(h_block_tables, -1, sizeof(h_block_tables));
    h_block_tables[0 * MAX_BLOCKS_PER_SEQ + 0] = 3;
    h_block_tables[0 * MAX_BLOCKS_PER_SEQ + 1] = 7;
    h_block_tables[0 * MAX_BLOCKS_PER_SEQ + 2] = 12;
    h_block_tables[1 * MAX_BLOCKS_PER_SEQ + 0] = 1;
    h_block_tables[1 * MAX_BLOCKS_PER_SEQ + 1] = 5;
    h_block_tables[2 * MAX_BLOCKS_PER_SEQ + 0] = 8;   // 注意 seq2 有四块，
    h_block_tables[2 * MAX_BLOCKS_PER_SEQ + 1] = 4;   // 物理块 8、4、0、15
    h_block_tables[2 * MAX_BLOCKS_PER_SEQ + 2] = 0;
    h_block_tables[2 * MAX_BLOCKS_PER_SEQ + 3] = 15;
    cudaMemcpy(d_block_tables, h_block_tables, sizeof(h_block_tables), cudaMemcpyHostToDevice);

    // ---- 模拟 prefill: 给 seq0 head=0 填初始 KV ----
    // 先用 host 端临时 buffer 算好 float 值，转 half，再一把 cudaMemcpy 拷到 device。
    // 不能直接 ((half*)device_ptr)[i] = value —— 那是 device 端地址，
    // host 端解引用会 segfault。
    half h_one = __float2half(0.1f);
    float* temp = (float*)malloc(MAX_BLOCKS * NUM_HEADS * BLOCK_SIZE * HEAD_DIM * sizeof(float));
    for (int i = 0; i < MAX_BLOCKS * NUM_HEADS * BLOCK_SIZE * HEAD_DIM; i++) temp[i] = 0.0f;

    // seq0 的 token 0~34: KV 值 = 全局 token 索引 × 0.1
    int blocks_0[] = {3, 7, 12};
    for (int t_global = 0; t_global < 35; t_global++) {
        int lb = t_global / BLOCK_SIZE;
        int off = t_global % BLOCK_SIZE;
        int phy = blocks_0[lb];
        int base = phy * (NUM_HEADS * BLOCK_SIZE * HEAD_DIM)
                 + 0 * (BLOCK_SIZE * HEAD_DIM)    // head=0
                 + off * HEAD_DIM;
        for (int d = 0; d < HEAD_DIM; d++) temp[base + d] = t_global * 0.1f;
    }

    // float → half 在 host 端做完再拷上去
    {
        half* h_kv_temp = (half*)malloc(kv_bytes);
        for (int i = 0; i < MAX_BLOCKS * NUM_HEADS * BLOCK_SIZE * HEAD_DIM; i++) {
            h_kv_temp[i] = __float2half(temp[i]);
        }
        cudaMemcpy(d_kv_cache, h_kv_temp, kv_bytes, cudaMemcpyHostToDevice);
        free(h_kv_temp);
    }
    free(temp);

    // ---- 测试 store_kv: 给 seq0 写入第 35 个 token（token_idx=35） ----
    half h_kv[NUM_HEADS * HEAD_DIM];
    for (int i = 0; i < NUM_HEADS * HEAD_DIM; i++) h_kv[i] = __float2half(3.5f);

    half *d_k, *d_v;
    cudaMalloc(&d_k, NUM_HEADS * HEAD_DIM * sizeof(half));
    cudaMalloc(&d_v, NUM_HEADS * HEAD_DIM * sizeof(half));
    cudaMemcpy(d_k, h_kv, NUM_HEADS * HEAD_DIM * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_kv, NUM_HEADS * HEAD_DIM * sizeof(half), cudaMemcpyHostToDevice);

    store_kv_kernel<<<1, 512>>>(d_k, d_v, d_kv_cache, d_value_cache, d_block_tables, 35);
    cudaDeviceSynchronize();
    printf("[store_kv] token_idx=35 written to phy=%d\n",
           h_block_tables[0 * MAX_BLOCKS_PER_SEQ + (35 / BLOCK_SIZE)]);

    // ---- 测试 paged_attention_kernel: 3 个序列 × 32 heads 全算 ----
    float sm_scale = 1.0f / sqrtf((float)HEAD_DIM);

    dim3 grid_attn(num_seqs, NUM_HEADS);
    paged_attention_kernel<<<grid_attn, 256>>>(
        d_out, d_q, d_kv_cache, d_value_cache,
        d_block_tables, d_context_lens, sm_scale);
    cudaDeviceSynchronize();

    // 读回 seq0 head0 的输出前几维查看
    half h_out[HEAD_DIM];
    cudaMemcpy(h_out, d_out, HEAD_DIM * sizeof(half), cudaMemcpyDeviceToHost);
    printf("[paged_attention] seq0 head0 out[0..3]: %.4f %.4f %.4f %.4f\n",
           __half2float(h_out[0]), __half2float(h_out[1]),
           __half2float(h_out[2]), __half2float(h_out[3]));

    // cleanup
    cudaFree(d_kv_cache);
    cudaFree(d_q);
    cudaFree(d_out);
    cudaFree(d_block_tables);
    cudaFree(d_context_lens);
    cudaFree(d_k);
    cudaFree(d_v);

    printf("Done.\n");
    return 0;
}
