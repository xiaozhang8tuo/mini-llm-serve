// PagedAttention 的纯 CPU 实现——不依赖 CUDA，不依赖 GPU。
// 逻辑和 vLLM 的 CUDA kernel 一模一样，只是为了方便理解写成了
// 单线程 CPU 版本。可以直接 g++ 编译跑。
//
// 演示了两个最核心的操作:
//   store_kv:        把新算出来的 K/V 写入分页 KV cache（通过 block table 找地址）
//   paged_attention: 遍历 block table 把 KV 全读出来，做 online softmax attention
//
// 和传统连续 KV cache 的对比——寻址方式不一样:
//   连续: base + seq * max_len * H * D + token * H * D + head * D
//   分页: key_cache[block_tables[seq][token/BLOCK]][head][token%BLOCK][:]
//
// 多了一层 block table 翻译，但换来的是零碎片和按需分配。

#include <cmath>
#include <cstring>
#include <cstdio>

constexpr int BLOCK_SIZE   = 16;
constexpr int NUM_HEADS    = 32;
constexpr int HEAD_DIM     = 128;
constexpr int MAX_BLOCKS   = 256;

// 模拟 GPU 显存里的 KV cache（per layer，这里就一层，真系统有 32 层）
// 两个大数组，一维相当于连续物理块池，通过 block table 翻译来访问。
float key_cache  [MAX_BLOCKS][NUM_HEADS][BLOCK_SIZE][HEAD_DIM];
float value_cache[MAX_BLOCKS][NUM_HEADS][BLOCK_SIZE][HEAD_DIM];

// 每个序列的页表: block_tables[seq][逻辑块] = 物理块号
// context_lens: 序列当前的 context 总长
int   block_tables[32][256 / BLOCK_SIZE];
int   context_lens[32];

// ================================================================
//  store_kv: 把刚算出来的一个新 token 的 K、V 写进 paged cache
//
//  参数:
//    seq       - 序列 id
//    token_idx - 这个 token 在整个序列中的编号（从 0 数起）
//    k_all/v_all - 这个 token 所有 head 的 K/V，扁平成一维数组
//                 大小 [NUM_HEADS * HEAD_DIM]
//
//  寻址过程:
//    1. token_idx / BLOCK_SIZE → 算出在第几个逻辑块
//    2. token_idx % BLOCK_SIZE → 算出块内偏移
//    3. block_tables[seq][逻辑块] → 找出物理块号
//    4. key_cache[物理块][head][偏移][:] → 写到具体位置
//
//  注意 store_kv 和 paged_attention 不是互相调用的关系:
//  在一个 transformer layer 里，执行顺序是:
//    compute QKV → store_kv(K_new, V_new) → paged_attention(Q_new)
//  store_kv 写的是这一层刚算出来的 K/V，
//  paged_attention 读的是包含刚写入的整个 context 的 K/V。
void store_kv(int seq, int token_idx, const float* k_all, const float* v_all) {
  int lb = token_idx / BLOCK_SIZE;   // 第几个逻辑块
  int off = token_idx % BLOCK_SIZE;  // 块内偏移，也就是第几个 token 的位子
  int phy = block_tables[seq][lb];   // ★ 关键: 页表翻译，逻辑→物理

  for (int h = 0; h < NUM_HEADS; h++) {
    float* k_dst = &key_cache[phy][h][off][0];
    float* v_dst = &value_cache[phy][h][off][0];
    memcpy(k_dst, k_all + h * HEAD_DIM, HEAD_DIM * sizeof(float));
    memcpy(v_dst, v_all + h * HEAD_DIM, HEAD_DIM * sizeof(float));
  }
}

// ================================================================
//  paged_attention: 对一个序列的某个 head 做 scaled dot-product attention
//
//  用 online softmax 来算——这是 flash attention 里也用了的技巧，
//  不用先遍历完所有 token 的 score 才能开始算 exp。
//  核心想法: 一边累加一边记录 max_score，发现更大的 score 时
//  对之前的结果做 rescale（乘以 exp(old_max - new_max)）。
//
//  算法流程:
//    遍历 block_table 里的每个物理块，对块内的每个 token:
//      1. s = q · k[t] / sqrt(HEAD_DIM)     ← 拿 q 和当前 token 的 k 做点积
//      2. 如果 s > 之前的 max_score → rescale 之前的累加结果
//      3. w = exp(s - max_score)            ← 当前 token 的 softmax weight
//      4. acc += w * v[t]                   ← 加权累加 V
//    最后: out = acc / sum_w → 归一化
//
//  online softmax 好处: 不用先算完所有 score 再倒回来算 exp，
//  内存只需要一份 acc 和 sum_w，不需要存整个 [ctx_len] 的 score 数组。
void paged_attention(int seq, int head, int ctx_len,
                     const float* q, float* out) {
  memset(out, 0, HEAD_DIM * sizeof(float));

  float max_s = -INFINITY, sum_w = 0;
  float acc[HEAD_DIM] = {0};
  int n_blocks = (ctx_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

  for (int b = 0; b < n_blocks; b++) {
    int phy = block_tables[seq][b];  // ★ 页表翻译
    // 最后一个物理块可能没填满，要单独算里面有几个 token
    int n_tok = (b < n_blocks - 1) ? BLOCK_SIZE : (ctx_len - b * BLOCK_SIZE);

    for (int t = 0; t < n_tok; t++) {
      float* k = &key_cache[phy][head][t][0];
      float* v = &value_cache[phy][head][t][0];

      // q·k 点积
      float s = 0;
      for (int d = 0; d < HEAD_DIM; d++) s += q[d] * k[d];
      s /= sqrtf((float)HEAD_DIM);  // scale，避免 score 太大导致 softmax 梯度消失

      // online softmax 更新
      if (s > max_s) {
        // 发现更大的 score —— 之前的 acc 和 sum_w 都需要 rescale
        float rescale = expf(max_s - s);
        for (int d = 0; d < HEAD_DIM; d++) acc[d] *= rescale;
        sum_w *= rescale;
        max_s = s;
      }
      float w = expf(s - max_s);
      sum_w += w;
      for (int d = 0; d < HEAD_DIM; d++) acc[d] += w * v[d];
    }
  }

  // 最后归一化
  for (int d = 0; d < HEAD_DIM; d++) out[d] = acc[d] / sum_w;
}

// ================================================================
//  Batched decode attention: 多个序列的同一个 head 一起做 attention。
//  在 GPU 上可以用一个 kernel launch 覆盖所有 (seq, head) 对，
//  CPU 版就只能老实 for 循环了。但逻辑是完全一样的。
// ================================================================
void batched_paged_attention(
    int batch_size, const int* seq_ids, int head,
    const float* qs, float* outs)
{
  for (int i = 0; i < batch_size; i++) {
    paged_attention(seq_ids[i], head, context_lens[seq_ids[i]],
                    &qs[i * HEAD_DIM], &outs[i * HEAD_DIM]);
  }
}

// ================================================================
//  main: 造一些 mock 数据跑一遍，验证寻址对不对
// ================================================================
int main() {
  memset(key_cache, 0, sizeof(key_cache));
  memset(value_cache, 0, sizeof(value_cache));

  // 手工构造 3 个序列的 block table。物理块号故意不连续——
  // 这样才能验证"页表翻译能不能正确处理不连续的物理块"。
  block_tables[0][0]=3; block_tables[0][1]=7; block_tables[0][2]=12;
  block_tables[1][0]=1; block_tables[1][1]=5;
  block_tables[2][0]=8; block_tables[2][1]=4; block_tables[2][2]=0; block_tables[2][3]=15;

  // seq0: 35 tokens, seq1: 20 tokens, seq2: 60 tokens
  context_lens[0] = 35; context_lens[1] = 20; context_lens[2] = 60;

  // 模拟 prefill: 按每个序列的 context_len 填充初始 KV 数据。
  // 只填 head=0，其他 head 全是 0。
  // 值 = token 全局索引 × 0.1，这样可以通过数值反查是不是读到了正确的 token。
  for (int s = 0; s < 3; s++) {
    int phy = -1, lb = -1;
    for (int t = 0; t < context_lens[s]; t++) {
      if (t / BLOCK_SIZE != lb) { lb = t / BLOCK_SIZE; phy = block_tables[s][lb]; }
      int off = t % BLOCK_SIZE;
      for (int d = 0; d < HEAD_DIM; d++) {
        key_cache[phy][0][off][d] = t * 0.1f;
        value_cache[phy][0][off][d] = t * 0.1f;
      }
    }
  }

  // 验证 token 34 的数据位置:
  //   token 34 → ln=34/16=2(逻辑块2), off=34%16=2(偏移2)
  //   phy=block_tables[0][2]=12
  //   所以应该在 key_cache[12][0][2][:]
  //   值 = 34 * 0.1 = 3.40
  printf("seq0 token34: key=%f (expect 3.40)\n", key_cache[12][0][2][0]);

  // 测试 store_kv: 给 seq0 写入第 35 个 token
  float new_kv[NUM_HEADS * HEAD_DIM];
  for (int d = 0; d < NUM_HEADS * HEAD_DIM; d++) new_kv[d] = 3.5f;
  store_kv(0, 35, new_kv, new_kv);
  context_lens[0]++;
  // token 35 → ln=35/16=2, off=35%16=3, phy 还是 12（逻辑块 2 对应物理块 12）
  printf("seq0 token35 write: key=%f (expect 3.50)\n", key_cache[12][0][3][0]);

  // 测试 paged_attention: 给 seq0 head=0 做 attention
  float q[HEAD_DIM], out[HEAD_DIM];
  for (int d = 0; d < HEAD_DIM; d++) q[d] = 0.01f;
  paged_attention(0, 0, context_lens[0], q, out);
  printf("seq0 attn out: [%f,%f]\n", out[0], out[1]);

  // 测试 batched: 3 个序列一起做
  float qs[3*HEAD_DIM], outs[3*HEAD_DIM];
  memset(qs, 0x01, sizeof(qs));
  int sl[3] = {0,1,2};
  batched_paged_attention(3, sl, 0, qs, outs);
  printf("\nBatched(3seqs):\n");
  for (int i=0;i<3;i++) printf("  seq%d: [%f,%f]\n", i, outs[i*HEAD_DIM], outs[i*HEAD_DIM+1]);

  // 对比两种寻址方式
  printf("\n---- 寻址对比 ----\n");
  printf("传统连续KV: base + seq*max_len*H*D + token*H*D + head*D\n");
  printf("Paged KV  : key_cache[block_tables[seq][token/BLOCK]][head][token%%BLOCK][:]\n");
  printf("\nseq0 ctx=%d, block_table=[3,7,12]\n", context_lens[0]);
  printf("  物理块3: tokens 0..15\n");
  printf("  物理块7: tokens 16..31\n");
  printf("  物理块12: tokens 32..35\n");
  printf("物理块不连续, block_table 翻译后正确寻址\n");

  return 0;
}
