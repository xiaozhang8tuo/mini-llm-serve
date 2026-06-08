// CUDA 编程里一些比较重要的优化技巧，每个拆成一个独立的小 demo，
// 带详细的注释说明为什么这么写、每一步线程在做什么。
//
// 这东西主要是读的，不是跑的——main() 里只列技巧名，不跑实际 kernel。
// 每个技巧在注释块里给了具体的数据走读，跟着算一遍应该就懂了。
//
// 编译: nvcc -O2 cuda_techniques.cu -o cuda_techniques && ./cuda_techniques

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

// ================================================================
//  技巧 1: stride loop（跳步循环 / grid-stride loop 的单 block 版）
//  ================================================================
//  问题: 要处理 N 个元素，但一个 block 里只有 T 个线程，N 比 T 大很多。
//  怎么办? 让每个线程"负责"多个元素，跳着处理。
//
//  这是几乎所有 CUDA kernel 的模板操作——GPU 上线程虽多但数据更多，
//  每个线程处理 ceil(N/T) 个元素。
//
//  关键: 步长 = 线程数 T，不是 1。这样相邻的线程处理相邻元素
//  （thread0 管 idx=0,T,2T...；thread1 管 idx=1,T+1,2T+1...），
//  天然 coalesced。

__global__ void technique_1_stride_loop(float* a, float* b, float* c, int N) {
    int tid = threadIdx.x;
    int T   = blockDim.x;

    for (int i = tid; i < N; i += T) {
        c[i] = a[i] + b[i];
    }
}

/*
照着走一遍: N=1024, T=256

Thread 0:  i=0   → c[0]=a[0]+b[0]
Thread 0:  i=256 → c[256]=a[256]+b[256]
Thread 0:  i=512 → c[512]=a[512]+b[512]
Thread 0:  i=768 → c[768]=a[768]+b[768]
Thread 0:  i=1024 → 退出（>=N）

Thread 1:  i=1   → c[1]=a[1]+b[1]
Thread 1:  i=257 → c[257]=a[257]+b[257]
...

Thread 255: i=255 → c[255]=a[255]+b[255]
Thread 255: i=511 → c[511]=a[511]+b[511]
...

你看 Thread 0 处理的地址是 c[0]、c[256]、c[512]、c[768]，
都是连续的地址（间隔 256×4 = 1024 bytes）。如果 N 超级大
（百万级），stride 会变大到几万 → cache line 效率变差，
这时候就得上 grid-stride loop（技巧 4）了。
*/

// ================================================================
//  技巧 2: parallel reduction（并行归约 / 树形归约）
//  ================================================================
//  问题: N 个线程各有一个数，求总和（或最大值/最小值）。
//  串行: 一个线程遍历 N 个 → O(N)，浪费其他 N-1 个线程。
//  并行: 树形归约 → O(log₂N)，log₂N 步搞定。
//
//  怎么做: 用 shared memory 当"草稿纸"，每轮 stride 减半，
//  活着的线程把两个数加在一起。log₂N 轮后 smem[0] 就是总和。
//
//  下面这个例子用 256 个线程处理 N 个数（N ≤ 256）。

__global__ void technique_2_reduction(float* input, float* output, int N) {
    __shared__ float smem[256];  // 草稿纸——每个线程把自己的数抄到这上面

    int tid = threadIdx.x;

    // 第一步: 各自把数据从 HBM 搬到 shared memory
    smem[tid] = (tid < N) ? input[tid] : 0.0f;
    __syncthreads();  // 等所有人抄完

    // 第二步: 树形归约
    //         每轮活跃线程减半，stride 从 N/2 往下折半
    for (int stride = N / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();  // 每轮之间必须同步！shared memory 的读写顺序
                          // 不保证，不加这个可能会读到别人没写完的值。
    }

    // smem[0] 现在存着所有 N 个数的总和
    if (tid == 0) output[0] = smem[0];
}

/*
拿 8 个数具体走一遍: [a, b, c, d, e, f, g, h]

初始状态: smem = [a,  b,  c,  d,  e,  f,  g,  h]
                   0   1   2   3   4   5   6   7  ← thread id

stride=4 (干活的是 tid 0~3，tid 4~7 闲着):
  tid=0: smem[0] = a + smem[4] → a+e
  tid=1: smem[1] = b + smem[5] → b+f
  tid=2: smem[2] = c + smem[6] → c+g
  tid=3: smem[3] = d + smem[7] → d+h
  __syncthreads()
  smem = [a+e, b+f, c+g, d+h, e, f, g, h]

stride=2 (干活的是 tid 0~1):
  tid=0: smem[0] = (a+e)+(c+g) = a+e+c+g
  tid=1: smem[1] = (b+f)+(d+h) = b+f+d+h
  __syncthreads()
  smem = [a+e+c+g, b+f+d+h, ...]

stride=1 (干活的是 tid=0):
  tid=0: smem[0] = (a+e+c+g)+(b+f+d+h) = a+b+c+d+e+f+g+h ✓
  __syncthreads()

3 步搞定（log₂8=3），对比串行 7 次加法。数据规模越大差距越明显。
*/

// ================================================================
//  技巧 3: bank conflict 和 shared memory padding
//  ================================================================
//  shared memory 的硬件: 分成 32 个 bank，每个 bank 宽 4 bytes。
//  如果同一 warp 的 32 个线程同时访问同一个 bank → 串行化 → 慢。
//
//  怎么算 bank id？
//    bank_id = (address_in_bytes / 4) % 32
//
//  什么情况会 conflict？
//    warp 中两个不同的线程访问同一个 bank → conflict。
//    如果访问同一个 bank 的同一个地址，不算 conflict（broadcast）。
//
//  最常见的 trigger: 二维数组的列访问。
//    shared float arr[32][128]
//    // thread i 访问 arr[i][0]
//    // 地址差 = 128*4 = 512 bytes = 128 words, 128%32=0
//    // → 所有 32 个线程都在 bank 0 → 32-way conflict！
//
//  解法: padding，把第二维加 1，打破对齐。

__global__ void technique_3_bank_conflict() {
    int tid = threadIdx.x;

    // 坏写法: [32][128]，128 是 32 的倍数
    // thread 0~31 分别写 bad[0][0], bad[1][0], ..., bad[31][0]
    // 地址间隔 128*4 = 512 bytes = 128 words, 128%32=0
    // → 全落在 bank 0, 32-way conflict
    __shared__ float bad[32][128];
    bad[tid][0] = tid;

    // 好写法: [32][128+1]，129 不是 32 的倍数
    // thread i 写 good[i][0], 地址间隔 129*4 = 516 bytes = 129 words
    // 129%32 = 1 → 每个线程落不同 bank → 零 conflict
    __shared__ float good[32][128 + 1];
    good[tid][0] = tid;
}

/*
更详细地算一遍:

bad 数组[32][128] (第二维 128):
  thread 0:  good[0][0] → offset = 0     → 0 words  → bank 0
  thread 1:  good[1][0] → offset = 128   → 128%32=0 → bank 0  ← 冲突！
  thread 2:  good[2][0] → offset = 256   → 256%32=0 → bank 0  ← 冲突！
  ...
  32 个线程全部 hit bank 0，真跑起来要串行 32 次。

good 数组[32][129] (第二维 129):
  thread 0:  good[0][0] → offset = 0     → 0%32=0   → bank 0
  thread 1:  good[1][0] → offset = 129   → 129%32=1 → bank 1  ✓
  thread 2:  good[2][0] → offset = 258   → 258%32=2 → bank 2  ✓
  ...
  32 个线程各占一个 bank，无冲突。

代价: 每个 row 多占 1 个 float（4 bytes）的 padding。
32×1×4 = 128 bytes，对 shared memory 来说基本没影响。
*/

// ================================================================
//  技巧 4: grid-stride loop（多 block 版全局跳步）
//  ================================================================
//  技巧 1 只有 1 个 block，线程数 limit = 1024。
//  数据量几百万、几千万时得用多个 block。grid-stride loop
//  就是技巧 1 上加一个"全局线程 id"和"全局线程总数"。
//
//  好处: 无论开多少 block，负载自动均衡。比如你有 15 个 SM 但开了
//  64 个 block，runtime 会先调度 15 个，跑完再调度下一批 15 个。
//  每个线程干的工作量自动均匀分配。

__global__ void technique_4_grid_stride(float* a, float* b, float* c, int N) {
    // 全局线程 id: 跨所有 block 的唯一编号
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // 所有 block 的线程之和
    int total_threads = gridDim.x * blockDim.x;

    // 步长 = 全局总线程数
    for (int i = tid; i < N; i += total_threads) {
        c[i] = a[i] + b[i];
    }
}

/*
例子: N=1,000,000, grid=64 blocks, block=256 threads
  total_threads = 64 × 256 = 16,384
  每个线程负责 ceil(1,000,000 / 16,384) ≈ 62 个元素

Thread 0:     i = 0, 16384, 32768, 49152, ...
Thread 1:     i = 1, 16385, 32769, 49153, ...
...
Thread 16383: i = 16383, 32767, 49151, ...

和数据量比，步长 16384 不算大，cache 效率还行。
*/

// ================================================================
//  技巧 5: warp shuffle reduction（warp 内归约 / 寄存器归约）
//  ================================================================
//  技巧 2 用了 shared memory，每次需要 __syncthreads()。
//  但如果归约发生在 warp 内部（32 个线程之内），可以更狠一点:
//  用 __shfl_down_sync 直接在寄存器之间传数据，不过 shared memory。
//
//  寄存器到寄存器: ~1 cycle
//  shared memory: ~20-30 cycles
//  HBM: ~200-800 cycles
//
//  所以 warp shuffle 比 shared memory reduction 快一个数量级。
//
//  __shfl_down_sync(mask, val, offset):
//    线程 i 收到线程 i+offset 的 val 值。
//    mask 控制参与线程（0xffffffff = 全 32 线程）。

__global__ void technique_5_warp_reduce(float* input, float* output) {
    int tid = threadIdx.x;
    float val = input[tid];

    // warp shuffle: offset 从 16 → 8 → 4 → 2 → 1
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }

    // 每个 warp 的线程 0 存下自己 warp 的总和
    if (tid % 32 == 0) output[tid / 32] = val;
}

/*
和技巧 2 一样的归约，只是用寄存器替换了 shared memory:

offset=16: thread 0 收到 thread 16 的 val → val0 += val16
           thread 1 收到 thread 17 的 val → val1 += val17
           ...
offset=8:  thread 0 收到 thread 8 的 val（此时 val8 已经 = val8+val24）
           ...
offset=1:  thread 0 收到 thread 1 的 val → val0 = 全 warp 总和

0xffffffff: 32 个 bit 全是 1，表示这个 warp 的 32 个线程全参与。
如果你只想要前 16 个: 用 0x0000ffff。
*/

// ================================================================
//  技巧 6: double buffering（双缓冲 / 乒乓缓冲）
//  ================================================================
//  问题: 从 HBM 加载数据到 shared memory 要几百 cycle，
//  GPU 在这段时间里只能干等。
//
//  解法: 准备两块 shared memory，一块在"被计算"，另一块在"被加载"。
//  加载完就交换角色。把 HBM 的延迟藏在计算时间下面。
//
//  这招在 flash attention 和各种 matmul kernel 里到处都有。
//
//  注意: 下面这个不是真正的异步加载——真正的硬件异步 copy
//  需要 SM80+ (A100) 的 async copy 指令。这里用 __syncthreads()
//  做了个"伪双缓冲"来演示想法。

__global__ void technique_6_double_buffer(const float* input, float* output, int N) {
    __shared__ float buf0[128];
    __shared__ float buf1[128];

    int tid = threadIdx.x;
    int global_tid = blockIdx.x * blockDim.x + tid;

    // 先手动把第一批数据拽进来
    if (global_tid < N) buf0[tid] = input[global_tid];
    __syncthreads();

    for (int chunk = 0; chunk < N / 128; chunk++) {
        int next_offset = (chunk + 1) * 128 + global_tid;

        // 预加载下一批到 buf1（跟当前 buf0 的计算同时进行）
        if (chunk + 1 < N / 128 && next_offset < N) {
            buf1[tid] = input[next_offset];
        }

        // 同时做 buf0 上的计算
        float val = buf0[tid];
        output[chunk * 128 + global_tid] = val * val;

        __syncthreads();

        // 乒乓: 下一轮 buf1 变 buf0
        if (chunk + 1 < N / 128) {
            buf0[tid] = buf1[tid];
        }
        __syncthreads();
    }
}

/*
时间线对比:

没有双缓冲:
  [load_chunk0] [compute_chunk0] [load_chunk1] [compute_chunk1] ...
   ↑ 加载时 GPU 的 SM 在发呆             ↑ 又在发呆

双缓冲:
  [load_buf0] [load_buf1 + compute_buf0] [load_buf0 + compute_buf1] ...
              ↑ HBM 加载和 SM 计算时间重叠
  用户体感的时间 = 计算时间（加载被藏起来了）
*/

// ================================================================
//  技巧 7: arithmetic intensity（算术强度分析）
//  ================================================================
//  你写了一个 kernel，它到底慢在哪？GPU 算力不够还是带宽不够？
//  这个判断决定了优化方向。
//
//  算术强度 = FLOPs / Bytes_read
//    每次从 HBM 读 1 个 byte，能换来几次浮点运算？
//
//  判断标准:
//    AI < 硬件平衡点 → memory bound → 去优化数据复用、加 shared memory 缓存
//    AI > 硬件平衡点 → compute bound → 加并行度、换更快指令
//
//  硬件平衡点 = (GPU 峰值算力) / (显存带宽):
//    A100: 312 TFLOPS(fp16) / 2 TB/s = ~156 FLOPs/byte
//    H100: 989 TFLOPS(fp16) / 3.35 TB/s ≈ 295 FLOPs/byte
//
//  这也是 fp16 为什么对 LLM 这么关键——同样的带宽，fp16 比 fp32
//  多搬一倍的数据，等于 AI 翻倍。

/*
拿一个 tiled matmul 算一下（tile 大小 r×r）:

  计算量: r² 个输出元素，每个做 r 次乘加 = 2r³ FLOPs
  读取量: 2r² 个元素（A tile + B tile），每个 2 bytes(fp16) = 4r² bytes
  AI = 2r³ / 4r² = r/2

  A100 上 (平衡点 156):
    r=32  → AI=16 < 156  → memory bound
    r=128 → AI=64 < 156  → 还是 memory bound！
    r 要到 312 才打平，但 shared memory 装不下这么大的 tile。

  结论: matmul 大多数情况是 memory bound。优化重点是
  1. 用大 tile（shared memory 能装的最大值）
  2. 双缓冲把加载延迟藏起来
  3. tensor core 提高有效算力（等效于降低平衡点）
*/

// ================================================================
//  技巧 8: coalesced memory access（合并访存 / 对齐访存）
//  ================================================================
//  GPU 访问 HBM 的最小单位是 128 bytes（一个 cache line）。
//  同一 warp 的 32 个线程访问的地址如果在同一个 128-byte 段内，
//  硬件会把 32 次请求合成 1 次事务（coalescing）。
//
//  规则很简单:
//    相邻线程访问相邻地址 → 合并 → 1 次事务 ＝ 快
//    相邻线程访问的地址四处散落 → 不能合并 → 32 次事务 ＝ 慢
//
//  所以 GPU 上数据的 layout 特别重要。row-major + 相邻线程访问同一行的相邻列
//  = 完美 coalesced。

__global__ void technique_8_coalesced(float* matrix, int width) {
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // 好的写法: thread0 读 matrix[row][0], thread1 读 matrix[row][1],
    // thread2 读 matrix[row][2]... → 地址连续 → coalesced
    float good = matrix[row * width + col];

    // 坏的写法: thread0 读 matrix[0][row], thread1 读 matrix[1][row],
    // thread2 读 matrix[2][row]... → 地址间隔 = width*4 bytes → 不连续
    // → 每个线程落在不同 cache line → 32 次单独事务
    float bad = matrix[col * width + row];
}

/*
具体到 PagedAttention 里——因为 KV cache 存在不连续的物理块上，
每个块的访问模式要小心设计。vLLM 的 paged_attention kernel
把 block 内部维度排成 [head][tok][dim]，让 dim 是最内层维度，
这样同一 head 同一 token 内相邻线程访问相邻 dim → coalesced。

coalesced 例子 (width=256, 数据地址从 0x1000 起):
  Thread 0:  matrix[0][0] = 0x1000
  Thread 1:  matrix[0][1] = 0x1004
  Thread 2:  matrix[0][2] = 0x1008
  ...
  32 个线程访问 32 个连续 4-byte 地址 = 128 bytes 的连续区域
  → 1 次 128-byte 事务 ✓

non-coalesced 例子:
  Thread 0:  matrix[0][row] = 0x1000
  Thread 1:  matrix[1][row] = 0x1000 + 256*4 = 0x1400
  Thread 2:  matrix[2][row] = 0x1000 + 2*256*4 = 0x1800
  ...
  每个差 1024 bytes，全在不同 cache line → 32 次单独事务 ✗
  性能可能差 10-30 倍。
*/

// ================================================================
int main() {
    printf("此文件是 CUDA 技巧教学，不实际跑 kernel\n");
    printf("每个 kernel 函数和注释块可以独立阅读，建议顺序:\n");
    printf("  1 → 4 → 2 → 5 → 3 → 8 → 6 → 7\n\n");

    printf("技巧列表:\n");
    printf("  1. stride loop      — 数据比线程多，让每个线程跳步处理多个\n");
    printf("  2. parallel reduction — shared memory 树形归约，O(log N) 步\n");
    printf("  3. bank conflict    — shared memory 的 bank 及如何用 padding 避免冲突\n");
    printf("  4. grid-stride loop — 多 block 版全局跳步，自动负载均衡\n");
    printf("  5. warp shuffle     — 寄存器交换做归约，比 shared memory 快一个量级\n");
    printf("  6. double buffering — 乒乓缓冲，把 HBM 加载延迟藏在计算时间下面\n");
    printf("  7. arithmetic intensity — 判断 kernel 卡在算力还是带宽，决定优化方向\n");
    printf("  8. coalesced access — 让 32 个线程的访问合成 1 次事务，差 10-30 倍\n");
    return 0;
}
