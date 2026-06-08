#pragma once

#include <cstdint>
#include <vector>

namespace mini_serve {

// 所有全局常量都扔这了，省的到处散落 magic number。
// 这些值差不多是对应 Llama-7B 的规模，BLOCK_SIZE=16 是 vLLM 论文
// 里推荐的——作者说试过 8、16、32、64，16 在碎片率和 block table
// 开销之间最平衡。具体怎么测的我也不知道，就信了。
struct Config {
  static constexpr int BLOCK_SIZE = 16;        // 一个 block 能塞几个 token 的 KV
  static constexpr int NUM_LAYERS = 32;        // transformer 层数
  static constexpr int NUM_HEADS = 32;         // 每层 attention head 数
  static constexpr int HEAD_DIM = 128;         // 每个 head 的维度
  static constexpr int MAX_SEQ_LEN = 2048;     // 最长序列，超过的直接截断
  static constexpr int MAX_NUM_BLOCKS = 1024;  // 物理 block 总数，模拟显存上限
  static constexpr int MAX_BATCH_SIZE = 32;    // 一轮最多同时跑几个请求
};

// 物理块——可以理解成 GPU 显存里 KV cache 的一小块连续空间。
// 如果真在 GPU 上跑，一个 block 存的大概是:
//   [num_layers][2(K+V)][num_heads][block_size][head_dim]
// 这么个 float16 的大数组。我们这里不真分配 GPU 内存，只管理元数据。
//
// 顺手算了一下一个 block 多大:
//   2 * 32 * 32 * 16 * 128 * 2 bytes = 8,388,608 bytes ≈ 8 MB
// 1024 个 block 就是 8 GB 显存。再加上模型权重（Llama-7B 大概 14GB fp16），
// 差不多是一张 A10（24G）能装下的量。所以这个配置不是瞎拍的。
struct PhysicalBlock {
  int block_id;
  int ref_count;  // 引用计数——多少个序列在用这个块。主要给 CoW 用，
                  // 计数归零才能回收。beam search 的时候一个块可能被
                  // 好几个分支共享。
  int num_filled; // 这个块里面填了几个 token。0 到 BLOCK_SIZE，
                  // 最后一个块经常填不满（比如序列总长 35 个 token，
                  // 前两个块 16+16=32，第三个块只填了 3 个）。

  PhysicalBlock() : block_id(-1), ref_count(0), num_filled(0) {}
  PhysicalBlock(int id) : block_id(id), ref_count(0), num_filled(0) {}

  bool is_full() const { return num_filled >= Config::BLOCK_SIZE; }
};

// 页表项——逻辑块到物理块的映射，跟 OS 的页表一个意思。
// 每个序列有自己的 block table（就是个数组），第 i 项记录"我这个序列
// 的第 i 个逻辑块存在哪个物理块里"。physical_block_id == -1 就是还没分配。
//
// 为什么要搞这一层映射？因为物理块可能不连续——序列 A 的三个逻辑块
// 可能在物理块 3、7、12 上，中间插着别的序列的块。跟虚拟内存一样，
// 逻辑地址连续不代表物理地址连续。
struct BlockTableEntry {
  int physical_block_id;

  BlockTableEntry() : physical_block_id(-1) {}
  BlockTableEntry(int pid) : physical_block_id(pid) {}
};

}  // namespace mini_serve
