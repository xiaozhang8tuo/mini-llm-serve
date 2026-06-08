#pragma once

#include "block.h"

#include <deque>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <cassert>
#include <algorithm>

namespace mini_serve {

// KV Cache 的分页管理器。
//
// 这个东西是看了 vLLM 论文之后的一个理解笔记式的实现。
//
// 先说背景: 传统 LLM serving 给每个请求预分配的 KV cache 是连成一片的，
// 大小 = max_seq_len * num_layers * 2(K/V) * num_heads * head_dim。
// 举个例子——如果 max_seq_len = 2048，上面这套乘起来大概 1GB。
// 但大部分用户的 prompt 就几百个 token，剩下 1GB 全是空着的，
// 又因为每块是连续的，释放之后留下的空隙凑不够也给别人用不了——碎片化。
//
// vLLM 的解法: 把 KV cache 切成固定大小的 block，跟 OS 的虚拟内存一样，
// 不直接给物理地址，而是通过页表（block table）做映射。
// 每个序列需要几个 block 就分几个，物理上不要求连续，逻辑上通过
// block table 连起来看就是一段连续的 KV cache。
//
// 好处:
//   1. 没有内部碎片——block 大小固定，不会出现"差 3 个 token 放不下"的情况
//      因为最小分配单位是一个 block，里面的空间全都是你的
//   2. 按需分配——短的 prompt 少分几个 block，不像传统方式那样强制
//      按 max_seq_len 预分配
//   3. 引用计数 + CoW——beam search 的时候多个候选分支可以共享前缀 block，
//      新的物理内存只在真正需要写入不同的数据时才分配
//
// 你可能会想"那外部碎片呢？" —— 物理块是固定大小的池子，
// 任意空闲块可以分配给任意序列，不存在"这段空间放了只能相邻用"的问题，
// 所以没有外部碎片。这也是为什么 OS 用分页机制。
//
// 我们这里只模拟管理逻辑（分配/释放/CoW/Fork），不存真实的 KV 张量。
// 如果真跑在 GPU 上，每个 block 大约 8MB（按我们 Config 里的参数算）。
class KVCacheManager {
 public:
  explicit KVCacheManager(int num_blocks = Config::MAX_NUM_BLOCKS)
      : num_total_blocks_(num_blocks) {
    // 初始化物理块池子——一开始全是空闲的，每个 block 一个 id
    physical_blocks_.reserve(num_blocks);
    for (int i = 0; i < num_blocks; i++) {
      physical_blocks_.emplace_back(i);
      free_blocks_.push_back(i);
    }
  }

  // ========== 核心接口 ==========

  // 给一个新序列建立 block table（还是个空表，没绑定任何物理块）。
  // 分配物理块是在 append_block 时才真正做的，所以这里只是"注册"。
  bool allocate_sequence(int seq_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (block_tables_.count(seq_id)) return false;  // 已经存在了，不能重复
    block_tables_[seq_id] = {};
    return true;
  }

  // 给序列追加一个新的物理块。什么时候需要？
  //   - prefill 阶段: 第一次分配，一次可能加好几个 block
  //   - decode 阶段: 当前最后一个 block 填满了，但还要继续生成
  // 返回新物理块 id，-1 表示池子干了 = OOM，调用方得处理.
  int append_block(int seq_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (free_blocks_.empty()) return -1;  // 全分完了，没得给了

    int phy_id = free_blocks_.front();
    free_blocks_.pop_front();

    auto& block = physical_blocks_[phy_id];
    block.ref_count = 1;    // 一开始只有当前序列引用
    block.num_filled = 0;   // 全新的空块，啥都没有

    block_tables_[seq_id].emplace_back(phy_id);
    return phy_id;
  }

  // 释放一个序列占用的所有物理块。
  // 注意不是直接回收——要用引用计数。因为可能有 CoW 共享（fork 出来
  // 的几个序列指向同一个物理块），只有 ref_count 降到 0 才真正回收。
  void free_sequence(int seq_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = block_tables_.find(seq_id);
    if (it == block_tables_.end()) return;

    for (auto& entry : it->second) {
      int phy_id = entry.physical_block_id;
      if (phy_id < 0) continue;
      auto& block = physical_blocks_[phy_id];
      block.ref_count--;
      if (block.ref_count <= 0) {
        // 没人引用了，回池子里
        block.ref_count = 0;
        block.num_filled = 0;
        free_blocks_.push_back(phy_id);
      }
    }
    block_tables_.erase(it);
  }

  // Copy-on-Write: 当一个序列要写入某个共享的 block 时调用。
  //
  // 场景: 序列 A fork 出了 A1 和 A2，它们共享 A 的所有物理块。
  // 现在 A1 要继续 decode，如果它直接写共享块里的旧 KV 数据，
  // 就把 A2 的数据也改了——这是不对的。所以要 CoW: 先复制一份
  // 出来，在新的物理块上写，老的不动。
  //
  // 如果这个块只有自己引用（ref_count==1），那就不用复制，直接写。
  // 返回值是"安全写入的物理块 id"。
  int copy_on_write(int seq_id, int logical_block_idx) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto& table = block_tables_[seq_id];
    if (logical_block_idx >= (int)table.size()) return -1;

    int old_phy = table[logical_block_idx].physical_block_id;
    auto& old_block = physical_blocks_[old_phy];

    // 只有自己用，不用复制
    if (old_block.ref_count == 1) return old_phy;

    // 多个人共享——必须复制
    if (free_blocks_.empty()) return -1;  // 没空闲块了还复制个毛

    int new_phy = free_blocks_.front();
    free_blocks_.pop_front();

    auto& new_block = physical_blocks_[new_phy];
    new_block.ref_count = 1;
    new_block.num_filled = old_block.num_filled;
    // 如果真跑在 GPU 上，这里要 cudaMemcpy 把 KV 数据从旧块拷到新块。
    // 我们这个 mock 里跳过——反正没真数据。

    // 自己不再引用旧块了
    old_block.ref_count--;

    // 页表指向新物理块
    table[logical_block_idx].physical_block_id = new_phy;
    return new_phy;
  }

  // Fork: 给 beam search 用的。新序列直接共享源序列的所有物理块，
  // 每个被共享的块引用计数 +1。
  //
  // fork 之后的两个序列看到的 KV cache 是完全一样的（指向同一批物理块），
  // 后面哪个要写入了，CoW 会在那个点上自动分裂。
  bool fork_sequence(int src_seq_id, int dst_seq_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = block_tables_.find(src_seq_id);
    if (it == block_tables_.end()) return false;

    // 直接拷贝 block table，指向同一批物理块
    block_tables_[dst_seq_id] = it->second;
    for (auto& entry : block_tables_[dst_seq_id]) {
      if (entry.physical_block_id >= 0) {
        physical_blocks_[entry.physical_block_id].ref_count++;
      }
    }
    return true;
  }

  // ========== 查询接口 ==========

  int get_num_free_blocks() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return (int)free_blocks_.size();
  }

  int get_num_used_blocks() const {
    return num_total_blocks_ - get_num_free_blocks();
  }

  // 某个序列占了几个物理块
  int get_seq_num_blocks(int seq_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = block_tables_.find(seq_id);
    if (it == block_tables_.end()) return 0;
    return (int)it->second.size();
  }

  // 查最后一个 block 里填了几个 token。
  // 最后一块经常不满，比如序列有 35 个 token，BLOCK_SIZE=16，
  // 前两个块满 16+16=32，最后一个只填了 3 个。
  // decode 的时候要知道该写到 block 的哪个位置（offset）。
  int get_last_block_fill(int seq_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = block_tables_.find(seq_id);
    if (it == block_tables_.end()) return 0;
    if (it->second.empty()) return 0;
    int phy_id = it->second.back().physical_block_id;
    return physical_blocks_[phy_id].num_filled;
  }

  // 标记最后一个 block 里多了 n 个 token 的 KV 数据。
  // 调度器在 decode 每步后调这个，把新生成的 token 也算进去。
  void fill_last_block(int seq_id, int n) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = block_tables_.find(seq_id);
    if (it == block_tables_.end()) return;
    if (it->second.empty()) return;
    int phy_id = it->second.back().physical_block_id;
    physical_blocks_[phy_id].num_filled += n;
  }

  // 最后一个 block 满了没？满了的话下次 decode 要先 append_block。
  bool is_last_block_full(int seq_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = block_tables_.find(seq_id);
    if (it == block_tables_.end()) return true;
    if (it->second.empty()) return true;
    int phy_id = it->second.back().physical_block_id;
    return physical_blocks_[phy_id].is_full();
  }

  // 如果还要追加 num_new_tokens 个 token，需要几个新 block？
  // 调度器用这个来决定"要不要给这个 decode 请求新 block"、
  // "free block 够不够让这个请求继续跑"。
  int calc_needed_blocks(int seq_id, int num_new_tokens) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = block_tables_.find(seq_id);
    int current_fill = 0;
    if (it != block_tables_.end() && !it->second.empty()) {
      int phy_id = it->second.back().physical_block_id;
      current_fill = physical_blocks_[phy_id].num_filled;
    }
    int remaining_in_last = Config::BLOCK_SIZE - current_fill;
    if (num_new_tokens <= remaining_in_last) return 0;  // 最后一块还有坑
    int overflow = num_new_tokens - remaining_in_last;
    return (overflow + Config::BLOCK_SIZE - 1) / Config::BLOCK_SIZE;  // ceil
  }

  // 把 block table 导成 int 数组，方便给 attention kernel 用。
  // 真实 GPU kernel 拿到的就是这个 [phy_0, phy_1, ...] 数组，
  // kernel 内部用它做逻辑→物理的翻译。
  std::vector<int> get_block_table(int seq_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<int> result;
    auto it = block_tables_.find(seq_id);
    if (it == block_tables_.end()) return result;
    for (auto& entry : it->second) {
      result.push_back(entry.physical_block_id);
    }
    return result;
  }

  int num_total_blocks() const { return num_total_blocks_; }

 private:
  int num_total_blocks_;
  std::vector<PhysicalBlock> physical_blocks_;                           // 物理 block 池子
  std::deque<int> free_blocks_;                                          // 空闲 block 队列
  std::unordered_map<int, std::vector<BlockTableEntry>> block_tables_;   // seq_id → block table
  mutable std::mutex mutex_;  // 所有对外接口都加锁了，简单直接。这个项目
                              // 的调用模式比较简单，不用读写锁也不会是瓶颈。
};

}  // namespace mini_serve
