// KV Cache Manager 的单元测试。
// 主要测这些东西:
//   - 基本分配/释放: 分了、释放，空闲数对不对
//   - OOM: 全分完了再请求，应该返回 -1
//   - Copy-on-Write: fork 共享后写入，看是不是真的复制了一份
//   - block fill 跟踪: 最后一个块填了多少、满了没、需要几个新块
//   - 多序列并发: 几个序列一起分一起释放，确认不互相干扰

#include "../include/kv_cache_manager.h"
#include <cassert>
#include <iostream>

using namespace mini_serve;

// 最基本的分-放流程:
//   分几个 block → 确认空闲数减少了 → 释放 → 确认空闲数恢复了
void test_basic_allocation() {
  std::cout << "[TEST] Basic allocation... ";
  KVCacheManager mgr(16);  // 总共 16 个物理 block

  assert(mgr.get_num_free_blocks() == 16);

  // seq 0: 建序列 + 分一个 block
  assert(mgr.allocate_sequence(0));
  int blk = mgr.append_block(0);
  assert(blk >= 0);
  assert(mgr.get_num_free_blocks() == 15);
  assert(mgr.get_seq_num_blocks(0) == 1);

  // 再追加两个
  mgr.append_block(0);
  mgr.append_block(0);
  assert(mgr.get_num_free_blocks() == 13);
  assert(mgr.get_seq_num_blocks(0) == 3);

  // 释放整个序列，3 个 block 应该全部回收
  mgr.free_sequence(0);
  assert(mgr.get_num_free_blocks() == 16);
  assert(mgr.get_seq_num_blocks(0) == 0);

  std::cout << "PASS\n";
}

// OOM: 总共 4 个 block，全分完再要就该返回 -1 了
void test_oom() {
  std::cout << "[TEST] OOM handling... ";
  KVCacheManager mgr(4);

  mgr.allocate_sequence(0);
  mgr.append_block(0);
  mgr.append_block(0);
  mgr.append_block(0);
  mgr.append_block(0);
  assert(mgr.get_num_free_blocks() == 0);  // 全没了

  // 再分 → 失败
  int result = mgr.append_block(0);
  assert(result == -1);

  // 释放之后又有了
  mgr.free_sequence(0);
  assert(mgr.get_num_free_blocks() == 4);
  mgr.allocate_sequence(1);
  result = mgr.append_block(1);
  assert(result >= 0);

  std::cout << "PASS\n";
}

// Copy-on-Write:
//   1. seq0 分 2 个物理块
//   2. fork 给 seq1（两个序列共享同一批块，ref_count=2）
//   3. seq1 对第 0 个逻辑块做 CoW → 应该分一个新块，旧块 ref_count 减 1
//   4. 释放 seq0 → 被 CoW 走的旧块只有 seq0 自己了，应该回收；
//      seq1 的 CoW 新块和还没 CoW 的旧块保留
//   5. 释放 seq1 → 全回收
void test_copy_on_write() {
  std::cout << "[TEST] Copy-on-Write... ";
  KVCacheManager mgr(16);

  // seq0: 2 个物理块
  mgr.allocate_sequence(0);
  mgr.append_block(0);
  mgr.append_block(0);

  // fork: seq1 共享 seq0 的块，不消耗新块
  mgr.fork_sequence(0, 1);
  assert(mgr.get_seq_num_blocks(1) == 2);
  assert(mgr.get_num_free_blocks() == 14);  // 没增加消耗

  // seq1 对第 0 个逻辑块做 CoW，因为 ref_count=2，必须复制
  int new_blk = mgr.copy_on_write(1, 0);
  assert(new_blk >= 0);
  assert(mgr.get_num_free_blocks() == 13);  // 新分了一个

  // 释放 seq0:
  //   - 逻辑块 0: seq0 的 ref 之前已被 CoW 减为 1 → 回收
  //   - 逻辑块 1: seq1 还引用着（ref=2→1），不回收
  mgr.free_sequence(0);
  assert(mgr.get_num_free_blocks() == 14);  // 只收回了块 0

  // 释放 seq1: 剩下全回收
  mgr.free_sequence(1);
  assert(mgr.get_num_free_blocks() == 16);

  std::cout << "PASS\n";
}

// 测试 block 填充跟踪。
// 模拟往最后一个 block 填 token，检查 is_full、calc_needed_blocks
// 这些查询接口的逻辑对不对。
void test_block_fill() {
  std::cout << "[TEST] Block fill tracking... ";
  KVCacheManager mgr(16);

  mgr.allocate_sequence(0);
  mgr.append_block(0);

  // 填 10 个 token，没满
  mgr.fill_last_block(0, 10);
  assert(mgr.get_last_block_fill(0) == 10);
  assert(!mgr.is_last_block_full(0));

  // 再填 6 个，10+6=16=BLOCK_SIZE，满了
  mgr.fill_last_block(0, 6);
  assert(mgr.is_last_block_full(0));

  // 满了之后再要 5 个 token → 需要 1 个新 block
  int needed = mgr.calc_needed_blocks(0, 5);
  assert(needed == 1);

  // 真的追加一个并填 5 个 token
  mgr.append_block(0);
  mgr.fill_last_block(0, 5);
  assert(!mgr.is_last_block_full(0));

  std::cout << "PASS\n";
}

// 多个序列同时用，测并发分配/释放互不干扰
void test_multiple_sequences() {
  std::cout << "[TEST] Multiple sequences... ";
  KVCacheManager mgr(32);

  // 8 个序列各 2 个 block = 16 个
  for (int i = 0; i < 8; i++) {
    mgr.allocate_sequence(i);
    mgr.append_block(i);
    mgr.append_block(i);
  }
  assert(mgr.get_num_free_blocks() == 16);  // 32 - 16 = 16

  // 释放前 4 个 → 回收 8 个 block
  for (int i = 0; i < 4; i++) {
    mgr.free_sequence(i);
  }
  assert(mgr.get_num_free_blocks() == 24);  // 16 + 8 = 24

  // 序列 4 应该还剩 2 个 block
  auto table = mgr.get_block_table(4);
  assert(table.size() == 2);

  std::cout << "PASS\n";
}

int main() {
  std::cout << "=== KV Cache Manager Tests ===\n\n";

  test_basic_allocation();
  test_oom();
  test_copy_on_write();
  test_block_fill();
  test_multiple_sequences();

  std::cout << "\nAll tests passed!\n";
  return 0;
}
