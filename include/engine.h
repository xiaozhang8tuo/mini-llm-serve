#pragma once

#include "kv_cache_manager.h"
#include "mock_attention.h"
#include "request.h"
#include "scheduler.h"

#include <atomic>
#include <functional>
#include <iostream>
#include <vector>

namespace mini_serve {

// 推理引擎——把 scheduler、kv_cache、attention 串起来的胶水代码。
//
// 主循环大概是这样的:
//
//   while (还有请求没跑完) {
//     1. scheduler.schedule() → 决定这一轮谁 prefill、谁 decode、谁被踢
//     2. 对 prefill 请求: 分配的 KV cache 已经在 schedule 时做了，
//        这里模拟 prefill 计算，填充 cache，产出第一个 token
//     3. 对 decode 请求: 如果最后一个 block 满了就先追加新的，
//        然后模拟一次 decode，追加一个 token
//     4. 更新状态，标记完成的请求
//   }
//
// 整个引擎不碰真模型，用 MockAttention 来模拟计算耗时。
// 目的是验证调度策略和 KV cache 管理的逻辑对不对，不是跑推理。
class Engine {
 public:
  // 跑完一轮的汇总统计
  struct Stats {
    int total_iterations = 0;
    int total_prefill_tokens = 0;
    int total_decode_tokens = 0;
    double total_time_ms = 0;
  };

  Engine(int num_blocks = Config::MAX_NUM_BLOCKS,
         int max_batch_size = Config::MAX_BATCH_SIZE)
      : kv_cache_(num_blocks),
        scheduler_(kv_cache_, max_batch_size),
        running_(false) {}

  // 提交一个请求（会进等待队列，不是立刻跑）
  void submit_request(Request* req) {
    scheduler_.add_request(req);
  }

  // 启动主循环，block 住直到所有请求跑完。
  // 返回总的统计信息。
  Stats run() {
    running_ = true;
    Stats stats;
    auto start = std::chrono::steady_clock::now();

    while (scheduler_.has_pending_work()) {
      step(stats);
    }

    auto end = std::chrono::steady_clock::now();
    stats.total_time_ms =
        std::chrono::duration<double, std::milli>(end - start).count();
    running_ = false;
    return stats;
  }

  // 执行单个 iteration 的全部流程
  void step(Stats& stats) {
    stats.total_iterations++;

    // 1. 调度: 决定这轮谁干什么
    auto sched_output = scheduler_.schedule();

    int batch_size = (int)sched_output.prefill_batch.size() +
                     (int)sched_output.decode_batch.size();
    if (batch_size == 0) return;  // 空轮（理论上不该发生，除非所有请求都卡在
                                  // 队列里等内存释放）

    // 2. Prefill: 一次性算完整个 prompt 的 self-attention
    for (auto* req : sched_output.prefill_batch) {
      int prompt_len = (int)req->prompt_tokens.size();
      stats.total_prefill_tokens += prompt_len;

      // 模拟 prefill 的注意力计算 —— 最耗时的部分
      int first_token = attention_.simulate_prefill(prompt_len, batch_size);

      // prefill 完成，所有 prompt token 的 KV 都在 cache 里了。
      // 更新最后一个 block 的 fill 计数（最后一个 block 很可能不满）
      int num_blocks = (prompt_len + Config::BLOCK_SIZE - 1) / Config::BLOCK_SIZE;
      int last_block_fill = prompt_len - (num_blocks - 1) * Config::BLOCK_SIZE;
      kv_cache_.fill_last_block(req->seq_id, last_block_fill);

      // 记录第一个生成的 token（prefill 完就能产出第一个 output token）
      req->output_tokens.push_back(first_token);
      stats.total_decode_tokens++;

      // 切到 decode 状态
      scheduler_.update_after_step(req);
    }

    // 3. Decode: 每个请求生成一个 token
    for (auto* req : sched_output.decode_batch) {
      int ctx_len = req->context_len();

      // 当前最后一个 block 满了，先加一个新的物理块
      if (kv_cache_.is_last_block_full(req->seq_id)) {
        int new_block = kv_cache_.append_block(req->seq_id);
        if (new_block < 0) {
          // OOM 了——前面 scheduler 已经检查过，理论上不该走到这里。
          // 万一真发生了就跳过（生产代码应该报错 + abort request）
          continue;
        }
      }

      // 模拟一次 decode 的注意力计算
      int new_token = attention_.simulate_decode(
          ctx_len, (int)sched_output.decode_batch.size());

      // 标记最后一个 block 又多了一个 token
      kv_cache_.fill_last_block(req->seq_id, 1);

      req->output_tokens.push_back(new_token);
      stats.total_decode_tokens++;

      // 到 max_len 了就标记完成
      if (req->reached_max_len()) {
        scheduler_.mark_finished(req);
      }
    }
  }

  // ========== 几个 getter ==========

  KVCacheManager& kv_cache() { return kv_cache_; }
  Scheduler& scheduler() { return scheduler_; }
  bool is_running() const { return running_; }

 private:
  KVCacheManager kv_cache_;
  Scheduler scheduler_;
  MockAttention attention_;
  std::atomic<bool> running_;
};

}  // namespace mini_serve
