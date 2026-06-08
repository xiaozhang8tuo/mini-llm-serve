#pragma once

#include "block.h"
#include "kv_cache_manager.h"
#include "request.h"

#include <algorithm>
#include <deque>
#include <mutex>
#include <vector>

namespace mini_serve {

// 每次 schedule() 调度的结果。里面有三拨请求:
//   - prefill_batch: 这一轮要 prefill 的（新来的或者被 preempt 后恢复的）
//   - decode_batch:  这一轮要继续 decode 的
//   - preempted:     这一轮因为内存不够被踢掉的
struct SchedulerOutput {
  std::vector<Request*> prefill_batch;
  std::vector<Request*> decode_batch;
  std::vector<Request*> preempted;
};

// Continuous Batching 调度器，想法来自 Orca 论文(OSDI 2022)。
//
// 传统 static batching 的问题一眼就能看出来: 固定 batch 大小，
// 必须等 batch 里最慢的那个请求跑完，全 batch 才能一起退出。
// 短的早就完了但空等着（padding），GPU 浪费在发呆上。
//
// Orca 的想法: 每个 iteration（一次 forward）跑完就重新决定 batch 里
// 该有谁。完成的滚蛋，等着的新请求见缝插针进来。这样 GPU 利用率最高，
// 不需要等任何人。
//
// 几个关键设计点:
//   1. iteration-level scheduling: 每个 step 都调一次 schedule() 重新组 batch
//   2. prefill 优先: 新请求排在 decode 前面，优先降 TTFT
//   3. 内存感知: 根据 free_blocks 数量判断能不能放新请求进来
//   4. preemption: 实在没内存了就踢 decode 请求（FIFO 倒序——最后来的先踢，
//      因为它"投入"的 decode 轮次最少，recompute 的损失也最小）
class Scheduler {
 public:
  explicit Scheduler(KVCacheManager& kv_cache,
                     int max_batch_size = Config::MAX_BATCH_SIZE)
      : kv_cache_(kv_cache), max_batch_size_(max_batch_size) {}

  // 新请求扔进等待队列
  void add_request(Request* req) {
    std::lock_guard<std::mutex> lock(mutex_);
    req->state = RequestState::WAITING;
    waiting_queue_.push_back(req);
  }

  // 核心: 每个 iteration 调一次。返回这轮谁 prefill、谁 decode、谁被踢。
  //
  // 调度分两步:
  //
  // Step 1 — 保住 decode 请求:
  //   已经在跑的 decode 请求，每轮只需要 0 或 1 个新 block
  //   （最后一个 block 满了就 +1，不满就 +0）。
  //   如果 free_blocks 连这 1 个都给不出了，就 preempt 它——踢回等待队列。
  //   踢的顺序按 FIFO 倒序（后进先踢），因为刚进来没多久的请求重算成本低。
  //
  // Step 2 — 接纳新请求做 prefill:
  //   waiting 队列里的请求，按 prompt 长度估算需要几个 block。
  //   （比如 prompt 60 token, BLOCK_SIZE=16 → 需要 ceil(60/16)=4 个 block）
  //   够就放进来，不够就停了——连这个都不够，后面更大的大概率也不够。
  //   真实系统可以做 reordering，把小请求先提上来，但这里先不管。
  SchedulerOutput schedule() {
    std::lock_guard<std::mutex> lock(mutex_);
    SchedulerOutput output;

    // === Step 1: 先保 decode 请求 ===
    std::vector<Request*> kept_running;
    for (auto* req : running_queue_) {
      int needed = kv_cache_.calc_needed_blocks(req->seq_id, 1);
      if (needed <= kv_cache_.get_num_free_blocks()) {
        // 内存够，留着继续 decode
        kept_running.push_back(req);
        output.decode_batch.push_back(req);
      } else {
        // 内存不够，踢掉
        preempt_request(req);
        output.preempted.push_back(req);
      }

      // batch 也别太大，到了上限就停
      if ((int)output.decode_batch.size() >= max_batch_size_) break;
    }
    running_queue_ = kept_running;

    // === Step 2: 接纳等待中的请求 ===
    int remaining_slots = max_batch_size_ - (int)output.decode_batch.size();
    auto it = waiting_queue_.begin();
    while (it != waiting_queue_.end() && remaining_slots > 0) {
      Request* req = *it;

      // 向上取整: prompt 需要多少个 block
      int prompt_len = (int)req->prompt_tokens.size();
      int needed_blocks = (prompt_len + Config::BLOCK_SIZE - 1) / Config::BLOCK_SIZE;

      if (needed_blocks <= kv_cache_.get_num_free_blocks()) {
        // 够，放进来 prefill
        req->state = RequestState::PREFILL;

        // 分配 KV cache: 先建 block table，再逐个 append 物理块
        kv_cache_.allocate_sequence(req->seq_id);
        for (int i = 0; i < needed_blocks; i++) {
          kv_cache_.append_block(req->seq_id);
        }

        output.prefill_batch.push_back(req);
        it = waiting_queue_.erase(it);
        remaining_slots--;
      } else {
        // 这个都不够，后面可能更不够（假设 prompt 长度差不多）。
        // 真实系统这里可以 skip 掉大的，把小的先放进来，没做。
        break;
      }
    }

    return output;
  }

  // engine 每步跑完之后调这个，更新请求的状态和时间戳。
  // prefill 完了 → 切 DECODING + 记录 TTFT
  // decode 步 → 检查是不是到 max_len 了，是就结束
  void update_after_step(Request* req) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (req->state == RequestState::PREFILL) {
      // prefill 完啦，进入逐 token 生成模式
      req->state = RequestState::DECODING;
      req->num_computed_tokens = (int)req->prompt_tokens.size();
      req->first_token_time = Request::Clock::now();  // 记 TTFT
      running_queue_.push_back(req);
    } else if (req->state == RequestState::DECODING) {
      if (req->reached_max_len()) {
        finish_request(req);
      }
      // 没到 max，继续留在 running_queue_
    }
  }

  // 内部用的 finish（不加锁）
  void finish_request(Request* req) {
    req->state = RequestState::FINISHED;
    req->finish_time = Request::Clock::now();
    kv_cache_.free_sequence(req->seq_id);  // 回收它的 KV cache
    running_queue_.erase(
        std::remove(running_queue_.begin(), running_queue_.end(), req),
        running_queue_.end());
    finished_queue_.push_back(req);
  }

  // 带锁的版本给外部用
  void mark_finished(Request* req) {
    std::lock_guard<std::mutex> lock(mutex_);
    finish_request(req);
  }

  // ========== 几个查询 ==========

  int num_waiting() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return (int)waiting_queue_.size();
  }

  int num_running() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return (int)running_queue_.size();
  }

  int num_finished() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return (int)finished_queue_.size();
  }

  // engine 的主循环用这个判断要不要停: 没人排队也没人在跑了就停
  bool has_pending_work() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return !waiting_queue_.empty() || !running_queue_.empty();
  }

  const std::vector<Request*>& get_finished() const {
    return finished_queue_;
  }

 private:
  // 抢占一个请求: 释放它的 KV cache，清掉生成的内容，
  // 塞回队列最前面（让它尽快被重新 prefill，不用再从队尾排起）。
  // recompute 策略: 下次恢复时从头 prefill 一遍，不用 checkpoint。
  void preempt_request(Request* req) {
    req->state = RequestState::PREEMPTED;
    kv_cache_.free_sequence(req->seq_id);
    waiting_queue_.push_front(req);   // 排最前面，优先恢复
    req->num_computed_tokens = 0;     // 重新来过
    req->output_tokens.clear();
  }

  KVCacheManager& kv_cache_;
  int max_batch_size_;

  std::deque<Request*> waiting_queue_;     // 等着 prefill 的
  std::vector<Request*> running_queue_;    // 正在 decode 的
  std::vector<Request*> finished_queue_;   // 跑完的，留着收集统计

  mutable std::mutex mutex_;
};

}  // namespace mini_serve
