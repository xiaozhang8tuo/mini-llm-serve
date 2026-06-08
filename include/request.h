#pragma once

#include "block.h"

#include <chrono>
#include <string>
#include <vector>

namespace mini_serve {

// 请求的状态机。
//
// 一个请求从进来到出去，正常路径是:
//   WAITING → PREFILL → DECODING → FINISHED
//
// 但还有个 PREEMPTED 状态——KV cache 不够的时候，调度器可能把正在
// decode 的请求踢出去，释放它的 KV cache，让它滚回等待队列，下次
// 重新 prefill。这是一种"宁可重算也不 OOM"的策略，反正 prefill 是
// compute-bound，重算一遍也还好，总比崩了好。
//
// 各种状态的意思:
//   WAITING:   在排队，还没开始处理。这个阶段不占 KV cache
//   PREFILL:   正在一次性处理整个 prompt，把 prompt 所有 token 的
//               K、V 全算出来塞进 cache。这个过程很快（GPU 算力吃满）
//   DECODING:  进入逐 token 生成阶段。每步只算一个新 token 的 Q，
//              然后从 KV cache 里把整个 context 的 K、V 读出来做
//              attention。这个过程慢（卡在显存带宽上）
//   PREEMPTED: 被踢了，KV cache 已释放，下次要从头重新 prefill
//   FINISHED:  完了（到了 max_len 或者生成 EOS）
//
// 状态流转:
//   WAITING ──(调度在)──> PREFILL ──(prefill完成)──> DECODING
//   DECODING ──(没内存了被踢)──> PREEMPTED ──> WAITING ──> ...
//   DECODING ──(跑完了)──> FINISHED
//   PREFILL ──(跑完了)──> DECODING（正常不会从 PREFILL 直接结束）
enum class RequestState {
  WAITING,
  PREFILL,
  DECODING,
  PREEMPTED,
  FINISHED,
};

// 一个推理请求。字段比较多，真实系统里这些信息会拆到不同模块里，
// 这里偷懒全塞一个 struct 了。
//
// 你可能会问 prompt_tokens 里的 token id 是哪来的——真实场景是
// HTTP API 收到的请求里带的，经过 tokenizer 编码。我们这里不管
// tokenizer，直接手动造 token id。
struct Request {
  int request_id;
  RequestState state;

  // ---- 输入 ----
  std::vector<int> prompt_tokens;    // prompt 的 token id 序列
  int max_output_len;                // 最多生成多少个 token

  // ---- 运行时状态 ----
  std::vector<int> output_tokens;    // 已经生成的 token，逐个追加
  int num_computed_tokens;           // prefill 阶段已经处理了几个 prompt token
                                     // 正常 prefill完后应该等于 prompt_tokens.size()
                                     // 被 preempt 后清零，重来

  // ---- 调度相关 ----
  int seq_id;       // KV cache 里的序列 ID，这里简化成 seq_id = request_id
                    // 真实系统里一个 request 可能对应多个 seq（beam search）
  int priority;     // 优先级，0 最高。做 preemption 决策时可能用到
                    // 目前还没真用上，先留个口子

  // ---- 时间戳，用来算延迟指标 ----
  // steady_clock 保证单调递增，不会被 NTP 对时之类的搞乱
  using Clock = std::chrono::steady_clock;
  Clock::time_point arrival_time;       // 请求到达的时间
  Clock::time_point first_token_time;   // 第一个 token 产出的时间（prefill 完成那刻）
  Clock::time_point finish_time;        // 生成结束的时间

  Request(int id, std::vector<int> prompt, int max_out)
      : request_id(id),
        state(RequestState::WAITING),
        prompt_tokens(std::move(prompt)),
        max_output_len(max_out),
        num_computed_tokens(0),
        seq_id(id),   // 直接拿 request_id 当 seq_id，反正 demo 里一一对应
        priority(0),
        arrival_time(Clock::now()) {}

  // 当前 context 长度 = prompt 已计算的 + 已生成输出的。
  // 这个值很重要——decode 阶段的 attention 要遍历整个 context 的 KV cache，
  // context 越长越慢（memory-bound）。
  int context_len() const {
    return num_computed_tokens + (int)output_tokens.size();
  }

  // prefill 还剩几个 token 没处理。到 0 了才能切 DECODING。
  int remaining_prefill() const {
    return (int)prompt_tokens.size() - num_computed_tokens;
  }

  bool is_finished() const {
    return state == RequestState::FINISHED;
  }

  bool reached_max_len() const {
    return (int)output_tokens.size() >= max_output_len;
  }

  // ---- 下面几个是延迟指标，LLM serving 里最常看的三个数 ----

  // TTFT (Time To First Token): 从请求到达到第一个 token 出来的时间。
  // 这是用户体感最重要的指标——你点了发送，多久才能看到第一个字开始冒？
  // 主要由排队 + prefill 时间决定。
  double ttft_ms() const {
    return std::chrono::duration<double, std::milli>(
               first_token_time - arrival_time).count();
  }

  // 端到端延迟: 从请求到达到最后一个 token 出来。
  double total_latency_ms() const {
    return std::chrono::duration<double, std::milli>(
               finish_time - arrival_time).count();
  }

  // TPOT (Time Per Output Token): decode 阶段平均每个 token 要多久。
  // TPOT 越低说明 decode 越快。对比 TTFT：TTFT 看的是"首字响应"，
  // TPOT 看的是"生成速度"——前者高了用户觉得卡，后者低了用户觉得慢。
  double tpot_ms() const {
    if (output_tokens.empty()) return 0;
    double decode_time = std::chrono::duration<double, std::milli>(
                             finish_time - first_token_time).count();
    return decode_time / output_tokens.size();
  }
};

}  // namespace mini_serve
