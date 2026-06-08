#pragma once

#include "block.h"

#include <chrono>
#include <cmath>
#include <thread>
#include <vector>
#include <random>

namespace mini_serve {

// Mock Attention —— 假装在做 attention，其实只是 sleep 对应时长。
//
// 为什么不用真的？因为这个项目的重点不是模型的数值对不对，而是
// KV cache 管理和调度策略。只要把 prefill/decode 的计算耗时特征抓对，
// 调度器的行为就会和真实场景差不多。
//
// 真实的 attention kernel 怎么写参考 cuda/ 目录里的 .cu 文件。
//
// Prefill 和 decode 的性能特征完全不一样——这个差异是很多
// LLM serving 优化（包括 continuous batching）的出发点:
//
// ── Prefill: compute-bound ──────────────────────────
//   一次性把整个 prompt 的 QK^T 算出来。这是个大的矩阵乘法
//   (seq_len × seq_len 的 attention score matrix)。
//   瓶颈在 GPU 的浮点算力（FLOPS），显存带宽还顶得住。
//   所以 prefill 的时候把多个请求放在一个 batch 里并行不明显
//   加速每个请求，但能提高 GPU 总体利用率。
//
//   这里模拟为: time ∝ prompt_len^2（self-attention 是 O(n²) 的）
//
// ── Decode: memory-bound ───────────────────────────
//   每步只算一个新 token 的 query，但从显存里把整个 context
//   的 K-cache 和 V-cache 全读出来做 attention。
//   context 越长，要读的数据越多——但计算量很小，时间全花在等
//   显存带宽上了。batch 越大越好，因为一次 kernel launch
//   可以摊分 KV cache 的读取开销。
//
//   这里模拟为: time ∝ context_len / sqrt(batch_size)
//   sqrt 是个经验近似——batch 越大带宽利用率越高，但收益递减。
//
// 两个阶段都加了点高斯噪声，模仿真实 GPU 上的各种波动:
// kernel launch time、显存访问竞争、其他 GPU 任务干扰等。
class MockAttention {
 public:
  struct TimingConfig {
    double prefill_us_per_token_sq;  // prefill: 每个 token² 的微秒数
    double decode_us_per_token;      // decode: 每个 context token 的微秒数
    double overhead_us;              // kernel launch 等固定开销
    bool add_noise;                  // 要不要加随机抖动

    TimingConfig()
        : prefill_us_per_token_sq(0.001),
          decode_us_per_token(0.5),
          overhead_us(50.0),
          add_noise(true) {}
  };

  explicit MockAttention(TimingConfig config = TimingConfig()) : config_(config) {}

  // 模拟 prefill: 一次处理整个 prompt，产出第一个 token。
  //
  // 计算量估算（简化到极致）:
  //   核心是 Q @ K^T，Q 和 K 都是 [prompt_len × head_dim] 的矩阵，
  //   所以是 prompt_len × head_dim × prompt_len 次乘法，也就是
  //   O(prompt_len²)。softmax 和乘 V 差不多也是这个量级。
  //   所以直接取 k * prompt_len²。
  //
  // batch_size 对 prefill 的影响不大，因为每个序列的 self-attention
  // 是独立的（prefill 阶段没有跨序列的 attention，除非是 PPO/DPO 那种）。
  // 所以公式里没放 batch_size，只在 overhead 上有微弱影响。
  int simulate_prefill(int prompt_len, int batch_size) {
    double compute_us = config_.prefill_us_per_token_sq * prompt_len * prompt_len
                        + config_.overhead_us;
    if (config_.add_noise) {
      compute_us *= (1.0 + noise(0.1));  // ±10% 的随机抖动
    }
    sleep_us(compute_us);
    return random_token();
  }

  // 模拟 decode: 基于当前 context 生成一个新 token。
  //
  // Decode 的计算:
  //   Q @ K^T: Q 是 1×head_dim，K 是 context_len×head_dim
  //   → 1 × head_dim × context_len = O(context_len) 次乘法
  //   softmax + weighted V 也是 O(context_len)。
  //
  // 这点计算量跟 KV cache 的读取量比不值一提。每次 decode 要读:
  //   context_len × num_heads × head_dim × 2(K+V) × 2 bytes(fp16)
  //   以 Llama-7B 为例: 1 token × 32 heads × 128 dim × 4 bytes = 16KB
  //   1000 token context = 16MB，per layer，32 层 = 512MB
  //   这全是显存带宽的事了，跟算力没关系。
  //
  // batch 越大越好是因为同一次 kernel launch 可以服务多个序列，
  // 带宽利用率更高。但收益递减——batch 从 1 到 4 提升很大，
  // 从 32 到 128 提升就没那么多了，所以用 sqrt 而不是线性。
  int simulate_decode(int context_len, int batch_size) {
    double memory_us = config_.decode_us_per_token * context_len
                       / std::sqrt(batch_size)
                       + config_.overhead_us;
    if (config_.add_noise) {
      memory_us *= (1.0 + noise(0.05));  // decode 波动比 prefill 小
    }
    sleep_us(memory_us);
    return random_token();
  }

  // 混合 iteration 的耗时估算（splitfuse 风格）。
  // 真实系统可以把 prefill 和 decode 放在同一个 iteration 里执行，
  // 让 GPU 资源利用率更高。这里每个 iteration 可能同时有 prefill
  // 和 decode 的请求。
  void simulate_iteration(int num_prefill_tokens, int num_decode_tokens,
                          int batch_size) {
    double total_us = 0;
    if (num_prefill_tokens > 0) {
      total_us += config_.prefill_us_per_token_sq * num_prefill_tokens * 4
                  + config_.overhead_us;
    }
    if (num_decode_tokens > 0) {
      total_us += config_.decode_us_per_token * num_decode_tokens * 64
                  / std::sqrt(batch_size)
                  + config_.overhead_us;
    }
    if (config_.add_noise) {
      total_us *= (1.0 + noise(0.08));
    }
    sleep_us(total_us);
  }

 private:
  // 微秒 → chrono duration → sleep
  void sleep_us(double us) {
    if (us > 0) {
      std::this_thread::sleep_for(
          std::chrono::microseconds(static_cast<int64_t>(us)));
    }
  }

  // 高斯噪声: N(0, scale)，用来模拟真实 GPU 上的波动
  double noise(double scale) {
    static thread_local std::mt19937 rng(42);   // 固定种子，结果可复现
    std::normal_distribution<double> dist(0, scale);
    return dist(rng);
  }

  // 返回一个假的 token id，1~32000 之间（大概 vocab 大小）
  int random_token() {
    static thread_local std::mt19937 rng(123);
    std::uniform_int_distribution<int> dist(1, 32000);
    return dist(rng);
  }

  TimingConfig config_;
};

}  // namespace mini_serve
