// mini-llm-serve: 一个不跑真模型的 LLM 推理引擎原型。
// 用 mock attention 假装在算，实际在验证 KV cache 分页管理和
// Continuous Batching 调度对吞吐、延迟的影响。
//
// 用法: ./mini-llm-serve [并发请求数] [每个请求最多生成的 token 数]
// 默认 50 个请求，每个最多 64 个 token。
//
// 跑完会打印:
//   1. 吞吐、延迟（TTFT/TPOT）统计
//   2. Continuous Batching vs Static Batching 对比

#include "engine.h"
#include "request.h"

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

using namespace mini_serve;

// 造一批假请求。prompt 里全是 token id=1，反正 attention 是 mock 的，
// token id 是多少都一样。prompt 长度在 32~512 之间随机，模拟真实场景
// 里有人发了一句话也有人发了一大段的情况。
std::vector<Request> generate_requests(int num_requests, int max_output_len) {
  std::mt19937 rng(2024);  // 种子固定，每次跑结果一样
  std::uniform_int_distribution<int> prompt_len_dist(32, 512);

  std::vector<Request> requests;
  requests.reserve(num_requests);
  for (int i = 0; i < num_requests; i++) {
    int prompt_len = prompt_len_dist(rng);
    std::vector<int> prompt(prompt_len, 1);
    requests.emplace_back(i, std::move(prompt), max_output_len);
  }
  return requests;
}

// 打印统计结果。关注的几个指标:
//
//   Throughput (tokens/sec):
//     每秒能生成多少 token。越高越好，直接反映系统产能。
//
//   TTFT (Time To First Token):
//     从请求到达到看到第一个 token 的时间。对用户体验最关键——
//     你跟 AI 聊天，点了发送之后多久看到它开始回？TTFT 高了
//     你会觉得它卡住了。
//
//   TPOT (Time Per Output Token):
//     decode 阶段平均每个 token 的生成时间。反映的是 decode 效率。
//     TPOT 高 = 生成慢 = token 一个字一个字往外蹦。
//
//   KV Cache 使用:
//     显存占了多少。这个和吞吐有一个 tradeoff——KV cache 越多
//     能同时跑的请求越多（吞吐高），但留给模型权重的空间越少。
void print_stats(const std::vector<Request>& requests,
                 const Engine::Stats& stats) {
  std::cout << "\n========== Benchmark Results ==========\n";
  std::cout << "配置:\n";
  std::cout << "  Block Size:       " << Config::BLOCK_SIZE << " tokens\n";
  std::cout << "  Num Blocks:       " << Config::MAX_NUM_BLOCKS << "\n";
  std::cout << "  Max Batch Size:   " << Config::MAX_BATCH_SIZE << "\n";
  std::cout << "  Num Requests:     " << requests.size() << "\n";

  // 吞吐
  std::cout << "\n吞吐:\n";
  std::cout << "  Total Time:       " << std::fixed << std::setprecision(1)
            << stats.total_time_ms << " ms\n";
  std::cout << "  Total Iterations: " << stats.total_iterations << "\n";
  std::cout << "  Prefill Tokens:   " << stats.total_prefill_tokens << "\n";
  std::cout << "  Decode Tokens:    " << stats.total_decode_tokens << "\n";
  double throughput = stats.total_decode_tokens / (stats.total_time_ms / 1000.0);
  std::cout << "  Throughput:       " << std::fixed << std::setprecision(1)
            << throughput << " tokens/sec\n";

  // 延迟: 遍历全部完成的请求，算平均和最大值
  double total_ttft = 0, total_tpot = 0, total_latency = 0;
  double max_ttft = 0, max_tpot = 0;
  int num_finished = 0;
  for (auto& req : requests) {
    if (req.state == RequestState::FINISHED) {
      num_finished++;
      double ttft = req.ttft_ms();
      double tpot = req.tpot_ms();
      double lat = req.total_latency_ms();
      total_ttft += ttft;
      total_tpot += tpot;
      total_latency += lat;
      max_ttft = std::max(max_ttft, ttft);
      max_tpot = std::max(max_tpot, tpot);
    }
  }

  if (num_finished > 0) {
    std::cout << "\n延迟 (已完成 " << num_finished << " 请求):\n";
    std::cout << "  Avg TTFT:         " << std::fixed << std::setprecision(2)
              << total_ttft / num_finished << " ms\n";
    std::cout << "  Max TTFT:         " << std::fixed << std::setprecision(2)
              << max_ttft << " ms\n";
    std::cout << "  Avg TPOT:         " << std::fixed << std::setprecision(2)
              << total_tpot / num_finished << " ms\n";
    std::cout << "  Max TPOT:         " << std::fixed << std::setprecision(2)
              << max_tpot << " ms\n";
    std::cout << "  Avg Latency:      " << std::fixed << std::setprecision(1)
              << total_latency / num_finished << " ms\n";
  }

  // 显存使用（模拟值）
  std::cout << "\nKV Cache 使用:\n";
  std::cout << "  每个 Block 大小:  " << Config::BLOCK_SIZE << " tokens × "
            << Config::NUM_LAYERS << " layers × 2(K+V) × "
            << Config::NUM_HEADS << " heads × " << Config::HEAD_DIM
            << " dim × 2 bytes = "
            << (2 * Config::NUM_LAYERS * Config::NUM_HEADS *
                Config::BLOCK_SIZE * Config::HEAD_DIM * 2 / 1024 / 1024)
            << " MB\n";
  std::cout << "  总显存 (模拟):    "
            << (2L * Config::NUM_LAYERS * Config::NUM_HEADS *
                Config::MAX_NUM_BLOCKS * Config::BLOCK_SIZE *
                Config::HEAD_DIM * 2 / 1024 / 1024)
            << " MB\n";

  std::cout << "========================================\n";
}

// 同一批请求分别跑 Continuous Batching 和 Static Batching，对比吞吐和延迟。
//
// 预期结果:
//   Continuous Batching 的吞吐明显更高，因为:
//     1. 短请求跑完立刻腾出 batch slot，新请求马上进来——GPU 不等
//     2. 不用等 batch 里最长的那个——没有 padding
//     3. KV cache 按需分配，更多人能同时跑
//
//   Static Batching 的问题是:
//     一个 batch 固定 N 个，必须等最慢的那个跑完才换下一批。
//     短请求早就完了但 slot 空着（padding），白浪费。
void compare_batching_strategies(int num_requests, int max_output_len) {
  std::cout << "\n\n>>>>>>>>>> 对比: Static Batching vs Continuous Batching <<<<<<<<<<\n";

  // --- Continuous Batching ---
  std::cout << "\n[Continuous Batching]\n";
  {
    auto requests = generate_requests(num_requests, max_output_len);
    Engine engine;

    // 所有请求同时到达，模拟突发流量
    for (auto& req : requests) {
      engine.submit_request(&req);
    }

    auto stats = engine.run();
    print_stats(requests, stats);
  }

  // --- Static Batching ---
  // 自己实现 static batching，不经过 Engine。
  // batch_size 固定 8，每个 batch 先串行 prefill，再所有请求一起 decode，
  // decode 阶段必须等 batch 里最长的那个跑完 max_output_len 步。
  std::cout << "\n[Static Batching (batch_size=8, 必须等最长序列完成)]\n";
  {
    auto requests = generate_requests(num_requests, max_output_len);
    int static_batch_size = 8;
    auto start = std::chrono::steady_clock::now();
    int total_decode_tokens = 0;
    int total_iterations = 0;

    MockAttention attention;

    for (int batch_start = 0; batch_start < num_requests;
         batch_start += static_batch_size) {
      int batch_end = std::min(batch_start + static_batch_size, num_requests);
      int bs = batch_end - batch_start;

      // Prefill: 每个请求单独做
      for (int i = batch_start; i < batch_end; i++) {
        int prompt_len = (int)requests[i].prompt_tokens.size();
        requests[i].first_token_time = Request::Clock::now();
        attention.simulate_prefill(prompt_len, 1);
        requests[i].output_tokens.push_back(1);
        total_decode_tokens++;
      }

      // Decode: 每个 step 所有人都要跑一遍 decode。
      // 就算有些请求已经结束了，也假模假式地跑 simulate_decode——
      // 这就是 static batching 最大的浪费: 已经完成的 slot 还要空转。
      for (int step = 1; step < max_output_len; step++) {
        total_iterations++;
        for (int i = batch_start; i < batch_end; i++) {
          int ctx_len = (int)requests[i].prompt_tokens.size() + step;
          attention.simulate_decode(ctx_len, bs);
          requests[i].output_tokens.push_back(1);
          total_decode_tokens++;
        }
      }

      // 这批全部标记完成
      for (int i = batch_start; i < batch_end; i++) {
        requests[i].state = RequestState::FINISHED;
        requests[i].finish_time = Request::Clock::now();
      }
    }

    auto end = std::chrono::steady_clock::now();
    double total_ms =
        std::chrono::duration<double, std::milli>(end - start).count();
    double throughput = total_decode_tokens / (total_ms / 1000.0);

    std::cout << "  Total Time:       " << std::fixed << std::setprecision(1)
              << total_ms << " ms\n";
    std::cout << "  Decode Tokens:    " << total_decode_tokens << "\n";
    std::cout << "  Throughput:       " << std::fixed << std::setprecision(1)
              << throughput << " tokens/sec\n";

    // 平均延迟
    double total_latency = 0;
    for (int i = 0; i < num_requests; i++) {
      total_latency += requests[i].total_latency_ms();
    }
    std::cout << "  Avg Latency:      " << std::fixed << std::setprecision(1)
              << total_latency / num_requests << " ms\n";
  }
}

int main(int argc, char** argv) {
  int num_requests = 50;
  int max_output_len = 64;

  if (argc > 1) num_requests = std::atoi(argv[1]);
  if (argc > 2) max_output_len = std::atoi(argv[2]);

  std::cout << "=== Mini LLM Serving Engine ===\n";
  std::cout << "模拟 " << num_requests << " 个并发请求, 每个最多生成 "
            << max_output_len << " tokens\n";
  std::cout << "KV Cache: " << Config::MAX_NUM_BLOCKS << " blocks × "
            << Config::BLOCK_SIZE << " tokens/block = "
            << Config::MAX_NUM_BLOCKS * Config::BLOCK_SIZE << " token 容量\n\n";

  // 先单独跑一遍 continuous batching
  {
    auto requests = generate_requests(num_requests, max_output_len);
    Engine engine;
    for (auto& req : requests) {
      engine.submit_request(&req);
    }
    auto stats = engine.run();
    print_stats(requests, stats);
  }

  // 再跑对比实验（continuous vs static）
  compare_batching_strategies(num_requests, max_output_len);

  return 0;
}
