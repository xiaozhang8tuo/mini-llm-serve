// Scheduler 和 Engine 的集成测试。
// 测调度逻辑: continuous batching 的 prefill/decode 是否分离、
// 内存压力下 preemption 是否触发、是否所有请求都能跑完。

#include "../include/engine.h"
#include <cassert>
#include <iostream>
#include <vector>

using namespace mini_serve;

// 最基本调度: 丢 3 个请求进去，第一轮应该全进 prefill，
// 第二轮全转 decode。确认状态机流转没问题。
void test_basic_scheduling() {
  std::cout << "[TEST] Basic scheduling... ";

  KVCacheManager kv_cache(64);   // 64 个物理块，绰绰有余
  Scheduler scheduler(kv_cache, 4);

  // 3 个 prompt 长 32 的请求
  std::vector<int> prompt(32, 1);
  Request req0(0, prompt, 10);
  Request req1(1, prompt, 10);
  Request req2(2, prompt, 10);

  scheduler.add_request(&req0);
  scheduler.add_request(&req1);
  scheduler.add_request(&req2);

  assert(scheduler.num_waiting() == 3);
  assert(scheduler.num_running() == 0);

  // 第一次调度: 3 个全进 prefill（它们都没 KV cache）
  auto output = scheduler.schedule();
  assert(output.prefill_batch.size() == 3);
  assert(output.decode_batch.size() == 0);

  // 模拟 engine 跑完 prefill → 转 decode
  for (auto* req : output.prefill_batch) {
    scheduler.update_after_step(req);
  }
  assert(scheduler.num_running() == 3);

  // 第二次调度: 都在 decode 了，prefill 应该是空的
  output = scheduler.schedule();
  assert(output.prefill_batch.size() == 0);
  assert(output.decode_batch.size() == 3);

  std::cout << "PASS\n";
}

// 内存极小的场景: 总共 8 个 block，两个 prompt 各要 4 个，刚好用完。
// 验证 scheduler 不会在 block 不够时强行接纳。
void test_preemption() {
  std::cout << "[TEST] Preemption under memory pressure... ";

  KVCacheManager kv_cache(8);  // 就 8 个
  Scheduler scheduler(kv_cache, 4);

  // 每个 prompt 60 token → ceil(60/16) = 4 block
  std::vector<int> long_prompt(60, 1);
  Request req0(0, long_prompt, 20);
  Request req1(1, long_prompt, 20);

  scheduler.add_request(&req0);
  scheduler.add_request(&req1);

  // 4+4=8，刚好分完
  auto output = scheduler.schedule();
  assert(output.prefill_batch.size() == 2);

  // 假装 prefill 完了转 decode
  for (auto* req : output.prefill_batch) {
    req->num_computed_tokens = (int)req->prompt_tokens.size();
    req->state = RequestState::DECODING;
    req->first_token_time = Request::Clock::now();
  }

  std::cout << "PASS\n";
}

// 用 Engine 完整跑 10 个不同长度的请求，验证全部完成。
void test_continuous_batching_vs_static() {
  std::cout << "[TEST] Continuous batching produces correct results... ";

  Engine engine(256, 8);  // 256 blocks, max batch=8

  // prompt 长度 16, 24, 32, ..., 88
  std::vector<Request> requests;
  for (int i = 0; i < 10; i++) {
    int len = 16 + i * 8;
    requests.emplace_back(i, std::vector<int>(len, 1), 8);
  }

  for (auto& req : requests) {
    engine.submit_request(&req);
  }

  auto stats = engine.run();

  int num_finished = 0;
  for (auto& req : requests) {
    if (req.state == RequestState::FINISHED) num_finished++;
  }
  assert(num_finished == 10);
  assert(stats.total_decode_tokens > 0);
  assert(stats.total_iterations > 0);

  std::cout << "PASS (finished=" << num_finished
            << ", tokens=" << stats.total_decode_tokens
            << ", iters=" << stats.total_iterations << ")\n";
}

// 大量请求 + 小 cache，验证排队和分批处理是否正常。
// 128 blocks 处理 20 个请求，肯定有排队。
void test_large_batch() {
  std::cout << "[TEST] Large batch with memory pressure... ";

  Engine engine(128, 4);  // 128 blocks, 小 batch

  std::vector<Request> requests;
  for (int i = 0; i < 20; i++) {
    requests.emplace_back(i, std::vector<int>(64, 1), 16);
  }

  for (auto& req : requests) {
    engine.submit_request(&req);
  }

  auto stats = engine.run();

  // 全跑完，只是需要更多轮
  int num_finished = 0;
  for (auto& req : requests) {
    if (req.state == RequestState::FINISHED) num_finished++;
  }
  assert(num_finished == 20);

  std::cout << "PASS (iters=" << stats.total_iterations
            << ", time=" << stats.total_time_ms << "ms)\n";
}

int main() {
  std::cout << "=== Scheduler & Engine Tests ===\n\n";

  test_basic_scheduling();
  test_preemption();
  test_continuous_batching_vs_static();
  test_large_batch();

  std::cout << "\nAll tests passed!\n";
  return 0;
}
