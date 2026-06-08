cmake_minimum_required(VERSION 3.14)
project(mini-llm-serve CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 主程序
add_executable(mini-llm-serve src/main.cpp)
target_include_directories(mini-llm-serve PRIVATE include)
target_link_libraries(mini-llm-serve PRIVATE pthread)

# 测试: KV Cache
add_executable(test_kv_cache test/test_kv_cache.cpp)
target_include_directories(test_kv_cache PRIVATE include)
target_link_libraries(test_kv_cache PRIVATE pthread)

# 测试: Scheduler
add_executable(test_scheduler test/test_scheduler.cpp)
target_include_directories(test_scheduler PRIVATE include)
target_link_libraries(test_scheduler PRIVATE pthread)

# 演示: PagedAttention kernel walk-through
add_executable(paged_attention_demo test/paged_attention_walkthrough.cpp)

# CTest
enable_testing()
add_test(NAME kv_cache_test COMMAND test_kv_cache)
add_test(NAME scheduler_test COMMAND test_scheduler)
