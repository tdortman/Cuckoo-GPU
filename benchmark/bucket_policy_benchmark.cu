#include <benchmark/benchmark.h>
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#include <bucket_policies.cuh>
#include <CuckooFilter.cuh>
#include <helpers.cuh>
#include <new>
#include "benchmark_common.cuh"

namespace bm = benchmark;

constexpr size_t FIXED_CAPACITY = 1ULL << 28;
constexpr double LOAD_FACTOR = 0.95;

using XorConfig = cuckoogpu::Config<uint64_t, 16, 500, 128, 16, cuckoogpu::XorAltBucketPolicy>;
using AddSubConfig =
    cuckoogpu::Config<uint64_t, 16, 500, 128, 16, cuckoogpu::AddSubAltBucketPolicy>;
using OffsetConfig =
    cuckoogpu::Config<uint64_t, 16, 500, 128, 16, cuckoogpu::OffsetAltBucketPolicy>;

template <typename ConfigType>
class PolicyBenchmarkFixture : public benchmark::Fixture {
    using benchmark::Fixture::SetUp;
    using benchmark::Fixture::TearDown;

   public:
    void SetUp(const benchmark::State& state) override {
        auto [cap, num] = calculateCapacityAndSize(state.range(0), LOAD_FACTOR);
        capacity = cap;
        n = num;

        d_keys.resize(n);
        d_keysNegative.resize(n);
        d_output.resize(n);

        generateKeysGPURange(d_keys, n, uint64_t(0), uint64_t(UINT32_MAX));
        generateKeysGPURange(d_keysNegative, n, uint64_t(UINT32_MAX) + 1, UINT64_MAX);

        filter = std::make_unique<cuckoogpu::Filter<ConfigType>>(capacity);
        filterMemory = filter->sizeInBytes();
    }

    void TearDown(const benchmark::State&) override {
        filter.reset();
        d_keys.clear();
        d_keysNegative.clear();
        d_output.clear();
        d_keys.shrink_to_fit();
        d_keysNegative.shrink_to_fit();
        d_output.shrink_to_fit();
    }

    void setCounters(benchmark::State& state) {
        setCommonCounters(state, filterMemory, n);
        state.counters["num_buckets"] = benchmark::Counter(
            static_cast<double>(filter->getNumBuckets()), benchmark::Counter::kDefaults
        );
    }

    size_t capacity;
    size_t n;
    size_t filterMemory;
    thrust::device_vector<uint64_t> d_keys;
    thrust::device_vector<uint64_t> d_keysNegative;
    thrust::device_vector<uint8_t> d_output;
    std::unique_ptr<cuckoogpu::Filter<ConfigType>> filter;
    GPUTimer timer;
};

#define BENCHMARK_POLICY_CONFIG         \
    ->Arg(FIXED_CAPACITY)               \
        ->Unit(benchmark::kMillisecond) \
        ->UseManualTime()               \
        ->Iterations(10)                \
        ->Repetitions(5)                \
        ->ReportAggregatesOnly(true)

using XorFixture = PolicyBenchmarkFixture<XorConfig>;

BENCHMARK_DEFINE_F(XorFixture, Insert)(bm::State& state) {
    for (auto _ : state) {
        filter->clear();
        cudaDeviceSynchronize();
        timer.start();
        size_t inserted = adaptiveInsert(*filter, d_keys);
        state.SetIterationTime(timer.elapsed());
        bm::DoNotOptimize(inserted);
    }
    setCounters(state);
}

BENCHMARK_DEFINE_F(XorFixture, Query)(bm::State& state) {
    adaptiveInsert(*filter, d_keys);
    cudaDeviceSynchronize();
    for (auto _ : state) {
        timer.start();
        filter->containsMany(d_keys, d_output);
        state.SetIterationTime(timer.elapsed());
        bm::DoNotOptimize(d_output.data().get());
    }
    setCounters(state);
}

BENCHMARK_DEFINE_F(XorFixture, QueryNegative)(bm::State& state) {
    adaptiveInsert(*filter, d_keys);
    cudaDeviceSynchronize();
    for (auto _ : state) {
        timer.start();
        filter->containsMany(d_keysNegative, d_output);
        state.SetIterationTime(timer.elapsed());
        bm::DoNotOptimize(d_output.data().get());
    }
    setCounters(state);
}

BENCHMARK_DEFINE_F(XorFixture, Delete)(bm::State& state) {
    for (auto _ : state) {
        filter->clear();
        adaptiveInsert(*filter, d_keys);
        cudaDeviceSynchronize();
        timer.start();
        size_t remaining = filter->deleteMany(d_keys, d_output);
        state.SetIterationTime(timer.elapsed());
        bm::DoNotOptimize(remaining);
    }
    setCounters(state);
}

BENCHMARK_REGISTER_F(XorFixture, Insert) BENCHMARK_POLICY_CONFIG;
BENCHMARK_REGISTER_F(XorFixture, Query) BENCHMARK_POLICY_CONFIG;
BENCHMARK_REGISTER_F(XorFixture, QueryNegative) BENCHMARK_POLICY_CONFIG;
BENCHMARK_REGISTER_F(XorFixture, Delete) BENCHMARK_POLICY_CONFIG;

using AddSubFixture = PolicyBenchmarkFixture<AddSubConfig>;

BENCHMARK_DEFINE_F(AddSubFixture, Insert)(bm::State& state) {
    for (auto _ : state) {
        filter->clear();
        cudaDeviceSynchronize();
        timer.start();
        size_t inserted = adaptiveInsert(*filter, d_keys);
        state.SetIterationTime(timer.elapsed());
        bm::DoNotOptimize(inserted);
    }
    setCounters(state);
}

BENCHMARK_DEFINE_F(AddSubFixture, Query)(bm::State& state) {
    adaptiveInsert(*filter, d_keys);
    cudaDeviceSynchronize();
    for (auto _ : state) {
        timer.start();
        filter->containsMany(d_keys, d_output);
        state.SetIterationTime(timer.elapsed());
        bm::DoNotOptimize(d_output.data().get());
    }
    setCounters(state);
}

BENCHMARK_DEFINE_F(AddSubFixture, QueryNegative)(bm::State& state) {
    adaptiveInsert(*filter, d_keys);
    cudaDeviceSynchronize();
    for (auto _ : state) {
        timer.start();
        filter->containsMany(d_keysNegative, d_output);
        state.SetIterationTime(timer.elapsed());
        bm::DoNotOptimize(d_output.data().get());
    }
    setCounters(state);
}

BENCHMARK_DEFINE_F(AddSubFixture, Delete)(bm::State& state) {
    for (auto _ : state) {
        filter->clear();
        adaptiveInsert(*filter, d_keys);
        cudaDeviceSynchronize();
        timer.start();
        size_t remaining = filter->deleteMany(d_keys, d_output);
        state.SetIterationTime(timer.elapsed());
        bm::DoNotOptimize(remaining);
    }
    setCounters(state);
}

BENCHMARK_REGISTER_F(AddSubFixture, Insert) BENCHMARK_POLICY_CONFIG;
BENCHMARK_REGISTER_F(AddSubFixture, Query) BENCHMARK_POLICY_CONFIG;
BENCHMARK_REGISTER_F(AddSubFixture, QueryNegative) BENCHMARK_POLICY_CONFIG;
BENCHMARK_REGISTER_F(AddSubFixture, Delete) BENCHMARK_POLICY_CONFIG;

using OffsetFixture = PolicyBenchmarkFixture<OffsetConfig>;

BENCHMARK_DEFINE_F(OffsetFixture, Insert)(bm::State& state) {
    for (auto _ : state) {
        filter->clear();
        cudaDeviceSynchronize();
        timer.start();
        size_t inserted = adaptiveInsert(*filter, d_keys);
        state.SetIterationTime(timer.elapsed());
        bm::DoNotOptimize(inserted);
    }
    setCounters(state);
}

BENCHMARK_DEFINE_F(OffsetFixture, Query)(bm::State& state) {
    adaptiveInsert(*filter, d_keys);
    cudaDeviceSynchronize();
    for (auto _ : state) {
        timer.start();
        filter->containsMany(d_keys, d_output);
        state.SetIterationTime(timer.elapsed());
        bm::DoNotOptimize(d_output.data().get());
    }
    setCounters(state);
}

BENCHMARK_DEFINE_F(OffsetFixture, QueryNegative)(bm::State& state) {
    adaptiveInsert(*filter, d_keys);
    cudaDeviceSynchronize();
    for (auto _ : state) {
        timer.start();
        filter->containsMany(d_keysNegative, d_output);
        state.SetIterationTime(timer.elapsed());
        bm::DoNotOptimize(d_output.data().get());
    }
    setCounters(state);
}

BENCHMARK_DEFINE_F(OffsetFixture, Delete)(bm::State& state) {
    for (auto _ : state) {
        filter->clear();
        adaptiveInsert(*filter, d_keys);
        cudaDeviceSynchronize();
        timer.start();
        size_t remaining = filter->deleteMany(d_keys, d_output);
        state.SetIterationTime(timer.elapsed());
        bm::DoNotOptimize(remaining);
    }
    setCounters(state);
}

BENCHMARK_REGISTER_F(OffsetFixture, Insert) BENCHMARK_POLICY_CONFIG;
BENCHMARK_REGISTER_F(OffsetFixture, Query) BENCHMARK_POLICY_CONFIG;
BENCHMARK_REGISTER_F(OffsetFixture, QueryNegative) BENCHMARK_POLICY_CONFIG;
BENCHMARK_REGISTER_F(OffsetFixture, Delete) BENCHMARK_POLICY_CONFIG;

STANDARD_BENCHMARK_MAIN();
