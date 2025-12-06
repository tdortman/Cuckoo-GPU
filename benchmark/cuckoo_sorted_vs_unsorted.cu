#include <benchmark/benchmark.h>
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#include <cstdint>
#include <CuckooFilter.cuh>
#include <helpers.cuh>
#include "benchmark_common.cuh"

namespace bm = benchmark;

using Config = CuckooConfig<uint64_t, 16, 500, 256, 16, XorAltBucketPolicy>;

static constexpr double LOAD_FACTOR = 0.95;

class CF : public benchmark::Fixture {
    using benchmark::Fixture::SetUp;
    using benchmark::Fixture::TearDown;

   public:
    void SetUp(const benchmark::State& state) override {
        auto [cap, num] = calculateCapacityAndSize(state.range(0), LOAD_FACTOR);
        capacity = cap;
        n = num;

        d_keys.resize(n);
        d_output.resize(n);
        generateKeysGPU(d_keys);

        filter = std::make_unique<CuckooFilter<Config>>(capacity);
        filterMemory = filter->sizeInBytes();
    }

    void TearDown(const benchmark::State&) override {
        filter.reset();
        d_keys.clear();
        d_keys.shrink_to_fit();
        d_output.clear();
        d_output.shrink_to_fit();
    }

    void setCounters(benchmark::State& state) const {
        setCommonCounters(state, filterMemory, n);
    }

    size_t capacity;
    size_t n;
    size_t filterMemory;
    thrust::device_vector<uint64_t> d_keys;
    thrust::device_vector<uint8_t> d_output;
    std::unique_ptr<CuckooFilter<Config>> filter;
    GPUTimer timer;
};

BENCHMARK_DEFINE_F(CF, InsertUnsorted)(bm::State& state) {
    for (auto _ : state) {
        filter->clear();
        cudaDeviceSynchronize();

        timer.start();
        size_t inserted = filter->insertMany(d_keys);
        double elapsed = timer.elapsed();

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(inserted);
    }
    setCounters(state);
}

BENCHMARK_DEFINE_F(CF, InsertSorted)(bm::State& state) {
    for (auto _ : state) {
        filter->clear();
        cudaDeviceSynchronize();

        timer.start();
        size_t inserted = filter->insertManySorted(d_keys);
        double elapsed = timer.elapsed();

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(inserted);
    }
    setCounters(state);
}

BENCHMARK_REGISTER_F(CF, InsertUnsorted)
    BENCHMARK_CONFIG;

BENCHMARK_REGISTER_F(CF, InsertSorted)
    BENCHMARK_CONFIG;

STANDARD_BENCHMARK_MAIN();
