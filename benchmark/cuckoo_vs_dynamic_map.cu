#include <benchmark/benchmark.h>
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <cstdint>
#include <cuckoogpu/bucket_policies.cuh>
#include <cuckoogpu/CuckooFilter.cuh>
#include <cuckoogpu/helpers.cuh>
#include <cuco/dynamic_map.cuh>
#include <limits>
#include "benchmark_common.cuh"

namespace bm = benchmark;

using Config = cuckoogpu::Config<uint64_t, 16, 500, 128, 16, cuckoogpu::XorAltBucketPolicy>;

using DMKey = uint64_t;
using DMValue = uint32_t;
using DMPair = cuco::pair<DMKey, DMValue>;
using DMMap = cuco::dynamic_map<DMKey, DMValue>;

class DMFixture : public benchmark::Fixture {
    using benchmark::Fixture::SetUp;
    using benchmark::Fixture::TearDown;

   public:
    void SetUp(const benchmark::State& state) override {
        auto [cap, num] = calculateCapacityAndSize(state.range(0), 0.95);
        capacity = cap;
        n = num;

        d_Keys.resize(n);
        d_Pairs.resize(n);
        d_Output.resize(n);

        constexpr DMKey emptyKey = 0;
        constexpr DMKey erasedKey = std::numeric_limits<DMKey>::max();
        constexpr DMValue emptyValue = std::numeric_limits<DMValue>::max();
        constexpr DMValue insertValue = 1;

        generateKeysGPURange(d_Keys, n, static_cast<DMKey>(1), erasedKey - 1);

        thrust::transform(
            d_Keys.begin(), d_Keys.end(), d_Pairs.begin(), [insertValue] __device__(DMKey key) {
                return DMPair{key, insertValue};
            }
        );

        emptyKeySentinel = emptyKey;
        erasedKeySentinel = erasedKey;
        emptyValueSentinel = emptyValue;
        mapMemory = capacity * sizeof(DMPair);

        resetMap();
    }

    void TearDown(const benchmark::State&) override {
        map.reset();
        d_Keys.clear();
        d_Pairs.clear();
        d_Output.clear();
        d_Keys.shrink_to_fit();
        d_Pairs.shrink_to_fit();
        d_Output.shrink_to_fit();
    }

    void setCounters(benchmark::State& state) const {
        setCommonCounters(state, mapMemory, n);
    }

    void resetMap() {
        map = std::make_unique<DMMap>(
            capacity,
            cuco::empty_key<DMKey>{emptyKeySentinel},
            cuco::empty_value<DMValue>{emptyValueSentinel},
            cuco::erased_key<DMKey>{erasedKeySentinel}
        );
    }

    size_t capacity;
    size_t n;
    size_t mapMemory;
    DMKey emptyKeySentinel;
    DMKey erasedKeySentinel;
    DMValue emptyValueSentinel;
    thrust::device_vector<DMKey> d_Keys;
    thrust::device_vector<DMPair> d_Pairs;
    thrust::device_vector<uint8_t> d_Output;
    std::unique_ptr<DMMap> map;
    GPUTimer timer;
};

using GCFFixture = CuckooFilterFixture<Config>;

BENCHMARK_DEFINE_F(DMFixture, Insert)(bm::State& state) {
    for (auto _ : state) {
        state.PauseTiming();
        resetMap();
        cudaDeviceSynchronize();
        state.ResumeTiming();

        timer.start();
        map->insert(d_Pairs.begin(), d_Pairs.end());
        double elapsed = timer.elapsed();

        state.SetIterationTime(elapsed);
    }
    setCounters(state);
}

BENCHMARK_DEFINE_F(DMFixture, Query)(bm::State& state) {
    map->insert(d_Pairs.begin(), d_Pairs.end());
    cudaDeviceSynchronize();

    for (auto _ : state) {
        timer.start();
        map->contains(d_Keys.begin(), d_Keys.end(), d_Output.begin());
        double elapsed = timer.elapsed();

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(d_Output.data().get());
    }
    setCounters(state);
}

BENCHMARK_DEFINE_F(DMFixture, Delete)(bm::State& state) {
    for (auto _ : state) {
        state.PauseTiming();
        resetMap();
        map->insert(d_Pairs.begin(), d_Pairs.end());
        cudaDeviceSynchronize();
        state.ResumeTiming();

        timer.start();
        map->erase(d_Keys.begin(), d_Keys.end());
        double elapsed = timer.elapsed();

        state.SetIterationTime(elapsed);
    }
    setCounters(state);
}

#define DM_BENCHMARK_CONFIG             \
    ->RangeMultiplier(2)                \
        ->Range(1 << 16, 1ULL << 26)    \
        ->Unit(benchmark::kMillisecond) \
        ->UseManualTime()               \
        ->Iterations(10)                \
        ->Repetitions(5)                \
        ->ReportAggregatesOnly(true)

DEFINE_AND_REGISTER_CORE_BENCHMARKS(GCFFixture)
BENCHMARK_REGISTER_F(DMFixture, Insert) DM_BENCHMARK_CONFIG;
BENCHMARK_REGISTER_F(DMFixture, Query) DM_BENCHMARK_CONFIG;
BENCHMARK_REGISTER_F(DMFixture, Delete) DM_BENCHMARK_CONFIG;

STANDARD_BENCHMARK_MAIN();
