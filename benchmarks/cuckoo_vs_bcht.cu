#include <benchmark/benchmark.h>
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <bght/bcht.hpp>
#include <bght/pair.cuh>
#include <cstddef>
#include <cstdint>
#include <cuckoogpu/CuckooFilter.cuh>
#include <limits>
#include <memory>

#include <cuckoogpu/bucket_policies.cuh>
#include "benchmark_common.cuh"

namespace bm = benchmark;

constexpr double LOAD_FACTOR = 0.95;
using FilterConfig = cuckoogpu::Config<uint64_t, 16, 500, 128, 16, cuckoogpu::XorAltBucketPolicy>;

template <typename ConfigType, double loadFactor = 0.95>
class CuckooFilterUniqueFixture : public benchmark::Fixture {
    using benchmark::Fixture::SetUp;
    using benchmark::Fixture::TearDown;

   public:
    using KeyType = typename ConfigType::KeyType;

    void SetUp(const benchmark::State& state) override {
        auto [cap, num] = calculateCapacityAndSize(static_cast<size_t>(state.range(0)), loadFactor);
        capacity = cap;
        n = num;

        d_keys.resize(n);
        d_output.resize(n);
        thrust::sequence(d_keys.begin(), d_keys.end(), static_cast<KeyType>(1));

        filter = std::make_unique<cuckoogpu::Filter<ConfigType>>(capacity);
        filterMemory = filter->sizeInBytes();
    }

    void TearDown(const benchmark::State&) override {
        filter.reset();
        d_keys.clear();
        d_output.clear();
        d_keys.shrink_to_fit();
        d_output.shrink_to_fit();
    }

    void setCounters(benchmark::State& state) {
        setCommonCounters(state, filterMemory, n);
    }

    size_t capacity;
    size_t n;
    size_t filterMemory;
    thrust::device_vector<KeyType> d_keys;
    thrust::device_vector<uint8_t> d_output;
    std::unique_ptr<cuckoogpu::Filter<ConfigType>> filter;
    GPUTimer timer;
};

class BCHTFixture : public benchmark::Fixture {
    using benchmark::Fixture::SetUp;
    using benchmark::Fixture::TearDown;

   public:
    using KeyType = uint64_t;
    using ValueType = uint8_t;
    using PairType = bght::pair<KeyType, ValueType>;
    using TableType = bght::bcht<KeyType, ValueType>;

    static constexpr KeyType sentinelKey() {
        return std::numeric_limits<KeyType>::max();
    }

    static constexpr ValueType sentinelValue() {
        return std::numeric_limits<ValueType>::max();
    }

    static size_t memoryBytesForCapacity(size_t requestedCapacity) {
        constexpr size_t bucketSize = TableType::bucket_size;
        const size_t roundedCapacity = SDIV(requestedCapacity, bucketSize) * bucketSize;
        return roundedCapacity * sizeof(typename TableType::atomic_pair_type) + sizeof(bool);
    }

    void SetUp(const benchmark::State& state) override {
        auto [cap, num] =
            calculateCapacityAndSize(static_cast<size_t>(state.range(0)), LOAD_FACTOR);
        capacity = cap;
        n = num;

        d_keys.resize(n);
        d_pairs.resize(n);
        d_results.resize(n);

        thrust::sequence(d_keys.begin(), d_keys.end(), static_cast<KeyType>(1));
        thrust::transform(
            d_keys.begin(), d_keys.end(), d_pairs.begin(), [] __device__(KeyType key) {
                return PairType{key, static_cast<ValueType>(1)};
            }
        );

        tableMemory = memoryBytesForCapacity(capacity);
        resetTable();
    }

    void TearDown(const benchmark::State&) override {
        table.reset();
        d_keys.clear();
        d_pairs.clear();
        d_results.clear();
        d_keys.shrink_to_fit();
        d_pairs.shrink_to_fit();
        d_results.shrink_to_fit();
    }

    void setCounters(benchmark::State& state) const {
        setCommonCounters(state, tableMemory, n);
    }

    void resetTable() {
        table = std::make_unique<TableType>(capacity, sentinelKey(), sentinelValue());
    }

    size_t capacity;
    size_t n;
    size_t tableMemory;
    thrust::device_vector<KeyType> d_keys;
    thrust::device_vector<PairType> d_pairs;
    thrust::device_vector<ValueType> d_results;
    std::unique_ptr<TableType> table;
    GPUTimer timer;
};

using GCFFixture = CuckooFilterUniqueFixture<FilterConfig, LOAD_FACTOR>;

BENCHMARK_DEFINE_F(BCHTFixture, Insert)(bm::State& state) {
    for (auto _ : state) {
        state.PauseTiming();
        resetTable();
        cudaDeviceSynchronize();
        state.ResumeTiming();

        timer.start();
        bool inserted = table->insert(d_pairs.begin(), d_pairs.end());
        double elapsed = timer.elapsed();

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(inserted);
        if (!inserted) {
            state.SkipWithError(
                "BCHTBCHT insertion failed. Try lower load factor or smaller capacity."
            );
            break;
        }
    }
    setCounters(state);
}

BENCHMARK_DEFINE_F(BCHTFixture, Query)(bm::State& state) {
    bool inserted = table->insert(d_pairs.begin(), d_pairs.end());
    if (!inserted) {
        state.SkipWithError("BCHTBCHT insertion failed during query setup.");
        return;
    }
    cudaDeviceSynchronize();

    for (auto _ : state) {
        timer.start();
        table->find(d_keys.begin(), d_keys.end(), d_results.begin());
        double elapsed = timer.elapsed();

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(d_results.data().get());
    }
    setCounters(state);
}

DEFINE_INSERT_QUERY(GCFFixture)

#define BCHT_BENCHMARK_CONFIG \
    ->RangeMultiplier(2)                 \
        ->Range(1 << 16, 1ULL << 26)     \
        ->Unit(benchmark::kMillisecond)  \
        ->UseManualTime()                \
        ->Iterations(10)                 \
        ->Repetitions(5)                 \
        ->ReportAggregatesOnly(true)

BENCHMARK_REGISTER_F(GCFFixture, Insert) BCHT_BENCHMARK_CONFIG;
BENCHMARK_REGISTER_F(GCFFixture, Query) BCHT_BENCHMARK_CONFIG;
BENCHMARK_REGISTER_F(BCHTFixture, Insert) BCHT_BENCHMARK_CONFIG;
BENCHMARK_REGISTER_F(BCHTFixture, Query) BCHT_BENCHMARK_CONFIG;

STANDARD_BENCHMARK_MAIN();
