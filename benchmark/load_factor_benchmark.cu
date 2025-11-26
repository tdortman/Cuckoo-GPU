#include <benchmark/benchmark.h>
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <bulk_tcf_host.cuh>
#include <cstddef>
#include <cstdint>
#include <cuckoo/cuckoo_parameter.hpp>
#include <CuckooFilter.cuh>
#include <cuco/bloom_filter.cuh>
#include <cuda/std/cstdint>
#include <filter.hpp>
#include <hash_strategies.cuh>
#include <helpers.cuh>
#include <random>
#include <thread>
#include "benchmark_common.cuh"
#include "parameter/parameter.hpp"

namespace bm = benchmark;

using Config = CuckooConfig<uint64_t, 16, 500, 128, 16, XorAltBucketPolicy>;
using TCFType = host_bulk_tcf<uint64_t, uint16_t>;

using CPUFilterParam = filters::cuckoo::Standard4<Config::bitsPerTag>;
using CPUOptimParam = filters::parameter::PowerOfTwoMurmurScalar64PartitionedMT;
using PartitionedCuckooFilter =
    filters::Filter<filters::FilterType::Cuckoo, CPUFilterParam, Config::bitsPerTag, CPUOptimParam>;

constexpr size_t FIXED_CAPACITY = 1ULL << 24;
const size_t L2_CACHE_SIZE = getL2CacheSize();

template <double loadFactor>
using CFFixture = CuckooFilterFixture<Config, loadFactor>;

template <typename Filter>
size_t cucoNumBlocks(size_t n) {
    constexpr auto bitsPerWord = sizeof(typename Filter::word_type) * 8;
    return SDIV(n * Config::bitsPerTag, Filter::words_per_block * bitsPerWord);
}

template <double loadFactor>
class BloomFilterFixture : public benchmark::Fixture {
    using benchmark::Fixture::SetUp;
    using benchmark::Fixture::TearDown;

   public:
    using BloomFilter = cuco::bloom_filter<uint64_t>;

    void SetUp(const benchmark::State& state) override {
        auto [cap, num] = calculateCapacityAndSize(state.range(0), loadFactor);
        capacity = cap;
        n = num;

        const size_t numBlocks = cucoNumBlocks<BloomFilter>(capacity);
        d_keys.resize(n);
        d_output.resize(n);
        generateKeysGPU(d_keys);

        filter = std::make_unique<BloomFilter>(numBlocks);
        filterMemory = filter->block_extent() * BloomFilter::words_per_block *
                       sizeof(typename BloomFilter::word_type);
    }

    void TearDown(const benchmark::State&) override {
        filter.reset();
        d_keys.clear();
        d_keys.shrink_to_fit();
        d_output.clear();
        d_output.shrink_to_fit();
    }

    void setCounters(benchmark::State& state) {
        setCommonCounters(state, filterMemory, n);
    }

    size_t capacity;
    size_t n;
    size_t filterMemory;
    thrust::device_vector<uint64_t> d_keys;
    thrust::device_vector<uint8_t> d_output;
    std::unique_ptr<BloomFilter> filter;
    Timer timer;
};

template <double loadFactor>
class TCFFixture : public benchmark::Fixture {
    using benchmark::Fixture::SetUp;
    using benchmark::Fixture::TearDown;

   public:
    void SetUp(const benchmark::State& state) override {
        auto [cap, num] = calculateCapacityAndSize(state.range(0), loadFactor);
        capacity = cap;
        n = num;

        d_keys.resize(n);
        generateKeysGPU(d_keys);

        cudaMalloc(&d_misses, sizeof(uint64_t));
        filterMemory = capacity * sizeof(uint16_t);
    }

    void TearDown(const benchmark::State&) override {
        if (d_misses) {
            cudaFree(d_misses);
            d_misses = nullptr;
        }
        d_keys.clear();
        d_keys.shrink_to_fit();
    }

    void setCounters(benchmark::State& state) const {
        setCommonCounters(state, filterMemory, n);
    }

    size_t capacity;
    size_t n;
    size_t filterMemory;
    uint64_t* d_misses = nullptr;
    thrust::device_vector<uint64_t> d_keys;
    Timer timer;
};

template <double loadFactor>
class PartitionedCFFixture : public benchmark::Fixture {
    using benchmark::Fixture::SetUp;
    using benchmark::Fixture::TearDown;

   public:
    void SetUp(const benchmark::State& state) override {
        auto [cap, num] = calculateCapacityAndSize(state.range(0), loadFactor);
        capacity = cap;
        n = num;

        keys = generateKeysCPU<uint64_t>(n);

        s = static_cast<size_t>(100.0 / loadFactor);

        size_t expectedSize =
            (capacity * Config::bitsPerTag * (static_cast<double>(s) / 100)) / 8.0;
        n_partitions = 1;

        if (expectedSize > L2_CACHE_SIZE) {
            size_t partitionsNeeded = SDIV(expectedSize, L2_CACHE_SIZE);
            while (n_partitions < partitionsNeeded) {
                n_partitions *= 2;
            }
        }

        // FIXME: How to reliably get the number of physical cores?
        n_threads = std::min(n_partitions, size_t(std::thread::hardware_concurrency() / 2));
        n_tasks = 1;
    }

    void TearDown(const benchmark::State&) override {
        keys.clear();
    }

    void setCounters(benchmark::State& state, size_t filterMemory) const {
        setCommonCounters(state, filterMemory, n);
    }

    size_t capacity;
    size_t n;
    size_t s;
    size_t n_partitions;
    size_t n_threads;
    size_t n_tasks;
    std::vector<uint64_t> keys;
    Timer timer;
};

#define BENCHMARK_CONFIG_LF             \
    ->Arg(FIXED_CAPACITY)               \
        ->Unit(benchmark::kMillisecond) \
        ->UseManualTime()               \
        ->MinTime(0.5)                  \
        ->Repetitions(5)                \
        ->ReportAggregatesOnly(true)

#define DEFINE_AND_REGISTER_ALL_FOR_LOAD_FACTOR(LF)                                           \
    /* Cuckoo Filter */                                                                       \
    using CF_##LF = CFFixture<(LF) * 0.01>;                                                   \
    DEFINE_INSERT_QUERY(CF_##LF)                                                              \
    DEFINE_FILTER_DELETE_BENCHMARK(CF_##LF)                                                   \
    BENCHMARK_REGISTER_F(CF_##LF, Insert) BENCHMARK_CONFIG_LF;                                \
    BENCHMARK_REGISTER_F(CF_##LF, Query) BENCHMARK_CONFIG_LF;                                 \
    BENCHMARK_REGISTER_F(CF_##LF, Delete) BENCHMARK_CONFIG_LF;                                \
                                                                                              \
    /* Bloom Filter */                                                                        \
    using BBF_##LF = BloomFilterFixture<(LF) * 0.01>;                                         \
    BENCHMARK_DEFINE_F(BBF_##LF, Insert)(bm::State & state) {                                 \
        for (auto _ : state) {                                                                \
            filter->clear();                                                                  \
            cudaDeviceSynchronize();                                                          \
            timer.start();                                                                    \
            filter->add(d_keys.begin(), d_keys.end());                                        \
            double elapsed = timer.stop();                                                    \
            state.SetIterationTime(elapsed);                                                  \
        }                                                                                     \
        setCounters(state);                                                                   \
    }                                                                                         \
    BENCHMARK_DEFINE_F(BBF_##LF, Query)(bm::State & state) {                                  \
        filter->add(d_keys.begin(), d_keys.end());                                            \
        for (auto _ : state) {                                                                \
            timer.start();                                                                    \
            filter->contains(                                                                 \
                d_keys.begin(),                                                               \
                d_keys.end(),                                                                 \
                reinterpret_cast<bool*>(thrust::raw_pointer_cast(d_output.data()))            \
            );                                                                                \
            double elapsed = timer.stop();                                                    \
            state.SetIterationTime(elapsed);                                                  \
            bm::DoNotOptimize(d_output.data().get());                                         \
        }                                                                                     \
        setCounters(state);                                                                   \
    }                                                                                         \
    BENCHMARK_REGISTER_F(BBF_##LF, Insert) BENCHMARK_CONFIG_LF;                               \
    BENCHMARK_REGISTER_F(BBF_##LF, Query) BENCHMARK_CONFIG_LF;                                \
                                                                                              \
    /* TCF */                                                                                 \
    using TCF_##LF = TCFFixture<(LF) * 0.01>;                                                 \
    BENCHMARK_DEFINE_F(TCF_##LF, Insert)(bm::State & state) {                                 \
        for (auto _ : state) {                                                                \
            TCFType* filter = TCFType::host_build_tcf(capacity);                              \
            cudaMemset(d_misses, 0, sizeof(uint64_t));                                        \
            cudaDeviceSynchronize();                                                          \
            timer.start();                                                                    \
            filter->bulk_insert(thrust::raw_pointer_cast(d_keys.data()), n, d_misses);        \
            double elapsed = timer.stop();                                                    \
            state.SetIterationTime(elapsed);                                                  \
            TCFType::host_free_tcf(filter);                                                   \
        }                                                                                     \
        setCounters(state);                                                                   \
    }                                                                                         \
    BENCHMARK_DEFINE_F(TCF_##LF, Query)(bm::State & state) {                                  \
        TCFType* filter = TCFType::host_build_tcf(capacity);                                  \
        cudaMemset(d_misses, 0, sizeof(uint64_t));                                            \
        filter->bulk_insert(thrust::raw_pointer_cast(d_keys.data()), n, d_misses);            \
        for (auto _ : state) {                                                                \
            timer.start();                                                                    \
            bool* d_output = filter->bulk_query(thrust::raw_pointer_cast(d_keys.data()), n);  \
            double elapsed = timer.stop();                                                    \
            state.SetIterationTime(elapsed);                                                  \
            bm::DoNotOptimize(d_output);                                                      \
            cudaFree(d_output);                                                               \
        }                                                                                     \
        TCFType::host_free_tcf(filter);                                                       \
        setCounters(state);                                                                   \
    }                                                                                         \
    BENCHMARK_DEFINE_F(TCF_##LF, Delete)(bm::State & state) {                                 \
        for (auto _ : state) {                                                                \
            TCFType* filter = TCFType::host_build_tcf(capacity);                              \
            cudaMemset(d_misses, 0, sizeof(uint64_t));                                        \
            filter->bulk_insert(thrust::raw_pointer_cast(d_keys.data()), n, d_misses);        \
            cudaDeviceSynchronize();                                                          \
            timer.start();                                                                    \
            bool* d_output = filter->bulk_delete(thrust::raw_pointer_cast(d_keys.data()), n); \
            double elapsed = timer.stop();                                                    \
            state.SetIterationTime(elapsed);                                                  \
            bm::DoNotOptimize(d_output);                                                      \
            cudaFree(d_output);                                                               \
            TCFType::host_free_tcf(filter);                                                   \
        }                                                                                     \
        setCounters(state);                                                                   \
    }                                                                                         \
    BENCHMARK_REGISTER_F(TCF_##LF, Insert) BENCHMARK_CONFIG_LF;                               \
    BENCHMARK_REGISTER_F(TCF_##LF, Query) BENCHMARK_CONFIG_LF;                                \
    BENCHMARK_REGISTER_F(TCF_##LF, Delete) BENCHMARK_CONFIG_LF;                               \
                                                                                              \
    /* Partitioned Cuckoo Filter */                                                           \
    using PCF_##LF = PartitionedCFFixture<(LF) * 0.01>;                                       \
    BENCHMARK_DEFINE_F(PCF_##LF, Insert)(bm::State & state) {                                 \
        for (auto _ : state) {                                                                \
            PartitionedCuckooFilter tempFilter(s, n_partitions, n_threads, n_tasks);          \
            auto constructKeys = keys;                                                        \
            timer.start();                                                                    \
            bool success = tempFilter.construct(constructKeys.data(), constructKeys.size());  \
            double elapsed = timer.stop();                                                    \
            state.SetIterationTime(elapsed);                                                  \
            bm::DoNotOptimize(success);                                                       \
        }                                                                                     \
        PartitionedCuckooFilter finalFilter(s, n_partitions, n_threads, n_tasks);             \
        finalFilter.construct(keys.data(), keys.size());                                      \
        size_t filterMemory = finalFilter.size();                                             \
        setCounters(state, filterMemory);                                                     \
    }                                                                                         \
    BENCHMARK_DEFINE_F(PCF_##LF, Query)(bm::State & state) {                                  \
        PartitionedCuckooFilter filter(s, n_partitions, n_threads, n_tasks);                  \
        filter.construct(keys.data(), keys.size());                                           \
        size_t filterMemory = filter.size();                                                  \
        for (auto _ : state) {                                                                \
            timer.start();                                                                    \
            size_t found = filter.count(keys.data(), keys.size());                            \
            double elapsed = timer.stop();                                                    \
            state.SetIterationTime(elapsed);                                                  \
            bm::DoNotOptimize(found);                                                         \
        }                                                                                     \
        setCounters(state, filterMemory);                                                     \
    }                                                                                         \
    BENCHMARK_REGISTER_F(PCF_##LF, Insert) BENCHMARK_CONFIG_LF;                               \
    BENCHMARK_REGISTER_F(PCF_##LF, Query) BENCHMARK_CONFIG_LF;

DEFINE_AND_REGISTER_ALL_FOR_LOAD_FACTOR(5)
DEFINE_AND_REGISTER_ALL_FOR_LOAD_FACTOR(10)
DEFINE_AND_REGISTER_ALL_FOR_LOAD_FACTOR(15)
DEFINE_AND_REGISTER_ALL_FOR_LOAD_FACTOR(20)
DEFINE_AND_REGISTER_ALL_FOR_LOAD_FACTOR(25)
DEFINE_AND_REGISTER_ALL_FOR_LOAD_FACTOR(30)
DEFINE_AND_REGISTER_ALL_FOR_LOAD_FACTOR(35)
DEFINE_AND_REGISTER_ALL_FOR_LOAD_FACTOR(40)
DEFINE_AND_REGISTER_ALL_FOR_LOAD_FACTOR(45)
DEFINE_AND_REGISTER_ALL_FOR_LOAD_FACTOR(50)
DEFINE_AND_REGISTER_ALL_FOR_LOAD_FACTOR(55)
DEFINE_AND_REGISTER_ALL_FOR_LOAD_FACTOR(60)
DEFINE_AND_REGISTER_ALL_FOR_LOAD_FACTOR(65)
DEFINE_AND_REGISTER_ALL_FOR_LOAD_FACTOR(70)
DEFINE_AND_REGISTER_ALL_FOR_LOAD_FACTOR(75)
DEFINE_AND_REGISTER_ALL_FOR_LOAD_FACTOR(80)
DEFINE_AND_REGISTER_ALL_FOR_LOAD_FACTOR(85)
DEFINE_AND_REGISTER_ALL_FOR_LOAD_FACTOR(90)
DEFINE_AND_REGISTER_ALL_FOR_LOAD_FACTOR(95)

STANDARD_BENCHMARK_MAIN();
