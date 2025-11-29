#include <benchmark/benchmark.h>
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <bulk_tcf_host.cuh>
#include <cstddef>
#include <cstdint>
#include <CuckooFilter.cuh>
#include <cuco/bloom_filter.cuh>
#include <cuda/std/cstdint>
#include <hash_strategies.cuh>
#include <helpers.cuh>
#include <random>
#include <thread>
#include "benchmark_common.cuh"

namespace bm = benchmark;

using Config = CuckooConfig<uint64_t, 16, 500, 128, 16, XorAltBucketPolicy>;
using TCFType = host_bulk_tcf<uint64_t, uint16_t>;

constexpr size_t FIXED_CAPACITY = 1ULL << 24;
const size_t L2_CACHE_SIZE = getL2CacheSize();

template <double loadFactor>
using CFFixture = CuckooFilterFixture<Config, loadFactor>;

template <double loadFactor>
class CFFixtureLF : public benchmark::Fixture {
    using benchmark::Fixture::SetUp;
    using benchmark::Fixture::TearDown;

   public:
    void SetUp(const benchmark::State& state) override {
        auto [cap, num] = calculateCapacityAndSize(state.range(0), loadFactor);
        capacity = cap;
        n = num;

        d_keys.resize(n);
        d_keysNegative.resize(n);
        d_output.resize(n);

        generateKeysGPURange(d_keys, n, uint64_t(0), uint64_t(UINT32_MAX));
        generateKeysGPURange(d_keysNegative, n, uint64_t(UINT32_MAX) + 1, UINT64_MAX);

        filter = std::make_unique<CuckooFilter<Config>>(capacity);
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
    }

    size_t capacity;
    size_t n;
    size_t filterMemory;
    thrust::device_vector<uint64_t> d_keys;
    thrust::device_vector<uint64_t> d_keysNegative;
    thrust::device_vector<uint8_t> d_output;
    std::unique_ptr<CuckooFilter<Config>> filter;
    Timer timer;
};

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
        d_keysNegative.resize(n);
        d_output.resize(n);

        generateKeysGPURange(d_keys, n, uint64_t(0), uint64_t(UINT32_MAX));
        generateKeysGPURange(d_keysNegative, n, uint64_t(UINT32_MAX) + 1, UINT64_MAX);

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
    thrust::device_vector<uint64_t> d_keysNegative;
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
        d_keysNegative.resize(n);

        generateKeysGPURange(d_keys, n, uint64_t(0), uint64_t(UINT32_MAX));
        generateKeysGPURange(d_keysNegative, n, uint64_t(UINT32_MAX) + 1, UINT64_MAX);

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
    thrust::device_vector<uint64_t> d_keysNegative;
    Timer timer;
};

#define BENCHMARK_CONFIG_LF             \
    ->Arg(FIXED_CAPACITY)               \
        ->Unit(benchmark::kMillisecond) \
        ->UseManualTime()               \
        ->MinTime(0.5)                  \
        ->Repetitions(5)                \
        ->ReportAggregatesOnly(true)

#define DEFINE_AND_REGISTER_ALL_FOR_LOAD_FACTOR(ID, LF)                                       \
    /* Cuckoo Filter */                                                                       \
    using CF_##ID = CFFixtureLF<(LF) * 0.01>;                                                 \
    BENCHMARK_DEFINE_F(CF_##ID, Insert)(bm::State & state) {                                  \
        for (auto _ : state) {                                                                \
            filter->clear();                                                                  \
            cudaDeviceSynchronize();                                                          \
            timer.start();                                                                    \
            size_t inserted = adaptiveInsert(*filter, d_keys);                                \
            double elapsed = timer.stop();                                                    \
            state.SetIterationTime(elapsed);                                                  \
            bm::DoNotOptimize(inserted);                                                      \
        }                                                                                     \
        setCounters(state);                                                                   \
    }                                                                                         \
    BENCHMARK_DEFINE_F(CF_##ID, Query)(bm::State & state) {                                   \
        adaptiveInsert(*filter, d_keys);                                                      \
        for (auto _ : state) {                                                                \
            timer.start();                                                                    \
            filter->containsMany(d_keys, d_output);                                           \
            double elapsed = timer.stop();                                                    \
            state.SetIterationTime(elapsed);                                                  \
            bm::DoNotOptimize(d_output.data().get());                                         \
        }                                                                                     \
        setCounters(state);                                                                   \
    }                                                                                         \
    BENCHMARK_DEFINE_F(CF_##ID, QueryNegative)(bm::State & state) {                           \
        adaptiveInsert(*filter, d_keys);                                                      \
        for (auto _ : state) {                                                                \
            timer.start();                                                                    \
            filter->containsMany(d_keysNegative, d_output);                                   \
            double elapsed = timer.stop();                                                    \
            state.SetIterationTime(elapsed);                                                  \
            bm::DoNotOptimize(d_output.data().get());                                         \
        }                                                                                     \
        setCounters(state);                                                                   \
    }                                                                                         \
    BENCHMARK_DEFINE_F(CF_##ID, Delete)(bm::State & state) {                                  \
        for (auto _ : state) {                                                                \
            filter->clear();                                                                  \
            adaptiveInsert(*filter, d_keys);                                                  \
            cudaDeviceSynchronize();                                                          \
            timer.start();                                                                    \
            size_t remaining = filter->deleteMany(d_keys, d_output);                          \
            double elapsed = timer.stop();                                                    \
            state.SetIterationTime(elapsed);                                                  \
            bm::DoNotOptimize(remaining);                                                     \
            bm::DoNotOptimize(d_output.data().get());                                         \
        }                                                                                     \
        setCounters(state);                                                                   \
    }                                                                                         \
    BENCHMARK_REGISTER_F(CF_##ID, Insert) BENCHMARK_CONFIG_LF;                                \
    BENCHMARK_REGISTER_F(CF_##ID, Query) BENCHMARK_CONFIG_LF;                                 \
    BENCHMARK_REGISTER_F(CF_##ID, QueryNegative) BENCHMARK_CONFIG_LF;                         \
    BENCHMARK_REGISTER_F(CF_##ID, Delete) BENCHMARK_CONFIG_LF;                                \
                                                                                              \
    /* Bloom Filter */                                                                        \
    using BBF_##ID = BloomFilterFixture<(LF) * 0.01>;                                         \
    BENCHMARK_DEFINE_F(BBF_##ID, Insert)(bm::State & state) {                                 \
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
    BENCHMARK_DEFINE_F(BBF_##ID, Query)(bm::State & state) {                                  \
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
    BENCHMARK_DEFINE_F(BBF_##ID, QueryNegative)(bm::State & state) {                          \
        filter->add(d_keys.begin(), d_keys.end());                                            \
        for (auto _ : state) {                                                                \
            timer.start();                                                                    \
            filter->contains(                                                                 \
                d_keysNegative.begin(),                                                       \
                d_keysNegative.end(),                                                         \
                reinterpret_cast<bool*>(thrust::raw_pointer_cast(d_output.data()))            \
            );                                                                                \
            double elapsed = timer.stop();                                                    \
            state.SetIterationTime(elapsed);                                                  \
            bm::DoNotOptimize(d_output.data().get());                                         \
        }                                                                                     \
        setCounters(state);                                                                   \
    }                                                                                         \
    BENCHMARK_REGISTER_F(BBF_##ID, Insert) BENCHMARK_CONFIG_LF;                               \
    BENCHMARK_REGISTER_F(BBF_##ID, Query) BENCHMARK_CONFIG_LF;                                \
    BENCHMARK_REGISTER_F(BBF_##ID, QueryNegative) BENCHMARK_CONFIG_LF;                        \
                                                                                              \
    /* TCF */                                                                                 \
    using TCF_##ID = TCFFixture<(LF) * 0.01>;                                                 \
    BENCHMARK_DEFINE_F(TCF_##ID, Insert)(bm::State & state) {                                 \
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
    BENCHMARK_DEFINE_F(TCF_##ID, Query)(bm::State & state) {                                  \
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
    BENCHMARK_DEFINE_F(TCF_##ID, QueryNegative)(bm::State & state) {                          \
        TCFType* filter = TCFType::host_build_tcf(capacity);                                  \
        cudaMemset(d_misses, 0, sizeof(uint64_t));                                            \
        filter->bulk_insert(thrust::raw_pointer_cast(d_keys.data()), n, d_misses);            \
        for (auto _ : state) {                                                                \
            timer.start();                                                                    \
            bool* d_output =                                                                  \
                filter->bulk_query(thrust::raw_pointer_cast(d_keysNegative.data()), n);       \
            double elapsed = timer.stop();                                                    \
            state.SetIterationTime(elapsed);                                                  \
            bm::DoNotOptimize(d_output);                                                      \
            cudaFree(d_output);                                                               \
        }                                                                                     \
        TCFType::host_free_tcf(filter);                                                       \
        setCounters(state);                                                                   \
    }                                                                                         \
    BENCHMARK_DEFINE_F(TCF_##ID, Delete)(bm::State & state) {                                 \
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
    BENCHMARK_REGISTER_F(TCF_##ID, Insert) BENCHMARK_CONFIG_LF;                               \
    BENCHMARK_REGISTER_F(TCF_##ID, Query) BENCHMARK_CONFIG_LF;                                \
    BENCHMARK_REGISTER_F(TCF_##ID, QueryNegative) BENCHMARK_CONFIG_LF;                        \
    BENCHMARK_REGISTER_F(TCF_##ID, Delete) BENCHMARK_CONFIG_LF;

DEFINE_AND_REGISTER_ALL_FOR_LOAD_FACTOR(5, 5)
DEFINE_AND_REGISTER_ALL_FOR_LOAD_FACTOR(10, 10)
DEFINE_AND_REGISTER_ALL_FOR_LOAD_FACTOR(15, 15)
DEFINE_AND_REGISTER_ALL_FOR_LOAD_FACTOR(20, 20)
DEFINE_AND_REGISTER_ALL_FOR_LOAD_FACTOR(25, 25)
DEFINE_AND_REGISTER_ALL_FOR_LOAD_FACTOR(30, 30)
DEFINE_AND_REGISTER_ALL_FOR_LOAD_FACTOR(35, 35)
DEFINE_AND_REGISTER_ALL_FOR_LOAD_FACTOR(40, 40)
DEFINE_AND_REGISTER_ALL_FOR_LOAD_FACTOR(45, 45)
DEFINE_AND_REGISTER_ALL_FOR_LOAD_FACTOR(50, 50)
DEFINE_AND_REGISTER_ALL_FOR_LOAD_FACTOR(55, 55)
DEFINE_AND_REGISTER_ALL_FOR_LOAD_FACTOR(60, 60)
DEFINE_AND_REGISTER_ALL_FOR_LOAD_FACTOR(65, 65)
DEFINE_AND_REGISTER_ALL_FOR_LOAD_FACTOR(70, 70)
DEFINE_AND_REGISTER_ALL_FOR_LOAD_FACTOR(75, 75)
DEFINE_AND_REGISTER_ALL_FOR_LOAD_FACTOR(80, 80)
DEFINE_AND_REGISTER_ALL_FOR_LOAD_FACTOR(85, 85)
DEFINE_AND_REGISTER_ALL_FOR_LOAD_FACTOR(90, 90)
DEFINE_AND_REGISTER_ALL_FOR_LOAD_FACTOR(95, 95)
DEFINE_AND_REGISTER_ALL_FOR_LOAD_FACTOR(98, 98)
DEFINE_AND_REGISTER_ALL_FOR_LOAD_FACTOR(99, 99)
DEFINE_AND_REGISTER_ALL_FOR_LOAD_FACTOR(99_5, 99.5)

STANDARD_BENCHMARK_MAIN();
