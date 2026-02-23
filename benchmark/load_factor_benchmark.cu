#include <benchmark/benchmark.h>
#include <bght/bcht.hpp>
#include <bght/pair.cuh>
#include <cuckoofilter.h>
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <bulk_tcf_host.cuh>
#include <cstddef>
#include <cstdint>
#include <cuckoogpu/bucket_policies.cuh>
#include <cuckoogpu/CuckooFilter.cuh>
#include <cuckoogpu/helpers.cuh>
#include <cuco/bloom_filter.cuh>
#include <cuda/std/cstdint>
#include <gqf.cuh>
#include <gqf_int.cuh>
#include <random>
#include <thread>
#include "benchmark_common.cuh"

#ifdef __x86_64__
    #include <cuckoo/cuckoo_parameter.hpp>
    #include <filter.hpp>
    #include "parameter/parameter.hpp"
#endif

size_t getQFSizeHost(QF* d_qf) {
    QF h_qf;
    cudaMemcpy(&h_qf, d_qf, sizeof(QF), cudaMemcpyDeviceToHost);

    qfmetadata h_metadata;
    cudaMemcpy(&h_metadata, h_qf.metadata, sizeof(qfmetadata), cudaMemcpyDeviceToHost);

    return h_metadata.total_size_in_bytes;
}

namespace bm = benchmark;

using Config = cuckoogpu::Config<uint64_t, 16, 500, 128, 16, cuckoogpu::XorAltBucketPolicy>;
using TCFType = host_bulk_tcf<uint64_t, uint16_t>;

#ifdef __x86_64__
using CPUFilterParam = filters::cuckoo::Standard4<Config::bitsPerTag>;
using CPUOptimParam = filters::parameter::PowerOfTwoMurmurScalar64PartitionedMT;
using PartitionedCuckooFilter =
    filters::Filter<filters::FilterType::Cuckoo, CPUFilterParam, Config::bitsPerTag, CPUOptimParam>;
#endif

constexpr size_t FIXED_CAPACITY = 1ULL << 22;
const size_t L2_CACHE_SIZE = getL2CacheSize();

template <double loadFactor>
using GCFFixture = CuckooFilterFixture<Config, loadFactor>;

template <double loadFactor>
class GCFFixtureLF : public benchmark::Fixture {
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

        filter = std::make_unique<cuckoogpu::Filter<Config>>(capacity);
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
    std::unique_ptr<cuckoogpu::Filter<Config>> filter;
    GPUTimer timer;
};

template <typename Filter>
size_t cucoNumBlocks(size_t n) {
    constexpr auto bitsPerWord = sizeof(typename Filter::word_type) * 8;
    return SDIV(n * Config::bitsPerTag, Filter::words_per_block * bitsPerWord);
}

template <typename PairType, typename KeyType, typename ValueType>
__global__ void makePairsKernel(
    const KeyType* keys,
    PairType* pairs,
    size_t n,
    ValueType value
) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        pairs[idx] = PairType{keys[idx], value};
    }
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
        d_keysNegative.clear();
        d_keysNegative.shrink_to_fit();
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
    GPUTimer timer;
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
        d_keysNegative.clear();
        d_keysNegative.shrink_to_fit();
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
    GPUTimer timer;
};

template <double loadFactor>
class BGHTFixtureLF : public benchmark::Fixture {
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
        auto [cap, num] = calculateCapacityAndSize(state.range(0), loadFactor);
        capacity = cap;
        n = num;

        d_keys.resize(n);
        d_keysNegative.resize(n);
        d_pairs.resize(n);
        d_output.resize(n);

        generateKeysGPURange(d_keys, n, uint64_t(0), uint64_t(UINT32_MAX));
        generateKeysGPURange(d_keysNegative, n, uint64_t(UINT32_MAX) + 1, UINT64_MAX);

        constexpr int threadsPerBlock = 256;
        const int numBlocks = static_cast<int>(SDIV(n, static_cast<size_t>(threadsPerBlock)));
        makePairsKernel<<<numBlocks, threadsPerBlock>>>(
            thrust::raw_pointer_cast(d_keys.data()),
            thrust::raw_pointer_cast(d_pairs.data()),
            n,
            static_cast<ValueType>(1)
        );
        CUDA_CALL(cudaDeviceSynchronize());

        tableMemory = memoryBytesForCapacity(capacity);
        resetTable();
    }

    void TearDown(const benchmark::State&) override {
        table.reset();
        d_keys.clear();
        d_keys.shrink_to_fit();
        d_keysNegative.clear();
        d_keysNegative.shrink_to_fit();
        d_pairs.clear();
        d_pairs.shrink_to_fit();
        d_output.clear();
        d_output.shrink_to_fit();
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
    thrust::device_vector<KeyType> d_keysNegative;
    thrust::device_vector<PairType> d_pairs;
    thrust::device_vector<ValueType> d_output;
    std::unique_ptr<TableType> table;
    GPUTimer timer;
};

#ifdef __x86_64__
template <double loadFactor>
class PartitionedCFFixture : public benchmark::Fixture {
    using benchmark::Fixture::SetUp;
    using benchmark::Fixture::TearDown;

   public:
    void SetUp(const benchmark::State& state) override {
        auto [cap, num] = calculateCapacityAndSize(state.range(0), loadFactor);
        capacity = cap;
        n = num;

        keys = generateKeysCPU<uint64_t>(n, 42, 0, UINT32_MAX);
        keysNegative = generateKeysCPU<uint64_t>(n, 99999, uint64_t(UINT32_MAX) + 1, UINT64_MAX);

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

        n_threads = std::min(n_partitions, size_t(std::thread::hardware_concurrency()));
        n_tasks = 1;
    }

    void TearDown(const benchmark::State&) override {
        keys.clear();
        keys.shrink_to_fit();
        keysNegative.clear();
        keysNegative.shrink_to_fit();
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
    std::vector<uint64_t> keysNegative;
    CPUTimer timer;
};
#endif

template <double loadFactor>
class CCFFixture : public benchmark::Fixture {
    using benchmark::Fixture::SetUp;
    using benchmark::Fixture::TearDown;

   public:
    void SetUp(const benchmark::State& state) override {
        auto [cap, num] = calculateCapacityAndSize(state.range(0), loadFactor);
        capacity = cap;
        n = num;

        keys = generateKeysCPU<uint64_t>(n, 42, 0, UINT32_MAX);
        keysNegative = generateKeysCPU<uint64_t>(n, 99999, uint64_t(UINT32_MAX) + 1, UINT64_MAX);
    }

    void TearDown(const benchmark::State&) override {
        keys.clear();
        keys.shrink_to_fit();
        keysNegative.clear();
        keysNegative.shrink_to_fit();
    }

    void setCounters(benchmark::State& state, size_t filterMemory) const {
        setCommonCounters(state, filterMemory, n);
    }

    size_t capacity;
    size_t n;
    std::vector<uint64_t> keys;
    std::vector<uint64_t> keysNegative;
    CPUTimer timer;
};

template <double loadFactor>
class GQFFixtureLF : public benchmark::Fixture {
    using benchmark::Fixture::SetUp;
    using benchmark::Fixture::TearDown;

   public:
    void SetUp(const benchmark::State& state) override {
        auto [cap, num] = calculateCapacityAndSize(state.range(0), loadFactor);
        n = num;

        q = static_cast<uint32_t>(std::log2(cap));
        capacity = 1ULL << q;

        d_keys.resize(n);
        d_keysNegative.resize(n);
        d_results.resize(n);

        generateKeysGPURange(
            d_keys, n, static_cast<uint64_t>(0), static_cast<uint64_t>(UINT32_MAX)
        );
        generateKeysGPURange(d_keysNegative, n, static_cast<uint64_t>(UINT32_MAX) + 1, UINT64_MAX);

        qf_malloc_device(&qf, q, true);
        filterMemory = getQFSizeHost(qf);
    }

    void TearDown(const benchmark::State&) override {
        qf_destroy_device(qf);
        d_keys.clear();
        d_keys.shrink_to_fit();
        d_keysNegative.clear();
        d_keysNegative.shrink_to_fit();
        d_results.clear();
        d_results.shrink_to_fit();
    }

    void setCounters(benchmark::State& state) {
        setCommonCounters(state, filterMemory, n);
    }

    size_t capacity;
    uint32_t q;
    size_t n;
    size_t filterMemory;
    thrust::device_vector<uint64_t> d_keys;
    thrust::device_vector<uint64_t> d_keysNegative;
    thrust::device_vector<uint64_t> d_results;
    QF* qf;
    GPUTimer timer;
};

#define BENCHMARK_CONFIG_LF             \
    ->Arg(FIXED_CAPACITY)               \
        ->Unit(benchmark::kMillisecond) \
        ->UseManualTime()               \
        ->Iterations(10)                \
        ->Repetitions(5)                \
        ->ReportAggregatesOnly(true)

#ifdef __x86_64__
    #define DEFINE_PARTITIONED_FOR_LOAD_FACTOR(ID, LF)                                           \
        /* Partitioned Cuckoo Filter */                                                          \
        using PCF_##ID = PartitionedCFFixture<(LF) * 0.01>;                                      \
        BENCHMARK_DEFINE_F(PCF_##ID, Insert)(bm::State & state) {                                \
            for (auto _ : state) {                                                               \
                PartitionedCuckooFilter tempFilter(s, n_partitions, n_threads, n_tasks);         \
                auto constructKeys = keys;                                                       \
                timer.start();                                                                   \
                bool success = tempFilter.construct(constructKeys.data(), constructKeys.size()); \
                state.SetIterationTime(timer.elapsed());                                         \
                bm::DoNotOptimize(success);                                                      \
            }                                                                                    \
            PartitionedCuckooFilter finalFilter(s, n_partitions, n_threads, n_tasks);            \
            finalFilter.construct(keys.data(), keys.size());                                     \
            size_t filterMemory = finalFilter.size();                                            \
            setCounters(state, filterMemory);                                                    \
        }                                                                                        \
        BENCHMARK_DEFINE_F(PCF_##ID, Query)(bm::State & state) {                                 \
            PartitionedCuckooFilter filter(s, n_partitions, n_threads, n_tasks);                 \
            filter.construct(keys.data(), keys.size());                                          \
            size_t filterMemory = filter.size();                                                 \
            for (auto _ : state) {                                                               \
                timer.start();                                                                   \
                size_t found = filter.count(keys.data(), keys.size());                           \
                state.SetIterationTime(timer.elapsed());                                         \
                bm::DoNotOptimize(found);                                                        \
            }                                                                                    \
            setCounters(state, filterMemory);                                                    \
        }                                                                                        \
        BENCHMARK_DEFINE_F(PCF_##ID, QueryNegative)(bm::State & state) {                         \
            PartitionedCuckooFilter filter(s, n_partitions, n_threads, n_tasks);                 \
            filter.construct(keys.data(), keys.size());                                          \
            size_t filterMemory = filter.size();                                                 \
            for (auto _ : state) {                                                               \
                timer.start();                                                                   \
                size_t found = filter.count(keysNegative.data(), keysNegative.size());           \
                state.SetIterationTime(timer.elapsed());                                         \
                bm::DoNotOptimize(found);                                                        \
            }                                                                                    \
            setCounters(state, filterMemory);                                                    \
        }                                                                                        \
        BENCHMARK_REGISTER_F(PCF_##ID, Insert) BENCHMARK_CONFIG_LF;                              \
        BENCHMARK_REGISTER_F(PCF_##ID, Query) BENCHMARK_CONFIG_LF;                               \
        BENCHMARK_REGISTER_F(PCF_##ID, QueryNegative) BENCHMARK_CONFIG_LF;
#else
    #define DEFINE_PARTITIONED_FOR_LOAD_FACTOR(ID, LF)
#endif

#define DEFINE_AND_REGISTER_ALL_FOR_LOAD_FACTOR(ID, LF)                                       \
    /* GPU Cuckoo Filter */                                                                   \
    using GCF_##ID = GCFFixtureLF<(LF) * 0.01>;                                               \
    BENCHMARK_DEFINE_F(GCF_##ID, Insert)(bm::State & state) {                                 \
        for (auto _ : state) {                                                                \
            filter->clear();                                                                  \
            cudaDeviceSynchronize();                                                          \
            timer.start();                                                                    \
            size_t inserted = adaptiveInsert(*filter, d_keys);                                \
            state.SetIterationTime(timer.elapsed());                                          \
            bm::DoNotOptimize(inserted);                                                      \
        }                                                                                     \
        setCounters(state);                                                                   \
    }                                                                                         \
    BENCHMARK_DEFINE_F(GCF_##ID, Query)(bm::State & state) {                                  \
        adaptiveInsert(*filter, d_keys);                                                      \
        cudaDeviceSynchronize();                                                              \
        for (auto _ : state) {                                                                \
            timer.start();                                                                    \
            filter->containsMany(d_keys, d_output);                                           \
            state.SetIterationTime(timer.elapsed());                                          \
            bm::DoNotOptimize(d_output.data().get());                                         \
        }                                                                                     \
        setCounters(state);                                                                   \
    }                                                                                         \
    BENCHMARK_DEFINE_F(GCF_##ID, QueryNegative)(bm::State & state) {                          \
        adaptiveInsert(*filter, d_keys);                                                      \
        cudaDeviceSynchronize();                                                              \
        for (auto _ : state) {                                                                \
            timer.start();                                                                    \
            filter->containsMany(d_keysNegative, d_output);                                   \
            state.SetIterationTime(timer.elapsed());                                          \
            bm::DoNotOptimize(d_output.data().get());                                         \
        }                                                                                     \
        setCounters(state);                                                                   \
    }                                                                                         \
    BENCHMARK_DEFINE_F(GCF_##ID, Delete)(bm::State & state) {                                 \
        for (auto _ : state) {                                                                \
            filter->clear();                                                                  \
            adaptiveInsert(*filter, d_keys);                                                  \
            cudaDeviceSynchronize();                                                          \
            timer.start();                                                                    \
            size_t remaining = filter->deleteMany(d_keys, d_output);                          \
            state.SetIterationTime(timer.elapsed());                                          \
            bm::DoNotOptimize(remaining);                                                     \
            bm::DoNotOptimize(d_output.data().get());                                         \
        }                                                                                     \
        setCounters(state);                                                                   \
    }                                                                                         \
    BENCHMARK_REGISTER_F(GCF_##ID, Insert) BENCHMARK_CONFIG_LF;                               \
    BENCHMARK_REGISTER_F(GCF_##ID, Query) BENCHMARK_CONFIG_LF;                                \
    BENCHMARK_REGISTER_F(GCF_##ID, QueryNegative) BENCHMARK_CONFIG_LF;                        \
    BENCHMARK_REGISTER_F(GCF_##ID, Delete) BENCHMARK_CONFIG_LF;                               \
                                                                                              \
    /* CPU Cuckoo Filter (2014) */                                                            \
    using CCF_##ID = CCFFixture<(LF) * 0.01>;                                                 \
    BENCHMARK_DEFINE_F(CCF_##ID, Insert)(bm::State & state) {                                 \
        for (auto _ : state) {                                                                \
            cuckoofilter::CuckooFilter<uint64_t, Config::bitsPerTag> tempFilter(capacity);    \
            timer.start();                                                                    \
            for (const auto& key : keys) {                                                    \
                auto status = tempFilter.Add(key);                                            \
                bm::DoNotOptimize(status);                                                    \
            }                                                                                 \
            state.SetIterationTime(timer.elapsed());                                          \
        }                                                                                     \
        size_t filterMemory =                                                                 \
            cuckoofilter::CuckooFilter<uint64_t, Config::bitsPerTag>(capacity).SizeInBytes(); \
        setCounters(state, filterMemory);                                                     \
    }                                                                                         \
    BENCHMARK_DEFINE_F(CCF_##ID, Query)(bm::State & state) {                                  \
        cuckoofilter::CuckooFilter<uint64_t, Config::bitsPerTag> filter(capacity);            \
        for (const auto& key : keys) {                                                        \
            filter.Add(key);                                                                  \
        }                                                                                     \
        size_t filterMemory = filter.SizeInBytes();                                           \
        for (auto _ : state) {                                                                \
            timer.start();                                                                    \
            for (const auto& key : keys) {                                                    \
                auto status = filter.Contain(key);                                            \
                bm::DoNotOptimize(status);                                                    \
            }                                                                                 \
            state.SetIterationTime(timer.elapsed());                                          \
        }                                                                                     \
        setCounters(state, filterMemory);                                                     \
    }                                                                                         \
    BENCHMARK_DEFINE_F(CCF_##ID, QueryNegative)(bm::State & state) {                          \
        cuckoofilter::CuckooFilter<uint64_t, Config::bitsPerTag> filter(capacity);            \
        for (const auto& key : keys) {                                                        \
            filter.Add(key);                                                                  \
        }                                                                                     \
        size_t filterMemory = filter.SizeInBytes();                                           \
        for (auto _ : state) {                                                                \
            timer.start();                                                                    \
            for (const auto& key : keysNegative) {                                            \
                auto status = filter.Contain(key);                                            \
                bm::DoNotOptimize(status);                                                    \
            }                                                                                 \
            state.SetIterationTime(timer.elapsed());                                          \
        }                                                                                     \
        setCounters(state, filterMemory);                                                     \
    }                                                                                         \
    BENCHMARK_DEFINE_F(CCF_##ID, Delete)(bm::State & state) {                                 \
        for (auto _ : state) {                                                                \
            cuckoofilter::CuckooFilter<uint64_t, Config::bitsPerTag> tempFilter(capacity);    \
            for (const auto& key : keys) {                                                    \
                tempFilter.Add(key);                                                          \
            }                                                                                 \
            timer.start();                                                                    \
            for (const auto& key : keys) {                                                    \
                auto status = tempFilter.Delete(key);                                         \
                bm::DoNotOptimize(status);                                                    \
            }                                                                                 \
            state.SetIterationTime(timer.elapsed());                                          \
        }                                                                                     \
        size_t filterMemory =                                                                 \
            cuckoofilter::CuckooFilter<uint64_t, Config::bitsPerTag>(capacity).SizeInBytes(); \
        setCounters(state, filterMemory);                                                     \
    }                                                                                         \
    BENCHMARK_REGISTER_F(CCF_##ID, Insert) BENCHMARK_CONFIG_LF;                               \
    BENCHMARK_REGISTER_F(CCF_##ID, Query) BENCHMARK_CONFIG_LF;                                \
    BENCHMARK_REGISTER_F(CCF_##ID, QueryNegative) BENCHMARK_CONFIG_LF;                        \
    BENCHMARK_REGISTER_F(CCF_##ID, Delete) BENCHMARK_CONFIG_LF;                               \
                                                                                              \
    /* Bloom Filter */                                                                        \
    using BBF_##ID = BloomFilterFixture<(LF) * 0.01>;                                         \
    BENCHMARK_DEFINE_F(BBF_##ID, Insert)(bm::State & state) {                                 \
        for (auto _ : state) {                                                                \
            filter->clear();                                                                  \
            cudaDeviceSynchronize();                                                          \
            timer.start();                                                                    \
            filter->add(d_keys.begin(), d_keys.end());                                        \
            state.SetIterationTime(timer.elapsed());                                          \
        }                                                                                     \
        setCounters(state);                                                                   \
    }                                                                                         \
    BENCHMARK_DEFINE_F(BBF_##ID, Query)(bm::State & state) {                                  \
        filter->add(d_keys.begin(), d_keys.end());                                            \
        cudaDeviceSynchronize();                                                              \
        for (auto _ : state) {                                                                \
            timer.start();                                                                    \
            filter->contains(                                                                 \
                d_keys.begin(),                                                               \
                d_keys.end(),                                                                 \
                reinterpret_cast<bool*>(thrust::raw_pointer_cast(d_output.data()))            \
            );                                                                                \
            state.SetIterationTime(timer.elapsed());                                          \
            bm::DoNotOptimize(d_output.data().get());                                         \
        }                                                                                     \
        setCounters(state);                                                                   \
    }                                                                                         \
    BENCHMARK_DEFINE_F(BBF_##ID, QueryNegative)(bm::State & state) {                          \
        filter->add(d_keys.begin(), d_keys.end());                                            \
        cudaDeviceSynchronize();                                                              \
        for (auto _ : state) {                                                                \
            timer.start();                                                                    \
            filter->contains(                                                                 \
                d_keysNegative.begin(),                                                       \
                d_keysNegative.end(),                                                         \
                reinterpret_cast<bool*>(thrust::raw_pointer_cast(d_output.data()))            \
            );                                                                                \
            state.SetIterationTime(timer.elapsed());                                          \
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
            state.SetIterationTime(timer.elapsed());                                          \
            TCFType::host_free_tcf(filter);                                                   \
        }                                                                                     \
        setCounters(state);                                                                   \
    }                                                                                         \
    BENCHMARK_DEFINE_F(TCF_##ID, Query)(bm::State & state) {                                  \
        TCFType* filter = TCFType::host_build_tcf(capacity);                                  \
        cudaMemset(d_misses, 0, sizeof(uint64_t));                                            \
        filter->bulk_insert(thrust::raw_pointer_cast(d_keys.data()), n, d_misses);            \
        cudaDeviceSynchronize();                                                              \
        for (auto _ : state) {                                                                \
            timer.start();                                                                    \
            bool* d_output = filter->bulk_query(thrust::raw_pointer_cast(d_keys.data()), n);  \
            state.SetIterationTime(timer.elapsed());                                          \
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
        cudaDeviceSynchronize();                                                              \
        for (auto _ : state) {                                                                \
            timer.start();                                                                    \
            bool* d_output =                                                                  \
                filter->bulk_query(thrust::raw_pointer_cast(d_keysNegative.data()), n);       \
            state.SetIterationTime(timer.elapsed());                                          \
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
            state.SetIterationTime(timer.elapsed());                                          \
            bm::DoNotOptimize(d_output);                                                      \
            cudaFree(d_output);                                                               \
            TCFType::host_free_tcf(filter);                                                   \
        }                                                                                     \
        setCounters(state);                                                                   \
    }                                                                                         \
    BENCHMARK_REGISTER_F(TCF_##ID, Insert) BENCHMARK_CONFIG_LF;                               \
    BENCHMARK_REGISTER_F(TCF_##ID, Query) BENCHMARK_CONFIG_LF;                                \
    BENCHMARK_REGISTER_F(TCF_##ID, QueryNegative) BENCHMARK_CONFIG_LF;                        \
    BENCHMARK_REGISTER_F(TCF_##ID, Delete) BENCHMARK_CONFIG_LF;                               \
                                                                                              \
    /* BGHT */                                                                                \
    using BGHT_##ID = BGHTFixtureLF<(LF) * 0.01>;                                             \
    BENCHMARK_DEFINE_F(BGHT_##ID, Insert)(bm::State & state) {                                \
        for (auto _ : state) {                                                                \
            resetTable();                                                                      \
            cudaDeviceSynchronize();                                                          \
            timer.start();                                                                    \
            bool inserted = table->insert(d_pairs.begin(), d_pairs.end());                    \
            state.SetIterationTime(timer.elapsed());                                          \
            bm::DoNotOptimize(inserted);                                                      \
            if (!inserted) {                                                                  \
                state.SkipWithError("BGHT BCHT insertion failed.");                           \
                break;                                                                        \
            }                                                                                 \
        }                                                                                     \
        setCounters(state);                                                                   \
    }                                                                                         \
    BENCHMARK_DEFINE_F(BGHT_##ID, Query)(bm::State & state) {                                 \
        resetTable();                                                                         \
        bool inserted = table->insert(d_pairs.begin(), d_pairs.end());                        \
        if (!inserted) {                                                                      \
            state.SkipWithError("BGHT BCHT insertion failed during query setup.");            \
            return;                                                                           \
        }                                                                                     \
        cudaDeviceSynchronize();                                                              \
        for (auto _ : state) {                                                                \
            timer.start();                                                                    \
            table->find(d_keys.begin(), d_keys.end(), d_output.begin());                      \
            state.SetIterationTime(timer.elapsed());                                          \
            bm::DoNotOptimize(d_output.data().get());                                         \
        }                                                                                     \
        setCounters(state);                                                                   \
    }                                                                                         \
    BENCHMARK_DEFINE_F(BGHT_##ID, QueryNegative)(bm::State & state) {                         \
        resetTable();                                                                         \
        bool inserted = table->insert(d_pairs.begin(), d_pairs.end());                        \
        if (!inserted) {                                                                      \
            state.SkipWithError("BGHT BCHT insertion failed during query setup.");            \
            return;                                                                           \
        }                                                                                     \
        cudaDeviceSynchronize();                                                              \
        for (auto _ : state) {                                                                \
            timer.start();                                                                    \
            table->find(d_keysNegative.begin(), d_keysNegative.end(), d_output.begin());      \
            state.SetIterationTime(timer.elapsed());                                          \
            bm::DoNotOptimize(d_output.data().get());                                         \
        }                                                                                     \
        setCounters(state);                                                                   \
    }                                                                                         \
    BENCHMARK_REGISTER_F(BGHT_##ID, Insert) BENCHMARK_CONFIG_LF;                              \
    BENCHMARK_REGISTER_F(BGHT_##ID, Query) BENCHMARK_CONFIG_LF;                               \
    BENCHMARK_REGISTER_F(BGHT_##ID, QueryNegative) BENCHMARK_CONFIG_LF;                       \
                                                                                              \
    /* GQF */                                                                                 \
    using GQF_##ID = GQFFixtureLF<(LF) * 0.01>;                                               \
    BENCHMARK_DEFINE_F(GQF_##ID, Insert)(bm::State & state) {                                 \
        for (auto _ : state) {                                                                \
            qf_destroy_device(qf);                                                            \
            cudaFree(qf);                                                                     \
            qf_malloc_device(&qf, q, true);                                                   \
            cudaDeviceSynchronize();                                                          \
            timer.start();                                                                    \
            bulk_insert(qf, n, thrust::raw_pointer_cast(d_keys.data()), 0);                   \
            state.SetIterationTime(timer.elapsed());                                          \
        }                                                                                     \
        setCounters(state);                                                                   \
    }                                                                                         \
    BENCHMARK_DEFINE_F(GQF_##ID, Query)(bm::State & state) {                                  \
        bulk_insert(qf, n, thrust::raw_pointer_cast(d_keys.data()), 0);                       \
        cudaDeviceSynchronize();                                                              \
        for (auto _ : state) {                                                                \
            timer.start();                                                                    \
            bulk_get(                                                                         \
                qf,                                                                           \
                n,                                                                            \
                thrust::raw_pointer_cast(d_keys.data()),                                      \
                thrust::raw_pointer_cast(d_results.data())                                    \
            );                                                                                \
            state.SetIterationTime(timer.elapsed());                                          \
            bm::DoNotOptimize(d_results.data().get());                                        \
        }                                                                                     \
        setCounters(state);                                                                   \
    }                                                                                         \
    BENCHMARK_DEFINE_F(GQF_##ID, QueryNegative)(bm::State & state) {                          \
        bulk_insert(qf, n, thrust::raw_pointer_cast(d_keys.data()), 0);                       \
        cudaDeviceSynchronize();                                                              \
        for (auto _ : state) {                                                                \
            timer.start();                                                                    \
            bulk_get(                                                                         \
                qf,                                                                           \
                n,                                                                            \
                thrust::raw_pointer_cast(d_keysNegative.data()),                              \
                thrust::raw_pointer_cast(d_results.data())                                    \
            );                                                                                \
            state.SetIterationTime(timer.elapsed());                                          \
            bm::DoNotOptimize(d_results.data().get());                                        \
        }                                                                                     \
        setCounters(state);                                                                   \
    }                                                                                         \
    BENCHMARK_DEFINE_F(GQF_##ID, Delete)(bm::State & state) {                                 \
        for (auto _ : state) {                                                                \
            qf_destroy_device(qf);                                                            \
            cudaFree(qf);                                                                     \
            qf_malloc_device(&qf, q, true);                                                   \
            cudaDeviceSynchronize();                                                          \
            bulk_insert(qf, n, thrust::raw_pointer_cast(d_keys.data()), 0);                   \
            cudaDeviceSynchronize();                                                          \
            timer.start();                                                                    \
            bulk_delete(qf, n, thrust::raw_pointer_cast(d_keys.data()), 0);                   \
            state.SetIterationTime(timer.elapsed());                                          \
        }                                                                                     \
        setCounters(state);                                                                   \
    }                                                                                         \
    BENCHMARK_REGISTER_F(GQF_##ID, Insert) BENCHMARK_CONFIG_LF;                               \
    BENCHMARK_REGISTER_F(GQF_##ID, Query) BENCHMARK_CONFIG_LF;                                \
    BENCHMARK_REGISTER_F(GQF_##ID, QueryNegative) BENCHMARK_CONFIG_LF;                        \
    BENCHMARK_REGISTER_F(GQF_##ID, Delete) BENCHMARK_CONFIG_LF;                               \
                                                                                              \
    DEFINE_PARTITIONED_FOR_LOAD_FACTOR(ID, LF)

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
