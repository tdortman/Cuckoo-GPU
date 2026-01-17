#include <benchmark/benchmark.h>
#include <cuckoofilter.h>
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <bucket_policies.cuh>
#include <bulk_tcf_host.cuh>
#include <cstddef>
#include <cstdint>
#include <CuckooFilter.cuh>
#include <cuco/bloom_filter.cuh>
#include <cuda/std/cstdint>
#include <gqf.cuh>
#include <gqf_int.cuh>
#include <helpers.cuh>
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

using Config = CuckooConfig<uint64_t, 16, 500, 128, 16, XorAltBucketPolicy>;
using TCFType = host_bulk_tcf<uint64_t, uint16_t>;

#ifdef __x86_64__
using CPUFilterParam = filters::cuckoo::Standard4<Config::bitsPerTag>;
using CPUOptimParam = filters::parameter::PowerOfTwoMurmurScalar64PartitionedMT;
using PartitionedCuckooFilter =
    filters::Filter<filters::FilterType::Cuckoo, CPUFilterParam, Config::bitsPerTag, CPUOptimParam>;
#endif

constexpr size_t FIXED_CAPACITY = 1ULL << 28;
const size_t L2_CACHE_SIZE = getL2CacheSize();



template <int loadFactorPercent>
class GPUCuckooFixture : public benchmark::Fixture {
   public:
    static constexpr double loadFactor = loadFactorPercent / 100.0;

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

    size_t capacity;
    size_t n;
    size_t filterMemory;
    thrust::device_vector<uint64_t> d_keys;
    thrust::device_vector<uint64_t> d_keysNegative;
    thrust::device_vector<uint8_t> d_output;
    std::unique_ptr<CuckooFilter<Config>> filter;
    GPUTimer timer;
};

template <int loadFactorPercent>
class CPUCuckooFixture : public benchmark::Fixture {
   public:
    static constexpr double loadFactor = loadFactorPercent / 100.0;

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

    size_t capacity;
    size_t n;
    std::vector<uint64_t> keys;
    std::vector<uint64_t> keysNegative;
    CPUTimer timer;
};

template <typename Filter>
size_t cucoNumBlocks(size_t n) {
    constexpr auto bitsPerWord = sizeof(typename Filter::word_type) * 8;
    return SDIV(n * Config::bitsPerTag, Filter::words_per_block * bitsPerWord);
}

template <int loadFactorPercent>
class BloomFilterFixture : public benchmark::Fixture {
   public:
    using BloomFilter = cuco::bloom_filter<uint64_t>;
    static constexpr double loadFactor = loadFactorPercent / 100.0;

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

    size_t capacity;
    size_t n;
    size_t filterMemory;
    thrust::device_vector<uint64_t> d_keys;
    thrust::device_vector<uint64_t> d_keysNegative;
    thrust::device_vector<uint8_t> d_output;
    std::unique_ptr<BloomFilter> filter;
    GPUTimer timer;
};

template <int loadFactorPercent>
class TCFFixture : public benchmark::Fixture {
   public:
    static constexpr double loadFactor = loadFactorPercent / 100.0;

    void SetUp(const benchmark::State& state) override {
        auto [cap, num] = calculateCapacityAndSize(state.range(0), loadFactor);
        n = num;

        constexpr double TCF_CAPACITY_FACTOR = 0.85;
        auto requiredUsableCapacity = static_cast<size_t>(n / loadFactor);
        capacity = static_cast<size_t>(requiredUsableCapacity / TCF_CAPACITY_FACTOR);

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

    size_t capacity;
    size_t n;
    size_t filterMemory;
    uint64_t* d_misses = nullptr;
    thrust::device_vector<uint64_t> d_keys;
    thrust::device_vector<uint64_t> d_keysNegative;
    GPUTimer timer;
};

template <int loadFactorPercent>
class GQFFixture : public benchmark::Fixture {
   public:
    static constexpr double loadFactor = loadFactorPercent / 100.0;

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

#ifdef __x86_64__
template <int loadFactorPercent>
class PartitionedCFFixture : public benchmark::Fixture {
   public:
    static constexpr double loadFactor = loadFactorPercent / 100.0;

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
#endif  // __x86_64__

#define BENCHMARK_CONFIG_LF             \
    ->Arg(FIXED_CAPACITY)               \
        ->Unit(benchmark::kMillisecond) \
        ->UseManualTime()               \
        ->Iterations(10)                \
        ->Repetitions(5)                \
        ->ReportAggregatesOnly(true)

template <int LF>
void GPUCuckoo_Insert(benchmark::State& state, GPUCuckooFixture<LF>* f) {
    for (auto _ : state) {
        f->filter->clear();
        cudaDeviceSynchronize();
        f->timer.start();
        size_t inserted = adaptiveInsert(*f->filter, f->d_keys);
        state.SetIterationTime(f->timer.elapsed());
        bm::DoNotOptimize(inserted);
    }
    setCommonCounters(state, f->filterMemory, f->n);
}

template <int LF>
void GPUCuckoo_Query(benchmark::State& state, GPUCuckooFixture<LF>* f) {
    adaptiveInsert(*f->filter, f->d_keys);
    cudaDeviceSynchronize();
    for (auto _ : state) {
        f->timer.start();
        f->filter->containsMany(f->d_keys, f->d_output);
        state.SetIterationTime(f->timer.elapsed());
        bm::DoNotOptimize(f->d_output.data().get());
    }
    setCommonCounters(state, f->filterMemory, f->n);
}

template <int LF>
void GPUCuckoo_QueryNegative(benchmark::State& state, GPUCuckooFixture<LF>* f) {
    adaptiveInsert(*f->filter, f->d_keys);
    cudaDeviceSynchronize();
    for (auto _ : state) {
        f->timer.start();
        f->filter->containsMany(f->d_keysNegative, f->d_output);
        state.SetIterationTime(f->timer.elapsed());
        bm::DoNotOptimize(f->d_output.data().get());
    }
    setCommonCounters(state, f->filterMemory, f->n);
}

template <int LF>
void GPUCuckoo_Delete(benchmark::State& state, GPUCuckooFixture<LF>* f) {
    for (auto _ : state) {
        f->filter->clear();
        adaptiveInsert(*f->filter, f->d_keys);
        cudaDeviceSynchronize();
        f->timer.start();
        size_t remaining = f->filter->deleteMany(f->d_keys, f->d_output);
        state.SetIterationTime(f->timer.elapsed());
        bm::DoNotOptimize(remaining);
        bm::DoNotOptimize(f->d_output.data().get());
    }
    setCommonCounters(state, f->filterMemory, f->n);
}

template <int LF>
void CPUCuckoo_Insert(benchmark::State& state, CPUCuckooFixture<LF>* f) {
    for (auto _ : state) {
        cuckoofilter::CuckooFilter<uint64_t, Config::bitsPerTag> tempFilter(f->capacity);
        f->timer.start();
        size_t inserted = 0;
        for (const auto& key : f->keys) {
            if (tempFilter.Add(key) == cuckoofilter::Ok) {
                inserted++;
            }
        }
        state.SetIterationTime(f->timer.elapsed());
        bm::DoNotOptimize(inserted);
    }
    size_t filterMemory =
        cuckoofilter::CuckooFilter<uint64_t, Config::bitsPerTag>(f->capacity).SizeInBytes();
    setCommonCounters(state, filterMemory, f->n);
}

template <int LF>
void CPUCuckoo_Query(benchmark::State& state, CPUCuckooFixture<LF>* f) {
    cuckoofilter::CuckooFilter<uint64_t, Config::bitsPerTag> filter(f->capacity);
    for (const auto& key : f->keys) {
        filter.Add(key);
    }
    size_t filterMemory = filter.SizeInBytes();
    for (auto _ : state) {
        f->timer.start();
        size_t found = 0;
        for (const auto& key : f->keys) {
            if (filter.Contain(key) == cuckoofilter::Ok) {
                found++;
            }
        }
        state.SetIterationTime(f->timer.elapsed());
        bm::DoNotOptimize(found);
    }
    setCommonCounters(state, filterMemory, f->n);
}

template <int LF>
void CPUCuckoo_QueryNegative(benchmark::State& state, CPUCuckooFixture<LF>* f) {
    cuckoofilter::CuckooFilter<uint64_t, Config::bitsPerTag> filter(f->capacity);
    for (const auto& key : f->keys) {
        filter.Add(key);
    }
    size_t filterMemory = filter.SizeInBytes();
    for (auto _ : state) {
        f->timer.start();
        size_t found = 0;
        for (const auto& key : f->keysNegative) {
            if (filter.Contain(key) == cuckoofilter::Ok) {
                found++;
            }
        }
        state.SetIterationTime(f->timer.elapsed());
        bm::DoNotOptimize(found);
    }
    setCommonCounters(state, filterMemory, f->n);
}

template <int LF>
void Bloom_Insert(benchmark::State& state, BloomFilterFixture<LF>* f) {
    for (auto _ : state) {
        f->filter->clear();
        cudaDeviceSynchronize();
        f->timer.start();
        f->filter->add(f->d_keys.begin(), f->d_keys.end());
        state.SetIterationTime(f->timer.elapsed());
    }
    setCommonCounters(state, f->filterMemory, f->n);
}

template <int LF>
void Bloom_Query(benchmark::State& state, BloomFilterFixture<LF>* f) {
    f->filter->add(f->d_keys.begin(), f->d_keys.end());
    cudaDeviceSynchronize();
    for (auto _ : state) {
        f->timer.start();
        f->filter->contains(
            f->d_keys.begin(),
            f->d_keys.end(),
            reinterpret_cast<bool*>(thrust::raw_pointer_cast(f->d_output.data()))
        );
        state.SetIterationTime(f->timer.elapsed());
        bm::DoNotOptimize(f->d_output.data().get());
    }
    setCommonCounters(state, f->filterMemory, f->n);
}

template <int LF>
void Bloom_QueryNegative(benchmark::State& state, BloomFilterFixture<LF>* f) {
    f->filter->add(f->d_keys.begin(), f->d_keys.end());
    cudaDeviceSynchronize();
    for (auto _ : state) {
        f->timer.start();
        f->filter->contains(
            f->d_keysNegative.begin(),
            f->d_keysNegative.end(),
            reinterpret_cast<bool*>(thrust::raw_pointer_cast(f->d_output.data()))
        );
        state.SetIterationTime(f->timer.elapsed());
        bm::DoNotOptimize(f->d_output.data().get());
    }
    setCommonCounters(state, f->filterMemory, f->n);
}

template <int LF>
void TCF_Insert(benchmark::State& state, TCFFixture<LF>* f) {
    for (auto _ : state) {
        TCFType* filter = TCFType::host_build_tcf(f->capacity);
        cudaMemset(f->d_misses, 0, sizeof(uint64_t));
        cudaDeviceSynchronize();
        f->timer.start();
        filter->bulk_insert(thrust::raw_pointer_cast(f->d_keys.data()), f->n, f->d_misses);
        state.SetIterationTime(f->timer.elapsed());
        TCFType::host_free_tcf(filter);
    }
    setCommonCounters(state, f->filterMemory, f->n);
}

template <int LF>
void TCF_Query(benchmark::State& state, TCFFixture<LF>* f) {
    TCFType* filter = TCFType::host_build_tcf(f->capacity);
    cudaMemset(f->d_misses, 0, sizeof(uint64_t));
    filter->bulk_insert(thrust::raw_pointer_cast(f->d_keys.data()), f->n, f->d_misses);
    cudaDeviceSynchronize();
    for (auto _ : state) {
        f->timer.start();
        bool* d_output = filter->bulk_query(thrust::raw_pointer_cast(f->d_keys.data()), f->n);
        state.SetIterationTime(f->timer.elapsed());
        bm::DoNotOptimize(d_output);
        cudaFree(d_output);
    }
    TCFType::host_free_tcf(filter);
    setCommonCounters(state, f->filterMemory, f->n);
}

template <int LF>
void TCF_QueryNegative(benchmark::State& state, TCFFixture<LF>* f) {
    TCFType* filter = TCFType::host_build_tcf(f->capacity);
    cudaMemset(f->d_misses, 0, sizeof(uint64_t));
    filter->bulk_insert(thrust::raw_pointer_cast(f->d_keys.data()), f->n, f->d_misses);
    cudaDeviceSynchronize();
    for (auto _ : state) {
        f->timer.start();
        bool* d_output =
            filter->bulk_query(thrust::raw_pointer_cast(f->d_keysNegative.data()), f->n);
        state.SetIterationTime(f->timer.elapsed());
        bm::DoNotOptimize(d_output);
        cudaFree(d_output);
    }
    TCFType::host_free_tcf(filter);
    setCommonCounters(state, f->filterMemory, f->n);
}

template <int LF>
void TCF_Delete(benchmark::State& state, TCFFixture<LF>* f) {
    for (auto _ : state) {
        TCFType* filter = TCFType::host_build_tcf(f->capacity);
        cudaMemset(f->d_misses, 0, sizeof(uint64_t));
        filter->bulk_insert(thrust::raw_pointer_cast(f->d_keys.data()), f->n, f->d_misses);
        cudaDeviceSynchronize();
        f->timer.start();
        bool* d_output = filter->bulk_delete(thrust::raw_pointer_cast(f->d_keys.data()), f->n);
        state.SetIterationTime(f->timer.elapsed());
        bm::DoNotOptimize(d_output);
        cudaFree(d_output);
        TCFType::host_free_tcf(filter);
    }
    setCommonCounters(state, f->filterMemory, f->n);
}

template <int LF>
void GQF_Insert(benchmark::State& state, GQFFixture<LF>* f) {
    for (auto _ : state) {
        qf_destroy_device(f->qf);
        cudaFree(f->qf);
        qf_malloc_device(&f->qf, f->q, true);
        cudaDeviceSynchronize();
        f->timer.start();
        bulk_insert(f->qf, f->n, thrust::raw_pointer_cast(f->d_keys.data()), 0);
        state.SetIterationTime(f->timer.elapsed());
    }
    setCommonCounters(state, f->filterMemory, f->n);
}

template <int LF>
void GQF_Query(benchmark::State& state, GQFFixture<LF>* f) {
    bulk_insert(f->qf, f->n, thrust::raw_pointer_cast(f->d_keys.data()), 0);
    cudaDeviceSynchronize();
    for (auto _ : state) {
        f->timer.start();
        bulk_get(
            f->qf,
            f->n,
            thrust::raw_pointer_cast(f->d_keys.data()),
            thrust::raw_pointer_cast(f->d_results.data())
        );
        state.SetIterationTime(f->timer.elapsed());
        bm::DoNotOptimize(f->d_results.data().get());
    }
    setCommonCounters(state, f->filterMemory, f->n);
}

template <int LF>
void GQF_QueryNegative(benchmark::State& state, GQFFixture<LF>* f) {
    bulk_insert(f->qf, f->n, thrust::raw_pointer_cast(f->d_keys.data()), 0);
    cudaDeviceSynchronize();
    for (auto _ : state) {
        f->timer.start();
        bulk_get(
            f->qf,
            f->n,
            thrust::raw_pointer_cast(f->d_keysNegative.data()),
            thrust::raw_pointer_cast(f->d_results.data())
        );
        state.SetIterationTime(f->timer.elapsed());
        bm::DoNotOptimize(f->d_results.data().get());
    }
    setCommonCounters(state, f->filterMemory, f->n);
}

template <int LF>
void GQF_Delete(benchmark::State& state, GQFFixture<LF>* f) {
    for (auto _ : state) {
        qf_destroy_device(f->qf);
        cudaFree(f->qf);
        qf_malloc_device(&f->qf, f->q, true);
        cudaDeviceSynchronize();
        bulk_insert(f->qf, f->n, thrust::raw_pointer_cast(f->d_keys.data()), 0);
        cudaDeviceSynchronize();
        f->timer.start();
        bulk_delete(f->qf, f->n, thrust::raw_pointer_cast(f->d_keys.data()), 0);
        state.SetIterationTime(f->timer.elapsed());
    }
    setCommonCounters(state, f->filterMemory, f->n);
}

#ifdef __x86_64__
template <int LF>
void PartitionedCuckoo_Insert(benchmark::State& state, PartitionedCFFixture<LF>* f) {
    for (auto _ : state) {
        PartitionedCuckooFilter tempFilter(f->s, f->n_partitions, f->n_threads, f->n_tasks);
        auto constructKeys = f->keys;
        f->timer.start();
        bool success = tempFilter.construct(constructKeys.data(), constructKeys.size());
        state.SetIterationTime(f->timer.elapsed());
        bm::DoNotOptimize(success);
    }
    PartitionedCuckooFilter finalFilter(f->s, f->n_partitions, f->n_threads, f->n_tasks);
    finalFilter.construct(f->keys.data(), f->keys.size());
    size_t filterMemory = finalFilter.size();
    setCommonCounters(state, filterMemory, f->n);
}

template <int LF>
void PartitionedCuckoo_Query(benchmark::State& state, PartitionedCFFixture<LF>* f) {
    PartitionedCuckooFilter filter(f->s, f->n_partitions, f->n_threads, f->n_tasks);
    filter.construct(f->keys.data(), f->keys.size());
    size_t filterMemory = filter.size();
    for (auto _ : state) {
        f->timer.start();
        size_t found = filter.count(f->keys.data(), f->keys.size());
        state.SetIterationTime(f->timer.elapsed());
        bm::DoNotOptimize(found);
    }
    setCommonCounters(state, filterMemory, f->n);
}

template <int LF>
void PartitionedCuckoo_QueryNegative(benchmark::State& state, PartitionedCFFixture<LF>* f) {
    PartitionedCuckooFilter filter(f->s, f->n_partitions, f->n_threads, f->n_tasks);
    filter.construct(f->keys.data(), f->keys.size());
    size_t filterMemory = filter.size();
    for (auto _ : state) {
        f->timer.start();
        size_t found = filter.count(f->keysNegative.data(), f->keysNegative.size());
        state.SetIterationTime(f->timer.elapsed());
        bm::DoNotOptimize(found);
    }
    setCommonCounters(state, filterMemory, f->n);
}
#endif  // __x86_64__

#define DEFINE_BENCHMARKS_FOR_LF(LF)                                                             \
    BENCHMARK_TEMPLATE_DEFINE_F(GPUCuckooFixture, Insert_##LF, LF)(bm::State & state) {          \
        GPUCuckoo_Insert<LF>(state, this);                                                       \
    }                                                                                            \
    BENCHMARK_TEMPLATE_DEFINE_F(GPUCuckooFixture, Query_##LF, LF)(bm::State & state) {           \
        GPUCuckoo_Query<LF>(state, this);                                                        \
    }                                                                                            \
    BENCHMARK_TEMPLATE_DEFINE_F(GPUCuckooFixture, QueryNegative_##LF, LF)(bm::State & state) {   \
        GPUCuckoo_QueryNegative<LF>(state, this);                                                \
    }                                                                                            \
    BENCHMARK_TEMPLATE_DEFINE_F(GPUCuckooFixture, Delete_##LF, LF)(bm::State & state) {          \
        GPUCuckoo_Delete<LF>(state, this);                                                       \
    }                                                                                            \
    BENCHMARK_REGISTER_F(GPUCuckooFixture, Insert_##LF) BENCHMARK_CONFIG_LF;                     \
    BENCHMARK_REGISTER_F(GPUCuckooFixture, Query_##LF) BENCHMARK_CONFIG_LF;                      \
    BENCHMARK_REGISTER_F(GPUCuckooFixture, QueryNegative_##LF) BENCHMARK_CONFIG_LF;              \
    BENCHMARK_REGISTER_F(GPUCuckooFixture, Delete_##LF) BENCHMARK_CONFIG_LF;                     \
                                                                                                 \
    BENCHMARK_TEMPLATE_DEFINE_F(CPUCuckooFixture, Insert_##LF, LF)(bm::State & state) {          \
        CPUCuckoo_Insert<LF>(state, this);                                                       \
    }                                                                                            \
    BENCHMARK_TEMPLATE_DEFINE_F(CPUCuckooFixture, Query_##LF, LF)(bm::State & state) {           \
        CPUCuckoo_Query<LF>(state, this);                                                        \
    }                                                                                            \
    BENCHMARK_TEMPLATE_DEFINE_F(CPUCuckooFixture, QueryNegative_##LF, LF)(bm::State & state) {   \
        CPUCuckoo_QueryNegative<LF>(state, this);                                                \
    }                                                                                            \
    BENCHMARK_REGISTER_F(CPUCuckooFixture, Insert_##LF) BENCHMARK_CONFIG_LF;                     \
    BENCHMARK_REGISTER_F(CPUCuckooFixture, Query_##LF) BENCHMARK_CONFIG_LF;                      \
    BENCHMARK_REGISTER_F(CPUCuckooFixture, QueryNegative_##LF) BENCHMARK_CONFIG_LF;              \
                                                                                                 \
    BENCHMARK_TEMPLATE_DEFINE_F(BloomFilterFixture, Insert_##LF, LF)(bm::State & state) {        \
        Bloom_Insert<LF>(state, this);                                                           \
    }                                                                                            \
    BENCHMARK_TEMPLATE_DEFINE_F(BloomFilterFixture, Query_##LF, LF)(bm::State & state) {         \
        Bloom_Query<LF>(state, this);                                                            \
    }                                                                                            \
    BENCHMARK_TEMPLATE_DEFINE_F(BloomFilterFixture, QueryNegative_##LF, LF)(bm::State & state) { \
        Bloom_QueryNegative<LF>(state, this);                                                    \
    }                                                                                            \
    BENCHMARK_REGISTER_F(BloomFilterFixture, Insert_##LF) BENCHMARK_CONFIG_LF;                   \
    BENCHMARK_REGISTER_F(BloomFilterFixture, Query_##LF) BENCHMARK_CONFIG_LF;                    \
    BENCHMARK_REGISTER_F(BloomFilterFixture, QueryNegative_##LF) BENCHMARK_CONFIG_LF;            \
                                                                                                 \
    BENCHMARK_TEMPLATE_DEFINE_F(TCFFixture, Insert_##LF, LF)(bm::State & state) {                \
        TCF_Insert<LF>(state, this);                                                             \
    }                                                                                            \
    BENCHMARK_TEMPLATE_DEFINE_F(TCFFixture, Query_##LF, LF)(bm::State & state) {                 \
        TCF_Query<LF>(state, this);                                                              \
    }                                                                                            \
    BENCHMARK_TEMPLATE_DEFINE_F(TCFFixture, QueryNegative_##LF, LF)(bm::State & state) {         \
        TCF_QueryNegative<LF>(state, this);                                                      \
    }                                                                                            \
    BENCHMARK_TEMPLATE_DEFINE_F(TCFFixture, Delete_##LF, LF)(bm::State & state) {                \
        TCF_Delete<LF>(state, this);                                                             \
    }                                                                                            \
    BENCHMARK_REGISTER_F(TCFFixture, Insert_##LF) BENCHMARK_CONFIG_LF;                           \
    BENCHMARK_REGISTER_F(TCFFixture, Query_##LF) BENCHMARK_CONFIG_LF;                            \
    BENCHMARK_REGISTER_F(TCFFixture, QueryNegative_##LF) BENCHMARK_CONFIG_LF;                    \
    BENCHMARK_REGISTER_F(TCFFixture, Delete_##LF) BENCHMARK_CONFIG_LF;                           \
                                                                                                 \
    BENCHMARK_TEMPLATE_DEFINE_F(GQFFixture, Insert_##LF, LF)(bm::State & state) {                \
        GQF_Insert<LF>(state, this);                                                             \
    }                                                                                            \
    BENCHMARK_TEMPLATE_DEFINE_F(GQFFixture, Query_##LF, LF)(bm::State & state) {                 \
        GQF_Query<LF>(state, this);                                                              \
    }                                                                                            \
    BENCHMARK_TEMPLATE_DEFINE_F(GQFFixture, QueryNegative_##LF, LF)(bm::State & state) {         \
        GQF_QueryNegative<LF>(state, this);                                                      \
    }                                                                                            \
    BENCHMARK_TEMPLATE_DEFINE_F(GQFFixture, Delete_##LF, LF)(bm::State & state) {                \
        GQF_Delete<LF>(state, this);                                                             \
    }                                                                                            \
    BENCHMARK_REGISTER_F(GQFFixture, Insert_##LF) BENCHMARK_CONFIG_LF;                           \
    BENCHMARK_REGISTER_F(GQFFixture, Query_##LF) BENCHMARK_CONFIG_LF;                            \
    BENCHMARK_REGISTER_F(GQFFixture, QueryNegative_##LF) BENCHMARK_CONFIG_LF;                    \
    BENCHMARK_REGISTER_F(GQFFixture, Delete_##LF) BENCHMARK_CONFIG_LF;

#ifdef __x86_64__
    #define DEFINE_PARTITIONED_BENCHMARKS_FOR_LF(LF)                                            \
        BENCHMARK_TEMPLATE_DEFINE_F(PartitionedCFFixture, Insert_##LF, LF)(bm::State & state) { \
            PartitionedCuckoo_Insert<LF>(state, this);                                          \
        }                                                                                       \
        BENCHMARK_TEMPLATE_DEFINE_F(PartitionedCFFixture, Query_##LF, LF)(bm::State & state) {  \
            PartitionedCuckoo_Query<LF>(state, this);                                           \
        }                                                                                       \
        BENCHMARK_TEMPLATE_DEFINE_F(PartitionedCFFixture, QueryNegative_##LF, LF)(              \
            bm::State & state                                                                   \
        ) {                                                                                     \
            PartitionedCuckoo_QueryNegative<LF>(state, this);                                   \
        }                                                                                       \
        BENCHMARK_REGISTER_F(PartitionedCFFixture, Insert_##LF) BENCHMARK_CONFIG_LF;            \
        BENCHMARK_REGISTER_F(PartitionedCFFixture, Query_##LF) BENCHMARK_CONFIG_LF;             \
        BENCHMARK_REGISTER_F(PartitionedCFFixture, QueryNegative_##LF) BENCHMARK_CONFIG_LF;
#else
    #define DEFINE_PARTITIONED_BENCHMARKS_FOR_LF(LF)
#endif

#define DEFINE_ALL_BENCHMARKS_FOR_LF(LF) \
    DEFINE_BENCHMARKS_FOR_LF(LF)         \
    DEFINE_PARTITIONED_BENCHMARKS_FOR_LF(LF)

DEFINE_ALL_BENCHMARKS_FOR_LF(5)
DEFINE_ALL_BENCHMARKS_FOR_LF(10)
DEFINE_ALL_BENCHMARKS_FOR_LF(15)
DEFINE_ALL_BENCHMARKS_FOR_LF(20)
DEFINE_ALL_BENCHMARKS_FOR_LF(25)
DEFINE_ALL_BENCHMARKS_FOR_LF(30)
DEFINE_ALL_BENCHMARKS_FOR_LF(35)
DEFINE_ALL_BENCHMARKS_FOR_LF(40)
DEFINE_ALL_BENCHMARKS_FOR_LF(45)
DEFINE_ALL_BENCHMARKS_FOR_LF(50)
DEFINE_ALL_BENCHMARKS_FOR_LF(55)
DEFINE_ALL_BENCHMARKS_FOR_LF(60)
DEFINE_ALL_BENCHMARKS_FOR_LF(65)
DEFINE_ALL_BENCHMARKS_FOR_LF(70)
DEFINE_ALL_BENCHMARKS_FOR_LF(75)
DEFINE_ALL_BENCHMARKS_FOR_LF(80)
DEFINE_ALL_BENCHMARKS_FOR_LF(85)
DEFINE_ALL_BENCHMARKS_FOR_LF(90)
DEFINE_ALL_BENCHMARKS_FOR_LF(95)
DEFINE_ALL_BENCHMARKS_FOR_LF(98)
DEFINE_ALL_BENCHMARKS_FOR_LF(99)

STANDARD_BENCHMARK_MAIN();
