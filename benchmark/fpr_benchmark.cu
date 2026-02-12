
#include <benchmark/benchmark.h>
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

namespace bm = benchmark;

using Config = cuckoogpu::Config<uint64_t, 16, 500, 128, 16, cuckoogpu::XorAltBucketPolicy>;
using TCFType = host_bulk_tcf<uint64_t, uint16_t>;
using BloomFilter = cuco::bloom_filter<uint64_t>;

size_t getQFSizeHost(QF* d_qf) {
    QF h_qf;
    cudaMemcpy(&h_qf, d_qf, sizeof(QF), cudaMemcpyDeviceToHost);

    qfmetadata h_metadata;
    cudaMemcpy(&h_metadata, h_qf.metadata, sizeof(qfmetadata), cudaMemcpyDeviceToHost);

    return h_metadata.total_size_in_bytes;
}

/**
 * GQF doesn't return a bit vector, but rather a count for each key
 */
void convertGQFResults(thrust::device_vector<uint64_t>& d_results) {
    thrust::device_ptr<uint64_t> d_resultsPtr(d_results.data().get());
    thrust::transform(
        d_resultsPtr, d_resultsPtr + d_results.size(), d_resultsPtr, [] __device__(uint64_t val) {
            return val > 0;
        }
    );
}

#ifdef __x86_64__
using CPUFilterParam = filters::cuckoo::Standard4<Config::bitsPerTag>;
using CPUOptimParam = filters::parameter::PowerOfTwoMurmurScalar64PartitionedMT;
using PartitionedCuckooFilter =
    filters::Filter<filters::FilterType::Cuckoo, CPUFilterParam, Config::bitsPerTag, CPUOptimParam>;
#endif

constexpr double LOAD_FACTOR = 0.95;
const size_t L2_CACHE_SIZE = getL2CacheSize();
constexpr size_t FPR_TEST_SIZE = 1'000'000;

static void GCF_FPR(bm::State& state) {
    GPUTimer timer;
    size_t targetMemory = state.range(0);

    size_t capacity = (targetMemory * 8) / Config::bitsPerTag;
    auto n = static_cast<size_t>(capacity * LOAD_FACTOR);

    thrust::device_vector<uint64_t> d_keys(n);
    generateKeysGPURange(d_keys, n, uint64_t(0), uint64_t(UINT32_MAX));

    auto filter = std::make_unique<cuckoogpu::Filter<Config>>(capacity);
    size_t filterMemory = filter->sizeInBytes();
    adaptiveInsert(*filter, d_keys);

    thrust::device_vector<uint64_t> d_neverInserted(FPR_TEST_SIZE);
    thrust::device_vector<uint8_t> d_output(FPR_TEST_SIZE);

    generateKeysGPURange(d_neverInserted, FPR_TEST_SIZE, uint64_t(UINT32_MAX) + 1, UINT64_MAX);

    for (auto _ : state) {
        timer.start();
        filter->containsMany(d_neverInserted, d_output);
        double elapsed = timer.elapsed();

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(d_output.data().get());
    }

    size_t falsePositives =
        thrust::reduce(d_output.begin(), d_output.end(), 0ULL, cuda::std::plus<size_t>());
    double fpr = static_cast<double>(falsePositives) / static_cast<double>(FPR_TEST_SIZE);

    setFPRCounters(state, filterMemory, n, fpr, falsePositives, FPR_TEST_SIZE);
}

static void CCF_FPR(bm::State& state) {
    CPUTimer timer;
    size_t targetMemory = state.range(0);
    size_t capacity = (targetMemory * 8) / Config::bitsPerTag;
    auto n = static_cast<size_t>(capacity * LOAD_FACTOR);

    auto keys = generateKeysCPU<uint64_t>(n, 42, 0, UINT32_MAX);

    cuckoofilter::CuckooFilter<uint64_t, Config::bitsPerTag> filter(capacity);
    for (const auto& k : keys) {
        filter.Add(k);
    }

    auto neverInserted =
        generateKeysCPU<uint64_t>(FPR_TEST_SIZE, 99999, uint64_t(UINT32_MAX) + 1, UINT64_MAX);

    size_t falsePositives = 0;
    for (auto _ : state) {
        timer.start();
        falsePositives = 0;
        for (const auto& k : neverInserted) {
            if (filter.Contain(k) == cuckoofilter::Ok) {
                ++falsePositives;
            }
        }
        double elapsed = timer.elapsed();

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(falsePositives);
    }

    double fpr = static_cast<double>(falsePositives) / static_cast<double>(FPR_TEST_SIZE);
    size_t filterMemory = filter.SizeInBytes();

    setFPRCounters(state, filterMemory, n, fpr, falsePositives, FPR_TEST_SIZE);
}

static void BBF_FPR(bm::State& state) {
    GPUTimer timer;
    size_t targetMemory = state.range(0);

    size_t numBlocks =
        targetMemory / (BloomFilter::words_per_block * sizeof(typename BloomFilter::word_type));
    if (numBlocks == 0)
        numBlocks = 1;

    size_t capacity =
        (numBlocks * BloomFilter::words_per_block * sizeof(typename BloomFilter::word_type) * 8) /
        Config::bitsPerTag;
    auto n = static_cast<size_t>(capacity * LOAD_FACTOR);

    thrust::device_vector<uint64_t> d_keys(n);
    generateKeysGPURange(d_keys, n, uint64_t(0), uint64_t(UINT32_MAX));

    auto filter = std::make_unique<BloomFilter>(numBlocks);
    size_t filterMemory = filter->block_extent() * BloomFilter::words_per_block *
                          sizeof(typename BloomFilter::word_type);

    thrust::device_vector<uint8_t> d_output(n);
    filter->add(d_keys.begin(), d_keys.end());

    thrust::device_vector<uint64_t> d_neverInserted(FPR_TEST_SIZE);
    thrust::device_vector<uint8_t> d_output_fpr(FPR_TEST_SIZE);

    generateKeysGPURange(d_neverInserted, FPR_TEST_SIZE, uint64_t(UINT32_MAX) + 1, UINT64_MAX);

    for (auto _ : state) {
        timer.start();
        filter->contains(
            d_neverInserted.begin(),
            d_neverInserted.end(),
            reinterpret_cast<bool*>(thrust::raw_pointer_cast(d_output_fpr.data()))
        );
        double elapsed = timer.elapsed();

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(d_output_fpr.data().get());
    }

    size_t falsePositives =
        thrust::reduce(d_output_fpr.begin(), d_output_fpr.end(), 0ULL, cuda::std::plus<size_t>());
    double fpr = static_cast<double>(falsePositives) / static_cast<double>(FPR_TEST_SIZE);

    setFPRCounters(state, filterMemory, n, fpr, falsePositives, FPR_TEST_SIZE);
}

static void TCF_FPR(bm::State& state) {
    GPUTimer timer;
    size_t targetMemory = state.range(0);

    size_t capacity = targetMemory / sizeof(uint16_t);
    auto n = static_cast<size_t>(capacity * LOAD_FACTOR);

    thrust::device_vector<uint64_t> d_keys(n);
    generateKeysGPURange(d_keys, n, uint64_t(0), uint64_t(UINT32_MAX));

    TCFType* filter = TCFType::host_build_tcf(capacity);
    size_t filterMemory = capacity * sizeof(uint16_t);

    uint64_t* d_misses;
    cudaMalloc(&d_misses, sizeof(uint64_t));
    cudaMemset(d_misses, 0, sizeof(uint64_t));
    filter->bulk_insert(thrust::raw_pointer_cast(d_keys.data()), n, d_misses);

    thrust::device_vector<uint64_t> d_neverInserted(FPR_TEST_SIZE);
    generateKeysGPURange(d_neverInserted, FPR_TEST_SIZE, uint64_t(UINT32_MAX) + 1, UINT64_MAX);

    bool* d_output = nullptr;
    for (auto _ : state) {
        timer.start();
        d_output =
            filter->bulk_query(thrust::raw_pointer_cast(d_neverInserted.data()), FPR_TEST_SIZE);
        double elapsed = timer.elapsed();

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(d_output);
        cudaFree(d_output);
    }

    // Perform one last query to get results for FPR calculation
    d_output = filter->bulk_query(thrust::raw_pointer_cast(d_neverInserted.data()), FPR_TEST_SIZE);

    thrust::device_ptr<bool> d_outputPtr(d_output);
    size_t falsePositives =
        thrust::reduce(d_outputPtr, d_outputPtr + FPR_TEST_SIZE, 0ULL, cuda::std::plus<size_t>());
    double fpr = static_cast<double>(falsePositives) / static_cast<double>(FPR_TEST_SIZE);

    setFPRCounters(state, filterMemory, n, fpr, falsePositives, FPR_TEST_SIZE);

    cudaFree(d_output);
    cudaFree(d_misses);
    TCFType::host_free_tcf(filter);
}

#ifdef __x86_64__
static void PCF_FPR(bm::State& state) {
    CPUTimer timer;
    size_t targetMemory = state.range(0);

    size_t capacity = (targetMemory * 8) / Config::bitsPerTag;
    auto n = static_cast<size_t>(capacity * LOAD_FACTOR);

    auto keys = generateKeysCPU<uint64_t>(n, 42, 0, UINT32_MAX);

    auto s = static_cast<size_t>(100.0 / LOAD_FACTOR);
    size_t expectedSize = (capacity * Config::bitsPerTag * (static_cast<double>(s) / 100)) / 8.0;
    size_t n_partitions = 1;

    if (expectedSize > L2_CACHE_SIZE) {
        size_t partitionsNeeded = SDIV(expectedSize, L2_CACHE_SIZE);
        while (n_partitions < partitionsNeeded) {
            n_partitions *= 2;
        }
    }

    size_t n_threads = std::min(n_partitions, size_t(std::thread::hardware_concurrency()));
    size_t n_tasks = 1;

    PartitionedCuckooFilter filter(s, n_partitions, n_threads, n_tasks);
    filter.construct(keys.data(), keys.size());

    auto neverInserted =
        generateKeysCPU<uint64_t>(FPR_TEST_SIZE, 99999, uint64_t(UINT32_MAX) + 1, UINT64_MAX);

    size_t falsePositives = 0;
    for (auto _ : state) {
        timer.start();
        falsePositives = 0;
        for (const auto& k : neverInserted) {
            if (filter.contains(k)) {
                ++falsePositives;
            }
        }
        double elapsed = timer.elapsed();

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(falsePositives);
    }

    double fpr = static_cast<double>(falsePositives) / static_cast<double>(FPR_TEST_SIZE);
    size_t filterMemory = filter.size();

    setFPRCounters(state, filterMemory, n, fpr, falsePositives, FPR_TEST_SIZE);
}
#endif  // __x86_64__

#define FPR_CONFIG                   \
    ->RangeMultiplier(2)             \
        ->Range(1 << 15, 1ULL << 28) \
        ->Unit(bm::kMillisecond)     \
        ->UseManualTime()            \
        ->Iterations(10)             \
        ->Repetitions(5)             \
        ->ReportAggregatesOnly(true);

static void GQF_FPR(bm::State& state) {
    GPUTimer timer;
    size_t targetMemory = state.range(0);

    // Estimate capacity based on target memory and bits per slot
    // GQF uses QF_BITS_PER_SLOT (16) + metadata overhead
    // We'll approximate capacity
    size_t capacity = (targetMemory * 8) / QF_BITS_PER_SLOT;
    auto n = static_cast<size_t>(capacity * LOAD_FACTOR);

    auto q = static_cast<uint32_t>(std::log2(capacity));
    capacity = 1ULL << q;

    thrust::device_vector<uint64_t> d_keys(n);
    generateKeysGPURange(d_keys, n, static_cast<uint64_t>(0), static_cast<uint64_t>(UINT32_MAX));

    QF* qf;
    qf_malloc_device(&qf, q, true);
    bulk_insert(qf, n, thrust::raw_pointer_cast(d_keys.data()), 0);
    cudaDeviceSynchronize();

    size_t filterMemory = getQFSizeHost(qf);

    thrust::device_vector<uint64_t> d_neverInserted(FPR_TEST_SIZE);
    thrust::device_vector<uint64_t> d_results(FPR_TEST_SIZE);

    generateKeysGPURange(
        d_neverInserted, FPR_TEST_SIZE, static_cast<uint64_t>(UINT32_MAX) + 1, UINT64_MAX
    );

    for (auto _ : state) {
        timer.start();
        bulk_get(
            qf,
            FPR_TEST_SIZE,
            thrust::raw_pointer_cast(d_neverInserted.data()),
            thrust::raw_pointer_cast(d_results.data())
        );
        double elapsed = timer.elapsed();

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(d_results.data().get());
    }

    convertGQFResults(d_results);

    size_t falsePositives =
        thrust::reduce(d_results.begin(), d_results.end(), 0ULL, thrust::plus<size_t>());

    double fpr = static_cast<double>(falsePositives) / static_cast<double>(FPR_TEST_SIZE);

    setFPRCounters(state, filterMemory, n, fpr, falsePositives, FPR_TEST_SIZE);

    qf_destroy_device(qf);
}

BENCHMARK(GCF_FPR) FPR_CONFIG;
BENCHMARK(CCF_FPR) FPR_CONFIG;
BENCHMARK(BBF_FPR) FPR_CONFIG;
BENCHMARK(TCF_FPR) FPR_CONFIG;

#ifdef __x86_64__
BENCHMARK(PCF_FPR) FPR_CONFIG;
#endif

BENCHMARK(GQF_FPR) FPR_CONFIG;

BENCHMARK_MAIN();
