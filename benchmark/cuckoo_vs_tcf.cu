#include <benchmark/benchmark.h>
#include <cuda_runtime_api.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <bulk_tcf_host.cuh>
#include <cstdint>
#include <CuckooFilter.cuh>
#include <cuda/std/cstdint>
#include <hash_strategies.cuh>
#include <helpers.cuh>
#include <random>
#include "benchmark_common.cuh"

namespace bm = benchmark;

constexpr double TARGET_LOAD_FACTOR = 0.85;
constexpr double TCF_LOAD_FACTOR = 0.85;
using Config = CuckooConfig<uint64_t, 16, 500, 128, 16, XorAltBucketPolicy>;
using TCFType = host_bulk_tcf<uint64_t, uint16_t>;

static void CF_Insert(bm::State& state) {
    auto [capacity, n] = calculateCapacityAndSize<Config>(state.range(0), TARGET_LOAD_FACTOR);

    thrust::device_vector<uint64_t> d_keys(n);
    generateKeysGPU(d_keys);
    CuckooFilter<Config> filter(capacity);

    size_t filterMemory = filter.sizeInBytes();

    for (auto _ : state) {
        state.PauseTiming();
        filter.clear();
        state.ResumeTiming();

        size_t inserted = adaptiveInsert(filter, d_keys);
        cudaDeviceSynchronize();
        bm::DoNotOptimize(inserted);
    }

    setCommonCounters(state, filterMemory, n);
}

static void CF_Query(bm::State& state) {
    auto [capacity, n] = calculateCapacityAndSize<Config>(state.range(0), TARGET_LOAD_FACTOR);

    thrust::device_vector<uint64_t> d_keys(n);
    generateKeysGPU(d_keys);
    CuckooFilter<Config> filter(capacity);
    thrust::device_vector<uint8_t> d_output(n);

    adaptiveInsert(filter, d_keys);

    size_t filterMemory = filter.sizeInBytes();

    for (auto _ : state) {
        filter.containsMany(d_keys, d_output);
        cudaDeviceSynchronize();
        bm::DoNotOptimize(d_output.data().get());
    }

    setCommonCounters(state, filterMemory, n);
}

static void CF_Delete(bm::State& state) {
    auto [capacity, n] = calculateCapacityAndSize<Config>(state.range(0), TARGET_LOAD_FACTOR);

    thrust::device_vector<uint64_t> d_keys(n);
    generateKeysGPU(d_keys);
    CuckooFilter<Config> filter(capacity);
    thrust::device_vector<uint8_t> d_output(n);

    size_t filterMemory = filter.sizeInBytes();

    for (auto _ : state) {
        state.PauseTiming();
        filter.clear();
        adaptiveInsert(filter, d_keys);
        state.ResumeTiming();

        size_t remaining = filter.deleteMany(d_keys, d_output);
        cudaDeviceSynchronize();
        bm::DoNotOptimize(remaining);
        bm::DoNotOptimize(d_output.data().get());
    }

    setCommonCounters(state, filterMemory, n);
}

static void CF_InsertAndQuery(bm::State& state) {
    auto [capacity, n] = calculateCapacityAndSize<Config>(state.range(0), TARGET_LOAD_FACTOR);

    thrust::device_vector<uint64_t> d_keys(n);
    generateKeysGPU(d_keys);
    thrust::device_vector<uint8_t> d_output(n);
    CuckooFilter<Config> filter(capacity);

    size_t filterMemory = filter.sizeInBytes();

    for (auto _ : state) {
        state.PauseTiming();
        filter.clear();
        state.ResumeTiming();

        size_t inserted = adaptiveInsert(filter, d_keys);
        filter.containsMany(d_keys, d_output);

        cudaDeviceSynchronize();

        bm::DoNotOptimize(inserted);
        bm::DoNotOptimize(d_output.data().get());
    }

    setCommonCounters(state, filterMemory, n);
}

static void CF_InsertQueryDelete(bm::State& state) {
    auto [capacity, n] = calculateCapacityAndSize<Config>(state.range(0), TARGET_LOAD_FACTOR);

    thrust::device_vector<uint64_t> d_keys(n);
    generateKeysGPU(d_keys);
    thrust::device_vector<uint8_t> d_output(n);
    CuckooFilter<Config> filter(capacity);

    size_t filterMemory = filter.sizeInBytes();

    for (auto _ : state) {
        state.PauseTiming();
        filter.clear();
        state.ResumeTiming();

        size_t inserted = adaptiveInsert(filter, d_keys);
        filter.containsMany(d_keys, d_output);
        size_t remaining = filter.deleteMany(d_keys, d_output);

        cudaDeviceSynchronize();

        bm::DoNotOptimize(inserted);
        bm::DoNotOptimize(remaining);
        bm::DoNotOptimize(d_output.data().get());
    }

    setCommonCounters(state, filterMemory, n);
}

static void CF_FPR(bm::State& state) {
    auto [capacity, n] = calculateCapacityAndSize<Config>(state.range(0), TARGET_LOAD_FACTOR);

    thrust::device_vector<uint64_t> d_keys(n);
    generateKeysGPU<uint64_t>(d_keys, UINT32_MAX);

    CuckooFilter<Config> filter(capacity);
    adaptiveInsert(filter, d_keys);

    size_t fprTestSize = std::min(n, size_t(1'000'000));
    thrust::device_vector<uint64_t> d_neverInserted(fprTestSize);
    thrust::device_vector<uint8_t> d_output(fprTestSize);

    thrust::transform(
        thrust::counting_iterator<size_t>(0),
        thrust::counting_iterator<size_t>(fprTestSize),
        d_neverInserted.begin(),
        [] __device__(size_t idx) {
            thrust::default_random_engine rng(99999);
            thrust::uniform_int_distribution<uint64_t> dist(
                static_cast<uint64_t>(UINT32_MAX) + 1, UINT64_MAX
            );
            rng.discard(idx);
            return dist(rng);
        }
    );

    size_t filterMemory = filter.sizeInBytes();

    for (auto _ : state) {
        filter.containsMany(d_neverInserted, d_output);
        cudaDeviceSynchronize();
        bm::DoNotOptimize(d_output.data().get());
    }

    size_t falsePositives =
        thrust::reduce(d_output.begin(), d_output.end(), 0ULL, cuda::std::plus<size_t>());
    double fpr = static_cast<double>(falsePositives) / static_cast<double>(fprTestSize);

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * fprTestSize));
    state.counters["fpr_percentage"] = bm::Counter(fpr * 100);
    state.counters["false_positives"] = bm::Counter(static_cast<double>(falsePositives));
    state.counters["bits_per_item"] = bm::Counter(
        static_cast<double>(filterMemory * 8) / static_cast<double>(n),
        bm::Counter::kDefaults,
        bm::Counter::kIs1024
    );
    state.counters["memory_bytes"] = bm::Counter(
        static_cast<double>(filterMemory), bm::Counter::kDefaults, bm::Counter::kIs1024
    );
}

static void TCF_Insert(bm::State& state) {
    auto [capacity, n] = calculateCapacityAndSize<Config>(state.range(0), TCF_LOAD_FACTOR);

    thrust::device_vector<uint64_t> d_keys(n);
    generateKeysGPU(d_keys);

    uint64_t* d_misses;
    cudaMalloc(&d_misses, sizeof(uint64_t));

    size_t filterMemory = capacity * sizeof(uint16_t);

    for (auto _ : state) {
        state.PauseTiming();
        TCFType* filter = TCFType::host_build_tcf(capacity);
        cudaMemset(d_misses, 0, sizeof(uint64_t));
        state.ResumeTiming();

        filter->bulk_insert(thrust::raw_pointer_cast(d_keys.data()), n, d_misses);
        cudaDeviceSynchronize();

        state.PauseTiming();
        TCFType::host_free_tcf(filter);
        state.ResumeTiming();
    }

    cudaFree(d_misses);
    setCommonCounters(state, filterMemory, n);
}

static void TCF_Query(bm::State& state) {
    auto [capacity, n] = calculateCapacityAndSize<Config>(state.range(0), TCF_LOAD_FACTOR);

    thrust::device_vector<uint64_t> d_keys(n);
    generateKeysGPU(d_keys);

    TCFType* filter = TCFType::host_build_tcf(capacity);
    uint64_t* d_misses;
    cudaMalloc(&d_misses, sizeof(uint64_t));
    cudaMemset(d_misses, 0, sizeof(uint64_t));

    filter->bulk_insert(thrust::raw_pointer_cast(d_keys.data()), n, d_misses);

    size_t filterMemory = capacity * sizeof(uint16_t);

    for (auto _ : state) {
        bool* d_output = filter->bulk_query(thrust::raw_pointer_cast(d_keys.data()), n);
        cudaDeviceSynchronize();
        bm::DoNotOptimize(d_output);
        cudaFree(d_output);
    }

    cudaFree(d_misses);
    TCFType::host_free_tcf(filter);
    setCommonCounters(state, filterMemory, n);
}

static void TCF_Delete(bm::State& state) {
    auto [capacity, n] = calculateCapacityAndSize<Config>(state.range(0), TCF_LOAD_FACTOR);

    thrust::device_vector<uint64_t> d_keys(n);
    generateKeysGPU(d_keys);

    uint64_t* d_misses;
    cudaMalloc(&d_misses, sizeof(uint64_t));

    size_t filterMemory = capacity * sizeof(uint16_t);

    for (auto _ : state) {
        state.PauseTiming();
        TCFType* filter = TCFType::host_build_tcf(capacity);
        cudaMemset(d_misses, 0, sizeof(uint64_t));
        filter->bulk_insert(thrust::raw_pointer_cast(d_keys.data()), n, d_misses);
        state.ResumeTiming();

        bool* d_output = filter->bulk_delete(thrust::raw_pointer_cast(d_keys.data()), n);
        cudaDeviceSynchronize();

        bm::DoNotOptimize(d_output);

        state.PauseTiming();
        cudaFree(d_output);
        TCFType::host_free_tcf(filter);
        state.ResumeTiming();
    }

    cudaFree(d_misses);
    setCommonCounters(state, filterMemory, n);
}

static void TCF_InsertAndQuery(bm::State& state) {
    auto [capacity, n] = calculateCapacityAndSize<Config>(state.range(0), TCF_LOAD_FACTOR);

    thrust::device_vector<uint64_t> d_keys(n);
    generateKeysGPU(d_keys);

    uint64_t* d_misses;
    cudaMalloc(&d_misses, sizeof(uint64_t));

    size_t filterMemory = capacity * sizeof(uint16_t);

    for (auto _ : state) {
        state.PauseTiming();
        TCFType* filter = TCFType::host_build_tcf(capacity);
        cudaMemset(d_misses, 0, sizeof(uint64_t));
        state.ResumeTiming();

        filter->bulk_insert(thrust::raw_pointer_cast(d_keys.data()), n, d_misses);
        bool* d_output = filter->bulk_query(thrust::raw_pointer_cast(d_keys.data()), n);

        cudaDeviceSynchronize();

        bm::DoNotOptimize(d_output);

        state.PauseTiming();
        cudaFree(d_output);
        TCFType::host_free_tcf(filter);
        state.ResumeTiming();
    }

    cudaFree(d_misses);
    setCommonCounters(state, filterMemory, n);
}

static void TCF_InsertQueryDelete(bm::State& state) {
    auto [capacity, n] = calculateCapacityAndSize<Config>(state.range(0), TCF_LOAD_FACTOR);

    thrust::device_vector<uint64_t> d_keys(n);
    generateKeysGPU(d_keys);

    uint64_t* d_misses;
    cudaMalloc(&d_misses, sizeof(uint64_t));

    size_t filterMemory = capacity * sizeof(uint16_t);

    for (auto _ : state) {
        state.PauseTiming();
        TCFType* filter = TCFType::host_build_tcf(capacity);
        cudaMemset(d_misses, 0, sizeof(uint64_t));
        state.ResumeTiming();

        filter->bulk_insert(thrust::raw_pointer_cast(d_keys.data()), n, d_misses);
        bool* d_query_output = filter->bulk_query(thrust::raw_pointer_cast(d_keys.data()), n);
        bool* d_delete_output = filter->bulk_delete(thrust::raw_pointer_cast(d_keys.data()), n);

        cudaDeviceSynchronize();

        bm::DoNotOptimize(d_query_output);
        bm::DoNotOptimize(d_delete_output);

        state.PauseTiming();
        cudaFree(d_query_output);
        cudaFree(d_delete_output);
        TCFType::host_free_tcf(filter);
        state.ResumeTiming();
    }

    cudaFree(d_misses);
    setCommonCounters(state, filterMemory, n);
}

static void TCF_FPR(bm::State& state) {
    auto [capacity, n] = calculateCapacityAndSize<Config>(state.range(0), TCF_LOAD_FACTOR);

    thrust::device_vector<uint64_t> d_keys(n);
    generateKeysGPU<uint64_t>(d_keys, UINT32_MAX);

    TCFType* filter = TCFType::host_build_tcf(capacity);
    uint64_t* d_misses;
    cudaMalloc(&d_misses, sizeof(uint64_t));
    cudaMemset(d_misses, 0, sizeof(uint64_t));

    filter->bulk_insert(thrust::raw_pointer_cast(d_keys.data()), n, d_misses);

    size_t fprTestSize = std::min(n, size_t(1'000'000));
    thrust::device_vector<uint64_t> d_neverInserted(fprTestSize);

    thrust::transform(
        thrust::counting_iterator<size_t>(0),
        thrust::counting_iterator<size_t>(fprTestSize),
        d_neverInserted.begin(),
        [] __device__(size_t idx) {
            thrust::default_random_engine rng(99999);
            thrust::uniform_int_distribution<uint64_t> dist(
                static_cast<uint64_t>(UINT32_MAX) + 1, UINT64_MAX
            );
            rng.discard(idx);
            return dist(rng);
        }
    );

    size_t filterMemory = capacity * sizeof(uint16_t);

    for (auto _ : state) {
        bool* d_output =
            filter->bulk_query(thrust::raw_pointer_cast(d_neverInserted.data()), fprTestSize);
        cudaDeviceSynchronize();

        state.PauseTiming();
        thrust::device_ptr<bool> d_output_ptr(d_output);
        size_t falsePositives = thrust::reduce(
            d_output_ptr, d_output_ptr + fprTestSize, 0ULL, cuda::std::plus<size_t>()
        );
        cudaFree(d_output);
        state.ResumeTiming();

        bm::DoNotOptimize(falsePositives);
    }

    // Calculate FPR for the counter
    bool* d_output =
        filter->bulk_query(thrust::raw_pointer_cast(d_neverInserted.data()), fprTestSize);
    cudaDeviceSynchronize();
    thrust::device_ptr<bool> d_output_ptr(d_output);
    size_t falsePositives =
        thrust::reduce(d_output_ptr, d_output_ptr + fprTestSize, 0ULL, cuda::std::plus<size_t>());
    cudaFree(d_output);

    double fpr = static_cast<double>(falsePositives) / static_cast<double>(fprTestSize);

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * fprTestSize));
    state.counters["fpr_percentage"] = bm::Counter(fpr * 100);
    state.counters["false_positives"] = bm::Counter(static_cast<double>(falsePositives));
    state.counters["bits_per_item"] = bm::Counter(
        static_cast<double>(filterMemory * 8) / static_cast<double>(n),
        bm::Counter::kDefaults,
        bm::Counter::kIs1024
    );
    state.counters["memory_bytes"] = bm::Counter(
        static_cast<double>(filterMemory), bm::Counter::kDefaults, bm::Counter::kIs1024
    );

    cudaFree(d_misses);
    TCFType::host_free_tcf(filter);
}

BENCHMARK(CF_Insert)->RangeMultiplier(2)->Range(1 << 16, 1ULL << 28)->Unit(bm::kMillisecond);
BENCHMARK(TCF_Insert)->RangeMultiplier(2)->Range(1 << 16, 1ULL << 28)->Unit(bm::kMillisecond);

BENCHMARK(CF_Query)->RangeMultiplier(2)->Range(1 << 16, 1ULL << 28)->Unit(bm::kMillisecond);
BENCHMARK(TCF_Query)->RangeMultiplier(2)->Range(1 << 16, 1ULL << 28)->Unit(bm::kMillisecond);

BENCHMARK(CF_Delete)->RangeMultiplier(2)->Range(1 << 16, 1ULL << 28)->Unit(bm::kMillisecond);
BENCHMARK(TCF_Delete)->RangeMultiplier(2)->Range(1 << 16, 1ULL << 28)->Unit(bm::kMillisecond);

BENCHMARK(CF_InsertAndQuery)
    ->RangeMultiplier(2)
    ->Range(1 << 16, 1ULL << 28)
    ->Unit(bm::kMillisecond);
BENCHMARK(TCF_InsertAndQuery)
    ->RangeMultiplier(2)
    ->Range(1 << 16, 1ULL << 28)
    ->Unit(bm::kMillisecond);

BENCHMARK(CF_InsertQueryDelete)
    ->RangeMultiplier(2)
    ->Range(1 << 16, 1ULL << 28)
    ->Unit(bm::kMillisecond);
BENCHMARK(TCF_InsertQueryDelete)
    ->RangeMultiplier(2)
    ->Range(1 << 16, 1ULL << 28)
    ->Unit(bm::kMillisecond);

BENCHMARK(CF_FPR)->RangeMultiplier(2)->Range(1 << 16, 1ULL << 28)->Unit(bm::kMillisecond);
BENCHMARK(TCF_FPR)->RangeMultiplier(2)->Range(1 << 16, 1ULL << 28)->Unit(bm::kMillisecond);

BENCHMARK_MAIN();
