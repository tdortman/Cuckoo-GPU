#include <benchmark/benchmark.h>
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <cstdint>
#include <CuckooFilter.cuh>
#include <helpers.cuh>
#include "benchmark_common.cuh"

namespace bm = benchmark;

constexpr double TARGET_LOAD_FACTOR = 0.95;

template <size_t bucketSize>
static void CF_Insert(bm::State& state) {
    using Config = CuckooConfig<uint32_t, 16, 500, 128, bucketSize>;
    auto [capacity, n] = calculateCapacityAndSize<Config>(state.range(0), TARGET_LOAD_FACTOR);

    thrust::device_vector<uint32_t> d_keys(n);
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
    state.counters["bucket_size"] = bm::Counter(bucketSize);
}

template <size_t bucketSize>
static void CF_Query(bm::State& state) {
    using Config = CuckooConfig<uint32_t, 16, 500, 128, bucketSize>;
    auto [capacity, n] = calculateCapacityAndSize<Config>(state.range(0), TARGET_LOAD_FACTOR);

    thrust::device_vector<uint32_t> d_keys(n);
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
    state.counters["bucket_size"] = bm::Counter(bucketSize);
}

template <size_t bucketSize>
static void CF_Delete(bm::State& state) {
    using Config = CuckooConfig<uint32_t, 16, 500, 128, bucketSize>;
    auto [capacity, n] = calculateCapacityAndSize<Config>(state.range(0), TARGET_LOAD_FACTOR);

    thrust::device_vector<uint32_t> d_keys(n);
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
    state.counters["bucket_size"] = bm::Counter(bucketSize);
}

template <size_t bucketSize>
static void CF_InsertAndQuery(bm::State& state) {
    using Config = CuckooConfig<uint32_t, 16, 500, 128, bucketSize>;
    auto [capacity, n] = calculateCapacityAndSize<Config>(state.range(0), TARGET_LOAD_FACTOR);

    thrust::device_vector<uint32_t> d_keys(n);
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
    state.counters["bucket_size"] = bm::Counter(bucketSize);
}

template <size_t bucketSize>
static void CF_InsertQueryDelete(bm::State& state) {
    using Config = CuckooConfig<uint32_t, 16, 500, 128, bucketSize>;
    auto [capacity, n] = calculateCapacityAndSize<Config>(state.range(0), TARGET_LOAD_FACTOR);

    thrust::device_vector<uint32_t> d_keys(n);
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
    state.counters["bucket_size"] = bm::Counter(bucketSize);
}

template <size_t bucketSize>
static void CF_FalsePositiveRate(bm::State& state) {
    using FPRConfig = CuckooConfig<uint64_t, 16, 500, 128, bucketSize>;
    auto [capacity, n] = calculateCapacityAndSize<FPRConfig>(state.range(0), TARGET_LOAD_FACTOR);

    thrust::device_vector<uint64_t> d_keys(n);
    generateKeysGPU<uint64_t>(d_keys, UINT32_MAX);

    CuckooFilter<FPRConfig> filter(capacity);
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
            thrust::uniform_int_distribution<uint64_t> dist(UINT32_MAX + 1, UINT64_MAX);
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
    state.counters["bucket_size"] = bm::Counter(bucketSize);
}

BENCHMARK(CF_Insert<4>)->RangeMultiplier(2)->Range(1 << 16, 1ULL << 28)->Unit(bm::kMillisecond);
BENCHMARK(CF_Insert<8>)->RangeMultiplier(2)->Range(1 << 16, 1ULL << 28)->Unit(bm::kMillisecond);
BENCHMARK(CF_Insert<16>)->RangeMultiplier(2)->Range(1 << 16, 1ULL << 28)->Unit(bm::kMillisecond);
BENCHMARK(CF_Insert<32>)->RangeMultiplier(2)->Range(1 << 16, 1ULL << 28)->Unit(bm::kMillisecond);
BENCHMARK(CF_Insert<64>)->RangeMultiplier(2)->Range(1 << 16, 1ULL << 28)->Unit(bm::kMillisecond);
BENCHMARK(CF_Insert<128>)->RangeMultiplier(2)->Range(1 << 16, 1ULL << 28)->Unit(bm::kMillisecond);

BENCHMARK(CF_Query<4>)->RangeMultiplier(2)->Range(1 << 16, 1ULL << 28)->Unit(bm::kMillisecond);
BENCHMARK(CF_Query<8>)->RangeMultiplier(2)->Range(1 << 16, 1ULL << 28)->Unit(bm::kMillisecond);
BENCHMARK(CF_Query<16>)->RangeMultiplier(2)->Range(1 << 16, 1ULL << 28)->Unit(bm::kMillisecond);
BENCHMARK(CF_Query<32>)->RangeMultiplier(2)->Range(1 << 16, 1ULL << 28)->Unit(bm::kMillisecond);
BENCHMARK(CF_Query<64>)->RangeMultiplier(2)->Range(1 << 16, 1ULL << 28)->Unit(bm::kMillisecond);
BENCHMARK(CF_Query<128>)->RangeMultiplier(2)->Range(1 << 16, 1ULL << 28)->Unit(bm::kMillisecond);

BENCHMARK(CF_Delete<4>)->RangeMultiplier(2)->Range(1 << 16, 1ULL << 28)->Unit(bm::kMillisecond);
BENCHMARK(CF_Delete<8>)->RangeMultiplier(2)->Range(1 << 16, 1ULL << 28)->Unit(bm::kMillisecond);
BENCHMARK(CF_Delete<16>)->RangeMultiplier(2)->Range(1 << 16, 1ULL << 28)->Unit(bm::kMillisecond);
BENCHMARK(CF_Delete<32>)->RangeMultiplier(2)->Range(1 << 16, 1ULL << 28)->Unit(bm::kMillisecond);
BENCHMARK(CF_Delete<64>)->RangeMultiplier(2)->Range(1 << 16, 1ULL << 28)->Unit(bm::kMillisecond);
BENCHMARK(CF_Delete<128>)->RangeMultiplier(2)->Range(1 << 16, 1ULL << 28)->Unit(bm::kMillisecond);

BENCHMARK(CF_InsertAndQuery<4>)
    ->RangeMultiplier(2)
    ->Range(1 << 16, 1ULL << 28)
    ->Unit(bm::kMillisecond);
BENCHMARK(CF_InsertAndQuery<8>)
    ->RangeMultiplier(2)
    ->Range(1 << 16, 1ULL << 28)
    ->Unit(bm::kMillisecond);
BENCHMARK(CF_InsertAndQuery<16>)
    ->RangeMultiplier(2)
    ->Range(1 << 16, 1ULL << 28)
    ->Unit(bm::kMillisecond);
BENCHMARK(CF_InsertAndQuery<32>)
    ->RangeMultiplier(2)
    ->Range(1 << 16, 1ULL << 28)
    ->Unit(bm::kMillisecond);
BENCHMARK(CF_InsertAndQuery<64>)
    ->RangeMultiplier(2)
    ->Range(1 << 16, 1ULL << 28)
    ->Unit(bm::kMillisecond);
BENCHMARK(CF_InsertAndQuery<128>)
    ->RangeMultiplier(2)
    ->Range(1 << 16, 1ULL << 28)
    ->Unit(bm::kMillisecond);

BENCHMARK(CF_InsertQueryDelete<4>)
    ->RangeMultiplier(2)
    ->Range(1 << 16, 1ULL << 28)
    ->Unit(bm::kMillisecond);
BENCHMARK(CF_InsertQueryDelete<8>)
    ->RangeMultiplier(2)
    ->Range(1 << 16, 1ULL << 28)
    ->Unit(bm::kMillisecond);
BENCHMARK(CF_InsertQueryDelete<16>)
    ->RangeMultiplier(2)
    ->Range(1 << 16, 1ULL << 28)
    ->Unit(bm::kMillisecond);
BENCHMARK(CF_InsertQueryDelete<32>)
    ->RangeMultiplier(2)
    ->Range(1 << 16, 1ULL << 28)
    ->Unit(bm::kMillisecond);
BENCHMARK(CF_InsertQueryDelete<64>)
    ->RangeMultiplier(2)
    ->Range(1 << 16, 1ULL << 28)
    ->Unit(bm::kMillisecond);
BENCHMARK(CF_InsertQueryDelete<128>)
    ->RangeMultiplier(2)
    ->Range(1 << 16, 1ULL << 28)
    ->Unit(bm::kMillisecond);

BENCHMARK(CF_FalsePositiveRate<4>)
    ->RangeMultiplier(2)
    ->Range(1 << 16, 1ULL << 28)
    ->Unit(bm::kMillisecond);
BENCHMARK(CF_FalsePositiveRate<8>)
    ->RangeMultiplier(2)
    ->Range(1 << 16, 1ULL << 28)
    ->Unit(bm::kMillisecond);
BENCHMARK(CF_FalsePositiveRate<16>)
    ->RangeMultiplier(2)
    ->Range(1 << 16, 1ULL << 28)
    ->Unit(bm::kMillisecond);
BENCHMARK(CF_FalsePositiveRate<32>)
    ->RangeMultiplier(2)
    ->Range(1 << 16, 1ULL << 28)
    ->Unit(bm::kMillisecond);
BENCHMARK(CF_FalsePositiveRate<64>)
    ->RangeMultiplier(2)
    ->Range(1 << 16, 1ULL << 28)
    ->Unit(bm::kMillisecond);
BENCHMARK(CF_FalsePositiveRate<128>)
    ->RangeMultiplier(2)
    ->Range(1 << 16, 1ULL << 28)
    ->Unit(bm::kMillisecond);

BENCHMARK_MAIN();