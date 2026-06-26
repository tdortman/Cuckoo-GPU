// Compares the default cuCo Bloom filter policy against a paper-recommended
// parametric policy (arXiv:2512.15595): 4x uint64 words per 256-bit block,
// k=4 pattern bits, fully-horizontal add (Theta=4) and fully-vertical
// contains (Phi=4), with conditional_add for ~95% load factor.

#include <benchmark/benchmark.h>
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#include <cstdint>
#include <cuckoogpu/helpers.cuh>
#include <cuco/bloom_filter.cuh>
#include <cuco/bloom_filter_policies.cuh>
#include <cuda/atomic>
#include <cuda/std/cstdint>
#include <memory>
#include "benchmark_common.cuh"

namespace bm = benchmark;

// Default policy: cuco::bloom_filter<uint64_t> (8x uint32 words, k=8).
using DefaultBloom = cuco::bloom_filter<uint64_t>;

// Parametric policy: 4x uint64 words per 256-bit block, k=4, conditional_add.
using ParametricPolicy = cuco::parametric_filter_policy<
    cuco::xxhash_64<uint64_t>,  // 64-bit hash (paper default)
    uint64_t,                   // word type, S=64
    4,                          // words per block (B=256 / S=64)
    4,                          // pattern bits, k=4
    4,                          // add: fully horizontal (Theta=4, Phi=1)
    1,                          // add: fully horizontal (Theta=4, Phi=1)
    1,                          // contains: fully vertical (Theta=1, Phi=4)
    4,                          // contains: fully vertical (Theta=1, Phi=4)
    true,                       // conditional_add (we target 95% LF)
    false>;                     // early_exit_contains

using ParametricBloom = cuco::
    bloom_filter<uint64_t, cuco::extent<std::size_t>, cuda::thread_scope_device, ParametricPolicy>;

template <typename BloomFilter, double loadFactor = 0.95>
class CucoBloomFixtureBase : public benchmark::Fixture {
    using benchmark::Fixture::SetUp;
    using benchmark::Fixture::TearDown;

   public:
    void SetUp(const benchmark::State& state) override {
        auto [cap, num] = calculateCapacityAndSize(state.range(0), loadFactor);
        capacity = cap;
        n = num;

        const size_t numBlocks = cucoNumBlocks<BloomFilter, 16>(capacity);
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

    size_t capacity{};
    size_t n{};
    size_t filterMemory{};
    thrust::device_vector<uint64_t> d_keys;
    thrust::device_vector<uint8_t> d_output;
    std::unique_ptr<BloomFilter> filter;
    GPUTimer timer;
};

class CucoDefaultFixture : public CucoBloomFixtureBase<DefaultBloom> {};
class CucoParametricFixture : public CucoBloomFixtureBase<ParametricBloom> {};

template <typename Fixture>
void bloomInsertBody(Fixture& f, bm::State& state) {
    for (auto _ : state) {
        f.filter->clear();
        cudaDeviceSynchronize();

        f.timer.start();
        f.filter->add(f.d_keys.begin(), f.d_keys.end());
        double elapsed = f.timer.elapsed();

        state.SetIterationTime(elapsed);
    }
    f.setCounters(state);
}

template <typename Fixture>
void bloomQueryBody(Fixture& f, bm::State& state) {
    f.filter->add(f.d_keys.begin(), f.d_keys.end());
    cudaDeviceSynchronize();

    for (auto _ : state) {
        f.timer.start();
        f.filter->contains(
            f.d_keys.begin(),
            f.d_keys.end(),
            reinterpret_cast<bool*>(thrust::raw_pointer_cast(f.d_output.data()))
        );
        double elapsed = f.timer.elapsed();

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(f.d_output.data().get());
    }
    f.setCounters(state);
}

template <typename Fixture>
void bloomInsertAndQueryBody(Fixture& f, bm::State& state) {
    for (auto _ : state) {
        f.filter->clear();
        cudaDeviceSynchronize();

        f.timer.start();
        f.filter->add(f.d_keys.begin(), f.d_keys.end());
        f.filter->contains(
            f.d_keys.begin(),
            f.d_keys.end(),
            reinterpret_cast<bool*>(thrust::raw_pointer_cast(f.d_output.data()))
        );
        double elapsed = f.timer.elapsed();

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(f.d_output.data().get());
    }
    f.setCounters(state);
}

BENCHMARK_DEFINE_F(CucoDefaultFixture, Insert)(bm::State& state) {
    bloomInsertBody(*this, state);
}
BENCHMARK_DEFINE_F(CucoDefaultFixture, Query)(bm::State& state) {
    bloomQueryBody(*this, state);
}
BENCHMARK_DEFINE_F(CucoDefaultFixture, InsertAndQuery)(bm::State& state) {
    bloomInsertAndQueryBody(*this, state);
}
REGISTER_BENCHMARK(CucoDefaultFixture, Insert);
REGISTER_BENCHMARK(CucoDefaultFixture, Query);
REGISTER_BENCHMARK(CucoDefaultFixture, InsertAndQuery);

BENCHMARK_DEFINE_F(CucoParametricFixture, Insert)(bm::State& state) {
    bloomInsertBody(*this, state);
}
BENCHMARK_DEFINE_F(CucoParametricFixture, Query)(bm::State& state) {
    bloomQueryBody(*this, state);
}
BENCHMARK_DEFINE_F(CucoParametricFixture, InsertAndQuery)(bm::State& state) {
    bloomInsertAndQueryBody(*this, state);
}
REGISTER_BENCHMARK(CucoParametricFixture, Insert);
REGISTER_BENCHMARK(CucoParametricFixture, Query);
REGISTER_BENCHMARK(CucoParametricFixture, InsertAndQuery);

STANDARD_BENCHMARK_MAIN();
