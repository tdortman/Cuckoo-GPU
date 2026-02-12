#include <benchmark/benchmark.h>
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cstdint>
#include <cuckoogpu/CuckooFilter.cuh>
#include <cuckoogpu/helpers.cuh>
#include "benchmark_common.cuh"

namespace bm = benchmark;

using Config64 = cuckoogpu::Config<
    uint64_t,
    8,
    500,
    256,
    16,
    cuckoogpu::XorAltBucketPolicy,
    cuckoogpu::EvictionPolicy::DFS,
    uint64_t>;
using Config32 = cuckoogpu::Config<
    uint64_t,
    8,
    500,
    256,
    16,
    cuckoogpu::XorAltBucketPolicy,
    cuckoogpu::EvictionPolicy::DFS,
    uint32_t>;

using Fixture64 = CuckooFilterFixture<Config64>;
DEFINE_AND_REGISTER_CORE_BENCHMARKS(Fixture64)

using Fixture32 = CuckooFilterFixture<Config32>;
DEFINE_AND_REGISTER_CORE_BENCHMARKS(Fixture32)

BENCHMARK_MAIN();
