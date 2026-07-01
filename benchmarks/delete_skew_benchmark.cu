// Enable delete CAS counting for the CAS cost breakdown.
#define CUCKOO_FILTER_COUNT_DELETE_CAS

#include <benchmark/benchmark.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>
#include <thrust/transform.h>
#include <cmath>
#include <cstdint>
#include <cuckoogpu/CuckooFilter.cuh>
#include <limits>
#include <numeric>
#include <vector>
#include "benchmark_common.cuh"

namespace bm = benchmark;

// ponytail: one config, one filter type. Add more if shepherding requires it.
using Config = cuckoogpu::Config<
    uint64_t,
    8,
    500,
    256,
    16,
    cuckoogpu::XorAltBucketPolicy,
    cuckoogpu::EvictionPolicy::BFS,
    uint64_t>;

using KeyType = Config::KeyType;

// Delete fractions to sweep. Each is the fraction of inserted keys deleted
// during the timed phase.
// ponytail: explicit array, no config parsing needed.
static constexpr double DELETE_FRACTIONS[] = {0.25, 0.50, 0.75, 0.90, 1.0};

// Values are percentages in CSV output. Finite values use N(N/2, sigma*N);
// infinity samples uniformly with replacement.
static constexpr double UNIFORM_SENTINEL = std::numeric_limits<double>::infinity();
static constexpr double STDDEV_FRACTIONS[] =
    {0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0, UNIFORM_SENTINEL};

// Prefill the filter to this load factor before each timed delete phase.
static constexpr double PREFILL_LOAD_FACTOR = 0.95;
static constexpr int MIN_CAPACITY_EXP = 16;
static constexpr int MAX_CAPACITY_EXP = 28;

static std::vector<int64_t> capacityRange() {
    std::vector<int64_t> capacities;
    capacities.reserve(MAX_CAPACITY_EXP - MIN_CAPACITY_EXP + 1);
    for (int exp = MIN_CAPACITY_EXP; exp <= MAX_CAPACITY_EXP; ++exp) {
        capacities.push_back(int64_t{1} << exp);
    }
    return capacities;
}

template <size_t N>
static std::vector<int64_t> indexRange(const double (&)[N]) {
    std::vector<int64_t> indices(N);
    std::iota(indices.begin(), indices.end(), int64_t{0});
    return indices;
}

/**
 * Generate a delete request set with controllable request distribution.
 *
 * finite stdevFraction: samples indices with replacement from N(mean=N/2,
 * stdev=stdevFraction*N), clamped to the inserted-key range.
 * infinite stdevFraction: samples uniformly with replacement.
 *
 * CAS counters and successful-delete counters separate atomic retry cost from misses.
 */
static void generateDeleteKeys(
    thrust::device_vector<KeyType>& d_deleteKeys,
    size_t numDeletes,
    const thrust::device_vector<KeyType>& d_insertedKeys,
    double stdevFraction,
    unsigned int seed = 42
) {
    size_t totalInserted = d_insertedKeys.size();
    double mean = static_cast<double>(totalInserted - 1) * 0.5;

    d_deleteKeys.resize(numDeletes);

    bool useUniform = std::isinf(stdevFraction);
    size_t centerIndex = totalInserted / 2;

    const KeyType* inserted = thrust::raw_pointer_cast(d_insertedKeys.data());
    thrust::transform(
        thrust::counting_iterator<size_t>(0),
        thrust::counting_iterator<size_t>(numDeletes),
        d_deleteKeys.begin(),
        [=] __device__(size_t idx) {
            thrust::default_random_engine rng(seed);
            rng.discard(idx);
            size_t selected = centerIndex;
            if (useUniform) {
                selected = thrust::uniform_int_distribution<size_t>(0, totalInserted - 1)(rng);
            } else {
                double sample =
                    thrust::normal_distribution<double>(mean, stdevFraction * totalInserted)(rng);
                auto clamped = lround(sample + 0.5);
                clamped = clamped < 0 ? 0 : clamped;
                clamped = clamped >= static_cast<long long>(totalInserted)
                              ? static_cast<long long>(totalInserted - 1)
                              : clamped;
                selected = static_cast<size_t>(clamped);
            }
            return inserted[selected];
        }
    );
}

/**
 * Benchmark: delete a fraction of inserted keys at a given stdev level.
 *
 * State ranges:
 *   range(0): filter capacity (number of slots)
 *   range(1): delete fraction index (into DELETE_FRACTIONS)
 *   range(2): normal stdev index (into STDDEV_FRACTIONS)
 */
static void DeleteSkew(bm::State& state) {
    const auto capacity = static_cast<size_t>(state.range(0));
    const double deleteFraction = DELETE_FRACTIONS[state.range(1)];
    const double stdevFraction = STDDEV_FRACTIONS[state.range(2)];

    const auto numInserted = static_cast<size_t>(capacity * PREFILL_LOAD_FACTOR);
    const auto numDeletes = static_cast<size_t>(numInserted * deleteFraction);

    cuckoogpu::Filter<Config> filter(capacity);
    const size_t filterMemory = filter.sizeInBytes();

    thrust::device_vector<KeyType> d_keys(numInserted);
    thrust::device_vector<uint8_t> d_output(numInserted);
    thrust::device_vector<KeyType> d_deleteKeys(numDeletes);
    thrust::device_vector<uint8_t> d_deleteOutput(numDeletes);

    generateKeysGPU(d_keys);

    // Prefill once. The keys we delete are drawn from this set.
    filter.insertMany(d_keys);
    cudaDeviceSynchronize();

    const size_t occupiedAfterPrefill = filter.occupiedSlots();

    size_t lastRemaining = occupiedAfterPrefill;

    GPUTimer timer;

    for (auto _ : state) {
        // Re-generate the delete set each iteration (stdev changes the contention pattern).
        generateDeleteKeys(d_deleteKeys, numDeletes, d_keys, stdevFraction);

        // Re-insert keys. clear() already resets CAS counters so the timed delete is isolated.
        filter.clear();
        filter.insertMany(d_keys);
        cudaDeviceSynchronize();

        timer.start();
        lastRemaining = filter.deleteMany(d_deleteKeys, d_deleteOutput);
        double elapsed = timer.elapsed();

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(lastRemaining);
        bm::DoNotOptimize(d_deleteOutput.data().get());
    }

    // Snapshot CAS counts from the last iteration (benchmarks report aggregates).
    const size_t casAttempts = filter.deleteCasAttempts();
    const size_t casFailures = filter.deleteCasFailures();
    const size_t successfulDeletes = occupiedAfterPrefill - lastRemaining;

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * numDeletes));
    state.counters["memory_bytes"] = bm::Counter(
        static_cast<double>(filterMemory), bm::Counter::kDefaults, bm::Counter::kIs1024
    );
    state.counters["bits_per_item"] = bm::Counter(
        static_cast<double>(filterMemory * 8) / static_cast<double>(numInserted),
        bm::Counter::kDefaults,
        bm::Counter::kIs1024
    );
    state.counters["inserted"] = bm::Counter(static_cast<double>(numInserted));
    state.counters["deletes"] = bm::Counter(static_cast<double>(numDeletes));
    state.counters["delete_fraction"] = bm::Counter(deleteFraction * 100);
    state.counters["stddev_fraction"] = bm::Counter(stdevFraction * 100);
    state.counters["prefill_load_factor"] = bm::Counter(PREFILL_LOAD_FACTOR * 100);
    state.counters["occupied_after_prefill"] =
        bm::Counter(static_cast<double>(occupiedAfterPrefill));
    state.counters["successful_deletes"] = bm::Counter(static_cast<double>(successfulDeletes));
    state.counters["successful_delete_fraction"] = bm::Counter(
        numDeletes > 0
            ? (static_cast<double>(successfulDeletes) / static_cast<double>(numDeletes)) * 100.0
            : 0.0
    );
    state.counters["delete_cas_attempts"] = bm::Counter(static_cast<double>(casAttempts));
    state.counters["delete_cas_failures"] = bm::Counter(static_cast<double>(casFailures));
    state.counters["delete_cas_fail_rate"] = bm::Counter(
        casAttempts > 0 ? static_cast<double>(casFailures) / static_cast<double>(casAttempts) : 0.0
    );
    state.counters["delete_cas_failure_rate"] = bm::Counter(
        casAttempts > 0 ? static_cast<double>(casFailures) / static_cast<double>(casAttempts) : 0.0
    );
    state.counters["cas_attempts_per_delete"] = bm::Counter(
        numDeletes > 0 ? static_cast<double>(casAttempts) / static_cast<double>(numDeletes) : 0.0
    );
    state.counters["delete_cas_attempts_per_key"] = bm::Counter(
        numDeletes > 0 ? static_cast<double>(casAttempts) / static_cast<double>(numDeletes) : 0.0
    );
    state.counters["cas_failures_per_delete"] = bm::Counter(
        numDeletes > 0 ? static_cast<double>(casFailures) / static_cast<double>(numDeletes) : 0.0
    );
}

// Register across capacities x delete fractions x normal stdev levels.
BENCHMARK(DeleteSkew)
    ->ArgsProduct({
        capacityRange(),
        indexRange(DELETE_FRACTIONS),
        indexRange(STDDEV_FRACTIONS),
    })
    ->Unit(bm::kMillisecond)
    ->UseManualTime()
    ->Iterations(10)
    ->Repetitions(5)
    ->ReportAggregatesOnly(true);

STANDARD_BENCHMARK_MAIN();
