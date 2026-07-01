// Enable delete CAS counting for the CAS cost breakdown.
#define CUCKOO_FILTER_COUNT_DELETE_CAS

#include <benchmark/benchmark.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>
#include <thrust/transform.h>
#include <cstdint>
#include <numeric>
#include <vector>
#include <cuckoogpu/CuckooFilter.cuh>
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

// Skew levels: probability that a delete request is sampled from the first
// 10% "hot" prefix. Skewed modes intentionally include repeated delete requests.
static constexpr double SKEW_LEVELS[] = {0.0, 0.5, 0.9};

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
 * Generate a delete request set with controllable skew.
 *
 * skew=0.0: uses distinct inserted keys, so the delete-heavy sweep measures valid deletes.
 * skew>0.0: requests are sampled with replacement from the hot prefix with the
 * configured probability, creating a separate hotspot stress case.
 *
 * CAS counters and successful-delete counters separate atomic retry cost from misses.
 */
static void generateDeleteKeys(
    thrust::device_vector<KeyType>& d_deleteKeys,
    size_t numDeletes,
    const thrust::device_vector<KeyType>& d_insertedKeys,
    double skew,
    unsigned int seed = 42
) {
    size_t totalInserted = d_insertedKeys.size();
    size_t hotCount = std::max(size_t(1), totalInserted / 10);

    d_deleteKeys.resize(numDeletes);

    if (skew == 0.0) {
        thrust::copy_n(d_insertedKeys.begin(), numDeletes, d_deleteKeys.begin());
        return;
    }

    const KeyType* inserted = thrust::raw_pointer_cast(d_insertedKeys.data());
    thrust::transform(
        thrust::counting_iterator<size_t>(0),
        thrust::counting_iterator<size_t>(numDeletes),
        d_deleteKeys.begin(),
        [=] __device__(size_t idx) {
            thrust::default_random_engine rng(seed);
            rng.discard(idx);
            bool pickHot = thrust::uniform_real_distribution<double>(0.0, 1.0)(rng) < skew;
            size_t sourceMax = pickHot ? hotCount : totalInserted;
            size_t selected = thrust::uniform_int_distribution<size_t>(0, sourceMax - 1)(rng);
            return inserted[selected];
        }
    );
}

/**
 * Benchmark: delete a fraction of inserted keys at a given skew level.
 *
 * State ranges:
 *   range(0): filter capacity (number of slots)
 *   range(1): delete fraction index (into DELETE_FRACTIONS)
 *   range(2): skew level index (into SKEW_LEVELS)
 */
static void DeleteSkew(bm::State& state) {
    const auto capacity = static_cast<size_t>(state.range(0));
    const double deleteFraction = DELETE_FRACTIONS[state.range(1)];
    const double skew = SKEW_LEVELS[state.range(2)];

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
        // Re-generate the delete set each iteration (skew changes the contention pattern).
        generateDeleteKeys(d_deleteKeys, numDeletes, d_keys, skew);

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
    state.counters["skew"] = bm::Counter(skew * 100);
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

// Register across capacities x delete fractions x skew levels.
BENCHMARK(DeleteSkew)
    ->ArgsProduct({
        capacityRange(),
        indexRange(DELETE_FRACTIONS),
        indexRange(SKEW_LEVELS),
    })
    ->Unit(bm::kMillisecond)
    ->UseManualTime()
    ->Iterations(10)
    ->Repetitions(3)
    ->ReportAggregatesOnly(true);

STANDARD_BENCHMARK_MAIN();
