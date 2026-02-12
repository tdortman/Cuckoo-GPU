// Enable eviction counting for this benchmark
#define CUCKOO_FILTER_COUNT_EVICTIONS

#include <benchmark/benchmark.h>
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#include <algorithm>
#include <cstdint>
#include <CuckooFilter.cuh>
#include <filesystem>
#include <fstream>
#include <helpers.cuh>
#include <map>
#include <string>
#include <tuple>
#include <vector>
#include "benchmark_common.cuh"

namespace bm = benchmark;

enum class HistogramPolicy : uint8_t {
    BFS,
    DFS,
};

constexpr const char* toString(HistogramPolicy policy) {
    switch (policy) {
        case HistogramPolicy::BFS:
            return "BFS";
        case HistogramPolicy::DFS:
            return "DFS";
    }

    return "unknown";
}

struct HistogramKey {
    HistogramPolicy policy;
    size_t capacity;
    size_t loadFactorIndex;
    double loadFactor;
    size_t binStart;
    int64_t binEnd;

    bool operator<(const HistogramKey& other) const {
        return std::tie(policy, capacity, loadFactorIndex, loadFactor, binStart, binEnd) <
               std::tie(
                   other.policy,
                   other.capacity,
                   other.loadFactorIndex,
                   other.loadFactor,
                   other.binStart,
                   other.binEnd
               );
    }
};

std::map<HistogramKey, uint64_t> histogramCounts;
std::string histogramOutputPath;

static constexpr int64_t OVERFLOW_BIN_END = -1;

bool histogramEnabled() {
    return !histogramOutputPath.empty();
}

void accumulateHistogram(
    HistogramPolicy policy,
    size_t capacity,
    size_t loadFactorIndex,
    double loadFactor,
    const std::vector<uint32_t>& evictionAttempts,
    const std::vector<uint8_t>& insertionSuccess,
    size_t maxEvictions
) {
    const size_t overflowBin = maxEvictions + 1;

    std::vector<uint64_t> binCounts(overflowBin + 1, 0);

    size_t sampleCount = std::min(evictionAttempts.size(), insertionSuccess.size());
    for (size_t i = 0; i < sampleCount; ++i) {
        uint32_t attemptCount = evictionAttempts[i];
        bool success = insertionSuccess[i] != 0;
        // failed insertions go into the overflow bin
        size_t bin = success ? std::min<size_t>(attemptCount, maxEvictions) : overflowBin;
        binCounts[bin]++;
    }

    for (size_t bin = 0; bin <= overflowBin; ++bin) {
        uint64_t count = binCounts[bin];
        if (count == 0) {
            continue;
        }

        bool isOverflow = bin == overflowBin;
        HistogramKey key{
            policy,
            capacity,
            loadFactorIndex,
            loadFactor,
            bin,
            isOverflow ? OVERFLOW_BIN_END : static_cast<int64_t>(bin),
        };
        histogramCounts[key] += count;
    }
}

void writeHistogramCSV(const std::string& outputPath) {
    if (outputPath.empty()) {
        return;
    }

    std::filesystem::path path(outputPath);
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }

    std::ofstream out(path);
    if (!out.is_open()) {
        std::cerr << "Failed to open histogram output file: " << outputPath << std::endl;
        return;
    }

    out << "policy,capacity,load_factor_index,load_factor,bin_start,bin_end,count\n";
    for (const auto& [key, count] : histogramCounts) {
        out << toString(key.policy) << ',' << key.capacity << ',' << key.loadFactorIndex << ','
            << key.loadFactor << ',' << key.binStart << ',' << key.binEnd << ',' << count << '\n';
    }
}

void parseCustomArgs(int argc, char** argv, std::vector<char*>& benchmarkArgv) {
    benchmarkArgv.clear();
    benchmarkArgv.reserve(argc);
    benchmarkArgv.push_back(argv[0]);

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        constexpr const char* histogramPrefix = "--eviction-histogram-out=";

        if (arg.rfind(histogramPrefix, 0) == 0) {
            histogramOutputPath = arg.substr(std::char_traits<char>::length(histogramPrefix));
            continue;
        }

        if (arg == "--eviction-histogram-out") {
            if (i + 1 < argc) {
                histogramOutputPath = argv[++i];
            } else {
                std::cerr << "Missing value for --eviction-histogram-out" << std::endl;
                std::exit(1);
            }
            continue;
        }

        benchmarkArgv.push_back(argv[i]);
    }
}

using BFSConfig = cuckoogpu::Config<
    uint64_t,
    16,
    500,
    256,
    16,
    cuckoogpu::XorAltBucketPolicy,
    cuckoogpu::EvictionPolicy::BFS>;
using DFSConfig = cuckoogpu::Config<
    uint64_t,
    16,
    500,
    256,
    16,
    cuckoogpu::XorAltBucketPolicy,
    cuckoogpu::EvictionPolicy::DFS>;

static constexpr double PREFILL_FRACTION = 0.75;

static constexpr double LOAD_FACTORS[] = {
    0.76,
    0.78,
    0.80,
    0.82,
    0.84,
    0.86,
    0.88,
    0.90,
    0.92,
    0.94,
    0.95,
    0.96,
    0.97,
    0.98,
    0.99,
    0.995,
    0.999
};

static constexpr size_t NUM_LOAD_FACTORS = sizeof(LOAD_FACTORS) / sizeof(LOAD_FACTORS[0]);

template <typename ConfigType>
class EvictionBenchmarkFixture : public benchmark::Fixture {
    using benchmark::Fixture::SetUp;
    using benchmark::Fixture::TearDown;

   public:
    void SetUp(const benchmark::State& state) override {
        capacity = state.range(0);
        loadFactorIndex = static_cast<size_t>(state.range(1));
        loadFactor = LOAD_FACTORS[loadFactorIndex];

        auto totalKeys = static_cast<size_t>(capacity * loadFactor);
        nPrefill = static_cast<size_t>(totalKeys * PREFILL_FRACTION);
        nMeasured = totalKeys - nPrefill;

        d_keysPrefill.resize(nPrefill);
        d_keysMeasured.resize(nMeasured);
        generateKeysGPU(d_keysPrefill);
        generateKeysGPURange(
            d_keysMeasured, nMeasured, static_cast<uint64_t>(UINT32_MAX) + 1, UINT64_MAX
        );

        filter = std::make_unique<cuckoogpu::Filter<ConfigType>>(capacity);
        filterMemory = filter->sizeInBytes();
    }

    void TearDown(const benchmark::State&) override {
        filter.reset();
        d_keysPrefill.clear();
        d_keysPrefill.shrink_to_fit();
        d_keysMeasured.clear();
        d_keysMeasured.shrink_to_fit();
    }

    void setCounters(benchmark::State& state, size_t evictions, size_t inserted) {
        state.counters["load_factor"] = bm::Counter(loadFactor * 100);
        state.counters["prefill_fraction"] = bm::Counter(PREFILL_FRACTION * 100);
        state.counters["evictions"] = bm::Counter(static_cast<double>(evictions));
        state.counters["inserted"] = bm::Counter(static_cast<double>(inserted));
        state.counters["evictions_per_insert"] = bm::Counter(
            inserted > 0 ? static_cast<double>(evictions) / static_cast<double>(inserted) : 0
        );
        state.SetItemsProcessed(
            static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(nMeasured)
        );
        state.counters["memory_bytes"] = bm::Counter(
            static_cast<double>(filterMemory), bm::Counter::kDefaults, bm::Counter::kIs1024
        );
    }

    size_t capacity;
    size_t loadFactorIndex;
    double loadFactor;
    size_t nPrefill;
    size_t nMeasured;
    size_t filterMemory;
    thrust::device_vector<uint64_t> d_keysPrefill;
    thrust::device_vector<uint64_t> d_keysMeasured;
    std::unique_ptr<cuckoogpu::Filter<ConfigType>> filter;
    GPUTimer timer;
};

using BFSFixture = EvictionBenchmarkFixture<BFSConfig>;
using DFSFixture = EvictionBenchmarkFixture<DFSConfig>;

BENCHMARK_DEFINE_F(BFSFixture, Evictions)(bm::State& state) {
    size_t totalEvictions = 0;
    size_t numIterations = 0;
    const bool collectHistogram = histogramEnabled();
    const size_t maxEvictions = filter->maxEvictions;
    thrust::device_vector<uint32_t> d_evictionAttempts;
    thrust::device_vector<uint8_t> d_insertSuccess;
    std::vector<uint32_t> h_evictionAttempts;
    std::vector<uint8_t> h_insertSuccess;
    if (collectHistogram) {
        h_evictionAttempts.resize(nMeasured);
        h_insertSuccess.resize(nMeasured);
    }

    for (auto _ : state) {
        filter->clear();
        cudaDeviceSynchronize();
        filter->insertMany(d_keysPrefill);
        filter->resetEvictionCount();
        cudaDeviceSynchronize();

        timer.start();
        size_t inserted = collectHistogram
                              ? filter->insertManyWithEvictionCounts(
                                    d_keysMeasured, d_evictionAttempts, &d_insertSuccess
                                )
                              : filter->insertMany(d_keysMeasured);
        double elapsed = timer.elapsed();

        size_t evictions = filter->evictionCount();
        if (collectHistogram) {
            if (h_evictionAttempts.size() != d_evictionAttempts.size()) {
                h_evictionAttempts.resize(d_evictionAttempts.size());
            }
            if (h_insertSuccess.size() != d_insertSuccess.size()) {
                h_insertSuccess.resize(d_insertSuccess.size());
            }
            CUDA_CALL(cudaMemcpy(
                h_evictionAttempts.data(),
                thrust::raw_pointer_cast(d_evictionAttempts.data()),
                d_evictionAttempts.size() * sizeof(uint32_t),
                cudaMemcpyDeviceToHost
            ));
            CUDA_CALL(cudaMemcpy(
                h_insertSuccess.data(),
                thrust::raw_pointer_cast(d_insertSuccess.data()),
                d_insertSuccess.size() * sizeof(uint8_t),
                cudaMemcpyDeviceToHost
            ));
            accumulateHistogram(
                HistogramPolicy::BFS,
                capacity,
                loadFactorIndex,
                loadFactor,
                h_evictionAttempts,
                h_insertSuccess,
                maxEvictions
            );
        }

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(inserted);

        totalEvictions += evictions;
        numIterations++;
    }

    size_t avgEvictions = numIterations > 0 ? totalEvictions / numIterations : 0;
    setCounters(state, avgEvictions, nMeasured);
}

BENCHMARK_DEFINE_F(DFSFixture, Evictions)(bm::State& state) {
    size_t totalEvictions = 0;
    size_t numIterations = 0;
    const bool collectHistogram = histogramEnabled();
    const size_t maxEvictions = filter->maxEvictions;
    thrust::device_vector<uint32_t> d_evictionAttempts;
    thrust::device_vector<uint8_t> d_insertSuccess;
    std::vector<uint32_t> h_evictionAttempts;
    std::vector<uint8_t> h_insertSuccess;
    if (collectHistogram) {
        h_evictionAttempts.resize(nMeasured);
        h_insertSuccess.resize(nMeasured);
    }

    for (auto _ : state) {
        filter->clear();
        cudaDeviceSynchronize();
        filter->insertMany(d_keysPrefill);
        filter->resetEvictionCount();
        cudaDeviceSynchronize();

        timer.start();
        size_t inserted = collectHistogram
                              ? filter->insertManyWithEvictionCounts(
                                    d_keysMeasured, d_evictionAttempts, &d_insertSuccess
                                )
                              : filter->insertMany(d_keysMeasured);
        double elapsed = timer.elapsed();

        size_t evictions = filter->evictionCount();
        if (collectHistogram) {
            if (h_evictionAttempts.size() != d_evictionAttempts.size()) {
                h_evictionAttempts.resize(d_evictionAttempts.size());
            }
            if (h_insertSuccess.size() != d_insertSuccess.size()) {
                h_insertSuccess.resize(d_insertSuccess.size());
            }
            CUDA_CALL(cudaMemcpy(
                h_evictionAttempts.data(),
                thrust::raw_pointer_cast(d_evictionAttempts.data()),
                d_evictionAttempts.size() * sizeof(uint32_t),
                cudaMemcpyDeviceToHost
            ));
            CUDA_CALL(cudaMemcpy(
                h_insertSuccess.data(),
                thrust::raw_pointer_cast(d_insertSuccess.data()),
                d_insertSuccess.size() * sizeof(uint8_t),
                cudaMemcpyDeviceToHost
            ));
            accumulateHistogram(
                HistogramPolicy::DFS,
                capacity,
                loadFactorIndex,
                loadFactor,
                h_evictionAttempts,
                h_insertSuccess,
                maxEvictions
            );
        }

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(inserted);

        totalEvictions += evictions;
        numIterations++;
    }

    size_t avgEvictions = numIterations > 0 ? totalEvictions / numIterations : 0;
    setCounters(state, avgEvictions, nMeasured);
}

#define EVICTION_BENCHMARK_CONFIG                                                       \
    ->ArgsProduct({{1 << 22}, benchmark::CreateDenseRange(0, NUM_LOAD_FACTORS - 1, 1)}) \
        ->Unit(benchmark::kMillisecond)                                                 \
        ->UseManualTime()                                                               \
        ->Iterations(10)                                                                \
        ->Repetitions(5)                                                                \
        ->ReportAggregatesOnly(true)

BENCHMARK_REGISTER_F(BFSFixture, Evictions)
EVICTION_BENCHMARK_CONFIG;

BENCHMARK_REGISTER_F(DFSFixture, Evictions)
EVICTION_BENCHMARK_CONFIG;

int main(int argc, char** argv) {
    std::vector<char*> benchmarkArgv;
    parseCustomArgs(argc, argv, benchmarkArgv);

    int benchmarkArgc = static_cast<int>(benchmarkArgv.size());
    benchmark::Initialize(&benchmarkArgc, benchmarkArgv.data());
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();

    writeHistogramCSV(histogramOutputPath);

    return 0;
}
