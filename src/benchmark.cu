#include <chrono>
#include <cstdint>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>
#include "BucketsTableCpu.cuh"
#include "BucketsTableGpu.cuh"
#include "common.cuh"
#include "HybridTable.cuh"

template <typename T>
size_t countOnes(T* data, size_t n) {
    size_t count = 0;
    for (size_t i = 0; i < n; ++i) {
        if (data[i]) {
            count++;
        }
    }
    return count;
}

struct BenchmarkResult {
    int exponent{};
    size_t n{};
    std::string tableType;
    double avgTimeMs{};
    double minTimeMs{};
    double maxTimeMs{};
    size_t itemsInserted{};
    size_t itemsFound{};
};

template <typename Func>
std::vector<double> benchmarkFunction(Func func, int iterations = 5) {
    std::vector<double> times(iterations);

    for (int i = 0; i < iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        func();
        auto end = std::chrono::high_resolution_clock::now();

        auto duration =
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        times.push_back(duration.count() / 1000.0 / 1000.0);
    }

    return times;
}

BenchmarkResult benchmarkHybridTable(uint32_t* input, size_t n, int exponent) {
    BenchmarkResult result;
    result.exponent = exponent;
    result.n = n;
    result.tableType = "HybridTable";

    size_t count = 0;
    size_t found = 0;

    auto benchmarkFunc = [&]() {
        auto table = HybridTable<uint32_t, 32, 1000, 256>(n * 2);
        count = 0;

        for (size_t i = 0; i < n; ++i) {
            count += size_t(table.insert(input[i]));
        }

        auto mask = table.containsMany(input, n);
        found = countOnes(mask, n);
    };

    auto times = benchmarkFunction(benchmarkFunc);

    result.itemsInserted = count;
    result.itemsFound = found;
    result.minTimeMs = *std::min_element(times.begin(), times.end());
    result.maxTimeMs = *std::max_element(times.begin(), times.end());

    double sum = 0.0;
    for (double time : times) {
        sum += time;
    }
    result.avgTimeMs = sum / times.size();

    return result;
}

BenchmarkResult benchmarkCpuTable(uint32_t* input, size_t n, int exponent) {
    BenchmarkResult result;
    result.exponent = exponent;
    result.n = n;
    result.tableType = "BucketsTableCpu";

    size_t count = 0;
    size_t found = 0;

    auto benchmarkFunc = [&]() {
        auto table = BucketsTableCpu<uint32_t, 32, 32, 1000>(n / 32);

        for (size_t i = 0; i < n; ++i) {
            count += size_t(table.insert(input[i]));
        }

        auto mask = table.containsMany(input, n);
        found = countOnes(mask, n);
    };

    auto times = benchmarkFunction(benchmarkFunc);

    result.itemsInserted = count;
    result.itemsFound = found;
    result.minTimeMs = *std::min_element(times.begin(), times.end());
    result.maxTimeMs = *std::max_element(times.begin(), times.end());

    double sum = 0.0;
    for (double time : times) {
        sum += time;
    }
    result.avgTimeMs = sum / times.size();

    return result;
}

BenchmarkResult benchmarkGpuTable(uint32_t* input, size_t n, int exponent) {
    BenchmarkResult result;
    result.exponent = exponent;
    result.n = n;
    result.tableType = "BucketsTableGpu";

    size_t count = 0;
    size_t found = 0;
    bool* output = nullptr;
    CUDA_CALL(cudaMallocHost(&output, sizeof(bool) * n));

    auto benchmarkFunc = [&]() {
        auto table = BucketsTableGpu<uint32_t, 32, 32, 1000>(n / 32);

        count = table.insertMany(input, n);

        table.containsMany(input, n, output);
        found = countOnes(output, n);
    };

    auto times = benchmarkFunction(benchmarkFunc);

    result.itemsInserted = count;
    result.itemsFound = found;
    result.minTimeMs = *std::min_element(times.begin(), times.end());
    result.maxTimeMs = *std::max_element(times.begin(), times.end());

    double sum = 0.0;
    for (double time : times) {
        sum += time;
    }
    result.avgTimeMs = sum / times.size();

    CUDA_CALL(cudaFreeHost(output));
    return result;
}

void writeResultsToCsv(
    const std::vector<BenchmarkResult>& results,
    const std::string& filename
) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing"
                  << std::endl;
        return;
    }

    file << "exponent,n,tableType,avgTimeMs,minTimeMs,maxTimeMs,items_"
            "inserted,itemsFound\n";

    for (const auto& result : results) {
        file << result.exponent << "," << result.n << "," << result.tableType
             << "," << std::fixed << std::setprecision(6) << result.avgTimeMs
             << "," << std::fixed << std::setprecision(6) << result.minTimeMs
             << "," << std::fixed << std::setprecision(6) << result.maxTimeMs
             << "," << result.itemsInserted << "," << result.itemsFound << "\n";
    }

    file.close();
    std::cout << "Results written to " << filename << std::endl;
}

int main(int argc, char** argv) {
    std::string output_file = "benchmark_results.csv";

    if (argc > 1) {
        output_file = argv[1];
    }

    const int min_exponent = 10;
    const int max_exponent = 30;

    std::cout << "Generating test data..." << std::endl;

    std::mt19937 rng(42);
    std::uniform_int_distribution<uint32_t> dist(1, UINT32_MAX);

    size_t max_n = 1ULL << max_exponent;
    uint32_t* input = nullptr;
    CUDA_CALL(cudaMallocHost(&input, sizeof(uint32_t) * max_n));

    for (size_t i = 0; i < max_n; ++i) {
        input[i] = dist(rng);
    }

    std::vector<BenchmarkResult> results;

    std::cout << "Running benchmarks..." << std::endl;
    std::cout << "Format: [Table Type] - 2^" << "exp"
              << " (size) - avg_time (min-max) ms" << std::endl;
    std::cout << std::string(70, '-') << std::endl;

    for (int exponent = min_exponent; exponent <= max_exponent; ++exponent) {
        size_t n = 1ULL << exponent;

        std::cout << "Testing size 2^" << exponent << " (" << n
                  << " elements):" << std::endl;

        {
            auto result = benchmarkHybridTable(input, n, exponent);
            results.push_back(result);

            std::cout << "  [HybridTable] - \t" << std::fixed
                      << std::setprecision(4) << result.avgTimeMs << " ("
                      << result.minTimeMs << "-" << result.maxTimeMs << ") ms"
                      << std::endl;
        }

        {
            auto result = benchmarkCpuTable(input, n, exponent);
            results.push_back(result);

            std::cout << "  [BucketsTableCpu] - \t" << std::fixed
                      << std::setprecision(4) << result.avgTimeMs << " ("
                      << result.minTimeMs << "-" << result.maxTimeMs << ") ms"
                      << std::endl;
        }

        {
            auto result = benchmarkGpuTable(input, n, exponent);
            results.push_back(result);

            std::cout << "  [BucketsTableGpu] - \t" << std::fixed
                      << std::setprecision(4) << result.avgTimeMs << " ("
                      << result.minTimeMs << "-" << result.maxTimeMs << ") ms"
                      << std::endl;
        }

        std::cout << std::endl;
    }

    writeResultsToCsv(results, output_file);

    CUDA_CALL(cudaFreeHost(input));

    std::cout << "Benchmark completed successfully!" << std::endl;

    return 0;
}