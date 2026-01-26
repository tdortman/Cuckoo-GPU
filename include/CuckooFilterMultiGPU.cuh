#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <numeric>
#include <thread>
#include <type_traits>
#include <vector>
#include "CuckooFilter.cuh"
#include "helpers.cuh"

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/scatter.h>

#include <gossip.cuh>
#include <plan_parser.hpp>

/**
 * @brief A multi-GPU implementation of the Cuckoo Filter.
 *
 * This class partitions keys across multiple GPUs using the gossip library
 * for efficient multi-GPU communication. It handles data distribution using
 * gossip's multisplit and all-to-all primitives, and aggregates results.
 *
 * @tparam Config The configuration structure for the Cuckoo Filter.
 */
template <typename Config>
class CuckooFilterMultiGPU {
   public:
    using T = typename Config::KeyType;

    /**
     * @brief Functor for partitioning keys across GPUs.
     *
     * Uses a hash function to assign each key to a specific GPU index.
     * Compatible with gossip's multisplit which requires __host__ __device__ functor.
     */
    struct Partitioner {
        size_t numGPUs;

        __host__ __device__ gossip::gpu_id_t operator()(const T& key) const {
            uint64_t hash = CuckooFilter<Config>::hash64(key);
            return static_cast<gossip::gpu_id_t>(hash % numGPUs);
        }
    };

    /// Default fraction of free GPU memory to use for buffers (after filter allocation)
    static constexpr float defaultMemoryFactor = 0.8f;

   private:
    size_t numGPUs;
    size_t capacityPerGPU;
    float memoryFactor;
    std::vector<CuckooFilter<Config>*> filters;

    gossip::context_t gossipContext;
    gossip::multisplit_t multisplit;
    gossip::all2all_t all2all;
    gossip::all2all_t all2allResults;

    // Pre-allocated per-GPU buffers for gossip operations
    std::vector<T*> srcBuffers;
    std::vector<T*> dstBuffers;
    std::vector<size_t> bufferCapacities;

    std::vector<bool*> resultSrcBuffers;
    std::vector<bool*> resultDstBuffers;

    size_t totalBufferCapacity;

    /**
     * @brief Gets the available free memory for each GPU.
     * @return A vector where the i-th element is the free memory on GPU i.
     */
    [[nodiscard]] std::vector<size_t> getGpuMemoryInfo() const {
        std::vector<size_t> freeMem(numGPUs);
        parallelForGPUs([&](size_t gpuId) {
            size_t free, total;
            CUDA_CALL(cudaMemGetInfo(&free, &total));
            freeMem[gpuId] = free;
        });
        return freeMem;
    }

    /**
     * @brief Pre-allocates buffers on each GPU based on available VRAM.
     *
     * Called after filter allocation. Each GPU gets buffers sized proportionally
     * to its available free memory. This allows processing large datasets in chunks
     * without requiring buffer reallocation.
     */
    void allocateBuffers() {
        // Bytes per element: 2 key buffers (src + dst) + 2 result buffers (src + dst)
        const size_t bytesPerKey = 2 * sizeof(T) + 2 * sizeof(bool);

        totalBufferCapacity = 0;

        parallelForGPUs([&](size_t gpuId) {
            size_t freeMem, totalMem;
            CUDA_CALL(cudaMemGetInfo(&freeMem, &totalMem));

            // Calculate max keys this GPU can buffer
            auto availableBytes = static_cast<size_t>(freeMem * memoryFactor);
            size_t maxKeys = availableBytes / bytesPerKey;

            // Allocate key buffers
            CUDA_CALL(cudaMalloc(&srcBuffers[gpuId], maxKeys * sizeof(T)));
            CUDA_CALL(cudaMalloc(&dstBuffers[gpuId], maxKeys * sizeof(T)));

            // Allocate result buffers
            CUDA_CALL(cudaMalloc(&resultSrcBuffers[gpuId], maxKeys * sizeof(bool)));
            CUDA_CALL(cudaMalloc(&resultDstBuffers[gpuId], maxKeys * sizeof(bool)));

            bufferCapacities[gpuId] = maxKeys;
            totalBufferCapacity += maxKeys;
        });
    }

    /**
     * @brief Free all pre-allocated buffers.
     */
    void freeBuffers() {
        parallelForGPUs([&](size_t gpuId) {
            if (srcBuffers[gpuId]) {
                cudaFree(srcBuffers[gpuId]);
                srcBuffers[gpuId] = nullptr;
            }
            if (dstBuffers[gpuId]) {
                cudaFree(dstBuffers[gpuId]);
                dstBuffers[gpuId] = nullptr;
            }
            if (resultSrcBuffers[gpuId]) {
                cudaFree(resultSrcBuffers[gpuId]);
                resultSrcBuffers[gpuId] = nullptr;
            }
            if (resultDstBuffers[gpuId]) {
                cudaFree(resultDstBuffers[gpuId]);
                resultDstBuffers[gpuId] = nullptr;
            }
            bufferCapacities[gpuId] = 0;
        });
    }

    /**
     * @brief Generic function for processing keys using gossip primitives.
     *
     * The workflow is:
     * 1. Distribute input keys to GPUs
     * 2. Use gossip multisplit to partition keys by target GPU
     * 3. Use gossip all2all to shuffle keys to correct GPUs
     * 4. Execute filter operation locally
     * 5. Use gossip all2all to return results (if hasOutput)
     * 6. Reorder results to match original input order
     *
     * @param h_keys Pointer to host memory containing keys to process.
     * @param n Number of keys to process.
     * @param h_output Optional pointer to host memory to store the boolean results.
     * @param filterOp The specific CuckooFilter operation to execute.
     * @tparam returnOccupied If true, returns total occupied slots after operation.
     * @tparam hasOutput If true, expects an output buffer for results.
     * @return Total occupied slots if returnOccupied is true, otherwise 0.
     */
    template <bool returnOccupied, bool hasOutput, typename FilterFunc>
    size_t executeOperation(const T* h_keys, size_t n, bool* h_output, FilterFunc filterOp) {
        if (n == 0) {
            return returnOccupied ? totalOccupiedSlots() : 0;
        }

        size_t processed = 0;

        while (processed < n) {
            size_t chunkSize = std::min(n - processed, totalBufferCapacity);

            // Distribute chunk proportionally based on each GPU's buffer capacity
            std::vector<size_t> inputLens(numGPUs);
            std::vector<size_t> inputOffsets(numGPUs + 1, 0);

            size_t remaining = chunkSize;
            for (size_t gpu = 0; gpu < numGPUs; ++gpu) {
                if (gpu == numGPUs - 1) {
                    inputLens[gpu] = std::min(remaining, bufferCapacities[gpu]);
                } else {
                    double proportion =
                        static_cast<double>(bufferCapacities[gpu]) / totalBufferCapacity;
                    inputLens[gpu] = std::min(
                        static_cast<size_t>(chunkSize * proportion), bufferCapacities[gpu]
                    );
                    inputLens[gpu] = std::min(inputLens[gpu], remaining);
                }
                remaining -= inputLens[gpu];
                inputOffsets[gpu + 1] = inputOffsets[gpu] + inputLens[gpu];
            }

            // Copy input data to source buffers on each GPU
            parallelForGPUs([&](size_t gpuId) {
                if (inputLens[gpuId] > 0) {
                    CUDA_CALL(cudaMemcpy(
                        srcBuffers[gpuId],
                        h_keys + processed + inputOffsets[gpuId],
                        inputLens[gpuId] * sizeof(T),
                        cudaMemcpyHostToDevice
                    ));
                }
            });
            gossipContext.sync_hard();

            // Partition keys by target GPU
            std::vector<std::vector<size_t>> partitionTable(numGPUs, std::vector<size_t>(numGPUs));

            Partitioner partitioner{numGPUs};
            multisplit.execAsync(
                srcBuffers,        // source pointers (per GPU)
                inputLens,         // source lengths (per GPU)
                dstBuffers,        // destination pointers (per GPU)
                bufferCapacities,  // destination capacities (per GPU)
                partitionTable,    // output: partition counts [src][dst]
                partitioner
            );
            multisplit.sync();

            std::swap(srcBuffers, dstBuffers);

            // Calculate how many keys each GPU will receive after all2all
            std::vector<size_t> recvCounts(numGPUs, 0);
            for (size_t dst = 0; dst < numGPUs; ++dst) {
                for (size_t src = 0; src < numGPUs; ++src) {
                    recvCounts[dst] += partitionTable[src][dst];
                }
            }

            // Shuffle partitioned keys to correct GPUs
            all2all.execAsync(
                srcBuffers,        // partitioned source data
                bufferCapacities,  // source buffer capacities
                dstBuffers,        // destination for received data
                bufferCapacities,  // destination buffer capacities
                partitionTable     // partition counts from multisplit
            );
            all2all.sync();

            // If no output is required, execute filter ops and continue
            if constexpr (!hasOutput) {
                parallelForGPUs([&](size_t gpuId) {
                    size_t localCount = recvCounts[gpuId];
                    if (localCount == 0) {
                        return;
                    }
                    auto stream = gossipContext.get_streams(gpuId)[0];
                    filterOp(filters[gpuId], dstBuffers[gpuId], nullptr, localCount, stream);
                });
                gossipContext.sync_all_streams();
            } else {
                // Transpose partitionTable in-place for reverse all-to-all
                for (size_t i = 0; i < numGPUs; ++i) {
                    for (size_t j = i + 1; j < numGPUs; ++j) {
                        std::swap(partitionTable[i][j], partitionTable[j][i]);
                    }
                }

                // Execute filter operations
                parallelForGPUs([&](size_t gpuId) {
                    size_t localCount = recvCounts[gpuId];
                    if (localCount == 0) {
                        return;
                    }
                    auto stream = gossipContext.get_streams(gpuId)[0];
                    filterOp(
                        filters[gpuId],
                        dstBuffers[gpuId],
                        resultSrcBuffers[gpuId],
                        localCount,
                        stream
                    );
                });
                gossipContext.sync_all_streams();

                all2allResults.execAsync(
                    resultSrcBuffers, recvCounts, resultDstBuffers, bufferCapacities, partitionTable
                );
                all2allResults.sync();

                parallelForGPUs([&](size_t gpuId) {
                    size_t localCount = inputLens[gpuId];
                    if (localCount == 0) {
                        return;
                    }

                    CUDA_CALL(cudaMemcpy(
                        h_output + processed + inputOffsets[gpuId],
                        resultDstBuffers[gpuId],
                        localCount * sizeof(bool),
                        cudaMemcpyDeviceToHost
                    ));
                });
                gossipContext.sync_hard();
            }

            processed += chunkSize;
        }

        return returnOccupied ? totalOccupiedSlots() : 0;
    }

   public:
    /**
     * @brief Constructs a new CuckooFilterMultiGPU with default transfer plan.
     *
     * Initializes gossip context, multisplit, all-to-all primitives, and CuckooFilter instances
     * on each available GPU.
     *
     * @param numGPUs Number of GPUs to use.
     * @param capacity Total capacity of the distributed filter.
     */
    CuckooFilterMultiGPU(size_t numGPUs, size_t capacity, float memFactor = defaultMemoryFactor)
        : numGPUs(numGPUs),
          capacityPerGPU(static_cast<size_t>(SDIV(capacity, numGPUs) * 1.02)),
          memoryFactor(memFactor),
          gossipContext(numGPUs),
          multisplit(gossipContext),
          all2all(gossipContext, gossip::all2all::default_plan(numGPUs)),
          all2allResults(gossipContext, gossip::all2all::default_plan(numGPUs)),
          srcBuffers(numGPUs, nullptr),
          dstBuffers(numGPUs, nullptr),
          bufferCapacities(numGPUs, 0),
          resultSrcBuffers(numGPUs, nullptr),
          resultDstBuffers(numGPUs, nullptr),
          totalBufferCapacity(0) {
        assert(numGPUs > 0 && "Number of GPUs must be at least 1");

        filters.resize(numGPUs);

        for (size_t i = 0; i < numGPUs; ++i) {
            CUDA_CALL(cudaSetDevice(gossipContext.get_device_id(i)));
            CuckooFilter<Config>* filter;
            CUDA_CALL(cudaMallocManaged(&filter, sizeof(CuckooFilter<Config>)));
            new (filter) CuckooFilter<Config>(capacityPerGPU);
            filters[i] = filter;
        }
        gossipContext.sync_hard();

        allocateBuffers();
    }

    /**
     * @brief Constructs a new CuckooFilterMultiGPU with custom transfer plan.
     *
     * Initializes gossip context, multisplit, all-to-all primitives with provided
     * transfer plan loaded from file, and CuckooFilter instances on each available GPU.
     *
     * @param numGPUs Number of GPUs to use.
     * @param capacity Total capacity of the distributed filter.
     * @param transferPlanPath Path to gossip transfer plan file for optimized topology-aware
     * transfers.
     */
    CuckooFilterMultiGPU(
        size_t numGPUs,
        size_t capacity,
        const char* transferPlanPath,
        float memFactor = defaultMemoryFactor
    )
        : numGPUs(numGPUs),
          capacityPerGPU(static_cast<size_t>(SDIV(capacity, numGPUs) * 1.02)),
          memoryFactor(memFactor),
          gossipContext(numGPUs),
          multisplit(gossipContext),
          all2all(
              gossipContext,
              [&]() {
                  auto plan = parse_plan(transferPlanPath);
                  if (plan.num_gpus() == 0) {
                      return gossip::all2all::default_plan(numGPUs);
                  }
                  return plan;
              }()
          ),
          all2allResults(
              gossipContext,
              [&]() {
                  auto plan = parse_plan(transferPlanPath);
                  if (plan.num_gpus() == 0) {
                      return gossip::all2all::default_plan(numGPUs);
                  }
                  return plan;
              }()
          ),
          srcBuffers(numGPUs, nullptr),
          dstBuffers(numGPUs, nullptr),
          bufferCapacities(numGPUs, 0),
          resultSrcBuffers(numGPUs, nullptr),
          resultDstBuffers(numGPUs, nullptr),
          totalBufferCapacity(0) {
        assert(numGPUs > 0 && "Number of GPUs must be at least 1");

        filters.resize(numGPUs);

        for (size_t i = 0; i < numGPUs; ++i) {
            CUDA_CALL(cudaSetDevice(gossipContext.get_device_id(i)));
            CuckooFilter<Config>* filter;
            CUDA_CALL(cudaMallocManaged(&filter, sizeof(CuckooFilter<Config>)));
            new (filter) CuckooFilter<Config>(capacityPerGPU);
            filters[i] = filter;
        }
        gossipContext.sync_hard();

        allocateBuffers();
    }

    /**
     * @brief Destroys the CuckooFilterMultiGPU.
     *
     * Cleans up filter instances and pre-allocated buffers.
     */
    ~CuckooFilterMultiGPU() {
        freeBuffers();
        for (size_t i = 0; i < numGPUs; ++i) {
            CUDA_CALL(cudaSetDevice(gossipContext.get_device_id(i)));
            filters[i]->~CuckooFilter<Config>();
            CUDA_CALL(cudaFree(filters[i]));
        }
    }

    CuckooFilterMultiGPU(const CuckooFilterMultiGPU&) = delete;
    CuckooFilterMultiGPU& operator=(const CuckooFilterMultiGPU&) = delete;

    /**
     * @brief Inserts a batch of keys into the distributed filter.
     * Uses gossip primitives for efficient multi-GPU data distribution.
     * @param h_keys Pointer to host memory containing keys to insert.
     * @param n Number of keys to insert.
     * @param h_output Optional pointer to host memory to store results (true if successfully
     * inserted).
     * @return The total number of occupied slots across all GPUs after insertion.
     */
    size_t insertMany(const T* h_keys, size_t n, bool* h_output = nullptr) {
        if (h_output) {
            return executeOperation<true, true>(
                h_keys,
                n,
                h_output,
                [](CuckooFilter<Config>* filter,
                   const T* keys,
                   bool* results,
                   size_t count,
                   cudaStream_t stream) { filter->insertMany(keys, count, results, stream); }
            );
        } else {
            return executeOperation<true, false>(
                h_keys,
                n,
                nullptr,
                [](CuckooFilter<Config>* filter,
                   const T* keys,
                   bool* /*unused results*/,
                   size_t count,
                   cudaStream_t stream) { filter->insertMany(keys, count, nullptr, stream); }
            );
        }
    }

    /**
     * @brief Checks for the presence of multiple keys in the filter.
     * @param h_keys Pointer to host memory containing keys to check.
     * @param n Number of keys to check.
     * @param h_output Pointer to host memory to store results (true if present, false otherwise).
     */
    void containsMany(const T* h_keys, size_t n, bool* h_output) {
        executeOperation<false, true>(
            h_keys,
            n,
            h_output,
            [](CuckooFilter<Config>* filter,
               const T* keys,
               bool* results,
               size_t count,
               cudaStream_t stream) { filter->containsMany(keys, count, results, stream); }
        );
    }

    /**
     * @brief Deletes multiple keys from the filter.
     * @param h_keys Pointer to host memory containing keys to delete.
     * @param n Number of keys to delete.
     * @param h_output Optional pointer to host memory to store results (true if found and deleted).
     * @return The total number of occupied slots across all GPUs after deletion.
     */
    size_t deleteMany(const T* h_keys, size_t n, bool* h_output = nullptr) {
        if (h_output) {
            return executeOperation<true, true>(
                h_keys,
                n,
                h_output,
                [](CuckooFilter<Config>* filter,
                   const T* keys,
                   bool* results,
                   size_t count,
                   cudaStream_t stream) { filter->deleteMany(keys, count, results, stream); }
            );
        } else {
            return executeOperation<true, false>(
                h_keys,
                n,
                nullptr,
                [](CuckooFilter<Config>* filter,
                   const T* keys,
                   bool* /*unused results*/,
                   size_t count,
                   cudaStream_t stream) { filter->deleteMany(keys, count, nullptr, stream); }
            );
        }
    }

    /**
     * @brief Calculates the global load factor.
     * @return float Load factor (total occupied / total capacity).
     */
    [[nodiscard]] float loadFactor() const {
        return static_cast<float>(totalOccupiedSlots()) / static_cast<float>(totalCapacity());
    }

    /**
     * @brief Executes a function in parallel across all GPUs.
     *
     * Spawns a thread for each GPU to run the provided function.
     *
     * @tparam Func Type of the function to execute.
     * @param func The function to execute, taking the GPU index as an argument.
     */
    template <typename Func>
    void parallelForGPUs(Func func) const {
        std::vector<std::thread> threads;
        for (size_t i = 0; i < numGPUs; ++i) {
            threads.emplace_back([=, this]() {
                CUDA_CALL(cudaSetDevice(gossipContext.get_device_id(i)));
                func(i);
            });
        }

        for (auto& t : threads) {
            t.join();
        }
    }

    /**
     * @brief Synchronizes all GPU streams used by this filter.
     */
    void synchronizeAllGPUs() {
        gossipContext.sync_all_streams();
    }

    /**
     * @brief Returns the total number of occupied slots across all GPUs.
     * @return size_t Total occupied slots.
     */
    [[nodiscard]] size_t totalOccupiedSlots() const {
        std::atomic<size_t> total(0);
        parallelForGPUs([&](size_t i) {
            total.fetch_add(filters[i]->occupiedSlots(), std::memory_order_relaxed);
        });

        return total.load();
    }

    /**
     * @brief Clears all filters on all GPUs.
     */
    void clear() {
        parallelForGPUs([&](size_t i) { filters[i]->clear(); });
    }

    /**
     * @brief Returns the total capacity of the distributed filter.
     * @return size_t Total capacity.
     */
    [[nodiscard]] size_t totalCapacity() const {
        std::atomic<size_t> total(0);
        parallelForGPUs([&](size_t i) {
            total.fetch_add(filters[i]->capacity(), std::memory_order_relaxed);
        });
        return total.load();
    }

    [[nodiscard]] size_t sizeInBytes() const {
        std::atomic<size_t> total(0);
        parallelForGPUs([&](size_t i) {
            total.fetch_add(filters[i]->sizeInBytes(), std::memory_order_relaxed);
        });
        return total.load();
    }

    /**
     * @brief Inserts keys from a Thrust host vector.
     * @param h_keys Vector of keys to insert.
     * @param h_output Vector to store results (bool). Resized if necessary.
     * @return size_t Total number of occupied slots.
     */
    size_t insertMany(const thrust::host_vector<T>& h_keys, thrust::host_vector<bool>& h_output) {
        h_output.resize(h_keys.size());
        return insertMany(
            thrust::raw_pointer_cast(h_keys.data()),
            h_keys.size(),
            thrust::raw_pointer_cast(h_output.data())
        );
    }

    /**
     * @brief Inserts keys from a Thrust host vector (uint8_t output).
     * @param h_keys Vector of keys to insert.
     * @param h_output Vector to store results (uint8_t). Resized if necessary.
     * @return size_t Total number of occupied slots.
     */
    size_t
    insertMany(const thrust::host_vector<T>& h_keys, thrust::host_vector<uint8_t>& h_output) {
        h_output.resize(h_keys.size());
        return insertMany(
            thrust::raw_pointer_cast(h_keys.data()),
            h_keys.size(),
            reinterpret_cast<bool*>(thrust::raw_pointer_cast(h_output.data()))
        );
    }

    /**
     * @brief Inserts keys from a Thrust host vector without outputting results.
     * @param h_keys Vector of keys to insert.
     * @return size_t Total number of occupied slots.
     */
    size_t insertMany(const thrust::host_vector<T>& h_keys) {
        return insertMany(thrust::raw_pointer_cast(h_keys.data()), h_keys.size(), nullptr);
    }

    /**
     * @brief Checks for existence of keys in a Thrust host vector.
     * @param h_keys Vector of keys to check.
     * @param h_output Vector to store results (bool). Resized if necessary.
     */
    void containsMany(const thrust::host_vector<T>& h_keys, thrust::host_vector<bool>& h_output) {
        h_output.resize(h_keys.size());
        containsMany(
            thrust::raw_pointer_cast(h_keys.data()),
            h_keys.size(),
            thrust::raw_pointer_cast(h_output.data())
        );
    }

    /**
     * @brief Checks for existence of keys in a Thrust host vector (uint8_t output).
     * @param h_keys Vector of keys to check.
     * @param h_output Vector to store results (uint8_t). Resized if necessary.
     */
    void
    containsMany(const thrust::host_vector<T>& h_keys, thrust::host_vector<uint8_t>& h_output) {
        h_output.resize(h_keys.size());
        containsMany(
            thrust::raw_pointer_cast(h_keys.data()),
            h_keys.size(),
            reinterpret_cast<bool*>(thrust::raw_pointer_cast(h_output.data()))
        );
    }

    /**
     * @brief Deletes keys in a Thrust host vector.
     * @param h_keys Vector of keys to delete.
     * @param h_output Vector to store results (bool). Resized if necessary.
     * @return size_t Total number of occupied slots.
     */
    size_t deleteMany(const thrust::host_vector<T>& h_keys, thrust::host_vector<bool>& h_output) {
        h_output.resize(h_keys.size());
        return deleteMany(
            thrust::raw_pointer_cast(h_keys.data()),
            h_keys.size(),
            thrust::raw_pointer_cast(h_output.data())
        );
    }

    /**
     * @brief Deletes keys in a Thrust host vector (uint8_t output).
     * @param h_keys Vector of keys to delete.
     * @param h_output Vector to store results (uint8_t). Resized if necessary.
     * @return size_t Total number of occupied slots.
     */
    size_t
    deleteMany(const thrust::host_vector<T>& h_keys, thrust::host_vector<uint8_t>& h_output) {
        h_output.resize(h_keys.size());
        return deleteMany(
            thrust::raw_pointer_cast(h_keys.data()),
            h_keys.size(),
            reinterpret_cast<bool*>(thrust::raw_pointer_cast(h_output.data()))
        );
    }

    /**
     * @brief Deletes keys in a Thrust host vector without outputting results.
     * @param h_keys Vector of keys to delete.
     * @return size_t Total number of occupied slots.
     */
    size_t deleteMany(const thrust::host_vector<T>& h_keys) {
        return deleteMany(thrust::raw_pointer_cast(h_keys.data()), h_keys.size(), nullptr);
    }
};