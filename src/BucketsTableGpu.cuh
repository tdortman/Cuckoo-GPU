#pragma once

#include <curand_kernel.h>
#include <cstdint>
#include <ctime>
#include <cuco/hash_functions.cuh>
#include <cuda/std/atomic>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <iostream>
#include <vector>
#include "common.cuh"

struct SpinLock {
    int lock_flag = 0;

    __device__ void lock() {
        while (atomicCAS(&lock_flag, 0, 1) != 0) {
            // Spin until lock is acquired
        }
    }

    __device__ void unlock() {
        atomicExch(&lock_flag, 0);
    }
};

template <size_t numBuckets>
__global__ void
initRandStatesKernel(curandState* states, unsigned long long seed) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numBuckets) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

template <
    typename T,
    size_t bitsPerTag,
    size_t bucketSize,
    size_t numBuckets,
    size_t maxProbes,
    size_t blockSize>
class BucketsTableGpu;

template <
    typename T,
    size_t bitsPerTag,
    size_t bucketSize,
    size_t numBuckets,
    size_t maxProbes,
    size_t blockSize>
__global__ void insertKernel(
    const T* keys,
    size_t n,
    typename BucketsTableGpu<
        T,
        bitsPerTag,
        bucketSize,
        numBuckets,
        maxProbes,
        blockSize>::DeviceTableView table_view
);

template <
    typename T,
    size_t bitsPerTag,
    size_t bucketSize,
    size_t numBuckets,
    size_t maxProbes,
    size_t blockSize>
__global__ void containsKernel(
    const T* keys,
    bool* output,
    size_t n,
    typename BucketsTableGpu<
        T,
        bitsPerTag,
        bucketSize,
        numBuckets,
        maxProbes,
        blockSize>::DeviceTableView table_view
);

template <
    typename T,
    size_t bitsPerTag,
    size_t bucketSize = 32,
    size_t numBuckets = 256,
    size_t maxProbes = 500,
    size_t blockSize = 256>
class BucketsTableGpu {
    static_assert(bitsPerTag <= 32, "The tag cannot be larger than 32 bits");
    static_assert(bitsPerTag >= 1, "The tag must be at least 1 bit");
    static_assert(
        bitsPerTag <= 8 * sizeof(T),
        "The tag cannot be larger than the size of the type"
    );
    static_assert(
        powerOfTwo(numBuckets),
        "Number of buckets must be a power of 2"
    );
    static_assert(bucketSize > 0, "Bucket size must be greater than 0");
    static_assert(
        bucketSize <= 32,
        "Bucket size must be <= 32 for warp operations"
    );

   public:
    using TagType = typename std::conditional<
        bitsPerTag <= 8,
        uint8_t,
        typename std::conditional<bitsPerTag <= 16, uint16_t, uint32_t>::type>::
        type;

    static constexpr TagType EMPTY = 0;
    static constexpr size_t tagMask = (1ULL << bitsPerTag) - 1;

    struct __align__(alignof(TagType)) Bucket {
        TagType tags[bucketSize];

        __device__ int findEmptySlotWarp() const {
            unsigned int lane_id = threadIdx.x % 32;
            bool has_empty = false;

            if (lane_id < bucketSize) {
                has_empty = (tags[lane_id] == EMPTY);
            }

            unsigned int mask = __ballot_sync(0xffffffff, has_empty);
            if (mask == 0) {
                return -1;
            }

            return __ffs(mask) - 1;
        }

        __device__ bool containsWarp(TagType tag) const {
            unsigned int lane_id = threadIdx.x % 32;
            bool found = false;

            if (lane_id < bucketSize) {
                found = (tags[lane_id] == tag);
            }

            unsigned int mask = __ballot_sync(0xffffffff, found);
            return mask != 0;
        }

        __device__ int findEmptySlot() const {
            for (size_t i = 0; i < bucketSize; ++i) {
                if (tags[i] == EMPTY) {
                    return static_cast<int>(i);
                }
            }
            return -1;
        }

        __device__ bool contains(TagType tag) const {
            for (size_t i = 0; i < bucketSize; ++i) {
                if (tags[i] == tag) {
                    return true;
                }
            }
            return false;
        }

        __device__ void insertAt(size_t slot, TagType tag) {
            tags[slot] = tag;
        }

        __device__ bool remove(TagType tag) {
            for (size_t i = 0; i < bucketSize; ++i) {
                if (tags[i] == tag) {
                    tags[i] = EMPTY;
                    return true;
                }
            }
            return false;
        }

        __device__ TagType getTagAt(size_t slot) const {
            return tags[slot];
        }
    };

    struct InsertionCandidate {
        size_t bucket_idx;
        TagType fingerprint;
    };

   private:
    Bucket* d_buckets;
    SpinLock* d_locks{};
    curandState* d_rand_states{};

    size_t* d_numOccupied{};
    size_t h_numOccupied = 0;

    template <typename H>
    static __host__ __device__ uint32_t hash(const H& key) {
        auto bytes = reinterpret_cast<const cuda::std::byte*>(&key);
        cuco::xxhash_32<H> hasher;
        return hasher.compute_hash(bytes, sizeof(H));
    }

    static __host__ __device__ TagType fingerprint(const T& key) {
        uint32_t hash_val = hash(key);
        auto fp = static_cast<TagType>(hash_val & tagMask);
        return fp == 0 ? 1 : fp;
    }

    static __host__ __device__ cuda::std::tuple<size_t, size_t, TagType>
    getCandidateBuckets(const T& key) {
        TagType fp = fingerprint(key);
        size_t h1 = hash(key) & (numBuckets - 1);
        size_t h2 = h1 ^ (hash(fp) & (numBuckets - 1));
        return {h1, h2, fp};
    }

    static __host__ __device__ size_t
    getAlternateBucket(size_t bucket, TagType fp) {
        return bucket ^ (hash(fp) & (numBuckets - 1));
    }

   public:
    explicit BucketsTableGpu() {
        CUDA_CALL(cudaMalloc(&d_buckets, numBuckets * sizeof(Bucket)));
        CUDA_CALL(cudaMalloc(&d_locks, numBuckets * sizeof(SpinLock)));
        CUDA_CALL(
            cudaMalloc(&d_numOccupied, sizeof(cuda::std::atomic<size_t>))
        );
        CUDA_CALL(cudaMalloc(&d_rand_states, numBuckets * sizeof(curandState)));

        size_t numBlocks = (numBuckets + blockSize - 1) / blockSize;
        initRandStatesKernel<numBuckets>
            <<<numBlocks, blockSize>>>(d_rand_states, time(nullptr));
        CUDA_CALL(cudaDeviceSynchronize());

        clear();
    }

    ~BucketsTableGpu() {
        if (d_buckets) {
            CUDA_CALL(cudaFree(d_buckets));
        }
        if (d_locks) {
            CUDA_CALL(cudaFree(d_locks));
        }
        if (d_numOccupied) {
            CUDA_CALL(cudaFree(d_numOccupied));
        }
        if (d_rand_states) {
            CUDA_CALL(cudaFree(d_rand_states));
        }
    }

    size_t insertMany(const T* keys, const size_t n) {
        T* d_keys;
        CUDA_CALL(cudaMalloc(&d_keys, n * sizeof(T)));

        constexpr size_t numStreams = 4;
        const size_t chunkSize = SDIV(n, numStreams);
        cudaStream_t streams[numStreams];

        for (auto& stream : streams) {
            CUDA_CALL(cudaStreamCreate(&stream));
        }

        for (size_t i = 0; i < numStreams; ++i) {
            size_t offset = i * chunkSize;
            size_t currentChunkSize = std::min(chunkSize, n - offset);
            if (currentChunkSize > 0) {
                CUDA_CALL(cudaMemcpyAsync(
                    d_keys + offset,
                    keys + offset,
                    currentChunkSize * sizeof(T),
                    cudaMemcpyHostToDevice,
                    streams[i]
                ));
            }
        }

        for (auto& stream : streams) {
            CUDA_CALL(cudaStreamSynchronize(stream));
            CUDA_CALL(cudaStreamDestroy(stream));
        }

        size_t numBlocks = SDIV(n, blockSize);
        insertKernel<
            T,
            bitsPerTag,
            bucketSize,
            numBuckets,
            maxProbes,
            blockSize><<<numBlocks, blockSize>>>(d_keys, n, get_device_view());

        CUDA_CALL(cudaDeviceSynchronize());
        CUDA_CALL(cudaFree(d_keys));

        CUDA_CALL(cudaMemcpy(
            &h_numOccupied,
            d_numOccupied,
            sizeof(size_t),
            cudaMemcpyDeviceToHost
        ));

        return h_numOccupied;
    }

    bool* containsMany(const T* keys, const size_t n) {
        T* d_keys;
        bool* d_output;
        bool* h_output;

        CUDA_CALL(cudaMallocHost(&h_output, n * sizeof(bool)));
        CUDA_CALL(cudaMalloc(&d_keys, n * sizeof(T)));
        CUDA_CALL(cudaMalloc(&d_output, n * sizeof(bool)));

        constexpr size_t numStreams = 4;
        const size_t chunkSize = SDIV(n, numStreams);
        cudaStream_t streams[numStreams];

        for (auto& stream : streams) {
            CUDA_CALL(cudaStreamCreate(&stream));
        }

        for (size_t i = 0; i < numStreams; ++i) {
            size_t offset = i * chunkSize;
            size_t currentChunkSize = std::min(chunkSize, n - offset);

            if (currentChunkSize > 0) {
                CUDA_CALL(cudaMemcpyAsync(
                    d_keys + offset,
                    keys + offset,
                    currentChunkSize * sizeof(T),
                    cudaMemcpyHostToDevice,
                    streams[i]
                ));

                size_t numBlocks = SDIV(currentChunkSize, blockSize);
                containsKernel<
                    T,
                    bitsPerTag,
                    bucketSize,
                    numBuckets,
                    maxProbes,
                    blockSize><<<numBlocks, blockSize, 0, streams[i]>>>(
                    d_keys + offset,
                    d_output + offset,
                    currentChunkSize,
                    get_device_view()
                );

                CUDA_CALL(cudaMemcpyAsync(
                    h_output + offset,
                    d_output + offset,
                    currentChunkSize * sizeof(bool),
                    cudaMemcpyDeviceToHost,
                    streams[i]
                ));
            }
        }

        for (auto& stream : streams) {
            CUDA_CALL(cudaStreamSynchronize(stream));
            CUDA_CALL(cudaStreamDestroy(stream));
        }

        CUDA_CALL(cudaFree(d_keys));
        CUDA_CALL(cudaFree(d_output));
        return h_output;
    }
    void clear() {
        CUDA_CALL(cudaMemset(d_buckets, 0, numBuckets * sizeof(Bucket)));
        CUDA_CALL(cudaMemset(d_locks, 0, numBuckets * sizeof(SpinLock)));
        CUDA_CALL(
            cudaMemset(d_numOccupied, 0, sizeof(cuda::std::atomic<size_t>))
        );
        h_numOccupied = 0;
    }

    float loadFactor() {
        CUDA_CALL(cudaMemcpy(
            &h_numOccupied,
            d_numOccupied,
            sizeof(size_t),
            cudaMemcpyDeviceToHost
        ));
        return static_cast<float>(h_numOccupied) / (numBuckets * bucketSize);
    }

    [[nodiscard]] double expectedFalsePositiveRate() const {
        return (2.0 * bucketSize) / (1ULL << bitsPerTag);
    }

    struct DeviceTableView {
        Bucket* d_buckets;
        SpinLock* d_locks;
        curandState* d_rand_states;
        size_t* d_numOccupied;

        __device__ bool tryInsertAtBucketWarp(size_t bucketIdx, TagType tag) {
            unsigned int lane_id = threadIdx.x & 31;
            bool success = false;

            if (lane_id == 0) {
                d_locks[bucketIdx].lock();
            }
            __syncwarp();

            int slot = d_buckets[bucketIdx].findEmptySlotWarp();

            if (slot != -1) {
                if (lane_id == slot) {
                    d_buckets[bucketIdx].insertAt(slot, tag);
                    atomicAdd(
                        reinterpret_cast<unsigned long long*>(d_numOccupied),
                        1ULL
                    );
                    success = true;
                }
            }

            bool was_successful = (__ballot_sync(0xffffffff, success) != 0);

            if (lane_id == 0) {
                d_locks[bucketIdx].unlock();
            }
            __syncwarp();

            return was_successful;
        }

        __device__ bool tryInsertAtBucket(size_t bucketIdx, TagType tag) {
            d_locks[bucketIdx].lock();

            int slot = d_buckets[bucketIdx].findEmptySlot();
            if (slot != -1) {
                d_buckets[bucketIdx].insertAt(slot, tag);
                atomicAdd(
                    reinterpret_cast<unsigned long long*>(d_numOccupied), 1ULL
                );
                d_locks[bucketIdx].unlock();
                return true;
            }
            d_locks[bucketIdx].unlock();
            return false;
        }

        __device__ bool insertWithEvictionWarp(TagType fp, size_t startBucket) {
            TagType currentFp = fp;
            size_t currentBucket = startBucket;
            unsigned int lane_id = threadIdx.x % 32;

            for (size_t evictions = 0; evictions < maxProbes; ++evictions) {
                if (tryInsertAtBucketWarp(currentBucket, currentFp)) {
                    return true;
                }

                TagType evictedFp = 0;
                if (lane_id == 0) {
                    d_locks[currentBucket].lock();

                    curandState* state = &d_rand_states[currentBucket];
                    auto evictSlot =
                        static_cast<size_t>(curand_uniform(state) * bucketSize);

                    evictedFp = d_buckets[currentBucket].getTagAt(evictSlot);
                    d_buckets[currentBucket].insertAt(evictSlot, currentFp);

                    d_locks[currentBucket].unlock();
                }

                currentFp = __shfl_sync(0xffffffff, evictedFp, 0);

                currentBucket = BucketsTableGpu::getAlternateBucket(
                    currentBucket, currentFp
                );
            }

            return false;
        }

        __device__ bool insertWithEviction(TagType fp, size_t startBucket) {
            TagType currentFp = fp;
            size_t currentBucket = startBucket;

            for (size_t evictions = 0; evictions < maxProbes; ++evictions) {
                d_locks[currentBucket].lock();

                int slot = d_buckets[currentBucket].findEmptySlot();
                if (slot != -1) {
                    d_buckets[currentBucket].insertAt(slot, currentFp);
                    atomicAdd(
                        reinterpret_cast<unsigned long long*>(d_numOccupied),
                        1ULL
                    );
                    d_locks[currentBucket].unlock();
                    return true;
                }

                curandState* state = &d_rand_states[currentBucket];
                auto evictSlot =
                    static_cast<size_t>(curand_uniform(state) * bucketSize);
                TagType evictedFp =
                    d_buckets[currentBucket].getTagAt(evictSlot);
                d_buckets[currentBucket].insertAt(evictSlot, currentFp);

                d_locks[currentBucket].unlock();

                currentFp = evictedFp;
                currentBucket = BucketsTableGpu::getAlternateBucket(
                    currentBucket, evictedFp
                );
            }
            return false;
        }

        __device__ bool insertWarp(const T& key) {
            auto [h1, h2, fp] = BucketsTableGpu::getCandidateBuckets(key);
            if (tryInsertAtBucketWarp(h1, fp) ||
                tryInsertAtBucketWarp(h2, fp)) {
                return true;
            }
            return insertWithEvictionWarp(fp, h1);
        }

        __device__ bool insert(const T& key) {
            auto [h1, h2, fp] = BucketsTableGpu::getCandidateBuckets(key);
            if (tryInsertAtBucket(h1, fp) || tryInsertAtBucket(h2, fp)) {
                return true;
            }
            return insertWithEviction(fp, h1);
        }

        __device__ bool containsWarp(const T& key) const {
            auto [h1, h2, fp] = BucketsTableGpu::getCandidateBuckets(key);
            return d_buckets[h1].containsWarp(fp) ||
                   d_buckets[h2].containsWarp(fp);
        }

        __device__ bool contains(const T& key) const {
            auto [h1, h2, fp] = BucketsTableGpu::getCandidateBuckets(key);
            return d_buckets[h1].contains(fp) || d_buckets[h2].contains(fp);
        }
    };

    DeviceTableView get_device_view() {
        return DeviceTableView{
            d_buckets, d_locks, d_rand_states, d_numOccupied
        };
    }
};

template <
    typename T,
    size_t bitsPerTag,
    size_t bucketSize,
    size_t numBuckets,
    size_t maxProbes,
    size_t blockSize>
__global__ void insertKernel(
    const T* keys,
    size_t n,
    typename BucketsTableGpu<
        T,
        bitsPerTag,
        bucketSize,
        numBuckets,
        maxProbes,
        blockSize>::DeviceTableView table_view
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // table_view.insert(keys[idx]);
        if (blockSize % 32 == 0) {
            table_view.insertWarp(keys[idx]);
        } else {
            table_view.insert(keys[idx]);
        }
    }
}

template <
    typename T,
    size_t bitsPerTag,
    size_t bucketSize,
    size_t numBuckets,
    size_t maxProbes,
    size_t blockSize>
__global__ void containsKernel(
    const T* keys,
    bool* output,
    size_t n,
    typename BucketsTableGpu<
        T,
        bitsPerTag,
        bucketSize,
        numBuckets,
        maxProbes,
        blockSize>::DeviceTableView table_view
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if (blockSize % 32 == 0) {
            output[idx] = table_view.containsWarp(keys[idx]);
        } else {
            output[idx] = table_view.contains(keys[idx]);
        }
    }
}