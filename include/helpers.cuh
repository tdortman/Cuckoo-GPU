#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>

/**
 * @brief Checks if a number is a power of two.
 * @param n Number to check.
 * @return true if n is a power of two, false otherwise.
 */
constexpr bool powerOfTwo(size_t n) {
    return n != 0 && (n & (n - 1)) == 0;
}

/**
 * @brief Calculates the global thread ID in a 1D grid.
 * @return uint32_t Global thread ID.
 */
__host__ __device__ __forceinline__ uint32_t globalThreadId() {
    return blockIdx.x * blockDim.x + threadIdx.x;
}

/**
 * @brief Calculates the next power of two greater than or equal to n.
 * @param n Input number.
 * @return size_t Next power of two.
 */
constexpr size_t nextPowerOfTwo(size_t n) {
    if (powerOfTwo(n))
        return n;

    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n |= n >> 32;
    n++;

    return n;
}

/**
 * @brief Counts the number of non-zero elements in an array.
 * @tparam T Type of elements.
 * @param data Pointer to the array.
 * @param n Number of elements.
 * @return size_t Number of non-zero elements.
 */
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

/**
 * @brief Returns a bitmask indicating which slots in a packed word are zero.
 *
 * Uses SWAR (SIMD Within A Register) to check multiple items in parallel.
 * See https://graphics.stanford.edu/~seander/bithacks.html#ZeroInWord
 *
 * The high bit of each slot that is zero will be set in the result.
 *
 * @tparam T The type of the individual items (uint8_t, uint16_t, or uint32_t)
 * @param v The packed 64-bit integer
 * @return A bitmask with the high bit of each zero slot set
 */
template <typename T>
__host__ __device__ __forceinline__ constexpr uint64_t getZeroMask(uint64_t v) {
    if constexpr (sizeof(T) == 1) {
        return (v - 0x0101010101010101ULL) & ~v & 0x8080808080808080ULL;
    } else if constexpr (sizeof(T) == 2) {
        return (v - 0x0001000100010001ULL) & ~v & 0x8000800080008000ULL;
    } else if constexpr (sizeof(T) == 4) {
        return (v - 0x0000000100000001ULL) & ~v & 0x8000000080000000ULL;
    } else {
        return 0;
    }
}

/**
 * @brief Checks if a packed word contains a zero byte/word.
 *
 * @tparam T The type of the individual items (uint8_t, uint16_t, or uint32_t)
 * @param v The packed 64-bit integer
 * @return true if any of the items in v are zero
 */
template <typename T>
__host__ __device__ __forceinline__ constexpr bool hasZero(uint64_t v) {
    return getZeroMask<T>(v) != 0;
}

/**
 * @brief Replicates a tag value across all slots in a 64-bit word.
 *
 * @tparam T The type of the tag (uint8_t, uint16_t, or uint32_t)
 * @param tag The tag value to replicate
 * @return A 64-bit word with the tag replicated in every slot
 */
template <typename T>
__host__ __device__ __forceinline__ constexpr uint64_t replicateTag(T tag) {
    if constexpr (sizeof(T) == 1) {
        return static_cast<uint64_t>(tag) * 0x0101010101010101ULL;
    } else if constexpr (sizeof(T) == 2) {
        return static_cast<uint64_t>(tag) * 0x0001000100010001ULL;
    } else if constexpr (sizeof(T) == 4) {
        return static_cast<uint64_t>(tag) * 0x0000000100000001ULL;
    } else {
        return tag;
    }
}

/**
 * @brief Integer division with rounding up (ceiling).
 */
#define SDIV(x, y) (((x) + (y) - 1) / (y))

/**
 * @brief Macro for checking CUDA errors.
 * Prints error message and exits if an error occurs.
 */
#define CUDA_CALL(err)                                                      \
    do {                                                                    \
        cudaError_t err_ = (err);                                           \
        if (err_ == cudaSuccess) [[likely]] {                               \
            break;                                                          \
        }                                                                   \
        printf("%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(err_)); \
        exit(err_);                                                         \
    } while (0)

/**
 * @brief Calculates the maximum occupancy grid size for a kernel.
 *
 * @tparam Kernel Type of the kernel function.
 * @param blockSize Block size (threads per block).
 * @param kernel The kernel function.
 * @param dynamicSMemSize Dynamic shared memory size per block.
 * @return size_t The calculated grid size (number of blocks).
 */
template <typename Kernel>
constexpr size_t maxOccupancyGridSize(int32_t blockSize, Kernel kernel, size_t dynamicSMemSize) {
    int device = 0;
    cudaGetDevice(&device);

    int numSM = -1;
    cudaDeviceGetAttribute(&numSM, cudaDevAttrMultiProcessorCount, device);

    int maxActiveBlocksPerSM{};
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxActiveBlocksPerSM, kernel, blockSize, dynamicSMemSize
    );

    return maxActiveBlocksPerSM * numSM;
}
