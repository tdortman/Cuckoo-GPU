#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>

constexpr bool powerOfTwo(size_t n) {
    return n != 0 && (n & (n - 1)) == 0;
}

__host__ __device__ __forceinline__ uint32_t globalThreadId() {
    return blockIdx.x * blockDim.x + threadIdx.x;
}

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

#define SDIV(x, y) (((x) + (y) - 1) / (y))

#define CUDA_CALL(err)                                                      \
    do {                                                                    \
        cudaError_t err_ = (err);                                           \
        if (err_ == cudaSuccess) [[likely]] {                               \
            break;                                                          \
        }                                                                   \
        printf("%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(err_)); \
        exit(err_);                                                         \
    } while (0)

template <typename Kernel>
constexpr auto maxOccupancyGridSize(int32_t blockSize, Kernel kernel, size_t dynamicSMemSize) {
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
