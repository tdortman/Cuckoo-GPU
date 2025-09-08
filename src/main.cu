#include <cstdint>
#include <cuco/hash_functions.cuh>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/cstdlib>
#include <iostream>
#include <random>
#include "common.cuh"
#include "NaiveTable.cuh"

cuda::std::byte* rand_string(size_t size, size_t seed = 0) {
    std::mt19937 mt(seed);
    std::uniform_int_distribution<uint8_t> dist(48, 122);

    cuda::std::byte* string;
    CUDA_CALL(cudaMallocHost(&string, size));

    for (size_t i = 0; i < size; ++i) {
        string[i] = static_cast<cuda::std::byte>(dist(mt));
    }

    return string;
}

__global__ void kernel(
    cuco::default_hash_function<char> hf,
    cuda::std::byte** strings,
    size_t* sizes,
    uint32_t* hash_values,
    uint32_t n
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) {
        return;
    }

    auto hash_value = hf.compute_hash(strings[idx], sizes[idx]);
    hash_values[idx] = hash_value;
}

int main() {
    auto table = NaiveTable<uint32_t, 16>();

    const uint32_t input[] = {1, 2, 3, 4, 5};

    for (auto key : input) {
        table.insert(key);
    }

    size_t size = sizeof(input) / sizeof(input[0]);

    auto mask = table.containsMany(input, size);

    for (size_t i = 0; i < size; ++i) {
        std::cout << input[i] << ": " << mask[i] << std::endl;
    }
}