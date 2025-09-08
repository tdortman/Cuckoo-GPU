#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/cstdlib>
#include <iostream>
#include <random>
#include "common.cuh"
#include "NaiveTable.cuh"


int main() {
    auto table = NaiveTable<uint32_t, 16>();

    const uint32_t input[] = {1, 2, 3, 4, 5};

    for (auto key : input) {
        table.insert(key);
    }

    const uint32_t test_keys[] = {1, 2, 3, 4, 5, 6, 7, 8};
    size_t size = sizeof(test_keys) / sizeof(test_keys[0]);

    auto mask = table.containsMany(test_keys, size);
    for (size_t i = 0; i < size; ++i) {
        std::cout << test_keys[i] << ": " << mask[i] << std::endl;
    }
}