#pragma once

#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>
#include <thrust/transform.h>
#include <cstdint>
#include <hash_strategies.cuh>
#include <limits>

template <typename ConfigType>
std::pair<size_t, size_t> calculateCapacityAndSize(size_t capacity, double loadFactor) {
    return {capacity, capacity * loadFactor};
}

/**
 * @brief Generate random keys on the GPU
 *
 * @tparam T The key type
 * @param d_keys Device vector to fill with random keys
 * @param max Maximum value for generated keys (default: max value of type T)
 * @param seed Random seed (default: 42)
 */
template <typename T>
void generateKeysGPU(
    thrust::device_vector<T>& d_keys,
    T max = std::numeric_limits<T>::max(),
    unsigned int seed = 42
) {
    size_t n = d_keys.size();
    thrust::transform(
        thrust::counting_iterator<size_t>(0),
        thrust::counting_iterator<size_t>(n),
        d_keys.begin(),
        [max, seed] __device__(size_t idx) {
            thrust::default_random_engine rng(seed);
            thrust::uniform_int_distribution<T> dist(1, max);
            rng.discard(idx);
            return dist(rng);
        }
    );
}
