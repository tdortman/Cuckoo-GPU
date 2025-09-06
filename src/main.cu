#include <cuco/hash_functions.cuh>
#include <cuda/std/cstddef>
#include <iostream>

int main() {
    cuda::std::byte bytes[] = {
        cuda::std::byte('H'),
        cuda::std::byte('i'),
        cuda::std::byte('!'),
    };

    auto hf = cuco::default_hash_function<char>();

    auto hash_value = hf.compute_hash(bytes, sizeof(bytes));
    std::cout << "Hash: " << hash_value << std::endl;

    return 0;
}