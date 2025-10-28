#include <cstddef>

/**
 * XXHash_64 implementation from
 * https://github.com/Cyan4973/xxHash
 * -----------------------------------------------------------------------------
 * xxHash - Extremely Fast Hash algorithm
 * Header File
 * Copyright (C) 2012-2021 Yann Collet
 *
 * BSD 2-Clause License (https://www.opensource.org/licenses/bsd-license.php)
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 *    * Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *    * Redistributions in binary form must reproduce the above
 *      copyright notice, this list of conditions and the following disclaimer
 *      in the documentation and/or other materials provided with the
 *      distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * You can contact the author at:
 *   - xxHash homepage: https://www.xxhash.com
 *   - xxHash source repository: https://github.com/Cyan4973/xxHash
 */
namespace xxhash {

constexpr uint64_t PRIME64_1 = 11400714785074694791ULL;
constexpr uint64_t PRIME64_2 = 14029467366897019727ULL;
constexpr uint64_t PRIME64_3 = 1609587929392839161ULL;
constexpr uint64_t PRIME64_4 = 9650029242287828579ULL;
constexpr uint64_t PRIME64_5 = 2870177450012600261ULL;

__host__ __device__ __forceinline__ uint64_t rotl64(uint64_t x, int8_t r) {
    return (x << r) | (x >> (64 - r));
}

template <typename T>
__host__ __device__ __forceinline__ T load_chunk(const uint8_t* data, size_t index) {
    T chunk;
    memcpy(&chunk, data + index * sizeof(T), sizeof(T));
    return chunk;
}

__host__ __device__ __forceinline__ uint64_t finalize(uint64_t h) {
    h ^= h >> 33;
    h *= PRIME64_2;
    h ^= h >> 29;
    h *= PRIME64_3;
    h ^= h >> 32;
    return h;
}

template <typename T>
__host__ __device__ inline uint64_t xxhash64(const T& key, uint64_t seed = 0) {
    const auto* bytes = reinterpret_cast<const uint8_t*>(&key);
    size_t size = sizeof(T);
    size_t offset = 0;
    uint64_t h64;

    // Process 32-byte chunks
    if (size >= 32) {
        size_t limit = size - 32;
        uint64_t v1 = seed + PRIME64_1 + PRIME64_2;
        uint64_t v2 = seed + PRIME64_2;
        uint64_t v3 = seed;
        uint64_t v4 = seed - PRIME64_1;

        do {
            const size_t pipeline_offset = offset / 8;
            v1 += load_chunk<uint64_t>(bytes, pipeline_offset + 0) * PRIME64_2;
            v1 = rotl64(v1, 31);
            v1 *= PRIME64_1;
            v2 += load_chunk<uint64_t>(bytes, pipeline_offset + 1) * PRIME64_2;
            v2 = rotl64(v2, 31);
            v2 *= PRIME64_1;
            v3 += load_chunk<uint64_t>(bytes, pipeline_offset + 2) * PRIME64_2;
            v3 = rotl64(v3, 31);
            v3 *= PRIME64_1;
            v4 += load_chunk<uint64_t>(bytes, pipeline_offset + 3) * PRIME64_2;
            v4 = rotl64(v4, 31);
            v4 *= PRIME64_1;
            offset += 32;
        } while (offset <= limit);

        h64 = rotl64(v1, 1) + rotl64(v2, 7) + rotl64(v3, 12) + rotl64(v4, 18);

        v1 *= PRIME64_2;
        v1 = rotl64(v1, 31);
        v1 *= PRIME64_1;
        h64 ^= v1;
        h64 = h64 * PRIME64_1 + PRIME64_4;

        v2 *= PRIME64_2;
        v2 = rotl64(v2, 31);
        v2 *= PRIME64_1;
        h64 ^= v2;
        h64 = h64 * PRIME64_1 + PRIME64_4;

        v3 *= PRIME64_2;
        v3 = rotl64(v3, 31);
        v3 *= PRIME64_1;
        h64 ^= v3;
        h64 = h64 * PRIME64_1 + PRIME64_4;

        v4 *= PRIME64_2;
        v4 = rotl64(v4, 31);
        v4 *= PRIME64_1;
        h64 ^= v4;
        h64 = h64 * PRIME64_1 + PRIME64_4;
    } else {
        h64 = seed + PRIME64_5;
    }

    h64 += size;

    // Process remaining 8-byte chunks
    if ((size % 32) >= 8) {
        for (; offset <= size - 8; offset += 8) {
            uint64_t k1 = load_chunk<uint64_t>(bytes, offset / 8) * PRIME64_2;
            k1 = rotl64(k1, 31) * PRIME64_1;
            h64 ^= k1;
            h64 = rotl64(h64, 27) * PRIME64_1 + PRIME64_4;
        }
    }

    // Process remaining 4-byte chunks
    if ((size % 8) >= 4) {
        for (; offset <= size - 4; offset += 4) {
            h64 ^= (load_chunk<uint32_t>(bytes, offset / 4) & 0xffffffffULL) * PRIME64_1;
            h64 = rotl64(h64, 23) * PRIME64_2 + PRIME64_3;
        }
    }

    // Process remaining bytes
    if (size % 4) {
        while (offset < size) {
            h64 ^= (bytes[offset] & 0xff) * PRIME64_5;
            h64 = rotl64(h64, 11) * PRIME64_1;
            ++offset;
        }
    }

    return finalize(h64);
}

}  // namespace xxhash