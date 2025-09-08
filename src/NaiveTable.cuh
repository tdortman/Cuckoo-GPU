#include <atomic>
#include <cuco/hash_functions.cuh>
#include <cuda/std/atomic>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/functional>
#include "common.cuh"

constexpr bool powerOfTwo(size_t n) {
    return n != 0 && (n & (n - 1)) == 0;
}

template <
    typename T,
    size_t bitsPerTag,
    size_t maxProbes,
    size_t blockSize,
    size_t initialNumSlots,
    float maxLoadFactor>
class NaiveTable;

template <
    typename T,
    size_t bitsPerTag,
    size_t maxProbes,
    size_t blockSize,
    size_t initialNumSlots,
    float maxLoadFactor>
__global__ void containsKernel(
    const T* keys,
    bool* output,
    size_t n,
    typename NaiveTable<
        T,
        bitsPerTag,
        maxProbes,
        blockSize,
        initialNumSlots,
        maxLoadFactor>::DeviceTableView table_view
);

template <
    typename T,
    size_t bitsPerTag,
    size_t maxProbes = 500,
    size_t blockSize = 256,
    size_t initialNumSlots = 256,
    float maxLoadFactor = 0.75f>
class NaiveTable {
    static_assert(bitsPerTag <= 64, "The tag cannot be larger than 64 bits");
    static_assert(
        bitsPerTag <= 8 * sizeof(T),
        "The tag cannot be larger than the size of the type"
    );

    using TagType = typename std::conditional<
        bitsPerTag <= 8,
        uint8_t,
        typename std::conditional<
            bitsPerTag <= 16,
            uint16_t,
            typename std::conditional<bitsPerTag <= 32, uint32_t, uint64_t>::
                type>::type>::type;

    struct Slot {
        TagType tag;
        T value;
    };

    static constexpr TagType EMPTY = 0;

    static constexpr size_t tagMask = (1ULL << bitsPerTag) - 1;

    Slot* h_slots;
    Slot* d_slots;

    size_t numEvictions = 0;
    size_t numSlots;
    cuda::std::atomic_size_t numOccupied = 0;

    using HashFn = uint32_t (*)(const T&, size_t);
    HashFn hashFns[2] = {nullptr, nullptr};

    static __host__ __device__ uint32_t murmur_hash(const T& key, size_t size) {
        auto bytes = reinterpret_cast<const cuda::std::byte*>(&key);
        cuco::murmurhash3_32<T> hasher;
        return hasher.compute_hash(bytes, size);
    }

    static __host__ __device__ uint32_t xxhash_hash(const T& key, size_t size) {
        auto bytes = reinterpret_cast<const cuda::std::byte*>(&key);
        cuco::xxhash_32<T> hasher;
        return hasher.compute_hash(bytes, size);
    }

    __device__ __host__ bool try_insert(TagType tag, const T& value, size_t i) {
        if (d_slots[i].tag == EMPTY) {
            d_slots[i].tag = tag;
            d_slots[i].value = value;
            numOccupied++;
            return true;
        }
        return false;
    }

    __host__ void rehash() {
        size_t oldNumSlots = numSlots;
        numSlots *= 2;

        Slot* new_h_slots;
        CUDA_CALL(cudaMallocHost(&new_h_slots, numSlots * sizeof(Slot)));
        CUDA_CALL(cudaMemset(new_h_slots, 0, numSlots * sizeof(Slot)));

        auto* temp_slots = new Slot[oldNumSlots];
        CUDA_CALL(cudaMemcpy(
            temp_slots,
            h_slots,
            oldNumSlots * sizeof(Slot),
            cudaMemcpyDeviceToHost
        ));

        CUDA_CALL(cudaFreeHost(h_slots));
        CUDA_CALL(cudaFree(d_slots));

        h_slots = new_h_slots;
        CUDA_CALL(cudaMalloc(&d_slots, numSlots * sizeof(Slot)));

        for (size_t i = 0; i < oldNumSlots; ++i) {
            if (temp_slots[i].tag != EMPTY) {
                T currentValue = temp_slots[i].value;
                TagType currentTag = temp_slots[i].tag;

                size_t evictions = 0;

                do {
                    auto i1 =
                        murmur_hash(currentValue, sizeof(T)) & (numSlots - 1);
                    auto i2 = i1 ^ (xxhash_hash(currentValue, sizeof(T)) &
                                    (numSlots - 1));

                    if (try_insert(currentTag, currentValue, i1) ||
                        try_insert(currentTag, currentValue, i2)) {
                        break;
                    }

                    Slot evicted = h_slots[i1];
                    h_slots[i1].tag = currentTag;
                    h_slots[i1].value = currentValue;

                    currentTag = evicted.tag;
                    currentValue = evicted.value;

                    evictions++;
                } while (evictions < maxProbes);

                // We have a BIG problem as there are A LOT of collisions
                // Go fix your hash functions I suppose
                throw std::runtime_error(
                    "Rehashing failed, too many evictions"
                );
            }

            CUDA_CALL(cudaMemcpy(
                d_slots,
                h_slots,
                numSlots * sizeof(Slot),
                cudaMemcpyHostToDevice
            ));

            delete[] temp_slots;
            numEvictions = 0;
        }
    };

   public:
    explicit NaiveTable(size_t numSlots = initialNumSlots)
        : numSlots(numSlots) {
        CUDA_CALL(cudaMallocHost(&h_slots, numSlots * sizeof(Slot)));
        CUDA_CALL(cudaMemset(h_slots, 0, numSlots * sizeof(Slot)));
        CUDA_CALL(cudaMalloc(&d_slots, numSlots * sizeof(Slot)));
        CUDA_CALL(cudaMemcpy(
            d_slots, h_slots, numSlots * sizeof(Slot), cudaMemcpyHostToDevice
        ));

        hashFns[0] = &murmur_hash;
        hashFns[1] = &xxhash_hash;
    }

    NaiveTable(T* items, size_t n, size_t numSlots = initialNumSlots)
        : numSlots(numSlots) {
        CUDA_CALL(cudaMallocHost(&h_slots, numSlots * sizeof(Slot)));
        CUDA_CALL(cudaMemset(h_slots, 0, numSlots * sizeof(Slot)));
        CUDA_CALL(cudaMalloc(&d_slots, numSlots * sizeof(Slot)));
        CUDA_CALL(cudaMemcpy(
            d_slots, h_slots, numSlots * sizeof(Slot), cudaMemcpyHostToDevice
        ));

        for (size_t i = 0; i < n; ++i) {
            insert(items[i]);
        }
    };

    __host__ void insert(const T& key) {
        while (true) {
            auto tag =
                static_cast<TagType>(murmur_hash(key, sizeof(T)) & tagMask);
            auto i1 = murmur_hash(key, sizeof(T)) & (numSlots - 1);
            auto i2 = i1 ^ (xxhash_hash(key, sizeof(T)) & (numSlots - 1));

            T currentValue = key;
            TagType currentTag = tag;

            do {
                if (try_insert(currentTag, currentValue, i1) ||
                    try_insert(currentTag, currentValue, i2)) {
                    break;
                }

                Slot evicted = d_slots[i1];
                d_slots[i1].tag = currentTag;
                d_slots[i1].value = currentValue;

                currentTag = evicted.tag;
                currentValue = evicted.value;

                i1 = murmur_hash(currentValue, sizeof(T)) & (numSlots - 1);
                i2 = i1 ^
                     (xxhash_hash(currentValue, sizeof(T)) & (numSlots - 1));

                numEvictions++;
            } while (numEvictions < maxProbes);

            rehash();
        }
    };

    void insertMany(const T* keys, bool* output, size_t n) {
        size_t numBlocks = (n + blockSize - 1) / blockSize;
    }

    const bool* containsMany(const T* keys, const size_t n) {
        bool* d_output;
        bool* h_output;
        CUDA_CALL(cudaMallocHost(&h_output, n * sizeof(bool)));

        CUDA_CALL(cudaMalloc(&d_output, n * sizeof(bool)));
        size_t numBlocks = (n + blockSize - 1) / blockSize;
        containsKernel<
            T,
            bitsPerTag,
            maxProbes,
            blockSize,
            initialNumSlots,
            maxLoadFactor><<<numBlocks, blockSize>>>(
            keys, d_output, n, this->get_device_view()
        );

        CUDA_CALL(cudaDeviceSynchronize());
        CUDA_CALL(cudaMemcpy(
            h_output, d_output, n * sizeof(bool), cudaMemcpyDeviceToHost
        ));
        CUDA_CALL(cudaFree(d_output));

        return h_output;
    };

    __device__ __host__ float loadFactor() const {
        return static_cast<float>(numOccupied.load()) / numSlots;
    };

    struct DeviceTableView {
        Slot* d_slots;
        size_t numSlots;
        TagType tagMask;
        HashFn hashFns[2];

        __device__ bool contains(const T& key) const {
            auto tag =
                static_cast<TagType>(hashFns[0](key, sizeof(T)) & tagMask);
            auto i1 = hashFns[0](key, sizeof(T)) & (numSlots - 1);
            auto i2 = i1 ^ (hashFns[1](key, sizeof(T)) & (numSlots - 1));

            return (d_slots[i1].tag == tag) || (d_slots[i2].tag == tag);
        };
    };

    __host__ DeviceTableView get_device_view() {
        return DeviceTableView{
            d_slots,
            numSlots,
            static_cast<TagType>(tagMask),
            {hashFns[0], hashFns[1]}
        };
    }

    ~NaiveTable() {
        if (h_slots) {
            CUDA_CALL(cudaFreeHost(h_slots));
        }
        if (d_slots) {
            CUDA_CALL(cudaFree(d_slots));
        }
    }
};

template <
    typename T,
    size_t bitsPerTag,
    size_t maxProbes = 500,
    size_t blockSize = 256,
    size_t initialNumSlots = 256,
    float maxLoadFactor = 0.75f>
__global__ void containsKernel(
    const T* keys,
    bool* output,
    size_t n,
    typename NaiveTable<
        T,
        bitsPerTag,
        maxProbes,
        blockSize,
        initialNumSlots,
        maxLoadFactor>::DeviceTableView table_view
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = table_view.contains(keys[idx]);
    }
}