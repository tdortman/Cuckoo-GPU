#pragma once

#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <cstddef>
#include <cstdint>
#include <gqf.cuh>
#include <gqf_int.cuh>

inline size_t getQFSizeHost(QF* d_qf) {
    QF h_qf;
    cudaMemcpy(&h_qf, d_qf, sizeof(QF), cudaMemcpyDeviceToHost);

    qfmetadata h_metadata;
    cudaMemcpy(&h_metadata, h_qf.metadata, sizeof(qfmetadata), cudaMemcpyDeviceToHost);

    return h_metadata.total_size_in_bytes;
}

inline void convertGQFResults(thrust::device_vector<uint64_t>& d_results) {
    thrust::device_ptr<uint64_t> d_resultsPtr(d_results.data().get());
    thrust::transform(
        d_resultsPtr, d_resultsPtr + d_results.size(), d_resultsPtr, [] __device__(uint64_t val) {
            return val > 0;
        }
    );
}
