#include <metal_stdlib>

using namespace metal;

kernel void compute_low(device const bool* matrix,
                        device const uint* matrixSize,
                        device int64_t* low,
                        uint index [[thread_position_in_grid]]) {
    for (int64_t i = *matrixSize - 1; i >= 0; i--) {
        if (matrix[(*matrixSize) * index + (uint64_t)i]) {
            low[index] = i;
            return;
        }
    }
    low[index] = -1;
}

kernel void reduce_matrix(device bool* matrix,
                          device const uint* matrixSize,
                          device const int64_t* lowClass,
                          uint index [[thread_position_in_grid]]) {
    if (lowClass[index] != -1 && (uint)lowClass[index] != index) {
        uint curClass = (uint64_t)lowClass[index];
        for (uint64_t i = 0; i < *matrixSize; i++) {
            matrix[*matrixSize * index + i] = (matrix[*matrixSize * index + i] != matrix[*matrixSize * curClass + i]);
        }
    }
}
