#include <metal_stdlib>
using namespace metal;

kernel void add_columns(device const uint32_t* col_start,
                        device uint32_t* col_end,
                        device uint32_t* row_index,
                        device uint32_t* row_index_buffer,
                        device uint32_t* to_add,
                        device const uint32_t* n,
                        device atomic_uint* need_widen_buffer,
                        uint add_to [[thread_position_in_grid]]) {
    if (to_add[add_to] == *n) {
        return;
    }

    uint32_t add_from = to_add[add_to];

    uint32_t end1 = col_end[add_to];
    uint32_t end2 = col_end[add_from];

    uint32_t i = col_start[add_to];
    uint32_t j = col_start[add_from];
    uint32_t k = col_start[add_to];
    while (i < end1 && j < end2) {
        if (row_index[i] < row_index[j]) {
            row_index_buffer[k] = row_index[i];
            i++;
            k++;
        } else if (row_index[i] > row_index[j]) {
            row_index_buffer[k] = row_index[j];
            j++;
            k++;
        } else {
            i++;
            j++;
        }
    }
    while (i < end1) {
        row_index_buffer[k] = row_index[i];
        i++;
        k++;
    }
    while (j < end2) {
        row_index_buffer[k] = row_index[j];
        j++;
        k++;
    }

    for (uint32_t i = col_start[add_to]; i < k; i++) {
        row_index[i] = row_index_buffer[i];
    }
    col_end[add_to] = k;

    to_add[add_to] = *n;

    uint32_t len = col_end[add_to] - col_start[add_to];
    uint32_t available;
    if (add_to + 1 != *n) {
        available = col_start[add_to + 1] - col_start[add_to];
    } else {
        available = *n - col_start[add_to];
    }

    if (len * 2 > available) {
        atomic_store_explicit(need_widen_buffer, 1, memory_order_relaxed);
    }
}
