#include <metal_stdlib>
using namespace metal;

kernel void run_twist(device const uint32_t* row_index,
                      device uint32_t* col_end,
                      device const uint32_t* col_start,
                      uint i [[thread_position_in_grid]]) {
    if (col_start[i] == col_end[i]) {
        return;
    }

    uint32_t low = row_index[col_end[i] - 1];
    col_end[low] = col_start[low];
}

kernel void count_inverse_low(device const uint32_t* row_index,
                              device const uint32_t* col_start,
                              device const uint32_t* col_end,
                              device atomic_uint* inverse_low,
                              device const uint32_t* n,
                              uint i [[thread_position_in_grid]]) {
    if (col_start[i] == col_end[i]) {
        return;
    }

    uint32_t low = row_index[col_end[i] - 1];
    while (true) {
        uint32_t cur_value =
            atomic_load_explicit(inverse_low + low, memory_order_relaxed);
        if (i > cur_value) {
            break;
        }
        if (atomic_compare_exchange_weak_explicit(inverse_low + low, &cur_value,
                                                  i, memory_order_relaxed,
                                                  memory_order_relaxed)) {
            break;
        }
    }
}

kernel void count_to_add(device const uint32_t* row_index,
                         device const uint32_t* col_start,
                         device const uint32_t* col_end,
                         device const uint32_t* inverse_low,
                         device uint32_t* to_add, device const uint32_t* n,
                         device atomic_uint* is_over,
                         uint i [[thread_position_in_grid]]) {
    if (col_start[i] == col_end[i]) {
        to_add[i] = *n;
        return;
    }

    uint32_t low = row_index[col_end[i] - 1];
    uint32_t low_inverse = inverse_low[low];
    if (low_inverse == i) {
        to_add[i] = *n;
    } else {
        to_add[i] = low_inverse;
        atomic_store_explicit(is_over, 0, memory_order_relaxed);
    }
}

kernel void copy_to_new_start(device const uint32_t* row_index,
                              device uint32_t* col_start,
                              device uint32_t* col_end,
                              device uint32_t* row_index_buffer,
                              device uint32_t* new_col_start,
                              uint i [[thread_position_in_grid]]) {
    uint32_t start = col_start[i];
    uint32_t end = col_end[i];
    uint32_t new_start = new_col_start[i];

    for (uint32_t j = 0; j < end - start; j++) {
        row_index_buffer[new_start + j] = row_index[start + j];
    }

    col_start[i] = new_start;
    col_end[i] = new_start + (end - start);
}

kernel void add_columns(device const uint32_t* col_start,
                        device uint32_t* col_end, device uint32_t* row_index,
                        device uint32_t* row_index_buffer,
                        device uint32_t* to_add, device const uint32_t* n,
                        device const uint32_t* row_index_size,
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
        available = *row_index_size - col_start[add_to];
    }

    if (len * 2 > available) {
        atomic_store_explicit(need_widen_buffer, 1, memory_order_relaxed);
    }
}
