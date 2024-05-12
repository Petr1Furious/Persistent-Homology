#include "SparseMatrix.hpp"

#include <fstream>
#include <sstream>

SparseMatrix::SparseMatrix(const std::string& file_path)
    : SparseMatrixBase(file_path) {}

void SparseMatrix::widenBuffer(std::vector<uint32_t>& row_index_buffer,
                               const std::vector<uint32_t>& to_add) {
    std::vector<uint32_t> new_col_start(n_);

    uint32_t cur_col_start = 0;
    for (size_t i = 0; i < n_; i++) {
        new_col_start[i] = cur_col_start;

        uint32_t len = col_end_[i] - col_start_[i];
        uint32_t cur_to_add = to_add[i];
        if (len != 0) {
            cur_col_start += (i + 1 != n_ ? col_start_[i + 1]
                                          : (uint32_t)row_index_.size()) -
                             col_start_[i];

            if (cur_to_add != n_) {
                uint32_t to_add_len =
                    col_end_[cur_to_add] - col_start_[cur_to_add];
                cur_col_start += std::max((int)to_add_len - 2, (int)len);
            }
        }
    }

    uint32_t new_size = cur_col_start;

    row_index_buffer.resize(new_size, 0);

    for (size_t i = 0; i < n_; i++) {
        int start = col_start_[i];
        int end = col_end_[i];
        int new_start = new_col_start[i];

        std::copy(row_index_.begin() + start, row_index_.begin() + end,
                  row_index_buffer.begin() + new_start);

        col_start_[i] = new_start;
        col_end_[i] = new_start + (end - start);
    }

    row_index_.resize(new_size, 0);
    std::swap(row_index_, row_index_buffer);
}

std::vector<uint32_t> SparseMatrix::reduce(bool run_twist) {
    if (run_twist) {
        runTwist();
    }

    bool need_widen_buffer = false;
    std::vector<uint32_t> row_index_buffer(row_index_.size(), 0);
    std::vector<uint32_t> to_add(n_, n_);
    std::vector<uint32_t> inverse_low;
    while (true) {
        bool is_over = true;
        inverse_low.assign(n_, n_);
        for (uint32_t i = 0; i < n_; i++) {
            uint32_t cur_low = getLow(i);
            if (cur_low == n_) {
                continue;
            }
            uint32_t cur_inverse_low = inverse_low[cur_low];

            if (cur_inverse_low == n_) {
                inverse_low[cur_low] = i;
            } else {
                to_add[i] = cur_inverse_low;
            }
            if (to_add[i] != n_) {
                is_over = false;
            }

            if (!enoughSizeForIteration(i, to_add[i])) {
                need_widen_buffer = true;
            }
        }
        if (is_over) {
            break;
        }

        if (need_widen_buffer) {
            widenBuffer(row_index_buffer, to_add);
            need_widen_buffer = false;
        }

        for (size_t i = 0; i < n_; i++) {
            if (to_add[i] != n_) {
                addColumn(i, to_add[i], row_index_buffer);
                to_add[i] = n_;
            }
        }
    }

    return getLowArray();
}
