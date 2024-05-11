#include "SparseMatrix.hpp"

#include <fstream>
#include <sstream>

SparseMatrix::SparseMatrix(const std::string& file_path)
    : SparseMatrixBase(file_path) {}

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
        }
        if (is_over) {
            break;
        }

        if (need_widen_buffer) {
            widenBuffer(row_index_buffer);
        }
        need_widen_buffer = false;

        for (size_t i = 0; i < n_; i++) {
            if (to_add[i] != n_) {
                addColumn(i, to_add[i], row_index_buffer);
                if (!enoughSizeForIteration(i)) {
                    need_widen_buffer = true;
                }
                to_add[i] = n_;
            }
        }
    }

    return getLowArray();
}