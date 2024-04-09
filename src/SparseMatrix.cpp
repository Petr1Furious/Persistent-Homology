#include <fstream>
#include <sstream>

#include "SparseMatrix.hpp"

SparseMatrix::SparseMatrix(const std::string& file_path, bool is_parallel) : is_parallel_(is_parallel) {
    readFromFile(file_path);
}

size_t SparseMatrix::size() const {
    return n_;
}

void SparseMatrix::readFromFile(const std::string& file_path) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file");
    }

    std::string line;
    size_t i = 0;
    while (std::getline(file, line)) {
        if (i > n_) {
            if (!line.empty()) {
                throw std::runtime_error("File too long");
            }
            break;
        }

        std::istringstream iss(line);
        if (i == 0) {
            iss >> n_;
            col_start_.resize(n_, 0);
            col_end_.resize(n_, 0);
        } else {
            uint32_t count = 0;
            uint32_t index;
            while (iss >> index) {
                count++;
                row_index_.push_back(index);
            }

            if (i < n_) {
                col_start_[i] = col_start_[i - 1] + count;
                col_end_[i - 1] = col_start_[i];
            } else {
                col_end_[n_ - 1] = col_start_[n_ - 1] + count;
            }
        }
        i++;
    }

    std::vector<uint32_t> row_index_buffer;
    widenBuffer(row_index_buffer);
}

uint32_t SparseMatrix::getLow(uint32_t col_index) const {
    return col_start_[col_index] == col_end_[col_index] ? n_ : row_index_[col_end_[col_index] - 1];
}

bool SparseMatrix::enoughSizeForIteration(uint32_t col) const {
    uint32_t col_size = col_end_[col] - col_start_[col];
    uint32_t available_size = (col + 1 < n_) ? col_start_[col + 1] - col_start_[col] :
        static_cast<uint32_t>(row_index_.size()) - col_start_[col];
    return col_size * 2 <= available_size;
}

void SparseMatrix::addColumn(uint32_t add_to, uint32_t add_from, std::vector<uint32_t>& row_index_buffer) {
    uint32_t end1 = col_end_[add_to];
    uint32_t end2 = col_end_[add_from];

    uint32_t i = col_start_[add_to];
    uint32_t j = col_start_[add_from];
    uint32_t k = col_start_[add_to];
    while (i < end1 && j < end2) {
        if (row_index_[i] < row_index_[j]) {
            row_index_buffer[k] = row_index_[i];
            i++;
            k++;
        } else if (row_index_[i] > row_index_[j]) {
            row_index_buffer[k] = row_index_[j];
            j++;
            k++;
        } else {
            i++;
            j++;
        }
    }
    while (i < end1) {
        row_index_buffer[k] = row_index_[i];
        i++;
        k++;
    }
    while (j < end2) {
        row_index_buffer[k] = row_index_[j];
        j++;
        k++;
    }

    uint32_t available_size;
    if (add_to + 1 < n_) {
        available_size = col_start_[add_to + 1] - col_start_[add_to];
    } else {
        available_size = row_index_.size() - col_start_[add_to];
    }
    if (k - col_start_[add_to] > available_size) {
        throw std::runtime_error("Not enough space");
    }

    std::copy(row_index_buffer.begin() + col_start_[add_to], row_index_buffer.begin() + k, row_index_.begin() + col_start_[add_to]);
    col_end_[add_to] = k;

    if (!enoughSizeForIteration(add_to)) {
        need_widen_buffer_ = true;
    }
}

void SparseMatrix::widenBuffer(std::vector<uint32_t>& row_index_buffer) {
    size_t new_size = row_index_.size() * widen_coef_;
    row_index_buffer.resize(new_size, 0);
    row_index_.resize(row_index_buffer.size(), 0);

    for (int i = n_ - 1; i >= 0; i--) {
        int start = col_start_[i];
        int end = col_end_[i];

        int new_start = start * widen_coef_;
        int new_end = new_start + (end - start);
        std::copy(row_index_.begin() + start, row_index_.begin() + end, row_index_buffer.begin() + new_start);

        col_start_[i] = new_start;
        col_end_[i] = new_end;
    }

    std::swap(row_index_, row_index_buffer);
}

std::vector<uint32_t> SparseMatrix::getLowArray() const {
    std::vector<uint32_t> lowArray(n_);
    for (uint32_t i = 0; i < n_; i++) {
        lowArray[i] = getLow(i);
    }
    return lowArray;
}

void SparseMatrix::runTwist() {
    for (uint32_t i = 0; i < n_; i++) {
        uint32_t curLow = getLow(i);
        if (curLow != n_) {
            col_end_[curLow] = col_start_[curLow];
        }
    }
}

std::vector<uint32_t> SparseMatrix::reduce(bool run_twist) {
    if (run_twist) {
        runTwist();
    }

    std::vector<uint32_t> row_index_buffer(row_index_.size(), 0);
    std::vector<uint32_t> to_add(n_, n_);
    while (true) {
        std::vector<uint32_t> inverse_low(n_, n_);
        bool is_over = true;
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

        if (need_widen_buffer_) {
            widenBuffer(row_index_buffer);
        }
        need_widen_buffer_ = false;

        for (int i = n_ - 1; i >= 0; i--) {
            if (to_add[i] != n_) {
                addColumn(i, to_add[i], row_index_buffer);
                to_add[i] = n_;
            }
        }
    }

    return getLowArray();
}
