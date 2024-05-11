#include <fstream>
#include <sstream>
#include <iostream>

#include "SparseMatrixBase.hpp"

SparseMatrixBase::SparseMatrixBase(const std::string& file_path) {
    readFromFile(file_path);
}

size_t SparseMatrixBase::size() const {
    return n_;
}

void SparseMatrixBase::readFromFile(const std::string& file_path) {
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

uint32_t SparseMatrixBase::getLow(uint32_t col_index) const {
    return col_start_[col_index] == col_end_[col_index] ? n_ : row_index_[col_end_[col_index] - 1];
}

bool SparseMatrixBase::enoughSizeForIteration(uint32_t col) const {
    uint32_t col_size = col_end_[col] - col_start_[col];
    uint32_t available_size = (col + 1 < n_) ? col_start_[col + 1] - col_start_[col] :
        static_cast<uint32_t>(row_index_.size()) - col_start_[col];
    return col_size * 2 <= available_size;
}

void SparseMatrixBase::addColumn(uint32_t add_to, uint32_t add_from, std::vector<uint32_t>& row_index_buffer) {
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

    for (uint32_t i = col_start_[add_to]; i < k; i++) {
        row_index_[i] = row_index_buffer[i];
    }
    col_end_[add_to] = k;
}

void SparseMatrixBase::widenBuffer(std::vector<uint32_t>& row_index_buffer) {
    std::vector<uint32_t> new_col_start(n_);

    uint32_t cur_col_start = 0;
    for (size_t i = 0; i < n_; i++) {
        new_col_start[i] = cur_col_start;

        uint32_t len = col_end_[i] - col_start_[i];
        cur_col_start += widen_coef_ * len;
        if (len != 0) {
            cur_col_start += (i == n_ - 1 ? (uint32_t)row_index_.size() : col_start_[i + 1]) - col_end_[i];
        }
    }

    uint32_t new_size = cur_col_start;

    row_index_buffer.resize(new_size, 0);

    for (size_t i = 0; i < n_; i++) {
        int start = col_start_[i];
        int end = col_end_[i];
        int new_start = new_col_start[i];

        std::copy(row_index_.begin() + start, row_index_.begin() + end, row_index_buffer.begin() + new_start);

        col_start_[i] = new_start;
        col_end_[i] = new_start + (end - start);
    }

    row_index_.resize(new_size, 0);
    std::swap(row_index_, row_index_buffer);
}

std::vector<uint32_t> SparseMatrixBase::getLowArray() const {
    std::vector<uint32_t> lowArray(n_);
    for (uint32_t i = 0; i < n_; i++) {
        lowArray[i] = getLow(i);
    }
    return lowArray;
}

void SparseMatrixBase::runTwist() {
    for (uint32_t i = 0; i < n_; i++) {
        uint32_t curLow = getLow(i);
        if (curLow != n_) {
            col_end_[curLow] = col_start_[curLow];
        }
    }
}
