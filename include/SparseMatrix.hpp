#pragma once

#include <string>
#include <vector>

#include "SparseMatrixBase.hpp"

class SparseMatrix : public SparseMatrixBase {
    public:
        SparseMatrix(const std::string& file_path);

        std::vector<uint32_t> reduce(bool run_twist = true) override;

    private:
        void widenBuffer(std::vector<uint32_t>& row_index_buffer,
                         const std::vector<uint32_t>& to_add);
};
