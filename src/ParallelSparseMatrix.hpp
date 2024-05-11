#pragma once

#include "SparseMatrixBase.hpp"

class ParallelSparseMatrix : public SparseMatrixBase {
    public:
        ParallelSparseMatrix(const std::string& file_path);

        std::vector<uint32_t> reduce(bool run_twist = true) override;
};
