#pragma once

#include "SparseMatrixBase.hpp"

#include <vector>
#include <string>

class SparseMatrix : public SparseMatrixBase {
public:
    SparseMatrix(const std::string& file_path);

    std::vector<uint32_t> reduce(bool run_twist = true) override;
};
