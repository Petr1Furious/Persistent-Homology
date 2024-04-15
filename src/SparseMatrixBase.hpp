#pragma once

#include "IMatrix.hpp"

#include <vector>
#include <string>

class SparseMatrixBase : public IMatrix {
public:
    SparseMatrixBase(const std::string& file_path);

    size_t size() const override;

protected:
    void readFromFile(const std::string& file_path);

    uint32_t getLow(uint32_t col_index) const;
    
    bool enoughSizeForIteration(uint32_t col) const;

    void addColumn(uint32_t col1, uint32_t col2, std::vector<uint32_t>& row_index_buffer);

    void widenBuffer(std::vector<uint32_t>& row_index_buffer);

    void runTwist();

    std::vector<uint32_t> getLowArray() const;

    size_t n_;
    std::vector<uint32_t> row_index_;
    std::vector<uint32_t> col_start_;
    std::vector<uint32_t> col_end_;
    const uint32_t widen_coef_ = 2;
};
