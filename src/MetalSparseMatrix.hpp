#pragma once

#include <Metal/Metal.hpp>

#include "IMatrix.hpp"

class MetalSparseMatrix : public IMatrix {
public:
    MetalSparseMatrix(const std::string& file_path);

    std::vector<uint32_t> reduce(bool run_twist = true) override;

    size_t size() const override;

    ~MetalSparseMatrix();

private:
    void widenBuffer();

    void readFromFile(const std::string& file_path);

    void sendAddColumnsCommand(MTL::Buffer* to_add, MTL::Buffer* n_buffer, MTL::Buffer* need_widen_buffer);

    void runTwist();

    void sendComputeCommand();

    void encodeComputeCommand(MTL::ComputeCommandEncoder*);

    bool enoughSizeForIteration(uint32_t col) const;

    size_t n_;
    MTL::Buffer* col_start_;
    MTL::Buffer* col_end_;

    size_t row_index_size_;
    MTL::Buffer* row_index_;
    MTL::Buffer* row_index_buffer_;
    const uint32_t widen_coef_ = 2;

    NS::AutoreleasePool* mPool;
    MTL::Device* mDevice;
    MTL::ComputePipelineState* mComputeFunctionPSO;
    MTL::CommandQueue* mCommandQueue;
};
