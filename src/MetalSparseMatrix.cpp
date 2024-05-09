#include <fstream>
#include <sstream>

#include <iostream>
#include <chrono>

#include "MetalSparseMatrix.hpp"

MetalSparseMatrix::MetalSparseMatrix(const std::string& file_path) {
    mPool = NS::AutoreleasePool::alloc()->init();
    mDevice = MTL::CreateSystemDefaultDevice();

    NS::Error* error;

    auto defaultLibrary = mDevice->newDefaultLibrary();
    if (!defaultLibrary) {
        throw std::runtime_error("failed to find the default library");
    }

    auto functionName = NS::String::string("add_columns", NS::ASCIIStringEncoding);
    auto computeFunction = defaultLibrary->newFunction(functionName);
    if(!computeFunction){
        throw std::runtime_error("failed to find the compute function");
    }

    mComputeFunctionPSO = mDevice->newComputePipelineState(computeFunction, &error);
    if (!computeFunction) {
        throw std::runtime_error("failed to create the pipeline state object");
    }
    
    mCommandQueue = mDevice->newCommandQueue();
    if (!mCommandQueue) {
        throw std::runtime_error("failed to find command queue");
    }

    readFromFile(file_path);
}

void MetalSparseMatrix::widenBuffer() {
    size_t new_size = row_index_size_ * widen_coef_;

    auto new_buffer = mDevice->newBuffer(new_size * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    auto new_buffer_ptr = (uint32_t*)new_buffer->contents();

    auto row_index_ptr = (uint32_t*)row_index_->contents();
    auto col_start_ptr = (uint32_t*)col_start_->contents();
    auto col_end_ptr = (uint32_t*)col_end_->contents();
    for (size_t i = 0; i < n_; i++) {
        uint32_t start = col_start_ptr[i];
        uint32_t end = col_end_ptr[i];

        uint32_t new_start = start * widen_coef_;
        uint32_t new_end = new_start + (end - start);
        std::memcpy(new_buffer_ptr + new_start, row_index_ptr + start, (end - start) * sizeof(uint32_t));

        col_start_ptr[i] = new_start;
        col_end_ptr[i] = new_end;
    }

    row_index_size_ = new_size;
    row_index_->release();
    row_index_ = new_buffer;

    if (row_index_buffer_) {
        row_index_buffer_->release();
    }
    row_index_buffer_ = mDevice->newBuffer(row_index_size_ * sizeof(uint32_t), MTL::ResourceStorageModeShared);
}

size_t MetalSparseMatrix::size() const {
    return n_;
}

void MetalSparseMatrix::readFromFile(const std::string& file_path) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file");
    }

    std::string line;
    size_t i = 0;
    std::vector<uint32_t> row_index;
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
            col_start_ = mDevice->newBuffer(n_ * sizeof(uint32_t), MTL::ResourceStorageModeShared);
            col_end_ = mDevice->newBuffer(n_ * sizeof(uint32_t), MTL::ResourceStorageModeShared);
        } else {
            uint32_t count = 0;
            uint32_t index;
            while (iss >> index) {
                count++;
                row_index.push_back(index);
            }

            auto col_start_ptr = (uint32_t*)col_start_->contents();
            auto col_end_ptr = (uint32_t*)col_end_->contents();
            if (i < n_) {
                col_start_ptr[i] = col_start_ptr[i - 1] + count;
                col_end_ptr[i - 1] = col_start_ptr[i];
            } else {
                col_end_ptr[n_ - 1] = col_start_ptr[n_ - 1] + count;
            }
        }
        i++;
    }

    row_index_size_ = row_index.size();
    row_index_ = mDevice->newBuffer(row_index_size_ * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    auto row_index_ptr = (uint32_t*)row_index_->contents();
    for (size_t i = 0; i < row_index.size(); i++) {
        row_index_ptr[i] = row_index[i];
    }
    row_index_buffer_ = mDevice->newBuffer(row_index_size_ * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    widenBuffer();
}

void MetalSparseMatrix::sendAddColumnsCommand(MTL::Buffer* to_add, MTL::Buffer* n_buffer, MTL::Buffer* need_widen_buffer) {
    MTL::CommandBuffer* commandBuffer = mCommandQueue->commandBuffer();
    assert(commandBuffer != nullptr);
    MTL::ComputeCommandEncoder* computeEncoder = commandBuffer->computeCommandEncoder();
    assert(computeEncoder != nullptr);

    computeEncoder->setComputePipelineState(mComputeFunctionPSO);
    computeEncoder->setBuffer(col_start_, 0, 0);
    computeEncoder->setBuffer(col_end_, 0, 1);
    computeEncoder->setBuffer(row_index_, 0, 2);
    computeEncoder->setBuffer(row_index_buffer_, 0, 3);
    computeEncoder->setBuffer(to_add, 0, 4);
    computeEncoder->setBuffer(n_buffer, 0, 5);
    computeEncoder->setBuffer(need_widen_buffer, 0, 6);

    MTL::Size gridSize = MTL::Size(n_, 1, 1);

    NS::UInteger threadGroupSize = mComputeFunctionPSO->maxTotalThreadsPerThreadgroup();
    if (threadGroupSize > n_) {
        threadGroupSize = n_;
    }
    MTL::Size threadgroupSize = MTL::Size(threadGroupSize, 1, 1);

    computeEncoder->dispatchThreads(gridSize, threadgroupSize);

    computeEncoder->endEncoding();
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
}

void MetalSparseMatrix::runTwist() {
    auto col_start_ptr = (uint32_t*)col_start_->contents();
    auto col_end_ptr = (uint32_t*)col_end_->contents();
    auto row_index_ptr = (uint32_t*)row_index_->contents();
    for (uint32_t i = 0; i < n_; i++) {
        uint32_t curLow = (col_start_ptr[i] == col_end_ptr[i] ? n_ : row_index_ptr[col_end_ptr[i] - 1]);
        if (curLow != n_) {
            col_end_ptr[curLow] = col_start_ptr[curLow];
        }
    }
}

bool MetalSparseMatrix::enoughSizeForIteration(uint32_t col) const {
    auto col_start_ptr = (uint32_t*)col_start_->contents();
    auto col_end_ptr = (uint32_t*)col_end_->contents();

    uint32_t col_size = col_end_ptr[col] - col_start_ptr[col];
    uint32_t available_size = (col + 1 < n_) ? col_start_ptr[col + 1] - col_start_ptr[col] :
        static_cast<uint32_t>(row_index_size_) - col_start_ptr[col];
    return col_size * 2 <= available_size;
}

std::vector<uint32_t> MetalSparseMatrix::reduce(bool run_twist) {
    if (run_twist) {
        runTwist();
    }

    MTL::Buffer* need_widen_buffer = mDevice->newBuffer(sizeof(uint32_t), MTL::ResourceStorageModeShared);
    uint32_t* need_widen_buffer_ptr = (uint32_t*)need_widen_buffer->contents();
    *need_widen_buffer_ptr = 0;

    MTL::Buffer* n_buffer = mDevice->newBuffer(sizeof(uint32_t), MTL::ResourceStorageModeShared);
    uint32_t* n_ptr = (uint32_t*)n_buffer->contents();
    *n_ptr = n_;

    MTL::Buffer* to_add = mDevice->newBuffer(n_ * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    uint32_t* to_add_ptr = (uint32_t*)to_add->contents();
    for (size_t i = 0; i < n_; i++) {
        to_add_ptr[i] = n_;
    }

    auto col_start_ptr = (uint32_t*)col_start_->contents();
    auto col_end_ptr = (uint32_t*)col_end_->contents();
    auto row_index_ptr = (uint32_t*)row_index_->contents();

    std::vector<uint32_t> inverse_low(n_, n_);
    while (true) {
        inverse_low.assign(n_, n_);
        bool is_over = true;
        auto start_time = std::chrono::high_resolution_clock::now();
        for (uint32_t i = 0; i < n_; i++) {
            uint32_t cur_low = (col_start_ptr[i] == col_end_ptr[i] ? n_ : row_index_ptr[col_end_ptr[i] - 1]);
            if (cur_low == n_) {
                continue;
            }
            uint32_t cur_inverse_low = inverse_low[cur_low];

            if (cur_inverse_low == n_) {
                inverse_low[cur_low] = i;
            } else {
                to_add_ptr[i] = cur_inverse_low;
            }
            if (to_add_ptr[i] != n_) {
                is_over = false;
            }
        }
        if (is_over) {
            break;
        }

        if (*need_widen_buffer_ptr == 1) {
            widenBuffer();
            col_start_ptr = (uint32_t*)col_start_->contents();
            col_end_ptr = (uint32_t*)col_end_->contents();
            row_index_ptr = (uint32_t*)row_index_->contents();
        }
        *need_widen_buffer_ptr = 0;

        std::cout << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start_time).count() << '\n';
        sendAddColumnsCommand(to_add, n_buffer, need_widen_buffer);
        std::cout << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start_time).count() << "\n\n";
    }

    to_add->release();
    n_buffer->release();

    std::vector<uint32_t> lowArray(n_);
    for (size_t i = 0; i < n_; i++) {
        lowArray[i] = (col_start_ptr[i] == col_end_ptr[i] ? n_ : row_index_ptr[col_end_ptr[i] - 1]);
    }
    return lowArray;
}

MetalSparseMatrix::~MetalSparseMatrix() {
    col_start_->release();
    col_end_->release();
    row_index_->release();
    row_index_buffer_->release();
    mPool->release();
}
