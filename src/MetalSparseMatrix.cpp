#include <fstream>
#include <sstream>

#include "MetalSparseMatrix.hpp"

MetalSparseMatrix::MetalSparseMatrix(const std::string& file_path) {
    m_pool = NS::AutoreleasePool::alloc()->init();
    m_device = MTL::CreateSystemDefaultDevice();

    NS::Error* error;

    auto default_library = m_device->newDefaultLibrary();
    if (!default_library) {
        throw std::runtime_error("failed to find the default library");
    }

    auto function_name = NS::String::string("count_inverse_low", NS::ASCIIStringEncoding);
    auto count_inverse_low_function = default_library->newFunction(function_name);
    if (!count_inverse_low_function) {
        throw std::runtime_error("failed to find the count_inverse_low function");
    }
    count_inverse_low_ps = m_device->newComputePipelineState(count_inverse_low_function, &error);
    if (!count_inverse_low_ps) {
        throw std::runtime_error("failed to create the count_inverse_low pipeline state object");
    }

    function_name = NS::String::string("count_to_add", NS::ASCIIStringEncoding);
    auto count_to_add_function = default_library->newFunction(function_name);
    if (!count_to_add_function) {
        throw std::runtime_error("failed to find the count_to_add function");
    }
    count_to_add_ps = m_device->newComputePipelineState(count_to_add_function, &error);
    if (!count_to_add_ps) {
        throw std::runtime_error("failed to create the count_to_add pipeline state object");
    }

    function_name = NS::String::string("copy_to_row_index_buffer", NS::ASCIIStringEncoding);
    auto copy_to_row_index_buffer_function = default_library->newFunction(function_name);
    if (!copy_to_row_index_buffer_function) {
        throw std::runtime_error("failed to find the copy_to_row_index_buffer function");
    }
    copy_to_row_index_buffer_ps = m_device->newComputePipelineState(copy_to_row_index_buffer_function, &error);
    if (!copy_to_row_index_buffer_ps) {
        throw std::runtime_error("failed to create the copy_to_row_index_buffer pipeline state object");
    }

    function_name = NS::String::string("add_columns", NS::ASCIIStringEncoding);
    auto add_columns_function = default_library->newFunction(function_name);
    if (!add_columns_function) {
        throw std::runtime_error("failed to find the add_columns function");
    }
    add_columns_ps = m_device->newComputePipelineState(add_columns_function, &error);
    if (!add_columns_ps) {
        throw std::runtime_error("failed to create the add_columns pipeline state object");
    }
    
    m_command_queue = m_device->newCommandQueue();
    if (!m_command_queue) {
        throw std::runtime_error("failed to find command queue");
    }

    readFromFile(file_path);
}

void MetalSparseMatrix::widenBuffer() {
    size_t new_size = row_index_size_ * widen_coef_;

    auto new_buffer = m_device->newBuffer(new_size * sizeof(uint32_t), MTL::ResourceStorageModeShared);
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
    row_index_buffer_ = m_device->newBuffer(row_index_size_ * sizeof(uint32_t), MTL::ResourceStorageModeShared);
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
            col_start_ = m_device->newBuffer(n_ * sizeof(uint32_t), MTL::ResourceStorageModeShared);
            col_end_ = m_device->newBuffer(n_ * sizeof(uint32_t), MTL::ResourceStorageModeShared);
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
    row_index_ = m_device->newBuffer(row_index_size_ * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    auto row_index_ptr = (uint32_t*)row_index_->contents();
    for (size_t i = 0; i < row_index.size(); i++) {
        row_index_ptr[i] = row_index[i];
    }
    row_index_buffer_ = m_device->newBuffer(row_index_size_ * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    widenBuffer();
}

void MetalSparseMatrix::sendComputeCommand(MTL::ComputePipelineState* ps, std::vector<MTL::Buffer*> buffers) {
    MTL::CommandBuffer* commandBuffer = m_command_queue->commandBuffer();
    assert(commandBuffer != nullptr);
    MTL::ComputeCommandEncoder* compute_encoder = commandBuffer->computeCommandEncoder();
    assert(compute_encoder != nullptr);

    compute_encoder->setComputePipelineState(ps);
    for (size_t i = 0; i < buffers.size(); i++) {
        compute_encoder->setBuffer(buffers[i], 0, i);
    }

    MTL::Size gridSize = MTL::Size(n_, 1, 1);
    NS::UInteger threadGroupSize = add_columns_ps->maxTotalThreadsPerThreadgroup();
    if (threadGroupSize > n_) {
        threadGroupSize = n_;
    }
    MTL::Size threadgroupSize = MTL::Size(threadGroupSize, 1, 1);

    compute_encoder->dispatchThreads(gridSize, threadgroupSize);

    compute_encoder->endEncoding();
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

    MTL::Buffer* need_widen_buffer = m_device->newBuffer(sizeof(uint32_t), MTL::ResourceStorageModeShared);
    auto* need_widen_buffer_ptr = (uint32_t*)need_widen_buffer->contents();
    *need_widen_buffer_ptr = 0;

    MTL::Buffer* n_buffer = m_device->newBuffer(sizeof(uint32_t), MTL::ResourceStorageModeShared);
    auto* n_ptr = (uint32_t*)n_buffer->contents();
    *n_ptr = n_;

    MTL::Buffer* to_add = m_device->newBuffer(n_ * sizeof(uint32_t), MTL::ResourceStorageModeShared);

    MTL::Buffer* inverse_low = m_device->newBuffer(n_ * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    auto* inverse_low_ptr = (uint32_t*)inverse_low->contents();
    for (size_t i = 0; i < n_; i++) {
        inverse_low_ptr[i] = n_;
    }

    MTL::Buffer* is_over = m_device->newBuffer(sizeof(uint32_t), MTL::ResourceStorageModeShared);
    auto* is_over_ptr = (uint32_t*)is_over->contents();

    MTL::Buffer* widen_coef_buffer_ = m_device->newBuffer(sizeof(uint32_t), MTL::ResourceStorageModeShared);
    auto* widen_coef_buffer_ptr = (uint32_t*)widen_coef_buffer_->contents();
    *widen_coef_buffer_ptr = widen_coef_;

    auto col_start_ptr = (uint32_t*)col_start_->contents();
    auto col_end_ptr = (uint32_t*)col_end_->contents();
    auto row_index_ptr = (uint32_t*)row_index_->contents();

    while (true) {
        std::vector<MTL::Buffer*> buffers = {
            row_index_,
            col_start_,
            col_end_,
            inverse_low,
            n_buffer,
        };
        sendComputeCommand(count_inverse_low_ps, buffers);

        *is_over_ptr = 1;
        buffers = {
            row_index_,
            col_start_,
            col_end_,
            inverse_low,
            to_add,
            n_buffer,
            is_over,
        };
        sendComputeCommand(count_to_add_ps, buffers);

        if (*is_over_ptr == 1) {
            break;
        }

        if (*need_widen_buffer_ptr == 1) {
            size_t new_size = row_index_size_ * widen_coef_;

            row_index_buffer_->release();
            row_index_buffer_ = m_device->newBuffer(new_size * sizeof(uint32_t), MTL::ResourceStorageModeShared);

            buffers = {
                row_index_,
                col_start_,
                col_end_,
                widen_coef_buffer_,
                row_index_buffer_,
            };
            sendComputeCommand(copy_to_row_index_buffer_ps, buffers);

            row_index_size_ = new_size;
            row_index_->release();
            row_index_ = row_index_buffer_;
            row_index_ptr = (uint32_t*)row_index_->contents();

            row_index_buffer_ = m_device->newBuffer(row_index_size_ * sizeof(uint32_t), MTL::ResourceStorageModeShared);
        }
        *need_widen_buffer_ptr = 0;

        buffers = {
            col_start_,
            col_end_,
            row_index_,
            row_index_buffer_,
            to_add,
            n_buffer,
            need_widen_buffer,
        };
        sendComputeCommand(add_columns_ps, buffers);
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
    m_pool->release();
}
