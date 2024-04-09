#include "SparseMetal.hpp"
#include "MetalComputeWrapper.hpp"

#include <iostream>

void runMetalCompute() {
    NS::AutoreleasePool* pPool   = NS::AutoreleasePool::alloc()->init();
    MTL::Device* pDevice = MTL::CreateSystemDefaultDevice();
    
    // Create the custom object used to encapsulate the Metal code.
    // Initializes objects to communicate with the GPU.
    MetalComputeWrapper* computer = new MetalComputeWrapper();
    computer->initWithDevice(pDevice);
    
    // Create buffers to hold data
    computer->prepareData();
    
    // Time the compute phase.
    auto start = std::chrono::steady_clock::now();
    
    // Send a command to the GPU to perform the calculation.
    computer->sendComputeCommand();
    
    // End of compute phase.
    auto end = std::chrono::steady_clock::now();
    auto delta_time = end - start;
    
    pPool->release();
    
    std::cout << "Computation completed in "
            << std::chrono::duration <double, std::milli> (delta_time).count()
            << " ms for array of size "
            << ARRAY_LENGTH
            <<".\n";
}
