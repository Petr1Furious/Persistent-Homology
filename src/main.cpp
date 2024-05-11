#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include <Metal/Metal.hpp>
#include <chrono>
#include <fstream>
#include <iostream>

#include "IMatrix.hpp"
#include "MetalSparseMatrix.hpp"
#include "ParallelSparseMatrix.hpp"
#include "SparseMatrix.hpp"

int main(int argc, const char* argv[]) {
    if (argc != 4) {
        std::cout << "Usage: " << argv[0]
                  << " <sparse/sparse-twist/sparse-parallel/"
                     "sparse-parallel-twist/sparse-metal/sparse-metal-twist> "
                     "<input file name> <output file name>\n";
        return 1;
    }

    std::string mode = argv[1];
    std::string inputFileName = argv[2];
    std::string outputFileName = argv[3];

    std::unique_ptr<IMatrix> matrix;
    if (mode == "sparse" || mode == "sparse-twist") {
        matrix = std::make_unique<SparseMatrix>(inputFileName);
    } else if (mode == "sparse-parallel" || mode == "sparse-parallel-twist") {
        matrix = std::make_unique<ParallelSparseMatrix>(inputFileName);
    } else if (mode == "sparse-metal" || mode == "sparse-metal-twist") {
        matrix = std::make_unique<MetalSparseMatrix>(inputFileName);
    } else {
        std::cout << "Unknown mode: " << mode << "\n";
        return 1;
    }

    auto start = std::chrono::high_resolution_clock::now();
    std::vector<uint32_t> result = matrix->reduce(
        mode == "sparse-twist" || mode == "sparse-parallel-twist" ||
        mode == "sparse-metal-twist");
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(
                     std::chrono::high_resolution_clock::now() - start)
                         .count() /
                     1'000'000.0
              << "\n";

    std::vector<std::pair<uint32_t, uint32_t>> result_pairs;
    for (size_t i = 0; i < result.size(); i++) {
        if (result[i] != matrix->size()) {
            result_pairs.emplace_back(result[i], i);
        }
    }
    std::sort(result_pairs.begin(), result_pairs.end());

    std::ofstream outputFile(outputFileName);
    for (const auto& [birth, death] : result_pairs) {
        outputFile << birth << " " << death << "\n";
    }
    return 0;
}
