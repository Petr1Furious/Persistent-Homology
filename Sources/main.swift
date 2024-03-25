import Foundation
import Metal

try main()

func main() throws -> Void {
    let arguments = CommandLine.arguments
    
    if arguments.count != 4 {
        print("Usage: " + arguments[0] + " <simple/twist/metal/sparse/sparse-twist/sparse-parallel/sparse-twist-parallel> <input file name> <output file name>")
        exit(1)
    }

    let mode = arguments[1]
    
    let input_file_path = arguments[2], output_file_path = arguments[3]

    var low = [Int]()
    if ["simple", "twist", "metal"].contains(mode) {
        guard var matrix = readFile(file_path: input_file_path) else {
            print("Error reading matrix from file")
            exit(1)
        }

        if mode == "simple" {
            low = reduceMatrixSimple(matrix: &matrix)
        } else if mode == "twist" {
            low = try reduceMatrixTwist(matrix: &matrix)
        } else if mode == "metal" {
            let device: MTLDevice = MTLCopyAllDevices().first!
            if let metal_compute: MetalPH = MetalPH(device) {
                low = metal_compute.run(matrix: matrix)
            } else {
                print("Failed to initialize MetalPH")
                exit(1)
            }
        }
    } else if ["sparse", "sparse-twist", "sparse-parallel", "sparse-twist-parallel"].contains(mode) {
        var matrix = SparseMatrix(widen_coef: 2)
        do {
            try matrix.readFromFile(file_path: input_file_path)
        } catch {
            print("Error reading sparse matrix from file: \(error)")
        }

        if mode == "sparse" || mode == "sparse-twist" {
            low = try reduceSparseMatrix(matrix: &matrix, twist: mode == "sparse-twist")
        } else {
            low = try reduceSparseParallel(matrix: &matrix, twist: mode == "sparse-twist-parallel")
        }
    } else {
        print("Wrong mode")
        exit(1)
    }

    if !writePairsToFile(file_path: output_file_path, low: low) {
        print("Failed to write to file")
        exit(1)
    }
}
