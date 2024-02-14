//
//  main.swift
//  Persistent Homology
//
//  Created by Petr Tsopa on 23.01.24.
//

import Foundation
import Metal

try main()

func main() throws -> Void {
    let arguments = CommandLine.arguments
    
    if arguments.count != 4 {
        print("Usage: " + arguments[0] + " <simple/twist/metal> <input file name> <output file name>")
        exit(1)
    }

    let mode = arguments[1]
    
    let input_file_path = arguments[2], output_file_path = arguments[3]
    
    if var matrix = readFile(file_path: input_file_path) {
        var low: [Int] = []
        if mode == "simple" {
            low = reduceMatrixSimple(matrix: &matrix)
        } else if mode == "twist" {
            low = try reduceMatrixTwist(matrix: &matrix)
        } else if mode == "metal" {
            let device: MTLDevice = MTLCopyAllDevices().first!
            let metal_compute: MetalPH = MetalPH(device)!

            low = metal_compute.run(matrix: matrix)
        } else {
            print("Wrong mode")
            exit(1)
        }
        if !writePairsToFile(file_path: output_file_path, low: low) {
            print("Failed to write to file")
            exit(1)
        }
    } else {
        print("Failed to read sparse matrix from file")
        exit(1)
    }
}
