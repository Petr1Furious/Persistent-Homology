//
//  main.swift
//  Persistent Homology
//
//  Created by Petr Tsopa on 23.01.24.
//

import Foundation

let arguments = CommandLine.arguments

if arguments.count != 3 {
    print("Usage: " + arguments[0] + " <input file name> <output file name>")
    exit(1)
}

let input_file_path = arguments[1], output_file_path = arguments[2]

if var matrix = readFile(file_path: input_file_path) {
    let low = reduceMatrixSimple(matrix: &matrix)
    if !writePairsToFile(file_path: output_file_path, low: low) {
        print("Failed to write to file")
    }
} else {
    print("Failed to read sparse matrix from file")
}
