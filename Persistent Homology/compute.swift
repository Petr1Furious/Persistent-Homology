//
//  compute.swift
//  Persistent Homology
//
//  Created by Petr Tsopa on 23.01.24.
//

import Foundation

struct Matrix {
    var n: size_t
    var elements: [[Bool]]
}

func readFile(file_path: String) -> Matrix? {
    var matrix: Matrix? = nil
    do {
        let text = try String(contentsOf: URL(fileURLWithPath: file_path), encoding: .utf8)
        for (i, line) in text.components(separatedBy: .newlines).enumerated() {
            if matrix == nil {
                let n = size_t(line)!
                matrix = Matrix(n: n, elements: [[Bool]](repeating: [Bool](repeating: false, count: n), count: n))
            } else {
                let indices = line.split(separator: " ").compactMap { size_t($0) }
                for j in indices {
                    matrix!.elements[i - 1][j] = true
                }
            }
        }
    }
    catch (let error) {
        print(error)
        return nil
    }
    return matrix
}

func getLow(column: [Bool]) -> Int {
    for i in (0...column.count-1).reversed() {
        if column[i] {
            return i
        }
    }
    return -1
}

func reduceMatrixSimple(matrix: inout Matrix) -> [Int] {
    var low = [Int](repeating: -1, count: matrix.n)
    for i in 1...matrix.n-1 {
        let column = matrix.elements[i]
        low[i] = getLow(column: column)
        for j in 0...i-1 {
            if low[j] == low[i] {
                for k in 0...matrix.n-1 {
                    matrix.elements[i][k] = matrix.elements[i][k] != matrix.elements[j][k]
                }
            }
        }
    }
    return low
}

func writePairsToFile(file_path: String, low: [Int]) -> Bool {
    var output = ""
    for i in 0...low.count-1 {
        if low[i] != -1 {
            output += String(i) + " " + String(low[i]) + "\n"
        }
    }
    do {
        try output.write(toFile: file_path, atomically: false, encoding: .utf8)
    }
    catch (let error) {
        print(error)
        return false
    }
    return true
}
