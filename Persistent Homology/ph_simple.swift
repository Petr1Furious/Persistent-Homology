//
//  compute.swift
//  Persistent Homology
//
//  Created by Petr Tsopa on 23.01.24.
//

import Foundation

struct Matrix {
    var n: size_t
    var columns: [[Bool]]
}

func readFile(file_path: String) -> Matrix? {
    var matrix: Matrix? = nil
    do {
        let text = try String(contentsOf: URL(fileURLWithPath: file_path), encoding: .utf8)
        for (i, line) in text.components(separatedBy: .newlines).enumerated() {
            if matrix == nil {
                let n = size_t(line)!
                matrix = Matrix(n: n, columns: [[Bool]](repeating: [Bool](repeating: false, count: n), count: n))
            } else {
                let indices = line.split(separator: " ").compactMap { size_t($0) }
                for j in indices {
                    matrix!.columns[i - 1][j] = true
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
    for i in (0 ..< column.count).reversed() {
        if column[i] {
            return i
        }
    }
    return -1
}

func getSum(column1: [Bool], column2: [Bool]) -> [Bool] {
    var res = [Bool](repeating: false, count: column1.count)
    for i in 0 ..< column1.count {
        res[i] = column1[i] != column2[i]
    }
    return res
}

func reduceMatrixSimple(matrix: inout Matrix) -> [Int] {
    var low = [Int](repeating: -1, count: matrix.n)
    var L = [Int](repeating: -1, count: matrix.n)
    for i in 0 ..< matrix.n {
        low[i] = getLow(column: matrix.columns[i])
        while low[i] != -1 && L[low[i]] != -1 {
            matrix.columns[i] = getSum(column1: matrix.columns[i], column2: matrix.columns[L[low[i]]])
            low[i] = getLow(column: matrix.columns[i])
        }
        if low[i] != -1 {
            L[low[i]] = i
        }
    }
    return low
}

enum DimensionsError: Error {
    case runtimeError(String)
}

func getDimensions(matrix: inout Matrix) throws -> [Int] {
    var dimensions = [Int](repeating: 0, count: matrix.n)
    for i in 0 ..< matrix.n {
        for j in 0 ..< matrix.n {
            if matrix.columns[i][j] {
                if dimensions[i] != 0 && dimensions[i] != dimensions[j] + 1 {
                    throw DimensionsError.runtimeError("Matrix does not represent valid simplicial complex")
                }
                dimensions[i] = dimensions[j] + 1
            }
        }
    }
    return dimensions
}

func reduceMatrixTwist(matrix: inout Matrix) throws -> [Int] {
    let dimensions = try getDimensions(matrix: &matrix)
    let max_dimension = dimensions.max()!
    var L = [Int](repeating: -1, count: matrix.n)
    var low = [Int](repeating: -1, count: matrix.n)

    for dimension in (0...max_dimension).reversed() {
        for i in 0 ..< matrix.n {
            if dimensions[i] == dimension {
                while true {
                    let low = getLow(column: matrix.columns[i])
                    if low == -1 || L[low] == -1 {
                        break
                    }
                    matrix.columns[i] = getSum(column1: matrix.columns[i], column2: matrix.columns[L[low]])
                }
                low[i] = getLow(column: matrix.columns[i])
                if (low[i] != -1) {
                    L[low[i]] = i
                    for k in 0 ..< matrix.n {
                        matrix.columns[low[i]][k] = false
                    }
                }
            }
        }
    }

    return low
}

func writePairsToFile(file_path: String, low: [Int]) -> Bool {
    var output = ""
    for i in 0 ..< low.count {
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
