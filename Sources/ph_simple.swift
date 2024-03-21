import Foundation

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
