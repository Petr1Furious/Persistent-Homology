import Foundation

func runTwist(matrix: inout SparseMatrix) {
    for i in 0 ..< matrix.n {
        if matrix.getLow(col_index: UInt32(i)) != UInt32.max {
            matrix.clearColumn(col_index: matrix.getLow(col_index: UInt32(i)))
        }
    }
}

func reduceSparseMatrix(matrix: inout SparseMatrix, twist: Bool) -> [Int] {
    if twist {
        runTwist(matrix: &matrix)
    }

    var result_row_index_buffer = [UInt32]()
    var row_index_buffer = [UInt32]()

    var L = [UInt32](repeating: UInt32.max, count: matrix.n)
    for i in 0 ..< matrix.n {
        var low = matrix.getLow(col_index: UInt32(i))
        while low != UInt32.max && L[Int(low)] != UInt32.max {
            matrix.addColumn(col1: UInt32(i), col2: L[Int(low)],
                row_index_buffer: &row_index_buffer, result_row_index_buffer: &result_row_index_buffer)
            low = matrix.getLow(col_index: UInt32(i))
        }

        if low != UInt32.max {
            L[Int(low)] = UInt32(i)
        }
    }
    return matrix.getLowArray()
}

func reduceSparseParallel(matrix: inout SparseMatrix, twist: Bool) -> [Int] {
    if twist {
        runTwist(matrix: &matrix)
    }

    var result_row_index_buffer = [UInt32]()
    var row_index_buffer = [UInt32]()

    var toAdd = [UInt32](repeating: UInt32.max, count: matrix.n)
    while true {
        var inverseLow = [UInt32](repeating: UInt32.max, count: matrix.n)
        var isOver = true
        for i in 0 ..< matrix.n {
            let curLow = matrix.getLow(col_index: UInt32(i))
            if curLow == UInt32.max {
                continue
            }
            let curInverseLow = inverseLow[Int(curLow)]

            if curInverseLow == UInt32.max {
                inverseLow[Int(matrix.getLow(col_index: UInt32(i)))] = UInt32(i)
            } else {
                toAdd[i] = curInverseLow
            }
            
            if toAdd[i] != UInt32.max {
                isOver = false
            }
        }
        if isOver {
            break
        }

        for i in 0 ..< matrix.n {
            if toAdd[i] != UInt32.max {
                matrix.addColumn(col1: UInt32(i), col2: toAdd[i],
                    row_index_buffer: &row_index_buffer, result_row_index_buffer: &result_row_index_buffer)
                toAdd[i] = UInt32.max
            }
        }
    }
    return matrix.getLowArray()
}
