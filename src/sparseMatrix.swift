func reduceSparseParallel(matrix: inout SparseMatrix, twist: Bool) throws -> [Int] {
    if twist {
        runTwist(matrix: &matrix)
    }

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

        if !matrix.enoughSizeForIteration() {
            matrix.resizeBuffer(row_index_buffer: &row_index_buffer)
            for i in (0 ..< matrix.n).reversed() {
                matrix.copyToBuffer(row_index_buffer: &row_index_buffer, col: i)
            }
            matrix.swapBuffers(row_index_buffer: &row_index_buffer)
        }

        for i in 0 ..< matrix.n {
            if toAdd[i] != UInt32.max {
                try matrix.addColumn(col1: UInt32(i), col2: toAdd[i],
                    row_index_buffer: &row_index_buffer, copy: false)
                toAdd[i] = UInt32.max
            } else {
                matrix.copyColumnToBuffer(row_index_buffer: &row_index_buffer, col: i)
            }
        }
        matrix.swapBuffers(row_index_buffer: &row_index_buffer)
    }
    return matrix.getLowArray()
}