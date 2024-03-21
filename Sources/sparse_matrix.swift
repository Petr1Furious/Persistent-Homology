import Foundation

struct SparseMatrix {
    // CSC-like format
    var n: size_t
    var row_index: [UInt32]
    var col_start: [UInt32]
    var col_end: [UInt32]
    var widen_coef: UInt32

    init(widen_coef: UInt32) {
        self.n = 0
        self.row_index = []
        self.col_start = []
        self.col_end = []
        self.widen_coef = widen_coef
    }

    mutating func readFromFile(file_path: String) throws {
        do {
            let text = try String(contentsOf: URL(fileURLWithPath: file_path), encoding: .utf8)
            for (i, line) in text.components(separatedBy: .newlines).enumerated() {
                if i > n {
                    print("Warning: file too long")
                    break
                }

                if i == 0 {
                    n = size_t(line)!
                    col_start = [UInt32](repeating: 0, count: n)
                    col_end = [UInt32](repeating: 0, count: n)
                } else {
                    let indices = line.split(separator: " ").compactMap { UInt32($0) }
                    for j in indices {
                        row_index.append(j)
                    }

                    if i < n {
                        col_start[i] = col_start[i - 1] + UInt32(indices.count)
                        col_end[i - 1] = col_start[i]
                    } else {
                        col_end[n - 1] = col_start[n - 1] + UInt32(indices.count)
                    }
                }
            }
        }
        catch (let error) {
            print(error)
            throw error
        }
    }

    func getLow(col_index: UInt32) -> UInt32 {
        if col_start[Int(col_index)] == col_end[Int(col_index)] {
            return UInt32.max
        }
        return row_index[Int(col_end[Int(col_index)] - 1)]
    }

    func getLowArray() -> [Int] {
        var low = [Int](repeating: 0, count: n)
        for i in 0 ..< n {
            low[i] = Int(getLow(col_index: UInt32(i)))
            if low[i] == Int(UInt32.max) {
                low[i] = -1
            }
        }
        return low
    }

    mutating func addColumn(col1: UInt32, col2: UInt32, row_index_buffer: inout [UInt32], result_row_index_buffer: inout [UInt32]) {
        let sum_size = Int((col_end[Int(col1)] - col_start[Int(col1)]) +
            (col_end[Int(col2)] - col_start[Int(col2)]))
        if result_row_index_buffer.count < sum_size {
            result_row_index_buffer += [UInt32](repeating: 0, count: sum_size - result_row_index_buffer.count)
        }

        let end1 = Int(col_end[Int(col1)])
        let end2 = Int(col_end[Int(col2)])

        var i = Int(col_start[Int(col1)])
        var j = Int(col_start[Int(col2)])
        var k = 0
        while i < end1 && j < end2 {
            if row_index[i] < row_index[j] {
                result_row_index_buffer[k] = row_index[i]
                i += 1
                k += 1
            } else if row_index[i] > row_index[j] {
                result_row_index_buffer[k] = row_index[j]
                j += 1
                k += 1
            } else {
                i += 1
                j += 1
            }
        }
        while i < end1 {
            result_row_index_buffer[k] = row_index[i]
            i += 1
            k += 1
        }
        while j < end2 {
            result_row_index_buffer[k] = row_index[j]
            j += 1
            k += 1
        }

        let available_size: Int
        if col1 + 1 < n {
            available_size = Int(col_start[Int(col1) + 1]) - Int(col_start[Int(col1)])
        } else {
            available_size = row_index.count - Int(col_start[Int(col1)])
        }
        if k > available_size {
            widenBuffer(row_index_buffer: &row_index_buffer)
        }
        row_index.replaceSubrange(Int(col_start[Int(col1)])..<Int(col_start[Int(col1)])+k, with: result_row_index_buffer[0..<k])
        col_end[Int(col1)] = col_start[Int(col1)] + UInt32(k)
    }

    mutating func widenBuffer(row_index_buffer: inout [UInt32]) {
        var new_size: size_t = 0
        for (start, end) in zip(col_start, col_end) {
            new_size += Int(end - start)
        }
        new_size *= Int(widen_coef)

        if new_size < row_index_buffer.count {
            new_size = row_index_buffer.count
        }
        row_index_buffer += [UInt32](repeating: 0, count: new_size - row_index_buffer.count)

        var cur_start = UInt32(new_size)
        for i in (0 ..< n).reversed() {
            let start = col_start[i]
            let end = col_end[i]

            cur_start -= widen_coef * (end - start)
            let cur_end = cur_start + (end - start)
            row_index_buffer.replaceSubrange(Int(cur_start)..<Int(cur_end), with: row_index[Int(start)..<Int(end)])

            col_start[i] = cur_start
            col_end[i] = cur_end
        }

        swap(&row_index, &row_index_buffer)
    }

    mutating func clearColumn(col_index: UInt32) {
        col_end[Int(col_index)] = col_start[Int(col_index)]
    }
}
