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
                    if line != "" {
                        print("Warning: file too long")
                    }
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

    func enoughSizeForIteration() -> Bool {
        for i in 0 ..< n {
            let col_size = col_end[Int(i)] - col_start[Int(i)]
            let available_size = i + 1 < n ? col_start[Int(i + 1)] - col_start[Int(i)] : UInt32(row_index.count) - col_start[Int(i)]
            if col_size * 2 > available_size {
                return false
            }
        }
        return true
    }

    func enoughSizeForIteration(col: Int) -> Bool {
        let col_size = col_end[col] - col_start[col]
        let available_size = col + 1 < n ? col_start[col + 1] - col_start[col] : UInt32(row_index.count) - col_start[col]
        return col_size * 2 <= available_size
    }

    mutating func addColumn(col1: UInt32, col2: UInt32, row_index_buffer: inout [UInt32], copy: Bool) throws {
        let end1 = Int(col_end[Int(col1)])
        let end2 = Int(col_end[Int(col2)])

        var i = Int(col_start[Int(col1)])
        var j = Int(col_start[Int(col2)])
        var k = Int(col_start[Int(col1)])
        while i < end1 && j < end2 {
            if row_index[i] < row_index[j] {
                row_index_buffer[k] = row_index[i]
                i += 1
                k += 1
            } else if row_index[i] > row_index[j] {
                row_index_buffer[k] = row_index[j]
                j += 1
                k += 1
            } else {
                i += 1
                j += 1
            }
        }
        while i < end1 {
            row_index_buffer[k] = row_index[i]
            i += 1
            k += 1
        }
        while j < end2 {
            row_index_buffer[k] = row_index[j]
            j += 1
            k += 1
        }

        let available_size: Int
        if col1 + 1 < n {
            available_size = Int(col_start[Int(col1) + 1]) - Int(col_start[Int(col1)])
        } else {
            available_size = row_index.count - Int(col_start[Int(col1)])
        }
        if k - Int(col_start[Int(col1)]) > available_size {
            throw NSError(domain: "SparseMatrix", code: 1, userInfo: ["message": "Not enough space"])
        }

        if copy {
            let start = Int(col_start[Int(col1)])
            row_index.replaceSubrange(start..<k, with: row_index_buffer[start..<k])
        }
        col_end[Int(col1)] = UInt32(k)
    }

    mutating func swapBuffers(row_index_buffer: inout [UInt32]) {
        swap(&row_index, &row_index_buffer)
    }

    mutating func resizeBuffer(row_index_buffer: inout [UInt32]) {
        let new_size: size_t = row_index.count * Int(widen_coef)
        row_index_buffer += [UInt32](repeating: 0, count: new_size - row_index_buffer.count)
        row_index += [UInt32](repeating: 0, count: row_index_buffer.count - row_index.count)
    }

    mutating func copyToBuffer(row_index_buffer: inout [UInt32], col: Int) {
        let start = col_start[col]
        let end = col_end[col]

        let new_start = start * widen_coef
        let new_end = new_start + (end - start)
        row_index_buffer.replaceSubrange(Int(new_start)..<Int(new_end), with: row_index[Int(start)..<Int(end)])

        col_start[col] = new_start
        col_end[col] = new_end
    }

    mutating func widenBuffer(row_index_buffer: inout [UInt32]) {
        resizeBuffer(row_index_buffer: &row_index_buffer)
        for i in (0 ..< n).reversed() {
            copyToBuffer(row_index_buffer: &row_index_buffer, col: i)
        }
        swapBuffers(row_index_buffer: &row_index_buffer)
    }

    mutating func copyColumnToBuffer(row_index_buffer: inout [UInt32], col: Int) {
        let start = Int(col_start[col])
        let end = Int(col_end[col])
        row_index_buffer.replaceSubrange(start..<end, with: row_index[start..<end])
    }

    mutating func clearColumn(col_index: UInt32) {
        col_end[Int(col_index)] = col_start[Int(col_index)]
    }
}
