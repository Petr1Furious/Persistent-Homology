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
