import Foundation

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
