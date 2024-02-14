import Foundation
import Metal

class MetalPH {
    let device: MTLDevice
    let computeLowFunctionPSO: MTLComputePipelineState
    let reduceMatrixFunctionPSO: MTLComputePipelineState
    let commandQueue: MTLCommandQueue

    var matrixSizeValue: size_t?
    var matrixSize: MTLBuffer?
    var matrix: MTLBuffer?
    var low: MTLBuffer?
    var lowClass: MTLBuffer?

    init?(_ device: MTLDevice) {
        self.device = device
        
        let defaultLibrary = self.device.makeDefaultLibrary()
        if (defaultLibrary == nil) {
            NSLog("Could not find library")
            return nil
        }
        
        let computeLowFunction = defaultLibrary!.makeFunction(name: "compute_low")
        if (computeLowFunction == nil) {
            NSLog("Could not find compute_low function")
            return nil
        }
        
        do {
            try self.computeLowFunctionPSO = self.device.makeComputePipelineState(function: computeLowFunction!)
        } catch {
            NSLog("Could not create compute_low pipeline")
            return nil
        }
        
        let reduceMatrixFunction = defaultLibrary!.makeFunction(name: "reduce_matrix")
        if (reduceMatrixFunction == nil) {
            NSLog("Could not find reduce_matrix function")
            return nil
        }
        
        do {
            try self.reduceMatrixFunctionPSO = self.device.makeComputePipelineState(function: reduceMatrixFunction!)
        } catch {
            NSLog("Could not create reduce_matrix pipeline")
            return nil
        }
        
        self.commandQueue = self.device.makeCommandQueue()!
    }
    
    func loadData(matrix: Matrix) {
        self.matrixSizeValue = matrix.n
        self.matrixSize = self.device.makeBuffer(length: MemoryLayout<UInt>.stride, options: MTLResourceOptions.storageModeShared)!
        self.matrix = self.device.makeBuffer(length: matrix.n * matrix.n * MemoryLayout<Bool>.stride, options: MTLResourceOptions.storageModeShared)!
        self.low = self.device.makeBuffer(length: matrix.n * MemoryLayout<Int>.stride, options: MTLResourceOptions.storageModeShared)!
        self.lowClass = self.device.makeBuffer(length: matrix.n * MemoryLayout<UInt>.stride, options: MTLResourceOptions.storageModeShared)!

        let matrixSizePtr = self.matrixSize!.contents().assumingMemoryBound(to: UInt.self)
        matrixSizePtr[0] = UInt(matrix.n)

        let matrixPtr: UnsafeMutablePointer<Bool> = self.matrix!.contents().assumingMemoryBound(to: Bool.self)
        for i in 0 ..< matrix.n {
            for j in 0 ..< matrix.n {
                let index = i * matrix.n + j
                matrixPtr[index] = matrix.columns[i][j]
            }
        }
    }
    
    func encodeComputeLowCommand(_ computeEncoder: MTLComputeCommandEncoder) {
        computeEncoder.setComputePipelineState(self.computeLowFunctionPSO)
        computeEncoder.setBuffer(self.matrix, offset: 0, index: 0)
        computeEncoder.setBuffer(self.matrixSize, offset: 0, index: 1)
        computeEncoder.setBuffer(self.low, offset: 0, index: 2)

        let gridSize: MTLSize = MTLSizeMake(self.matrixSizeValue!, 1, 1)

        var threadGroupSizeInt: Int = self.computeLowFunctionPSO.maxTotalThreadsPerThreadgroup
        if (threadGroupSizeInt > self.matrixSizeValue!) {
            threadGroupSizeInt = self.matrixSizeValue!
        }
        let threadGroupSize: MTLSize = MTLSizeMake(threadGroupSizeInt, 1, 1)

        computeEncoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupSize)
    }
    
    func runComputeLowCommand() {
        let commandBuffer: MTLCommandBuffer = self.commandQueue.makeCommandBuffer()!
        let computeEncoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()!

        self.encodeComputeLowCommand(computeEncoder)

        computeEncoder.endEncoding()
        commandBuffer.commit()

        commandBuffer.waitUntilCompleted()
    }
    
    func encodeReduceMatrixCommand(_ computeEncoder: MTLComputeCommandEncoder) {
        computeEncoder.setComputePipelineState(self.reduceMatrixFunctionPSO)
        computeEncoder.setBuffer(self.matrix, offset: 0, index: 0)
        computeEncoder.setBuffer(self.matrixSize, offset: 0, index: 1)
        computeEncoder.setBuffer(self.lowClass, offset: 0, index: 2)

        let gridSize: MTLSize = MTLSizeMake(self.matrixSizeValue!, 1, 1)

        var threadGroupSizeInt: Int = self.reduceMatrixFunctionPSO.maxTotalThreadsPerThreadgroup
        if (threadGroupSizeInt > self.matrixSizeValue!) {
            threadGroupSizeInt = self.matrixSizeValue!
        }
        let threadGroupSize: MTLSize = MTLSizeMake(threadGroupSizeInt, 1, 1)

        computeEncoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupSize)
    }
    
    func runReduceMatrixCommand() {
        let commandBuffer: MTLCommandBuffer = self.commandQueue.makeCommandBuffer()!
        let computeEncoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()!

        self.encodeReduceMatrixCommand(computeEncoder)

        computeEncoder.endEncoding()
        commandBuffer.commit()

        commandBuffer.waitUntilCompleted()
    }
    
    func run(matrix: Matrix) -> [Int] {
        loadData(matrix: matrix)

        while true {
            runComputeLowCommand()
            
            var isOver = true

            let lowPtr: UnsafeMutablePointer<Int> = self.low!.contents().assumingMemoryBound(to: Int.self)
            let lowClassPtr: UnsafeMutablePointer<Int> = self.lowClass!.contents().assumingMemoryBound(to: Int.self)
            var reverseLow = [Int](repeating: -1, count: self.matrixSizeValue!)
            for i in 0 ..< self.matrixSizeValue! {
                lowClassPtr[i] = -1
                if lowPtr[i] != -1 {
                    let curLow = lowPtr[i]
                    if reverseLow[curLow] == -1 {
                        reverseLow[curLow] = i
                    } else {
                        lowClassPtr[i] = reverseLow[curLow]
                        isOver = false
                    }
                }
            }

            if isOver {
                break
            }

            runReduceMatrixCommand()
        }
        
        var lowResult = [Int](repeating: -1, count: self.matrixSizeValue!)
        let lowPtr: UnsafeMutablePointer<Int> = self.low!.contents().assumingMemoryBound(to: Int.self)
        for i in 0 ..< self.matrixSizeValue! {
            lowResult[i] = lowPtr[i]
        }
        return lowResult
    }
}
