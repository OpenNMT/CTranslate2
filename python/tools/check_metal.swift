import Foundation
import Metal

guard let device = MTLCreateSystemDefaultDevice() else {
    fatalError("No Metal device available on this runner")
}

print("Metal device: \(device.name)")
print("Unified memory: \(device.hasUnifiedMemory)")
print("Recommended working set: \(device.recommendedMaxWorkingSetSize) bytes")

let source = """
#include <metal_stdlib>
using namespace metal;
kernel void increment(device uint *values [[buffer(0)]],
                      uint index [[thread_position_in_grid]]) {
    values[index] += 1;
}
"""

let library = try device.makeLibrary(source: source, options: nil)
guard let function = library.makeFunction(name: "increment"),
      let queue = device.makeCommandQueue(),
      let buffer = device.makeBuffer(length: 4 * MemoryLayout<UInt32>.stride,
                                     options: .storageModeShared),
      let commandBuffer = queue.makeCommandBuffer(),
      let encoder = commandBuffer.makeComputeCommandEncoder() else {
    fatalError("Could not create Metal compute resources")
}

let pipeline = try device.makeComputePipelineState(function: function)
let values = buffer.contents().bindMemory(to: UInt32.self, capacity: 4)
for index in 0..<4 {
    values[index] = UInt32(index)
}

encoder.setComputePipelineState(pipeline)
encoder.setBuffer(buffer, offset: 0, index: 0)
encoder.dispatchThreads(MTLSize(width: 4, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: 4, height: 1, depth: 1))
encoder.endEncoding()
commandBuffer.commit()
commandBuffer.waitUntilCompleted()

guard commandBuffer.status == .completed else {
    fatalError("Metal command failed: \(String(describing: commandBuffer.error))")
}

for index in 0..<4 {
    precondition(values[index] == UInt32(index + 1), "Incorrect Metal kernel result")
}

print("Metal compute probe passed")
