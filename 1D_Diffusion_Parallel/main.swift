//
//  main.swift
//  1D_Diffusion_Parallel
//
//  Created by 조일현 on 2023/07/19.
//

import Foundation
import MetalFFT

let device = MTLCreateSystemDefaultDevice()!
let commandQueue = device.makeCommandQueue()!
let batchSize : Int = 28672
let input = FFTSplitComplexBuffer(device: device, capacity: 128 * batchSize)
let output = FFTComplexBuffer(device: device, capacity: 128 * batchSize)

let fft = FastFourierTransform1D(device: device,
                                 transformWidth: 128,
                                 inputType: .splitComplex,
                                 batchSize: batchSize)

print("execution time in milliseconds")

for _ in 0..<20 {
    let commandBuffer = commandQueue.makeCommandBuffer()!

    fft.encode(commandBuffer: commandBuffer, input: input, output: output)

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()

    let elapsedTime = commandBuffer.gpuEndTime - commandBuffer.gpuStartTime
    print(elapsedTime * 1e3)
}

