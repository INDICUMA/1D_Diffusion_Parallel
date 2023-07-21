//
//  main.swift
//  1D_Diffusion_Parallel
//
//  Created by 조일현 on 2023/07/19.
//
import Foundation
import MetalFFT

let batchSize = 1
let n : Int = 128
//fft, ifft array
var fftInput = [SIMD2<Float>](repeating: SIMD2(1.0,0.0), count: n*batchSize)
var fftoutput = [SIMD2<Float>](repeating: SIMD2(0.0,0.0), count: n*batchSize)
var ifftInput = [SIMD2<Float>](repeating: SIMD2(0.0,0.0), count: n*batchSize)
var ifftOutput = [SIMD2<Float>](repeating: SIMD2(0.0,0.0), count: n*batchSize)

fftoutput = FFT1D(inputData : fftInput, n: n, batchSize: batchSize)


ifftInput = fftoutput


ifftOutput = IFFT1D(inputData : ifftInput, n: n, batchSize: batchSize)
for i in 0..<n*batchSize{
    print("\(i)th", ifftOutput[i].x)
}
func FFT1D(inputData : [SIMD2<Float>] ,n : Int,batchSize : Int) -> [SIMD2<Float>]{
    let device = MTLCreateSystemDefaultDevice()!
    let commandQueue = device.makeCommandQueue()!
    let numElement = n*batchSize
    let inputByte = numElement*MemoryLayout<SIMD2<Float>>.stride
    let input = FFTComplexBuffer(device: device, capacity: numElement)
    let output = FFTComplexBuffer(device: device, capacity: numElement)
    let fft = FastFourierTransform1D(device: device,
                                                 transformWidth: n,
                                                 inputType: .interleavedComplex,
                                                 batchSize: batchSize)
   // print("execution time in milliseconds")
    let commandBuffer = commandQueue.makeCommandBuffer()!

    let inputDataComplex = (input.buffer.contents()+input.byteOffset).assumingMemoryBound(to: SIMD2<Float>.self)
    //copy input from inputDataReal
    memcpy(inputDataComplex, inputData, inputByte)
    fft.encode(commandBuffer: commandBuffer, input: input, output: output)
    //fft
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    var outputData = [SIMD2<Float>](repeating: SIMD2(0.0,0.0), count: numElement)
    let outputDataComplex = output.buffer.contents().bindMemory(to: SIMD2<Float>.self, capacity: numElement)
    for i in 0..<numElement{
        outputData[i] = outputDataComplex[i]
    }
    /*time check
     let elapsedTime = commandBuffer.gpuEndTime - commandBuffer.gpuStartTime
     print(elapsedTime * 1e3)
     */
    return outputData
}

func IFFT1D(inputData : [SIMD2<Float>] ,n : Int,batchSize : Int) -> [SIMD2<Float>]{
    let device = MTLCreateSystemDefaultDevice()!
    let commandQueue = device.makeCommandQueue()!
    let numElement = n*batchSize
    let inputByte = numElement*MemoryLayout<SIMD2<Float>>.stride
    let input = FFTComplexBuffer(device: device, capacity: numElement)
    let output = FFTComplexBuffer(device: device, capacity: numElement)
    let fft = InverseFastFourierTransform1D(device: device,
                                                 transformWidth: n,
                                                 inputType: .interleavedComplex,
                                                 batchSize: batchSize)
   // print("execution time in milliseconds")
    let commandBuffer = commandQueue.makeCommandBuffer()!

    let inputDataComplex = (input.buffer.contents()+input.byteOffset).assumingMemoryBound(to: SIMD2<Float>.self)
    //copy input from inputDataReal
    memcpy(inputDataComplex, inputData, inputByte)
    fft.encode(commandBuffer: commandBuffer, input: input, output: output)
    //fft
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    var outputData = [SIMD2<Float>](repeating: SIMD2(0.0,0.0), count: numElement)
    let outputDataComplex = output.buffer.contents().bindMemory(to: SIMD2<Float>.self, capacity: numElement)
    for i in 0..<numElement{
        outputData[i] = outputDataComplex[i]
    }
    /*time check
     let elapsedTime = commandBuffer.gpuEndTime - commandBuffer.gpuStartTime
     print(elapsedTime * 1e3)
     */
    return outputData
}

