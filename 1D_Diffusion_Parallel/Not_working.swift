//
//  Not_working.swift
//  1D_Diffusion_Parallel
//
//  Created by 조일현 on 2023/07/21.
//

import Foundation
import Accelerate
import MetalFFT
//no use of ICB;
//func testRandom1D(width: Int, isInverse: Bool, usingICB: Bool = false) throws {

func testRandom1D(width: Int, isInverse: Bool) throws {
    // set GPU, commandQueue, defaultLibrary, kernelFunction
    guard
        let device  = MTLCreateSystemDefaultDevice(),
        let commandQueue = device.makeCommandQueue()
    else {fatalError()}

    /* no use of ICB
     
     // set ICB
    if usingICB {
        guard canUseICBs(device: device) else {
            return
        }
    }
    */
    //parameters
    let numElement = width
    let bufferSize = MemoryLayout<Float>.stride * numElement

    // allocate GPU-accessible buffers
    let input = FFTRealBuffer(device: device, capacity: numElement)
    let output = FFTComplexBuffer(device: device, capacity: numElement)
    // fill inputData
    var inputData : [Float] = [Float](Array(repeating: 0.0, count: numElement))
    for i in 0..<numElement{
        inputData[i] = 1.0
    }
    let inputPointer = input.buffer.contents()
    memcpy(inputPointer, inputData, bufferSize)
    let commandBuffer = commandQueue.makeDebugCommandBuffer()
#if os(macOS)
    if input.buffer.storageMode == .shared {
        // Synchronize `input.buffer` between CPU and GPU
    }
#endif
    var fft: FastFourierTransform
    if isInverse {
        fft = InverseFastFourierTransform1D(device: device,
                                            transformWidth: width,
                                            inputType: .interleavedComplex)
    } else {
        fft = FastFourierTransform1D(device: device,
                                     transformWidth: width,
                                     inputType: .interleavedComplex)
    }
   /* no use of ICB
    if usingICB {
        fft.encodeAsICB(commandBuffer: commandBuffer, input: input, output: output)
    } else {
        fft.encode(commandBuffer: commandBuffer, input: input, output: output)
    }
    */
    fft.encode(commandBuffer: commandBuffer, input: input, output: output)

#if os(macOS)
    if output.buffer.storageMode == .shared { // `.shared` on M1
    }
#endif
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    
}

