/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once
#include <iostream>
#include "cuda_runtime.h"
#include "helper_cuda.h"

namespace TinyYolov3PreProcessingJetsonKernel
{
void Convert_BGR24PlanarUINT8_to_BGR24PlanarFloat32(uint8_t* uint_b, uint8_t* uint_g, uint8_t* uint_r, float* float_bgr, int nSrcWidth, int nSrcHeight);
void Convert_BGR24InterleavedUINT8_to_BGR24PlanarFloat32(uint8_t* bgr24Interleaved, float* bgr24Planar, int nSrcWidth, int nSrcHeight);
void Convert_BGR24InterleavedUINT8_to_BGR24PlanarUINT8(uint8_t* bgr24Interleaved, uint8_t* b, uint8_t* g, uint8_t* r, int nSrcWidth, int nSrcHeight);
void Convert_RGBA32InterleavedUINT8_to_BGR24PlanarFloat32(uint8_t* rgba32Interleaved, float* bgr24Planar, int nSrcWidth, int nSrcHeight);
void ResizeAndConvert_BGR24PlanarFloat32_to_BGR24InterleavedFloat32_Batch(
    float* dpSrc, int nSrcPitch, int nSrcWidth, int nSrcHeight,
    float* dpDst, int nDstPitch, int nDstWidth, int nDstHeight,
    int nBatchSize, cudaStream_t stream = 0,
    int cropX = 0, int cropY = 0, int cropW = 0, int cropH = 0,
    bool whSameResizeRatio = false);

void Resize_BGR24PlanarFloat32_Batch(
    float* dpSrc, int nSrcPitch, int nSrcWidth, int nSrcHeight,
    float* dpDst, int nDstPitch, int nDstWidth, int nDstHeight,
    int nBatchSize, cudaStream_t stream = 0,
    int cropX = 0, int cropY = 0, int cropW = 0, int cropH = 0,
    bool whSameResizeRatio = false);
}

