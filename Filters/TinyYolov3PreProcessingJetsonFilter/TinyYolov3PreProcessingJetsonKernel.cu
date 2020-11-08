/*
 * Copyright 1993-2019 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include "TinyYolov3PreProcessingJetsonKernel.h"
namespace TinyYolov3PreProcessingJetsonKernel
{
	__global__ void Convert_BGR24PlanarUINT8_to_BGR24PlanarFloat32_Kernel(uint8_t *bSrcData, uint8_t *gSrcData, uint8_t *rSrcData, float *bgrPlanarDstData, int nSrcWidth, int nSrcHeight)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		float *p = NULL;

		p = bgrPlanarDstData + (y * nSrcWidth) + x;
		*p = *(bSrcData + x + (y * nSrcWidth));
		*(p + (nSrcWidth * nSrcHeight)) = *(gSrcData + x + (y * nSrcWidth));
		*(p + 2 * (nSrcWidth * nSrcHeight)) = *(rSrcData + x + (y * nSrcWidth));
	}

	void Convert_BGR24PlanarUINT8_to_BGR24PlanarFloat32_Core(uint8_t *bSrcData, uint8_t *gSrcData, uint8_t *rSrcData, float *bgrPlanarDstData, int nSrcWidth, int nSrcHeight)
	{
		dim3 threads(64, 10);
		size_t blockDimZ = 1;
		blockDimZ = (blockDimZ > 32) ? 32 : blockDimZ;
		dim3 blocks((nSrcWidth - 1) / threads.x + 1, (nSrcHeight - 1) / threads.y + 1, blockDimZ);
		Convert_BGR24PlanarUINT8_to_BGR24PlanarFloat32_Kernel<<<blocks, threads>>>(bSrcData, gSrcData, rSrcData, bgrPlanarDstData, nSrcWidth, nSrcHeight);
	}

	void Convert_BGR24PlanarUINT8_to_BGR24PlanarFloat32(uint8_t *bSrcData, uint8_t *gSrcData, uint8_t *rSrcData, float *bgrPlanarDstData, int nSrcWidth, int nSrcHeight)
	{
		Convert_BGR24PlanarUINT8_to_BGR24PlanarFloat32_Core(bSrcData, gSrcData, rSrcData, bgrPlanarDstData, nSrcWidth, nSrcHeight);
	}

	__global__ void Convert_BGR24InterleavedUINT8_to_BGR24PlanarFloat32_Kernel(uint8_t *bgr24Interleaved, float *bgr24Planar, int nSrcWidth, int nSrcHeight)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		float *p = NULL;

		p = bgr24Planar + (y * nSrcWidth) + x;
		*p = *(bgr24Interleaved + 3 * x + (y * nSrcWidth * 3));
		*(p + (nSrcWidth * nSrcHeight)) = *(bgr24Interleaved + (3 * x + 1) + (y * nSrcWidth * 3));
		*(p + 2 * (nSrcWidth * nSrcHeight)) = *(bgr24Interleaved + (3 * x + 2) + (y * nSrcWidth * 3));
	}

	void Convert_BGR24InterleavedUINT8_to_BGR24PlanarFloat32_Core(uint8_t *bgr24Interleaved, float *bgr24Planar, int nSrcWidth, int nSrcHeight)
	{
		dim3 threads(64, 10);
		size_t blockDimZ = 1;
		blockDimZ = (blockDimZ > 32) ? 32 : blockDimZ;
		dim3 blocks((nSrcWidth - 1) / threads.x + 1, (nSrcHeight - 1) / threads.y + 1, blockDimZ);
		Convert_BGR24InterleavedUINT8_to_BGR24PlanarFloat32_Kernel<<<blocks, threads>>>(bgr24Interleaved, bgr24Planar, nSrcWidth, nSrcHeight);
	}

	void Convert_BGR24InterleavedUINT8_to_BGR24PlanarFloat32(uint8_t *bgr24Interleaved, float *bgr24Planar, int nSrcWidth, int nSrcHeight)
	{
		Convert_BGR24InterleavedUINT8_to_BGR24PlanarFloat32_Core(bgr24Interleaved, bgr24Planar, nSrcWidth, nSrcHeight);
	}

	__global__ void Convert_BGR24InterleavedUINT8_to_BGR24PlanarUINT8_Kernel(uint8_t *bgr24Interleaved, uint8_t *b, uint8_t *g, uint8_t *r, int nSrcWidth, int nSrcHeight)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		uint8_t *b_dest = NULL;
		uint8_t *g_dest = NULL;
		uint8_t *r_dest = NULL;

		b_dest = b + (y * nSrcWidth) + x;
		g_dest = g + (y * nSrcWidth) + x;
		r_dest = r + (y * nSrcWidth) + x;

		*b_dest = *(bgr24Interleaved + 3 * x + (y * nSrcWidth * 3));
		*g_dest = *(bgr24Interleaved + (3 * x + 1) + (y * nSrcWidth * 3));
		*r_dest = *(bgr24Interleaved + (3 * x + 2) + (y * nSrcWidth * 3));
	}

	void Convert_BGR24InterleavedUINT8_to_BGR24PlanarUINT8_Core(uint8_t *bgr24Interleaved, uint8_t *b, uint8_t *g, uint8_t *r, int nSrcWidth, int nSrcHeight)
	{
		dim3 threads(64, 10);
		size_t blockDimZ = 1;
		blockDimZ = (blockDimZ > 32) ? 32 : blockDimZ;
		dim3 blocks((nSrcWidth - 1) / threads.x + 1, (nSrcHeight - 1) / threads.y + 1, blockDimZ);
		Convert_BGR24InterleavedUINT8_to_BGR24PlanarUINT8_Kernel<<<blocks, threads>>>(bgr24Interleaved, b, g, r, nSrcWidth, nSrcHeight);
	}

	void Convert_BGR24InterleavedUINT8_to_BGR24PlanarUINT8(uint8_t *bgr24Interleaved, uint8_t *b, uint8_t *g, uint8_t *r, int nSrcWidth, int nSrcHeight)
	{
		Convert_BGR24InterleavedUINT8_to_BGR24PlanarUINT8_Core(bgr24Interleaved, b, g, r, nSrcWidth, nSrcHeight);
	}

	__global__ void Convert_RGBA32InterleavedUINT8_to_BGR24PlanarFloat32_Kernel(uint8_t *rgba32Interleaved, float *bgr24Planar, int nSrcWidth, int nSrcHeight)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		float *p = NULL;

		p = bgr24Planar + (y * nSrcWidth) + x;
		*(p + 2 * (nSrcWidth * nSrcHeight)) = *(rgba32Interleaved + 4 * x + (y * nSrcWidth * 4));
		*(p + (nSrcWidth * nSrcHeight)) = *(rgba32Interleaved + (4 * x + 1) + (y * nSrcWidth * 4));
		*p = *(rgba32Interleaved + (4 * x + 2) + (y * nSrcWidth * 4));
	}

	void Convert_RGBA32InterleavedUINT8_to_BGR24PlanarFloat32_Core(uint8_t *rgba32Interleaved, float *bgr24Planar, int nSrcWidth, int nSrcHeight)
	{
		dim3 threads(64, 10);
		size_t blockDimZ = 1;
		blockDimZ = (blockDimZ > 32) ? 32 : blockDimZ;
		dim3 blocks((nSrcWidth - 1) / threads.x + 1, (nSrcHeight - 1) / threads.y + 1, blockDimZ);
		Convert_RGBA32InterleavedUINT8_to_BGR24PlanarFloat32_Kernel<<<blocks, threads>>>(rgba32Interleaved, bgr24Planar, nSrcWidth, nSrcHeight);
	}

	void Convert_RGBA32InterleavedUINT8_to_BGR24PlanarFloat32(uint8_t *rgba32Interleaved, float *bgr24Planar, int nSrcWidth, int nSrcHeight)
	{
		Convert_RGBA32InterleavedUINT8_to_BGR24PlanarFloat32_Core(rgba32Interleaved, bgr24Planar, nSrcWidth, nSrcHeight);
	}

	__global__ void ResizeAndConvert_BGR24PlanarFloat32_to_BGR24InterleavedFloat32_Batch_Kernel(cudaTextureObject_t texSrc,
																								float *pDst, int nDstPitch, int nDstHeight, int nSrcHeight,
																								int batch, float scaleX, float scaleY,
																								int cropX, int cropY, int cropW, int cropH)
	{
		int dstX = 3 * (threadIdx.x + blockIdx.x * blockDim.x);
		int dstY = threadIdx.y + blockIdx.y * blockDim.y;
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if (x >= (int)(cropW / scaleX) || y >= (int)(cropH / scaleY))
			return;

		int frameSize = nDstPitch * nDstHeight;
		
		//ToDo Handle Width Aspect Ratio
		int newH = ((cropH / (float)cropW) * nDstHeight);
		float newScaleH = nDstHeight / (float)newH;
		int padOffset = (nDstHeight - newH) / 2;

		float *p = NULL;
		for (int i = blockIdx.z; i < batch; i += gridDim.z)
		{
#pragma unroll
			for (int channel = 0; channel < 3; channel++)
			{
				p = pDst + i * frameSize + dstY * nDstPitch + dstX + channel;
				if (y < padOffset || y > padOffset + newH)
				{
					*p = 0.5f;
				}
				else
				{
					*p = tex2D<float>(texSrc, x * scaleX + cropX, ((3 * i + channel) * nSrcHeight + (y - padOffset) * newScaleH * scaleY + cropY));
				}
				//Tensorflow model we use includes Noramlization in Model. We don't need to normalize here.
			}
		}
	}

	void ResizeAndConvert_BGR24PlanarFloat32_to_BGR24InterleavedFloat32_Batch_Core(
		float *dpSrc, int nSrcPitch, int nSrcWidth, int nSrcHeight,
		float *dpDst, int nDstPitch, int nDstWidth, int nDstHeight,
		int nBatchSize, cudaStream_t stream, bool whSameResizeRatio,
		int cropX, int cropY, int cropW, int cropH)
	{
		cudaTextureObject_t texSrc[2];
		int nTiles = 1, h, iTile;

		h = nSrcHeight * 3 * nBatchSize;
		while ((h + nTiles - 1) / nTiles > 65536)
			nTiles++;

		if (nTiles > 2)
			return;

		int batchTile = nBatchSize / nTiles;
		int batchTileLast = nBatchSize - batchTile * (nTiles - 1);

		for (iTile = 0; iTile < nTiles; ++iTile)
		{
			int bs = (iTile == nTiles - 1) ? batchTileLast : batchTile;
			float *dpSrcNew = dpSrc +
							  iTile * (batchTile * 3 * nSrcHeight * nSrcPitch);

			cudaResourceDesc resDesc = {};
			resDesc.resType = cudaResourceTypePitch2D;
			resDesc.res.pitch2D.devPtr = dpSrcNew;
			resDesc.res.pitch2D.desc = cudaCreateChannelDesc<float>();
			resDesc.res.pitch2D.width = nSrcWidth;
			resDesc.res.pitch2D.height = bs * 3 * nSrcHeight;
			resDesc.res.pitch2D.pitchInBytes = nSrcPitch * sizeof(float);
			cudaTextureDesc texDesc = {};
			texDesc.filterMode = cudaFilterModeLinear;
			texDesc.readMode = cudaReadModeElementType;

			checkCudaErrors(cudaCreateTextureObject(&texSrc[iTile], &resDesc, &texDesc, NULL));
			float *dpDstNew = dpDst +
							  iTile * (batchTile * 3 * nDstHeight * nDstPitch);

			if (cropW == 0 || cropH == 0)
			{
				cropX = 0;
				cropY = 0;
				cropW = nSrcWidth;
				cropH = nSrcHeight;
			}

			float scaleX = (cropW * 1.0f / nDstWidth);
			float scaleY = (cropH * 1.0f / nDstHeight);

			if (whSameResizeRatio == true)
				scaleX = scaleY = scaleX > scaleY ? scaleX : scaleY;
			dim3 block(32, 32, 1);

			size_t blockDimZ = bs;
			// Restricting blocks in Z-dim till 32 to not launch too many blocks
			blockDimZ = (blockDimZ > 32) ? 32 : blockDimZ;
			dim3 grid((cropW * 1.0f / scaleX + block.x - 1) / block.x,
					  (cropH * 1.0f / scaleY + block.y - 1) / block.y, blockDimZ);

			ResizeAndConvert_BGR24PlanarFloat32_to_BGR24InterleavedFloat32_Batch_Kernel<<<grid, block, 0, stream>>>(texSrc[iTile], dpDstNew, nDstPitch, nDstHeight, nSrcHeight,
																													bs, scaleX, scaleY, cropX, cropY, cropW, cropH);
		}

		for (iTile = 0; iTile < nTiles; ++iTile)
			checkCudaErrors(cudaDestroyTextureObject(texSrc[iTile]));
	}

	void ResizeAndConvert_BGR24PlanarFloat32_to_BGR24InterleavedFloat32_Batch(
		float *dpSrc, int nSrcPitch, int nSrcWidth, int nSrcHeight,
		float *dpDst, int nDstPitch, int nDstWidth, int nDstHeight,
		int nBatchSize, cudaStream_t stream,
		int cropX, int cropY, int cropW, int cropH, bool whSameResizeRatio)
	{
		ResizeAndConvert_BGR24PlanarFloat32_to_BGR24InterleavedFloat32_Batch_Core(dpSrc, nSrcPitch, nSrcWidth, nSrcHeight,
																				  dpDst, nDstPitch, nDstWidth, nDstHeight, nBatchSize, stream,
																				  whSameResizeRatio, cropX, cropY, cropW, cropH);
	}

	__global__ void Resize_BGR24PlanarFloat32_Batch_Kernel(cudaTextureObject_t texSrc,
														   float *pDst, int nDstPitch, int nDstHeight, int nSrcHeight,
														   int batch, float scaleX, float scaleY,
														   int cropX, int cropY, int cropW, int cropH)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if (x >= (int)(cropW / scaleX) || y >= (int)(cropH / scaleY))
			return;

		int frameSize = nDstPitch * nDstHeight;
		float *p = NULL;

		//ToDo Handle Width Aspect Ratio
		int newH = ((cropH / (float)cropW) * nDstHeight);
		float newScaleH = nDstHeight / (float)newH;
		int padOffset = (nDstHeight - newH) / 2;

		for (int i = blockIdx.z; i < batch; i += gridDim.z)
		{
#pragma unroll
			for (int channel = 0; channel < 3; channel++)
			{
				p = pDst + i * 3 * frameSize + y * nDstPitch + x + channel * frameSize;

				if (y < padOffset || y > padOffset + newH)
				{
					*p = 0.5f;
				}
				else
				{
					*p = tex2D<float>(texSrc, x * scaleX + cropX, ((3 * i + channel) * nSrcHeight + (y - padOffset) * newScaleH * scaleY + cropY));
				}
				*p = (*p / 255.0f); //Normalization of TensorRT, Pytorch Model Input
			}
		}
	}

	void Resize_BGR24PlanarFloat32_Batch_Core(
		float *dpSrc, int nSrcPitch, int nSrcWidth, int nSrcHeight,
		float *dpDst, int nDstPitch, int nDstWidth, int nDstHeight,
		int nBatchSize, cudaStream_t stream, bool whSameResizeRatio,
		int cropX, int cropY, int cropW, int cropH)
	{
		cudaTextureObject_t texSrc[2];
		int nTiles = 1, h, iTile;

		h = nSrcHeight * 3 * nBatchSize;
		while ((h + nTiles - 1) / nTiles > 65536)
			nTiles++;

		if (nTiles > 2)
			return;

		int batchTile = nBatchSize / nTiles;
		int batchTileLast = nBatchSize - batchTile * (nTiles - 1);

		for (iTile = 0; iTile < nTiles; ++iTile)
		{
			int bs = (iTile == nTiles - 1) ? batchTileLast : batchTile;
			float *dpSrcNew = dpSrc +
							  iTile * (batchTile * 3 * nSrcHeight * nSrcPitch);

			cudaResourceDesc resDesc = {};
			resDesc.resType = cudaResourceTypePitch2D;
			resDesc.res.pitch2D.devPtr = dpSrcNew;
			resDesc.res.pitch2D.desc = cudaCreateChannelDesc<float>();
			resDesc.res.pitch2D.width = nSrcWidth;
			resDesc.res.pitch2D.height = bs * 3 * nSrcHeight;
			resDesc.res.pitch2D.pitchInBytes = nSrcPitch * sizeof(float);
			cudaTextureDesc texDesc = {};
			texDesc.filterMode = cudaFilterModeLinear;
			texDesc.readMode = cudaReadModeElementType;

			cudaCreateTextureObject(&texSrc[iTile], &resDesc, &texDesc, NULL);
			float *dpDstNew = dpDst +
							  iTile * (batchTile * 3 * nDstHeight * nDstPitch);

			if (cropW == 0 || cropH == 0)
			{
				cropX = 0;
				cropY = 0;
				cropW = nSrcWidth;
				cropH = nSrcHeight;
			}

			float scaleX = (cropW * 1.0f / nDstWidth);
			float scaleY = (cropH * 1.0f / nDstHeight);

			if (whSameResizeRatio == true)
				scaleX = scaleY = scaleX > scaleY ? scaleX : scaleY;
			dim3 block(32, 32, 1);

			size_t blockDimZ = bs;
			// Restricting blocks in Z-dim till 32 to not launch too many blocks
			blockDimZ = (blockDimZ > 32) ? 32 : blockDimZ;
			dim3 grid((cropW * 1.0f / scaleX + block.x - 1) / block.x,
					  (cropH * 1.0f / scaleY + block.y - 1) / block.y, blockDimZ);

			Resize_BGR24PlanarFloat32_Batch_Kernel<<<grid, block, 0, stream>>>(texSrc[iTile], dpDstNew, nDstPitch, nDstHeight, nSrcHeight, bs, scaleX, scaleY, cropX, cropY, cropW, cropH);
		}

		for (iTile = 0; iTile < nTiles; ++iTile)
			cudaDestroyTextureObject(texSrc[iTile]);
	}

	void Resize_BGR24PlanarFloat32_Batch(
		float *dpSrc, int nSrcPitch, int nSrcWidth, int nSrcHeight,
		float *dpDst, int nDstPitch, int nDstWidth, int nDstHeight,
		int nBatchSize, cudaStream_t stream,
		int cropX, int cropY, int cropW, int cropH, bool whSameResizeRatio)
	{
		Resize_BGR24PlanarFloat32_Batch_Core(dpSrc, nSrcPitch, nSrcWidth, nSrcHeight,
											 dpDst, nDstPitch, nDstWidth, nDstHeight, nBatchSize, stream,
											 whSameResizeRatio, cropX, cropY, cropW, cropH);
	}
} // namespace TinyYolov3PreProcessingKernel