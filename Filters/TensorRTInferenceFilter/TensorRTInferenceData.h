#pragma once

#include "../../Data/BaseData.h"

struct TensorRTInferenceInputData
{
	int imageWidth;
	int imageHeight;
	int resizedImageWidth;
	int resizedImageHeight;
	void *bgrData;
};

struct TensorRTInferenceInternalData
{
	int gpuID;
	std::string model;
	int batchSize;
};

struct TensorRTInferenceOutputData
{
	int imageWidth;
	int imageHeight;
	int resizedImageWidth;
	int resizedImageHeight;
	int tensorRTModelOutputCount;
	void *tensorRTModelOutputs[10];
	int64_t tensorRTModelOutputsSizes[10];
};
