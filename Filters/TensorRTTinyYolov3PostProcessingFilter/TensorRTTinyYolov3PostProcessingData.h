#pragma once
#include "../../Data/BaseData.h"

struct TensorRTTinyYolov3PostProcessingInternalData
{
	float objThreshold;
	int maxBoxCount;
};

struct TensorRTTinyYolov3PostProcessingInputData
{
	int imageWidth;
	int imageHeight;
	int resizedImageWidth;
	int resizedImageHeight;
	int tensorRTModelOutputCount;
	void *tensorRTModelOutputs[10];
	int64_t tensorRTModelOutputsSizes[10];
};

struct TensorRTTinyYolov3PostProcessingOutputData
{
	map<float, std::vector<Detection>> detections;
	int imageWidth;
	int imageHeight;
	int resizedImageWidth;
	int resizedImageHeight;
	vector<string> detectionClassNames;
};
