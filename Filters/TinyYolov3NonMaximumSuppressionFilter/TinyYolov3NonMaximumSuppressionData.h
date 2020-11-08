#pragma once

#include "../../Data/BaseData.h"

struct TinyYolov3NonMaximumSuppressionInputData
{
	map<float, std::vector<Detection>> detections;
	int imageWidth;
	int imageHeight;
	int resizedImageWidth;
	int resizedImageHeight;
	vector<string> detectionClassNames;
};

struct TinyYolov3NonMaximumSuppressionOutputData
{
	int imageWidth;
	int imageHeight;
	vector<Detection> nmsOutDetections;
	vector<string> detectionClassNames;
};

struct TinyYolov3NonMaximumSuppressionInternalData
{
	float nmsThreshold;
};
