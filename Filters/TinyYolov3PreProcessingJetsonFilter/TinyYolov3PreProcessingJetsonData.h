#pragma once

#include "../../Data/BaseData.h"

struct TinyYolov3PreProcessingJetsonInternalData
{
	int inferenceEngine; //0: TensorRT, 1: Tensorflow, 2: Pytorch
};

struct TinyYolov3PreProcessingJetsonInputData
{
	int imageWidth;
	int imageHeight;
	int dmaBufferFD;
};

struct TinyYolov3PreProcessingJetsonOutputData
{
	int imageWidth;
	int imageHeight;
	int resizedImageWidth;
	int resizedImageHeight;
	void *bgrData;
};
