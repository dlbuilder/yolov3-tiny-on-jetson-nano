#pragma once

#include "../../Data/BaseData.h"

struct DetectionEGLRenderInternalData
{
};

struct DetectionEGLRenderImageInputData
{
	int imageWidth;
	int imageHeight;
	int dmaBufferFD;
};

struct DetectionEGLRenderBoundingBoxInputData
{
	int imageWidth;
	int imageHeight;
	vector<Detection> nmsOutDetections;
	vector<string> detectionClassNames;
};
