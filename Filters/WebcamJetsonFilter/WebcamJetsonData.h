#pragma once

#include "../../Data/BaseData.h"

struct WebcamInternalData
{
	int webcamId;
};

struct WebcamOutputData
{
	int imageWidth;
	int imageHeight;
	int dmaBufferFD;
};
