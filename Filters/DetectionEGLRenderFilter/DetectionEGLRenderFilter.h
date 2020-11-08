#pragma once

#include "DetectionEGLRenderCore.h"

class DetectionEGLRenderFilter
{
public:
	DetectionEGLRenderFilter(DetectionEGLRenderInternalData* internalData,DetectionEGLRenderImageInputData* imageinputData, DetectionEGLRenderBoundingBoxInputData* boxinputData) 
	{
		mDetectionEGLRenderCore = std::make_unique<DetectionEGLRenderCore>(internalData, imageinputData, boxinputData);
	}

	~DetectionEGLRenderFilter()
	{

	}

	FilterStatus RunFilterCoreLogic()
	{
		return mDetectionEGLRenderCore->RunFilterCoreLogic();
	}

	bool IsInputFilter()
	{
		return false;
	}

	bool IsOutPutFilter()
	{
		return true;
	}

private:
	std::unique_ptr<DetectionEGLRenderCore> mDetectionEGLRenderCore;
};