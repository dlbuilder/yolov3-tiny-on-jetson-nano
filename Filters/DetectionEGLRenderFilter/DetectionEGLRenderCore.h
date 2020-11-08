#pragma once

#include <iostream>
#include <vector>
#include "NvEglRenderer.h"
#include "nvosd.h"
#include "DetectionEGLRenderData.h"

#define RENDER_FPS 30

class DetectionEGLRenderCore
{
public:
	DetectionEGLRenderCore(DetectionEGLRenderInternalData* internalData, DetectionEGLRenderImageInputData* imageInputData, DetectionEGLRenderBoundingBoxInputData* boxInputData);
	~DetectionEGLRenderCore();
	FilterStatus RunFilterCoreLogic();

private:
	void InitRenderer();
	void DestroyRenderer();
	void InitOSD();
	void DestroyOSD();
	void DrawBoundingBox(std::vector<Detection> detections);

	DetectionEGLRenderInternalData* mInternalData;
	DetectionEGLRenderImageInputData* mImageInputData;
	DetectionEGLRenderBoundingBoxInputData* mBoxInputData;

    NvEglRenderer *mRenderer;
	void *mNVOSDContext;
};