#include "DetectionEGLRenderCore.h"

DetectionEGLRenderCore::DetectionEGLRenderCore(DetectionEGLRenderInternalData *internalData, DetectionEGLRenderImageInputData *imageinputData, DetectionEGLRenderBoundingBoxInputData *boxinputData)
{
	mInternalData = internalData;
	mImageInputData = imageinputData;
	mBoxInputData = boxinputData;

	InitRenderer();
	InitOSD();
}

void DetectionEGLRenderCore::InitRenderer()
{
	mRenderer = NvEglRenderer::createEglRenderer("renderer0",
												 mImageInputData->imageWidth, mImageInputData->imageHeight, 0, 0);
	if (!mRenderer)
	{
		std::cerr << "Failed to create EGL renderer" << std::endl;
		exit(EXIT_FAILURE);
	}
	mRenderer->setFPS(RENDER_FPS);
}

void DetectionEGLRenderCore::DestroyRenderer()
{
	if (mRenderer != NULL)
	{
		delete mRenderer;
		mRenderer = NULL;
	}
}

void DetectionEGLRenderCore::InitOSD()
{
	mNVOSDContext = nvosd_create_context();
	if (!mNVOSDContext)
	{
		std::cerr << "Failed to create NVOSD" << std::endl;
		exit(EXIT_FAILURE);
	}
}

void DetectionEGLRenderCore::DestroyOSD()
{
	if (mNVOSDContext != NULL)
	{
		nvosd_destroy_context(mNVOSDContext);
		mNVOSDContext = NULL;
	}
}

DetectionEGLRenderCore::~DetectionEGLRenderCore()
{
	DestroyRenderer();
	DestroyOSD();
}

void DetectionEGLRenderCore::DrawBoundingBox(std::vector<Detection> detections)
{
	int boxCount = detections.size();
	NvOSD_RectParams rectParams[boxCount];
	NvOSD_TextParams textParams[boxCount];
	if (boxCount > 0)
	{
		for (int i = 0; i < boxCount; i++)
		{
			if (detections[i].bbox[0] < 0)
				detections[i].bbox[0] = 0;
			if (detections[i].bbox[1] < 0)
				detections[i].bbox[1] = 0;
			if (detections[i].bbox[0] + detections[i].bbox[2] >= mImageInputData->imageWidth)
				detections[i].bbox[2] = mImageInputData->imageWidth - detections[i].bbox[0];
			if (detections[i].bbox[1] + detections[i].bbox[3] >= mImageInputData->imageHeight)
				detections[i].bbox[3] = mImageInputData->imageHeight - detections[i].bbox[1];
			rectParams[i].left = detections[i].bbox[0];
			rectParams[i].top = detections[i].bbox[1];
			rectParams[i].width = detections[i].bbox[2];
			rectParams[i].height = detections[i].bbox[3];
			rectParams[i].border_width = 3;
			rectParams[i].border_color.red = 0;
			rectParams[i].border_color.green = 0;
			rectParams[i].border_color.blue = 1;
			rectParams[i].has_bg_color = 0;

			textParams[i].display_text = strdup(mBoxInputData->detectionClassNames[detections[i].class_id].c_str());
			textParams[i].x_offset = detections[i].bbox[0];
			textParams[i].y_offset = detections[i].bbox[1] - 20;
			textParams[i].font_params.font_name = strdup("Arial");
			textParams[i].font_params.font_size = 10;
			textParams[i].font_params.font_color.red = 0.0;
			textParams[i].font_params.font_color.green = 0.0;
			textParams[i].font_params.font_color.blue = 1.0;
			textParams[i].font_params.font_color.alpha = 1.0;
		}
		nvosd_draw_rectangles(mNVOSDContext,
							  MODE_HW,
							  mImageInputData->dmaBufferFD,
							  boxCount,
							  rectParams);
		nvosd_put_text(mNVOSDContext,
					   MODE_CPU,
					   mImageInputData->dmaBufferFD,
					   boxCount,
					   textParams);
		detections.clear();
	}
}

FilterStatus DetectionEGLRenderCore::RunFilterCoreLogic()
{
	DrawBoundingBox(mBoxInputData->nmsOutDetections);
	mRenderer->render(mImageInputData->dmaBufferFD);
	return FilterStatus::COMPLETE;
}
