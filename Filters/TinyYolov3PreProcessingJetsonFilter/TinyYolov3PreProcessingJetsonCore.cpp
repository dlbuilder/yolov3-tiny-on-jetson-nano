#include "TinyYolov3PreProcessingJetsonCore.h"

using namespace TinyYolov3PreProcessingJetsonKernel;

static void out_ppmInterleavedRGBA(const char *name, uint8_t *s_abgr, int width, int height)
{
	uint8_t *abgr, *d_abgr;
	int frameSize = width * height * 4 * sizeof(uint8_t);
	abgr = (uint8_t *)malloc(frameSize);
	d_abgr = s_abgr;
	FILE *fpOut = fopen(name, "wb");
	(void)fprintf(fpOut, "P6\n%d %d\n255\n", width, height);
	char filename[120];
	std::ofstream *outputFile;
	cudaMemcpy(abgr, d_abgr, frameSize, cudaMemcpyDeviceToHost);
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			uint8_t color[3];
			color[0] = (int)(abgr[y * width * 4 + x * 4 + 0]);
			color[1] = (int)(abgr[y * width * 4 + x * 4 + 1]);
			color[2] = (int)(abgr[y * width * 4 + x * 4 + 2]);
			(void)fwrite(color, 1, 3, fpOut);
		}
	}
	free(abgr);
	fclose(fpOut);
}

static void out_ppm(const char *name, float *s_bgr, int width, int height)
{
	float *bgr, *d_bgr;
	int frameSize = width * height * 3 * sizeof(float);
	bgr = (float *)malloc(frameSize);
	d_bgr = s_bgr;
	FILE *fpOut = fopen(name, "wb");
	(void)fprintf(fpOut, "P6\n%d %d\n255\n", width, height);
	char filename[120];
	std::ofstream *outputFile;
	cudaMemcpy(bgr, d_bgr, frameSize, cudaMemcpyDeviceToHost);
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			uint8_t color[3];
			color[2] = (int)((bgr[y * width + x]) * 255);
			color[1] = (int)((bgr[y * width + x + width * height]) * 255);
			color[0] = (int)((bgr[y * width + x + 2 * width * height]) * 255);
			(void)fwrite(color, 1, 3, fpOut);
		}
	}
	free(bgr);
	fclose(fpOut);
}

static void out_ppmInterleaved(const char *name, float *s_bgr, int width, int height)
{
	float *bgr, *d_bgr;
	int frameSize = width * height * 3 * sizeof(float);
	bgr = (float *)malloc(frameSize);
	d_bgr = s_bgr;
	FILE *fpOut = fopen(name, "wb");
	(void)fprintf(fpOut, "P6\n%d %d\n255\n", width, height);
	char filename[120];
	std::ofstream *outputFile;
	cudaMemcpy(bgr, d_bgr, frameSize, cudaMemcpyDeviceToHost);
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			uint8_t color[3];
			color[2] = (int)(bgr[y * width * 3 + x * 3]);
			color[1] = (int)(bgr[y * width * 3 + x * 3 + 1]);
			color[0] = (int)(bgr[y * width * 3 + x * 3 + 2]);
			(void)fwrite(color, 1, 3, fpOut);
		}
	}
	free(bgr);
	fclose(fpOut);
}

TinyYolov3PreProcessingJetsonCore::TinyYolov3PreProcessingJetsonCore(TinyYolov3PreProcessingJetsonInternalData *internalData, TinyYolov3PreProcessingJetsonInputData *inputData, TinyYolov3PreProcessingJetsonOutputData *outputData)
{
	mInternalData = internalData;
	mInputData = inputData;
	mOutputData = outputData;

	AllocateOutputData();
	InitCudaInteropResources();
}

TinyYolov3PreProcessingJetsonCore::~TinyYolov3PreProcessingJetsonCore()
{
	DestroyCudaInteropResources();
	cudaFree(mOutputData->bgrData);
	cudaFree(mIntermediateData);
}

void TinyYolov3PreProcessingJetsonCore::AllocateOutputData()
{
	int channelSize = 3;
	int pixelTypeSize = sizeof(float);
	cudaMalloc(&mIntermediateData, mInputData->imageWidth * mInputData->imageHeight * channelSize * pixelTypeSize);
	cudaMalloc(&mOutputData->bgrData, mOutputData->resizedImageWidth * mOutputData->resizedImageHeight * channelSize * pixelTypeSize);
}

void TinyYolov3PreProcessingJetsonCore::InitCudaInteropResources()
{
	mEGLDisplay = eglGetDisplay(EGL_DEFAULT_DISPLAY);
	if (mEGLDisplay == EGL_NO_DISPLAY)
	{
		std::cerr << "Failed to get EGL display connection" << std::endl;
		exit(EXIT_FAILURE);
	}
	if (!eglInitialize(mEGLDisplay, NULL, NULL))
	{
		std::cerr << "Failed to initialize EGL display connection" << std::endl;
		exit(EXIT_FAILURE);
	}
	mEGLImage = NvEGLImageFromFd(mEGLDisplay, mInputData->dmaBufferFD);
	if (mEGLImage == NULL)
	{
		std::cerr << "Failed to map dma Buffer to EGLImage" << std::endl;
		exit(EXIT_FAILURE);
	}
	cuGraphicsEGLRegisterImage(&mCUGraphicsResource, mEGLImage, CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE);
}

void TinyYolov3PreProcessingJetsonCore::DestroyCudaInteropResources()
{
	NvDestroyEGLImage(mEGLDisplay, mEGLImage);
	mEGLImage = NULL;
	cuGraphicsUnregisterResource(mCUGraphicsResource);
	if (mEGLDisplay)
	{
		eglTerminate(mEGLDisplay);
	}
}

FilterStatus TinyYolov3PreProcessingJetsonCore::RunFilterCoreLogic()
{
	cuGraphicsResourceGetMappedEglFrame(&mEGLFrame, mCUGraphicsResource, 0, 0);
	//out_ppmInterleavedRGBA("abgr32.ppm", (uint8_t*)(mEGLFrame.frame.pPitch[0]), mInputData->imageWidth, mInputData->imageHeight);
	Convert_RGBA32InterleavedUINT8_to_BGR24PlanarFloat32((uint8_t *)mEGLFrame.frame.pPitch[0], (float *)mIntermediateData, mInputData->imageWidth, mInputData->imageHeight);
	Resize_BGR24PlanarFloat32_Batch((float *)mIntermediateData, mInputData->imageWidth, mInputData->imageWidth, mInputData->imageHeight,
									(float *)mOutputData->bgrData, mOutputData->resizedImageWidth, mOutputData->resizedImageWidth, mOutputData->resizedImageHeight, 1);
	return FilterStatus::COMPLETE;
}