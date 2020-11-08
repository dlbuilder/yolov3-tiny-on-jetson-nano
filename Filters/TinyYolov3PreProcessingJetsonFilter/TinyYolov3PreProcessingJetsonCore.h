#pragma once

#include <cuda_runtime_api.h>
#include <memory>
#include "cudaEGL.h"
#include "nvbuf_utils.h"
#include "TinyYolov3PreProcessingJetsonData.h"
#include "TinyYolov3PreProcessingJetsonKernel.h"

class TinyYolov3PreProcessingJetsonCore
{
public:
	TinyYolov3PreProcessingJetsonCore(TinyYolov3PreProcessingJetsonInternalData* internalData, TinyYolov3PreProcessingJetsonInputData* inputData, TinyYolov3PreProcessingJetsonOutputData* outputData);
	~TinyYolov3PreProcessingJetsonCore();
	FilterStatus RunFilterCoreLogic();

private:
	void AllocateOutputData();
	void InitCudaInteropResources();
	void DestroyCudaInteropResources();
	TinyYolov3PreProcessingJetsonInternalData* mInternalData;
	TinyYolov3PreProcessingJetsonInputData* mInputData;
	TinyYolov3PreProcessingJetsonOutputData* mOutputData;
	EGLDisplay mEGLDisplay;
    EGLImageKHR mEGLImage;
    CUeglFrame mEGLFrame;
    CUgraphicsResource mCUGraphicsResource;	
	void* mIntermediateData;
};
