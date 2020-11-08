#pragma once

#include <fstream>
#include <memory>
#include <numeric>
#include "NvInferRuntime.h"
#include "logging.h"
#include "cuda_runtime.h"
#include "helper_cuda.h"
#include "plugins.h"
#include "TensorRTInferenceData.h"

using namespace nvinfer1;
static sample::Logger gLogger;

class TensorRTInferenceCore
{
public:
	TensorRTInferenceCore(TensorRTInferenceInternalData* internalData, TensorRTInferenceInputData* inputData, TensorRTInferenceOutputData* outputData);
	~TensorRTInferenceCore();
	FilterStatus RunFilterCoreLogic();

private:
	void AllocateModelOutputDatas(TensorRTInferenceOutputData* outputData);
	
	TensorRTInferenceInternalData* mInternalData;
	TensorRTInferenceInputData* mInputData;
	TensorRTInferenceOutputData* mOutputData;

	IRuntime* mRuntime;
	ICudaEngine* mEngine;
	IExecutionContext* mContext;
};