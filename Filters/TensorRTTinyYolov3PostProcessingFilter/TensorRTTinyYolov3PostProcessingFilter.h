#pragma once

#include "TensorRTTinyYolov3PostProcessingCore.h"

class TensorRTTinyYolov3PostProcessingFilter
{
public:
	TensorRTTinyYolov3PostProcessingFilter(TensorRTTinyYolov3PostProcessingInternalData* internalData, TensorRTTinyYolov3PostProcessingInputData* inputData, TensorRTTinyYolov3PostProcessingOutputData* outputData)
	{
		mTensorRTTinyYolov3PostProcessingCore = std::make_unique<TensorRTTinyYolov3PostProcessingCore>(internalData, inputData, outputData);
	}

	~TensorRTTinyYolov3PostProcessingFilter()
	{

	}

	FilterStatus RunFilterCoreLogic()
	{
		return mTensorRTTinyYolov3PostProcessingCore->RunFilterCoreLogic();
	}

	bool IsInputFilter()
	{
		return false;
	}

	bool IsOutPutFilter()
	{
		return false;
	}

private:
	std::unique_ptr<TensorRTTinyYolov3PostProcessingCore> mTensorRTTinyYolov3PostProcessingCore;
};
