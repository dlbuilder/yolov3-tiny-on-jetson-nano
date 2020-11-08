#pragma once

#include "TensorRTInferenceCore.h"

class TensorRTInferenceFilter
{
public:
	TensorRTInferenceFilter(TensorRTInferenceInternalData* internalData, TensorRTInferenceInputData* inputData, TensorRTInferenceOutputData* outputData) {
		mTensorRTInferenceCore = std::make_unique<TensorRTInferenceCore>(internalData, inputData, outputData);
	}

	~TensorRTInferenceFilter()
	{

	}

	FilterStatus RunFilterCoreLogic()
	{
		return mTensorRTInferenceCore->RunFilterCoreLogic();
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
	std::unique_ptr<TensorRTInferenceCore> mTensorRTInferenceCore;
};


