#pragma once

#include "TinyYolov3PreProcessingJetsonCore.h"

class TinyYolov3PreProcessingJetsonFilter
{
public:
	TinyYolov3PreProcessingJetsonFilter(TinyYolov3PreProcessingJetsonInternalData* internalData, TinyYolov3PreProcessingJetsonInputData* inputData, TinyYolov3PreProcessingJetsonOutputData* outputData) 
	{
		mTinyYolov3PreProcessingJetsonCore = std::make_unique<TinyYolov3PreProcessingJetsonCore>(internalData, inputData, outputData);
	}

	~TinyYolov3PreProcessingJetsonFilter()
	{

	}

	FilterStatus RunFilterCoreLogic()
	{
		return mTinyYolov3PreProcessingJetsonCore->RunFilterCoreLogic();
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
	std::unique_ptr<TinyYolov3PreProcessingJetsonCore> mTinyYolov3PreProcessingJetsonCore;
};
