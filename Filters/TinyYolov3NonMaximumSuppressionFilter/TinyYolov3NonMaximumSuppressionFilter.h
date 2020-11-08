#pragma once

#include "TinyYolov3NonMaximumSuppressionCore.h"

class TinyYolov3NonMaximumSuppressionFilter
{
public:
	TinyYolov3NonMaximumSuppressionFilter(TinyYolov3NonMaximumSuppressionInternalData* internalData, TinyYolov3NonMaximumSuppressionInputData* inputData, TinyYolov3NonMaximumSuppressionOutputData* outputData)
	{
		mTinyYolov3NonMaximumSuppressionCore=std::make_unique<TinyYolov3NonMaximumSuppressionCore>(internalData, inputData, outputData);
	}

	~TinyYolov3NonMaximumSuppressionFilter()
	{

	}

	FilterStatus RunFilterCoreLogic()
	{
		return mTinyYolov3NonMaximumSuppressionCore->RunFilterCoreLogic();
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
	std::unique_ptr<TinyYolov3NonMaximumSuppressionCore> mTinyYolov3NonMaximumSuppressionCore;
};

