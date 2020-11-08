#pragma once

#include "TinyYolov3NonMaximumSuppressionData.h"
#include <iostream>
#include <numeric>
#include <vector>
#include <algorithm>

class TinyYolov3NonMaximumSuppressionCore
{
public:
	TinyYolov3NonMaximumSuppressionCore(TinyYolov3NonMaximumSuppressionInternalData* internalData, TinyYolov3NonMaximumSuppressionInputData* inputData, TinyYolov3NonMaximumSuppressionOutputData* outputData);
	~TinyYolov3NonMaximumSuppressionCore();
	FilterStatus RunFilterCoreLogic();

private:
	void DoNms();
	float IOU(float lbox[4], float rbox[4]);
	void ChangeRectSize(float* bbox);

	TinyYolov3NonMaximumSuppressionInternalData* mInternalData;
	TinyYolov3NonMaximumSuppressionInputData* mInputData;
	TinyYolov3NonMaximumSuppressionOutputData* mOutputData;
};

