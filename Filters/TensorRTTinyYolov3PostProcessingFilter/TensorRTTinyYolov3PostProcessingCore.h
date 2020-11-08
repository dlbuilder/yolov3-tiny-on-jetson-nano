#pragma once

#include "TensorRTTinyYolov3PostProcessingData.h"
#include <iostream>
#include <fstream>
#include <memory>
#include <vector>
#include <cstring>
#include <string>
#include "cuda_runtime.h"

class TensorRTTinyYolov3PostProcessingCore
{
public:
	TensorRTTinyYolov3PostProcessingCore(TensorRTTinyYolov3PostProcessingInternalData* internalData, TensorRTTinyYolov3PostProcessingInputData* inputData, TensorRTTinyYolov3PostProcessingOutputData* outputData);
	~TensorRTTinyYolov3PostProcessingCore();
	FilterStatus RunFilterCoreLogic();

private:
	void ReadDetectionClassNames(string filePath, vector<string> &detectionClassNames);
	TensorRTTinyYolov3PostProcessingInternalData* mInternalData;
	TensorRTTinyYolov3PostProcessingInputData* mInputData;
	TensorRTTinyYolov3PostProcessingOutputData* mOutputData;

	float* mModelOutputHost;
};
