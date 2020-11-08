#include "TensorRTTinyYolov3PostProcessingCore.h"

TensorRTTinyYolov3PostProcessingCore::TensorRTTinyYolov3PostProcessingCore(TensorRTTinyYolov3PostProcessingInternalData * internalData, TensorRTTinyYolov3PostProcessingInputData * inputData, TensorRTTinyYolov3PostProcessingOutputData * outputData)
{
	mInternalData = internalData;
	mInputData = inputData;
	mOutputData = outputData;

	mModelOutputHost = new float[mInputData->tensorRTModelOutputsSizes[0] / sizeof(float)];

	string filePath = "./DetectionClass.names";
	ReadDetectionClassNames(filePath, mOutputData->detectionClassNames);
}

TensorRTTinyYolov3PostProcessingCore::~TensorRTTinyYolov3PostProcessingCore()
{
	delete[] mModelOutputHost;
}

FilterStatus TensorRTTinyYolov3PostProcessingCore::RunFilterCoreLogic()
{
	float * modelOutput = (float*)mInputData->tensorRTModelOutputs[0];
	int64_t modelOutputSize = mInputData->tensorRTModelOutputsSizes[0];
	cudaMemcpy(mModelOutputHost, modelOutput, modelOutputSize, cudaMemcpyDeviceToHost);

	mOutputData->detections.clear();
	for (int i = 0; i < mModelOutputHost[0] && i < mInternalData->maxBoxCount; i++) {
		if (mModelOutputHost[1 + 7 * i + 4] <= mInternalData->objThreshold) continue;
		Detection det;
		std::memcpy(&det, &mModelOutputHost[1 + 7 * i], 7 * sizeof(float));
		if (mOutputData->detections.count(det.class_id) == 0) mOutputData->detections.emplace(det.class_id, std::vector<Detection>());
		mOutputData->detections[det.class_id].push_back(det);
	}
	return FilterStatus::COMPLETE;
}

void TensorRTTinyYolov3PostProcessingCore::ReadDetectionClassNames(string filePath, vector<string>& detectionClassNames)
{
	ifstream openFile(filePath.data());
	if (openFile.is_open()) {
		string line;
		while (getline(openFile, line)) {
			detectionClassNames.push_back(line);
		}
		openFile.close();
	}
	else
	{
		std::cerr << "Error Opening Detection Class Name File" << std::endl;
		exit(EXIT_FAILURE);
	}
}
