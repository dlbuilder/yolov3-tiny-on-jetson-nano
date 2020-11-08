#include "TensorRTInferenceCore.h"

inline int64_t volume(const nvinfer1::Dims &d)
{
	return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
}

inline unsigned int getElementSize(nvinfer1::DataType t)
{
	switch (t)
	{
	case nvinfer1::DataType::kINT32:
		return 4;
	case nvinfer1::DataType::kFLOAT:
		return 4;
	case nvinfer1::DataType::kHALF:
		return 2;
	case nvinfer1::DataType::kINT8:
		return 1;
	}
	throw std::runtime_error("Invalid DataType.");
	return 0;
}

TensorRTInferenceCore::TensorRTInferenceCore(TensorRTInferenceInternalData* internalData, TensorRTInferenceInputData* inputData, TensorRTInferenceOutputData* outputData)
{
	mInternalData = internalData;
	mInputData = inputData;
	mOutputData = outputData;

	int devID = gpuDeviceInit(mInternalData->gpuID);
	if (devID < 0)
	{
		std::cerr << "GPU Init Failture" << std::endl;
		exit(EXIT_FAILURE);
	}

	mRuntime = createInferRuntime(gLogger);
	assert(mRuntime != nullptr);

	std::ifstream engineFile(mInternalData->model, std::ios::binary);

	if (!engineFile)
	{
		std::cerr << "Model File Read Failed" << std::endl;
		exit(EXIT_FAILURE);
	}

	engineFile.seekg(0, engineFile.end);
	long int fsize = engineFile.tellg();
	engineFile.seekg(0, engineFile.beg);

	std::vector<char> engineData(fsize);
	engineFile.read(engineData.data(), fsize);

	mEngine = mRuntime->deserializeCudaEngine(engineData.data(), fsize, nullptr);
	assert(mEngine != nullptr);

	mContext = mEngine->createExecutionContext();
	assert(mContext != nullptr);

	AllocateModelOutputDatas(mOutputData);
}

TensorRTInferenceCore::~TensorRTInferenceCore()
{
	mContext->destroy();
	mEngine->destroy();
	mRuntime->destroy();

	for (int i = 0; i < mOutputData->tensorRTModelOutputCount; ++i) 
	{
		cudaFree(mOutputData->tensorRTModelOutputs[i]);
	}
}

FilterStatus TensorRTInferenceCore::RunFilterCoreLogic()
{
	std::vector<void *> buffers;
	buffers.push_back(mInputData->bgrData);

	for (int i = 0; i < mOutputData->tensorRTModelOutputCount; ++i)
	{
		buffers.push_back(mOutputData->tensorRTModelOutputs[i]);
	}
	mContext->execute(mInternalData->batchSize, &buffers[0]);
	return FilterStatus::COMPLETE;
}

void TensorRTInferenceCore::AllocateModelOutputDatas(TensorRTInferenceOutputData* outputData)
{
	for (int i = 0; i < mEngine->getNbBindings(); ++i)
	{
		if (!mEngine->bindingIsInput(i))
		{
			outputData->tensorRTModelOutputCount++;
			nvinfer1::Dims dims = mEngine->getBindingDimensions(i);
			nvinfer1::DataType dtype = mEngine->getBindingDataType(i);
			int64_t outputSize = volume(dims) * 1 * getElementSize(dtype);
			cudaMalloc(&outputData->tensorRTModelOutputs[i - 1], outputSize);
			outputData->tensorRTModelOutputsSizes[i - 1] = outputSize; //Assume Input is first and only one
		}
	}
}