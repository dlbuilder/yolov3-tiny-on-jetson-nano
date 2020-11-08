// reference : https://github.com/wang-xinyu/tensorrtx

#pragma once

#include <vector>
#include <string>
#include "NvInfer.h"
#include <algorithm>
#include "cudnn.h"
#include <iostream>

#ifndef CUDA_CHECK

#define CUDA_CHECK(callstr)                                                                    \
    {                                                                                          \
        cudaError_t error_code = callstr;                                                      \
        if (error_code != cudaSuccess) {                                                       \
            std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":" << __LINE__; \
            assert(0);                                                                         \
        }                                                                                      \
    }

#endif

namespace Tn
{
	class Profiler : public nvinfer1::IProfiler
	{
	public:
		void printLayerTimes(int itrationsTimes)
		{
			float totalTime = 0;
			for (size_t i = 0; i < mProfile.size(); i++)
			{
				printf("%-40.40s %4.3fms\n", mProfile[i].first.c_str(), mProfile[i].second / itrationsTimes);
				totalTime += mProfile[i].second;
			}
			printf("Time over all layers: %4.3f\n", totalTime / itrationsTimes);
		}
	private:
		typedef std::pair<std::string, float> Record;
		std::vector<Record> mProfile;

		virtual void reportLayerTime(const char* layerName, float ms)
		{
			auto record = std::find_if(mProfile.begin(), mProfile.end(), [&](const Record& r) { return r.first == layerName; });
			if (record == mProfile.end())
				mProfile.push_back(std::make_pair(layerName, ms));
			else
				record->second += ms;
		}
	};

	//Logger for TensorRT info/warning/errors
	class Logger : public nvinfer1::ILogger
	{
	public:

		Logger() : Logger(Severity::kWARNING) {}

		Logger(Severity severity) : reportableSeverity(severity) {}

		void log(Severity severity, const char* msg) override
		{
			// suppress messages with severity enum value greater than the reportable
			if (severity > reportableSeverity) return;

			switch (severity)
			{
			case Severity::kINTERNAL_ERROR: std::cerr << "INTERNAL_ERROR: "; break;
			case Severity::kERROR: std::cerr << "ERROR: "; break;
			case Severity::kWARNING: std::cerr << "WARNING: "; break;
			case Severity::kINFO: std::cerr << "INFO: "; break;
			default: std::cerr << "UNKNOWN: "; break;
			}
			std::cerr << msg << std::endl;
		}

		Severity reportableSeverity{ Severity::kWARNING };
	};

	template<typename T>
	void write(char*& buffer, const T& val)
	{
		*reinterpret_cast<T*>(buffer) = val;
		buffer += sizeof(T);
	}

	template<typename T>
	void read(const char*& buffer, T& val)
	{
		val = *reinterpret_cast<const T*>(buffer);
		buffer += sizeof(T);
	}
}

namespace Yolo
{
    static constexpr int CHECK_COUNT = 3;
    static constexpr float IGNORE_THRESH = 0.1f;
    static constexpr int MAX_OUTPUT_BBOX_COUNT = 1000;
    static constexpr int CLASS_NUM = 80;
    static constexpr int INPUT_H = 416;
    static constexpr int INPUT_W = 416;

    struct YoloKernel
    {
        int width;
        int height;
        float anchors[CHECK_COUNT*2];
    };

    static constexpr YoloKernel yolo1 = {
        INPUT_W / 32,
        INPUT_H / 32,
		{81,82,  135,169,  344,319}
    };

    static constexpr YoloKernel yolo2 = {
        INPUT_W / 16,
        INPUT_H / 16,
		{10,14,  23,27,  37,58}
    };

    static constexpr int LOCATIONS = 4;
    struct alignas(float) Detection{
        //x y w h
        float bbox[LOCATIONS];
        float det_confidence;
        float class_id;
        float class_confidence;
    };
}

namespace nvinfer1
{
    class YoloLayerPlugin: public IPluginV2IOExt
    {
        public:
            explicit YoloLayerPlugin();
            YoloLayerPlugin(const void* data, size_t length);

            ~YoloLayerPlugin();

            int getNbOutputs() const override
            {
                return 1;
            }

            Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;

            int initialize() override;

            virtual void terminate() override {};

            virtual size_t getWorkspaceSize(int maxBatchSize) const override { return 0;}

            virtual int enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream) override;

            virtual size_t getSerializationSize() const override;

            virtual void serialize(void* buffer) const override;

            bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) const override {
                return inOut[pos].format == TensorFormat::kLINEAR && inOut[pos].type == DataType::kFLOAT;
            }

            const char* getPluginType() const override;

            const char* getPluginVersion() const override;

            void destroy() override;

            IPluginV2IOExt* clone() const override;

            void setPluginNamespace(const char* pluginNamespace) override;

            const char* getPluginNamespace() const override;

            DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const override;

            bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const override;

            bool canBroadcastInputAcrossBatch(int inputIndex) const override;

            void attachToContext(
                    cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) override;

            void configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput) override;

            void detachFromContext() override;

        private:
            void forwardGpu(const float *const * inputs,float * output, cudaStream_t stream,int batchSize = 1);
            int mClassCount;
            int mKernelCount;
            std::vector<Yolo::YoloKernel> mYoloKernel;
            int mThreadCount = 256;
            const char* mPluginNamespace;
    };

    class YoloPluginCreator : public IPluginCreator
    {
        public:
            YoloPluginCreator();

            ~YoloPluginCreator() override = default;

            const char* getPluginName() const override;

            const char* getPluginVersion() const override;

            const PluginFieldCollection* getFieldNames() override;

            IPluginV2IOExt* createPlugin(const char* name, const PluginFieldCollection* fc) override;

            IPluginV2IOExt* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override;

            void setPluginNamespace(const char* libNamespace) override
            {
                mNamespace = libNamespace;
            }

            const char* getPluginNamespace() const override
            {
                return mNamespace.c_str();
            }

        private:
            std::string mNamespace;
            static PluginFieldCollection mFC;
            static std::vector<PluginField> mPluginAttributes;
    };
    REGISTER_TENSORRT_PLUGIN(YoloPluginCreator);
};

