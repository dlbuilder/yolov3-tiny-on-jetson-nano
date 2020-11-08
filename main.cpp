#include "./Data/DataPool.h"
#include "./API/InitializeAPI.h"
#include "./API/RuntimeAPI.h"
#include "./Filters/WebcamJetsonFilter/WebcamJetsonFilter.h"
#include "./Filters/TinyYolov3PreProcessingJetsonFilter/TinyYolov3PreProcessingJetsonFilter.h"
#include "./Filters/TensorRTInferenceFilter/TensorRTInferenceFilter.h"
#include "./Filters/TensorRTTinyYolov3PostProcessingFilter/TensorRTTinyYolov3PostProcessingFilter.h"
#include "./Filters/TinyYolov3NonMaximumSuppressionFilter/TinyYolov3NonMaximumSuppressionFilter.h"
#include "./Filters/DetectionEGLRenderFilter/DetectionEGLRenderFilter.h"

#include <chrono>

using namespace std;
int main()
{
    AllocaterFilterInitialData();
    InitFilterOutputDataandChainNextInputData(inputNode0_WebcamOutputData,preProcNode0_TinyYolov3PreProcessingJetsonInputData,preProcNode0_TinyYolov3PreProcessingJetsonOutputData,inferenceNode0_TensorRTInferenceInputData,inferenceNode0_TensorRTInferenceOutputData,postProcNode0_TensorRTTinyYolov3PostProcessingInputData,postProcNode0_TensorRTTinyYolov3PostProcessingOutputData,postProcNode1_TinyYolov3NonMaximumSuppressionInputData,postProcNode1_TinyYolov3NonMaximumSuppressionOutputData,outputNode0_DetectionEGLRenderBoundingBoxInputData,inputNode0_WebcamOutputData,outputNode0_DetectionEGLRenderImageInputData);
    WebcamJetsonFilter* inputNode0= new WebcamJetsonFilter(inputNode0_WebcamInternalData,inputNode0_WebcamOutputData);
    TinyYolov3PreProcessingJetsonFilter* preProcNode0= new TinyYolov3PreProcessingJetsonFilter(preProcNode0_TinyYolov3PreProcessingJetsonInternalData,preProcNode0_TinyYolov3PreProcessingJetsonInputData,preProcNode0_TinyYolov3PreProcessingJetsonOutputData);
    TensorRTInferenceFilter* inferenceNode0= new TensorRTInferenceFilter(inferenceNode0_TensorRTInferenceInternalData,inferenceNode0_TensorRTInferenceInputData,inferenceNode0_TensorRTInferenceOutputData);
    TensorRTTinyYolov3PostProcessingFilter* postProcNode0= new TensorRTTinyYolov3PostProcessingFilter(postProcNode0_TensorRTTinyYolov3PostProcessingInternalData,postProcNode0_TensorRTTinyYolov3PostProcessingInputData,postProcNode0_TensorRTTinyYolov3PostProcessingOutputData);
    TinyYolov3NonMaximumSuppressionFilter* postProcNode1= new TinyYolov3NonMaximumSuppressionFilter(postProcNode1_TinyYolov3NonMaximumSuppressionInternalData,postProcNode1_TinyYolov3NonMaximumSuppressionInputData,postProcNode1_TinyYolov3NonMaximumSuppressionOutputData);
    DetectionEGLRenderFilter* outputNode0= new DetectionEGLRenderFilter(outputNode0_DetectionEGLRenderInternalData,outputNode0_DetectionEGLRenderImageInputData,outputNode0_DetectionEGLRenderBoundingBoxInputData);
    while (true)
    {
	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        RunFilterSync(inputNode0);
        RunFilterSync(preProcNode0);
        RunFilterSync(inferenceNode0);
        RunFilterSync(postProcNode0);
        RunFilterSync(postProcNode1);
        RunFilterSync(outputNode0);
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
    }
    return 0;
}
