#pragma once
#include "../Filters/WebcamJetsonFilter/WebcamJetsonData.h"
#include "../Filters/TinyYolov3PreProcessingJetsonFilter/TinyYolov3PreProcessingJetsonData.h"
#include "../Filters/TensorRTInferenceFilter/TensorRTInferenceData.h"
#include "../Filters/TensorRTTinyYolov3PostProcessingFilter/TensorRTTinyYolov3PostProcessingData.h"
#include "../Filters/TinyYolov3NonMaximumSuppressionFilter/TinyYolov3NonMaximumSuppressionData.h"
#include "../Filters/DetectionEGLRenderFilter/DetectionEGLRenderData.h"
WebcamInternalData* inputNode0_WebcamInternalData= new WebcamInternalData();
WebcamOutputData* inputNode0_WebcamOutputData= new WebcamOutputData();
TinyYolov3PreProcessingJetsonInternalData* preProcNode0_TinyYolov3PreProcessingJetsonInternalData= new TinyYolov3PreProcessingJetsonInternalData();
TinyYolov3PreProcessingJetsonInputData* preProcNode0_TinyYolov3PreProcessingJetsonInputData;
TinyYolov3PreProcessingJetsonOutputData* preProcNode0_TinyYolov3PreProcessingJetsonOutputData= new TinyYolov3PreProcessingJetsonOutputData();
TensorRTInferenceInternalData* inferenceNode0_TensorRTInferenceInternalData= new TensorRTInferenceInternalData();
TensorRTInferenceInputData* inferenceNode0_TensorRTInferenceInputData;
TensorRTInferenceOutputData* inferenceNode0_TensorRTInferenceOutputData= new TensorRTInferenceOutputData();
TensorRTTinyYolov3PostProcessingInternalData* postProcNode0_TensorRTTinyYolov3PostProcessingInternalData= new TensorRTTinyYolov3PostProcessingInternalData();
TensorRTTinyYolov3PostProcessingInputData* postProcNode0_TensorRTTinyYolov3PostProcessingInputData;
TensorRTTinyYolov3PostProcessingOutputData* postProcNode0_TensorRTTinyYolov3PostProcessingOutputData= new TensorRTTinyYolov3PostProcessingOutputData();
TinyYolov3NonMaximumSuppressionInternalData* postProcNode1_TinyYolov3NonMaximumSuppressionInternalData= new TinyYolov3NonMaximumSuppressionInternalData();
TinyYolov3NonMaximumSuppressionInputData* postProcNode1_TinyYolov3NonMaximumSuppressionInputData;
TinyYolov3NonMaximumSuppressionOutputData* postProcNode1_TinyYolov3NonMaximumSuppressionOutputData= new TinyYolov3NonMaximumSuppressionOutputData();
DetectionEGLRenderInternalData* outputNode0_DetectionEGLRenderInternalData= new DetectionEGLRenderInternalData();
DetectionEGLRenderImageInputData* outputNode0_DetectionEGLRenderImageInputData;
DetectionEGLRenderBoundingBoxInputData* outputNode0_DetectionEGLRenderBoundingBoxInputData;

void AllocaterFilterInitialData()
{
    inputNode0_WebcamInternalData->webcamId=0;
    inputNode0_WebcamOutputData->imageHeight=1080;
    inputNode0_WebcamOutputData->imageWidth=1920;
    preProcNode0_TinyYolov3PreProcessingJetsonInternalData->inferenceEngine=0;
    preProcNode0_TinyYolov3PreProcessingJetsonOutputData->imageHeight=1080;
    preProcNode0_TinyYolov3PreProcessingJetsonOutputData->imageWidth=1920;
    preProcNode0_TinyYolov3PreProcessingJetsonOutputData->resizedImageHeight=416;
    preProcNode0_TinyYolov3PreProcessingJetsonOutputData->resizedImageWidth=416;
    inferenceNode0_TensorRTInferenceInternalData->batchSize=1;
    inferenceNode0_TensorRTInferenceInternalData->gpuID=0;
    inferenceNode0_TensorRTInferenceInternalData->model="./Models/TinyYolov3_coco_jetsonnano4.4_sm53_tensorrt7.1_fp16.engine";
    inferenceNode0_TensorRTInferenceOutputData->imageHeight=1080;
    inferenceNode0_TensorRTInferenceOutputData->imageWidth=1920;
    inferenceNode0_TensorRTInferenceOutputData->resizedImageHeight=416;
    inferenceNode0_TensorRTInferenceOutputData->resizedImageWidth=416;
    postProcNode0_TensorRTTinyYolov3PostProcessingInternalData->maxBoxCount=1000;
    postProcNode0_TensorRTTinyYolov3PostProcessingInternalData->objThreshold=0.4;
    postProcNode0_TensorRTTinyYolov3PostProcessingOutputData->imageHeight=1080;
    postProcNode0_TensorRTTinyYolov3PostProcessingOutputData->imageWidth=1920;
    postProcNode0_TensorRTTinyYolov3PostProcessingOutputData->resizedImageHeight=416;
    postProcNode0_TensorRTTinyYolov3PostProcessingOutputData->resizedImageWidth=416;
    postProcNode1_TinyYolov3NonMaximumSuppressionInternalData->nmsThreshold=0.5;
    postProcNode1_TinyYolov3NonMaximumSuppressionOutputData->imageHeight=1080;
    postProcNode1_TinyYolov3NonMaximumSuppressionOutputData->imageWidth=1920;
}
