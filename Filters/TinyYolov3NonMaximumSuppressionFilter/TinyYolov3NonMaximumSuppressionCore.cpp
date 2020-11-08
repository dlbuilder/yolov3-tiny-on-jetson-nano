#include "TinyYolov3NonMaximumSuppressionCore.h"

using namespace std;

inline bool cmp(const Detection &a, const Detection &b)
{
	return a.det_confidence > b.det_confidence;
}

TinyYolov3NonMaximumSuppressionCore::TinyYolov3NonMaximumSuppressionCore(TinyYolov3NonMaximumSuppressionInternalData *internalData, TinyYolov3NonMaximumSuppressionInputData *inputData, TinyYolov3NonMaximumSuppressionOutputData *outputData)
{
	mInternalData = internalData;
	mInputData = inputData;
	mOutputData = outputData;
}

TinyYolov3NonMaximumSuppressionCore::~TinyYolov3NonMaximumSuppressionCore()
{
}

FilterStatus TinyYolov3NonMaximumSuppressionCore::RunFilterCoreLogic()
{
	mOutputData->nmsOutDetections.clear();
	DoNms();
	for (size_t m = 0; m < mOutputData->nmsOutDetections.size(); ++m)
	{
		ChangeRectSize(mOutputData->nmsOutDetections[m].bbox);
	}
	if(mOutputData->detectionClassNames.size() == 0)
		mOutputData->detectionClassNames = mInputData->detectionClassNames;
	return FilterStatus::COMPLETE;
}

void TinyYolov3NonMaximumSuppressionCore::DoNms()
{
	for (auto it = mInputData->detections.begin(); it != mInputData->detections.end(); it++)
	{
		auto &dets = it->second;
		sort(dets.begin(), dets.end(), cmp);
		for (size_t m = 0; m < dets.size(); ++m)
		{
			auto &item = dets[m];
			mOutputData->nmsOutDetections.push_back(item);
			for (size_t n = m + 1; n < dets.size(); ++n)
			{
				if (IOU(item.bbox, dets[n].bbox) > mInternalData->nmsThreshold)
				{
					dets.erase(dets.begin() + n);
					--n;
				}
			}
		}
	}
}

float TinyYolov3NonMaximumSuppressionCore::IOU(float lbox[4], float rbox[4])
{
	float interBox[] = {
		std::max(lbox[0] - lbox[2] / 2.f, rbox[0] - rbox[2] / 2.f), //left
		std::min(lbox[0] + lbox[2] / 2.f, rbox[0] + rbox[2] / 2.f), //right
		std::max(lbox[1] - lbox[3] / 2.f, rbox[1] - rbox[3] / 2.f), //top
		std::min(lbox[1] + lbox[3] / 2.f, rbox[1] + rbox[3] / 2.f), //bottom
	};

	if (interBox[2] > interBox[3] || interBox[0] > interBox[1])
		return 0.0f;

	float interBoxS = (interBox[1] - interBox[0]) * (interBox[3] - interBox[2]);
	return interBoxS / (lbox[2] * lbox[3] + rbox[2] * rbox[3] - interBoxS);
}

void TinyYolov3NonMaximumSuppressionCore::ChangeRectSize(float *bbox)
{
	int l, r, t, b;
	float r_w = mInputData->resizedImageWidth / (mInputData->imageWidth * 1.0);
	float r_h = mInputData->resizedImageHeight / (mInputData->imageHeight * 1.0);
	if (r_h > r_w)
	{
		l = bbox[0] - bbox[2] / 2.f;
		r = bbox[0] + bbox[2] / 2.f;
		t = bbox[1] - bbox[3] / 2.f - (mInputData->resizedImageHeight - r_w * mInputData->imageHeight) / 2;
		b = bbox[1] + bbox[3] / 2.f - (mInputData->resizedImageHeight - r_w * mInputData->imageHeight) / 2;
		l = l / r_w;
		r = r / r_w;
		t = t / r_w;
		b = b / r_w;
	}
	else
	{
		l = bbox[0] - bbox[2] / 2.f - (mInputData->resizedImageWidth - r_h * mInputData->imageWidth) / 2;
		r = bbox[0] + bbox[2] / 2.f - (mInputData->resizedImageWidth - r_h * mInputData->imageWidth) / 2;
		t = bbox[1] - bbox[3] / 2.f;
		b = bbox[1] + bbox[3] / 2.f;
		l = l / r_h;
		r = r / r_h;
		t = t / r_h;
		b = b / r_h;
	}
	bbox[0] = l;
	bbox[1] = t;
	bbox[2] = r - l;
	bbox[3] = b - t;
}