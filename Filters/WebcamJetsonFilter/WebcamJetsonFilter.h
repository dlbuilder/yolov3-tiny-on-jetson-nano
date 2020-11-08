#pragma once

#include "WebcamJetsonCore.h"

class WebcamJetsonFilter
{
public:
	WebcamJetsonFilter(WebcamInternalData* internalData, WebcamOutputData *outputData) {
		mWebcamJetsonCore = std::make_unique<WebcamJetsonCore>(internalData, outputData);
	}

	~WebcamJetsonFilter()
	{

	}

	FilterStatus RunFilterCoreLogic()
	{
		return mWebcamJetsonCore->RunFilterCoreLogic();
	}

	bool IsInputFilter()
	{
		return true;
	}

	bool IsOutPutFilter()
	{
		return false;
	}

private:
	std::unique_ptr<WebcamJetsonCore> mWebcamJetsonCore;
};
