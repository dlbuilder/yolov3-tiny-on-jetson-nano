#pragma once

#include <iostream>
#include <memory>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <poll.h>
#include <fcntl.h>
#include "NvJpegDecoder.h"
#include "nvbuf_utils.h"
#include "WebcamJetsonData.h"
#include "cuda_runtime.h"

#define V4L2_BUFFERS_NUM 4
#define MJPEG_EOS_SEARCH_SIZE 4096

typedef struct
{
	uint8_t *start;
	unsigned int size;
	int dmabuff_fd;
} nv_buffer;

class WebcamJetsonCore
{
public:
	WebcamJetsonCore(WebcamInternalData *internalData, WebcamOutputData *outputData);
	~WebcamJetsonCore();
	FilterStatus RunFilterCoreLogic();

private:
	void InitializeCamera();
	void InitializeJpegDecoder();
	void SetTransformParam();
	void SetCameraEvent();
	void PrepareMJPEGBuffer();
	void RequestCameraBufferMMap();
	void StartCaptueStream();
	void StopCaptureStream();
	void CleanUp();

	WebcamInternalData *mInternalData;
	WebcamOutputData *mOutputData;
	NvBufferTransformParams mTransParams;
	NvJPEGDecoder *mJpegDecoder;
	int mCamFD;
	string mCamDevName;
	nv_buffer *mGlobalBuffer;
	struct pollfd mFDs[1];
};
