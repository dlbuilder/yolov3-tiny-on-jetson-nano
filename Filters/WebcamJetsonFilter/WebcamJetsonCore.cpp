#include "WebcamJetsonCore.h"

WebcamJetsonCore::WebcamJetsonCore(WebcamInternalData *internalData, WebcamOutputData *outputData)
{
	mInternalData = internalData;
	mOutputData = outputData;
	mCamDevName = "/dev/video" + to_string(internalData->webcamId);

	InitializeCamera();
	PrepareMJPEGBuffer();
	StartCaptueStream();
	InitializeJpegDecoder();
	SetTransformParam();
	SetCameraEvent();
}

WebcamJetsonCore::~WebcamJetsonCore()
{
	CleanUp();
}

void WebcamJetsonCore::CleanUp()
{
	StopCaptureStream();

	if (mJpegDecoder != NULL)
		delete mJpegDecoder;

	if (mCamFD > 0)
		close(mCamFD);

	if (mGlobalBuffer != NULL)
	{
		for (unsigned i = 0; i < V4L2_BUFFERS_NUM; i++)
		{
			if (mGlobalBuffer[i].dmabuff_fd)
				NvBufferDestroy(mGlobalBuffer[i].dmabuff_fd);
			munmap(mGlobalBuffer[i].start, mGlobalBuffer[i].size);
		}
		free(mGlobalBuffer);
	}
	NvBufferDestroy(mOutputData->dmaBufferFD);
}

void WebcamJetsonCore::InitializeCamera()
{
	struct v4l2_format fmt;

	mCamFD = open(mCamDevName.c_str(), O_RDWR);
	if (mCamFD == -1)
	{
		std::cerr << "Failed to open camera device : " << mCamDevName << std::endl;
		exit(EXIT_FAILURE);
	}

	memset(&fmt, 0, sizeof(fmt));
	fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	fmt.fmt.pix.width = mOutputData->imageWidth;
	fmt.fmt.pix.height = mOutputData->imageHeight;
	fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG;
	fmt.fmt.pix.field = V4L2_FIELD_INTERLACED;
	if (ioctl(mCamFD, VIDIOC_S_FMT, &fmt) < 0)
	{
		std::cerr << "Failed to set camera output format VIDIOC_S_FMT" << std::endl;
		exit(EXIT_FAILURE);
	}

	memset(&fmt, 0, sizeof(fmt));
	fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	if (ioctl(mCamFD, VIDIOC_G_FMT, &fmt) < 0)
	{
		std::cerr << "Failed to set camera output format VIDIOC_G_FMT" << std::endl;
		exit(EXIT_FAILURE);
	}
	if (fmt.fmt.pix.width != mOutputData->imageWidth ||
		fmt.fmt.pix.height != mOutputData->imageHeight ||
		fmt.fmt.pix.pixelformat != V4L2_PIX_FMT_MJPEG)
	{
		std::cerr << "The desired capture format is not supported" << std::endl;
		exit(EXIT_FAILURE);
	}

	struct v4l2_streamparm streamparm;
	memset(&streamparm, 0x00, sizeof(struct v4l2_streamparm));
	streamparm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	ioctl(mCamFD, VIDIOC_G_PARM, &streamparm);
}

void WebcamJetsonCore::InitializeJpegDecoder()
{
	mJpegDecoder = NvJPEGDecoder::createJPEGDecoder("jpegdec");
}

void WebcamJetsonCore::SetTransformParam()
{
	NvBufferRect src_rect, dest_rect;
	src_rect.top = 0;
	src_rect.left = 0;
	src_rect.width = mOutputData->imageWidth;
	src_rect.height = mOutputData->imageHeight;
	dest_rect.top = 0;
	dest_rect.left = 0;
	dest_rect.width = mOutputData->imageWidth;
	dest_rect.height = mOutputData->imageHeight;

	memset(&mTransParams, 0, sizeof(mTransParams));
	mTransParams.transform_flag = NVBUFFER_TRANSFORM_CROP_SRC;
	mTransParams.transform_filter = NvBufferTransform_Filter_Smart;
	mTransParams.transform_flip = NvBufferTransform_None;
	mTransParams.src_rect = src_rect;
	mTransParams.dst_rect = dest_rect;
}

void WebcamJetsonCore::SetCameraEvent()
{
	mFDs[0].fd = mCamFD;
	mFDs[0].events = POLLIN;
}

void WebcamJetsonCore::PrepareMJPEGBuffer()
{
	NvBufferCreateParams input_params = {0};

	mGlobalBuffer = (nv_buffer *)malloc(V4L2_BUFFERS_NUM * sizeof(nv_buffer));
	if (mGlobalBuffer == NULL)
	{
		std::cerr << "Failed to allocate global buffer" << std::endl;
		exit(EXIT_FAILURE);
	}

	memset(mGlobalBuffer, 0, V4L2_BUFFERS_NUM * sizeof(nv_buffer));

	input_params.payloadType = NvBufferPayload_SurfArray;
	input_params.width = mOutputData->imageWidth;
	input_params.height = mOutputData->imageHeight;
	input_params.layout = NvBufferLayout_Pitch;
	input_params.colorFormat = NvBufferColorFormat_ABGR32;
	input_params.nvbuf_tag = NvBufferTag_NONE;

	if (-1 == NvBufferCreateEx(&mOutputData->dmaBufferFD, &input_params))
	{
		std::cerr << "Failed to create OutputDMA NvBuffer" << std::endl;
		exit(EXIT_FAILURE);
	}
	RequestCameraBufferMMap();
	std::cout << "Succeed in Preparing JPEG Buffer" << std::endl;
}

void WebcamJetsonCore::RequestCameraBufferMMap()
{
	struct v4l2_requestbuffers rb;
	memset(&rb, 0, sizeof(rb));
	rb.count = V4L2_BUFFERS_NUM;
	rb.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	rb.memory = V4L2_MEMORY_MMAP;
	if (ioctl(mCamFD, VIDIOC_REQBUFS, &rb) < 0)
	{
		std::cerr << "Failed to request v4l2 buffers" << std::endl;
		exit(EXIT_FAILURE);
	}
	if (rb.count != V4L2_BUFFERS_NUM)
	{
		std::cerr << "V4l2 buffer number is not as desired" << std::endl;
		exit(EXIT_FAILURE);
	}

	for (unsigned int index = 0; index < V4L2_BUFFERS_NUM; index++)
	{
		struct v4l2_buffer buf;

		memset(&buf, 0, sizeof buf);
		buf.index = index;
		buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

		buf.memory = V4L2_MEMORY_MMAP;
		if (ioctl(mCamFD, VIDIOC_QUERYBUF, &buf) < 0)
		{
			std::cerr << "Failed to query buff VIDIOC_QUERYBUF" << std::endl;
			exit(EXIT_FAILURE);
		}

		mGlobalBuffer[index].size = buf.length;
		mGlobalBuffer[index].start = (uint8_t *)
			mmap(NULL,
				 buf.length,
				 PROT_READ | PROT_WRITE,
				 MAP_SHARED,
				 mCamFD, buf.m.offset);
		if (MAP_FAILED == mGlobalBuffer[index].start)
		{
			std::cerr << "Failed to map buffers" << std::endl;
			exit(EXIT_FAILURE);
		}

		if (ioctl(mCamFD, VIDIOC_QBUF, &buf) < 0)
		{
			std::cerr << "Failed to enqueue buffers" << std::endl;
			exit(EXIT_FAILURE);
		}
	}
}

void WebcamJetsonCore::StartCaptueStream()
{
	enum v4l2_buf_type type;

	type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	if (ioctl(mCamFD, VIDIOC_STREAMON, &type) < 0)
	{
		std::cerr << "Failed to start streaming" << std::endl;
		exit(EXIT_FAILURE);
	}

	usleep(200);

	std::cout << "Camera video streaming on ..." << std::endl;
}

void WebcamJetsonCore::StopCaptureStream()
{
	enum v4l2_buf_type type;

	type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	if (ioctl(mCamFD, VIDIOC_STREAMOFF, &type))
	{
		std::cerr << "Failed to stop streaming" << std::endl;
	}
	std::cout << "Camera video streaming off ..." << std::endl;
}

FilterStatus WebcamJetsonCore::RunFilterCoreLogic()
{
	if (poll(mFDs, 1, 5000) > 0)
	{
		if (mFDs[0].revents & POLLIN)
		{
			struct v4l2_buffer v4l2_buf;

			memset(&v4l2_buf, 0, sizeof(v4l2_buf));
			v4l2_buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
			v4l2_buf.memory = V4L2_MEMORY_MMAP;

			if (ioctl(mCamFD, VIDIOC_DQBUF, &v4l2_buf) < 0)
			{
				std::cerr << "Failed to dequeue camera buff" << std::endl;
				return FilterStatus::ERR;
			}

			int fd = 0;
			uint32_t width, height, pixfmt;
			unsigned int i = 0;
			unsigned int eos_search_size = MJPEG_EOS_SEARCH_SIZE;
			unsigned int bytesused = v4l2_buf.bytesused;
			uint8_t *p;

			if (eos_search_size > bytesused)
				eos_search_size = bytesused;
			for (i = 0; i < eos_search_size; i++)
			{
				p = (uint8_t *)(mGlobalBuffer[v4l2_buf.index].start + bytesused);
				if ((*(p - 2) == 0xff) && (*(p - 1) == 0xd9))
				{
					break;
				}
				bytesused--;
			}

			if (mJpegDecoder->decodeToFd(fd, mGlobalBuffer[v4l2_buf.index].start,
										 bytesused, pixfmt, width, height) < 0)
			{
				std::cerr << "Cannot decode MJPEG" << std::endl;
				return FilterStatus::ERR;
			}

			if (-1 == NvBufferTransform(fd, mOutputData->dmaBufferFD,
										&mTransParams))
			{
				std::cerr << "Failed to convert the buffer" << std::endl;
				return FilterStatus::ERR;
			}

			if (ioctl(mCamFD, VIDIOC_QBUF, &v4l2_buf))
			{
				std::cerr << "Failed to queue camera buffers" << std::endl;
				return FilterStatus::ERR;
			}
		}
	}
	else
	{
		std::cout << "Error in WebCamp Capture" << std::endl;
		FilterStatus::ERR;
	}

	return FilterStatus::COMPLETE;
}
