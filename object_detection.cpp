// Object Recognition example code from NVIDIA
// See https://github.com/dusty-nv/jetson-inference/blob/master/examples/my-recognition/my-recognition.cpp

#include <jetson-inference/detectNet.h>
#include <jetson-utils/loadImage.h>
#include <jetson-utils/cudaMappedMemory.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

// loadImageRGBA
bool cvt2CudaRGBA(const cv::Mat& mimg, float4** cpu, float4** gpu, int* width, int* height, const float4& mean=make_float4(0,0,0,0))
{
        // validate parameters
        if( !cpu || !gpu || !width || !height )
        {
                printf("loadImageRGBA() - invalid parameter(s)\n");
                return NULL;
        }

        // attempt to load the data from disk
        int imgWidth = *width;
        int imgHeight = *height;
        int imgChannels = 4;

        unsigned char* img = mimg.data;

        if( !img )
                return false;


        // allocate CUDA buffer for the image
        const size_t imgSize = imgWidth * imgHeight * sizeof(float) * 4;

        if( !cudaAllocMapped((void**)cpu, (void**)gpu, imgSize) )
        {
                printf("failed to allocate %zu bytes for image \n", imgSize);
                return false;
        }


        // convert uint8 image to float4
        float4* cpuPtr = *cpu;

        for( int y=0; y < imgHeight; y++ )
        {
                const size_t yOffset = y * imgWidth * imgChannels * sizeof(unsigned char);

                for( int x=0; x < imgWidth; x++ )
                {
                        #define GET_PIXEL(channel)	    float(img[offset + channel])
                        #define SET_PIXEL_FLOAT4(r,g,b,a) cpuPtr[y*imgWidth+x] = make_float4(r,g,b,a)

                        const size_t offset = yOffset + x * imgChannels * sizeof(unsigned char);

                        switch(imgChannels)
                        {
                                case 1:
                                {
                                        const float grey = GET_PIXEL(0);
                                        SET_PIXEL_FLOAT4(grey - mean.x, grey - mean.y, grey - mean.z, 255.0f - mean.w);
                                        break;
                                }
                                case 2:
                                {
                                        const float grey = GET_PIXEL(0);
                                        SET_PIXEL_FLOAT4(grey - mean.x, grey - mean.y, grey - mean.z, GET_PIXEL(1) - mean.w);
                                        break;
                                }
                                case 3:
                                {
                                        SET_PIXEL_FLOAT4(GET_PIXEL(0) - mean.x, GET_PIXEL(1) - mean.y, GET_PIXEL(2) - mean.z, 255.0f - mean.w);
                                        break;
                                }
                                case 4:
                                {
                                        SET_PIXEL_FLOAT4(GET_PIXEL(0) - mean.x, GET_PIXEL(1) - mean.y, GET_PIXEL(2) - mean.z, GET_PIXEL(3) - mean.w);
                                        break;
                                }
                        }
                }
        }

        *width  = imgWidth;
        *height = imgHeight;

//        free(img);
        return true;
}

// loadImageBGR
bool cvtCudaBGR( const cv::Mat& mimg, float3** cpu, float3** gpu, int* width, int* height, const float3& mean=make_float3(0,0,0) )
{
        // validate parameters
        if(!cpu || !gpu || !width || !height )
        {
                printf("loadImageRGB() - invalid parameter(s)\n");
                return NULL;
        }

        // attempt to load the data from disk
        int imgWidth = *width;
        int imgHeight = *height;
        int imgChannels = 3;

        unsigned char* img = mimg.data;

        if( !img )
                return false;


        // allocate CUDA buffer for the image
        const size_t imgSize = imgWidth * imgHeight * sizeof(float) * 3;

        if( !cudaAllocMapped((void**)cpu, (void**)gpu, imgSize) )
        {
                printf("failed to allocate %zu bytes for image \n", imgSize);
                return false;
        }


        // convert uint8 image to float4
        float3* cpuPtr = *cpu;

        for( int y=0; y < imgHeight; y++ )
        {
                const size_t yOffset = y * imgWidth * imgChannels * sizeof(unsigned char);

                for( int x=0; x < imgWidth; x++ )
                {
                        #define SET_PIXEL_FLOAT3(r,g,b) cpuPtr[y*imgWidth+x] = make_float3(r,g,b)

                        const size_t offset = yOffset + x * imgChannels * sizeof(unsigned char);

                        switch(imgChannels)
                        {
                                case 1:
                                {
                                        const float grey = GET_PIXEL(0);
                                        SET_PIXEL_FLOAT3(grey - mean.x, grey - mean.y, grey - mean.z);
                                        break;
                                }
                                case 2:
                                {
                                        const float grey = GET_PIXEL(0);
                                        SET_PIXEL_FLOAT3(grey - mean.x, grey - mean.y, grey - mean.z);
                                        break;
                                }
                                case 3:
                                case 4:
                                {
                                        SET_PIXEL_FLOAT3(GET_PIXEL(2) - mean.x, GET_PIXEL(1) - mean.y, GET_PIXEL(0) - mean.z);
                                        break;
                                }
                        }
                }
        }

        *width  = imgWidth;
        *height = imgHeight;

        return true;
}

int main( int argc, char** argv ){
	if( argc < 2 ) {
		printf("object_recognition:  expected image filename as argument\n");
		printf("example usage:   ./object_recognition image.jpg\n");
		return 0;
	}

	const char* imgFilename = argv[1];

	cv::Mat img = cv::imread(imgFilename);

	if(img.empty()) {
		printf("failed to load the image %s\n", imgFilename);
		return 0;
	}

//	cv::cvtColor(img,img,CV_BGRA2RGBA);

//	img.convertTo(img,CV_32FC4, 1/255.0);


	/*
	 * create detection network
	 */
	detectNet* net = detectNet::Create(argc, argv);

	if( !net )
	{
		printf("detectnet-console:   failed to initialize detectNet\n");
		return 0;
	}

	// parse overlay flags
	const uint32_t overlayFlags = detectNet::OverlayFlagsFromStr("box,labels,conf");

	
	/*
	 * load image from disk
	 */
	float* imgCPU    = NULL;
        float* imgCUDA   = NULL;
        int    imgWidth  = img.cols;
        int    imgHeight = img.rows;
		
        if( !cvtCudaBGR(img, (float4**)&imgCPU, (float4**)&imgCUDA, &imgWidth, &imgHeight) )
        {
                printf("failed to load image '%s'\n", imgFilename);
                return 0;
        }
	

	/*
	 * detect objects in image
	 */
	detectNet::Detection* detections = NULL;

	const int numDetections = net->Detect(imgCUDA, imgWidth, imgHeight, &detections, overlayFlags);

	// print out the detection results
	printf("%i objects detected\n", numDetections);
	
	for( int n=0; n < numDetections; n++ )
	{
		printf("detected obj %u  class #%u (%s)  confidence=%f\n", detections[n].Instance, detections[n].ClassID, net->GetClassDesc(detections[n].ClassID), detections[n].Confidence);
		printf("bounding box %u  (%f, %f)  (%f, %f)  w=%f  h=%f\n", detections[n].Instance, detections[n].Left, detections[n].Top, detections[n].Right, detections[n].Bottom, detections[n].Width(), detections[n].Height()); 
	}
	
        // print out timing info
	net->PrintProfilerTimes();


	/*
	 * destroy resources
	 */
	printf("detectnet-console:  shutting down...\n");

        CUDA(cudaFreeHost(imgCPU));
	SAFE_DELETE(net);

	printf("detectnet-console:  shutdown complete\n");
	return 0;
}
