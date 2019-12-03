// Object Recognition example code from NVIDIA
// See https://github.com/dusty-nv/jetson-inference/blob/master/examples/my-recognition/my-recognition.cpp

#include <jetson-inference/detectNet.h>
#include <jetson-utils/loadImage.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>


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

	cv::cvtColor(img,img,cv::COLOR_BGRA2RGBA);

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
	const uint32_t overlayFlags = detectNet::OverlayFlagsFromStr(cmdLine.GetString("overlay", "box,labels,conf"));

	
	/*
	 * load image from disk
	 */
	float* imgCPU    = NULL;
	float* imgCUDA   = NULL;
	int    imgWidth  = 0;
	int    imgHeight = 0;
		
	if( !loadImageRGBA(imgFilename, (float4**)&imgCPU, (float4**)&imgCUDA, &imgWidth, &imgHeight) )
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
	
	// save image to disk
	const char* outputFilename = cmdLine.GetPosition(1);
	
	if( outputFilename != NULL )
	{
		printf("detectnet-console:  writing %ix%i image to '%s'\n", imgWidth, imgHeight, outputFilename);
		
		if( !saveImageRGBA(outputFilename, (float4*)imgCPU, imgWidth, imgHeight, 255.0f) )
			printf("detectnet-console:  failed saving %ix%i image to '%s'\n", imgWidth, imgHeight, outputFilename);
		else	
			printf("detectnet-console:  successfully wrote %ix%i image to '%s'\n", imgWidth, imgHeight, outputFilename);
	}


	/*
	 * destroy resources
	 */
	printf("detectnet-console:  shutting down...\n");

	CUDA(cudaFreeHost(imgCPU));
	SAFE_DELETE(net);

	printf("detectnet-console:  shutdown complete\n");
	return 0;
}
