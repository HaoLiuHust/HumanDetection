#define _CRT_SECURE_NO_WARNINGS
#include <cv.h>
#include <highgui.h>
#include <cxcore.h>
#include <iostream>
#include "DepthHumanDetector.h"
using namespace cv;

int main()
{
	DepthHumanDetector depthdetector;
	Mat depthimg = imread("depth01.png", CV_LOAD_IMAGE_ANYDEPTH);
	vector<Rect> faces;
	depthdetector.detect_face_depth(depthimg, faces);
	cout << faces.size() << endl;
	system("pause");
}