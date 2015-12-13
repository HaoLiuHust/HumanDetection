#ifndef HUMANDETECTOR_H
#define HUMANDETECTOR_H
#include<iostream>
#include<vector>
#include<cassert>
#include <exception>

#include <cv.h>
#include <highgui.h>
#include <cxcore.h>
#include <opencv2/contrib/contrib.hpp>


using namespace std;
using cv::Mat;

const int lowThreshold = 100;
const int Ratio = 3;
const int EdgeThreshold = 20;
const float matchThreshold = 0.1000;

class HumanDetector
{

public:
	static void prePrecessing(Mat& src,Mat& dst);
	static void chamferMatch(Mat& src,Mat& temp,vector<vector<cv::Point> >& matchpos);
	static void edgeProcessing(Mat& src, Mat& dst);
	static void prePyramid(Mat& src,vector<Mat>& pyramid);
};



#endif
