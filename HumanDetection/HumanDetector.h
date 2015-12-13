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
const float scalefactor = 0.75;
class HumanDetector
{
public:
	static void chamferMatch(Mat& src, Mat& temp, vector<vector<cv::Point> >& matchpos);

private:
	static void prePrecessing(Mat& src,Mat& dst);
	static void edgeProcessing(Mat& src, Mat& dst,const int edgethreshold=EdgeThreshold);
	static void prePyramid(Mat& src,vector<Mat>& pyramid);
	static cv::Point getCenter(vector<cv::Point>points);
};



#endif
