#ifndef UTILS_H
#define UTILS_H
#include <cv.h>
#include <highgui.h>
#include <cxcore.h>
#include <iostream>

#include<opencv2/imgproc/imgproc.hpp>
#include <opencv2/photo/photo.hpp>

using namespace std;
using namespace cv;
class DepthHumanDetector;

class Utils
{
public:
	Mat rgb2bw(Mat& img_rgb, Mat& img_bw);
	Mat preProcessing(Mat& src, Mat& dst);
	void convert16to8U(Mat& src, Mat& dst);
	void get_non_zeros(Mat& img, Mat& prob, vector<Point3f>& points, Point pdiff = Point(0, 0), double scale = 1);
	double euclidean_distance(Point3f a, Point3f b);
	double euclidean_distance(Point a, Point b);
	bool depth_cmp(const Point& lhs, const Point& rhs,Mat& depthimg );
	inline bool isZero(double num);
};
#endif
