#define _CRT_SECURE_NO_WARNINGS
#include "Utils.h"

void Utils::convert16to8U(Mat& src, Mat& dst)
{
	double maxvalue;
	cv::minMaxLoc(src, 0, &maxvalue);
	src.convertTo(dst, CV_8U, 255 / maxvalue);
}

void Utils::get_non_zeros(Mat & img, Mat & prob, vector<Point3f>& points, Point pdiff, double scale)
{
	for (int i = 0; i < img.rows; i++)
	{
		float *rowi = img.ptr<float>(i);
		for (int j = 0; j < img.cols; j++)
		{
			if (rowi[j] != 0)
			{
				Point3f point;
				point.x = (Point(j, i).x + pdiff.x) * scale;
				point.y = (Point(j, i).y + pdiff.y) * scale;
				point.z = prob.at<float>(i, j);
				points.push_back(point);
			}
		}
	}
}

double Utils::euclidean_distance(Point3f a, Point3f b)
{
	return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2) + pow(a.z - b.z, 2));
}

double Utils::euclidean_distance(Point a, Point b)
{
	return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2));
}

bool Utils::depth_cmp(const Point & lhs, const Point & rhs, Mat & depthimg)
{
	return depthimg.at<ushort>(lhs) < depthimg.at<ushort>(rhs);
}

inline bool Utils::isZero(double num)
{
	double mind = 1e-6;
	return num<=mind&&num>=mind;
}

Mat Utils::rgb2bw(Mat& im_rgb,Mat& img_bw)
{
	Mat im_gray;
	if (im_rgb.channels() == 3)
	{
		cvtColor(im_rgb, im_gray, CV_RGB2GRAY);
	}
	else
	{
		im_gray = im_rgb;
	}

	threshold(im_gray, img_bw, 128, 255, CV_THRESH_BINARY);
	return img_bw;
}

Mat Utils::preProcessing(Mat& src,Mat& dst)
{
	Mat temp;

	//GaussianBlur( image, image, Size(2*3+1,2*3+1), 0.0, 0.0, BORDER_DEFAULT );
	medianBlur(src, src, 5);//3
	
	inpaint(src, (src == 0), temp, 5, INPAINT_NS);
	medianBlur(temp, dst, 5);
	GaussianBlur(dst, dst, Size(2 * 2 + 1, 2 * 2 + 1), 0.0, 0.0, BORDER_DEFAULT);

	return dst;
}