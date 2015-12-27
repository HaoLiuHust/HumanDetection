#define _CRT_SECURE_NO_WARNINGS
#include <cv.h>
#include <highgui.h>
#include <cxcore.h>
#include<opencv2/contrib/contrib.hpp>
#include <iostream>
#include <omp.h>
#include "DepthHumanDetector.h"
using namespace cv;

int main()
{
	DepthHumanDetector depthdetector;
	Mat depthimg = imread("6.png", CV_LOAD_IMAGE_ANYDEPTH);
	Mat disparityimg = imread("6d.png", CV_LOAD_IMAGE_ANYDEPTH);

	vector<Rect> faces;
	double t = (double)getTickCount();
	/*Mat colorimg;
	applyColorMap(disparityimg, colorimg, COLORMAP_HOT);
	imshow("colrmap", colorimg);
	waitKey();*/
	/*depthdetector.detect_face_depth(depthimg, disparityimg,faces);
	t = (double)getTickCount() - t;
	cout << faces.size() << endl;
	cout << t*1000. / cv::getTickFrequency() << endl;
	for (int i = 0; i < faces.size();++i)
	{
		rectangle(disparityimg, faces[i], Scalar(255));
	}
	imshow("faces", disparityimg);
	waitKey();*/

	/*vector<Point> detected_heads = depthdetector.get_detectedheads();
	Mat regin = Mat::zeros(depthimg.size(),CV_8U);
	ushort temp = depthimg.at<ushort>(detected_heads[0]);
	for (int i = 0; i < regin.rows;++i)
	{
		for (int j = 0; j < regin.cols;++j)
		{
			if (depthimg.at<ushort>(i,j)>=temp)
			{
				regin.at<uchar>(i, j) = 1;
			}
		}
	}

	Mat body1;
	disparityimg.copyTo(body1, (regin != 0));
	imshow("body", body1);

	waitKey();*/


	vector<Mat> detected_body;
	t = (double)getTickCount();
	depthdetector.detect_body_depth(depthimg, disparityimg, detected_body);
	t = (double)getTickCount() - t;
	cout << t*1000. / cv::getTickFrequency() << endl;
	//imshow("depth", depthimg);
	//waitKey();
	for (int i = 0; i < detected_body.size();++i)
	{
		Mat body;
		disparityimg.copyTo(body, (detected_body[i]!=0));
		imshow("body", body);
	
		waitKey();
	}
	system("pause");
}