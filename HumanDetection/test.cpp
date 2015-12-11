#define _CRT_SECURE_NO_WARNINGS
#include "HumanDetector.h"
#include <fstream>
#include<sstream>
#include<string>
void readtxt(const string& filename, Mat& img)
{
	img.create(480, 640, CV_16U);
	ushort* imgdata = img.ptr<ushort>(0);
	ifstream f(filename.c_str());
	if (!f.good())
	{
		cout << "file not opened" << endl;
	}
	ushort n;
	int index = 0;
	string line;
	while (getline(f,line))
	{
		istringstream is(line);
		int n;

		while (is>>n)
		{
			imgdata[index++] = n;
		}
	}
	f.close();
}

int main()
{
	/*Mat src = cv::imread("depthimage.png",0);
	Mat dst;

	HumanDetector::prePrecessing(src, dst);
	Mat edges;
	HumanDetector::edgeProcessing(dst, edges);*/
	//HumanDetector::chamferMatch(edges);

	//Mat drawing = Mat::zeros(src.size(), CV_8U);
	//for (int i = 0; i < contours.size(); ++i)
	//{
	//	cv::drawContours(drawing, contours, i, cv::Scalar(255, 255, 255), 1, 8, hierarchy, 0, cv::Point(0, 0));

	//}
	////cv::drawContours(drawing, contours, -1, cv::Scalar(255), 2, 8, hierarchy, 0, cv::Point(0, 0));
	//cv::imshow("edges", edges);
	//cv::imshow("contours", drawing);
	
	Mat img;
	readtxt("depth003.txt", img);
	Mat img8U;

	cv::imshow("img", img);
	cv::imwrite("003.png", img);

	Mat img2 = cv::imread("003.png", 0);
	cv::imshow("img2", img2);
	cv::waitKey();

	system("pause");
}