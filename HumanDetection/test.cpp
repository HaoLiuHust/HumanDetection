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
	Mat src = cv::imread("003.png",CV_LOAD_IMAGE_ANYDEPTH);
	Mat temp = cv::imread("template1.bmp", CV_LOAD_IMAGE_ANYDEPTH);
	vector<vector<cv::Point> > matpos;
	vector<float> matcosts;
	HumanDetector::chamferMatch(src, temp, matpos);
	if (matpos.size()>0)
	{
		cout << "matched" << endl;
	}
	Mat src8U;
	double maxv;
	cv::minMaxLoc(src, 0, &maxv);
	src.convertTo(src8U, CV_8U, 255 / maxv);

	Mat srcRGB;
	cv::cvtColor(src8U, srcRGB, CV_GRAY2BGR);
	Mat mask=Mat::zeros(src8U.size(), CV_8U);
	//uchar* maskdata = mask.ptr<uchar>(0);
	//int widthstep = mask.step[0] / mask.elemSize();
	for (int i = 0; i < matpos.size();++i)
	{
		for (int j = 0; j < matpos[i].size()-1;j+=2)
		{
			cv::Point pt = matpos[i][j];
			cv::Point pt2 = matpos[i][j + 1];
			cv::line(mask, pt, pt2, cv::Scalar(255), 2);
		}
		
	}
	srcRGB.setTo(cv::Scalar(0, 255, 0), mask);
	cv::imshow("match", srcRGB);
	cv::waitKey();
	//Mat drawing = Mat::zeros(src.size(), CV_8U);
	//for (int i = 0; i < contours.size(); ++i)
	//{
	//	cv::drawContours(drawing, contours, i, cv::Scalar(255, 255, 255), 1, 8, hierarchy, 0, cv::Point(0, 0));

	//}
	////cv::drawContours(drawing, contours, -1, cv::Scalar(255), 2, 8, hierarchy, 0, cv::Point(0, 0));
	//cv::imshow("edges", edges);
	//cv::imshow("contours", drawing);
	
	/*Mat img;
	readtxt("depth003.txt", img);
	Mat img8U;

	cv::imshow("img", img);
	cv::imwrite("003.png", img);

	Mat img2 = cv::imread("003.png", 0);
	cv::imshow("img2", img2);
	cv::waitKey();*/

	system("pause");
}