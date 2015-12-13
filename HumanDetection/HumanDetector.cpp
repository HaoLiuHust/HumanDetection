#define _CRT_SECURE_NO_WARNINGS

#include "HumanDetector.h"
#include "chamfermatch.h"

void HumanDetector::prePrecessing(Mat& src, Mat& dst)
{
	if (dst.empty())
	{
		dst.create(src.size(), src.type());
	}
	assert(src.size() == dst.size() && src.type() == dst.type()&&src.channels()==1);

	ushort* dstdata = dst.ptr<ushort>(0);
	ushort* srcdata = src.ptr<ushort>(0);

	int width = src.cols;
	int height = src.rows;
	int widthstep = src.step[0] / src.elemSize();
	 for (int rowindex = 0; rowindex < height;++rowindex)
	{
		for (int colindex = 0; colindex < width;++colindex)
		{
			int depthindex = colindex + rowindex*widthstep;
			int fillindex = depthindex;
			if (srcdata[depthindex]==0)
			{
				int searchwin = 1;
				bool found = false;
				while (!found)
				{
					int rangex1 = colindex - searchwin;
					int rangex2 = colindex + searchwin;
					int rangey1 = rowindex - searchwin;
					int rangey2 = rowindex + searchwin;

					if (rangex1<0&&rangex2>width&&rangey1<0&&rangey2>height)
					{
						found = true;
					}
					if (!found&&rangey1 >=0)
					{
						int range1 = rangex1 >= 0 ? rangex1 : 0;
						int range2 = rangex2 < width ? rangex2 : width - 1;

						for (int i = range1; i < range2 + 1;++i)
						{
							int index = rangey1*widthstep + i;
							if (srcdata[index]!=0)
							{
								fillindex = index;
								found = true;
								break;
							}
						}
					}
					if (!found&&rangey2<height)
					{
						int range1 = rangex1 > 0 ? rangex1 : 0;
						int range2 = rangex2 < width ? rangex2 : width - 1;

						for (int i = range1; i < range2 + 1; ++i)
						{
							int index = rangey2*widthstep + i;
							if (srcdata[index] != 0)
							{
								fillindex = index;
								found = true;
								break;
							}
						}
					}

					if (!found&&rangex1>=0)
					{
						int range1 = rangey1 > 0 ? rangey1 : 0;
						int range2 = rangey2 < height ? rangey2 : height - 1;

						for (int i = range1; i < range2 + 1; ++i)
						{
							int index = i*widthstep + rangex1;
							if (srcdata[index] != 0)
							{
								fillindex = index;
								found = true;
								break;
							}
						}
					}

					if (!found&&rangex2<width)
					{
						int range1 = rangey1 > 0 ? rangey1 : 0;
						int range2 = rangey2 < height ? rangey2 : height - 1;

						for (int i = range1; i < range2 + 1; ++i)
						{
							int index = i*widthstep + rangex2;
							if (srcdata[index] != 0)
							{
								fillindex = index;
								found = true;
								break;
							}
						}
					}

					++searchwin;
				}
			}

			dstdata[depthindex] = srcdata[fillindex];
		}
	}

	Mat blured;
	cv::medianBlur(dst, blured, 3);
	blured.copyTo(dst);
}

void HumanDetector::edgeProcessing(Mat& src, Mat& edges)
{
	Mat src8U;
	double maxv;
	cv::minMaxLoc(src, 0, &maxv);
	src.convertTo(src8U, CV_8U, 255 / maxv);

	cout << maxv << endl;

	cv::Canny(src8U, edges, lowThreshold, lowThreshold * 3);
	vector<vector<cv::Point> >contours;
	vector<cv::Vec4i> hierarchy;
	cv::findContours(edges, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
	uchar* edgedata = edges.ptr<uchar>(0);
	int widthstep = edges.step[0] / edges.elemSize();
	cout << contours.size() << endl;

	//统计轮廓长度
	for (int i = 0; i < contours.size(); ++i)
	{
		if (contours[i].size() < EdgeThreshold)
		{
			for (int j = 0; j < contours[i].size(); ++j)
			{
				cv::Point pos = contours[i][j];
				int index = pos.x + pos.y*widthstep;
				edgedata[index] = 0;
			}
		}
	}

}

void HumanDetector::chamferMatch(Mat& src,Mat& temp, vector<vector<cv::Point> >& matchpos)
{
	
	//Mat binarymap;
	//cv::threshold(src, binarymap, 0, 255, CV_THRESH_BINARY_INV);

	//cv::distanceTransform(binarymap, dist, CV_DIST_L2, CV_DIST_MASK_5);
	//Mat distmap32S;
	//Mat distmap8U(src.size(), CV_8U);
	//dist.convertTo(distmap32S, CV_32F, 1.0, 0.5);
	//distmap32S.convertTo(distmap8U, CV_8U);
	////cv::convertScaleAbs(distmap32S, distmap8U);
	//cv::imshow("dismap", distmap8U);
	//cv::waitKey(-1);
	
	Mat dst,edges;
	prePrecessing(src, dst);
	edgeProcessing(dst, edges);
	cv::imwrite("edges.png", edges);
	vector<vector<cv::Point> > results;
	vector<float> costs;
	int best = mychamerMatching(edges, temp, results, costs);
	
	for (int i = 0; i < costs.size();++i)
	{
		if (costs[i]<matchThreshold)
		{
			matchpos.push_back(results[i]);
		}
	}
	/*if (best>=0)
	{
		for (int i = 0; i < results[best].size();++i)
		{
			matchpos.push_back(results[best][i]);
		}
	}*/
	results.clear();
}

void HumanDetector::prePyramid(Mat& src, vector<Mat>& pyramid)
{
	int width = src.cols;
	int height = src.rows;
	int scalefactor = 0.75;
	int levels = 10;
	pyramid.resize(levels);
	pyramid.push_back(src);
	for (int i = 1; i < levels;++i)
	{
		Mat temp;
		width *= scalefactor;
		height *= scalefactor;
		if (width<1||height<1)
		{
			break;
		}
		cv::pyrDown(src, temp, cv::Size(width, height));
		pyramid.push_back(temp);
	}
}