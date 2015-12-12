#pragma once
#ifndef CHAMFERMATCH_H
#define CHAMFERMATCH_H
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <queue>
using namespace cv;
using std::vector;
using std::queue;

int mychamerMatching(Mat& img, Mat& templ,
	CV_OUT vector<vector<Point> >& results, CV_OUT vector<float>& cost,
	double templScale = 1, int maxMatches = 20,
	double minMatchDistance = 1.0, int padX = 3,
	int padY = 3, int scales = 5, double minScale = 0.6, double maxScale = 1.6,
	double orientationWeight = 0.5, double truncate = 20);
#endif