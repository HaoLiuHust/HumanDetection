#ifndef TEMPLMATCH_H
#define TEMPLMATCH_H

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>

using namespace cv;

void matchTemplateParallel ( InputArray _img, InputArray _templ, OutputArray _result, int method );

#endif