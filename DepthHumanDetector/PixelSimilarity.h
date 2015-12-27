#ifndef PIXELSIMILARITY_H
#define PIXELSIMILARITY_H

#include <opencv2/core/core.hpp>

/** 
* @class PixelSimilarity
*
* @brief This class provides the location, radius and likelihood of a given potential head.
* 
* @author Social Robot
* 
*/


class PixelSimilarity
  {
  public:
    cv::Point point;
    float radius;
    float similarity;

    PixelSimilarity( );
    PixelSimilarity ( cv::Point point, float radius, float similarity );
  };

class DepthSimilarity
{
public:
	cv::Point point;
	float similarity;
	DepthSimilarity():point(0,0),similarity(0.0){}
	DepthSimilarity(cv::Point p,float s):point(p),similarity(s){}
	bool operator<(const DepthSimilarity& rhs)const
	{
		return similarity < rhs.similarity;
	}
};

#endif // PIXELSIMILARITY_H
