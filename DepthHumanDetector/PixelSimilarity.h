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

#endif // PIXELSIMILARITY_H
