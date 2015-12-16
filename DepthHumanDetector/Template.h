#ifndef TEMPLATE_H
#define TEMPLATE_H

/** 
* @class Template
*
* @brief This class contains the 2D and 3D templates for matching.
* 
* @author Social Robot
* 
*/

#include <opencv2/core/core.hpp>

class Template
  {
  public:
    cv::Mat template2d;
    cv::Mat template3d;

    Template( );
    Template ( cv::Mat, cv::Mat );
    bool operator < ( const Template& ) const;
  };

#endif // TEMPLATE_H