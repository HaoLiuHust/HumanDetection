#include "PixelSimilarity.h"

PixelSimilarity::PixelSimilarity( )
{
  point = cv::Point ( 0,0 );
  radius = 0;
  similarity = 0;
}
/**<
* Default Constructor.
 */
PixelSimilarity::PixelSimilarity ( cv::Point point, float radius, float similarity )
{
  this->point = point;
  this->radius = radius;
  this->similarity = similarity;
}
/**<
* Constructor given location, radius and likelihood.
* @param point A point containing the x and y coordinates.
* @param radius A float for the radius of the head in pixels.
* @param similarity A float between 0 and 1 for the likelihood.
 */
