#include "Template.h"

Template::Template( )
{

}
/**<
* Default Constructor.
 */

Template::Template ( cv::Mat template2d, cv::Mat template3d )
{
  this->template2d = template2d;
  this->template3d = template3d;
}
/**<
* Constructor given 2D and 3D templates.
* @param template2d A Mat containing a binary image.
* @param template3d A Mat containing a grayscale image.
* Notice that template3d is a grayscale image, it is used 3d when the matching is done with the depth image.
 */

bool Template::operator< ( const Template& another_template ) const
{
  return (this->template2d.rows > another_template.template2d.rows) || (this->template2d.cols > another_template.template2d.cols);
}
