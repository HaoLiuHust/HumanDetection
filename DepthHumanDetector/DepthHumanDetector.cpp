#define _CRT_SECURE_NO_WARNINGS
#include "DepthHumanDetector.h"
#include "Utils.h"
#include "templmatch.h"
#include <omp.h>
#include <algorithm>
//#include <device_launch_parameters.h>
//#include <cuda_runtime.h>
//#include <opencv2/gpu/gpu.hpp>

const Size2i MORPH_SIZE = Size(11, 11);
Utils utils;
vector<Mat> pyramid;
vector<Mat> chamfer;
vector<Mat> matching;
enum MASKFLAG
{
	NOVISITED,
	CONTOURS,
	VISITED
};


DepthHumanDetector::DepthHumanDetector()
{
	load_templates();
	canny_thr1 = 5;
	canny_thr2 = 7;
	chamfer_thr = 7;
	arc_thr_low = 7;
	arc_thr_high = 20;
	scale_factor = 0.75;
	match3D_thr = 0.4;
	scales = 6;
	scales_default = 6;
	depth_thr = 300;
	depth_thrf = 100 * 10.0;
}

void DepthHumanDetector::detect_face_depth(Mat & depth_image, Mat& disparity_image,vector<Rect>& detected_faces)
{
	detected_faces.clear();
	final_head_features.clear();
	detected_heads.clear();
	//utils.convert16to8U(depth_image, disparity_image);
	Mat element = getStructuringElement(MORPH_RECT, MORPH_SIZE, Point(5, 5));

	utils.preProcessing(disparity_image, disparity_image);
	//imshow("8U", disparity_image);
	depth_image.setTo(0, (disparity_image == 0));
	depth_precessing(depth_image, depth_image);
	depthimage = depth_image;
	//dilate(depth_image, depth_image, element);
	/*imshow("depth", depth_image);
	waitKey();*/

	
	vector<PixelSimilarity> all_head_features;

	if (!check_dimensions(depth_image))
	{
		if (scales < 0)
		{
			scales = 0;
			return;
		}
	}

	
	pyramid.reserve(scales);
	pyramid.resize(scales);
	chamfer.reserve(scales);
	chamfer.resize(scales);
	matching.reserve(scales);
	matching.resize(scales);

	int arc_thr_lovv = arc_thr_low;
	int arc_thr_highv = arc_thr_high;
	Mat canny_im;
//#pragma omp parallel for default(none) shared(disparity_image,all_head_features,depth_image)
	for (int k = 0; k < templates.size(); k++)
	{
		vector<Point3f> head_matched_points;
		
		double t = (double)getTickCount();
		chamfer_matching(disparity_image, templates[k].template2d, canny_im,head_matched_points);
		t = (double)getTickCount() - t;
		//cout << t*1000./cv::getTickFrequency() << endl;
		vector<PixelSimilarity> head_features;
		compute_headparameters(depth_image, head_matched_points, head_features);
		vector<PixelSimilarity> new_head_features;
		t = (double)getTickCount();
		false_positives(head_features, canny_im,arc_thr_low, arc_thr_high, new_head_features);
		t = (double)getTickCount() - t;
		//cout << t*1000. / cv::getTickFrequency() << endl;
		match_template3D(disparity_image, new_head_features, all_head_features, k);
	}
	
	merge_rectangles(all_head_features, final_head_features);
	Rect rect;
	for (unsigned int i = 0; i < final_head_features.size(); i++)
	{
		int wh = final_head_features[i].radius * 2;
		rect = Rect(final_head_features[i].point.x - final_head_features[i].radius, final_head_features[i].point.y - final_head_features[i].radius, wh, wh);
		detected_faces.push_back(rect);
		detected_heads.push_back(final_head_features[i].point);
	}

	sort_heads(detected_heads);

	pyramid.clear();
	chamfer.clear();
	matching.clear();
}

void DepthHumanDetector::detect_body_depth(Mat & depth_image, Mat & disparity_image, vector<Mat>& detected_bodies)
{
	vector<Rect> detected_faces;
	double t = (double)getTickCount();
	
	detect_face_depth(depth_image, disparity_image, detected_faces);
	t = (double)getTickCount() - t;
	cout << t*1000. / cv::getTickFrequency() << endl;
	extract_contours(depth_image, detected_heads, detected_bodies);
}

vector<Point> DepthHumanDetector::get_detectedheads()
{
	return detected_heads;
}

DepthHumanDetector::~DepthHumanDetector()
{
	templates.clear();
	detected_heads.clear();
	final_head_features.clear();
	depthimage.release();
}



void DepthHumanDetector::load_templates()
{
	string head_template[4];
	head_template[0] = "./pictures/template3.png";
	head_template[1] = "./pictures/template.png";
	head_template[2] = "./pictures/only_head_template.png";
	head_template[3] = "./pictures/only_head_template_small.png";

	string head_template3D[4];
	head_template3D[0] = "./pictures/template3D.png";
	head_template3D[1] = "./pictures/template3D.png";
	head_template3D[2] = "./pictures/only_head_template_3d.png";
	head_template3D[3] = "./pictures/only_head_template_3d_small.png";

	Mat head_templateim[4];
	Mat head_template3D_im[4];
#pragma omp parallel for default(none) shared(head_templateim,head_template3D_im,utils,head_template,head_template3D)
	for (int i = 0; i < 4;++i)
	{
		head_templateim[i]=imread(head_template[i], CV_LOAD_IMAGE_ANYDEPTH);
		head_template3D_im[i] = imread(head_template3D[i], CV_LOAD_IMAGE_ANYDEPTH);
		utils.rgb2bw(head_templateim[i], head_templateim[i]);
		head_templateim[i].convertTo(head_templateim[i], CV_32F);
		if (head_template3D_im[i].depth() == 3)
		{
			cvtColor(head_template3D_im[i], head_template3D_im[i], CV_RGB2GRAY);
		}
	}
	
	for (int i = 0; i < 4;++i)
	{
		templates.push_back(Template(head_templateim[i], head_template3D_im[i]));
	}

	sort(templates.begin(), templates.end());
}

bool DepthHumanDetector::check_dimensions(Mat & image_depth)
{
	double max_scale, scale_rows, scale_cols, log_scale;

	max_scale = pow(scale_factor, scales_default);

	log_scale = log(scale_factor);

	if (image_depth.rows * max_scale < templates[0].template2d.rows || image_depth.cols * max_scale < templates[0].template2d.cols)
	{
		scale_rows = (log(templates[0].template2d.rows) - log(image_depth.rows)) / log_scale;
		scale_cols = (log(templates[0].template2d.cols) - log(image_depth.cols)) / log_scale;

		scales = round(min(scale_rows, scale_cols));
		return false;
	}
	else if (image_depth.rows * max_scale < templates[1].template2d.rows || image_depth.cols * max_scale < templates[1].template2d.cols)
	{
		scale_rows = (log(templates[1].template2d.rows) - log(image_depth.rows)) / log_scale;
		scale_cols = (log(templates[1].template2d.cols) - log(image_depth.cols)) / log_scale;

		scales = round(min(scale_rows, scale_cols));
		return false;
	}
	else
	{
		return true;
	}
}

void DepthHumanDetector::chamfer_matching(Mat & image, Mat & template_im, Mat& canny_im, vector<Point3f>& chamfer_heads)
{
	double xdiff = template_im.cols / 2;
	double ydiff = template_im.rows / 2;
	Point pdiff = Point(xdiff, ydiff);
	
	

	canny_im.create(image.rows, image.cols, image.depth());
	Canny(image, canny_im, canny_thr1, canny_thr2, 3, true);

	Size size0 = Size();

#pragma omp parallel for default(none) shared(chamfer, pyramid,size0,canny_im)
	for (int i = 0; i < scales; i++)
	{
		resize(canny_im, pyramid[i], size0, pow(scale_factor, i), pow(scale_factor, i), INTER_NEAREST);
		distanceTransform((255 - pyramid[i]), chamfer[i], CV_DIST_C, 3);
	}

	//Mat matching_thr;

#pragma omp parallel for default(none) shared(chamfer, matching, chamfer_heads, template_im, pdiff, utils) //private(matching_thr)
	for (int j = 0; j < scales; j++)
	{
		double t = (double)getTickCount();
		//       matchTemplate ( chamfer[j], template_im, matching[j], CV_TM_CCOEFF );
		matchTemplateParallel(chamfer[j], template_im, matching[j], CV_TM_CCOEFF);
		t = (double)getTickCount() - t;
		//cout << t*1000. / cv::getTickFrequency() << endl;

		double minVal, maxVal;
		Point minLoc, maxLoc;
		normalize(matching[j], matching[j], 0.0, 1.0, NORM_MINMAX);
		minMaxLoc(matching[j], &minVal, &maxVal, &minLoc, &maxLoc);
		Mat matching_thr;
		threshold(matching[j], matching_thr, 1.0 / chamfer_thr, 1.0, CV_THRESH_BINARY_INV);
		double scale = pow(1.0 / scale_factor, j);
		vector<Point3f> chamfer_heads_private;
		utils.get_non_zeros(matching_thr, matching[j], chamfer_heads_private, pdiff, scale);

		#pragma omp critical
		{
			//utils.get_non_zeros(matching_thr, matching[j], chamfer_heads, pdiff, scale);
			chamfer_heads.insert(chamfer_heads.end(), chamfer_heads_private.begin(), chamfer_heads_private.end());
		}
	}
}

void DepthHumanDetector::compute_headparameters(Mat & image, vector<Point3f>& chamfer, vector<PixelSimilarity>& parameters_head)
{
	parameters_head.reserve(chamfer.size());
	parameters_head.resize(chamfer.size());
	float p1 = -1.3835 * pow(10, -9);
	float p2 = 1.8435 * pow(10, -5);
	float p3 = -0.091403;
	float p4 = 189.38;

	for (unsigned int i = 0; i < chamfer.size(); i++)
	{
		int position_x = chamfer[i].x;
		int position_y = chamfer[i].y;

		float x;

		if (image.type() == 5)
		{
			x = image.at<float>(position_y, position_x) * 1000;
		}
		else
		{
			unsigned short xshort = image.at<unsigned short>(position_y, position_x);
			x = (float)xshort;
		}

		float h = (p1 * pow(x, 3) + p2 * pow(x, 2) + p3 * x + p4);

		float R = 1.33 * h * 0.5;

		float Rp = round(R / 1.3);

		parameters_head[i].point = Point(position_x, position_y);
		parameters_head[i].radius = Rp;
		parameters_head[i].similarity = chamfer[i].z;
	}
}

void DepthHumanDetector::false_positives(vector<PixelSimilarity>& potential_heads,Mat& canny_im, int thr, int thr2, vector<PixelSimilarity>& false_positive_removed_heads)
{
	
//#pragma omp parallel for default(none) shared(potential_heads,canny_im,thr,thr2,false_positive_removed_heads)
	for (int i = 0; i < potential_heads.size(); i++)
	{
		Rect roi(potential_heads[i].point.x - potential_heads[i].radius, potential_heads[i].point.y - potential_heads[i].radius, potential_heads[i].radius * 2, potential_heads[i].radius * 2);
		if (!(0 <= roi.x && 0 <= roi.width && roi.x + roi.width < canny_im.cols && 0 <= roi.y && 0 <= roi.height && roi.y + roi.height < canny_im.rows))
		{
			continue;
		}
		Mat cannyroi;
		cannyroi=canny_im(roi);
		
		if (!cannyroi.empty())
		{
			vector<vector<Point> > contour;
			vector<Point> approx;
			findContours(cannyroi, contour, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
			
			for (int j = 0; j < contour.size(); j++)
			{
				approx.clear();
				approxPolyDP(Mat(contour[j]), approx, 5, false);
				if (approx.size() > (unsigned int) thr && approx.size() < (unsigned int)thr2)
				{
					false_positive_removed_heads.push_back(potential_heads[i]);
					break;
				}
			}
		}
	}


}

void DepthHumanDetector::match_template3D(Mat& image_disparity, vector<PixelSimilarity>& potentials, vector<PixelSimilarity>& heads, int n)
{
	Mat match;
	double minVal, maxVal;
	Point minLoc, maxLoc;

	if (potentials.empty())
	{
		return;
	}
#pragma omp parallel for default(none) shared(potentials,image_disparity,heads,n) private(minVal,maxVal,match,minLoc,maxLoc)
	for (int i = 0; i < potentials.size(); i++)
	{
		Rect rect_roi(potentials[i].point.x - potentials[i].radius, potentials[i].point.y - potentials[i].radius, 2 * potentials[i].radius, 2 * potentials[i].radius);
		Mat roi(image_disparity, rect_roi);
		resize(roi, roi, templates[n].template3d.size());
		minMaxLoc(roi, &minVal, &maxVal, 0, 0);
		roi = roi - minVal;
		normalize(roi, roi, 0.0, 255.0, NORM_MINMAX);

		matchTemplate(roi, templates[n].template3d, match, CV_TM_CCOEFF_NORMED);

		minMaxLoc(match, &minVal, &maxVal, &minLoc, &maxLoc);

		vector<PixelSimilarity> heads_private;
		if (minVal >= match3D_thr)
		{
			heads_private.push_back(PixelSimilarity(potentials[i].point, potentials[i].radius, potentials[i].similarity));
		}
#pragma omp critical
		heads.insert(heads.end(),heads_private.begin(), heads_private.end());
	}
}

void DepthHumanDetector::merge_rectangles(vector<PixelSimilarity>& rectangles, vector<PixelSimilarity>& merged_rects)
{
	PixelSimilarity mean_rect;
	vector<PixelSimilarity> queue;
	double tol = 75;

	while (rectangles.size() > 0)
	{
		mean_rect = rectangles[0];

		for (unsigned int i = 1; i < rectangles.size(); i++)
		{
			if (utils.euclidean_distance(mean_rect.point, rectangles[i].point) < tol)
			{
				if (rectangles[i].similarity < mean_rect.similarity)
				{
					mean_rect = rectangles[i];
				}
			}
			else
			{
				queue.push_back(rectangles[i]);
			}
		}
		merged_rects.push_back(mean_rect);
		rectangles.swap(queue);
		queue.clear();
	}
}

void DepthHumanDetector::boundary_filter(Mat& src, Mat& filtered)
{
	Mat kern = (Mat_<char>(1, 6) << 1, 1, 1, -1, -1, -1);
	filter2D(src, filtered, src.depth(), kern);
	add(src, filtered, filtered);
}

void DepthHumanDetector::extract_contours(Mat & depthimg, vector<Point>& heads, vector<Mat>& bodies)
{
	bodies.reserve(heads.size());
	bodies.resize(heads.size());
	Mat region = Mat::zeros(depthimg.size(), CV_8U);
//#pragma omp parallel for default(none) shared(bodies,depthimg,heads,region)
	for (int i = 0; i < heads.size();++i)
	{
		Point seed = heads[i];
		float head_height = 2 * final_head_features[i].radius;

		if (i+1<heads.size())
		{
			ushort depth_diff = depthimg.at<ushort>(heads[i + 1]) - depthimg.at<ushort>(heads[i]);
			depth_thr = depth_diff / 2.0;
		}
		region_grow(depthimg, seed, region,head_height);
		Mat body;
		depthimg.copyTo(body, (region == CONTOURS));

		region.setTo(NOVISITED, (region == VISITED));
		region.setTo(VISITED, (region == CONTOURS));
		//bodies.push_back(body);
		bodies[i] = body;
	}
}



void DepthHumanDetector::get_neighborings(Mat & depthimg, Mat & region, vector<DepthSimilarity>& edgesbefore, float depthbefore,vector<DepthSimilarity>& edgesafter)
{
	Point offset[4] = { {-1,0},{0,-1},{1,0},{0,1} };
	DepthSimilarity ds;
	int width = depthimg.cols;
	int height = depthimg.rows;
	for (int i = 0; i < edgesbefore.size();++i)
	{
		#pragma omp parallel for default(none) shared(offset,edgesbefore,width,height,region,depthbefore,depthimg,ds,edgesafter,i) 
		for (int j = 0; j < 4;++j)
		{
			Point pt = edgesbefore[i].point + offset[j];
			if (pt.x>=width||pt.y>=height||pt.x<0||pt.y<0)
			{
				continue;
			}
			if (region.at<uchar>(pt)==NOVISITED)
			{
				float similarity = (float)depthimg.at<ushort>(pt) - depthbefore;
				
				if (similarity <= 0.0&&abs(similarity) <= depth_thrf)
				{
				#pragma omp critical 
				{
					ds.point = pt;
					ds.similarity = similarity;

					edgesafter.push_back(ds);
				}
				region.at<uchar>(pt) = CONTOURS;
				}
				else if (similarity >= 0.0&&similarity <= depth_thr)
				{
				#pragma omp critical 
				{
					ds.point = pt;
					ds.similarity = similarity;

					edgesafter.push_back(ds);
				}
				region.at<uchar>(pt) = CONTOURS;
				}
				else
					region.at<uchar>(pt) = VISITED;
			}
		}
	}
	
	//sort(edgesafter.begin(), edgesafter.end());
}

void DepthHumanDetector::region_grow(Mat& depthimg, Point seed, Mat& region,float head_height)
{
	float depthavg = depthimg.at<ushort>(seed);
	vector<DepthSimilarity> edgesbefore,edgesafer;
	edgesbefore.push_back(DepthSimilarity(seed, 0));
	region.at<uchar>(seed) = CONTOURS;
	while (true)
	{
		get_neighborings(depthimg, region, edgesbefore, depthavg, edgesafer);
		
		if (edgesafer.empty())
		{
			//check for whether needs to grow again from bottom
			int body_height = static_cast<int>(head_height * 6.5);
			int body_width = static_cast<int>(head_height/2);
			int start_x = seed.x - body_width;
			int start_y = seed.y;
			bool needs_regrow = false;
			for (int i = start_x; i < start_x + body_width;++i)
			{
				for (int j = start_y; j < start_y + body_height;++j)
				{
					Point pt = Point(i, j);
					if (pt.x<0||pt.x>=depthimg.cols||pt.y<0||pt.y>=depthimg.rows||region.at<uchar>(pt)!=NOVISITED)
					{
						continue;
					}
					
					if (is_body(depthimage,depthavg,pt))
					{
						needs_regrow = true;
						region.at<uchar>(pt) = CONTOURS;
						float similarity = (float)depthimg.at<ushort>(pt) - depthavg;
						DepthSimilarity ds(pt, similarity);
						edgesafer.push_back(ds);
						break;
					}
					else
						region.at<uchar>(pt) = VISITED;
				}
			}
			if (!needs_regrow)
			{
				break;
			}
		}

		edgesbefore.swap(edgesafer);
		depthavg = mean(depthimg, (region == CONTOURS))[0];
		edgesafer.clear();
	}

}

void DepthHumanDetector::depth_precessing(Mat& src, Mat& dst)
{

		if (dst.empty())
		{
			dst = Mat::zeros(src.size(), src.type());
		}
		assert(src.size() == dst.size() && src.type() == dst.type() && src.channels() == 1);

		ushort* dstdata = dst.ptr<ushort>(0);
		ushort* srcdata = src.ptr<ushort>(0);

		int width = src.cols;
		int height = src.rows;
		int widthstep = src.step[0] / src.elemSize();
		Point minloc,maxloc;

		minloc.x =0;
		minloc.y = 0;
		maxloc.x = width;
		maxloc.y = height;
		//find roi
		/*for (int rowindex = 0; rowindex < height; ++rowindex)
		{
			for (int colindex = 0; colindex < width; ++colindex)
			{
				int depthindex = colindex + rowindex*widthstep;
				if (srcdata[depthindex] != 0)
				{
					if (minloc.x>colindex)
					{
						minloc.x = colindex;
					}
					if (minloc.y>rowindex)
					{
						minloc.y = rowindex;
					}
					if (maxloc.x<colindex)
					{
						maxloc.x = colindex;
					}
					if (maxloc.y<rowindex)
					{
						maxloc.y = rowindex;
					}

				}
			}
		}*/

		Range rowRange, colRange;
		rowRange.start = minloc.y;
		rowRange.end = maxloc.y;
		colRange.start = minloc.x;
		colRange.end = maxloc.x;

		for (int rowindex = minloc.y; rowindex <= maxloc.y; ++rowindex)
		{
			for (int colindex = minloc.x; colindex <= maxloc.x; ++colindex)
			{
				int depthindex = colindex + rowindex*widthstep;
				int fillindex = depthindex;
				if (srcdata[depthindex] == 0)
				{
					int searchwin = 1;
					bool found = false;
					while (!found)
					{
						int rangex1 = colindex - searchwin;
						int rangex2 = colindex + searchwin;
						int rangey1 = rowindex - searchwin;
						int rangey2 = rowindex + searchwin;

						if (rangex1<minloc.x&&rangex2>maxloc.x&&rangey1<minloc.y&&rangey2>maxloc.y)
						{
							found = true;
						}
						if (!found&&rangey1 >= minloc.y)
						{
							int range1 = rangex1 >= minloc.x ? rangex1 : minloc.x;
							int range2 = rangex2 <= maxloc.x ? rangex2 : maxloc.x;

							for (int i = range1; i < range2 + 1; ++i)
							{
								int index = rangey1*widthstep + i;
								if (srcdata[index] != 0)
								{
									fillindex = index;
									found = true;
									break;
								}
							}
						}
						if (!found&&rangey2 <= maxloc.y)
						{
							int range1 = rangex1 >= minloc.x ? rangex1 : minloc.x;
							int range2 = rangex2 <= maxloc.x ? rangex2 : maxloc.x;

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

						if (!found&&rangex1 >= minloc.x)
						{
							int range1 = rangey1 >= minloc.y ? rangey1 : minloc.y;
							int range2 = rangey2 <= maxloc.y ? rangey2 : maxloc.y;

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

						if (!found&&rangex2<maxloc.x)
						{
							int range1 = rangey1 >= minloc.y ? rangey1 : minloc.y;
							int range2 = rangey2 <= maxloc.y ? rangey2 : maxloc.y;

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

		//Mat dst8U;
		//convert16to8U(src, dst8U);
		//cv::imshow("ori", dst8U);
		//convert16to8U(dst, dst8U);
		//cv::imshow("filled", dst8U);
		Mat blured;
		cv::medianBlur(dst, blured, 3);
		blured.copyTo(dst);
		//convert16to8U(dst, dst8U);
		//cv::imshow("filterd", dst8U);
		//cv::waitKey(0);
}

void DepthHumanDetector::sort_heads(vector<Point>& detected_heads)
{
	if (detected_heads.size()<=1)
	{
		return;
	}

	
	for (int i = 1; i < detected_heads.size();++i)
	{
		int insertpos = i-1;
		Point curpos = detected_heads[i];
		ushort cur = depthimage.at<ushort>(detected_heads[i]);
		for (insertpos = i - 1; insertpos >= 0;--insertpos)
		{
			if (cur>depthimage.at<ushort>(detected_heads[insertpos]))
			{
				break;
			}
			else
			{
				detected_heads[insertpos + 1] = detected_heads[insertpos];
				final_head_features[insertpos + 1] = final_head_features[insertpos];
			}
		}
		detected_heads[insertpos+1] = curpos;
		final_head_features[insertpos + 1] = final_head_features[i];

	}
}

bool DepthHumanDetector::is_body(Mat & depthimag,float depthavg, Point pt)
{
	float similarity = (float)depthimag.at<ushort>(pt) - depthavg;

	if (similarity <= 0.0&&abs(similarity) <= depth_thrf)
	{	
		return true;
	}
	else if (similarity >= 0.0&&similarity <= depth_thr)
	{
		return true;
	}
	else
		return false;
}



