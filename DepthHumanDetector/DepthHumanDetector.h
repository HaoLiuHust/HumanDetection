#ifndef DEPTHHUMANDETECTOR_H
#define DEPTHHUMANDETECTOR_H
#include "PixelSimilarity.h"
#include "Template.h"

#include <iostream>
#include <vector>
#include <cv.h>
#include <highgui.h>
#include <cxcore.h>

using namespace std;
using namespace cv;
class DepthHumanDetector
{
public:
	DepthHumanDetector();
	void detect_face_depth(Mat& depth_image, Mat& disparity_image,vector<Rect>& detected_faces);
	void detect_body_depth(Mat& depth_image, Mat& disparity_image,vector<Mat>& detected_bodies);
	vector<Point> get_detectedheads();
	~DepthHumanDetector();
	double canny_thr1;
	double canny_thr2;
	double chamfer_thr;
	double arc_thr_low;
	double arc_thr_high;
	double approx_poly_thr;
	double max_suppression;
	double scale_factor;
	double match3D_thr;
	double depth_thr;
	double depth_thrf;
	int scales;
	int scales_default;
	int framenum;
	int update_rate;

private:
	void load_templates();
	bool check_dimensions(Mat& img_depth);
	void chamfer_matching(Mat& image, Mat& template_im, Mat& canny_im,vector<Point3f>& heads);
	void compute_headparameters(Mat& image, vector<Point3f>& chamfer,vector<PixelSimilarity>& parameters_head);
	void false_positives(vector<PixelSimilarity>& potential_heads, Mat& canny_im, int thr, int thr2,vector<PixelSimilarity>& false_positive_removed_heads);
	void match_template3D(Mat& image_disparity, vector<PixelSimilarity>& potentials, vector<PixelSimilarity>& heads, int n);
	void merge_rectangles(vector<PixelSimilarity>& tmpcont,vector<PixelSimilarity>& merged_rect);
	void boundary_filter(Mat& src, Mat& filtered);
	void extract_contours(Mat& depthimg, vector<Point>& heads, vector<Mat>& bodies);
	void get_neighborings(Mat& depthimg, Mat& region, vector<DepthSimilarity>& edgesbefore,float depthbefore, vector<DepthSimilarity>& edgesafter);
	void region_grow(Mat& depthimg, Point seed, Mat& region,float head_height);
	void depth_precessing(Mat& src, Mat& dst);
	void sort_heads(vector<Point>& detected_heads);
	bool is_body(Mat& depthimag, float depthavg,Point pt);
private:
	vector<Template> templates;
	vector<PixelSimilarity> final_head_features;
	vector<Point> detected_heads;
	Mat depthimage;
	//Mat canny_im;
};

#endif
