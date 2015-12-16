#define _CRT_SECURE_NO_WARNINGS
#include "DepthHumanDetector.h"
#include "Utils.h"
#include "templmatch.h"
#include <omp.h>

const Size2i MORPH_SIZE = Size(11, 11);
Utils utils;

DepthHumanDetector::DepthHumanDetector()
{
	load_templates();
	canny_thr1 = 5;
	canny_thr2 = 7;
	chamfer_thr = 10;
	arc_thr_low = 7;
	arc_thr_high = 20;
	scale_factor = 0.75;
	match3D_thr = 0.4;
	scales = 6;
	scales_default = 6;
}

void DepthHumanDetector::detect_face_depth(Mat & depth_image,vector<Rect>& detected_faces)
{
	Mat disparity_image;
	disparity_image = imread("disparity01.png", CV_LOAD_IMAGE_ANYDEPTH);
	//utils.convert16to8U(depth_image, disparity_image);
	Mat element = getStructuringElement(MORPH_RECT, MORPH_SIZE, Point(5, 5));

	utils.preProcessing(disparity_image, disparity_image);
	//imshow("8U", disparity_image);
	depth_image.setTo(0, (disparity_image == 0));
	dilate(depth_image, depth_image, element);
	/*imshow("depth", depth_image);
	waitKey();*/

	vector<PixelSimilarity> final_head_features;
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

	for (unsigned int k = 0; k < templates.size(); k++)
	{
		double t = (double)getTickCount();
		vector<Point3f> head_matched_points;
		chamfer_matching(disparity_image, templates[k].template2d, head_matched_points);
		t = (double)getTickCount() - t;
		//       cout << t*1000./cv::getTickFrequency() << endl;
		vector<PixelSimilarity> head_features;
		compute_headparameters(depth_image, head_matched_points, head_features);
		vector<PixelSimilarity> new_head_features;
		false_positives(head_features, arc_thr_low, arc_thr_high, new_head_features);
		match_template3D(disparity_image, new_head_features, all_head_features, k);
	}
	
	merge_rectangles(all_head_features, final_head_features);
	Rect rect;
	for (unsigned int i = 0; i < final_head_features.size(); i++)
	{
		int wh = final_head_features[i].radius * 2;
		rect = Rect(final_head_features[i].point.x - final_head_features[i].radius, final_head_features[i].point.y - final_head_features[i].radius, wh, wh);
		detected_faces.push_back(rect);
	}
	pyramid.clear();
	chamfer.clear();
	matching.clear();
}



void DepthHumanDetector::load_templates()
{
	string head_template1 = "./pictures/template3.png";
	string head_template2 = "./pictures/template.png";
	string head_template3 = "./pictures/only_head_template.png";
	string head_template4 = "./pictures/only_head_template_small.png";

	string head_template3D1 = "./pictures/template3D.png";
	string head_template3D2 = "./pictures/template3D.png";
	string head_template3D3 = "./pictures/only_head_template_3d.png";
	string head_template3D4 = "./pictures/only_head_template_3d_small.png";

	Mat head_template_im1 = imread(head_template1, CV_LOAD_IMAGE_ANYDEPTH);
	Mat head_template_im2 = imread(head_template2, CV_LOAD_IMAGE_ANYDEPTH);
	Mat head_template_im3 = imread(head_template3, CV_LOAD_IMAGE_ANYDEPTH);
	Mat head_template_im4 = imread(head_template4, CV_LOAD_IMAGE_ANYDEPTH);
	Mat head_template3D_im1 = imread(head_template3D1, CV_LOAD_IMAGE_ANYDEPTH);
	Mat head_template3D_im2 = imread(head_template3D2, CV_LOAD_IMAGE_ANYDEPTH);
	Mat head_template3D_im3 = imread(head_template3D3, CV_LOAD_IMAGE_ANYDEPTH);
	Mat head_template3D_im4 = imread(head_template3D4, CV_LOAD_IMAGE_ANYDEPTH);

	utils.rgb2bw(head_template_im1, head_template_im1);
	head_template_im1.convertTo(head_template_im1, CV_32F);
	utils.rgb2bw(head_template_im2, head_template_im2);
	head_template_im2.convertTo(head_template_im2, CV_32F);
	utils.rgb2bw(head_template_im3, head_template_im3);
	head_template_im3.convertTo(head_template_im3, CV_32F);
	utils.rgb2bw(head_template_im4, head_template_im4);
	head_template_im4.convertTo(head_template_im4, CV_32F);

	if (head_template3D_im1.depth() == 3)
	{
		cvtColor(head_template3D_im1, head_template3D_im1, CV_RGB2GRAY);
	}

	if (head_template3D_im2.depth() == 3)
	{
		cvtColor(head_template3D_im2, head_template3D_im2, CV_RGB2GRAY);
	}

	if (head_template3D_im3.depth() == 3)
	{
		cvtColor(head_template3D_im3, head_template3D_im3, CV_RGB2GRAY);
	}

	if (head_template3D_im4.depth() == 3)
	{
		cvtColor(head_template3D_im4, head_template3D_im4, CV_RGB2GRAY);
	}

	templates.push_back(Template(head_template_im1, head_template3D_im1));
	templates.push_back(Template(head_template_im2, head_template3D_im2));
	templates.push_back(Template(head_template_im3, head_template3D_im3));
	templates.push_back(Template(head_template_im4, head_template3D_im4));

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

void DepthHumanDetector::chamfer_matching(Mat & image, Mat & template_im, vector<Point3f>& chamfer_heads)
{
	double xdiff = template_im.cols / 2;
	double ydiff = template_im.rows / 2;
	Point pdiff = Point(xdiff, ydiff);
	
	Mat matching_thr;

	canny_im.create(image.rows, image.cols, image.depth());
	Canny(image, canny_im, canny_thr1, canny_thr2, 3, true);

	// #pragma omp parallel for default(none) shared(chamfer, pyramid)
	for (int i = 0; i < scales; i++)
	{
		resize(canny_im, pyramid[i], Size(), pow(scale_factor, i), pow(scale_factor, i), INTER_NEAREST);
		distanceTransform((255 - pyramid[i]), chamfer[i], CV_DIST_C, 3);
	}

	// #pragma omp parallel for default(none) shared(chamfer, matching, chamfer_heads, matching_thr, template_im, pdiff, cv_utils)
	for (int j = 0; j < scales; j++)
	{
		double t = (double)getTickCount();
		//       matchTemplate ( chamfer[j], template_im, matching[j], CV_TM_CCOEFF );
		matchTemplateParallel(chamfer[j], template_im, matching[j], CV_TM_CCOEFF);
		t = (double)getTickCount() - t;
		cout << t*1000. / cv::getTickFrequency() << endl;

		double minVal, maxVal;
		Point minLoc, maxLoc;
		normalize(matching[j], matching[j], 0.0, 1.0, NORM_MINMAX);
		minMaxLoc(matching[j], &minVal, &maxVal, &minLoc, &maxLoc);

		threshold(matching[j], matching_thr, 1.0 / chamfer_thr, 1.0, CV_THRESH_BINARY_INV);
		double scale = pow(1.0 / scale_factor, j);

		// #pragma omp critical
		{
			utils.get_non_zeros(matching_thr, matching[j], chamfer_heads, pdiff, scale);
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

void DepthHumanDetector::false_positives(vector<PixelSimilarity>& potential_heads, int thr, int thr2, vector<PixelSimilarity>& false_positive_removed_heads)
{
	vector<PixelSimilarity> updated_heads;

	for (unsigned int i = 0; i < potential_heads.size(); i++)
	{
		Rect roi(potential_heads[i].point.x - potential_heads[i].radius, potential_heads[i].point.y - potential_heads[i].radius, potential_heads[i].radius * 2, potential_heads[i].radius * 2);
		if (!(0 <= roi.x && 0 <= roi.width && roi.x + roi.width < canny_im.cols && 0 <= roi.y && 0 <= roi.height && roi.y + roi.height < canny_im.rows))
		{
			continue;
		}

		Mat cannyroi(canny_im, roi);

		if (!cannyroi.empty())
		{
			vector<vector<Point> > contour;
			findContours(cannyroi, contour, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
			for (unsigned int j = 0; j < contour.size(); j++)
			{
				vector<Point> approx;
				approxPolyDP(contour[j], approx, 5, false);
				if (approx.size() > (unsigned int) thr && approx.size() < (unsigned int)thr2)
				{
					updated_heads.push_back(potential_heads[i]);
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

	for (unsigned int i = 0; i < potentials.size(); i++)
	{
		Rect rect_roi(potentials[i].point.x - potentials[i].radius, potentials[i].point.y - potentials[i].radius, 2 * potentials[i].radius, 2 * potentials[i].radius);
		Mat roi(image_disparity, rect_roi);
		resize(roi, roi, templates[n].template3d.size());
		minMaxLoc(roi, &minVal, &maxVal, 0, 0);
		roi = roi - minVal;
		normalize(roi, roi, 0.0, 255.0, NORM_MINMAX);

		matchTemplate(roi, templates[n].template3d, match, CV_TM_CCOEFF_NORMED);

		minMaxLoc(match, &minVal, &maxVal, &minLoc, &maxLoc);

		if (minVal >= match3D_thr)
		{
			heads.push_back(PixelSimilarity(potentials[i].point, potentials[i].radius, potentials[i].similarity));
		}
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
