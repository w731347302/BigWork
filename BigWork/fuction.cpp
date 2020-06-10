#include <iostream>
#include <opencv2/opencv.hpp>
#include "fuction.h"

using namespace std;
using namespace cv;

int Rt = 70;  //红色阈值
int St = 7;  //S饱和度阈值


void fire_check(Mat frame,Mat r_frame,Point &fire_point)     //火焰检测标记
{
	vector<Mat> srcs;
	Mat fire;
	fire.create(frame.size(), CV_8UC1);
	split(frame, srcs);
	for (int i = 0; i < frame.rows; i++)
	{
		for (int j = 0; j < frame.cols; j++)
		{
			float B, G, R;
			B = float(srcs[0].at<uchar>(i, j));
			G = float(srcs[1].at<uchar>(i, j));
			R = float(srcs[2].at<uchar>(i, j));

			int maxmal = max(max(B, G), R);
			int minmal = min(min(B, G), R);
			double s = 1 - 3 * minmal / (R + G + B);     //RGB 转 S计算公式
			if (R > Rt && R >= G && G >= B && s > 0.15 && s > ((255 - R) * St / Rt))   //三个判决条件
				fire.at<uchar>(i, j) = 255;
			else
				fire.at<uchar>(i, j) = 0;
		}
	}
	Mat kernal = getStructuringElement(MORPH_RECT, Size(9, 9));
	dilate(fire, fire, kernal, Point(-1, -1), 2);

	vector<vector<Point>> con;
	findContours(fire, con, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	//drawContours(frame, con, -1, Scalar(255, 0, 0));
	for (vector<Point> c : con)
	{
		Rect rect = boundingRect(c);
		rectangle(r_frame, rect, Scalar(255, 0, 0));
		fire_point.x = rect.x + rect.width / 2;
		fire_point.y = rect.y + rect.height / 2;
		putText(r_frame, "fire", Point(rect.x, rect.y), FONT_HERSHEY_PLAIN, 1.0, Scalar(255, 0, 0));
	}
}

void diff_backgroud(Mat gray_frame, Mat back_ground,Mat &abs_frame,Mat &out_frame)   //背景差分
{
	absdiff(gray_frame, back_ground, abs_frame);
	threshold(abs_frame, out_frame, 75, 255, THRESH_BINARY);
	Mat kernal = getStructuringElement(MORPH_RECT, Size(5, 5));
	dilate(out_frame, out_frame, kernal, Point(-1, -1), 1);
	morphologyEx(out_frame, out_frame, MORPH_OPEN, kernal);
}

void curve_fitting(vector<Point> &points, int n, Mat &src)    //曲线拟合
{
	int N = points.size();
	Mat mask_X = Mat::zeros(n + 1, n + 1, CV_64FC1);
	for (int i = 0; i < n + 1; i++)
	{
		for (int j = 0; j < n + 1; j++)
		{
			for (int k = 0; k < N; k++)
			{
				mask_X.at<double>(i, j) = mask_X.at<double>(i, j) + pow(points[k].x, i + j);
			}
		}
	}

	Mat mask_Y = Mat::zeros(n + 1, 1, CV_64FC1);
	for (int i = 0; i < n + 1; i++)
	{
		for (int k = 0; k < N; k++)
		{
			mask_Y.at<double>(i, 0) = mask_Y.at<double>(i, 0) + pow(points[k].x, i)*points[k].y;
		}
	}

	src = Mat::zeros(n + 1, 1, CV_64FC1);
	solve(mask_X, mask_Y, src, DECOMP_LU);
}

void draw_curve_fit(Mat &frame, Mat A)  //绘制曲线
{
	vector<Point> points_fit;
	for (int i = 0; i < 400; i++)
	{
		double y = A.at<double>(0, 0) + A.at<double>(1, 0)*i + A.at<double>(2, 0)*pow(i, 2) + A.at<double>(3, 0)*pow(i, 3);
		points_fit.push_back(Point(i, y));
	}
	polylines(frame, points_fit, false, Scalar(255, 0, 0),2);
}

void background_diff(Mat frame, Ptr<BackgroundSubtractor> model, Mat &foregroud, Mat &background,Mat &foregroundMask,bool doUpdateModel,bool doSmoothMask)
{
	model->apply(frame, foregroundMask, doUpdateModel ? -1 : 0);
	//imshow("image", frame);
	if (doSmoothMask)
	{
		GaussianBlur(foregroundMask, foregroundMask, Size(11, 11), 3.5, 3.5);
		threshold(foregroundMask, foregroundMask, 75, 255, THRESH_BINARY);
		Mat kernal = getStructuringElement(MORPH_RECT, Size(5, 5));
		dilate(foregroundMask, foregroundMask, kernal);
	}
	if (foregroud.empty())
		foregroud.create(frame.size(), frame.type());
	foregroud = Scalar::all(0);
	frame.copyTo(foregroud, foregroundMask);
	//imshow("foreground mask", foregroundMask);
	//imshow("foreground image", foregroud);
	model->getBackgroundImage(background);
	//if (!background.empty())
		//imshow("mean background image", background);
}

void connectfind(Mat out_frame,Mat &frame,Point p,Point fire_point)
{
	Mat labels, states, cent;
	vector<Point> points;
	int num = connectedComponentsWithStats(out_frame, labels, states, cent);
	cout << num - 1 << endl;
	for (int i = 0; i < num; i++)
	{
		Rect rect;
		rect.x = states.at<int>(i, 0);
		rect.y = states.at<int>(i, 1);
		rect.width = states.at<int>(i, 2);
		rect.height = states.at<int>(i, 3);
		double r = sqrt(pow(rect.width, 2) + pow(rect.height, 2));
		if (rect.x >= p.x && rect.x < 380 && rect.width>85 && rect.width <100)
		{
			//rectangle(frame, rect, Scalar(255, 0, 0));
			Point mid(rect.x + rect.width, rect.y + rect.height);
			circle(frame, mid, 3, Scalar(255, 0, 0));
			points.push_back(mid);
			//waterPoints.push_back(mid);
		}
	}
	if (!points.empty())
	{
		points.push_back(p);
		points.push_back(fire_point);
		Mat A;
		curve_fitting(points, 3, A);
		draw_curve_fit(frame, A);
	}
}