#pragma once
#ifndef _FUNC_
#define _FUNC_
#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;


void fire_check(Mat frame, Mat r_frame, Point &fire_point);     //��������
void diff_backgroud(Mat gray_frame, Mat back_ground, Mat &abs_frame, Mat &out_frame);   //�������
void curve_fitting(vector<Point> &points, int n, Mat &src);    //�������
void draw_curve_fit(Mat &frame, Mat A);  //��������
void background_diff(Mat frame, Ptr<BackgroundSubtractor> model, Mat &foregroud, Mat &background, Mat &foregroundMask, bool doUpdateModel, bool doSmoothMask);
void connectfind(Mat out_frame, Mat &frame, Point p, Point fire_point);

#endif // !_Fuction_
