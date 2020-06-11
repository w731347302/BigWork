#include <iostream>
#include <opencv2/opencv.hpp>
#include <stdlib.h>
#include "fuction.h"
#include <ctime>

using namespace std;
using namespace cv;

clock_t c_start;
clock_t c_end;

vector<Point> waterPoints;
Point p;
Point fire_point;
void on_mouse(int EVENT, int x, int y, int flags, void* userdata);
Mat frame;

int main()
{
	VideoCapture cap("Work.mp4");
	Mat gray_frame;
	Mat back_ground;    //背景图
	Mat abs_frame;
	Mat res_frame;    //去除背景的图
	Mat out_frame;      //背景差分后二值图
	Point choose_point;
	//
	bool doUpdateModel = true;
	bool doSmoothMask = false;
	//Mat foreground, backgournd, foregroundMask;
	Ptr<BackgroundSubtractor> model = createBackgroundSubtractorKNN();
	//
	int cnt = 0;   //第一帧选取喷射点
	while (1)
	{
		
		cap >> frame;
		if (frame.empty())
		{
			cout << "finished" << endl;
			return 0;
		}
		c_start = clock();
		cvtColor(frame, gray_frame, COLOR_BGR2GRAY);
		if (cnt == 0)
		{
			gray_frame.copyTo(back_ground);
			namedWindow("final_window");
			imshow("final_window", frame);
			setMouseCallback("final_window", on_mouse, 0);
			waitKey(0);
			destroyAllWindows();
		}
		else
		{
			//background_diff(frame, model, res_frame, back_ground, out_frame, doUpdateModel, doSmoothMask);
			diff_backgroud(gray_frame, back_ground, abs_frame, out_frame);
			frame.copyTo(res_frame, out_frame);
			fire_check(res_frame, frame, fire_point);
			//fire_check(res_frame, frame, fire_point);
			Mat kernal = getStructuringElement(MORPH_CROSS, Size(5, 3));
			erode(out_frame, out_frame, kernal);
			dilate(out_frame, out_frame, kernal);
			connectfind(out_frame, frame, p, fire_point);
			imshow("res_frame", res_frame);
			imshow("frame", frame);
			imshow("out_frame", out_frame);
			waitKey(30);
		}
		cnt++;
		res_frame = 0;
		c_end = clock();
		double endtime = (double)(c_end - c_start) / CLOCKS_PER_SEC;
		cout << endtime * 1000 << "ms" << endl;
	}

	cout << "finished" << endl;
	return 0;
}

void on_mouse(int EVENT, int x, int y, int flags, void* )   //鼠标回调函数
{
	switch (EVENT)
	{
	case EVENT_LBUTTONDOWN:
	{
		p.x = x;
		p.y = y;
		waterPoints.push_back(p);
		putText(frame, "point", Point(x, y), FONT_HERSHEY_PLAIN, 1.0, Scalar(255, 0, 0));
		circle(frame, p, 2, Scalar(255, 0, 0));
		imshow("final_window", frame);
	}
	}
}