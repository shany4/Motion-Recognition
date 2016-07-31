/////////////////////////////////////////////////////////////////////////////
//
// TOPIC: Motion Recognition
// Designer: Ying Shan
//
/////////////////////////////////////////////////////////////////////////////

/** header inclusion */
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/opencv.hpp"
#include "opencv/cxcore.h"
#include "opencv/cvaux.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <math.h>
#include <opencv2/core/operations.hpp>

using namespace std;
using namespace cv;

//Defined different cluster basing on position
struct PointLike{
	    PointLike(int threshold){
              this->threshold = threshold;
		}
		bool operator()(cv::Point p1,cv::Point p2){
               int x = p1.x - p2.x;
               int y = p1.y - p2.y;
               return x*x+y*y <= threshold*threshold;
		}
		int threshold;
 };

void DrawArrow(cv::Mat& img, cv::Point pStart, cv::Point pEnd, int len, int alpha,             
			   cv::Scalar color, int thickness, int lineType);
void GetIxIyIt( Mat frame, Mat paddedInput[2], Mat &ddxMat, Mat &ddyMat, Mat &ddtMat, 
				double &xmin, double &xmax, double &ymin, double &ymax, double &tmin, double &tmax );
void ShowIxIyIt( Mat frame,Mat &ddxMat, Mat &ddyMat, Mat &ddtMat, Mat &dxMat,Mat &dyMat, Mat &dtMat,
				double xmin, double xmax, double ymin, double ymax, double tmin, double tmax );
void GetVelocity( Mat frame, int regionsize, Mat &ddxMat, Mat &ddyMat, Mat &ddtMat, 
				Mat &velocityX, Mat &velocityY, Mat &Movement );
Point moveball( Mat game, int orientation1, int orientation2, Point center );

int main( )
{
	cv::VideoCapture cap;             // Open camera
    
	cap.open(0);
	
	if(!cap.isOpened())
	{
		printf("Error: could not load a camera or video.\n");
	}
	Mat frame[2],paddedInput[2],dxMat,dyMat,dtMat, displaypicture, Movement;
	Mat ddxMat,ddyMat,ddtMat,game;

	double xmax = -1000,xmin = 1000, ymax = -1000, ymin = 1000, tmax = -1000, tmin = 1000;
	namedWindow("video", 1);
                                              //Get a frame and initialize it
	cap >> frame[1]; 
	resize(frame[1],frame[1],Size(1920/3,1080/3));
	frame[1].copyTo(displaypicture);

	cvtColor(frame[1],frame[1],CV_BGR2GRAY);
	frame[1].copyTo(frame[0]);
	dxMat.create( frame[1].size(),frame[1].type() );
	dyMat.create( frame[1].size(),frame[1].type() );
	dtMat.create( frame[1].size(),frame[1].type() );
	Movement.create( frame[1].size(),frame[1].type() );
	game.create( displaypicture.size(),displaypicture.type() );

	game.setTo(0);
	Point Pointcenter( game.size().width/2 , game.size().height/2 );
	circle( game, Pointcenter, 30 , Scalar(255,255,255), 3);

	ddxMat.create( frame[1].size(),CV_64F );
	ddyMat.create( frame[1].size(),CV_64F );
	ddtMat.create( frame[1].size(),CV_64F );
	for( int i=1;;)
	{
		waitKey(30);
		dtMat.setTo(0);
		Movement.setTo(0);

		
		if(!frame[i].data)
		{
			printf("Error: no frame data.\n");
			break;
		}
		copyMakeBorder(frame[0],paddedInput[0],0,1,0,1,cv::BORDER_REPLICATE);      //Make boeder
		copyMakeBorder(frame[1],paddedInput[1],0,1,0,1,cv::BORDER_REPLICATE);
        
        //1.Calculate Ix, Iy and It using two continual frames
		GetIxIyIt(frame[1],paddedInput,ddxMat,ddyMat,ddtMat,xmin,xmax,ymin,ymax,tmin,tmax);

		double Sumxx = 0.0, Sumxy = 0.0, Sumyy = 0.0, Sumxt = 0.0, Sumyt = 0.0;
		Mat velocityX, velocityY; 

		velocityX.create( frame[1].size(),CV_64F );
		velocityY.create( frame[1].size(),CV_64F );

		velocityX.setTo(0);
		velocityY.setTo(0);
		int regionsize = 10;

        //2.Calculate velocity of moving pixels by Lk method
		GetVelocity( frame[1],regionsize,ddxMat,ddyMat,ddtMat,velocityX,velocityY,Movement );

        //3.Start to sort out moveing pixels
		PointLike plike(regionsize + 150);
		std::vector<int> labels;
		std::vector<cv::Point> pts_v; 

		vector<Point>::iterator it = pts_v.begin();
		int flag = 0;
		pts_v.resize(1);
		it = pts_v.begin();
		for( int j=0 ; j < Movement.rows; j++ )
		{
			for( int k=0; k< Movement.cols; k++ )
			{	
				if( Movement.at<uchar>(j,k) > 200 ) 
				{
					flag++;
					*it = Point(j,k);
					it++;
					if (it == pts_v.end())
					{
						pts_v.resize(pts_v.size()+1);
						it = pts_v.begin() + flag;
					}
				}
			}
		}

		Scalar colourtable[5];
		colourtable[0] = Scalar(0,0,255);
		colourtable[1] = Scalar(0,255,0);
		colourtable[2] = Scalar(255,0,0);
		colourtable[3] = Scalar(125,125,125);
		colourtable[4] = Scalar(125,0,125);

        //4.Seperate different pixels in different clusters
		int count;
		count = partition(pts_v,labels,plike);

        //6.Seperate orientation into eight parts
        // and called a vote to identify compound speed of each cluster
		for( int j=0; j < pts_v.size(); j++ )
		{
			circle(displaypicture,Point(pts_v[j].y,pts_v[j].x),2,colourtable[labels[j]],2);
		}


		double Vxsum[10] = {0.0}, Vysum[10] = {0.0}, Vsum[10] = {0.0};
		int num[10] = {0},xsum[10] = {0}, ysum[10] = {0};
		int maxlabel = -1;
		int right = 0, left = 0, up = 0, down = 0, up_right, down_right, up_left, down_left ;
		double angle = 0;
		for( int j=0 ; j < pts_v.size(); j++ )
		{ 
			if( labels[j] > maxlabel ) maxlabel = labels[j];
		}

		for( int k=0; k<maxlabel; k++ )
		{		
			right = 0; left = 0; up = 0; down = 0;
			up_right = 0; down_right = 0; up_left = 0; down_left = 0;
			for( int j=0 ; j < pts_v.size(); j++ )
			{ 
				if( labels[j] == k )
				{
					angle = atan2(velocityY.at<double>(pts_v[j].x,pts_v[j].y),velocityX.at<double>(pts_v[j].x,pts_v[j].y));
					if( angle >= -0.5 && angle <= 0.5 ) right ++;
					else if( angle > 0.5 && angle < 1.3 ) down_right ++;
					else if( angle >= 1.3 && angle <= 1.9 ) down ++;
					else if( angle > 1.9 && angle < 2.7 ) down_left ++;
					else if( angle >= 2.7 || angle <= -2.7 ) left ++;
					else if( angle > -1.3 && angle < -0.5 ) up_right ++;
					else if( angle >= -1.9 && angle <= -1.3 ) up ++;
					else if( angle > -2.7 && angle < -1.9 ) up_left ++;
					
					Vxsum[k] = Vxsum[k] + velocityX.at<double>(pts_v[j].x,pts_v[j].y);
					Vysum[k] = Vysum[k] + velocityY.at<double>(pts_v[j].x,pts_v[j].y);
					xsum[k] = pts_v[j].y + xsum[k];
					ysum[k] = pts_v[j].x + ysum[k];
					num[k]++;
				}

				Vsum[k] = sqrt(Vxsum[k]*Vxsum[k] + Vysum[k]*Vysum[k])/num[k];	 
			}
			if( right > left && right >= up && right >= down && right >= down_right && right >= down_left && right >= up_left && right >= up_right )
			{
				DrawArrow( displaypicture, Point((int)xsum[k]/num[k], (int)ysum[k]/num[k]), 
					Point((int)(xsum[k]/num[k]+20*Vsum[k]), (int)(ysum[k]/num[k])),35, 45, Scalar(255,0,0), 16, CV_AA);
				Pointcenter = moveball( game, 1, 0, Pointcenter);
			}
			if( up > left && up > right && up > down && up > down_right && up > down_left && up > up_right && up > up_left )
			{
				DrawArrow( displaypicture, Point((int)xsum[k]/num[k], (int)ysum[k]/num[k]), 
					Point((int)(xsum[k]/num[k]), (int)(ysum[k]/num[k]-20*Vsum[k])),35, 45, Scalar(255,0,0), 16, CV_AA);
				Pointcenter = moveball( game, 0, -1, Pointcenter);
			}
			if( left > right && left >= up && left >= down && left >= up_right && left >= up_left && left >= down_right && left >= down_left )
			{
				DrawArrow( displaypicture, Point((int)xsum[k]/num[k], (int)ysum[k]/num[k]), 
					Point((int)(xsum[k]/num[k]-20*Vsum[k]), (int)(ysum[k]/num[k])),35, 45, Scalar(255,0,0), 16, CV_AA);
				Pointcenter = moveball( game, -1, 0, Pointcenter);
			}
			if( down > left && down > up && down > right && down > up_right && down > up_left && down > down_right && down > down_left )
			{
				DrawArrow( displaypicture, Point((int)xsum[k]/num[k], (int)ysum[k]/num[k]), 
					Point((int)(xsum[k]/num[k]), (int)(ysum[k]/num[k]+20*Vsum[k])),35, 45, Scalar(255,0,0), 16, CV_AA);
				Pointcenter = moveball( game, 0, 1,Pointcenter);
			}
			if( down_right > up && down_right > down && down_right > right && down_right > left && down_right > down_left && down_right > up_left && down_right > up_right )
			{
				DrawArrow( displaypicture, Point((int)xsum[k]/num[k], (int)ysum[k]/num[k]), 
					Point((int)(xsum[k]/num[k]+20*Vsum[k]), (int)(ysum[k]/num[k]+20*Vsum[k])),35, 45, Scalar(255,0,0), 16, CV_AA);
				Pointcenter = moveball( game, 1, 1,Pointcenter);
			}
			if( down_left > up && down_left > right && down_left > left && down_left > down && down_left > down_right && down_left > up_right && down_left > up_left )
			{
				DrawArrow( displaypicture, Point((int)xsum[k]/num[k], (int)ysum[k]/num[k]), 
					Point((int)(xsum[k]/num[k]-20*Vsum[k]), (int)(ysum[k]/num[k]+20*Vsum[k])),35, 45, Scalar(255,0,0), 16, CV_AA);
				Pointcenter = moveball( game, 1, -1, Pointcenter);
			}
			if( up_right > up && up_right > down && up_right > right && up_right > left && up_right > down_right && up_right > down_left && up_right > up_left )
			{
				DrawArrow( displaypicture, Point((int)xsum[k]/num[k], (int)ysum[k]/num[k]), 
					Point((int)(xsum[k]/num[k]+20*Vsum[k]), (int)(ysum[k]/num[k]-20*Vsum[k])),35, 45, Scalar(255,0,0), 16, CV_AA);
				Pointcenter = moveball( game, -1, 1, Pointcenter);
			}
			if( up_left > up && up_left > down && up_left > right && up_left > left && up_left > down_right && up_left > down_left && up_left > up_right )
			{
				DrawArrow( displaypicture, Point((int)xsum[k]/num[k], (int)ysum[k]/num[k]), 
					Point((int)(xsum[k]/num[k]-20*Vsum[k]), (int)(ysum[k]/num[k]-20*Vsum[k])),35, 45, Scalar(255,0,0), 16, CV_AA);
				Pointcenter = moveball( game, -1, -1, Pointcenter);
			}
		}
        
        //7.Show the results
		ShowIxIyIt(frame[1],ddxMat,ddyMat,ddtMat,dxMat,dyMat,dtMat,xmin,xmax,ymin,ymax,tmin,tmax);

		frame[1].copyTo(frame[0]);
		cap >> frame[1];
		resize(frame[1],frame[1],Size(1920/3,1080/3));
		imshow("video", displaypicture);
		imshow("dx",dxMat);
		imshow("dy",dyMat);
		imshow("dt",dtMat);
		imshow("move",Movement);
		imshow("game",game);
		frame[1].copyTo(displaypicture);
		cvtColor(frame[1],frame[1],CV_BGR2GRAY);
	}
}

void DrawArrow(cv::Mat& img, cv::Point pStart, cv::Point pEnd, int len, int alpha,             
			   cv::Scalar color, int thickness, int lineType)
{    
	const double PI = 3.1415926;    
	Point arrow;    
 
	double angle = atan2((double)(pStart.y - pEnd.y), (double)(pStart.x - pEnd.x));  
	line(img, pStart, pEnd, color, thickness, lineType);  

    arrow.x = pEnd.x + len * cos(angle + PI * alpha / 180);     
	arrow.y = pEnd.y + len * sin(angle + PI * alpha / 180);  
	line(img, pEnd, arrow, color, thickness, lineType);   
	arrow.x = pEnd.x + len * cos(angle - PI * alpha / 180);     
	arrow.y = pEnd.y + len * sin(angle - PI * alpha / 180);    
    line(img, pEnd, arrow, color, thickness, lineType);
}

void GetIxIyIt( Mat frame, Mat paddedInput[2], Mat &ddxMat, Mat &ddyMat, Mat &ddtMat, 
				double &xmin, double &xmax, double &ymin, double &ymax, double &tmin, double &tmax )
{
	double x[4], y[4], t[4];
	double dx, dy, dt;
	for( int j=0; j<frame.rows; j++ )
	{
		for( int k=0; k<frame.cols; k++ )
		{		
			x[0] = (double) paddedInput[0].at<uchar>(j,k+1) - paddedInput[0].at<uchar>(j,k);
			x[1] = (double) paddedInput[0].at<uchar>(j+1,k+1) - paddedInput[0].at<uchar>(j+1,k);
			x[2] = (double) paddedInput[1].at<uchar>(j,k+1) - paddedInput[1].at<uchar>(j,k);
			x[3] = (double) paddedInput[1].at<uchar>(j+1,k+1) - paddedInput[1].at<uchar>(j+1,k);
			dx = ( x[0]+x[1]+x[2]+x[3] ) / 4;
			if( dx < xmin ) xmin = dx;
			if( dx > xmax ) xmax = dx;
			ddxMat.at<double>(j,k) = dx;

			y[0] = (double) paddedInput[0].at<uchar>(j+1,k) - paddedInput[0].at<uchar>(j,k);
			y[1] = (double) paddedInput[0].at<uchar>(j+1,k+1) - paddedInput[0].at<uchar>(j,k+1);
			y[2] = (double) paddedInput[1].at<uchar>(j+1,k) - paddedInput[1].at<uchar>(j,k);
			y[3] = (double) paddedInput[1].at<uchar>(j+1,k+1) - paddedInput[1].at<uchar>(j,k+1);
			dy = ( y[0]+y[1]+y[2]+y[3] ) / 4;
			if( dy < ymin ) ymin = dy;
			if( dy > ymax ) ymax = dy;
			ddyMat.at<double>(j,k) = dy;

			t[0] =( double ) (paddedInput[1].at<uchar>(j,k) - paddedInput[0].at<uchar>(j,k));
			t[1] =( double ) (paddedInput[1].at<uchar>(j+1,k) - paddedInput[0].at<uchar>(j+1,k));
			t[2] =( double ) (paddedInput[1].at<uchar>(j,k+1) - paddedInput[0].at<uchar>(j,k+1));
			t[3] =( double ) (paddedInput[1].at<uchar>(j+1,k+1) - paddedInput[0].at<uchar>(j+1,k+1));
			dt = ( t[0] + t[1] + t[2] + t[3] ) / 4;
			if( dt < tmin ) tmin = dt;
			if( dt > tmax ) tmax = dt;
			ddtMat.at<double>(j,k) = dt;
		}
	}
}

void ShowIxIyIt( Mat frame,Mat &ddxMat, Mat &ddyMat, Mat &ddtMat, Mat &dxMat,Mat &dyMat, Mat &dtMat,
				double xmin, double xmax, double ymin, double ymax, double tmin, double tmax )
{
	for( int j=0; j<frame.rows; j++ )
	{
		for( int k=0; k<frame.cols; k++ )
		{		
			dxMat.at<uchar>(j,k) = (uchar)  ( ( ddxMat.at<double>(j,k)+ abs(xmin) )* 255 / (abs(xmin) + abs(xmax) ) );
			dyMat.at<uchar>(j,k) = (uchar)  ( ( ddyMat.at<double>(j,k)+ abs(ymin) )* 255 / (abs(ymin) + abs(ymax) ) );
			dtMat.at<uchar>(j,k) = (uchar)  ( ( ddtMat.at<double>(j,k)+ abs(tmin) )* 255 / (abs(tmin) + abs(tmax) ) );
		}
	}
}

void GetVelocity( Mat frame, int regionsize, Mat &ddxMat, Mat &ddyMat, Mat &ddtMat, Mat &velocityX, Mat &velocityY, Mat &Movement )
{
	double Sumxx = 0.0, Sumxy = 0.0, Sumyy = 0.0, Sumxt = 0.0, Sumyt = 0.0;
	Mat A( 2, 2, CV_64F);
	Mat Ainv( 2, 2, CV_64F);
	Mat B( 2, 1, CV_64F);
	Mat V( 2, 1, CV_64F);
	for( int j=0; j<frame.rows-regionsize; j=j+regionsize )
	{
		for( int k=0; k<frame.cols-regionsize; k=k+regionsize )
		{							
			Sumxx = 0.0; Sumxy = 0.0; Sumyy = 0.0; Sumxt = 0.0; Sumyt = 0.0;
			if( abs( ddtMat.at<double>(j,k) ) > 40 )
			{
				for( int a=0; a<regionsize; a++ )
				{
					for( int b=0; b<regionsize; b++ )
					{
						Sumxx = Sumxx + ddxMat.at<double>(j+a,k+b) * ddxMat.at<double>(j+a,k+b);
						Sumxy = Sumxy + ddxMat.at<double>(j+a,k+b) * ddyMat.at<double>(j+a,k+b);
						Sumyy = Sumyy + ddyMat.at<double>(j+a,k+b) * ddyMat.at<double>(j+a,k+b);
						Sumxt = Sumxt - ddxMat.at<double>(j+a,k+b) * ddtMat.at<double>(j+a,k+b);
						Sumyt = Sumyt - ddyMat.at<double>(j+a,k+b) * ddtMat.at<double>(j+a,k+b);
					}
				}
				A.at<double>(0,0) = Sumxx;
				A.at<double>(1,0) = Sumxy;
				A.at<double>(0,1) = Sumxy;
				A.at<double>(1,1) = Sumyy;
				B.at<double>(0,0) = Sumxt;
				B.at<double>(1,0) = Sumyt;
				invert(A,Ainv);
				V = Ainv*B;
				double Vx = V.at<double>(0,0);
				double Vy = V.at<double>(1,0);

				velocityX.at<double>(j,k) = Vx;
				velocityY.at<double>(j,k) = Vy;

				if( ( abs(Vx) + abs(Vy) ) > 3 && ( abs(Vx) + abs(Vy) ) < 20 )
				{
					Movement.at<uchar>(j,k) = 255;
				}
			}
		}
	}

}

Point moveball( Mat game, int orientation1, int orientation2, Point center )
{
	if( center.x > game.size().width || center.y > game.size().height || center.x < 0 || center.y < 0 )
	{ 
		center = Point( game.size().width/2 , game.size().height/2 ); 
	}
	game.setTo(0);
	center = Point( center.x + 5*orientation1*2 , center.y + 5*orientation2*2 );
	if( orientation1 > 0 )
		circle( game, center, 30, Scalar(0,0,255),3);
	else if( orientation1 < 0 )
		circle( game, center, 30, Scalar(0,255,0),3);
	else
		circle( game, center, 30, Scalar(0,0,255),3);
	return center;
}
