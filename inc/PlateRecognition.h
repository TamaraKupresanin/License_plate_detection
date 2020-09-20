#ifndef NOISEREDUCTION_H_
#define NOISEREDUCTION_H_
#include <QDebug>


#ifndef MY_MAIN       
#define MY_MAIN

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

#include "DetectPlates.h"
#include "PossiblePlate.h"
#include "DetectChars.h"

#include<iostream>
#include<conio.h>          

const cv::Scalar SCALAR_BLACK = cv::Scalar(0.0, 0.0, 0.0);
const cv::Scalar SCALAR_WHITE = cv::Scalar(255.0, 255.0, 255.0);
const cv::Scalar SCALAR_YELLOW = cv::Scalar(0.0, 255.0, 255.0);
const cv::Scalar SCALAR_GREEN = cv::Scalar(0.0, 255.0, 0.0);
const cv::Scalar SCALAR_RED = cv::Scalar(0.0, 0.0, 255.0);

const int MIN_THRESHOLD_VALUE = 80;

void licensePlateRecognition(const QImage& inImgs, char U_buff[],char V_buff[],uchar input[], int xSize, int ySize);
void detectEdgeSobel(uchar Y_buff_H[], int xSize, int ySize);
void performGaussFilter(uchar input[], int xSize, int ySize, int N, double sigma);
void performMedianFilter(uchar input[], int xSize, int ySize, int N);

void drawRedRectangleAroundPlate(QImage& copyImage, PossiblePlate& licPlate);

void threshold(uchar Y_buff[], int xSize,int ySize);
void writeLicensePlateCharsOnImage(QImage& copyImage, PossiblePlate& licPlate);

# endif	// MAIN

#endif //NOISEREDUCTION_H_