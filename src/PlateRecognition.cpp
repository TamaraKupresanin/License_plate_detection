#include "PlateRecognition.h"
#include "ImageFilter.h"
#include "DetectChars.h"
#include <QImage.h>
#include "ColorSpaces.h"
#include "baseapi.h"
#include"allheaders.h"

#include <vector>
#include <algorithm>
#include <fstream> 
#include <QPainter>

using namespace std;

void licensePlateRecognition(const QImage& inImgs, char U_buff[], char V_buff[], uchar Y_buff[], int xSize, int ySize)
{
    cv::Mat CVImgGray;
    cv::Mat result;
    QImage QImg = QImage(xSize, ySize, inImgs.format());
    std::vector<PossibleChar> vectorOfPossibleChars;
    int intCountOfPossibleChars = 0;
    QImage copyImage = QImage(xSize, ySize, inImgs.format());
    procesing_YUV420(Y_buff, U_buff, V_buff, inImgs.width(), inImgs.height(), 1, 1, 1);
    YUV420toRGB(Y_buff, U_buff, V_buff, inImgs.width(), inImgs.height(), copyImage.bits());

    std::vector<std::vector<cv::Point> > contours;
    std::vector<PossiblePlate> vectorOfPossiblePlates;

    //Median filter better resuls
    //performGaussFilter(Y_buff, xSize, ySize, 5, 0.001);
    performMedianFilter(Y_buff, xSize,ySize,5);
    
    //Threshold
    threshold(Y_buff, xSize, ySize);


    //SOBEL DETECT EDGE
    detectEdgeSobel(Y_buff, xSize, ySize);

    //CV image and cv gray image
    procesing_YUV420(Y_buff, U_buff, V_buff, xSize, ySize, 1, 0, 0);
    YUV420toRGB(Y_buff, U_buff, V_buff, xSize, ySize, QImg.bits());

    cv::Mat CVImg(QImg.height(), QImg.width(), CV_8UC3, (uchar*)QImg.bits(), QImg.bytesPerLine());
    cvtColor(CVImg, CVImgGray, CV_BGR2GRAY);
    cv::resize(CVImgGray, CVImgGray, cv::Size(1024, 640));


    cv::Mat CVOrgImg(inImgs.height(), inImgs.width(), CV_8UC3, (uchar*)inImgs.bits(), inImgs.bytesPerLine());
    cvtColor(CVOrgImg, result, CV_BGR2RGB);
    cv::resize(result, result, cv::Size(1024, 640));

    //Find contours on CV gray image
    cv::findContours(CVImgGray, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

    //check contours if possible char
    for (unsigned int i = 0; i < contours.size(); i++) {

        PossibleChar possibleChar(contours[i]);

        if (checkIfPossibleChar(possibleChar)) {
            intCountOfPossibleChars++;
            vectorOfPossibleChars.push_back(possibleChar);
        }
    }

    
    std::vector<std::vector<PossibleChar> > vectorOfVectorsOfMatchingCharsInScene = findVectorOfVectorsOfMatchingChars(vectorOfPossibleChars);

    //extract plate from cv image
    for (auto& vectorOfMatchingChars : vectorOfVectorsOfMatchingCharsInScene) {
        PossiblePlate possiblePlate = extractPlate(result, vectorOfMatchingChars); 

        if (possiblePlate.imgPlate.empty() == false) {
            vectorOfPossiblePlates.push_back(possiblePlate);
        }
    }

    vectorOfPossiblePlates = detectPlateInPlates(vectorOfPossiblePlates);

    if (vectorOfPossiblePlates.empty()) {

        QPainter p(&copyImage);
        p.setPen(QPen(Qt::red));
        p.setFont(QFont("Times", 50, QFont::Bold));
        p.drawText(copyImage.rect(), Qt::AlignCenter, "No license plates were detected");

        RGBtoYUV420(copyImage.bits(), xSize, ySize, Y_buff, U_buff, V_buff);
    }
    else {
        sort(vectorOfPossiblePlates.begin(), vectorOfPossiblePlates.end(), PossiblePlate::sortDescendingByNumberOfChars); 

        PossiblePlate licPlate = vectorOfPossiblePlates.front();
        string text;

        //filterPlateImage(licPlate);

        cv::imshow("imgPlate", licPlate.imgPlate);
        cv::imshow("imgPlateT", licPlate.imgThresh);
        cv::waitKey(0);

        drawRedRectangleAroundPlate(copyImage,licPlate);

        cv::imwrite("out.jpg", licPlate.imgThresh);
        
        RGBtoYUV420(copyImage.bits(), xSize, ySize, Y_buff, U_buff, V_buff);

    }


}

void threshold(uchar Y_buff[], int xSize, int ySize) {
    int i;
    for (i = 0; i < (xSize * ySize); i++)
    {
        if (Y_buff[i] > MIN_THRESHOLD_VALUE)
        {
            Y_buff[i] = 0;
        }
        else
        {
            Y_buff[i] = 255;
        }
    }
}


void performMedianFilter(uchar input[], int xSize, int ySize, int N)
{
    std::vector<uchar> temp(N * N);
    uchar* input_copy = new uchar[xSize * ySize];
    for (int i = 0; i < xSize * ySize; i++)
    {
        input_copy[i] = input[i];
    }


    for (int i = 0; i < xSize; i++)
    {
        for (int j = 0; j < ySize; j++)
        {
            int k = 0;
            for (int m = i - N / 2; m <= i + N / 2; m++)
            {
                for (int n = j - N / 2; n <= j + N / 2; n++)
                {
                    if (m >= 0 && m < xSize && n >= 0 && n < ySize)
                    {
                        temp[k] = input_copy[m + n * xSize];
                    }
                    else
                    {
                        temp[k] = 0;
                    }
                    k++;
                }

                std::sort(temp.begin(), temp.end());
                input[i + j * xSize] = temp[N * N / 2];
            }

        }
    }
    delete[] input_copy;

}


void detectEdgeSobel(uchar Y_buff_H[], int xSize, int ySize)
{
    uchar* Y_buff_V = new uchar[xSize * ySize];

    uchar* img = new uchar[xSize * ySize];

    // Create a copy of Y buffer for second convolution.
    memcpy(Y_buff_V, Y_buff_H, xSize * ySize);

    // Horizontal sobel kernel.
    double sobel_h[9] = {
         0.25,  0.5,  0.25,
            0,    0,     0,
        -0.25, -0.5, -0.25
    };

    // Vertical sobel kernel.
    double sobel_v[9] = {
        -0.25,    0, 0.25,
         -0.5,    0,  0.5,
        -0.25,    0, 0.25
    };

    /* Filter both images with corresponding Sobel operator */
    convolve2D(Y_buff_H, xSize, ySize, sobel_h, 3);
    convolve2D(Y_buff_V, xSize, ySize, sobel_v, 3);

    int max = -200;
    int min = 2000;

    for (int i = 0; i < xSize * ySize; i++) {
        img[i] = sqrt(pow(Y_buff_H[i], 2) + pow(Y_buff_V[i], 2));
        if (img[i] > max) max = img[i];
        if (img[i] < min) min = img[i];
    }

    int diff = max - min;

    for (int i = 0; i < xSize * ySize; i++) {
        float abc = (img[i] - min) / (diff * 1.0);
        img[i] = abc * 255;
    }


    for (int i = 0; i < xSize * ySize; i++) {
        Y_buff_H[i] = img[i];
    }


    delete[] Y_buff_V;
}

void drawRedRectangleAroundPlate(QImage& copyImg, PossiblePlate& licPlate) {
    double widthParam = (double)copyImg.width() / 1024;
    double heightParam = (double)copyImg.height() / 640;

    QPainter p(&copyImg);
    p.setPen(Qt::red);
    p.setFont(QFont("Times", 30, QFont::Bold));
    QSize size = copyImg.size();
    if (licPlate.angleInDeg > 1.3) {
        p.translate(size.height() / 2, size.height() / 2);
        p.rotate(licPlate.angleInDeg);
        p.translate(size.height() / -2, size.height() / -2);

        p.drawRect(licPlate.dblPlateCenterX * widthParam, licPlate.dblPlateCenterY * heightParam, licPlate.intPlateWidth * widthParam * PLATE_WIDTH_PADDING_FACTOR, licPlate.intPlateHeight * heightParam * PLATE_HEIGHT_PADDING_FACTOR);
    }
    else {
        p.drawRect((licPlate.dblPlateCenterX - 30) * widthParam, (licPlate.dblPlateCenterY - 30) * heightParam, licPlate.intPlateWidth * widthParam * PLATE_WIDTH_PADDING_FACTOR, licPlate.intPlateHeight * heightParam * PLATE_HEIGHT_PADDING_FACTOR);
    }
}

void writeLicensePlateCharsOnImage(QImage& copyImg, PossiblePlate& licPlate) 
{
    QPainter p(&copyImg);
    p.setPen(Qt::red);
    p.setFont(QFont("Times", 30, QFont::Bold));
    QString str = QString::fromUtf8(licPlate.strChars.c_str());
    p.drawText(licPlate.dblPlateCenterX,licPlate.dblPlateCenterY - 50, str );
    //p.drawText(copyImg.rect(), Qt::AlignCenter,str);

}


void calculateGaussKernel(double kernel[], int N, double sigma)
{
	double C = 0;
	for(int n = 0; n < N; n++)
    {
        for(int k = 0; k < N; k++)
        {
            kernel[n*N+k] = exp( -((n - N/2)*(n - N/2) + (k - N/2)*(k - N/2)) / (2 * sigma * sigma));
			C += kernel[n*N+k];
		}
	}

	C = 1.0 / C;

	for(int n = 0; n < N; n++)
    {
        for(int k = 0; k < N; k++)
        {
            kernel[n*N+k] *= C;
		}
	}
}

void performGaussFilter(uchar input[], int xSize, int ySize, int N, double sigma)
{
    double* coeffs = new double[N * N];
    calculateGaussKernel(coeffs, N, sigma);

    convolve2D(input, xSize, ySize, coeffs, N);
}


