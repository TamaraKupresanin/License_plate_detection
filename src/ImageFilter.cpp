#include "ImageFilter.h"
#include <cmath>


void convolve2D(uchar image[], int xSize, int ySize, double* filterCoeff, int N)
{
	int newXSize = xSize + (N - 1);
	int newYSize = ySize + (N - 1);

	uchar* extendedImage = new uchar[newXSize*newYSize];

	extendBorders(image, xSize, ySize, extendedImage, (N - 1) / 2);

	for (int i = 0; i < xSize; i++)
	{
		for (int j = 0; j < ySize; j++)
		{
			double accum = 0;
			for (int m = 0; m < N; m++)
			{
				for (int n = 0; n < N; n++)
				{

					accum += extendedImage[(j + n) * newXSize + i + m] * filterCoeff[(N - n) * N - m - 1];
				}
			}
			if (accum > 255.0)
				image[j * xSize + i] = 255;
			else if (accum < 0.0)
				image[j * xSize + i] = 0;
			else
				image[j * xSize + i] = floor(accum + 0.5);

		}
	}

	delete[] extendedImage;
}

void extendBorders(uchar input[], int xSize, int ySize, uchar output[], int delta)
{
	int newXSize = xSize + 2 * delta;
	int newYSize = ySize + 2 * delta;

	for (int i = 0; i < xSize; i++)
	{
		for (int j = 0; j < ySize; j++)
		{
			output[(j + delta) * newXSize + i + delta] = input[j * xSize + i];
		}
	}

	for (int i = 0; i < delta; i++)
	{
		memcpy(&output[i * newXSize + delta], &input[0], xSize);
		memcpy(&output[(ySize + delta + i)*newXSize + delta], &input[(ySize - 1) * xSize], xSize);
	}

	for (int i = 0; i < newYSize; i++)
	{
		memset(&output[i * newXSize], output[i * newXSize + delta], delta);
		memset(&output[i * newXSize + delta + xSize], output[i * newXSize + xSize + delta - 1], delta);
	}

}


void performNFFilter(uchar input[], int xSize, int ySize)
{
	double* coeff = new double[9];
	for (int i = 0; i < 9; i++)
	{
		coeff[i] = 1.0 / 9.0;
	}

	convolve2D(input, xSize, ySize, coeff, 3);

	delete[] coeff;
}

void performVFFilter(uchar input[], int xSize, int ySize)
{
	double coeff[] = { 0, -1, 0, -1, 4, -1, 0, -1, 0 };

	convolve2D(input, xSize, ySize, coeff, 3);
}

void performSuccessiveNFFilter(uchar input[], int xSize, int ySize, int stages)
{
	double* coeff = new double[9];
	for (int i = 0; i < 9; i++)
	{
		coeff[i] = 1.0 / 9.0;
	}

	for (int i = 0; i < stages; i++)
	{
		convolve2D(input, xSize, ySize, coeff, 3);
	}
	delete[] coeff;
}

void performSobelEdgeDetection(uchar input[], int xSize, int ySize, double threshold)
{
	int N = 3;
	double hCoeff[] = { -0.25, -0.5, -0.25, 0, 0, 0, 0.25, 0.5, 0.25 };
	double vCoeff[] = { -0.25, 0, 0.25, -0.5, 0, 0.5, -0.25, 0, 0.25 };

	int newXSize = xSize + 2;
	int newYSize = ySize + 2;

	uchar* extendedImage = new uchar[newXSize*newYSize];

	extendBorders(input, xSize, ySize, extendedImage, 1);

	for (int i = 0; i < xSize; i++)
	{
		for (int j = 0; j < ySize; j++)
		{
			double hAccum = 0, vAccum = 0;
			for (int m = 0; m < N; m++)
			{
				for (int n = 0; n < N; n++)
				{
					hAccum += extendedImage[(j + n) * newXSize + i + m] * hCoeff[(N - n) * N - m - 1];
					vAccum += extendedImage[(j + n) * newXSize + i + m] * vCoeff[(N - n) * N - m - 1];
				}
			}

			input[j * xSize + i] = (sqrt(hAccum * hAccum + vAccum * vAccum) > threshold) * 255;

		}
	}
	delete[] extendedImage;

}

void performNFplusSobelEdgeDetection(uchar input[], int xSize, int ySize, int stages, double threshold)
{
	performSuccessiveNFFilter(input, xSize, ySize, stages);
	performSobelEdgeDetection(input, xSize, ySize, threshold);

}
