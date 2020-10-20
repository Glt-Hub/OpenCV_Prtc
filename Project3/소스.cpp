#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>  
#include <sstream>

using namespace cv;
using namespace std;

Mat histCalc(const Mat& src);
Mat histToGraph(const Mat& hist);
void noise_gaussian();
void filter_bilateral();
void filter_median();
void edge_detection();
void edge_detection_kenny();
void edge_detection_kenny_manual();
void edge_detection_manual();
void hough_line();
void hough_line_possibility();
void hough_circle();
void erode_dilate();
void labeling();
void labelwithstats();

float testData[] = { 50, 20, 10, 1, 255, 255
					,20, 0, 20, 1, 1, 1
					,1, 20, 1, 1, 1, 1
					,0, 0, 0, 0, 0, 0
					,1, 1, 1, 1, 1, 1
					,0, 0, 0, 0, 0, 0 };

Mat testMat(6, 6, CV_32FC1, testData);

void edge_detection_manual()
{
	Mat src = testMat;
	Mat gaussian;
	Mat mag;
	Mat sobel;
	Mat dx, dy;
	GaussianBlur(src, gaussian, Size(), (double)1.0);
	Sobel(src, dx, CV_32FC1, 1, 0);
	Sobel(src, dy, CV_32FC1, 0, 1);
	cout << gaussian << endl;
	magnitude(dx, dy, mag);
	mag.convertTo(sobel, CV_8UC1);
	cout << sobel << endl;
}

int main(int argc, char** argv)
{
	/*
	Mat lenna = imread("image.png", IMREAD_GRAYSCALE);
	Mat copyImg = lenna.clone();

	for (int i = 0; i < copyImg.rows; i++)
	{
		for (int j = 0; j < copyImg.cols; j++)
		{
			copyImg.at<uchar>(i, j) = saturate_cast<uchar>(copyImg.at<uchar>(i,j) + 100);
			// saturate contrl : 255를 over하거나 0 이하로 내려가면 보정함. (0~255 사이의 값만 수용하고, 넘치거나 부족하면 0 or 255처리함)
		}
	}

	namedWindow("original");
	namedWindow("Processed");

	imshow("original", lenna);
	imshow("Processed", copyImg);

	Bright Control
	*/

	/*
	Mat src = imread("image.png", IMREAD_GRAYSCALE);

	float s = 2.f;
	Mat dst = s * src;
	// 명암비 조절
	// 명암 : 밝은 영역 ~ 어두운 영역 사이 드러나느 밝기 차이의 강도
	// 전체적으로 어둡거나 밝다 ; 명암비가 낮다

	// 전체 픽셀 * 적절한 실수 -> 곱셈 수식 적용에 따라 영상품질 영향 끼침!

	// 위 방법은 픽셀값이 포화되기 때문에 실제로 사용되는 방법이 아님.
	*/

	/*
	Mat src = imread("image.png", IMREAD_GRAYSCALE);
	Mat dst;

	double total = 0.0;

	for (int i = 0; i < src.rows;i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			total += src.at<uchar>(i, j);
		}
	}
	double avg = total / (src.rows * src.cols);
	cout << avg << endl;

	float alpha = 1.f;
	uchar threshold = (uchar)avg;

	dst = src + (src - threshold) * alpha;

	// 명암비를 증가시키는 효과적인 공식
	// alpha : 명암비 그래프의 기울기를 결정
	*/

	/*
	Mat src = imread("image.png", IMREAD_GRAYSCALE);
	Mat dst;
	Mat hist = histCalc(src);

	double gmin, gmax;
	minMaxLoc(src, &gmin, &gmax);
	dst = (src - gmin) / (gmax - gmin) * 255;
	// Histogram Stretch 공식
	// 명암비가 올라간당
	Mat dsthist = histCalc(dst);

	imshow("src", src);
	imshow("srchist", histToGraph(hist));
	imshow("dst", dst);
	imshow("dsthist", histToGraph(dsthist));
	*/
	Mat src = imread("image.png", IMREAD_GRAYSCALE);
	/*
	float embossdata[] = {  -1,-1, 0
					, -1, 0, 1
					,  0, 1, 1 };

	Mat emboss(3, 3, CV_32FC1, embossdata);
	// 영상 엠보싱.. 대각선 방향으로 픽셀값 급격히 변화시
	// 에지부분을 강조




	Mat dst;

	filter2D(src, dst, -1, emboss, Point(-1, -1), 128);
	*/
	Mat dst;
	Mat manualDst;
	Mat testDst;
	int blurSize = 7;
	double sigma = 1.0;
	
	/*
	float blurKernel = 1.0 / 9.0;
	float blurData[] = {blurKernel,blurKernel ,blurKernel, 
						blurKernel,blurKernel,blurKernel,
						blurKernel, blurKernel, blurKernel};
	Mat blurMat(blurSize, blurSize, CV_32FC1, blurData);

	filter2D(src, dst, -1, blurMat);
	filter2D(testMat, testDst, -1, blurMat);
	// 수동 커널 적용하기
	*/
	/*
	blur(src, dst, Size(blurSize, blurSize));
	blur(testMat, testDst, Size(blurSize, blurSize));
	// 자동 커널 적용하기
	*/
	// 이상 평균값 필터 Blurring

	/*
	float gaussianKernel[] = {0.0001, 0.0044, 0.0540, 0.2420, 0.3989, 0.2420, 0.0540, 0.0044, 0.0001};
	Mat gaussianHorizontal(1, 9, CV_32FC1, gaussianKernel);
	Mat gaussianVertical(9, 1, CV_32FC1, gaussianKernel);
	filter2D(src, manualDst, CV_8UC1, gaussianHorizontal);
	filter2D(manualDst, manualDst, CV_8UC1, gaussianVertical);
	// Sigma(표준편차) 1짜리 수동 가우시안

	GaussianBlur(src, dst, Size(), (double)sigma);


	Mat hist = histCalc(dst);
	Mat srcHist = histCalc(src);
	*/
	// noise_gaussian();
	// filter_bilateral();
	// filter_median();
	// edge_detection();
	// edge_detection_kenny_manual();
	// edge_detection_manual();
	// hough_line_possibility();
	// hough_circle();
	// erode_dilate();
	// labeling();
	labelwithstats();
	waitKey(0);
}

Mat histCalc(const Mat& src)
{
	Mat hist;
	
	int channels[] = { 0 };
	int dims = 1;
	const int histSize[] = { 256 };
	float graylevel[] = { 0,256 };
	const float* ranges[] = { graylevel };
	calcHist(&src, 1, channels, noArray(), hist, dims, histSize, ranges);
	return hist;
}

Mat histToGraph(const Mat& hist)
{
	double histMax;
	minMaxLoc(hist, 0, &histMax);
	Mat imgHist(100, 256, CV_8UC1, Scalar(255));
	for (int i = 0; i < 256; i++)
	{
		line(imgHist, Point(i, 100), Point(i, 100 - cvRound(hist.at<float>(i, 0) * 100 / histMax)), Scalar(0));
	}
	return imgHist;
}

// 가우시안 잡음 추가 함수
void noise_gaussian()
{
	Mat src = imread("image.png", IMREAD_GRAYSCALE);

	imshow("src", src);

	for (int i = 10; i <= 30; i += 10)
	{
		// 표준편차 10짜리 랜덤 Data
		Mat noise(src.size(), CV_32SC1);
		randn(noise, 0, i);
		
		// 합영상을 통해 Source에 잡음 추가
		Mat dst;
		add(src, noise, dst, Mat(), CV_8U);
		

		imshow("dst", dst);
		waitKey(0);
	}
}

void filter_bilateral()
{
	double duration;

	Mat src = imread("image.png", IMREAD_GRAYSCALE);
	Mat noise(src.size(), CV_32SC1);
	randn(noise, 0, 10);

	Mat dst;
	add(src, noise, dst, Mat(), CV_8U);

	duration = static_cast<double>(getTickCount());
	Mat filterRes;
	bilateralFilter(dst, filterRes, -1, 10, 5);
	duration = static_cast<double>(getTickCount() - duration);
	duration /= getTickFrequency();
	cout << "Artificial noise" << duration << " ms" << endl;

	duration = static_cast<double>(getTickCount());
	Mat oriFilter;
	bilateralFilter(src, oriFilter, -1, 10, 5);
	duration = static_cast<double>(getTickCount() - duration);
	duration /= getTickFrequency();
	cout << "natural noise" << duration << " ms" << endl;

	imshow("src", src);
	imshow("dst", dst);
	imshow("oriFilter", oriFilter);
	imshow("filterRes", filterRes);
}

void filter_median()
{
	Mat src = imread("image.png", IMREAD_GRAYSCALE);
	/*
	int num = (int)(src.total() * 0.1);
	for (int i = 0; i < num; i++)
	{
		int x = rand() % src.cols;
		int y = rand() % src.rows;
		src.at<uchar>(y, x) = (i % 2) * 255;
	}
	*/
	
	Mat medfilterMat(3, 3, CV_32FC1);

	Mat medBlur;
	medianBlur(src, medBlur, 3);

	imshow("src", src);
	imshow("medBlur", medBlur);
}

void edge_detection()
{
	Mat src = imread("image.png", IMREAD_GRAYSCALE);
	Mat dx, dy;

	// x,y 방향 소벨커널 연산 (3x3)
	Sobel(src, dx, CV_32FC1, 1, 0);
	Sobel(src, dy, CV_32FC1, 0, 1);
	
	Mat fmag, mag;
	// dx, dy는 CV_32F또는 CV_64F
	magnitude(dx, dy, fmag);
	fmag.convertTo(mag, CV_8UC1);

	Mat edge = mag > 100;

	imshow("src", src);
	imshow("mag", mag);
	imshow("edge", edge);
}

void edge_detection_kenny()
{
	Mat src = imread("image.png", IMREAD_GRAYSCALE);

	Mat dst1, dst2;
	Canny(src, dst1, 50, 100);
	Canny(src, dst2, 50, 150);

	imshow("src", src);
	imshow("dst1", dst1);
	imshow("dst2", dst2);
}

void edge_detection_kenny_manual()
{
	Mat src = imread("3030rand.png", IMREAD_GRAYSCALE);
	Mat gaussian;
	Mat mag;
	Mat sobel;
	Mat dx, dy;
	GaussianBlur(src, gaussian, Size(), (double)1.0);
	imshow("gaussian", gaussian);
	Sobel(src, dx, CV_32FC1, 1, 0);
	Sobel(src, dy, CV_32FC1, 0, 1);
	Mat dst;
	Canny(src, dst, 20, 100);

	magnitude(dx,dy,mag);
	mag.convertTo(sobel, CV_8UC1);
	cout << sobel << endl << endl;
	cout << dst << endl;

	imshow("sobel", sobel);
}

void hough_line()
{
	Mat src = imread("houghline.jpg", IMREAD_GRAYSCALE);
	Mat cannyDst;
	Canny(src, cannyDst, 50, 90);

	vector<Vec2f> lines;
	HoughLines(cannyDst, lines, 1, CV_PI / 180, 240);

	Mat dst;
	cvtColor(cannyDst, dst, COLOR_GRAY2BGR);

	for (size_t i = 0; i < lines.size(); i++)
	{
		float r = lines[i][0], t = lines[i][1];
		double cos_t = cos(t), sin_t = sin(t);
		double x0 = r * cos_t, y0 = r * sin_t;
		double alpha = 200;

		Point pt1(cvRound(x0 + alpha * (-sin_t)), cvRound(y0 + alpha * cos_t));
		Point pt2(cvRound(x0 - alpha * (-sin_t)), cvRound(y0 - alpha * cos_t));
		line(dst, pt1, pt2, Scalar(0, 0, 255), 2, LINE_AA);
	}

	imshow("src", src);
	imshow("dst", dst);

	waitKey();
	destroyAllWindows();
}

void hough_line_possibility()
{
	Mat src = imread("houghline.jpg", IMREAD_GRAYSCALE);
	Mat cannyDst;
	Canny(src, cannyDst, 50, 150);

	vector<Vec4i> lines;
	HoughLinesP(cannyDst, lines, 1, CV_PI / 180, 180, 100, 100);

	Mat dst;
	cvtColor(cannyDst, dst, COLOR_GRAY2BGR);

	for (Vec4i l : lines)
	{
		line(dst, Point(l[0], l[1]), Point(l[2], l[3]),Scalar(0,0,255), 2, LINE_AA);
	}
	imshow("src", src);
	imshow("dst", dst);
}

void hough_circle()
{
	Mat src = imread("coin.jpg", IMREAD_GRAYSCALE);

	Mat blurred;
	blur(src, blurred, Size(3, 3));
	int tot = 0;

	for (int i = 0; i < src.size().width; i++)
	{
		for (int j = 0; j < src.size().height; j++)
		{
			tot += src.at<uchar>(j, i);
		}
	}

	int avg = tot / (src.size().width * src.size().height);
	cout << avg << endl;

	vector<Vec3f> circles;
	HoughCircles(blurred, circles, HOUGH_GRADIENT, 1,100, avg, 10);

	Mat dst;
	cvtColor(src, dst, COLOR_GRAY2BGR);

	for (Vec3f c : circles)
	{
		Point center(cvRound(c[0]), cvRound(c[1]));
		int radius = cvRound(c[2]);
		circle(dst, center, radius, Scalar(0, 0, 255), 2, LINE_AA);
	}
	imshow("src", src);
	imshow("dst", dst);
	waitKey(0);
}

void erode_dilate()
{ 
	Mat src = imread("image.png", IMREAD_GRAYSCALE);

	Mat bin;
	threshold(src, bin, 130, 255, THRESH_BINARY);

	Mat erodeMat, dilateMat;
	
	erode(bin, erodeMat, Mat());
	dilate(bin, dilateMat, Mat());

	imshow("src", src);
	imshow("Binary", bin);
	imshow("erode", erodeMat);
	imshow("dilate", dilateMat);
}

void labeling()
{
	uchar data[] = {
		0,0,1,1,0,0,0,0,
		1,1,1,1,0,0,1,0,
		1,1,1,1,0,0,0,0,
		0,0,0,0,0,1,1,0,
		0,0,0,1,1,1,1,0,
		0,0,0,1,0,0,1,0,
		0,0,1,1,1,1,1,0,
		0,0,0,0,0,0,0,0,
	};

	Mat src = Mat(8, 8, CV_8UC1, data) * 255;

	Mat labels;

	int cnt = connectedComponents(src, labels);

	cout << "src\n" << src << endl;
	cout << "labels\n" << labels << endl;
	cout << "cnt\n" << cnt << endl;
}

void labelwithstats()
{
	Mat src = imread("Keyboard.jpg", IMREAD_GRAYSCALE);

	Mat bin;
	threshold(src, bin, 0, 255, THRESH_BINARY | THRESH_OTSU);

	Mat labels, stats, centroids;
	int cnt = connectedComponentsWithStats(bin, labels, stats, centroids);

	Mat dst;
	cvtColor(src, dst, COLOR_GRAY2BGR);

	for (int i = 1; i < cnt; i++)
	{
		int* p = stats.ptr<int>(i);

		if (p[4] < 20) continue;

		rectangle(dst, Rect(p[0], p[1], p[2], p[3]), Scalar(0, 255, 255), 2);
	}

	cout << centroids << endl;

	imshow("src", src);
	imshow("binary", bin);
	imshow("dst", dst);

	waitKey(0);
}