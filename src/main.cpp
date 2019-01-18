
#include <iostream>

#include "PoissonImage.h"

int main(int argc, char* argv[])
{
    std::cout << "Poisson Image Editing ..." << std::endl;

//    cv::Mat src = cv::imread("tests/iloveyouticket.png");
    cv::Mat src = cv::imread("tests/src_1.jpg");
    cv::Mat dst = cv::imread("tests/wood.png");
//    cv::Mat dst = cv::imread("tests/dst.jpg");
//    cv::Mat mask = 255 * cv::Mat::ones(src.rows, src.cols, CV_8UC1);
    cv::Mat mask = cv::imread("tests/mask_1.jpg", cv::IMREAD_GRAYSCALE);
//    cv::Mat mask = cv::imread("tests/iloveyouticket_m.png", cv::IMREAD_GRAYSCALE);
    cv::Mat output;
    PoissonImage PI(PoissonImage::Maximum); PI.seamlessClone(src, dst, mask, cv::Point(0, 0), output);
    cv::imwrite("tests/result.png", output);

    return 0;
}
