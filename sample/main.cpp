
#include <iostream>

#include "PoissonImage.h"

int main(int argc, char* argv[])
{
    std::cout << "Poisson Image Editing ..." << std::endl;

    if(argc < 4){
        std::cout << "Compulsory arguments are missing ..." << std::endl;
        std::cout << "Usage: PoissonImageEditor <source-path> <destination-path> <output-path> \n"
                  << "[ <mask-path> [ <x-offset> <y-offset> [ <-r|-a|-m> [ <-b|-f|-c> <-b|-f|-c> ] ] ] ]"
                  << std::endl
                  ;
        return 0;
    }

    cv::Mat src = cv::imread(argv[1]);
    cv::Mat dst = cv::imread(argv[2]);
    std::string outputPath = argv[3];

    cv::Mat mask;
    if(argc >= 5) mask = cv::imread(argv[4], cv::IMREAD_GRAYSCALE);
    else mask = 255 * cv::Mat::ones(src.rows, src.cols, CV_8UC1);

    cv::Point offset(0, 0);
    if(argc >= 7){
        offset = cv::Point(std::atoi(argv[5]), std::atoi(argv[6]));
    }

    PoissonImage::GradientScheme scheme = PoissonImage::Maximum;
    if(argc >= 8){
        std::string a = argv[7];
        if(a == "-r") scheme = PoissonImage::Replace;
        else if(a == "-a") scheme = PoissonImage::Average;
        else if(a == "-m") scheme = PoissonImage::Maximum;
    }

    PoissonImage::DiffOp gradientOp = PoissonImage::Backward, divOp = PoissonImage::Forward;
    if(argc >= 10){
        std::string a = argv[8], b = argv[9];
        if(a == "-f") gradientOp = PoissonImage::Forward;
        else if(a == "-b") gradientOp = PoissonImage::Backward;
        else if(a == "-c") gradientOp = PoissonImage::Centered;

        if(b == "-f") divOp = PoissonImage::Forward;
        else if(b == "-b") divOp = PoissonImage::Backward;
        else if(b == "-c") divOp = PoissonImage::Centered;
    }

    cv::Mat output;
    PoissonImage PI(scheme, gradientOp, divOp);
    PI.seamlessClone(src, dst, mask, offset, output);
    cv::imwrite(outputPath, output);

    return 0;
}
