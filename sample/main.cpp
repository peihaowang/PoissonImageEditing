#include <iostream>

#include "PoissonImage.h"

int main(int argc, char* argv[])
{
    if(argc < 4){
        std::cout << "Compulsory arguments are missing ..." << std::endl;
        std::cout << "Usage: PoissonImageEditor <source-path> <destination-path> <output-path> \n"
                  << "[ <mask-path> [ <x-offset> <y-offset> [ <-r|-a|-m> [ <-b|-f|-c> <-b|-f|-c> ] ] ] ]"
                  << std::endl
                  ;
        return 0;
    }

    std::cout << "Poisson Image Editing ..." << std::endl;

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
    PoissonImage::PerfMetric perf;
    if(PoissonImage::seamlessClone(src, dst, mask, offset, output, scheme, gradientOp, divOp, &perf)){
        cv::imwrite(outputPath, output);

        std::cout << "Initialization: " << perf.m_tInit << "s" << std::endl;
        std::cout << "Calculate Gradient: " << perf.m_tGradient << "s" << std::endl;
        std::cout << "Poisson Solving: " << perf.m_tSolver << "s" << std::endl;
    }else{
        std::cerr << "Failed to run Poisson image editing." << std::endl;
    }

    return 0;
}
