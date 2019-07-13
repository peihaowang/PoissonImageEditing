#include <iostream>
#include "StopWatch.h"
#include "PoissonImageImpl.h"
#include "PoissonImage.h"

bool PoissonImage::seamlessClone(cv::InputArray src, cv::InputArray dst, cv::InputArray mask, const cv::Point& offset
    , cv::OutputArray output, GradientScheme gradientSchm, DiffOp gradientOp, DiffOp divOp, PerfMetric* perfMetric
) {
    if (src.size() != mask.size())
    {
        std::cerr << "The sizes of source and mask should be the same!" << std::endl;
        return false;
    }

    // Size
    int w = dst.cols();
    int h = dst.rows();

    // Format check
    if (src.type() != CV_8UC3)
    {
        std::cerr << "Format Check: The source mat have type: " << src.type() << ", but should have type: " << CV_8UC3 << std::endl;
        return false;
    }
    if (dst.type() != CV_8UC3)
    {
        std::cerr << "Format Check: The destination mat have type: " << dst.type() << ", but should have type: " << CV_8UC3 << std::endl;
        return false;
    }
    if (mask.type() != CV_8UC1)
    {
        std::cerr << "Format Check: The mask mat have type: " << mask.type() << ", but should have type: " << CV_8UC1 << std::endl;
        return false;
    }

    // Adjust the position of the src image and mask
    cv::Mat srcMat = cv::Mat::zeros(h, w, CV_8UC3);
    cv::Mat maskMat = cv::Mat::zeros(h, w, CV_8UC1);
    cv::Mat dstMat = dst.getMat();
    {
        int left = (src.cols() / 2) - offset.x - (w / 2);
        if (left < 0)
            left = 0;
        int right = (src.cols() / 2) - offset.x + (w / 2);
        if (right >= src.cols())
            right = src.cols() - 1;
        int top = (src.rows() / 2) - offset.y - (h / 2);
        if (top < 0)
            top = 0;
        int bottom = (src.rows() / 2) - offset.y + (h / 2);
        if (bottom >= src.rows())
            bottom = src.rows() - 1;
        cv::Rect rcSrc(left, top, right - left, bottom - top);
        cv::Point center = cv::Point(w / 2, h / 2) + offset + cv::Point((left + right) / 2, (top + bottom) / 2) - cv::Point(src.cols() / 2, src.rows() / 2);
        cv::Rect rcDst(center.x - rcSrc.width / 2, center.y - rcSrc.height / 2, rcSrc.width, rcSrc.height);
        src.getMat()(rcSrc).copyTo(srcMat(rcDst));
        mask.getMat()(rcSrc).copyTo(maskMat(rcDst));
    }

    // Make input data compact
    srcMat = PoissonImageImpl::makeContinuous(srcMat);
    dstMat = PoissonImageImpl::makeContinuous(dstMat);
    maskMat = PoissonImageImpl::makeContinuous(maskMat);

    PoissonImageImpl PI(gradientSchm, gradientOp, divOp);
    cv::Mat outputMat;
    if(!PI.seamlessClone(srcMat, dstMat, maskMat, outputMat)){
        return false;
    }

    // 2019.7.13 Bugfix to float-pointing matrix. Thanks to ZHAO
    // Since the cv::imshow only present normalized float matrix
    // normally, we need to return the matrix of byte type.
    outputMat.convertTo(output, CV_8U);
    
    // Fill out the perfermance metric
    if (perfMetric){
        (*perfMetric) = PI.m_perfMetric;
    }

    return true;
}
