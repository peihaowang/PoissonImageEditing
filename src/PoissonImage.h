
#ifndef POISSON_IMAGE_H
#define POISSON_IMAGE_H

#include <opencv2/opencv.hpp>

class PoissonImage
{

public:

    // Differential Operators
    enum DiffOp {
        Forward
        , Backward
        , Centered
        , Fourier   // No implementation
    };

    // Gradient Strategy
    enum GradientScheme {
        Replace
        , Average
        , Maximum
    };

    // Performance Metric
    struct PerfMetric {
        double  m_tInit;
        double  m_tGradient;
        double  m_tSolver;

        PerfMetric& operator=(const PerfMetric& rhs)
        {
            m_tInit = rhs.m_tInit;
            m_tGradient = rhs.m_tGradient;
            m_tSolver = rhs.m_tSolver;
            return (*this);
        }
    };

public:

    static bool seamlessClone(cv::InputArray src, cv::InputArray dst, cv::InputArray mask
        , const cv::Point& offset, cv::OutputArray output, PerfMetric* perfMetric = NULL
        , GradientScheme gradientSchm = GradientScheme::Maximum
        , DiffOp gradientOp = DiffOp::Backward
        , DiffOp divOp = DiffOp::Forward
    );

};

#endif // POISSON_IMAGE_H
