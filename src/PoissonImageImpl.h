
#ifndef POISSON_IMAGE_IMPL_H
#define POISSON_IMAGE_IMPL_H

#include <opencv2/opencv.hpp>

#include <Eigen/Core>
#include <Eigen/Sparse>

#include "PoissonImage.h"

class PoissonImageImpl
{

protected:

    // All channels of pixels are saved in column-major order
    Eigen::MatrixXf                 m_srcImage;
    Eigen::MatrixXf                 m_dstImage;
    Eigen::Matrix<unsigned char, Eigen::Dynamic, 1> m_maskMap;

    // Gradient field
    Eigen::MatrixXf                 m_gradientX;
    Eigen::MatrixXf                 m_gradientY;

    int                             m_width;
    int                             m_height;

    PoissonImage::GradientScheme    m_gradientScheme;

    PoissonImage::DiffOp            m_gradientOperator;
    PoissonImage::DiffOp            m_divOperator;

    PoissonImage::PerfMetric        m_perfMetric;

    friend class PoissonImage;

protected:

    static cv::Mat makeContinuous(const cv::Mat& m);

    template<typename T, int RowNum, int ColNum>
    void cvMat2EigenMat(const cv::Mat& cvMat, Eigen::Matrix<T, RowNum, ColNum>& eigenMat);
    template<typename T, int RowNum, int ColNum>
    void eigenMat2CvMat(const Eigen::Matrix<T, RowNum, ColNum>& eigenMat, cv::Mat& cvMat);

    inline int _idx(int x, int y) const { return x * m_height + y; }

    void laplacianOperator(Eigen::SparseMatrix<float, Eigen::RowMajor>& L) const;
    void projectionMask(Eigen::SparseMatrix<float, Eigen::RowMajor>& M) const;
    void projectionEdge(Eigen::SparseMatrix<float, Eigen::RowMajor>& E) const;
    void projectionSampler(Eigen::SparseMatrix<float, Eigen::RowMajor>& S) const;
    void diffOperator(Eigen::SparseMatrix<float, Eigen::RowMajor> &Dx, Eigen::SparseMatrix<float, Eigen::RowMajor> &Dy, PoissonImage::DiffOp op) const;

    void poissonSolver(Eigen::MatrixXf& R, bool wholeSpace = false) const;

    PoissonImageImpl(PoissonImage::GradientScheme gradientSchm = PoissonImage::Maximum, PoissonImage::DiffOp gradientOp = PoissonImage::Backward, PoissonImage::DiffOp divOp = PoissonImage::Forward);
    virtual ~PoissonImageImpl() { return; }

    bool seamlessClone(const cv::Mat& srcMat, const cv::Mat& dstMat, const cv::Mat& maskMat, cv::Mat& output);

};

#endif // POISSON_IMAGE_IMPL_H
