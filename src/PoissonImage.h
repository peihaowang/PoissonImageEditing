
#ifndef POISSON_IMAGE_H
#define POISSON_IMAGE_H

#include <opencv2/opencv.hpp>

#include <Eigen/Core>
#include <Eigen/Sparse>

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

protected:

    // All channels of pixels are saved in column-major order
    Eigen::MatrixXf             m_srcImage;
    Eigen::MatrixXf             m_dstImage;
    Eigen::Matrix<unsigned char, Eigen::Dynamic, 1> m_maskMap;

    // Gradient field
    Eigen::MatrixXf             m_gradientX;
    Eigen::MatrixXf             m_gradientY;

    int                     m_width;
    int                     m_height;

    GradientScheme          m_gradientScheme;

    DiffOp                  m_gradientOperator;
    DiffOp                  m_divOperator;

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
    void projectionSimpler(Eigen::SparseMatrix<float, Eigen::RowMajor>& S) const;
    void diffOperator(Eigen::SparseMatrix<float, Eigen::RowMajor>& Dx, Eigen::SparseMatrix<float, Eigen::RowMajor>& Dy, DiffOp op) const;

    void poissonSolver(Eigen::MatrixXf& R, bool wholeSpace = false) const;

    PoissonImage(GradientScheme gradientSchm = GradientScheme::Maximum, DiffOp gradientOp = DiffOp::Backward, DiffOp divOp = DiffOp::Forward);
    virtual ~PoissonImage() { return; }

    void seamlessClone(const cv::Mat& srcMat, const cv::Mat& dstMat, const cv::Mat& maskMat, cv::Mat& output);

public:

    static void seamlessClone(cv::InputArray src, cv::InputArray dst, cv::InputArray mask, const cv::Point& offset, cv::OutputArray output, GradientScheme gradientSchm = GradientScheme::Maximum, DiffOp gradientOp = DiffOp::Backward, DiffOp divOp = DiffOp::Forward);

};

#endif // POISSON_IMAGE_H
