
#include <opencv2/photo.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

#include <Eigen/Core>
#include <Eigen/Sparse>


class StopWatch
{
private:
    clock_t         m_startTime;
public:
    StopWatch() : m_startTime(std::clock()) { return; }
    void tick(const char* title)
    {
        std::cout << title << ": " << (double)(std::clock() - m_startTime) / CLOCKS_PER_SEC << "s" << std::endl;
        m_startTime = std::clock();
    }
};


///////////////////////////////////////////////////////////////////////////////

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

    cv::Mat makeContinuous(const cv::Mat& m) const;

    template<typename T, int RowNum, int ColNum>
    void cvMat2EigenMat(const cv::Mat& cvMat, Eigen::Matrix<T, RowNum, ColNum>& eigenMat);
    template<typename T, int RowNum, int ColNum>
    void eigenMat2CvMat(const Eigen::Matrix<T, RowNum, ColNum>& eigenMat, cv::Mat& cvMat);

    inline int _idx(int x, int y) const { return x * m_height + y; }
    inline bool isOnEdge(int x, int y) const;

    void laplacianOperator(Eigen::SparseMatrix<float, Eigen::RowMajor>& L) const;
    void projectionMask(Eigen::SparseMatrix<float, Eigen::RowMajor>& M) const;
    void projectionEdge(Eigen::SparseMatrix<float, Eigen::RowMajor>& E) const;
    void projectionSimpler(Eigen::SparseMatrix<float, Eigen::RowMajor>& S) const;
    void diffOperator(Eigen::SparseMatrix<float, Eigen::RowMajor>& Dx, Eigen::SparseMatrix<float, Eigen::RowMajor>& Dy, DiffOp op) const;

    void poissonSolver(Eigen::MatrixXf& R, bool wholeSpace = false) const;

public:

    PoissonImage(GradientScheme gradientSchm = GradientScheme::Maximum, DiffOp gradientOp = DiffOp::Backward, DiffOp divOp = DiffOp::Forward);
    virtual ~PoissonImage() { return; }

    void seamlessClone(cv::InputArray src, cv::InputArray dst, cv::InputArray mask, const cv::Point& offset, cv::OutputArray output);

};
