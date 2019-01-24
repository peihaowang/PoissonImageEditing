
#include <time.h>
#include "StopWatch.h"
#include "PoissonImage.h"

PoissonImage::PoissonImage(GradientScheme gradientSchm, DiffOp gradientOp, DiffOp divOp)
    : m_gradientScheme(gradientSchm)
    , m_gradientOperator(gradientOp)
    , m_divOperator(divOp)
{
    return;
}

cv::Mat PoissonImage::makeContinuous(const cv::Mat& m)
{
    if (!m.isContinuous()) {
        return m.clone();
    }
    return m;
}

template<typename T, int RowNum, int ColNum>
void PoissonImage::cvMat2EigenMat(const cv::Mat& cvMat, Eigen::Matrix<T, RowNum, ColNum>& eigenMat)
{
    typedef Eigen::Matrix<unsigned char, Eigen::Dynamic, 1> V;
    for(int x = 0; x < m_width; x++){
        for(int y = 0; y < m_height; y++){
            eigenMat.row(_idx(x, y)) = Eigen::Map<const V>(cvMat.ptr<unsigned char>(y, x), cvMat.channels()).cast<T>();
        }
    }
}

template<typename T, int RowNum, int ColNum>
void PoissonImage::eigenMat2CvMat(const Eigen::Matrix<T, RowNum, ColNum>& eigenMat, cv::Mat& cvMat)
{
    cvMat = cv::Mat::zeros(m_height, m_width, CV_32FC3);
    typedef Eigen::Matrix<T, Eigen::Dynamic, 1> V;
    for(int x = 0; x < m_width; x++){
        for(int y = 0; y < m_height; y++){
            Eigen::Map<V>(cvMat.ptr<float>(y, x), cvMat.channels()) = eigenMat.row(_idx(x, y));
        }
    }
}

void PoissonImage::laplacianOperator(Eigen::SparseMatrix<float, Eigen::RowMajor>& L) const
{
    Eigen::SparseMatrix<float, Eigen::RowMajor> m(m_width * m_height, m_width * m_height);
    m.reserve(Eigen::VectorXi::Constant(m_width * m_height, 5));
    for(int x = 0; x < m_width; x++){
        for(int y = 0; y < m_height; y++){
            int coeff = 0;
            if(x >= 1){
                m.insert(_idx(x, y), _idx(x-1, y)) = 1;
                coeff--;
            }
            if(x <= m_width - 2){
                m.insert(_idx(x, y), _idx(x+1, y)) = 1;
                coeff--;
            }
            if(y >= 1){
                m.insert(_idx(x, y), _idx(x, y-1)) = 1;
                coeff--;
            }
            if(y <= m_height - 2){
                m.insert(_idx(x, y), _idx(x, y+1)) = 1;
                coeff--;
            }
            m.insert(_idx(x, y), _idx(x, y)) = coeff;
        }
    }
    m.makeCompressed();
    m.swap(L);
}

void PoissonImage::projectionMask(Eigen::SparseMatrix<float, Eigen::RowMajor>& M) const
{
    Eigen::SparseMatrix<float, Eigen::RowMajor> m(m_width * m_height, m_width * m_height);
    m.reserve(Eigen::VectorXi::Constant(m_width * m_height, 1));
    for(int x = 0; x < m_width; x++){
        for(int y = 0; y < m_height; y++){
            if(m_maskMap[_idx(x, y)] > 0){
                m.insert(_idx(x, y), _idx(x, y)) = 1;
            }
        }
    }
    m.makeCompressed();
    m.swap(M);
}

void PoissonImage::projectionEdge(Eigen::SparseMatrix<float, Eigen::RowMajor>& E) const
{
    Eigen::SparseMatrix<float, Eigen::RowMajor> m(m_width * m_height, m_width * m_height);
    m.reserve(Eigen::VectorXi::Constant(m_width * m_height, 1));
    for(int x = 0; x < m_width; x++){
        for(int y = 0; y < m_height; y++){
            if(m_maskMap[_idx(x, y)] > 0) continue;
            bool isEdge = (x >= 1 && m_maskMap[_idx(x-1, y)] > 0)
                || (x <= m_width - 2 && m_maskMap[_idx(x+1, y)] > 0)
                || (y >= 1 && m_maskMap[_idx(x, y-1)] > 0)
                || (y <= m_height - 2 && m_maskMap[_idx(x, y+1)] > 0)
                ;
            if(isEdge) m.insert(_idx(x, y), _idx(x, y)) = 1;
        }
    }
    m.makeCompressed();
    m.swap(E);
}

void PoissonImage::projectionSimpler(Eigen::SparseMatrix<float, Eigen::RowMajor>& S) const
{
    std::vector<Eigen::Triplet<float>> triplets;
    for(int x = 0; x < m_width; x++){
        for(int y = 0; y < m_height; y++){
            if(m_maskMap[_idx(x, y)] > 0){
                triplets.push_back(Eigen::Triplet<float>(triplets.size(), _idx(x, y), 1.0f));
            }
        }
    }
    Eigen::SparseMatrix<float, Eigen::RowMajor> m(triplets.size(), m_width * m_height);
    m.reserve(Eigen::VectorXi::Constant(m_width * m_height, 1));
    m.setFromTriplets(triplets.begin(), triplets.end());
    m.makeCompressed();
    m.swap(S);
}

void PoissonImage::diffOperator(Eigen::SparseMatrix<float, Eigen::RowMajor>& Dx, Eigen::SparseMatrix<float, Eigen::RowMajor>& Dy, DiffOp op) const
{
    Eigen::SparseMatrix<float, Eigen::RowMajor> m1(m_width * m_height, m_width * m_height);
    m1.reserve(Eigen::VectorXi::Constant(m_width * m_height, 2));
    Eigen::SparseMatrix<float, Eigen::RowMajor> m2(m_width * m_height, m_width * m_height);
    m2.reserve(Eigen::VectorXi::Constant(m_width * m_height, 2));
    for(int x = 0; x < m_width; x++){
        for(int y = 0; y < m_height; y++){
            {
                switch(op){
                    case DiffOp::Backward:
                        if(x == 0) continue;
                        if(m_maskMap[_idx(x-1, y)] == 0) continue;
                        m1.insert(_idx(x, y), _idx(x, y)) = 1;
                        m1.insert(_idx(x, y), _idx(x-1, y)) = -1;
                        break;
                    case DiffOp::Forward:
                        if(x == m_width - 1) continue;
                        if(m_maskMap[_idx(x+1, y)] == 0) continue;
                        m1.insert(_idx(x, y), _idx(x+1, y)) = 1;
                        m1.insert(_idx(x, y), _idx(x, y)) = -1;
                        break;
                    case DiffOp::Centered:
                        if(x == 0 || x == m_width - 1) continue;
                        if(m_maskMap[_idx(x-1, y)] == 0 || m_maskMap[_idx(x+1, y)] == 0) continue;
                        m1.insert(_idx(x, y), _idx(x+1, y)) = 0.5;
                        m1.insert(_idx(x, y), _idx(x-1, y)) = -0.5;
                        break;
                    case DiffOp::Fourier:
                        // No implement
                        break;
                }
            }

            {
                switch(op){
                    case DiffOp::Backward:
                        if(y == 0) continue;
                        if(m_maskMap[_idx(x, y-1)] == 0) continue;
                        m2.insert(_idx(x, y), _idx(x, y)) = 1;
                        m2.insert(_idx(x, y), _idx(x, y-1)) = -1;
                        break;
                    case DiffOp::Forward:
                        if(y == m_height - 1) continue;
                        if(m_maskMap[_idx(x, y+1)] == 0) continue;
                        m2.insert(_idx(x, y), _idx(x, y+1)) = 1;
                        m2.insert(_idx(x, y), _idx(x, y)) = -1;
                        break;
                    case DiffOp::Centered:
                        if(y == 0 || y == m_height - 1) continue;
                        if(m_maskMap[_idx(x, y-1)] == 0 || m_maskMap[_idx(x, y+1)] == 0) continue;
                        m2.insert(_idx(x, y), _idx(x, y+1)) = 0.5;
                        m2.insert(_idx(x, y), _idx(x, y-1)) = -0.5;
                        break;
                    case DiffOp::Fourier:
                        // No implement
                        break;
                }
            }
        }
    }
    m1.makeCompressed(); m1.swap(Dx);
    m2.makeCompressed(); m2.swap(Dy);
}

void PoissonImage::poissonSolver(Eigen::MatrixXf& R, bool wholeSpace) const
{
    Eigen::SparseMatrix<float, Eigen::RowMajor> L, M, E, S, ST, Dx, Dy;
    laplacianOperator(L); projectionMask(M); projectionEdge(E); projectionSimpler(S); ST = S.transpose(); diffOperator(Dx, Dy, m_divOperator);
    Eigen::SparseMatrix<float> A = S * L * M * ST;
    Eigen::MatrixXf b = S * (Dx * m_gradientX + Dy * m_gradientY - L * E * m_dstImage);

    Eigen::SparseLU<Eigen::SparseMatrix<float>> solver;
    solver.analyzePattern(A);
    solver.factorize(A);
    R = solver.solve(b);
    if(wholeSpace){
        R = m_dstImage - M * m_dstImage + ST * R;
    }
}

void PoissonImage::seamlessClone(const cv::Mat& srcMat, const cv::Mat& dstMat, const cv::Mat& maskMat, cv::Mat& output)
{
    StopWatch timer;

    if(srcMat.size() != maskMat.size()){
        std::cerr << "The sizes of source and mask should be the same!" << std::endl;
        return;
    }

    if(srcMat.size() != dstMat.size()){
        std::cerr << "The sizes of source and destination should be the same!" << std::endl;
        return;
    }

    // Size
    m_width = dstMat.cols;
    m_height = dstMat.rows;

    // Initialize eigen matrices
    m_srcImage = std::move(Eigen::MatrixXf(m_width * m_height, srcMat.channels()));
    m_dstImage = std::move(Eigen::MatrixXf(m_width * m_height, dstMat.channels()));
    m_maskMap = std::move(Eigen::Matrix<unsigned char, Eigen::Dynamic, 1>(m_width * m_height, 1));

    // Convert cv mat into eigen mat
    cvMat2EigenMat(srcMat, m_srcImage);
    cvMat2EigenMat(dstMat, m_dstImage);
    cvMat2EigenMat(maskMat, m_maskMap);

    timer.tick("Initialization");

    // Calculate gradient
    Eigen::SparseMatrix<float, Eigen::RowMajor> Dx, Dy; diffOperator(Dx, Dy, m_gradientOperator);
    switch(m_gradientScheme){
        case GradientScheme::Replace:
            {
                m_gradientX = Dx * m_srcImage;
                m_gradientY = Dy * m_srcImage;
            }
            break;
        case GradientScheme::Average:
            {
                m_gradientX = 0.5 * (Dx * m_srcImage + Dx * m_dstImage);
                m_gradientY = 0.5 * (Dy * m_srcImage + Dy * m_dstImage);
            }
            break;
        case GradientScheme::Maximum:
            {
                Eigen::MatrixXf srcGradX = Dx * m_srcImage, srcGradY = Dy * m_srcImage;
                Eigen::MatrixXf dstGradX = Dx * m_dstImage, dstGradY = Dy * m_dstImage;

                m_gradientX = std::move(Eigen::MatrixXf(m_width * m_height, srcMat.channels()));
                m_gradientY = std::move(Eigen::MatrixXf(m_width * m_height, srcMat.channels()));
                for(int x = 0; x < srcMat.channels(); x++){
                    for(int y = 0; y < m_width * m_height; y++){
                        float srcGx = srcGradX(y, x), srcGy = srcGradY(y, x);
                        float dstGx = dstGradX(y, x), dstGy = dstGradY(y, x);
                        if(srcGx*srcGx + srcGy*srcGy >= dstGx*dstGx + dstGy*dstGy){
                            m_gradientX(y, x) = srcGx;
                            m_gradientY(y, x) = srcGy;
                        }else{
                            m_gradientX(y, x) = dstGx;
                            m_gradientY(y, x) = dstGy;
                        }
                    }
                }
            }
            break;
    }

    timer.tick("Calculate Gradient");

    Eigen::MatrixXf R;
    poissonSolver(R, true);
    eigenMat2CvMat(R, output);

    timer.tick("Poisson Solving");
}

void PoissonImage::seamlessClone(cv::InputArray src, cv::InputArray dst, cv::InputArray mask, const cv::Point& offset, cv::OutputArray output, GradientScheme gradientSchm, DiffOp gradientOp, DiffOp divOp)
{
    if(src.size() != mask.size()){
        std::cerr << "The sizes of source and mask should be the same!" << std::endl;
        return;
    }

    // Size
    int w = dst.cols();
    int h = dst.rows();

    // Format check
    if(src.type() != CV_8UC3){
        std::cerr << "Format Check: The source mat have type: " << src.type() << ", but should have type: " << CV_8UC3 << std::endl;
        return;
    }
    if(dst.type() != CV_8UC3){
        std::cerr << "Format Check: The destination mat have type: " << dst.type() << ", but should have type: " << CV_8UC3 << std::endl;
        return;
    }
    if(mask.type() != CV_8UC1){
        std::cerr << "Format Check: The mask mat have type: " << mask.type() << ", but should have type: " << CV_8UC1 << std::endl;
        return;
    }

    // Adjust the position of the src image and mask
    cv::Mat srcMat = cv::Mat::zeros(h, w, CV_8UC3);
    cv::Mat maskMat = cv::Mat::zeros(h, w, CV_8UC1);
    cv::Mat dstMat = dst.getMat();
    {
        int left = (src.cols()/2) - offset.x - (w/2); if(left < 0) left = 0;
        int right = (src.cols()/2) - offset.x + (w/2); if(right >= src.cols()) right = src.cols() - 1;
        int top = (src.rows()/2) - offset.y - (h/2); if(top < 0) top = 0;
        int bottom = (src.rows()/2) - offset.y + (h/2); if(bottom >= src.rows()) bottom = src.rows() - 1;
        cv::Rect rcSrc(left, top, right - left, bottom - top);
        cv::Point center = cv::Point(w / 2, h / 2) + offset + cv::Point((left + right) / 2, (top + bottom) / 2) - cv::Point(src.cols() / 2, src.rows() / 2);
        cv::Rect rcDst(center.x - rcSrc.width/2, center.y - rcSrc.height/2, rcSrc.width, rcSrc.height);
        src.getMat()(rcSrc).copyTo(srcMat(rcDst));
        mask.getMat()(rcSrc).copyTo(maskMat(rcDst));
    }

    // Make input data compact
    srcMat = makeContinuous(srcMat);
    dstMat = makeContinuous(dstMat);
    maskMat = makeContinuous(maskMat);

    PoissonImage PI(gradientSchm, gradientOp, divOp);
    cv::Mat outputMat; PI.seamlessClone(srcMat, dstMat, maskMat, outputMat);
    outputMat.copyTo(output);
}
