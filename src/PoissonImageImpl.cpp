#include "StopWatch.h"
#include "PoissonImageImpl.h"

PoissonImageImpl::PoissonImageImpl(PoissonImage::GradientScheme gradientSchm, PoissonImage::DiffOp gradientOp, PoissonImage::DiffOp divOp)
    : m_gradientScheme(gradientSchm)
    , m_gradientOperator(gradientOp)
    , m_divOperator(divOp)
{
    return;
}

cv::Mat PoissonImageImpl::makeContinuous(const cv::Mat& m)
{
    if (!m.isContinuous()) {
        return m.clone();
    }
    return m;
}

template<typename T, int RowNum, int ColNum>
void PoissonImageImpl::cvMat2EigenMat(const cv::Mat& cvMat, Eigen::Matrix<T, RowNum, ColNum>& eigenMat)
{
    typedef Eigen::Matrix<unsigned char, Eigen::Dynamic, 1> V;
    for(int x = 0; x < m_width; x++){
        for(int y = 0; y < m_height; y++){
            eigenMat.row(_idx(x, y)) = Eigen::Map<const V>(cvMat.ptr<unsigned char>(y, x), cvMat.channels()).cast<T>();
        }
    }
}

template<typename T, int RowNum, int ColNum>
void PoissonImageImpl::eigenMat2CvMat(const Eigen::Matrix<T, RowNum, ColNum>& eigenMat, cv::Mat& cvMat)
{
    cvMat = cv::Mat::zeros(m_height, m_width, CV_32FC3);
    typedef Eigen::Matrix<T, Eigen::Dynamic, 1> V;
    for(int x = 0; x < m_width; x++){
        for(int y = 0; y < m_height; y++){
            Eigen::Map<V>(cvMat.ptr<float>(y, x), cvMat.channels()) = eigenMat.row(_idx(x, y));
        }
    }
}

void PoissonImageImpl::laplacianOperator(Eigen::SparseMatrix<float, Eigen::RowMajor>& L) const
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

void PoissonImageImpl::projectionMask(Eigen::SparseMatrix<float, Eigen::RowMajor>& M) const
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

void PoissonImageImpl::projectionEdge(Eigen::SparseMatrix<float, Eigen::RowMajor>& E) const
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

void PoissonImageImpl::projectionSampler(Eigen::SparseMatrix<float, Eigen::RowMajor>& S) const
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

void PoissonImageImpl::diffOperator(Eigen::SparseMatrix<float, Eigen::RowMajor> &Dx, Eigen::SparseMatrix<float, Eigen::RowMajor> &Dy, PoissonImage::DiffOp op) const
{
    Eigen::SparseMatrix<float, Eigen::RowMajor> m1(m_width * m_height, m_width * m_height);
    m1.reserve(Eigen::VectorXi::Constant(m_width * m_height, 2));
    Eigen::SparseMatrix<float, Eigen::RowMajor> m2(m_width * m_height, m_width * m_height);
    m2.reserve(Eigen::VectorXi::Constant(m_width * m_height, 2));
    for(int x = 0; x < m_width; x++){
        for(int y = 0; y < m_height; y++){
            {
                switch(op){
                    case PoissonImage::Backward:
                        if(x == 0) continue;
                        if(m_maskMap[_idx(x-1, y)] == 0) continue;
                        m1.insert(_idx(x, y), _idx(x, y)) = 1;
                        m1.insert(_idx(x, y), _idx(x-1, y)) = -1;
                        break;
                    case PoissonImage::Forward:
                        if(x == m_width - 1) continue;
                        if(m_maskMap[_idx(x+1, y)] == 0) continue;
                        m1.insert(_idx(x, y), _idx(x+1, y)) = 1;
                        m1.insert(_idx(x, y), _idx(x, y)) = -1;
                        break;
                    case PoissonImage::Centered:
                        if(x == 0 || x == m_width - 1) continue;
                        if(m_maskMap[_idx(x-1, y)] == 0 || m_maskMap[_idx(x+1, y)] == 0) continue;
                        m1.insert(_idx(x, y), _idx(x+1, y)) = 0.5;
                        m1.insert(_idx(x, y), _idx(x-1, y)) = -0.5;
                        break;
                    case PoissonImage::Fourier:
                        // No implement
                        break;
                }
            }

            {
                switch(op){
                    case PoissonImage::Backward:
                        if(y == 0) continue;
                        if(m_maskMap[_idx(x, y-1)] == 0) continue;
                        m2.insert(_idx(x, y), _idx(x, y)) = 1;
                        m2.insert(_idx(x, y), _idx(x, y-1)) = -1;
                        break;
                    case PoissonImage::Forward:
                        if(y == m_height - 1) continue;
                        if(m_maskMap[_idx(x, y+1)] == 0) continue;
                        m2.insert(_idx(x, y), _idx(x, y+1)) = 1;
                        m2.insert(_idx(x, y), _idx(x, y)) = -1;
                        break;
                    case PoissonImage::Centered:
                        if(y == 0 || y == m_height - 1) continue;
                        if(m_maskMap[_idx(x, y-1)] == 0 || m_maskMap[_idx(x, y+1)] == 0) continue;
                        m2.insert(_idx(x, y), _idx(x, y+1)) = 0.5;
                        m2.insert(_idx(x, y), _idx(x, y-1)) = -0.5;
                        break;
                    case PoissonImage::Fourier:
                        // No implement
                        throw "No implementation currently for Fourier differential operator.";
                }
            }
        }
    }
    m1.makeCompressed(); m1.swap(Dx);
    m2.makeCompressed(); m2.swap(Dy);
}

void PoissonImageImpl::poissonSolver(Eigen::MatrixXf& R, bool wholeSpace) const
{
    Eigen::SparseMatrix<float, Eigen::RowMajor> L, M, E, S, ST, Dx, Dy;
    laplacianOperator(L); projectionMask(M); projectionEdge(E); projectionSampler(S); ST = S.transpose(); diffOperator(Dx, Dy, m_divOperator);
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


bool PoissonImageImpl::seamlessClone(const cv::Mat& srcMat, const cv::Mat& dstMat, const cv::Mat& maskMat, cv::Mat& output)
{
    StopWatch timer;
    // Clear metric
    m_perfMetric = { 0.0, 0.0, 0.0 };

    if (srcMat.size() != maskMat.size()){
        std::cerr << "The sizes of source and mask should be the same!" << std::endl;
        return false;
    }

    if(srcMat.size() != dstMat.size()){
        std::cerr << "The sizes of source and destination should be the same!" << std::endl;
        return false;
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

    // Initialization time
    m_perfMetric.m_tInit = timer.tick();

    // Calculate gradient
    Eigen::SparseMatrix<float, Eigen::RowMajor> Dx, Dy; diffOperator(Dx, Dy, m_gradientOperator);
    switch(m_gradientScheme){
        case PoissonImage::Replace:
            {
                m_gradientX = Dx * m_srcImage;
                m_gradientY = Dy * m_srcImage;
            }
            break;
        case PoissonImage::Average:
            {
                m_gradientX = 0.5 * (Dx * m_srcImage + Dx * m_dstImage);
                m_gradientY = 0.5 * (Dy * m_srcImage + Dy * m_dstImage);
            }
            break;
        case PoissonImage::Maximum:
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

    // Gradient calculation time
    m_perfMetric.m_tGradient = timer.tick();

    Eigen::MatrixXf R;
    poissonSolver(R, true);
    eigenMat2CvMat(R, output);

    // Poisson solving time
    m_perfMetric.m_tSolver = timer.tick();

    return true;
}
