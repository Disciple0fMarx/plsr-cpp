#include "fourier.hpp"
#include <numbers>
#include <sstream>
#include <iomanip>


using namespace std::numbers;


FourierRegressor::FourierRegressor(double T, int K)
    : T_(T), K_(std::max(1, K))
{
    if (T_ <= 0.0)
        throw std::invalid_argument("Period T must be positive");
}

void FourierRegressor::build_design_matrix(const Eigen::VectorXd& t)
{
    const int P = t.size();
    const int n_params = 1 + 2 * K_;
    Phi_.resize(P, n_params);
    time_grid_ = t;

    Phi_.col(0).setOnes();  // intercept

    for (int j = 0; j < P; ++j)
    {
        double tj = t(j);
        for (int k = 1; k <= K_; ++k)
        {
            double arg = 2.0 * pi * k * tj / T_;
            Phi_(j, 2*k - 1) = std::cos(arg);   // cosine term for harmonic k
            Phi_(j, 2*k)     = std::sin(arg);   // sine term for harmonic k
        }
    }
}

// void FourierRegressor::fit_beta(const Eigen::VectorXd& beta_true)
// {
//     const int P = beta_true.size();
//     Eigen::VectorXd t = Eigen::VectorXd::LinSpaced(P, 0.0, T_ * (P-1.0)/(P-1.0)); // reconstruct time

//     build_design_matrix(t);

//     // Solve least squares: θ = (ΦᵀΦ)⁻¹ Φᵀ β
//     // Using CompleteOrthogonalDecomposition for numerical robustness
//     Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> cod(Phi_);
//     coeffs_ = cod.solve(beta_true);
// }
void FourierRegressor::fit_beta(const Eigen::VectorXd& beta_true, const Eigen::VectorXd& t)
{
    if (beta_true.size() != t.size())
        throw std::invalid_argument("beta_true and t must have same size");

    build_design_matrix(t);

    Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> cod(Phi_);
    coeffs_ = cod.solve(beta_true);
}

// void FourierRegressor::fit_y(const Eigen::MatrixXd& X, const Eigen::VectorXd& y)
// {
//     const int P = X.cols();
//     Eigen::VectorXd t = Eigen::VectorXd::LinSpaced(P, 0.0, T_ * (P-1.0)/(P-1.0));

//     build_design_matrix(t);

//     // True linear predictor without noise/intercept: X * β_true
//     // We recover β_true first, then use same design matrix
//     Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> cod(Phi_);
//     coeffs_ = cod.solve(X.transpose() * y);  // Equivalent to fitting projected coefficients
// }
void FourierRegressor::fit_y(const Eigen::MatrixXd& X, const Eigen::VectorXd& y, const Eigen::VectorXd& t)
{
    if (X.cols() != t.size())
        throw std::invalid_argument("X columns must match t.size()");

    build_design_matrix(t);

    // Correct way: y ≈ X * (Phi * theta)  →  min ||y - (X*Phi) theta||
    Eigen::MatrixXd design = X * Phi_;                 // [N x (1+2K)]
    Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> cod(design);
    coeffs_ = cod.solve(y);
}

Eigen::VectorXd FourierRegressor::predict_beta(const Eigen::VectorXd& t) const
{
    if (coeffs_.size() == 0)
        throw std::runtime_error("Model has not been fitted yet");

    const int n_params = 1 + 2 * K_;
    Eigen::VectorXd pred(t.size());

    for (int j = 0; j < t.size(); ++j)
    {
        double tj = t(j);
        double val = coeffs_(0);  // intercept

        for (int k = 1; k <= K_; ++k)
        {
            double arg = 2.0 * pi * k * tj / T_;
            val += coeffs_(2*k - 1) * std::cos(arg);
            val += coeffs_(2*k)     * std::sin(arg);
        }
        pred(j) = val;
    }
    return pred;
}

Eigen::VectorXd FourierRegressor::predict_y(const Eigen::MatrixXd& X) const
{
    if (coeffs_.size() == 0)
        throw std::runtime_error("Model has not been fitted yet");

    const int P = X.cols();
    Eigen::VectorXd t = Eigen::VectorXd::LinSpaced(P, 0.0, T_ * (P-1.0)/(P-1.0));
    Eigen::VectorXd beta_hat = predict_beta(t);
    return X * beta_hat;
}

double FourierRegressor::cos_coeff(int k) const
{
    if (k < 1 || k > K_) throw std::out_of_range("Harmonic k out of range");
    return coeffs_(2*k - 1);
}

double FourierRegressor::sin_coeff(int k) const
{
    if (k < 1 || k > K_) throw std::out_of_range("Harmonic k out of range");
    return coeffs_(2*k);
}

std::string FourierRegressor::summary() const
{
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << "Fourier Regression Summary (T = " << T_ << ", K = " << K_ << ")\n";
    oss << "------------------------------------------------\n";
    oss << "β₀ (intercept)     = " << coeffs_(0) << "\n";

    for (int k = 1; k <= K_; ++k)
    {
        oss << "Harmonic k=" << k << ":\n";
        oss << "   cos coefficient = " << cos_coeff(k) << "\n";
        oss << "   sin coefficient = " << sin_coeff(k) << "\n";
    }

    return oss.str();
}
