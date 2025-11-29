/**
 * PLSR-CPP - Partial Least Squares Regression in C++ for Functional Data
 * 
 * Copyright (C) 2025  Dhiaa Eddine Bahri <dhya.bahri@proton.me>
 *                     Malek Rihani <malek.rihani090@gmail.com>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */
#pragma once

#include <Eigen/Dense>
#include <vector>
#include <string>


/**
 * @class FourierRegressor
 * @brief Performs Fourier (trigonometric) regression with user-specified number of harmonics.
 *
 * The model is:
 *   y_i(t) = β₀ + Σ_{k=1}^{K} [β_{c,k} cos(2π k t / T) + β_{s,k} sin(2π k t / T)] + δ_i
 *
 * This is equivalent to fitting a linear model in the Fourier basis up to harmonic K.
 *
 * Key features:
 *  - Exact least-squares solution using normal equations (Eigen's CompleteOrthogonalDecomposition for stability)
 *  - Can fit either the true beta(t) coefficients or the response y directly
 *  - Provides reconstructed signal and coefficients with interpretable names
 *
 * @note Perfect for periodic data with known fundamental period T.
 *
 * @authors Dhiaa Eddine Bahri <dhya.bahri@proton.me>
 *          Malek Rihani <malek.rihani090@gmail.com>
 *
 * @date 2025
 *
 */
class FourierRegressor
{
public:
    /**
     * @brief Construct a Fourier regressor.
     * @param T Period of the fundamental frequency (same as used in data generation)
     * @param K Maximum harmonic index to include (K=1 recovers exactly your generating model)
     */
    FourierRegressor(double T, int K = 1);

    /**
     * @brief Fit the model to a vector of true coefficients β(t_j) (P-dimensional)
     * @param beta_true Vector of true time-varying coefficients [P x 1]
     */
    void fit_beta(const Eigen::VectorXd& beta_true, const Eigen::VectorXd& t);

    /**
     * @brief Fit the model directly to observed responses y_i
     * @param X Design matrix [N x P] (each row is a sine wave sample)
     * @param y Response vector [N x 1]
     */
    void fit_y(const Eigen::MatrixXd& X, const Eigen::VectorXd& y, const Eigen::VectorXd& t);

    /**
     * @brief Reconstruct the fitted coefficient function β̂(t) on the original time grid
     * @param t Time points [P x 1], same as used in data generation
     * @return Reconstructed β̂(t) vector
     */
    Eigen::VectorXd predict_beta(const Eigen::VectorXd& t) const;

    /**
     * @brief Predict responses for new data matrix X
     * @param X Input matrix [N x P]
     * @return Predicted y values
     */
    Eigen::VectorXd predict_y(const Eigen::MatrixXd& X) const;

    // --- Accessors ---

    /// Intercept term β₀
    double intercept() const { return coeffs_(0); }

    /// Cosine coefficient for harmonic k (k starts at 1)
    double cos_coeff(int k) const;

    /// Sine coefficient for harmonic k (k starts at 1)
    double sin_coeff(int k) const;

    /// Total number of harmonics used
    int num_harmonics() const { return K_; }

    /// Get all coefficients as vector: [β₀, β_c1, β_s1, β_c2, β_s2, ..., β_cK, β_sK]
    const Eigen::VectorXd& coefficients() const { return coeffs_; }

    /// Human-readable summary of fitted coefficients
    std::string summary() const;

private:
    void build_design_matrix(const Eigen::VectorXd& t);

    double T_;                     // Period
    int K_;                        // Number of harmonics
    Eigen::MatrixXd Phi_;          // Design matrix [P x (1+2K)]
    Eigen::VectorXd coeffs_;       // Fitted coefficients
    Eigen::VectorXd time_grid_;    // Stored time points (for prediction)
};