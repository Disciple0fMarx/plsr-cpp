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

// FORBID any other definition of pls_nipals
// #ifdef pls_nipals
// #error "pls_nipals already defined elsewhere — you have duplicate definitions!"
// #endif

#include <Eigen/Dense>


/**
 * @struct PLSR_Result
 * @brief Container for all outputs of the NIPALS PLS1 algorithm.
 *
 * This structure holds the decomposition components and regression coefficients
 * required for interpretation and prediction in Partial Least Squares Regression
 * (PLS1 — univariate response).
 */
struct PLSR_Result {
	Eigen::MatrixXd W;  ///< X-weights (P × A): direction of maximum covariance
	Eigen::MatrixXd T;  ///< X-scores (N × A): latent projections of X
	Eigen::MatrixXd P;  ///< X-loadings (P × A): how scores relate back to original X-space
	Eigen::VectorXd q;  ///< y-loadings (A × 1): regression of y on scores t_a
	Eigen::VectorXd b;  ///< Inner relation coefficients (A × 1): ŷ = T b (in score space)
};


/**
 * @brief NIPALS algorithm for PLS1 (Partial Least Squares Regression with univariate response)
 *
 * Implements the original Nonlinear Iterative Partial Least Squares (NIPALS)
 * algorithm for PLS1 as described by Wold et al. (1984) and widely used in
 * chemometrics and functional data analysis.
 *
 * The algorithm iteratively extracts latent components that maximize the
 * covariance between X and y, with orthogonal deflation of both X and y.
 *
 * @param X                Predictor matrix (N × P), rows = observations, columns = variables
 * @param y                Univariate response vector (N × 1)
 * @param max_components   Maximum number of latent components to extract (default: 5)
 * @param tol              Convergence tolerance for weight vector norm (default: 1e-12)
 *
 * @return PLSR_Result     Structure containing weights, scores, loadings, and regression coefficients
 *
 * @note Automatically stops early if residual variance in X or y becomes negligible.
 *       The returned matrices are trimmed to the actual number of components used.
 *
 * @warning Assumes X and y are centered (or will be handled externally). For functional
 *          data, X typically contains coefficients of basis expansions (B-spline, Fourier).
 *
 * @see PLSR_Result
 *
 * @authors Dhiaa Eddine Bahri <dhya.bahri@proton.me>
 *          Malek Rihani <malek.rihani090@gmail.com>
 *
 * @date 2025
 */
PLSR_Result pls_nipals(const Eigen::MatrixXd& X,
		       const Eigen::VectorXd& y,
		       int max_components = 5,
		       double tol = 1e-12);

