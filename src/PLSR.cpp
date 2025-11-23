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
#include "PLSR.hpp"
#include <iostream>


/**
 * @brief Implementation of the NIPALS PLS1 algorithm
 *
 * See detailed documentation in PLSR.hpp for mathematical formulation
 * and references.
 */
PLSR_Result pls_nipals(const Eigen::MatrixXd& X,
		       const Eigen::VectorXd& y,
		       int max_components,
		       double tol)
{
	Eigen::MatrixXd X_work = X;
	Eigen::VectorXd y_work = y;

	int N = X.rows();
	int P = X.cols();
	int A = std::min({max_components, N-1, P});

	PLSR_Result result;
	result.W.resize(P, A);
	result.T.resize(N, A);
	result.P.resize(P, A);
	result.q.resize(A);
	result.b.resize(A);
	
	int comp = 0;
	for (; comp < A; ++comp) {
		// 1. X-weights w (direction of max covariance)
		Eigen::VectorXd w = X_work.transpose() * y_work;

		double w_norm = w.norm();
		if (w_norm < tol) break;
		w /= w_norm;
		
		// 2. X-scores t = X w
		Eigen::VectorXd t = X_work * w;

		// 3. X-loadings p = Xᵀ t / ||t||²
		double t_norm2 = t.dot(t);
		Eigen::VectorXd p = (X_work.transpose() * t) / t_norm2;

		// 4. y-loading q (scalar in PLS1)
		double q = t.dot(y_work) / t_norm2;

		// Deflation: remove explained variation
		X_work -= t * p.transpose();  // X := X - t pᵀ
		y_work -= t * q;              // y := y - t q

		// Store component
		result.W.col(comp) = w;
		result.T.col(comp) = t;
		result.P.col(comp) = p;
		result.q(comp) = q;
	}
	
	// Trim to actually extracted components
	int A_used = comp;
	if (A_used < A) {
		result.W.conservativeResize(Eigen::NoChange, A_used);
		result.T.conservativeResize(Eigen::NoChange, A_used);
		result.P.conservativeResize(Eigen::NoChange, A_used);
		result.q.conservativeResize(A_used);
		result.b.conservativeResize(A_used);
	}

	// Compute regression coefficients in score space: b = (TᵀT)⁻¹Tᵀy
	if (A_used > 0) {
		result.b = result.T.leftCols(A_used).householderQr().solve(y);
	} else {
		result.b.setZero();
	}

	return result;
}

