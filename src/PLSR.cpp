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
	// Eigen::MatrixXd X_work = X;
	// Eigen::VectorXd y_work = y;
	int N = X.rows();
	int P = X.cols();

	// 1. Center data
	Eigen::VectorXd X_mean = X.colwise().mean();
	Eigen::MatrixXd Xc = X.rowwise() - X_mean.transpose();
	double y_mean = y.mean();
	Eigen::VectorXd yc = y.array() - y_mean;

	PLSR_Result result;
	result.W.resize(P, max_components);
	result.T.resize(N, max_components);
	result.P.resize(P, max_components);
	result.q.resize(max_components);
	result.b.resize(max_components);

	Eigen::MatrixXd X0 = Xc;
	Eigen::VectorXd y0 = yc;
	
	int comp = 0;
	for (; comp < max_components; ++comp) {
		// 2. Initialize u_a = yc_{a-1}
		Eigen::VectorXd u = y0;

		int iter = 0;
		int max_iter = 3;
		double w_change;

		Eigen::VectorXd w = Eigen::VectorXd::Zero(P);
		Eigen::VectorXd t = Eigen::VectorXd::Zero(N);
		double t_norm2;
		double w_norm;
		double q;

		do {
			Eigen::VectorXd w_old = w;

			w = X0.transpose() * u;
			w_norm = w.norm();

			if (w_norm < tol) break;
			w /= w_norm;
	
			// 3. X-scores t = X w
			t = X0 * w;

			// Eigen::VectorXd p = (X_work.transpose() * t) / t_norm2;

			// 4. y-loadings q = uᵀ t / (tᵀ t) (scalar in PLS1)
			t_norm2 = t.dot(t);
			q = t.dot(y0) / t_norm2;

			// 5. New y-scores
			u = y0 * q;

			w_change = (w - w_old).norm();
			// std::cout << "  Comp " << (comp+1) << " | iter " << iter+1
			// 				<< " | w_change: " << w_change
			// 				<< " | q: " << q
			// 				<< " | t.norm(): " << t.norm()
			// 				<< "\n";
		} while (w_change >= tol && ++iter < max_iter);

		// 6. X-loadings p = Xᵀ t / ||t||²
		Eigen::VectorXd p = (X0.transpose() * t) / t_norm2;

		// 7. Regression coefficient for component b = q
		double b = t.dot(y0) / t_norm2;

		// Deflation: remove explained variation
		X0 -= t * p.transpose();  // X := X - t pᵀ
		y0 -= t * q;              // y := y - t q
		
		// double y_explained = b * b * t.squaredNorm();
		// double y_total_var = yc.squaredNorm();
		// double y_residual_norm = y0.norm();

		// std::cout << "\n>>> COMPONENT " << (comp+1) << " FINISHED <<<\n";
    	// 	std::cout << "    iterations used      : " << iter << "\n";
    	// 	std::cout << "    q (y-loading)        : " << q << "\n";
    	// 	std::cout << "    b_comp               : " << b << "\n";
    	// 	std::cout << "    t.norm()             : " << t.norm() << "\n";
    	// 	std::cout << "    y_explained variance : " << y_explained << " ("
        //       		  << (y_explained / y_total_var * 100.0) << "%)\n";
    	// 	std::cout << "    y_residual.norm()    : " << y_residual_norm << "\n";
    	// 	std::cout << "    X0 residual norm     : " << X0.norm() << "\n";
		
		double abs_q = std::abs(q);
		if (abs_q < tol) {
    			// std::cout << "    EARLY STOP: |q| = " << abs_q 
              	// 		  << " < tol = " << tol << " -- Stopping.\n";
    			break;
		}

		// Store component
		result.W.col(comp) = w;
		result.T.col(comp) = t;
		result.P.col(comp) = p;
		result.q(comp) = q;
        result.b(comp) = b;
	}
	
	// Trim to actually extracted components
	if (comp < max_components) {
		result.W.conservativeResize(Eigen::NoChange, comp);
		result.T.conservativeResize(Eigen::NoChange, comp);
		result.P.conservativeResize(Eigen::NoChange, comp);
		result.q.conservativeResize(comp);
		result.b.conservativeResize(comp);
	}

	// // Compute β_PLS = W (Pᵀ W)⁻¹ Qᵀ from the returned result in your code
    // Eigen::VectorXd beta_pls = result.W * (result.P.transpose() * result.W).ldl().solve(result.q);

	return result;
}
