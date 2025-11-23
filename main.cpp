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
#include <iostream>
#include <iomanip>
#include <fstream>
#include "DataGenerator.hpp"
#include "ResponseGenerator.hpp"
#include "PLSR.hpp"

using Eigen::MatrixXd;
using Eigen::VectorXd;

void new_main();

// Helper functions
double R2(const VectorXd& y_true, const VectorXd& y_pred);
MatrixXd fourier_basis(int P, int nbasis, double period = 1.0);
MatrixXd bspline_basis(int P, int nknots_interior, int order = 4);

int main() {
    new_main();
    return 0;
}


void new_main() {
    std::cout << "=== PLSR on Sinusoidal Functional Predictors ===\n";

    // std::cout << "[DEBUG 1] Generating data...\n";
    auto dg = DataGenerator::make_default();
    dg.generate();
    const MatrixXd& X = dg.X();
    const VectorXd& A = dg.amplitudes();
    const VectorXd& t = dg.t();

    auto rg = ResponseGenerator::make_default(A);
    rg.generate();
    const VectorXd& y = rg.y();
    const VectorXd& y_true = rg.y_true();

    // std::cout << "[DEBUG 2] Centering data...\n";
    VectorXd Xmean = X.colwise().mean();
    MatrixXd Xc = X.rowwise() - Xmean.transpose();
    double y_mean = y.mean();
    VectorXd yc = y.array() - y_mean;

    std::cout << "Data generated: N=" << X.rows() << ", P=" << X.cols() << "\n";

    std::vector<double> pls_r2;
    VectorXd y_pred_pls_best;
    PLSR_Result pls_res_1comp;

    // std::cout << "[DEBUG 3] Starting PLS loop...\n";
	// MatrixXd Phi = fourier_basis(X.cols(), 21);  // 21 = 1 + 10 sin/cos pairs
	// MatrixXd X_func = Xc * Phi;                  // 1000 × 21 smooth coefficients

    for (int a = 1; a <= 5; ++a) {
        // std::cout << "[DEBUG 3." << a << "] Running PLS with " << a << " component(s)...\n";
        PLSR_Result res = pls_nipals(Xc, yc, a);
		// PLSR_Result res = pls_nipals(X_func, yc, a);

        // CRITICAL FIX: Use the correct number of components for prediction
        VectorXd y_fitted_centered = res.T.leftCols(a) * res.b.head(a);
        VectorXd y_fitted = y_fitted_centered.array() + y_mean;

        double r2 = R2(y, y_fitted);
        pls_r2.push_back(r2);

        std::cout << "PLS " << a << " component(s) → R² = "
                  << std::fixed << std::setprecision(12) << r2 << "\n";

        if (a == 1) {
            y_pred_pls_best = y_fitted;
            pls_res_1comp = std::move(res);
        }
    }

	std::cout << "\n=== CHECKING WHICH COMPONENT IS THE SINE WAVE ===\n";
	PLSR_Result res5 = pls_nipals(Xc, yc, 5);

	// Check correlation of each loading with true sine
	VectorXd true_sine = (t.array() * 2 * std::numbers::pi).sin();

	for (int i = 0; i < 5; ++i) {
		double corr = res5.P.col(i).normalized().dot(true_sine.normalized());
		std::cout << "Component " << (i+1) 
				<< " loading correlation with sin(2πt): " << corr << "\n";
	}

    // std::cout << "[DEBUG 4] PLS loop finished successfully.\n";

    // std::cout << "[DEBUG 5] Writing data.txt and loadings.txt...\n";
    {
        std::ofstream out("results/data.txt");
        out << std::scientific << std::setprecision(12);
		out << "y_true y_pred\n";
        for (int i = 0; i < y.size(); ++i)
            out << y_true(i) << " " << y_pred_pls_best(i) << "\n";
    }
    {
        std::ofstream out("results/loadings.txt");
		out << "t true_beta pls_loading\n";
        for (int j = 0; j < t.size(); ++j) {
            double true_sine = std::sin(2.0 * std::numbers::pi * t(j));
            out << t(j) << " " << true_sine << " " << pls_res_1comp.P.col(0)(j) << "\n";
        }
    }
    // std::cout << "[DEBUG 6] Files written.\n";

    // std::cout << "[DEBUG 7] Starting OLS...\n";
    double r2_ols_raw = -999;
    try {
        VectorXd beta_ols_raw = (Xc.transpose() * Xc).ldlt().solve(Xc.transpose() * yc);
        VectorXd y_pred = Xc * beta_ols_raw + y_mean * VectorXd::Ones(y.size());
        r2_ols_raw = R2(y, y_pred);
        // std::cout << "[DEBUG 8] OLS done. R² = " << r2_ols_raw << "\n";
    } catch (...) {
        std::cout << "OLS on raw X failed (as expected)\n";
    }

    // std::cout << "[DEBUG 9] Starting Fourier basis...\n";
    int nbasis_fourier = 9;
    MatrixXd Phi_f = fourier_basis(X.cols(), nbasis_fourier);
    MatrixXd X_fourier = Xc * Phi_f;
    MatrixXd coef_f = (X_fourier.transpose() * X_fourier).ldlt().solve(X_fourier.transpose() * yc);
    VectorXd y_pred_f = X_fourier * coef_f + y_mean * VectorXd::Ones(y.size());
    double r2_fourier = R2(y, y_pred_f);
    // std::cout << "[DEBUG 10] Fourier done. R² = " << r2_fourier << "\n";

    // std::cout << "[DEBUG 11] Starting B-spline basis...\n";
    MatrixXd Phi_b = bspline_basis(X.cols(), 12, 4);
    // std::cout << "[DEBUG 12] B-spline basis created successfully!\n";

    MatrixXd X_bs = Xc * Phi_b;
    // std::cout << "[DEBUG 13] Projected onto B-spline basis\n";

    MatrixXd coef_b = (X_bs.transpose() * X_bs).ldlt().solve(X_bs.transpose() * yc);
    // std::cout << "[DEBUG 14] B-spline coefficients solved\n";

    VectorXd y_pred_b = X_bs * coef_b + y_mean * VectorXd::Ones(y.size());
    // std::cout << "[DEBUG 15] B-spline predictions done\n";

    double r2_bspline = R2(y, y_pred_b);
    // std::cout << "[DEBUG 16] B-spline R² computed: " << r2_bspline << "\n";

    // std::cout << "[DEBUG 17] Writing comparison table...\n";
    std::ofstream comp("results/comparison.txt");
    comp << std::fixed << std::setprecision(12);
    comp << "Method & Components / Basis size & $R^2$ \\\\\\hline\n";
    comp << "PLS (1 comp) & 1 & " << pls_r2[0] << " \\\\\n";
    comp << "PLS (5 comp) & 5 & " << pls_r2[4] << " \\\\\n";
    comp << "OLS raw X & " << X.cols() << " & " << (r2_ols_raw > -100 ? std::to_string(r2_ols_raw) : "failed") << " \\\\\n";
    comp << "Fourier basis & " << nbasis_fourier << " & " << r2_fourier << " \\\\\n";
    comp << "B-spline (cubic) & 16 & " << r2_bspline << " \\\\\\hline\n";

    std::cout << "\n=== FINAL RESULTS ===\n";
    std::cout << "PLS 1 component  → R² = " << pls_r2[0] << "\n";
    std::cout << "Fourier (9 basis) → R² = " << r2_fourier << "\n";
    std::cout << "B-spline (16)     → R² = " << r2_bspline << "\n";

	// Save Fourier and B-spline coefficient functions for plotting
	{
		std::ofstream f("results/fourier_coefs.txt");
		f << "t true_beta fourier_beta\n";
		for (int j = 0; j < t.size(); ++j)
			f << t(j) << " " << std::sin(2*M_PI*t(j)) << " " 
			<< (Phi_f.row(j) * coef_f) << "\n";
	}

	{
		std::ofstream b("results/bspline_coefs.txt");
		b << "t true_beta bspline_beta\n";
		for (int j = 0; j < t.size(); ++j)
			b << t(j) << " " << std::sin(2*M_PI*t(j)) << " " 
			<< (Phi_b.row(j) * coef_b) << "\n";
	}
}

double R2(const VectorXd& y_true, const VectorXd& y_pred) {
    double ss_res = (y_true - y_pred).squaredNorm();
    double ss_tot = (y_true.array() - y_true.mean()).square().sum();
    return 1.0 - ss_res / ss_tot;
}

MatrixXd fourier_basis(int P, int nbasis, double period) {
    if (nbasis % 2 == 0) ++nbasis;
    MatrixXd phi(P, nbasis);
    VectorXd t = VectorXd::LinSpaced(P, 0.0, period);
    double omega = 2 * std::numbers::pi / period;
    phi.col(0).setConstant(0.7071067811865475);
    for (int k = 1; k <= (nbasis-1)/2; ++k) {
        phi.col(2*k-1) = (omega * k * t.array()).sin();
        phi.col(2*k)   = (omega * k * t.array()).cos();
    }
    return phi;
}

MatrixXd bspline_basis(int P, int nknots_interior, int order) {
    int nbasis = nknots_interior + order;
    MatrixXd B = MatrixXd::Zero(P, nbasis);
    VectorXd t = VectorXd::LinSpaced(P, 0.0, 1.0);
    double h = 1.0 / nknots_interior;

    for (int i = 0; i < P; ++i) {
        double x = t(i);
        int j0 = std::max(0, static_cast<int>(x/h) - order + 1);
        int j1 = std::min(nbasis, static_cast<int>(x/h) + 1);
        for (int j = j0; j < j1; ++j) {
            double c = x/h - j;
            double ac = std::abs(c);
            if (ac < 1) B(i,j) = (4.0 + 3.0*ac*(ac*ac - 4.0))/6.0;
            else if (ac < 2) B(i,j) = (8.0 - 3.0*ac*(ac*ac - 4.0))/6.0;
            else if (ac < 3) B(i,j) = std::pow(3.0 - ac, 3)/6.0;
        }
    }
    return B;
}
