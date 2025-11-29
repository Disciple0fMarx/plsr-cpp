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
#include "fourier.hpp"
#include "bspline.hpp"
// #include "include/ResponseGenerator.hpp"


// Helper functions
double R2(const Eigen::VectorXd& y_true, const Eigen::VectorXd& y_pred);
// MatrixXd fourier_basis(int P, int nbasis, double period = 1.0);
// MatrixXd bspline_basis(int P, int nknots_interior, int order = 4);


int main() {
    std::cout << "=== PLSR on Sinusoidal Functional Predictors ===\n";
    std::cout << std::fixed << std::setprecision(7);

    // std::cout << "[DEBUG 1] Generating data...\n";
    auto dg = DataGenerator::make_default();
    dg.generate();
    const Eigen::MatrixXd& X = dg.X();
    // const Eigen::VectorXd& A = dg.amplitudes();
    const Eigen::VectorXd& t = dg.t();

    std::cout << "Data generated: N=" << X.rows() << ", P=" << X.cols() << "\n";

    const double T = 1.0;
    auto rg = ResponseGenerator::make_default(X, T);
    rg.generate();
    const Eigen::VectorXd& y = rg.y();
    // const Eigen::VectorXd& lin_pred = rg.lin_pred();
    const Eigen::VectorXd& beta_true = rg.beta_true();

    PLSR_Result res = pls_nipals(X, y, 5, 1);

    int components = res.W.cols();  // should be 1
    std::cout << "Components extracted: " << components << "\n";

    // Compute β_PLS = W (Pᵀ W)⁻¹ Qᵀ
    Eigen::VectorXd beta_pls = res.W * (res.P.transpose() * res.W).ldlt().solve(res.q);
    Eigen::VectorXd y_pred = X * beta_pls;  // because beta0 = 0

    double y_r2 = R2(y, y_pred);
    std::cout << "y_true vs y_pred: R² = " << y_r2 << "\n";

    double beta_r2 = R2(beta_true, beta_pls);
    std::cout << "beta_true vs beta_pls: R² = " << beta_r2 << "\n";

    std::cout << "\n--- Fourier Regression Baseline (K=1) ---\n";
    FourierRegressor fourier(T, 1);
    fourier.fit_y(X, y, t);  // ← this is the one that works perfectly

    Eigen::VectorXd beta_fourier = fourier.predict_beta(t);
    Eigen::VectorXd y_pred_fourier = X * beta_fourier;

    double y_r2_fourier = R2(y, y_pred_fourier);
    double beta_r2_fourier = R2(beta_true, beta_fourier);

    std::cout << "y_true vs y_pred (Fourier): R² = " << y_r2_fourier << "\n";
    std::cout << "beta_true vs beta_fourier: R² = " << beta_r2_fourier << "\n";

    std::cout << "Fourier intercept (β₀)   : " << fourier.intercept() << "\n";
    std::cout << "Fourier sin coeff (k=1)  : " << fourier.sin_coeff(1) << "\n";

    std::cout << "\n--- B-Spline Regression Baseline (15 interior knots) ---\n";
    BSplineRegressor bspline(15, 3);  // cubic, 15 interior knots → very flexible
    bspline.fit_y(X, y, t);

    Eigen::VectorXd beta_bs = bspline.predict_beta(t);
    Eigen::VectorXd y_pred_bs = X * beta_bs;

    double y_r2_bs = R2(y, y_pred_bs);
    double beta_r2_bs = R2(beta_true, beta_bs);

    std::cout << "y_true vs y_pred (B-Spline): R² = " << y_r2_bs << "\n";
    std::cout << "beta_true vs beta_bs: R² = " << beta_r2_bs << "\n";

    // Optional: pretty summary
    // std::cout << fourier.summary();

    // ==================================================================
    // Final comparison summary
    // ==================================================================
    std::cout << "\n=== FINAL COMPARISON ===\n";
    std::cout << "Method       | R²(y)     | R²(β)\n";
    std::cout << "-------------|-----------|-----------\n";
    std::cout << "PLSR         | " << std::setw(8) << y_r2
              << " | " << std::setw(8) << beta_r2 << "\n";
    std::cout << "Fourier (K=1)| " << std::setw(8) << y_r2_fourier
              << " | " << std::setw(8) << beta_r2_fourier << "\n";
    std::cout << "B-Spline     | " << std::setw(8) << y_r2_bs
              << " | " << std::setw(8) << beta_r2_bs << "\n";

    {
        std::ofstream out("results/response.txt");
        out << std::scientific << std::setprecision(12);
        out << "y_true y_pls y_fourier y_bspline\n";

        for (int i = 0; i < y.size(); ++i) {
            double yp_pls     = (X.row(i) * beta_pls).value();
            double yp_fourier = (X.row(i) * beta_fourier).value();
            double yp_bspline = (X.row(i) * beta_bs).value();

            out << y(i) << " "
                << yp_pls << " "
                << yp_fourier << " "
                << yp_bspline << "\n";
        }
    }

    {
        std::ofstream out("results/coef_functions.txt");
        out << std::scientific << std::setprecision(12);
        out << "t beta_true beta_pls beta_fourier beta_bspline\n";

        for (int j = 0; j < t.size(); ++j) {
            out << t(j) << " "
                << beta_true(j) << " "
                << beta_pls(j) << " "
                << beta_fourier(j) << " "
                << beta_bs(j) << "\n";
        }
    }

    return 0;
}


double R2(const Eigen::VectorXd& y_true, const Eigen::VectorXd& y_pred) {
    double ss_res = (y_true - y_pred).squaredNorm();
    double ss_tot = (y_true.array() - y_true.mean()).square().sum();
    return 1.0 - ss_res / ss_tot;
}
