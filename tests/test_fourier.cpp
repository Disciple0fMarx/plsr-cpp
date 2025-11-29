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
#include <Eigen/Dense>
#include <cmath>
#include <numbers>

#include "../include/DataGenerator.hpp"
#include "../include/ResponseGenerator.hpp"
#include "../include/fourier.hpp"


int main() {
    std::cout << std::fixed << std::setprecision(12);
    std::cout << "=== FourierRegressor Unit Tests ===\n\n";

    bool all_ok = true;
    const double T = 1.0;

    // Use fixed seed and known config for reproducibility
    auto dg = DataGenerator::make_default();
    // dg.config_.seed = 12345;
    dg.generate();

    const auto& X = dg.X();
    const auto& t = dg.t();

    // ==================================================================
    // Test 1: With K=1 and noiseless data → perfect recovery of beta_true
    // ==================================================================
    {
        std::cout << "Test 1: K=1 recovers exact beta_true (noiseless)\n";

        ResponseGenerator::Config cfg;
        cfg.c_sin = 5.0;
        cfg.beta0 = -2.0;
        cfg.add_noise = false;

        ResponseGenerator rg(X, T, cfg);
        rg.generate();

        FourierRegressor fourier(T, 1);
        fourier.fit_beta(rg.beta_true(), t);

        Eigen::VectorXd beta_recovered = fourier.predict_beta(t);
        double error = (beta_recovered - rg.beta_true()).norm();

        std::cout << "   Reconstruction error ||beta_hat - beta_true|| = " << error << "\n";

        if (error < 1e-10) {
            std::cout << "  PASS: Perfect recovery with K=1 (expected for single sine)\n";
        } else {
            std::cerr << "  FAIL: Did not recover true beta(t) accurately\n";
            all_ok = false;
        }

        // Also check coefficients directly
        double c_sin_recovered = fourier.sin_coeff(1);
        if (std::abs(c_sin_recovered - 5.0) > 1e-10) {
            std::cerr << "  FAIL: sin_coeff(1) = " << c_sin_recovered << ", expected 5.0\n";
            all_ok = false;
        }
        if (std::abs(fourier.intercept() + 2.0) > 1e-10) {
            std::cerr << "  FAIL: intercept = " << fourier.intercept() << ", expected -2.0\n";
            all_ok = false;
        }
        if (std::abs(fourier.cos_coeff(1)) > 1e-10) {
            std::cerr << "  FAIL: cos_coeff(1) should be ~0, got " << fourier.cos_coeff(1) << "\n";
            all_ok = false;
        }
    }

    std::cout << "\n";

    // ==================================================================
    // Test 2: fit_y() recovers same model from noisy observations
    // ==================================================================
    {
        std::cout << "Test 2: fit_y() recovers true beta from noisy y (K=1)\n";

        ResponseGenerator::Config cfg;
        cfg.c_sin = 4.0;
        cfg.beta0 = 0.0;
        cfg.add_noise = true;
        cfg.noise_std = 0.05;
        cfg.seed = 777;

        ResponseGenerator rg(X, T, cfg);
        rg.generate();

        FourierRegressor fourier(T, 1);
        fourier.fit_y(X, rg.y(), t);

        Eigen::VectorXd beta_recovered = fourier.predict_beta(t);
        double error = (beta_recovered - rg.beta_true()).norm() / rg.beta_true().norm();

        std::cout << "   Relative error in beta recovery = " << error << "\n";
        std::cout << "   Recovered sin_coeff(1) = " << fourier.sin_coeff(1) << " (true: 4.0)\n";

        if (error < 0.05) {  // ~5% error acceptable with σ=0.05 noise
            std::cout << "  PASS: Good recovery from noisy observations\n";
        } else {
            std::cerr << "  FAIL: Poor recovery of beta(t) from noisy y\n";
            all_ok = false;
        }
    }

    std::cout << "\n";

    // ==================================================================
    // Test 3: Higher K does not hurt (overfitting control)
    // ==================================================================
    {
        std::cout << "Test 3: Using K>1 does not degrade fit on pure sine data\n";

        ResponseGenerator::Config cfg;
        cfg.c_sin = 3.0;
        cfg.beta0 = 1.0;
        cfg.add_noise = false;

        ResponseGenerator rg(X, T, cfg);
        rg.generate();

        FourierRegressor fourier(T, 5);  // Over-specified
        fourier.fit_beta(rg.beta_true(), t);

        Eigen::VectorXd beta_rec = fourier.predict_beta(t);
        double error = (beta_rec - rg.beta_true()).norm();

        bool higher_harmonics_zero = true;
        for (int k = 2; k <= 5; ++k) {
            if (std::abs(fourier.cos_coeff(k)) > 1e-8 || std::abs(fourier.sin_coeff(k)) > 1e-8) {
                higher_harmonics_zero = false;
            }
        }

        std::cout << "   Reconstruction error (K=5) = " << error << "\n";

        if (error < 1e-10 && higher_harmonics_zero) {
            std::cout << "  PASS: K=5 still perfectly fits and higher coeffs ≈ 0\n";
        } else {
            if (error >= 1e-10) std::cerr << "  FAIL: Reconstruction degraded with higher K\n";
            if (!higher_harmonics_zero) std::cerr << "  FAIL: Higher harmonics not zero in clean data\n";
            all_ok = false;
        }
    }

    std::cout << "\n";

    // ==================================================================
    // Test 4: predict_y() matches X * beta_hat
    // ==================================================================
    {
        std::cout << "Test 4: predict_y(X) == X * beta_hat(t)\n";

        ResponseGenerator::Config cfg;
        cfg.c_sin = 6.0;
        cfg.beta0 = 0.0;
        cfg.add_noise = false;

        ResponseGenerator rg(X, T, cfg);
        rg.generate();

        FourierRegressor fourier(T, 1);
        fourier.fit_beta(rg.beta_true(), t);

        Eigen::VectorXd y_pred = fourier.predict_y(X);
        Eigen::VectorXd y_manual = X * fourier.predict_beta(t);

        double diff = (y_pred - y_manual).norm();

        if (diff < 1e-12) {
            std::cout << "  PASS: predict_y() is consistent with matrix multiplication\n";
        } else {
            std::cerr << "  FAIL: predict_y() differs from X * beta_hat\n";
            all_ok = false;
        }
    }

    std::cout << "\n";

    // ==================================================================
    // Test 5: summary() reports correct coefficients
    // ==================================================================
    {
        std::cout << "Test 5: summary() string contains correct coefficient values\n";

        ResponseGenerator::Config cfg;
        cfg.c_sin = 10.0;
        cfg.beta0 = -1.0;
        cfg.add_noise = false;

        ResponseGenerator rg(X, T, cfg);
        rg.generate();

        FourierRegressor fourier(T, 1);
        fourier.fit_beta(rg.beta_true(), t);

        std::string summary = fourier.summary();

        bool has_intercept = summary.find("-1.") != std::string::npos ||
                             summary.find("-1.0000") != std::string::npos;
        bool has_sin_coeff = summary.find("10.") != std::string::npos ||
                             summary.find("10.0000") != std::string::npos;

        if (has_intercept && has_sin_coeff) {
            std::cout << "  PASS: summary() reports correct intercept and sin coefficient\n";
        } else {
            std::cerr << "  FAIL: summary() missing correct coefficient values\n";
            std::cout << "     Got:\n" << summary << "\n";
            all_ok = false;
        }
    }

    std::cout << "\n";

    // ==================================================================
    // Test 6: Exception safety — call predict before fit
    // ==================================================================
    {
        std::cout << "Test 6: Throws if predict called before fitting\n";

        FourierRegressor fourier(T, 1);
        bool threw = false;

        try {
            fourier.predict_beta(t);
        } catch (const std::runtime_error&) {
            threw = true;
        }

        if (threw) {
            std::cout << "  PASS: Correctly throws when not fitted\n";
        } else {
            std::cerr << "  FAIL: Did not throw when predicting before fit\n";
            all_ok = false;
        }
    }

    // ==================================================================
    // Final result
    // ==================================================================
    std::cout << "\n";
    if (all_ok) {
        std::cout << "ALL FOURIER REGRESSOR TESTS PASSED!\n\n";
        return 0;
    } else {
        std::cout << "SOME FOURIER REGRESSOR TESTS FAILED!\n\n";
        return 1;
    }
}
