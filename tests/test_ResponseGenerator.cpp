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

int main() {
    std::cout << std::fixed << std::setprecision(12);
    std::cout << "=== ResponseGenerator Unit Tests ===\n\n";

    bool all_ok = true;
    const double T = 1.0;

    // ==================================================================
    // Test 1: beta_true_ is exactly c_sin * sin(2πt/T)
    // ==================================================================
    {
        std::cout << "Test 1: beta_true_ matches c_sin * sin(2πt/T)\n";

        auto dg = DataGenerator::make_default();
        dg.generate();

        ResponseGenerator::Config cfg;
        cfg.c_sin = 7.0;
        cfg.beta0 = -2.5;

        ResponseGenerator rg(dg.X(), T, cfg);

        const auto& beta = rg.beta_true();
        const auto& t    = dg.t();
        // double T = dg.T;

        bool ok = true;
        for (int j = 0; j < beta.size(); ++j) {
            double t_j = t(j);
            double expected = 7.0 * std::sin(2.0 * std::numbers::pi * t_j / T);
            if (std::abs(beta(j) - expected) > 1e-12) {
                ok = false;
                break;
            }
        }

        if (ok) {
            std::cout << "  PASS: Coefficient function beta(t) is exact\n";
        } else {
            std::cerr << "  FAIL: beta_true_ does not match theoretical sin wave\n";
            all_ok = false;
        }
    }

    std::cout << "\n";

    // ==================================================================
    // Test 2: linear_predictor = X * beta_true_ + beta0
    // ==================================================================
    {
        std::cout << "Test 2: linear_predictor = X * beta_true_ + beta0\n";

        auto dg = DataGenerator::make_default();
        dg.generate();

        ResponseGenerator rg(dg.X(), T, ResponseGenerator::Config{.beta0 = 1.5, .c_sin = 3.0});
        rg.generate();

        Eigen::VectorXd manual_eta = dg.X() * rg.beta_true();
        manual_eta.array() += 1.5;

        double diff = (rg.linear_predictor() - manual_eta).norm();

        std::cout << "   ||computed η - manual η|| = " << diff << "\n";

        if (diff < 1e-12) {
            std::cout << "  PASS: Linear predictor computed correctly\n";
        } else {
            std::cerr << "  FAIL: Linear predictor does not match X*beta + beta0\n";
            all_ok = false;
        }
    }

    std::cout << "\n";

    // ==================================================================
    // Test 3: Noiseless case → y == linear_predictor
    // ==================================================================
    {
        std::cout << "Test 3: Noiseless response equals linear predictor\n";

        auto dg = DataGenerator::make_default();
        dg.generate();

        ResponseGenerator::Config cfg;
        cfg.add_noise = false;

        ResponseGenerator rg(dg.X(), T, cfg);
        rg.generate();

        double diff = (rg.y() - rg.linear_predictor()).norm();

        if (diff < 1e-14) {
            std::cout << "  PASS: Noiseless y exactly equals linear predictor\n";
        } else {
            std::cerr << "  FAIL: y differs from linear predictor in noiseless case\n";
            all_ok = false;
        }
    }

    std::cout << "\n";

    // ==================================================================
    // Test 4: Theoretical value: η_i ≈ beta0 + c_sin * A_i
    // ==================================================================
    // {
    //     std::cout << "Test 4: η_i ≈ beta0 + c_sin * A_i (due to perfect sine alignment)\n";

    //     auto dg = DataGenerator::make_default();
    //     dg.generate();

    //     ResponseGenerator rg(dg.X(), T, ResponseGenerator::Config{.c_sin = 4.0, .beta0 = -1.0});
    //     rg.generate();

    //     const auto& A = dg.amplitudes();
    //     Eigen::VectorXd expected_eta = -1.0 + 4.0 * A;

    //     double rmse = std::sqrt( (rg.linear_predictor() - expected_eta).squaredNorm() / A.size() );

    //     std::cout << "   RMSE from theoretical η = " << rmse << "\n";

    //     if (rmse < 1e-10) {
    //         std::cout << "  PASS: Linear predictor matches theoretical c_sin·A + beta0\n";
    //     } else {
    //         std::cerr << "  FAIL: Large deviation from theoretical inner product\n";
    //         all_ok = false;
    //     }
    // }

    // std::cout << "\n";

    // ==================================================================
    // Test 5: Noise has mean ≈ 0 and correct std
    // ==================================================================
    {
        std::cout << "Test 5: Added noise has correct statistics\n";

        auto dg = DataGenerator::make_default();
        dg.generate();

        ResponseGenerator::Config cfg;
        cfg.add_noise = true;
        cfg.noise_std = 0.15;
        cfg.seed = 424242;

        ResponseGenerator rg(dg.X(), T, cfg);
        rg.generate();

        Eigen::VectorXd noise = rg.y() - rg.linear_predictor();

        double mean = noise.mean();
        double std  = std::sqrt( noise.squaredNorm() / noise.size() );

        std::cout << "   Noise mean = " << mean << "\n";
        std::cout << "   Noise std  = " << std  << " (target 0.150000)\n";

        if (std::abs(mean) < 0.02 && std::abs(std - 0.15) < 0.01) {
            std::cout << "  PASS: Noise statistics correct\n";
        } else {
            std::cerr << "  FAIL: Noise mean or std out of tolerance\n";
            all_ok = false;
        }
    }

    std::cout << "\n";

    // ==================================================================
    // Test 6: Reproducibility with same seed
    // ==================================================================
    {
        std::cout << "Test 6: Same seed → identical response\n";

        auto dg = DataGenerator::make_default();
        dg.generate();

        ResponseGenerator::Config cfg{.seed = 987654};

        ResponseGenerator rg1(dg.X(), T, cfg);
        ResponseGenerator rg2(dg.X(), T, cfg);

        rg1.generate();
        rg2.generate();

        double diff = (rg1.y() - rg2.y()).norm();

        if (diff < 1e-12) {
            std::cout << "  PASS: Same seed produces identical y\n";
        } else {
            std::cerr << "  FAIL: Different y with same seed\n";
            all_ok = false;
        }
    }

    // ==================================================================
    // Final result
    // ==================================================================
    std::cout << "\n";
    if (all_ok) {
        std::cout << "ALL TESTS PASSED!\n\n";
        return 0;
    } else {
        std::cout << "SOME TESTS FAILED!\n\n";
        return 1;
    }
}
