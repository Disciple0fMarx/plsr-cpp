#include <iostream>
#include <iomanip>
#include <Eigen/Dense>
#include <cmath>
// #include <numbers>

#include "../include/PLSR.hpp"
#include "../include/DataGenerator.hpp"
#include "../include/ResponseGenerator.hpp"

int main() {
    std::cout << std::fixed << std::setprecision(12);
    std::cout << "=== PLS-NIPALS Functional Regression Tests ===\n\n";

    bool all_ok = true;
    const double T = 1.0;

    // Helper: absolute correlation (ignores sign flip)
    auto abs_corr = [](const Eigen::VectorXd& a, const Eigen::VectorXd& b) -> double {
        return std::abs(a.dot(b) / (a.norm() * b.norm() + 1e-15));
    };

    // ==================================================================
    // Test 1: First PLS component recovers the response y
    // ==================================================================
    {
        std::cout << "Test 1: First component recovers response y\n";

        auto dg = DataGenerator::make_default();
        dg.generate();

        auto rg = ResponseGenerator::make_default(dg.X(), T);
        rg.generate();

        PLSR_Result pls = pls_nipals(dg.X(), rg.y(), 5, 1);

        if (pls.T.cols() < 1) {
            std::cerr << "  FAIL: No components extracted!\n";
            all_ok = false;
        } else {
            Eigen::VectorXd beta_pls = pls.W * (pls.P.transpose() * pls.W).ldlt().solve(pls.q);
            Eigen::VectorXd y_pred = dg.X() * beta_pls;

            std::cout << "  [*] info: Number of extracted components: " << pls.T.cols() << "\n";

            double y_mean = rg.y().mean();
	        Eigen::VectorXd yc = rg.y().array() - y_mean;
            double y_explained = pls.b(0) * pls.b(0) * pls.T.col(0).squaredNorm();
            double y_total_var = yc.squaredNorm();

            std::cout << "            Component explains "
                      << (y_explained / y_total_var * 100.0)
                      << "% of y\n";

            double corr_beta = abs_corr(beta_pls, rg.beta_true());
            double corr_y_noisy = abs_corr(y_pred, rg.y());
            double corr_y_pure = abs_corr(y_pred, rg.linear_predictor());

            std::cout << "   |β_PLS ⋅ β_true| correlation = " << corr_beta << "\n";
            std::cout << "   |y_pred ⋅ y_true| correlation (noisy) = " << corr_y_noisy << "\n";
            std::cout << "   |y_pred ⋅ y_true| correlation (pure) = " << corr_y_pure << "\n";

            if (corr_beta < 0.98 || corr_y_noisy < 0.98 || corr_y_pure < 0.98) {
                std::cerr << "  FAIL: First component does not recover response (corr < 0.98)\n";
                all_ok = false;
            } else {
                std::cout << "  PASS: Excellent recovery of response\n";
            }
        }
    }

    std::cout << "\n";

    // ==================================================================
    // Test 2: Weights follow true coefficient function sin(2πt)
    // ==================================================================
    {
        std::cout << "Test 2: Weights w₁ match true β(t) = sin(2πt)\n";

        auto dg = DataGenerator::make_default();
        dg.generate();
        auto rg = ResponseGenerator::make_default(dg.X(), T);
        rg.generate();

        PLSR_Result pls = pls_nipals(dg.X(), rg.y(), 3, 1e-12);

        if (pls.W.cols() == 0) {
            std::cerr << "  FAIL: No weights computed\n";
            all_ok = false;
        } else {
            Eigen::VectorXd true_beta = (2.0 * std::numbers::pi * dg.t().array()).sin();
            double corr = abs_corr(pls.W.col(0), true_beta);

            std::cout << "   |w₁ ⋅ sin(2πt)| correlation = " << corr << "\n";

            if (corr < 0.90) {
                std::cerr << "  FAIL: Weights do not resemble true sinusoidal coefficient\n";
                all_ok = false;
            } else {
                std::cout << "  PASS: Weights correctly identify active time region\n";
            }
        }
    }

    std::cout << "\n";

    // ==================================================================
    // Test 3: Scores are orthogonal (core NIPALS property)
    // ==================================================================
    {
        std::cout << "Test 3: Successive score vectors t are orthogonal\n";

        auto dg = DataGenerator::make_default();
        dg.generate();
        auto rg = ResponseGenerator::make_default(dg.X(), T);
        rg.generate();

        PLSR_Result pls = pls_nipals(dg.X(), rg.y(), 10, 1e-12);

        bool ortho_ok = true;
        for (int i = 0; i < pls.T.cols(); ++i) {
            for (int j = i + 1; j < pls.T.cols(); ++j) {
                double dot = pls.T.col(i).dot(pls.T.col(j));
                if (std::abs(dot) > 1e-10) {
                    ortho_ok = false;
                    std::cerr << "  FAIL: t" << i+1 << " and t" << j+1
                              << " not orthogonal (dot = " << dot << ")\n";
                }
            }
        }

        if (pls.T.cols() >= 2 && ortho_ok) {
            std::cout << "  PASS: All " << pls.T.cols() << " scores are orthogonal\n";
        } else if (pls.T.cols() < 2) {
            std::cout << "  SKIP: Only " << pls.T.cols() << " component(s) extracted\n";
        } else {
            all_ok = false;
        }
    }

    std::cout << "\n";

    // ==================================================================
    // Test 4: Perfect recovery in noise-free case (rank-1 signal)
    // ==================================================================
    {
        std::cout << "Test 4: Perfect recovery when no noise in X or y\n";

        DataGenerator::Config dg_cfg;
        dg_cfg.add_noise = false;
        dg_cfg.seed = 42;

        ResponseGenerator::Config rg_cfg;
        rg_cfg.add_noise = false;
        rg_cfg.seed = 123;

        DataGenerator dg(dg_cfg);
        dg.generate();
        ResponseGenerator rg(dg.X(), T, rg_cfg);
        rg.generate();

        PLSR_Result pls = pls_nipals(dg.X(), rg.y(), 3, 1);

        if (pls.T.cols() == 0) {
            std::cerr << "  FAIL: No components extracted in clean case\n";
            all_ok = false;
        } else {
            Eigen::VectorXd beta_pls = pls.W * (pls.P.transpose() * pls.W).ldlt().solve(pls.q);
            Eigen::VectorXd y_pred = dg.X() * beta_pls;

            // double corr_beta = abs_corr(beta_pls, rg.beta_true());
            double corr_y = abs_corr(y_pred, rg.y());

            // double corr = abs_corr(pls.T.col(0), dg.amplitudes());
            std::cout << "   Correlation with true y = " << corr_y << "\n";

            if (corr_y > 0.9999) {
                std::cout << "  PASS: Near-perfect recovery in noise-free setting\n";
            } else {
                std::cerr << "  FAIL: Did not recover y perfectly (corr = " << corr_y << ")\n";
                all_ok = false;
            }
        }
    }

    std::cout << "\n";

    // ==================================================================
    // Test 5: Prediction error decreases monotonically
    // ==================================================================
    {
        std::cout << "Test 5: Prediction error decreases with more components\n";

        auto dg = DataGenerator::make_default();
        dg.generate();
        auto rg = ResponseGenerator::make_default(dg.X(), T);
        rg.generate();

        double prev_mse = 1e9;
        bool decreasing = true;

        for (int nc = 1; nc <= 5; ++nc) {
            PLSR_Result pls = pls_nipals(dg.X(), rg.y(), nc, 1e-12);

            Eigen::VectorXd y_pred = Eigen::VectorXd::Zero(rg.y().size());
            for (int k = 0; k < pls.T.cols(); ++k) {
                y_pred += pls.T.col(k) * pls.b(k);
            }

            // Center both
            double y_mean = rg.y().mean();
            Eigen::VectorXd yc = rg.y().array() - y_mean;
            Eigen::VectorXd yc_pred = y_pred.array() - y_mean;

            double mse = (yc - yc_pred).squaredNorm() / yc.size();

            std::cout << "   " << std::setw(2) << nc << " component(s) → MSE = " << mse << "\n";

            if (mse > prev_mse + 1e-12) {
                decreasing = false;
            }
            prev_mse = mse;
        }

        if (decreasing) {
            std::cout << "  PASS: MSE decreases monotonically\n";
        } else {
            std::cerr << "  FAIL: MSE increased when adding components\n";
            all_ok = false;
        }
    }

    // ==================================================================
    // Final verdict
    // ==================================================================
    std::cout << "\n";
    if (all_ok) {
        std::cout << "ALL TESTS PASSED!\n";
        return 0;
    } else {
        std::cout << "SOME TESTS FAILED!\n";
        return 1;
    }
}
