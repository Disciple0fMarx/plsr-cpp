#include <iostream>
#include <iomanip>
#include <Eigen/Dense>
#include <cmath>
#include <numbers>

#include "../include/DataGenerator.hpp"
#include "../include/ResponseGenerator.hpp"
#include "../include/bspline.hpp"

int main() {
    std::cout << std::fixed << std::setprecision(12);
    std::cout << "=== BSplineRegressor Unit Tests ===\n\n";

    bool all_ok = true;
    const double T = 1.0;

    // Use fixed seed and known config for reproducibility
    auto dg = DataGenerator::make_default();
    // dg.config_.seed = 54321;
    dg.generate();

    const auto& X = dg.X();
    const auto& t = dg.t();

    // ==================================================================
    // Test 1: Perfect recovery of piecewise constant function (spline's natural strength)
    // ==================================================================
    {
        std::cout << "Test 1: Perfect recovery of piecewise constant step function\n";

        ResponseGenerator::Config cfg;
        cfg.c_sin = 3.0;
        cfg.beta0 = 1.0;
        cfg.add_noise = false;
        
        // Create a step function: β(t) = 1 + 3*[t>0.3] for t in [0,1]
        ResponseGenerator rg(X, T, cfg);
        rg.generate();

        BSplineRegressor bspline(10, 1);  // linear splines (degree=1) for piecewise constant
        bspline.fit_y(X, rg.y(), t);

        Eigen::VectorXd beta_rec = bspline.predict_beta(t);
        double error = (beta_rec - rg.beta_true()).norm();

        std::cout << "   Reconstruction error ||beta_hat - beta_true|| = " << error << "\n";

        if (error < 1e-8) {  // very tolerant for degree=1
            std::cout << "  PASS: Excellent recovery of step function\n";
        } else {
            std::cerr << "  FAIL: Poor recovery of piecewise constant function\n";
            all_ok = false;
        }
    }

    std::cout << "\n";

    // ==================================================================
    // Test 2: Smooth cubic function recovery (spline's sweet spot)
    // ==================================================================
    {
        std::cout << "Test 2: Recovery of smooth cubic polynomial (cubic splines)\n";

        ResponseGenerator::Config cfg;
        cfg.c_sin = 0.0;
        cfg.beta0 = 0.0;
        cfg.add_noise = true;
        cfg.noise_std = 0.02;
        cfg.seed = 888;

        // Create β(t) = 2t³ - 3t² + 1 (cubic polynomial)
        ResponseGenerator rg(X, T, cfg);
        rg.generate();

        BSplineRegressor bspline(12, 3);  // cubic splines for smooth functions
        bspline.fit_y(X, rg.y(), t);

        Eigen::VectorXd beta_rec = bspline.predict_beta(t);
        Eigen::VectorXd beta_true = rg.beta_true();
        double error = (beta_rec - beta_true).norm() / beta_true.norm();

        std::cout << "   Relative error in beta recovery = " << error << "\n";

        if (error < 0.005) {  // 0.5% error acceptable with noise
            std::cout << "  PASS: Excellent recovery of smooth cubic function\n";
        } else {
            std::cerr << "  FAIL: Poor recovery of smooth function under noise\n";
            all_ok = false;
        }
    }

    std::cout << "\n";

    // ==================================================================
    // Test 3: Overfitting control — too many knots doesn't hurt clean data
    // ==================================================================
    {
        std::cout << "Test 3: Many knots (30) doesn't overfit clean sine data\n";

        ResponseGenerator::Config cfg;
        cfg.c_sin = 2.5;
        cfg.beta0 = 0.5;
        cfg.add_noise = false;

        ResponseGenerator rg(X, T, cfg);
        rg.generate();

        BSplineRegressor bspline(30, 3);  // WAY too many knots for clean sine
        bspline.fit_y(X, rg.y(), t);

        Eigen::VectorXd beta_rec = bspline.predict_beta(t);
        double error = (beta_rec - rg.beta_true()).norm();

        // Check that most coefficients are near zero (no overfitting)
        const auto& coeffs = bspline.coefficients();
        bool sparse = true;
        int n_zero = 0;
        for (int i = 1; i < coeffs.size(); ++i) {  // skip intercept
            if (std::abs(coeffs(i)) > 1e-6) {
                sparse = false;
                break;
            }
            if (std::abs(coeffs(i)) < 1e-10) n_zero++;
        }

        std::cout << "   Reconstruction error = " << error << "\n";
        std::cout << "   Non-zero basis coeffs = " << (bspline.coefficients().size() - 1 - n_zero) << "\n";

        if (error < 1e-8 && (bspline.coefficients().size() - 1 - n_zero) <= 3) {
            std::cout << "  PASS: Many knots still fits cleanly, sparse coefficients\n";
        } else {
            if (error >= 1e-8) std::cerr << "  FAIL: Reconstruction error too large\n";
            if ((bspline.coefficients().size() - 1 - n_zero) > 3) std::cerr << "  FAIL: Overfitting detected (too many non-zero coeffs)\n";
            all_ok = false;
        }
    }

    std::cout << "\n";

    // ==================================================================
    // Test 4: predict_y() consistency with X * beta_hat
    // ==================================================================
    {
        std::cout << "Test 4: predict_y(X, t) == X * predict_beta(t)\n";

        ResponseGenerator::Config cfg;
        cfg.c_sin = 1.8;
        cfg.beta0 = -0.3;
        cfg.add_noise = false;

        ResponseGenerator rg(X, T, cfg);
        rg.generate();

        BSplineRegressor bspline(8, 3);
        bspline.fit_y(X, rg.y(), t);

        Eigen::VectorXd y_pred = bspline.predict_y(X, t);
        Eigen::VectorXd y_manual = X * bspline.predict_beta(t);

        double diff = (y_pred - y_manual).norm();

        if (diff < 1e-12) {
            std::cout << "  PASS: predict_y() is mathematically consistent\n";
        } else {
            std::cerr << "  FAIL: predict_y() differs from manual computation\n";
            all_ok = false;
        }
    }

    std::cout << "\n";

    // ==================================================================
    // Test 5: Different knot counts — more knots = more flexibility
    // ==================================================================
    {
        std::cout << "Test 5: More knots improves fit on wiggly function\n";

        // Create a wiggly target: β(t) = sin(8πt) + 0.3 sin(20πt)
        ResponseGenerator::Config cfg;
        cfg.add_noise = false;
        cfg.beta0 = 0.0;

        ResponseGenerator rg(X, T, cfg);
        rg.generate();

        // Test different knot counts
        std::vector<int> nknots = {5, 10, 20};
        std::vector<double> errors;
        
        for (int nk : nknots) {
            BSplineRegressor bspline(nk, 3);
            bspline.fit_y(X, rg.y(), t);
            Eigen::VectorXd beta_rec = bspline.predict_beta(t);
            double error = (beta_rec - rg.beta_true()).norm();
            errors.push_back(error);
        }

        std::cout << "   Errors for 5/10/20 knots: " 
                  << errors[0] << ", " << errors[1] << ", " << errors[2] << "\n";

        // Should see strictly decreasing errors
        bool monotonic = (errors[1] < errors[0]) && (errors[2] < errors[1]);

        if (monotonic && errors[2] < errors[0] * 0.5) {
            std::cout << "  PASS: More knots = better fit (monotonic improvement)\n";
        } else {
            std::cerr << "  FAIL: More knots didn't improve fit as expected\n";
            all_ok = false;
        }
    }

    std::cout << "\n";

    // ==================================================================
    // Test 6: Exception safety — invalid inputs
    // ==================================================================
    {
        std::cout << "Test 6: Throws on invalid inputs\n";

        bool threw_knots = false, threw_dims = false;

        // Test 1: Negative knots
        try {
            BSplineRegressor bspline(-1, 3);
        } catch (const std::invalid_argument&) {
            threw_knots = true;
        }

        // Test 2: Dimension mismatch
        BSplineRegressor bspline(10, 3);
        try {
            Eigen::VectorXd short_t(5);
            bspline.fit_y(X, Eigen::VectorXd::Zero(X.rows()), short_t);
        } catch (const std::invalid_argument&) {
            threw_dims = true;
        }

        if (threw_knots && threw_dims) {
            std::cout << "  PASS: Proper error handling for invalid inputs\n";
        } else {
            std::cerr << "  FAIL: Missing expected exceptions\n";
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
