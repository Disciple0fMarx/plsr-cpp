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
#include "bspline.hpp"
#include <algorithm>
#include <stdexcept>

BSplineRegressor::BSplineRegressor(int nknots_interior, int degree)
    : degree_(degree)
{
    if (nknots_interior < 1) throw std::invalid_argument("Need at least 1 interior knot");
    if (degree < 1)          throw std::invalid_argument("Degree must be >= 1");
}

static double bspline_basis(double x, int i, int k, const std::vector<double>& knots)
{
    // Cox-de Boor recursion (k = degree+1)
    if (k == 1) {
        return (x >= knots[i] && x < knots[i+1]) ? 1.0 : 0.0;
    }
    double denom1 = knots[i+k-1] - knots[i];
    double denom2 = knots[i+k]   - knots[i+1];

    double term1 = (denom1 > 0) ? (x - knots[i])     / denom1 * bspline_basis(x, i,     k-1, knots) : 0.0;
    double term2 = (denom2 > 0) ? (knots[i+k] - x) / denom2 * bspline_basis(x, i + 1, k-1, knots) : 0.0;

    return term1 + term2;
}

void BSplineRegressor::build_design_matrix(const Eigen::VectorXd& t)
{
    const int P = t.size();
    const int n_basis = knots_.size() - degree_ - 1;

    Phi_.resize(P, 1 + n_basis);  // +1 for intercept
    Phi_.col(0).setOnes();

    for (int j = 0; j < P; ++j) {
        double tj = t(j);
        for (int b = 0; b < n_basis; ++b) {
            Phi_(j, 1 + b) = bspline_basis(tj, b, degree_ + 1, knots_);
        }
    }
}

void BSplineRegressor::fit_y(const Eigen::MatrixXd& X,
                             const Eigen::VectorXd& y,
                             const Eigen::VectorXd& t)
{
    // Uniform knots with clamping
    const int n_interior = 15;
    std::vector<double> interior;
    for (int i = 1; i <= n_interior; ++i)
        interior.push_back(0.0 + i * 1.0 / (n_interior + 1));

    knots_.clear();
    for (int i = 0; i < degree_; ++i) knots_.push_back(0.0);
    knots_.insert(knots_.end(), interior.begin(), interior.end());
    for (int i = 0; i < degree_; ++i) knots_.push_back(1.0);

    build_design_matrix(t);

    Eigen::MatrixXd design = X * Phi_;
    Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> cod(design);
    coeffs_ = cod.solve(y);
}

Eigen::VectorXd BSplineRegressor::predict_beta(const Eigen::VectorXd& t) const
{
    if (coeffs_.size() == 0) throw std::runtime_error("Not fitted yet");
    Eigen::MatrixXd Phi_t(t.size(), Phi_.cols());
    Phi_t.col(0).setOnes();
    for (int j = 0; j < t.size(); ++j) {
        for (int b = 0; b < Phi_.cols()-1; ++b) {
            Phi_t(j, 1+b) = bspline_basis(t(j), b, degree_+1, knots_);
        }
    }
    return Phi_t * coeffs_;
}

Eigen::VectorXd BSplineRegressor::predict_y(const Eigen::MatrixXd& X,
                                            const Eigen::VectorXd& t) const
{
    return X * predict_beta(t);
}
