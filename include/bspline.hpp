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

#include <Eigen/Dense>
#include <vector>


/**
 * @class BSplineRegressor
 * @brief Fits β(t) using a B-spline basis (cubic by default)
 *        Perfect for localized, non-periodic, or irregular coefficient functions.
 *
 * @authors Dhiaa Eddine Bahri <dhya.bahri@proton.me>
 *          Malek Rihani <malek.rihani090@gmail.com>
 *
 * @date 2025
 */
class BSplineRegressor
{
public:
    BSplineRegressor(int nknots_interior = 15, int degree = 3);

    void fit_y(const Eigen::MatrixXd& X,
               const Eigen::VectorXd& y,
               const Eigen::VectorXd& t);

    Eigen::VectorXd predict_beta(const Eigen::VectorXd& t) const;
    Eigen::VectorXd predict_y(const Eigen::MatrixXd& X,
                              const Eigen::VectorXd& t) const;

    // Accessors
    double intercept() const { return coeffs_(0); }
    const Eigen::VectorXd& coefficients() const { return coeffs_; }
    int num_knots() const { return knots_.size(); }

private:
    void build_design_matrix(const Eigen::VectorXd& t);

    int degree_;
    std::vector<double> knots_;
    Eigen::MatrixXd Phi_;      // [P x (nknots + degree)]
    Eigen::VectorXd coeffs_;
};
