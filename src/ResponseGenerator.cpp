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
#include "ResponseGenerator.hpp"
#include <cmath>


ResponseGenerator::ResponseGenerator(const Eigen::MatrixXd& X,
									 double T,
				    				 const Config& config)
	: config_(config),
	  X_(X),
	  T_(T),
	  rng_(config.seed),
	  noise_dist_(0.0, config.noise_std)
{
	const int P = X_.cols();
	beta_true_.resize(P);

	const double dt = T_ / (P - 1);
	const double pi = std::numbers::pi_v<double>;
	const double omega = 2.0 * pi / T;
	for (int j = 0; j < P; ++j) {
		double t_j = j * dt;
		beta_true_(j) = config_.c_sin * std::sin(omega * t_j); 
	}

	lin_pred_.resize(X_.rows());
	y_.resize(X_.rows());
}


void ResponseGenerator::generate()
{
	lin_pred_ = X_ * beta_true_;
	lin_pred_.array() += config_.beta0;

	y_ = lin_pred_;

	if (config_.add_noise) {
		for (int i = 0; i < y_.size(); ++i) {
			y_(i) += noise_dist_(rng_);
		}
	}
}
