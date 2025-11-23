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


ResponseGenerator::ResponseGenerator(const Eigen::VectorXd& amplitudes,
				     const Config& config)
	: config_(config),
	  A_(amplitudes),
	  rng_(config.seed),
	  noise_dist_(0.0, config.noise_std)
{
	y_true_.resize(A_.size());
	y_.resize(A_.size());
}


void ResponseGenerator::generate()
{
	const double b0 = config_.beta0;
	const double b1 = config_.beta1;

	for (int i = 0; i < A_.size(); ++i) {
		double z = b0 + b1 * A_(i);
		y_true_(i) = 1.0 / (1.0 + std::exp(-z));
	}

	y_ = y_true_;

	if (config_.add_noise) {
		for (int i = 0; i < y_.size(); ++i) {
			y_(i) += noise_dist_(rng_);
		}
	}
}

