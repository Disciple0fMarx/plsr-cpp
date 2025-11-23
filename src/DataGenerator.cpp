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
#include "DataGenerator.hpp"
#include <numbers>  // std::numbers::pi_v
#include <cmath>


DataGenerator::DataGenerator(const Config& config)
	: config_(config),
	  rng_(config.seed),
	  noise_dist_(0.0, config.noise_std)
{
	X_.resize(config_.N, config_.P);
	t_.resize(config_.P);
	A_.resize(config_.N);
}


void DataGenerator::generate()
{
	generate_time_points();
	generate_amplitudes();
	generate_sine_waves();
	add_noise();
}


void DataGenerator::generate_time_points()
{
	const double dt = config_.T / (config_.P - 1);
	for (int j = 0; j < config_.P; ++j) {
		t_(j) = j * dt;
	}
}


void DataGenerator::generate_amplitudes()
{
	std::uniform_real_distribution<double> dist(0.5, 2.5);
	for (int i = 0; i < config_.N; ++i) {
		A_(i) = dist(rng_);
	}
}


void DataGenerator::generate_sine_waves()
{
	const double pi = std::numbers::pi_v<double>;
	const double omega = 2.0 * pi / config_.T;

	for (int i = 0; i < config_.N; ++i) {
		const double amp = A_(i);
		for (int j = 0; j < config_.P; ++j) {
			X_(i, j) = amp * std::sin(omega * t_(j));
		}
	}
}


void DataGenerator::add_noise()
{
	if (!config_.add_noise) return;

	for (int i = 0; i < config_.N; ++i) {
		for (int j = 0; j < config_.P; ++j) {
			X_(i, j) += noise_dist_(rng_);
		}
	}
}

