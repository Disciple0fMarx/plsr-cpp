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
#include "../include/DataGenerator.hpp"
#include <cmath>
#include <numbers>


int main() {
	std::cout << std::fixed << std::setprecision(12);

	std::cout << "=== DataGenerator Unit Tests ===\n\n";

	// Test 1: Default config + basic properties
	
	DataGenerator gen1 = DataGenerator::make_default();
	gen1.generate();

	bool ok = true;

	if (gen1.X().rows() != 1000 || gen1.X().cols() != 500) {
		std::cerr << "FAIL: Wrong matrix dimensions\n";
		ok = false;
	}

	if (gen1.t().size() != 500) {
		std::cerr << "FAIL: Time vector wrong size\n";
		ok = false;
	}

	// Test 2: Time vector exactly [0, 1] inclusive
	
	double t0 = gen1.t()(0);
	double t_end = gen1.t()(gen1.t().size() - 1);
	double expected_end = 1.0;

	if (std::abs(t0 - 0.0) > 1e-15 || std::abs(t_end - expected_end) > 1e-12) {
		std::cerr << "FAIL: Time vector should be [0, 1] inclusive\n";
		std::cerr << "	t0 = " << t0 << ", t_end = " << t_end << "\n";
		ok = false;
	} else {
		std::cout << "PASS: Time vector spans [0, 1] inclusive\n";
	}

	// Test 3: Noiseless case - Perfect sine, rank 1, starts/ends at 0
	
	DataGenerator::Config cfg_clean;
	cfg_clean.add_noise = false;
	cfg_clean.N = 200;
	cfg_clean.P = 100;
	cfg_clean.seed = 999;

	DataGenerator gen_clean(cfg_clean);
	gen_clean.generate();

	const auto& X_clean = gen_clean.X();
	const auto& t = gen_clean.t();
	const auto& A = gen_clean.amplitudes();

	bool sine_ok = true;
	for (int i = 0; i < X_clean.rows(); ++i) {
		double a = A(i);
		if (std::abs(X_clean(i, 0)) > 1e-14) sine_ok = false;
		if (std::abs(X_clean(i, X_clean.cols() - 1)) > 1e-14) sine_ok = false;

		for (int j = 0; j < X_clean.cols(); ++j) {
			double expected = a * std::sin(2.0 * std::numbers::pi * t(j));
			if (std::abs(X_clean(i, j) - expected) > 1e-12) {
				sine_ok = false;
			}
		}
	}

	if (!sine_ok) {
		std::cerr << "FAIL: Noiseless sine waves are inaccurate\n";
		ok = false;
	} else {
		std::cout << "PASS: Noiseless sine waves are perfect\n";
	}

	// Rank should be exactly 1 (up to numerical precision)
    	Eigen::JacobiSVD<Eigen::MatrixXd> svd(X_clean, Eigen::ComputeThinU | Eigen::ComputeThinV);
    	int rank = (svd.singularValues().array() > 1e-10).count();

    	if (rank != 1) {
        	std::cerr << "FAIL: Noiseless X should be rank 1 (found rank " << rank << ")\n";
        	ok = false;
    	} else {
        	std::cout << "PASS: Noiseless X has exact rank 1\n";
    	}

    	// ──────────────────────────────────────────────────────────────
    	// Test 4: Reproducibility with same seed
    	// ──────────────────────────────────────────────────────────────
    	DataGenerator gen_a(cfg_clean);
    	gen_a.generate();
    	DataGenerator gen_b(cfg_clean);
    	gen_b.generate();

    	bool reproducible = (gen_a.X() - gen_b.X()).norm() < 1e-12;

    	if (!reproducible) {
        	std::cerr << "FAIL: Same seed did not produce identical data\n";
        	ok = false;
    	} else {
        	std::cout << "PASS: Same seed → identical data\n";
    	}

    	// ──────────────────────────────────────────────────────────────
    	// Test 5: Noise statistics (mean ≈ 0, std ≈ config)
    	// ──────────────────────────────────────────────────────────────
    	DataGenerator::Config cfg_noise;
    	cfg_noise.N = 1000;
    	cfg_noise.P = 500;
    	cfg_noise.add_noise = true;
    	cfg_noise.noise_std = 0.2;
    	cfg_noise.seed = 123;

    	DataGenerator gen_noise(cfg_noise);
    	gen_noise.generate();

    	// Clean signal reconstruction
    	Eigen::MatrixXd X_signal = gen_noise.amplitudes() *
        	(2.0 * std::numbers::pi * gen_noise.t().array()).sin().matrix().transpose();

    	Eigen::MatrixXd noise_matrix = gen_noise.X() - X_signal;

    	double noise_mean = noise_matrix.mean();
    	double noise_std  = std::sqrt(noise_matrix.squaredNorm() / noise_matrix.size());

    	std::cout << "Noise mean approximately " << noise_mean << " (expected ~0)\n";
    	std::cout << "Noise std  approximately " << noise_std  << " (target 0.200)\n";

    	if (std::abs(noise_mean) > 0.005 || std::abs(noise_std - 0.2) > 0.01) {
        	std::cerr << "FAIL: Noise statistics out of tolerance\n";
        	ok = false;
    	} else {
        	std::cout << "PASS: Noise has correct mean and std\n";
    	}

    	// ──────────────────────────────────────────────────────────────
    	// Final result
    	// ──────────────────────────────────────────────────────────────
    	std::cout << "\n";
    	if (ok) {
        	std::cout << "ALL TESTS PASSED!\n";
        	return 0;
    	} else {
        	std::cout << "SOME TESTS FAILED!\n";
        	return 1;
    	}
}

