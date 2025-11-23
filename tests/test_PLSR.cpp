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
#include <numbers>
#include "../include/DataGenerator.hpp"
#include "../include/ResponseGenerator.hpp"
#include "../include/PLSR.hpp"


int main() {
	std::cout << std::fixed << std::setprecision(12);
	std::cout << "=== PLSR Unit Tests ===\n\n";

	bool all_ok = true;

	// Small reproducible dataset
	DataGenerator::Config cfg;
	cfg.N = 300;
	cfg.P = 120;
	cfg.add_noise = true;
	cfg.noise_std = 0.03;
	cfg.seed = 123;
	auto gen = DataGenerator(cfg);
	gen.generate();

	ResponseGenerator resp = ResponseGenerator::make_default(gen.amplitudes());
	resp.generate();

	auto pls = pls_nipals(gen.X(), resp.y(), 1, 1e-12);
	
	// Test 1: First score recovers true latent amplitude A
	double corr_score_A = std::abs(
		pls.T.col(0).normalized().dot(gen.amplitudes().normalized())	
	);

	// Test 2: First loading recovers sine shape
	Eigen::VectorXd true_sine = (2.0 * std::numbers::pi * gen.t().array()).sin();
	double corr_loading = std::abs(
		pls.P.col(0).normalized().dot(true_sine.normalized())
	);

	// Test 3: Prediction with 1 component is already good (R² > 0.85 even with sigmoid)
	double corr_y_t = pls.T.col(0).dot(resp.y()) / (pls.T.col(0).norm() * resp.y().norm());
	double R2 = corr_y_t * corr_y_t;
	
	// double ss_res = (resp.y() - y_pred_1).squaredNorm();

	// Eigen::VectorXd y_centered = resp.y().array() - resp.y().mean();
	// double ss_tot = y_centered.squaredNorm();
	// double ss_tot = (resp.y().array() - resp.y().mean()).square().sum();
	// double R2_1 = 1.0 - ss_res / ss_tot;

	std::cout << "Score ↔ True A correlation: " << corr_score_A << " ... "
		  << (corr_score_A > 0.998 ? "PASS" : "FAIL") << "\n";
	std::cout << "Loading ↔ sine correlation: " << corr_loading << " ... "
		  << (corr_loading > 0.995 ? "PASS" : "FAIL") << "\n";
	std::cout << "R² (1 component):           " << R2 << " ... "
		  << (R2 > 0.85 ? "PASS" : "FAIL") << "\n";

	all_ok &= (corr_score_A > 0.998) && (corr_loading > 0.995) && (R2 > 0.85);

	std::cout << "\n";
	std::cout << (all_ok ? "ALL TESTS PASSED!\n" : "SOME TESTS FAILED!\n");
	return all_ok ? 0 : 1;
}

