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
#include "../include/ResponseGenerator.hpp"


int main() {
	std::cout << std::fixed << std::setprecision(12);
	std::cout << "=== ResponseGenerator Unit Tests ===\n\n";

	bool all_ok = true;

	// Create fake amplitudes (monotonically increasing)
	Eigen::VectorXd A = Eigen::VectorXd::LinSpaced(200, 0.4, 2.6);

	// Test 1: No noise, steep sigmoid → perfect monotonocity and [0, 1] range
	{
		ResponseGenerator::Config cfg;
		cfg.beta0 = -6.0;  // center at A = 1.5
		cfg.beta1 = 8.0;   // steep transition
		cfg.add_noise = false;

		ResponseGenerator rg(A, cfg);
		rg.generate();

		bool monotonic = true;
		for (int i = 1; i < rg.y().size(); ++i) {
			if (rg.y()(i) <= rg.y()(i-1)) monotonic = false;
		}

		bool in_range = (rg.y().array() >= 0.0).all() && (rg.y().array() <= 1.0).all();

		std::cout << "No noise: monotonic ........ " << (monotonic ? "PASS" : "FAIL") << "\n";
		std::cout << "No noise: in [0, 1] ........ " << (in_range ? "PASS" : "FAIL") << "\n";
		all_ok &= (monotonic & in_range);
	}

	// Test 2: With noise → mean(y) ≈ mean(y_true), std reasonable
	{
		ResponseGenerator rg = ResponseGenerator::make_default(A);
		rg.generate();

		double mean_diff = (rg.y() - rg.y_true()).mean();
		double std_emp = std::sqrt((rg.y() - rg.y_true()).squaredNorm() / rg.y().size());

		bool ok = std::abs(mean_diff) < 0.01 && std::abs(std_emp - 0.05) < 0.02;

		std::cout << "With noise: std ≈ 0.05 ..... " << (ok ? "PASS" : "FAIL") << " (actual std = " << std_emp << ")\n";
		all_ok &= ok;
	}

	std::cout << "\n";
	std::cout << (all_ok ? "ALL TESTS PASSED!\n" : "SOME TESTS FAILED!\n");
	return all_ok ? 0 : 1;
}

