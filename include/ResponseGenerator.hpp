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

#include <random>
#include <numbers>
#include <Eigen/Dense>


/**
 * @class ResponseGenerator
 * @brief Generates a scalar response y from latent amplitudes A
 * via a nonlinear link function.
 *
 * True model: y_i = σ(β₀ + β₁ · A_i) + ε
 * where σ(x) = 1/(1+exp(-x)) is the logistic sigmoid.
 *
 * @authors Dhiaa Eddine Bahri <dhya.bahri@proton.me>
 *          Malek Rihani <malek.rihani090@gmail.com>
 *
 * @date 2025
 */
class ResponseGenerator {
	public:
		struct Config {
			double beta0 = 0.0;       ///< Intercept of the linear predictor
			double beta1 = 4.0;       ///< Slope (controls steepness)
			bool add_noise = true;    ///< Add Gaussian noise to the response?
			double noise_std = 0.05;  ///< Standard deviation of response noise
			unsigned int seed = 999;  ///< RNG seed for reproducibility

		};

		/**
		 * @brief Construct generator from true latent amplitudes and model configuration
		 * @param amplitudes Reference to the vector of true A_i (from DataGenerator)
		 * @param config     Model and noise parameters
		 */
		explicit ResponseGenerator(const Eigen::VectorXd& amplitudes,
					   			   const Config& config);

		/**
		 * @brief Factory method using sensible default parameters (β₀=0, β₁=4, mild noise)
		 * @param amplitudes True latent amplitudes A
		 * @return Configured ResponseGenerator
		 */
		static ResponseGenerator make_default(const Eigen::VectorXd& amplitudes) {
			return ResponseGenerator(amplitudes, Config{});
		};

		/**
		 * @brief Generate (or regenerate) the response vector y
		 *
		 * Applies the sigmoid transformation and optionally adds noise.
		 * Safe to call multiple times with the same seed → identical results.
		 */
		void generate();

		/** @brief Observed response vector y (N × 1) (with noise, if enabled) */
		const Eigen::VectorXd& y() const noexcept { return y_; }

		/** @brief Noise-free "true" response σ(β₀ + β₁ A) (N × 1) */
		const Eigen::VectorXd& y_true() const noexcept { return y_true_; }  // noiseless

	private:
		Config config_;
		const Eigen::VectorXd& A_;  ///< reference to true latent amplitudes
		
		Eigen::VectorXd y_true_;    ///< sigmoid output without noise
		Eigen::VectorXd y_;         ///< final observed response (may include noise)
		
		std::mt19937 rng_;          ///< Random number engine
		std::normal_distribution<double> noise_dist_;  ///< noise distribution
};
