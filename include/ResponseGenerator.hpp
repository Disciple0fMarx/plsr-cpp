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
 * @brief Generates a scalar response y from functional predictors X(t)
 *
 * The true data-generating model is:
 *
 *     y_i = β₀ + ∫ β(t) X_i(t) dt + δ_i   (linear predictor)
 * 
 * Programmatically, this translates to:
 *
 *     y_i = β₀ + ∑ β(t_j) X_i(t_j) + δ_i
 *
 * where:
 *   • t_j = j·Δt,  Δt = T/(P-1)   (equidistant grid)
 *   • β(t) = c ⋅ sin(2π t / T)    (true coefficient function (pure sine wave))
 *   • δ_i is Gaussian noise      (optional)
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
			double c_sin = 5.0;       ///< Amplitude of the sinusoidal coefficient function β(t) = c_sin * sin(2πt)
			bool add_noise = true;    ///< Add Gaussian noise to the response?
			double noise_std = 0.05;  ///< Standard deviation of response noise
			unsigned int seed = 999;  ///< RNG seed for reproducibility

		};

		/**
		 * @brief Construct generator
		 * @param X The data matrix
		 * @param T The period of X's sine waves
		 * @param config     Model and noise parameters
		 */
		explicit ResponseGenerator(const Eigen::MatrixXd& X,
								   double T,
					   			   const Config& config);

		/**
		 * @brief Factory method using the default 
		 * @param X The data matrix
		 * @param T The period of X's sine waves
		 * @return Configured ResponseGenerator
		 */
		static ResponseGenerator make_default(const Eigen::MatrixXd& X, double T) {
			return ResponseGenerator(X, T, Config{});
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
		const Eigen::VectorXd& beta_true() const noexcept { return beta_true_; }  // noiseless

		const Eigen::VectorXd& linear_predictor() const noexcept { return lin_pred_; }

	private:
		Config config_;
		const Eigen::MatrixXd& X_;
		double T_;                                 // ← now stored

		Eigen::VectorXd beta_true_;
		Eigen::VectorXd lin_pred_;
		Eigen::VectorXd y_;
		
		std::mt19937 rng_;          ///< Random number engine
		std::normal_distribution<double> noise_dist_;  ///< noise distribution
};
