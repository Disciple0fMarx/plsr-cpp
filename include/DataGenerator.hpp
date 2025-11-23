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

#include <vector>
#include <random>
#include <Eigen/Dense>


/**
 * @class DataGenerator
 * @brief Generates synthetic data for PLSR demonstration with sinusoidal predictors.
 * 
 * Each row i of X is a discrete sampling of:
 *     X_i(t) = A_i · sin(2πf t + φ) + ε
 * over one full period of a 1 Hz sine wave.
 *
 * where:
 *   - A_i ∈ [0.5, 2.5] is a latent amplitude (drawn uniformly)
 *   - Frequency is fixed at f = 1 Hz (one full period over t ∈ [0,1))
 *   - Phase is fixed at φ = 0
 *
 * Optional Gaussian noise can be added to simulate real-world measurements.
 *
 * @note Time points t_j are uniformly spaced in the closed interval [0, 1],
 * so t_0 = 0 and t_{P-1} = 1 (inclusive). This ensures exactly one
 * complete oscillation is observed.
 *
 * @authors Dhiaa Eddine Bahri <dhya.bahri@proton.me>
 *          Malek Rihani <malek.rihani090@gmail.com>
 *
 * @date 2025
 */
class DataGenerator {
	public:
		/// Configuration parameters for data generation
		struct Config {
			int     N = 1000;          ///< Number of samples (rows)
			int     P = 500;           ///< Number of variables/time points (columns)
			double  T = 1.0;           ///< Observation window in seconds (one period)
			bool    add_noise = true;  ///< Whether to add i.i.d. Gaussian noise
			double  noise_std = 0.1;   ///< Standard deviation of the noise
			unsigned int seed = 42;    ///< RNG seed for reproducibility
		};

		/**
		 * @brief Construct a DataGenerator with given configuration.
		 * @param config Configuration struct (default values used if not provided)
		 */
		explicit DataGenerator(const Config& config);

		/**
		 * @brief Factory function returning a generator with default settings
		 * @return DataGenerator with standard configuration
		 */
		static DataGenerator make_default() {
			return DataGenerator(Config{});
		}

		/**
		 * @brief Generate (or regenerate) the entire synthetic dataset
		 *
		 * Populates X_, t_, and A_. Safe to call multiple times.
		 */
		void generate();

		// Accessors

		/** @brief Returns the generated predictor matrix X (N × P) */
		const Eigen::MatrixXd& X() const noexcept { return X_; }

		/** @brief Returns the time vector t (P × 1) (uniform grid on [0,1]) */
		const Eigen::VectorXd& t() const noexcept { return t_; }

		/** @brief Returns the vector of true latent amplitudes A (N × 1) (ground truth for response generation) */
		const Eigen::VectorXd& amplitudes() const noexcept { return A_; }
	
	private:
		Config config_;

		Eigen::MatrixXd X_;  ///< Predictor matrix (N × P)
		Eigen::VectorXd t_;  ///< Time points (P)
		Eigen::VectorXd A_;  ///< True latent amplitudes (N)
		
		std::mt19937 rng_;   ///< Mersenne Twister RNG
		std::normal_distribution<double> noise_dist_;   ///< Noise distribution

		/// Generate uniformly spaced time points over [0, 1]
		void generate_time_points();

		/// Sample latent amplitudes A_i ~ Uniform[0.5, 2.5]
		void generate_amplitudes();

		/// Fill X with clean sine waves X_i,j = A_i * sin(2π t_j)
		void generate_sine_waves();

		/// Add i.i.d. Gaussian noise if requested
		void add_noise();
};

