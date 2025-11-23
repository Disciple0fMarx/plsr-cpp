# PLSR-CPP — Functional Partial Least Squares Regression in C++

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) [![C++20](https://img.shields.io/badge/C%2B%2B-20-blue)](https://en.cppreference.com/w/cpp/20) [![Eigen](https://img.shields.io/badge/Dependencies-Eigen3-green)](#) [![Platforms](https://img.shields.io/badge/Platforms-Linux%20•%20macOS%20•%20Windows-lightgrey)](#)

A clean, fast, and fully documented C++20 implementation of the **NIPALS PLS1 algorithm** applied to **functional data**, developed as part of the course *Statistics and Data Analysis* — M. Sc. in Data Science and Information Retrieval.

Includes realistic synthetic data generation (amplitude-modulated sine waves), a challenging **nonlinear response model** (sigmoid link), and a complete simulation study framework.

Perfect for teaching, research, benchmarking, or extending to real functional datasets (spectroscopy, chemometrics, FDA, etc.).

## Scientific Motivation

We generate functional predictors  

$$X_i(t_j) = A_i \cdot \sin(2\pi t_j) + \varepsilon_{ij}, \quad t_j \in [0,1], \quad i=1,\dots,N$$

with latent amplitude $A_i \sim \mathcal{U}[0.5, 2.5]$, and a scalar response  

$$y_i = \sigma(\beta_0 + \beta_1 A_i) + \delta_i$$  

where $\sigma$ is the logistic sigmoid.  
This creates a **strong but nonlinear relationship** that ordinary least squares and PCA fail to capture efficiently — while **PLSR recovers it with just 1–2 components**.

## Features

- Modern C++20 + Eigen3 (header-only dependency)
- Classic NIPALS PLS1 implementation (Wold et al., 1984)
- Reproducible synthetic functional data (B-spline/Fourier-ready structure)
- Nonlinear response via sigmoid link
- Full Doxygen documentation with LaTeX formulas
- Cross-platform: Linux • macOS • Windows (MSVC, Clang, GCC)
- Extremely simple build system (just a smart Makefile wrapper around CMake)

## Project Structure

```
plsr-cpp/
├── CMakeLists.txt
├── Makefile ← one-command magic
├── include/
│ ├── DataGenerator.hpp
│ ├── ResponseGenerator.hpp 
│ └── PLSR.hpp 
├── src/ ← implementations 
├── tests/ ← unit tests 
├── main.cpp ← full simulation demo 
├── results/ ← generated data & comparison 
├── LICENSE ← GPL-3.0 
└── README.md
```


## Build & Run (All Platforms)

### Prerequisites (same for Linux, macOS, Windows)

- A modern C++ compiler (GCC ≥ 10, Clang ≥ 10, MSVC ≥ 2019)
- CMake ≥ 3.16
- Eigen3 (header-only)

**Installation of Eigen3**

```bash
# Ubuntu/Debian
sudo apt install libeigen3-dev

# Arch Linux
sudo pacman -S eigen

# macOS (Homebrew)
brew install eigen

# Windows: use vcpkg
vcpkg install eigen3
# or Conan, or just drop the headers somewhere
```

### One-command workflow (Linux & macOS)

```bash
cd plsr-cpp
make go          # builds + runs the full demo
make tests       # runs all unit tests
```

Other useful targets:

```bash
make setup      # first-time configuration
make build      # fast incremental rebuild
make run        # run the demo (same as `make make go`)
make clean      # remove build artifacts
```

### Windows (Visual Studio or MinGW)

Open a terminal (PowerShell, CMD, or Git Bash) and run exactly the same `make` commands above if you have GNU Make installed.

**Alternative: pure CMake (works everywhere)**

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release   # Linux/macOS
# or
cmake .. -G "Visual Studio 17 2022" -A x64   # Windows MSVC
cmake --build . --config Release --parallel
./main                 # Linux/macOS
./Release/main.exe     # Windows
```

### Expected Output (in `results/`)

After running `make go` or `./build/main` you will get:

```
results/
├── data.txt              ← X matrix (first few rows)
├── bspline_coefs.txt     ← (placeholder for future basis coefficients)
├── fourier_coefs.txt     ← (placeholder)
├── loadings.txt          ← PLS weights/loadings
└── comparison.txt        ← True A vs recovered scores, R², etc.
```

## Authors

**Dhiaa Eddine Bahri** \
*dhya.bahri@proton.me* \
Main contributions: PLSR algorithm, project architecture, main integration and build system \

**Malek Rihani** \
*malek.rihani090@gmail.com* \
Main contributions: Functional data and nonlinear response generators \

Academic year 2025-2026 \
M. Sc. in Data Science and Information Retrieval \
[University of Manouba](https://uma.rnu.tn/) — [Higher Institute of Multimedia Arts](https://isa2m.rnu.tn) \

## Citation

If you use this code in your research or teaching, please cite:

```bibtex
@software{plsr_cpp_2025,
  author       = {Bahri, Dhiaa Eddine and Rihani, Malek},
  title        = {PLSR-CPP: A C++20 implementation of NIPALS PLS1 for functional data},
  year         = 2025,
  publisher    = {GitHub},
  url          = {https://github.com/Disciple0fMarx/plsr-cpp}
}
```

Enjoy the power of supervised dimension reduction on functional data!
