# Makefile — super-convenient wrapper around CMake
# You still get all the power of CMake, but now you can type `make run` like a god

.PHONY: all setup build run test tests debug clean distclean

# Default target
all: build/main

# First-time setup (creates build dir + configures)
setup:
	@cmake -B build -S . -DCMAKE_BUILD_TYPE=Release

# Normal fast rebuild (assumes build/ already configured)
build:
	@cmake --build build --config Release -j$(nproc)

# Debug build
debug:
	@cmake --build build --config Debug -j$(nproc)

# Run the demo
run: build/main
	@./build/main

# Build + run
go: build run

# Build and run all unit tests
test tests: build
	@echo "=== Running all tests ==="
	@./build/test_DataGenerator && \
	 ./build/test_ResponseGenerator && \
	 ./build/test_PLSR && \
	 ./build/test_Fourier && \
	 ./build/test_Bspline

# Individual tests (optional)
test-data: build
	@./build/test_DataGenerator

test-response: build
	@./build/test_ResponseGenerator

test-plsr: build
	@./build/test_PLSR

test-fourier: build
	@./build/test_Fourier

test-bspline: build
	@./build/test_Bspline

# Clean build directory (your old friend)
clean:
	@rm -rf build/*

# Nuclear option — also delete CMake cache
distclean: clean
	@rm -rf build/

# Force reconfiguration (useful if you add new files)
reconfig:
	@cmake -B build -S . -DCMAKE_BUILD_TYPE=Release

# Show what actually got compiled last time (debugging hero)
last:
	@cmake --build build --verbose --target help | tail -20

# The actual executables
build/main: setup
	@cmake --build build --target main -j$(nproc)

build/%: setup
	@cmake --build build --target $* -j$(nproc)

