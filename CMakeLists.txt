cmake_minimum_required(VERSION 3.16)
project(PLSR_Demo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED)

add_executable(test_DataGenerator tests/test_DataGenerator.cpp src/DataGenerator.cpp)
target_include_directories(test_DataGenerator PRIVATE include)
target_link_libraries(test_DataGenerator PRIVATE Eigen3::Eigen)

add_executable(test_ResponseGenerator
    tests/test_ResponseGenerator.cpp
    src/DataGenerator.cpp
    src/ResponseGenerator.cpp
)
target_include_directories(test_ResponseGenerator PRIVATE include)
target_link_libraries(test_ResponseGenerator PRIVATE Eigen3::Eigen)

add_executable(test_PLSR
    tests/test_PLSR.cpp
    src/DataGenerator.cpp
    src/ResponseGenerator.cpp
    src/PLSR.cpp
)
target_include_directories(test_PLSR PRIVATE include)
target_link_libraries(test_PLSR PRIVATE Eigen3::Eigen)

add_executable(main main.cpp
    src/DataGenerator.cpp
    src/ResponseGenerator.cpp
    src/PLSR.cpp
    src/fourier.cpp
    src/bspline.cpp
)
target_include_directories(main PRIVATE include)
target_link_libraries(main PRIVATE Eigen3::Eigen)

add_executable(test_Fourier
    tests/test_fourier.cpp
    src/DataGenerator.cpp
    src/ResponseGenerator.cpp
    src/fourier.cpp
)
target_include_directories(test_Fourier PRIVATE include)
target_link_libraries(test_Fourier PRIVATE Eigen3::Eigen)

add_executable(test_Bspline
    tests/test_bspline.cpp
    src/DataGenerator.cpp
    src/ResponseGenerator.cpp
    src/bspline.cpp
)
target_include_directories(test_Bspline PRIVATE include)
target_link_libraries(test_Bspline PRIVATE Eigen3::Eigen)

