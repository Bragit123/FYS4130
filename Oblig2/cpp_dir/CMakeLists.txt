## This file was written by ChatGPT, then modified by me

cmake_minimum_required(VERSION 3.10)
project(SpinSimulation)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Tell CMake where to look for Homebrew packages
set(CMAKE_PREFIX_PATH "/opt/homebrew")

# Add your source files
add_executable(spin_sim main.cpp spin_model.cpp)

set(ARMADILLO_INCLUDE_DIRS "/opt/homebrew/include")
set(ARMADILLO_LIBRARIES "/opt/homebrew/lib/")

find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})
target_link_libraries(spin_sim ${ARMADILLO_LIBRARIES})