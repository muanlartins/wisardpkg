cmake_minimum_required(VERSION 3.16)
project(wisard_perf CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Optimization flags are overridable so the benchmark can build a plain `-O2`
# rung of the ladder alongside the default `-O3 -march=native`:
#   cmake -S . -B build-o2 -DWP_OPT_FLAGS=-O2
if(NOT WP_OPT_FLAGS)
  set(WP_OPT_FLAGS "-O3 -march=native")
endif()
set(CMAKE_CXX_FLAGS_RELEASE "${WP_OPT_FLAGS}")

add_executable(wisard_bench
  src/wisard.cpp
  src/io.cpp
  src/bench_main.cpp
)
target_include_directories(wisard_bench PRIVATE include)

# OpenMP is optional. On Apple-clang it comes from Homebrew libomp; hint its
# location but never let a missing OpenMP break the build.
if(APPLE AND NOT DEFINED OpenMP_ROOT)
  if(EXISTS "/opt/homebrew/opt/libomp")
    set(OpenMP_ROOT "/opt/homebrew/opt/libomp")
  elseif(EXISTS "/usr/local/opt/libomp")
    set(OpenMP_ROOT "/usr/local/opt/libomp")
  endif()
endif()

if(APPLE AND DEFINED OpenMP_ROOT)
  set(OpenMP_CXX_FLAGS "-Xclang -fopenmp -I${OpenMP_ROOT}/include"
      CACHE STRING "" FORCE)
  set(OpenMP_CXX_LIB_NAMES "omp" CACHE STRING "" FORCE)
  set(OpenMP_omp_LIBRARY "${OpenMP_ROOT}/lib/libomp.dylib" CACHE FILEPATH "" FORCE)
endif()

find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
  target_link_libraries(wisard_bench PRIVATE OpenMP::OpenMP_CXX)
  target_compile_definitions(wisard_bench PRIVATE WP_OPENMP)
  message(STATUS "OpenMP enabled")
else()
  message(STATUS "OpenMP not found - building single-threaded")
endif()
