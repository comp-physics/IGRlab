cmake_minimum_required(VERSION 3.20)
project(dg_igr_hdg_cpu_oracle LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(MFEM 4.9 CONFIG REQUIRED)
find_package(MPI REQUIRED COMPONENTS CXX)

if(NOT MFEM_VERSION_INT EQUAL 40900)
  message(FATAL_ERROR "HDG oracle requires immutable MFEM v4.9, got ${MFEM_VERSION}")
endif()
if(NOT MFEM_GIT_STRING STREQUAL
   "tags/v4.9-0-gd9d6526cc1749980a2ba1da16e2c1ca1e07d82ec")
  message(FATAL_ERROR "HDG oracle requires pinned MFEM commit d9d6526")
endif()
if(MFEM_USE_CUDA OR MFEM_USE_HIP OR MFEM_USE_RAJA OR MFEM_USE_UMPIRE)
  message(FATAL_ERROR "HDG CPU oracle rejects accelerator-enabled MFEM installs")
endif()

add_executable(hdg_oracle
  hdg_oracle_main.cpp
  ../../src/numerics/hdg/scalar_hdg_oracle.cpp)
target_include_directories(hdg_oracle PRIVATE ../../include)
target_link_libraries(hdg_oracle PRIVATE mfem MPI::MPI_CXX)
target_compile_options(hdg_oracle PRIVATE -Wall -Wextra -Wpedantic -Werror)
