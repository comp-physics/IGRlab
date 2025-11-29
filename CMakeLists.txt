cmake_minimum_required(VERSION 3.14)
project(my_mfem_stuff LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(mfem REQUIRED)

add_subdirectory(src)
add_subdirectory(examples/supersonic_example)
add_subdirectory(examples/supersonic_example_w_igr)
add_subdirectory(examples/forward_facing_step)
add_subdirectory(examples/sedov_taylor_blast)
add_subdirectory(examples/shock_vortex_interaction)