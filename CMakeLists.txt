cmake_minimum_required(VERSION 3.14)
project(my_mfem_stuff LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(CTest)
option(DG_IGR_BUILD_TESTS "Build dg_igr unit tests" ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(MFEM REQUIRED)

add_subdirectory(src)
add_subdirectory(examples/supersonic_example)
add_subdirectory(examples/supersonic_example_w_igr)
add_subdirectory(examples/forward_facing_step)
add_subdirectory(examples/sedov_taylor_blast)
add_subdirectory(examples/shock_vortex_interaction)
add_subdirectory(examples/stationary_shock)
add_subdirectory(examples/double_mach_ref)
add_subdirectory(examples/dgsem_smooth_advection)
add_subdirectory(examples/dgsem_smooth_advection_periodic)
add_subdirectory(examples/dgsem_stationary_shock)
add_subdirectory(examples/dgsem_shock_vortex_interaction)
add_subdirectory(examples/dgsem_forward_facing_step)
add_subdirectory(examples/dgsem_sedov_taylor_blast)
add_subdirectory(examples/dgsem_supersonic_cylinder)

if(BUILD_TESTING AND DG_IGR_BUILD_TESTS)
    add_subdirectory(tests)
endif()
