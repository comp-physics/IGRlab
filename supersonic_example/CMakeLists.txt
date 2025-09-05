# CMakeLists.txt for MFEM Examples: Supersonic Cylinder and Inviscid Burgers
# 
# This CMakeLists.txt builds two MFEM examples:
# 1. supersonic_cylinder - Compressible flow over cylinder
# 2. invisc_burgers - Inviscid Burgers equation solver
#
# Usage:
#   mkdir build && cd build
#   cmake ..
#   make -j4
#
#   # Run examples:
#   mpirun -np 4 ./supersonic_cylinder -p 1 -rs 2 -rp 1 -o 2
#   mpirun -np 4 ./invisc_burgers -p 1 -rs 3 -rp 1 -o 2 -tf 1.0

cmake_minimum_required(VERSION 3.10)
project(mfem_hyperbolic_examples)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find MFEM package
find_package(mfem REQUIRED)

# Check if MFEM was built with MPI support
if(NOT MFEM_USE_MPI)
    message(WARNING "MFEM was not built with MPI support. Examples may not work properly in parallel.")
endif()

# Check for required MFEM features
if(NOT MFEM_USE_HYPRE)
    message(WARNING "MFEM was not built with HYPRE support. Parallel examples may not work.")
endif()

# Compiler flags for optimization and debugging
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

# Set default build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Function to create MFEM example executable
function(add_mfem_example target_name source_files)
    add_executable(${target_name} ${source_files})
    
    # Link with MFEM
    target_link_libraries(${target_name} mfem)
    
    # Include MFEM headers
    target_include_directories(${target_name} PRIVATE ${MFEM_INCLUDE_DIRS})
    
    # Set compiler definitions from MFEM
    target_compile_definitions(${target_name} PRIVATE ${MFEM_DEFINITIONS})
    
    # Add MPI compilation flags if available
    if(MFEM_USE_MPI)
        find_package(MPI REQUIRED)
        target_link_libraries(${target_name} MPI::MPI_CXX)
    endif()
    
    # Add OpenMP support if MFEM uses it
    if(MFEM_USE_OPENMP)
        find_package(OpenMP REQUIRED)
        target_link_libraries(${target_name} OpenMP::OpenMP_CXX)
    endif()
    
    # Add CUDA support if MFEM uses it
    if(MFEM_USE_CUDA)
        enable_language(CUDA)
        set_property(TARGET ${target_name} PROPERTY CUDA_ARCHITECTURES 60 70 75 80)
    endif()
endfunction()

#==============================================================================
# SUPERSONIC CYLINDER EXAMPLE
#==============================================================================

# Check if supersonic_cylinder source files exist
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/supersonic_cylinder.cpp")
    set(SUPERSONIC_CYLINDER_SOURCES
        supersonic_cylinder.cpp BdrHyperbolicFluxIntegrator.cpp
    )
    
    # Add header file if it exists
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/supersonic_cylinder.hpp")
        list(APPEND SUPERSONIC_CYLINDER_SOURCES supersonic_cylinder.hpp BdrHyperbolicFluxIntegrator.hpp)
    endif()
    
    # Create supersonic_cylinder executable
    add_mfem_example(supersonic_cylinder "${SUPERSONIC_CYLINDER_SOURCES}")
    
    message(STATUS "Added supersonic_cylinder example")
else()
    message(WARNING "supersonic_cylinder.cpp not found - skipping this example")
endif()

#==============================================================================
# INVISCID BURGERS EXAMPLE  
#==============================================================================

# Check if inviscid burgers source files exist
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/invisc_burgers.cpp")
    set(INVISC_BURGERS_SOURCES
        invisc_burgers.cpp
    )
    
    # Add header file if it exists
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/invisc_burgers.hpp")
        list(APPEND INVISC_BURGERS_SOURCES invisc_burgers.hpp)
    endif()
    
    # Create invisc_burgers executable
    add_mfem_example(invisc_burgers "${INVISC_BURGERS_SOURCES}")
    
    message(STATUS "Added invisc_burgers example")
else()
    message(WARNING "invisc_burgers.cpp not found - skipping this example")
endif()

#==============================================================================
# OPTIONAL: Add other hyperbolic examples
#==============================================================================

# Example for adding ex18 if you want to compare
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ex18.cpp" AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ex18.hpp")
    add_mfem_example(ex18 "ex18.cpp;ex18.hpp")
    message(STATUS "Added ex18 for comparison")
endif()

# Example for adding ex18p if you want to compare
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ex18p.cpp" AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ex18.hpp")
    add_mfem_example(ex18p "ex18p.cpp;ex18.hpp")
    message(STATUS "Added ex18p for comparison")
endif()

#==============================================================================
# INSTALLATION (OPTIONAL)
#==============================================================================

# Install executables to CMAKE_INSTALL_PREFIX/bin
if(TARGET supersonic_cylinder)
    install(TARGETS supersonic_cylinder RUNTIME DESTINATION bin)
endif()

if(TARGET invisc_burgers)
    install(TARGETS invisc_burgers RUNTIME DESTINATION bin)
endif()

#==============================================================================
# TESTING (OPTIONAL)
#==============================================================================

# Enable testing
enable_testing()

# Add basic smoke tests (just check that executables run without crashing)
if(TARGET supersonic_cylinder)
    add_test(NAME supersonic_cylinder_smoke_test 
             COMMAND supersonic_cylinder -tf 0.01 -dt 0.001 -no-vis)
    set_tests_properties(supersonic_cylinder_smoke_test PROPERTIES 
                        TIMEOUT 60)
endif()

if(TARGET invisc_burgers)
    add_test(NAME invisc_burgers_smoke_test 
             COMMAND invisc_burgers -p 1 -tf 0.01 -dt 0.001 -no-vis)
    set_tests_properties(invisc_burgers_smoke_test PROPERTIES 
                        TIMEOUT 60)
endif()

# Parallel tests (if MPI is available and we have more than 1 process)
if(MFEM_USE_MPI)
    find_program(MPIEXEC mpirun mpiexec)
    if(MPIEXEC)
        if(TARGET supersonic_cylinder)
            add_test(NAME supersonic_cylinder_parallel_test
                     COMMAND ${MPIEXEC} -np 2 $<TARGET_FILE:supersonic_cylinder> 
                             -tf 0.01 -dt 0.001 -no-vis)
            set_tests_properties(supersonic_cylinder_parallel_test PROPERTIES 
                                TIMEOUT 120)
        endif()
        
        if(TARGET invisc_burgers)
            add_test(NAME invisc_burgers_parallel_test
                     COMMAND ${MPIEXEC} -np 2 $<TARGET_FILE:invisc_burgers> 
                             -p 1 -tf 0.01 -dt 0.001 -no-vis)
            set_tests_properties(invisc_burgers_parallel_test PROPERTIES 
                                TIMEOUT 120)
        endif()
    endif()
endif()

#==============================================================================
# STATUS MESSAGES
#==============================================================================

message(STATUS "")
message(STATUS "MFEM Hyperbolic Examples Configuration Summary:")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  MFEM found: ${MFEM_FOUND}")
message(STATUS "  MFEM version: ${MFEM_VERSION}")
message(STATUS "  MFEM uses MPI: ${MFEM_USE_MPI}")
message(STATUS "  MFEM uses HYPRE: ${MFEM_USE_HYPRE}")
message(STATUS "  MFEM uses OpenMP: ${MFEM_USE_OPENMP}")
message(STATUS "  MFEM uses CUDA: ${MFEM_USE_CUDA}")
message(STATUS "")

# Print build instructions
message(STATUS "To build:")
message(STATUS "  mkdir build && cd build")
message(STATUS "  cmake ..")
message(STATUS "  make -j4")
message(STATUS "")
message(STATUS "To run examples:")
if(TARGET supersonic_cylinder)
    if(MFEM_USE_MPI)
        message(STATUS "  mpirun -np 4 ./supersonic_cylinder -p 1 -rs 2 -rp 1 -o 2")
    else()
        message(STATUS "  ./supersonic_cylinder -p 1 -r 3 -o 2")
    endif()
endif()
if(TARGET invisc_burgers)
    if(MFEM_USE_MPI)
        message(STATUS "  mpirun -np 4 ./invisc_burgers -p 1 -rs 3 -rp 1 -o 2 -tf 1.0")
    else()
        message(STATUS "  ./invisc_burgers -p 1 -r 4 -o 2 -tf 1.0")
    endif()
endif()
message(STATUS "")