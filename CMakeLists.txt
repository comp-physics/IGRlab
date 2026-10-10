cmake_minimum_required(VERSION 3.14)
project(IGRlab LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(CTest)
include(CheckLanguage)
include(cmake/IGRlabCompatibility.cmake)
igrlab_cache_alias(IGRLAB_BUILD_TESTS DG_IGR_BUILD_TESTS
                   BOOL ON "Build IGRlab correctness tests")
igrlab_cache_alias(IGRLAB_ENABLE_HDG_EULER_EXPERIMENT DG_IGR_ENABLE_HDG_EULER_EXPERIMENT
                   BOOL OFF "Build the serial smooth-Euler HDG experiment.")
igrlab_cache_alias(IGRLAB_ENABLE_HDG_EULER_DEVICE_EXPERIMENT DG_IGR_ENABLE_HDG_EULER_DEVICE_EXPERIMENT
                   BOOL OFF "Build the default-off smooth-Euler HDG CPU-fallback/device experiment.")
igrlab_cache_alias(IGRLAB_ENABLE_HDG_RUNTIME DG_IGR_ENABLE_HDG_RUNTIME
                   BOOL OFF "Build the default-off persistent HDG runtime component seam.")
igrlab_cache_alias(IGRLAB_ENABLE_HDG_AUXILIARY_EXPERIMENT DG_IGR_ENABLE_HDG_AUXILIARY_EXPERIMENT
                   BOOL OFF "Build the default-off CPU-only HDG auxiliary experiment seam.")
igrlab_cache_alias(IGRLAB_ENABLE_OBSERVABILITY DG_IGR_ENABLE_OBSERVABILITY
                   BOOL OFF "Compile the core observability contract (runtime remains default-off).")
igrlab_cache_alias(IGRLAB_OBSERVABILITY_USE_NVTX DG_IGR_OBSERVABILITY_USE_NVTX
                   BOOL OFF "Use the NVTX adapter for compiled observability regions.")
igrlab_cache_alias(IGRLAB_OBSERVABILITY_USE_CALIPER DG_IGR_OBSERVABILITY_USE_CALIPER
                   BOOL OFF "Use the Caliper adapter for compiled observability regions.")
if((DG_IGR_OBSERVABILITY_USE_NVTX OR DG_IGR_OBSERVABILITY_USE_CALIPER) AND
   NOT DG_IGR_ENABLE_OBSERVABILITY)
    message(FATAL_ERROR
            "An observability backend requires DG_IGR_ENABLE_OBSERVABILITY=ON.")
endif()
if(DG_IGR_OBSERVABILITY_USE_NVTX AND DG_IGR_OBSERVABILITY_USE_CALIPER)
    message(FATAL_ERROR
            "DG_IGR_OBSERVABILITY_USE_NVTX and DG_IGR_OBSERVABILITY_USE_CALIPER are mutually exclusive.")
endif()
if(DG_IGR_OBSERVABILITY_USE_CALIPER)
    # Caliper documents this package and imported target as `caliper`.
    find_package(caliper REQUIRED)
    if(NOT TARGET caliper)
        message(FATAL_ERROR
                "The discovered Caliper package does not export the required 'caliper' target.")
    endif()
endif()
igrlab_cache_alias(IGRLAB_REQUIRE_MFEM_CUDA DG_IGR_REQUIRE_MFEM_CUDA
                   BOOL OFF "Fail configure unless the discovered MFEM package has CUDA enabled.")
igrlab_cache_alias(IGRLAB_ENABLE_ACCELERATOR_LANGUAGE DG_IGR_ENABLE_ACCELERATOR_LANGUAGE
                   BOOL ON "Enable local CUDA/HIP language support when MFEM backend requires it.")
igrlab_cache_alias(IGRLAB_EXPECT_CUDA_ARCH DG_IGR_EXPECT_CUDA_ARCH
                   STRING ""
                   "Expected concrete CUDA dependency/device arch for CUDA builds (e.g. 80, sm_80, 120).")
igrlab_cache_alias(IGRLAB_BUILD_RESEARCH_TESTS DG_IGR_BUILD_RESEARCH_TESTS
                   BOOL OFF
                   "Build standalone numerical-investigation and campaign probes.")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(MFEM REQUIRED CONFIG)
if(DG_IGR_ENABLE_HDG_EULER_EXPERIMENT AND
   (MFEM_USE_CUDA OR MFEM_USE_HIP))
    message(FATAL_ERROR
            "DG_IGR_ENABLE_HDG_EULER_EXPERIMENT=ON requires a non-CUDA, non-HIP MFEM.")
endif()
if(DG_IGR_ENABLE_HDG_RUNTIME AND NOT MFEM_USE_MPI)
    message(FATAL_ERROR
            "DG_IGR_ENABLE_HDG_RUNTIME=ON requires an MPI/HYPRE-enabled MFEM profile.")
endif()
if(DG_IGR_ENABLE_HDG_RUNTIME AND MFEM_USE_HIP)
    message(FATAL_ERROR
            "DG_IGR_ENABLE_HDG_RUNTIME=ON has no HIP backend; use the CPU or CUDA profile.")
endif()
if(DG_IGR_ENABLE_HDG_AUXILIARY_EXPERIMENT AND
   (MFEM_USE_CUDA OR MFEM_USE_HIP))
    message(FATAL_ERROR
            "DG_IGR_ENABLE_HDG_AUXILIARY_EXPERIMENT=ON requires a CPU-only MFEM profile.")
endif()
find_package(ZLIB REQUIRED)

if(MFEM_USE_UMPIRE)
    find_package(umpire REQUIRED CONFIG)
endif()

if(MFEM_USE_MPI)
    find_package(MPI REQUIRED COMPONENTS CXX)
    set(DG_IGR_MPI_INCLUDE_DIRS "${MPI_CXX_INCLUDE_DIRS}")
    if((NOT DG_IGR_MPI_INCLUDE_DIRS) AND MPI_CXX_COMPILER)
        execute_process(
            COMMAND "${MPI_CXX_COMPILER}" --showme:incdirs
            RESULT_VARIABLE _dg_igr_mpi_incdirs_rc
            OUTPUT_VARIABLE _dg_igr_mpi_incdirs_raw
            ERROR_QUIET
            OUTPUT_STRIP_TRAILING_WHITESPACE)
        if(_dg_igr_mpi_incdirs_rc EQUAL 0 AND _dg_igr_mpi_incdirs_raw)
            string(REPLACE " " ";" DG_IGR_MPI_INCLUDE_DIRS "${_dg_igr_mpi_incdirs_raw}")
        endif()
    endif()
endif()

if(MFEM_USE_CUDA AND MFEM_USE_HIP)
    message(FATAL_ERROR
            "MFEM reports both CUDA and HIP enabled for this target; "
            "IGRlab expects one accelerator backend at a time.")
endif()

set(_dg_igr_expected_arch "")
if(DG_IGR_EXPECT_CUDA_ARCH)
    set(_dg_igr_expected_arch "${DG_IGR_EXPECT_CUDA_ARCH}")
    string(REPLACE "sm_" "" _dg_igr_expected_arch "${_dg_igr_expected_arch}")
    if(NOT _dg_igr_expected_arch MATCHES "^[0-9]+$")
        message(FATAL_ERROR
                "DG_IGR_EXPECT_CUDA_ARCH must be numeric or sm_<num>, got '${DG_IGR_EXPECT_CUDA_ARCH}'.")
    endif()
endif()

set(DG_IGR_ACCELERATOR_LANGUAGE "")
if(DG_IGR_ENABLE_ACCELERATOR_LANGUAGE)
    if(MFEM_USE_CUDA)
        if(_dg_igr_expected_arch AND NOT CMAKE_CUDA_ARCHITECTURES)
            set(_dg_igr_default_cuda_arch "${_dg_igr_expected_arch}")
            set(CMAKE_CUDA_ARCHITECTURES
                "${_dg_igr_default_cuda_arch}"
                CACHE STRING
                "CUDA architectures for local accelerator-compiled sources."
                FORCE)
        endif()

        check_language(CUDA)
        if(CMAKE_CUDA_COMPILER)
            enable_language(CUDA)
            if(NOT DEFINED CMAKE_CUDA_STANDARD)
                set(CMAKE_CUDA_STANDARD ${CMAKE_CXX_STANDARD})
                set(CMAKE_CUDA_STANDARD_REQUIRED ON)
            endif()
            set(DG_IGR_ACCELERATOR_LANGUAGE "CUDA")
        else()
            message(WARNING
                    "MFEM_USE_CUDA=ON but no CUDA compiler is available to CMake. "
                    "MFEM CUDA runtime paths remain usable, but local sources cannot "
                    "compile mfem::forall CUDA kernels until CMAKE_CUDA_COMPILER is set.")
        endif()
    elseif(MFEM_USE_HIP)
        if(CMAKE_VERSION VERSION_LESS 3.21)
            message(FATAL_ERROR
                    "MFEM_USE_HIP=ON requires CMake >= 3.21 to enable HIP language.")
        endif()
        check_language(HIP)
        if(CMAKE_HIP_COMPILER)
            enable_language(HIP)
            if(NOT DEFINED CMAKE_HIP_STANDARD)
                set(CMAKE_HIP_STANDARD ${CMAKE_CXX_STANDARD})
                set(CMAKE_HIP_STANDARD_REQUIRED ON)
            endif()
            set(DG_IGR_ACCELERATOR_LANGUAGE "HIP")
        else()
            message(WARNING
                    "MFEM_USE_HIP=ON but no HIP compiler is available to CMake. "
                    "MFEM HIP runtime paths remain usable, but local sources cannot "
                    "compile mfem::forall HIP kernels until CMAKE_HIP_COMPILER is set.")
        endif()
    endif()
endif()

if(DG_IGR_REQUIRE_MFEM_CUDA AND NOT MFEM_USE_CUDA)
    message(FATAL_ERROR
            "DG_IGR_REQUIRE_MFEM_CUDA=ON but discovered MFEM has MFEM_USE_CUDA=OFF. "
            "Point MFEM_DIR to a CUDA-enabled local MFEM install.")
endif()

if(MFEM_USE_CUDA)
    find_package(CUDAToolkit REQUIRED)

    # Required for explicit, fail-fast CUDA dep validation and explicit
    # downstream linkage in src/CMakeLists.txt.
    set(DG_IGR_REQUIRED_CUDA_TARGETS
        CUDA::cudart
        CUDA::cusparse
        CUDA::cublas
        CUDA::curand)

    set(_dg_igr_missing_cuda_targets "")
    foreach(_cuda_target IN LISTS DG_IGR_REQUIRED_CUDA_TARGETS)
        if(NOT TARGET ${_cuda_target})
            list(APPEND _dg_igr_missing_cuda_targets ${_cuda_target})
        endif()
    endforeach()
    if(_dg_igr_missing_cuda_targets)
        message(FATAL_ERROR
                "MFEM reports CUDA enabled, but required CUDA toolkit targets are missing: "
                "${_dg_igr_missing_cuda_targets}. "
                "Check CUDAToolkit_ROOT/CMAKE_PREFIX_PATH and CUDA installation.")
    endif()

    get_target_property(_dg_igr_mfem_iface_links mfem INTERFACE_LINK_LIBRARIES)
    set(_dg_igr_missing_cuda_link_names "")
    if(_dg_igr_mfem_iface_links)
        string(JOIN ";" _dg_igr_mfem_iface_links_joined ${_dg_igr_mfem_iface_links})
        # MFEM CUDA exports may omit cudart while still requiring it as a
        # transitive runtime dependency; we explicitly link CUDA::cudart in src/.
        foreach(_cuda_lib_name IN ITEMS cusparse cublas curand)
            if(NOT _dg_igr_mfem_iface_links_joined MATCHES "${_cuda_lib_name}")
                list(APPEND _dg_igr_missing_cuda_link_names ${_cuda_lib_name})
            endif()
        endforeach()
    else()
        list(APPEND _dg_igr_missing_cuda_link_names cusparse cublas curand)
    endif()

    if(_dg_igr_missing_cuda_link_names)
        message(FATAL_ERROR
                "MFEM reports CUDA enabled, but mfem INTERFACE_LINK_LIBRARIES is missing: "
                "${_dg_igr_missing_cuda_link_names}. "
                "This is a common transitive-link failure surface (e.g., cusparse not propagating). "
                "Rebuild local MFEM with CUDA-enabled HYPRE and verify MFEMTargets.cmake exports CUDA deps.")
    endif()
endif()

if(DG_IGR_EXPECT_CUDA_ARCH AND MFEM_USE_CUDA)
    if(CMAKE_CUDA_ARCHITECTURES)
        string(REPLACE ";" "," _dg_igr_arches "${CMAKE_CUDA_ARCHITECTURES}")
        set(_dg_igr_cuda_arch_ok OFF)
        foreach(_dg_igr_arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
            if(_dg_igr_arch STREQUAL "${_dg_igr_expected_arch}" OR
               _dg_igr_arch STREQUAL "${_dg_igr_expected_arch}-real")
                set(_dg_igr_cuda_arch_ok ON)
            endif()
        endforeach()
        if(NOT _dg_igr_cuda_arch_ok)
            message(FATAL_ERROR
                    "CUDA arch policy mismatch: expected concrete profile ${_dg_igr_expected_arch}, "
                    "configured CMAKE_CUDA_ARCHITECTURES=${_dg_igr_arches}. "
                    "Application CUDA sources must target the selected native profile.")
        endif()
    else()
        message(STATUS
                "DG_IGR_EXPECT_CUDA_ARCH=${_dg_igr_expected_arch}: local CUDA language is not enabled "
                "or imported MFEM package does not export architecture metadata; "
                "validate arch during local MFEM configure/build.")
    endif()
endif()

if(DG_IGR_ENABLE_HDG_RUNTIME AND MFEM_USE_CUDA)
    if(NOT _dg_igr_expected_arch STREQUAL "80" AND
       NOT _dg_igr_expected_arch STREQUAL "120")
        message(FATAL_ERROR
                "HDG CUDA components require native architecture 80 or 120.")
    endif()
    # Share the normalized, validated profile with every CUDA component and
    # final executable, including tests in sibling directories.
    set(DG_IGR_HDG_NATIVE_CUDA_ARCH "${_dg_igr_expected_arch}")
endif()

add_subdirectory(src)
add_subdirectory(examples/supersonic_example)
add_subdirectory(examples/supersonic_example_w_igr)
add_subdirectory(examples/forward_facing_step)
add_subdirectory(examples/sedov_taylor_blast)
add_subdirectory(examples/shock_vortex_interaction)
add_subdirectory(examples/stationary_shock)
add_subdirectory(examples/inoue_hattori_case_c)
add_subdirectory(examples/inoue_hattori_large_domain)
add_subdirectory(examples/inoue_hattori_stationary_shock)
add_subdirectory(examples/inoue_hattori_stationary_shock_1dy)
add_subdirectory(examples/double_mach_ref)
add_subdirectory(examples/dgsem_smooth_advection)
add_subdirectory(examples/dgsem_smooth_advection_periodic)
add_subdirectory(examples/hdg_smooth_transition_research)
add_subdirectory(examples/hdg_smooth_periodic)
add_subdirectory(examples/dgsem_stationary_shock)
add_subdirectory(examples/dgsem_shock_vortex_interaction)
add_subdirectory(examples/dgsem_forward_facing_step)
add_subdirectory(examples/dgsem_sedov_taylor_blast)
add_subdirectory(examples/dgsem_supersonic_cylinder)

if(BUILD_TESTING AND DG_IGR_BUILD_TESTS)
    add_subdirectory(tests)
endif()
