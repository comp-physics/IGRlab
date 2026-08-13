# Collect all source files (including module subdirectories)
set(DG_IGR_MFEM_SOURCE_ROOT
    "${CMAKE_SOURCE_DIR}/tpl/mfem"
    CACHE PATH
    "Exact MFEM source checkout supplying source-internal headers.")
foreach(_dg_igr_mfem_source_header IN ITEMS
        fem/picojson.h
        general/tinyxml2.h)
    if(NOT EXISTS
       "${DG_IGR_MFEM_SOURCE_ROOT}/${_dg_igr_mfem_source_header}" OR
       IS_DIRECTORY
       "${DG_IGR_MFEM_SOURCE_ROOT}/${_dg_igr_mfem_source_header}")
        message(FATAL_ERROR
                "Required MFEM source header is missing: "
                "${DG_IGR_MFEM_SOURCE_ROOT}/${_dg_igr_mfem_source_header}")
    endif()
endforeach()

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "*.cpp")

# W8 runtime and auxiliary sources are owned by strict named component targets.
# Exclude their complete C++ roots unconditionally so neither default-OFF nor
# feature-ON configurations can double-compile them through igr_mfem's glob.
file(GLOB_RECURSE _dg_igr_hdg_named_component_cpp CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/numerics/hdg/runtime/*.cpp"
     "${CMAKE_CURRENT_SOURCE_DIR}/numerics/hdg/auxiliary/*.cpp")
if(_dg_igr_hdg_named_component_cpp)
    list(REMOVE_ITEM SOURCES ${_dg_igr_hdg_named_component_cpp})
endif()

# The opt-in main-application HDG runner is compiled only by its named CPU
# component.  Keeping it out of igr_mfem preserves default-OFF examples and
# prevents a second copy of its generic runtime entry.
list(REMOVE_ITEM SOURCES
     "${CMAKE_CURRENT_SOURCE_DIR}/sim/hdg/main_backend.cpp")

if(NOT DG_IGR_ENABLE_HDG_EULER_EXPERIMENT)
    list(REMOVE_ITEM SOURCES
         "${CMAKE_CURRENT_SOURCE_DIR}/numerics/hdg/euler/smooth_euler_mpi.cpp")
endif()
if(NOT DG_IGR_ENABLE_HDG_EULER_EXPERIMENT AND
   NOT DG_IGR_ENABLE_HDG_EULER_DEVICE_EXPERIMENT)
    list(REMOVE_ITEM SOURCES
         "${CMAKE_CURRENT_SOURCE_DIR}/numerics/hdg/euler/smooth_euler.cpp")
endif()
if(NOT DG_IGR_ENABLE_HDG_EULER_DEVICE_EXPERIMENT)
    list(REMOVE_ITEM SOURCES
         "${CMAKE_CURRENT_SOURCE_DIR}/numerics/hdg/euler/smooth_euler_device.cpp")
endif()

set(_dg_igr_hdg_euler_full_device_enabled OFF)
if(DG_IGR_ENABLE_HDG_EULER_DEVICE_EXPERIMENT AND
   DG_IGR_ACCELERATOR_LANGUAGE STREQUAL "CUDA" AND
   MFEM_USE_CUDA AND MFEM_USE_MPI AND MFEM_USE_UMPIRE AND TARGET umpire)
    get_target_property(_dg_igr_mfem_link_dependencies mfem
                        INTERFACE_LINK_LIBRARIES)
    if(_dg_igr_mfem_link_dependencies MATCHES "HYPRE")
        set(_dg_igr_hdg_euler_full_device_enabled ON)
        list(APPEND SOURCES
             "${CMAKE_CURRENT_SOURCE_DIR}/numerics/hdg/euler/smooth_euler_device_full.cu")
    endif()
endif()

# Create library
add_library(igr_mfem STATIC ${SOURCES})

set(_dg_igr_default_accel_sources "")
if(DG_IGR_ACCELERATOR_LANGUAGE)
    list(APPEND _dg_igr_default_accel_sources
         "src/numerics/dg/DGHyperbolicConservationLaws.cpp"
         "src/numerics/dg/DGHyperbolicConservationLaws_face.cpp"
         "src/numerics/dg/DGHyperbolicConservationLaws_mass.cpp"
         "src/numerics/dg/DGHyperbolicConservationLaws_weakdiv.cpp"
         "src/numerics/dg/DGFaceQuadratureKernels.cpp"
         "src/numerics/dgsem/dgsem_device.cpp"
         "src/numerics/hdg/device/device_core.cpp"
         "src/numerics/limiters/zhang_shu.cpp"
         "src/physics/igr/IGR_device_coefficients.cpp")
    if(DG_IGR_ENABLE_HDG_EULER_DEVICE_EXPERIMENT)
        list(APPEND _dg_igr_default_accel_sources
             "src/numerics/hdg/euler/smooth_euler_device.cpp")
    endif()
    if(_dg_igr_hdg_euler_full_device_enabled)
        list(APPEND _dg_igr_default_accel_sources
             "src/numerics/hdg/euler/smooth_euler_device_full.cu")
    endif()
endif()

set(DG_IGR_AUTO_DG_ACCELERATOR_SOURCE
    ON
    CACHE BOOL
    "When ON, automatically compile DGHyperbolicConservationLaws.cpp with the accelerator language in CUDA/HIP builds.")

if(NOT DEFINED DG_IGR_ACCELERATOR_KERNEL_SOURCES)
    set(DG_IGR_ACCELERATOR_KERNEL_SOURCES
        "${_dg_igr_default_accel_sources}"
        CACHE STRING
        "Semicolon-separated source files (repo-relative or absolute) to compile with ${DG_IGR_ACCELERATOR_LANGUAGE} for MFEM_FORALL kernels.")
else()
    if(DG_IGR_AUTO_DG_ACCELERATOR_SOURCE AND
       DG_IGR_ACCELERATOR_LANGUAGE AND
       _dg_igr_default_accel_sources AND
       "${DG_IGR_ACCELERATOR_KERNEL_SOURCES}" STREQUAL "")
        set(DG_IGR_ACCELERATOR_KERNEL_SOURCES "${_dg_igr_default_accel_sources}")
    endif()
    set(DG_IGR_ACCELERATOR_KERNEL_SOURCES
        "${DG_IGR_ACCELERATOR_KERNEL_SOURCES}"
        CACHE STRING
        "Semicolon-separated source files (repo-relative or absolute) to compile with ${DG_IGR_ACCELERATOR_LANGUAGE} for MFEM_FORALL kernels.")
endif()

if(DG_IGR_AUTO_DG_ACCELERATOR_SOURCE AND
   DG_IGR_ACCELERATOR_LANGUAGE AND
   _dg_igr_default_accel_sources)
    foreach(_required_src IN LISTS _dg_igr_default_accel_sources)
        list(FIND DG_IGR_ACCELERATOR_KERNEL_SOURCES "${_required_src}" _required_idx)
        if(_required_idx EQUAL -1)
            list(APPEND DG_IGR_ACCELERATOR_KERNEL_SOURCES "${_required_src}")
        endif()
    endforeach()
    set(DG_IGR_ACCELERATOR_KERNEL_SOURCES
        "${DG_IGR_ACCELERATOR_KERNEL_SOURCES}"
        CACHE STRING
        "Semicolon-separated source files (repo-relative or absolute) to compile with ${DG_IGR_ACCELERATOR_LANGUAGE} for MFEM_FORALL kernels."
        FORCE)
endif()

# Include directories  
target_include_directories(igr_mfem PUBLIC 
    ${CMAKE_SOURCE_DIR}/include
)
target_include_directories(igr_mfem PRIVATE
    ${DG_IGR_MFEM_SOURCE_ROOT}/fem
    ${DG_IGR_MFEM_SOURCE_ROOT}/general
)
target_compile_definitions(igr_mfem PUBLIC
    DG_IGR_ENABLE_OBSERVABILITY=$<BOOL:${DG_IGR_ENABLE_OBSERVABILITY}>
    DG_IGR_OBSERVABILITY_USE_NVTX=$<BOOL:${DG_IGR_OBSERVABILITY_USE_NVTX}>
    DG_IGR_OBSERVABILITY_USE_CALIPER=$<BOOL:${DG_IGR_OBSERVABILITY_USE_CALIPER}>
)

# Link MFEM
target_link_libraries(igr_mfem PUBLIC mfem ZLIB::ZLIB)
if(DG_IGR_OBSERVABILITY_USE_CALIPER)
    target_link_libraries(igr_mfem PUBLIC caliper)
endif()
if(MFEM_USE_MPI)
    target_link_libraries(igr_mfem PUBLIC MPI::MPI_CXX)
    if(DG_IGR_MPI_INCLUDE_DIRS)
        target_include_directories(igr_mfem PUBLIC ${DG_IGR_MPI_INCLUDE_DIRS})
    endif()
endif()
if(_dg_igr_hdg_euler_full_device_enabled)
    target_link_libraries(igr_mfem PRIVATE umpire)
endif()

if(MFEM_USE_CUDA)
    # Keep CUDA TPL linkage explicit to avoid transitive-link drops when MFEM/HYPRE
    # exports are incomplete.
    target_link_libraries(igr_mfem PUBLIC ${DG_IGR_REQUIRED_CUDA_TARGETS})
endif()

if(DG_IGR_ACCELERATOR_KERNEL_SOURCES)
    if(NOT DG_IGR_ACCELERATOR_LANGUAGE)
        message(FATAL_ERROR
                "DG_IGR_ACCELERATOR_KERNEL_SOURCES is set, but local accelerator language support "
                "is unavailable. Ensure MFEM backend and CMAKE_<backend>_COMPILER are configured.")
    endif()

    set(_dg_igr_accel_sources_abs "")
    foreach(_src IN LISTS DG_IGR_ACCELERATOR_KERNEL_SOURCES)
        if(IS_ABSOLUTE "${_src}")
            set(_src_abs "${_src}")
        else()
            set(_src_abs "${CMAKE_SOURCE_DIR}/${_src}")
        endif()
        if(NOT EXISTS "${_src_abs}")
            message(FATAL_ERROR
                    "DG_IGR accelerator source not found: ${_src_abs}")
        endif()
        list(APPEND _dg_igr_accel_sources_abs "${_src_abs}")
    endforeach()

    set_source_files_properties(${_dg_igr_accel_sources_abs}
                                PROPERTIES
                                LANGUAGE ${DG_IGR_ACCELERATOR_LANGUAGE})

    if(DG_IGR_ACCELERATOR_LANGUAGE STREQUAL "CUDA")
        set_target_properties(igr_mfem
                              PROPERTIES
                              CUDA_STANDARD 17
                              CUDA_STANDARD_REQUIRED ON
                              CUDA_EXTENSIONS OFF
                              CUDA_SEPARABLE_COMPILATION ON)
        target_compile_options(igr_mfem PRIVATE
                               $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
                               $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
        if(CMAKE_CUDA_ARCHITECTURES)
            set_property(TARGET igr_mfem
                         PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
        endif()
    endif()

    message(STATUS
            "DG_IGR accelerator kernel sources (${DG_IGR_ACCELERATOR_LANGUAGE}): "
            "${_dg_igr_accel_sources_abs}")
endif()

if(DG_IGR_ENABLE_HDG_RUNTIME OR
   DG_IGR_ENABLE_HDG_AUXILIARY_EXPERIMENT)
    add_subdirectory(numerics/hdg/runtime)
endif()
