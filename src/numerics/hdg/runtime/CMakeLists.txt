include("${PROJECT_SOURCE_DIR}/cmake/DgIgrHdgComponents.cmake")

file(GLOB_RECURSE _dg_igr_hdg_runtime_common_sources CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/common/*.cpp")
file(GLOB_RECURSE _dg_igr_hdg_runtime_app_sources CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/app/*.cpp")
file(GLOB_RECURSE _dg_igr_hdg_runtime_app_cuda_sources CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/app/*.cu")
file(GLOB_RECURSE _dg_igr_hdg_runtime_cpu_sources CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/cpu/*.cpp")
file(GLOB_RECURSE _dg_igr_hdg_runtime_cuda_sources CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/cuda/*.cpp"
     "${CMAKE_CURRENT_SOURCE_DIR}/cuda/*.cu")
file(GLOB_RECURSE _dg_igr_hdg_runtime_all_cpp CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
file(GLOB_RECURSE _dg_igr_hdg_runtime_all_cuda CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")

# CUDA-enabled MFEM conditionally extends mfem::SparseMatrix when a translation
# unit is compiled by NVCC.  Any owner of that type must therefore use the same
# language as the MFEM library in CUDA profiles; mixing a C++ owner with MFEM's
# CUDA constructor is an object-layout ABI violation.
set(_dg_igr_hdg_mfem_sparse_owner_sources
    "${CMAKE_CURRENT_SOURCE_DIR}/cpu/persistent_sc25_scalar_igr.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/cpu/persistent_smooth_euler.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/cpu/traced_scalar_igr_p1_global_step.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../auxiliary/screened_poisson_cpu.cpp")
if(MFEM_USE_CUDA)
  set_source_files_properties(${_dg_igr_hdg_mfem_sparse_owner_sources}
    PROPERTIES LANGUAGE CUDA)
endif()

# The generic Euler CPU reference engine deliberately owns only the reusable
# request/session, borrowed-layout, and traced p1 pieces. The frozen SC25
# implementation remains a separate oracle component. Keep both inventories
# explicit so an added CPU source cannot silently reconstitute a monolith.
set(_dg_igr_hdg_generic_cpu_sources
    "${CMAKE_CURRENT_SOURCE_DIR}/cpu/borrowed_p1_dg_interface_layout.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/cpu/generic_euler.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/cpu/persistent_smooth_euler.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/cpu/traced_scalar_igr_p1_global_step.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/cpu/traced_scalar_igr_p1_local.cpp")
set(_dg_igr_hdg_sc25_oracle_cpu_sources
    "${CMAKE_CURRENT_SOURCE_DIR}/cpu/persistent_sc25_scalar_igr.cpp")
set(_dg_igr_hdg_runtime_cpu_expected_sources
    ${_dg_igr_hdg_generic_cpu_sources}
    ${_dg_igr_hdg_sc25_oracle_cpu_sources})
list(SORT _dg_igr_hdg_runtime_cpu_sources)
list(SORT _dg_igr_hdg_runtime_cpu_expected_sources)
if(NOT _dg_igr_hdg_runtime_cpu_sources STREQUAL
   _dg_igr_hdg_runtime_cpu_expected_sources)
  message(FATAL_ERROR
          "HDG CPU source inventory drifted: every runtime CPU source must be classified as either generic reference or frozen SC25 oracle.")
endif()
list(FIND _dg_igr_hdg_generic_cpu_sources
     "${CMAKE_CURRENT_SOURCE_DIR}/cpu/persistent_sc25_scalar_igr.cpp"
     _dg_igr_hdg_sc25_in_generic_index)
if(NOT _dg_igr_hdg_sc25_in_generic_index EQUAL -1)
  message(FATAL_ERROR
          "The generic HDG CPU component must not own persistent_sc25_scalar_igr.cpp.")
endif()

set(_dg_igr_hdg_runtime_unclassified_cpp
    ${_dg_igr_hdg_runtime_all_cpp})
list(REMOVE_ITEM _dg_igr_hdg_runtime_unclassified_cpp
     ${_dg_igr_hdg_runtime_common_sources}
     ${_dg_igr_hdg_runtime_app_sources}
     ${_dg_igr_hdg_runtime_cpu_sources}
     ${_dg_igr_hdg_runtime_cuda_sources})
set(_dg_igr_hdg_runtime_unclassified_cuda
    ${_dg_igr_hdg_runtime_all_cuda})
list(REMOVE_ITEM _dg_igr_hdg_runtime_unclassified_cuda
     ${_dg_igr_hdg_runtime_cuda_sources}
     ${_dg_igr_hdg_runtime_app_cuda_sources})
if(_dg_igr_hdg_runtime_unclassified_cpp OR
   _dg_igr_hdg_runtime_unclassified_cuda)
  message(FATAL_ERROR
          "Unclassified HDG runtime sources are excluded from igr_mfem but have no named component target: "
          "${_dg_igr_hdg_runtime_unclassified_cpp};${_dg_igr_hdg_runtime_unclassified_cuda}")
endif()

set(_dg_igr_hdg_scalar_igr_cpu_app_available OFF)
if(DG_IGR_ENABLE_HDG_AUXILIARY_EXPERIMENT)
  set(_dg_igr_hdg_scalar_igr_cpu_app_available ON)
endif()

if(DG_IGR_ENABLE_HDG_AUXILIARY_EXPERIMENT OR
   (DG_IGR_ENABLE_HDG_RUNTIME AND MFEM_USE_CUDA))
  file(GLOB_RECURSE _dg_igr_hdg_auxiliary_sources CONFIGURE_DEPENDS
       "${CMAKE_CURRENT_SOURCE_DIR}/../auxiliary/*.cpp")
  dg_igr_add_hdg_cxx17_component(
    TARGET dg_igr_hdg_auxiliary_cpu
    SOURCES ${_dg_igr_hdg_auxiliary_sources})
  if(MFEM_USE_CUDA)
    set_target_properties(dg_igr_hdg_auxiliary_cpu PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED ON
      CUDA_EXTENSIONS OFF)
    target_compile_options(dg_igr_hdg_auxiliary_cpu PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
  endif()
endif()

if(DG_IGR_ENABLE_HDG_RUNTIME)
  if(MFEM_USE_CUDA)
    set(_dg_igr_hdg_runtime_cuda_expected_sources
        "${CMAKE_CURRENT_SOURCE_DIR}/cuda/hdg_sc25_periodic_v3_device.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/cuda/hdg_sc25_periodic_v3_rank2_device.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/cuda/persistent_smooth_euler.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/cuda/traced_scalar_igr_p1_private.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/cuda/traced_scalar_igr_p1_private_numerics.cu")
    list(SORT _dg_igr_hdg_runtime_cuda_sources)
    list(SORT _dg_igr_hdg_runtime_cuda_expected_sources)
    if(NOT _dg_igr_hdg_runtime_cuda_sources STREQUAL
       _dg_igr_hdg_runtime_cuda_expected_sources)
      message(FATAL_ERROR
              "The frozen W8A+W8B CUDA component requires exactly persistent_smooth_euler.cu, the SC25 rank-one/rank-two device sources, and the private traced-scalar-IGR packed-batch/topology sources.")
    endif()
  endif()

  if(TARGET dg_igr_hdg_runtime_components OR
     TARGET dg_igr_hdg_runtime OR TARGET dg_igr_hdg)
    message(FATAL_ERROR
            "Canonical HDG component, future runtime, or executable target name is already in use.")
  endif()

  add_library(dg_igr_hdg_runtime_components INTERFACE)

  dg_igr_add_hdg_cxx17_component(
    TARGET dg_igr_hdg_runtime_common
    STANDARD_LIBRARY_ONLY
    SOURCES ${_dg_igr_hdg_runtime_common_sources})

  set(_dg_igr_hdg_runtime_app_contract_sources
      "${CMAKE_CURRENT_SOURCE_DIR}/app/application_contract.cpp"
      "${CMAKE_CURRENT_SOURCE_DIR}/app/checkpoint_codec.cpp"
      "${CMAKE_CURRENT_SOURCE_DIR}/app/command_line.cpp"
      "${CMAKE_CURRENT_SOURCE_DIR}/app/durable_io.cpp")
  set(_dg_igr_hdg_runtime_app_executable_sources
      "${CMAKE_CURRENT_SOURCE_DIR}/app/main.cpp")
  set(_dg_igr_hdg_runtime_app_cpu_sources
      "${CMAKE_CURRENT_SOURCE_DIR}/app/solver_cpu.cpp")
  set(_dg_igr_hdg_runtime_app_entry_source
      "${CMAKE_CURRENT_SOURCE_DIR}/app/solver_entry.cpp")
  set(_dg_igr_hdg_runtime_app_expected_sources
      ${_dg_igr_hdg_runtime_app_contract_sources}
      ${_dg_igr_hdg_runtime_app_executable_sources}
      ${_dg_igr_hdg_runtime_app_cpu_sources}
      ${_dg_igr_hdg_runtime_app_entry_source})
  list(SORT _dg_igr_hdg_runtime_app_sources)
  list(SORT _dg_igr_hdg_runtime_app_expected_sources)
  if(NOT _dg_igr_hdg_runtime_app_sources STREQUAL
     _dg_igr_hdg_runtime_app_expected_sources)
    message(FATAL_ERROR
            "The frozen W8A application slice source inventory drifted.")
  endif()
  add_library(dg_igr_hdg_runtime_app_contract STATIC
              ${_dg_igr_hdg_runtime_app_contract_sources})
  set_target_properties(dg_igr_hdg_runtime_app_contract PROPERTIES
                        CXX_STANDARD 17
                        CXX_STANDARD_REQUIRED ON
                        CXX_EXTENSIONS OFF)
  target_include_directories(dg_igr_hdg_runtime_app_contract PUBLIC
                             "${PROJECT_SOURCE_DIR}/include")
  target_compile_definitions(dg_igr_hdg_runtime_app_contract PUBLIC
    DG_IGR_HDG_APP_HAS_CPU_SERIAL=1
    DG_IGR_HDG_APP_HAS_CPU_RANK2=1
    DG_IGR_HDG_APP_HAS_CUDA_RANK1=$<BOOL:${MFEM_USE_CUDA}>
    DG_IGR_HDG_APP_HAS_CUDA_RANK2=$<BOOL:${MFEM_USE_CUDA}>
    DG_IGR_HDG_APP_HAS_SCALAR_IGR_CPU=$<BOOL:${_dg_igr_hdg_scalar_igr_cpu_app_available}>
    DG_IGR_HDG_APP_HAS_SCALAR_IGR_CUDA_RANK1=$<BOOL:${MFEM_USE_CUDA}>
    DG_IGR_HDG_APP_HAS_SCALAR_IGR_CUDA_RANK2=$<BOOL:${MFEM_USE_CUDA}>)
  target_compile_options(dg_igr_hdg_runtime_app_contract PRIVATE
    $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic;-Werror>)
  target_link_libraries(dg_igr_hdg_runtime_app_contract
                        PUBLIC dg_igr_hdg_runtime_common)

  dg_igr_add_hdg_cxx17_component(
    TARGET dg_igr_hdg_generic_cpu
    SOURCES ${_dg_igr_hdg_generic_cpu_sources}
    PUBLIC_LINK_LIBRARIES dg_igr_hdg_runtime_common)
  dg_igr_add_hdg_cxx17_component(
    TARGET dg_igr_hdg_sc25_oracle_cpu
    SOURCES ${_dg_igr_hdg_sc25_oracle_cpu_sources}
    PUBLIC_LINK_LIBRARIES dg_igr_hdg_runtime_common)
  # Keep the frozen oracle configured for the graph guard, but do not compile
  # it in a generic-only runtime build. Auxiliary/W8C consumers name it
  # explicitly and thereby opt it back into the default build closure.
  if(NOT _dg_igr_hdg_scalar_igr_cpu_app_available)
    set_target_properties(dg_igr_hdg_sc25_oracle_cpu PROPERTIES
      EXCLUDE_FROM_ALL TRUE)
  endif()
  if(MFEM_USE_CUDA)
    foreach(_dg_igr_hdg_cpu_component IN ITEMS
            dg_igr_hdg_generic_cpu
            dg_igr_hdg_sc25_oracle_cpu)
      set_target_properties(${_dg_igr_hdg_cpu_component} PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_EXTENSIONS OFF)
      target_compile_options(${_dg_igr_hdg_cpu_component} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
    endforeach()
  endif()

  if(NOT MFEM_USE_CUDA)
    add_library(dg_igr_hdg_main_backend_cpu STATIC
      "${PROJECT_SOURCE_DIR}/src/sim/hdg/main_backend.cpp")
    target_include_directories(dg_igr_hdg_main_backend_cpu PUBLIC
      "${PROJECT_SOURCE_DIR}/include")
    target_link_libraries(dg_igr_hdg_main_backend_cpu PUBLIC
      igr_mfem
      dg_igr_hdg_generic_cpu
      mfem
      MPI::MPI_CXX)
    target_compile_options(dg_igr_hdg_main_backend_cpu PRIVATE
      $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic;-Werror>)
    set_target_properties(dg_igr_hdg_main_backend_cpu PROPERTIES
      CXX_STANDARD 17
      CXX_STANDARD_REQUIRED ON
      CXX_EXTENSIONS OFF)
    dg_igr_hdg_require_target_link_closure_absent(
      dg_igr_hdg_main_backend_cpu dg_igr_hdg_sc25_oracle_cpu)
  endif()

  # Keep the entry as an object library so the static selector test can bind
  # it to its own mock downstreams.  Production users link the one combined
  # static archive below, avoiding an archive-order cycle between entry and
  # solver_cpu (each calls symbols defined by the other).
  add_library(dg_igr_hdg_runtime_app_entry OBJECT
              "${_dg_igr_hdg_runtime_app_entry_source}")
  target_include_directories(dg_igr_hdg_runtime_app_entry PRIVATE
                             "${PROJECT_SOURCE_DIR}/include")
  target_link_libraries(dg_igr_hdg_runtime_app_entry PRIVATE
                        dg_igr_hdg_runtime_app_contract MPI::MPI_CXX)
  target_compile_options(dg_igr_hdg_runtime_app_entry PRIVATE
    $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic;-Werror>)
  set_target_properties(dg_igr_hdg_runtime_app_entry PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF)

  add_library(dg_igr_hdg_runtime_app_cpu STATIC
              ${_dg_igr_hdg_runtime_app_cpu_sources}
              $<TARGET_OBJECTS:dg_igr_hdg_runtime_app_entry>)
  target_include_directories(dg_igr_hdg_runtime_app_cpu PRIVATE
                             "${PROJECT_SOURCE_DIR}/include")
  target_link_libraries(dg_igr_hdg_runtime_app_cpu PUBLIC
                        dg_igr_hdg_runtime_app_contract
                        dg_igr_hdg_generic_cpu
                        mfem
                        MPI::MPI_CXX)
  if(_dg_igr_hdg_scalar_igr_cpu_app_available)
    target_link_libraries(dg_igr_hdg_runtime_app_cpu PUBLIC
                          dg_igr_hdg_sc25_oracle_cpu)
  endif()
  target_compile_options(dg_igr_hdg_runtime_app_cpu PRIVATE
    $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic;-Werror>)
  set_target_properties(dg_igr_hdg_runtime_app_cpu PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF)
  if(MFEM_USE_CUDA)
    set_target_properties(dg_igr_hdg_runtime_app_cpu PROPERTIES
      CUDA_RESOLVE_DEVICE_SYMBOLS OFF)
  endif()

  target_link_libraries(dg_igr_hdg_runtime_components INTERFACE
                        dg_igr_hdg_runtime_common
                        dg_igr_hdg_runtime_app_contract
                        dg_igr_hdg_generic_cpu
                        dg_igr_hdg_sc25_oracle_cpu)

  if(MFEM_USE_CUDA)
    dg_igr_add_hdg_cuda17_component(
      TARGET dg_igr_hdg_runtime_cuda
      SOURCES ${_dg_igr_hdg_runtime_cuda_sources}
      PUBLIC_LINK_LIBRARIES dg_igr_hdg_runtime_common
      PRIVATE_LINK_LIBRARIES dg_igr_hdg_generic_cpu)
    target_link_libraries(dg_igr_hdg_runtime_components INTERFACE
                          dg_igr_hdg_runtime_cuda)

    # This application-facing CUDA adapter deliberately lives outside the
    # frozen runtime CUDA inventory above.  It is a separate profile-only
    # component over the existing generic session, not another W8A kernel or
    # allocator implementation.
    dg_igr_add_hdg_cuda17_component(
      TARGET dg_igr_hdg_main_backend_cuda
      SOURCES "${PROJECT_SOURCE_DIR}/src/sim/hdg/main_backend_cuda.cu"
      PRIVATE_LINK_LIBRARIES
        dg_igr_hdg_runtime_cuda
        dg_igr_hdg_generic_cpu)
  elseif(_dg_igr_hdg_runtime_cuda_sources)
    message(STATUS
            "HDG CUDA sources are reserved but not built by the requested CPU profile.")
  endif()

  if(MFEM_USE_CUDA)
    set(_dg_igr_hdg_runtime_expected_app_cuda_source
        "${CMAKE_CURRENT_SOURCE_DIR}/app/solver_cuda.cu")
    if(NOT _dg_igr_hdg_runtime_app_cuda_sources STREQUAL
       _dg_igr_hdg_runtime_expected_app_cuda_source)
      message(FATAL_ERROR
              "The CUDA application slice requires exactly app/solver_cuda.cu.")
    endif()
    list(APPEND _dg_igr_hdg_runtime_app_executable_sources
         ${_dg_igr_hdg_runtime_app_cuda_sources})
    add_library(dg_igr_hdg_runtime_scalar_cuda_rank2_app OBJECT
                "${CMAKE_CURRENT_SOURCE_DIR}/app/solver_cuda.cu")
    target_include_directories(dg_igr_hdg_runtime_scalar_cuda_rank2_app
                               PRIVATE "${PROJECT_SOURCE_DIR}/include")
    target_compile_definitions(
      dg_igr_hdg_runtime_scalar_cuda_rank2_app PRIVATE
      DG_IGR_HDG_SCALAR_IGR_CUDA_RANK2_TRANSLATION_UNIT=1)
    target_compile_options(dg_igr_hdg_runtime_scalar_cuda_rank2_app PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
    target_link_libraries(dg_igr_hdg_runtime_scalar_cuda_rank2_app PRIVATE
                          dg_igr_hdg_runtime_app_contract
                          dg_igr_hdg_generic_cpu
                          dg_igr_hdg_runtime_cuda
                          dg_igr_hdg_auxiliary_cpu mfem MPI::MPI_CXX)
    set_target_properties(dg_igr_hdg_runtime_scalar_cuda_rank2_app PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED ON
      CUDA_EXTENSIONS OFF
      CUDA_ARCHITECTURES 120
      CUDA_SEPARABLE_COMPILATION ON)
  elseif(_dg_igr_hdg_runtime_app_cuda_sources)
    message(STATUS
            "HDG application CUDA source is reserved but not built by the CPU profile.")
  endif()

  add_executable(dg_igr_hdg_runtime
                 ${_dg_igr_hdg_runtime_app_executable_sources})
  if(MFEM_USE_CUDA)
    target_sources(dg_igr_hdg_runtime PRIVATE
      $<TARGET_OBJECTS:dg_igr_hdg_runtime_scalar_cuda_rank2_app>)
    dg_igr_configure_hdg_cuda_final_consumer(dg_igr_hdg_runtime)
  endif()
  target_include_directories(dg_igr_hdg_runtime PRIVATE
                             "${PROJECT_SOURCE_DIR}/include")
  target_link_libraries(dg_igr_hdg_runtime PRIVATE
                        dg_igr_hdg_runtime_app_contract
                        dg_igr_hdg_runtime_app_cpu
                        mfem MPI::MPI_CXX)
  if(DG_IGR_ENABLE_HDG_AUXILIARY_EXPERIMENT OR MFEM_USE_CUDA)
    target_link_libraries(dg_igr_hdg_runtime PRIVATE dg_igr_hdg_auxiliary_cpu)
  endif()
  target_compile_options(dg_igr_hdg_runtime PRIVATE
    $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic;-Werror>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
  set_target_properties(dg_igr_hdg_runtime PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF)
  if(MFEM_USE_CUDA)
    target_link_libraries(dg_igr_hdg_runtime PRIVATE
                          dg_igr_hdg_runtime_cuda)
  endif()
endif()
