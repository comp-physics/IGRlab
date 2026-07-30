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

if(DG_IGR_ENABLE_HDG_RUNTIME)
  if(MFEM_USE_CUDA)
    set(_dg_igr_hdg_runtime_cuda_expected_source
        "${CMAKE_CURRENT_SOURCE_DIR}/cuda/persistent_smooth_euler.cu")
    list(LENGTH _dg_igr_hdg_runtime_cuda_sources
         _dg_igr_hdg_runtime_cuda_source_count)
    list(FIND _dg_igr_hdg_runtime_cuda_sources
         "${_dg_igr_hdg_runtime_cuda_expected_source}"
         _dg_igr_hdg_runtime_cuda_expected_index)
    if(NOT _dg_igr_hdg_runtime_cuda_source_count EQUAL 1 OR
       _dg_igr_hdg_runtime_cuda_expected_index EQUAL -1)
      message(FATAL_ERROR
              "The frozen W8A CUDA component requires exactly cuda/persistent_smooth_euler.cu.")
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
      "${CMAKE_CURRENT_SOURCE_DIR}/app/main.cpp"
      "${CMAKE_CURRENT_SOURCE_DIR}/app/solver_cpu.cpp")
  set(_dg_igr_hdg_runtime_app_expected_sources
      ${_dg_igr_hdg_runtime_app_contract_sources}
      ${_dg_igr_hdg_runtime_app_executable_sources})
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
    DG_IGR_HDG_APP_HAS_CUDA_RANK2=$<BOOL:${MFEM_USE_CUDA}>)
  target_compile_options(dg_igr_hdg_runtime_app_contract PRIVATE
    $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic;-Werror>)
  target_link_libraries(dg_igr_hdg_runtime_app_contract
                        PUBLIC dg_igr_hdg_runtime_common)

  dg_igr_add_hdg_cxx17_component(
    TARGET dg_igr_hdg_runtime_cpu
    SOURCES ${_dg_igr_hdg_runtime_cpu_sources}
    PUBLIC_LINK_LIBRARIES dg_igr_hdg_runtime_common)

  target_link_libraries(dg_igr_hdg_runtime_components INTERFACE
                        dg_igr_hdg_runtime_common
                        dg_igr_hdg_runtime_app_contract
                        dg_igr_hdg_runtime_cpu)

  if(MFEM_USE_CUDA)
    dg_igr_add_hdg_cuda17_component(
      TARGET dg_igr_hdg_runtime_cuda
      SOURCES ${_dg_igr_hdg_runtime_cuda_sources}
      PUBLIC_LINK_LIBRARIES dg_igr_hdg_runtime_common)
    target_link_libraries(dg_igr_hdg_runtime_components INTERFACE
                          dg_igr_hdg_runtime_cuda)
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
  elseif(_dg_igr_hdg_runtime_app_cuda_sources)
    message(STATUS
            "HDG application CUDA source is reserved but not built by the CPU profile.")
  endif()

  add_executable(dg_igr_hdg_runtime
                 ${_dg_igr_hdg_runtime_app_executable_sources})
  target_include_directories(dg_igr_hdg_runtime PRIVATE
                             "${PROJECT_SOURCE_DIR}/include")
  target_link_libraries(dg_igr_hdg_runtime PRIVATE
                        dg_igr_hdg_runtime_app_contract
                        dg_igr_hdg_runtime_cpu mfem MPI::MPI_CXX)
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
    set_target_properties(dg_igr_hdg_runtime PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED ON
      CUDA_EXTENSIONS OFF
      CUDA_ARCHITECTURES 120
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  endif()
endif()

if(DG_IGR_ENABLE_HDG_AUXILIARY_EXPERIMENT)
  file(GLOB_RECURSE _dg_igr_hdg_auxiliary_sources CONFIGURE_DEPENDS
       "${CMAKE_CURRENT_SOURCE_DIR}/../auxiliary/*.cpp")
  dg_igr_add_hdg_cxx17_component(
    TARGET dg_igr_hdg_auxiliary_cpu
    SOURCES ${_dg_igr_hdg_auxiliary_sources})
endif()
