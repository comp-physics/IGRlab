if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  message(FATAL_ERROR
    "The smooth-Euler HDG device experiment is parent-build-only; configure from the repository root")
endif()

option(DG_IGR_ENABLE_HDG_EULER_DEVICE_EXPERIMENT
  "Build the default-off smooth-Euler HDG CPU-fallback/device experiment" OFF)
if(NOT DG_IGR_ENABLE_HDG_EULER_DEVICE_EXPERIMENT)
  return()
endif()

if(NOT TARGET mfem OR NOT TARGET igr_mfem)
  message(FATAL_ERROR "smooth-Euler HDG device experiment requires parent MFEM discovery")
endif()

set(_dg_igr_hdg_euler_device_source
  ${DG_IGR_HDG_REPOSITORY_ROOT}/src/numerics/hdg/euler/smooth_euler_device.cpp)
if(MFEM_USE_CUDA)
  set_source_files_properties(${_dg_igr_hdg_euler_device_source}
    PROPERTIES LANGUAGE CUDA)
endif()
add_library(dg_igr_hdg_smooth_euler_device_cxx17_check OBJECT
  ${_dg_igr_hdg_euler_device_source})
add_executable(dg_igr_hdg_smooth_euler_device_test
  test_smooth_euler_device.cpp)
target_include_directories(dg_igr_hdg_smooth_euler_device_cxx17_check PRIVATE
  ${DG_IGR_HDG_REPOSITORY_ROOT}/include)
target_include_directories(dg_igr_hdg_smooth_euler_device_test PRIVATE
  ${DG_IGR_HDG_REPOSITORY_ROOT}/include)
target_link_libraries(dg_igr_hdg_smooth_euler_device_cxx17_check PRIVATE
  igr_mfem mfem)
target_link_libraries(dg_igr_hdg_smooth_euler_device_test PRIVATE igr_mfem mfem)
foreach(_target IN ITEMS
    dg_igr_hdg_smooth_euler_device_cxx17_check
    dg_igr_hdg_smooth_euler_device_test)
  target_compile_options(${_target} PRIVATE
    $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic;-Werror>)
  set_target_properties(${_target} PROPERTIES
    CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF
    CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_EXTENSIONS OFF)
endforeach()
if(MFEM_USE_CUDA)
  foreach(_target IN ITEMS
      dg_igr_hdg_smooth_euler_device_cxx17_check
      dg_igr_hdg_smooth_euler_device_test)
    set_target_properties(${_target} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_compile_options(${_target} PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
  endforeach()
endif()

if(MFEM_USE_CUDA)
  add_test(NAME dg_igr_hdg_smooth_euler_device_cuda
    COMMAND dg_igr_hdg_smooth_euler_device_test)
  set_tests_properties(dg_igr_hdg_smooth_euler_device_cuda PROPERTIES
    LABELS "hdg;euler;gpu;cuda;device;experimental"
    RESOURCE_LOCK dg_igr_sm120
    TIMEOUT 600)
elseif(NOT MFEM_USE_HIP)
  add_test(NAME dg_igr_hdg_smooth_euler_device_cpu_fallback
    COMMAND dg_igr_hdg_smooth_euler_device_test)
  set_tests_properties(dg_igr_hdg_smooth_euler_device_cpu_fallback PROPERTIES
    LABELS "hdg;euler;cpu;device;fallback;experimental"
    TIMEOUT 600)
endif()
