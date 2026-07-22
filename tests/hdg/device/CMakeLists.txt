add_executable(dg_igr_hdg_device_core_test test_device_core.cpp)
target_include_directories(dg_igr_hdg_device_core_test PRIVATE
  ${DG_IGR_HDG_REPOSITORY_ROOT}/include)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  target_sources(dg_igr_hdg_device_core_test PRIVATE
    ${DG_IGR_HDG_REPOSITORY_ROOT}/src/numerics/hdg/core/host_reference.cpp
    ${DG_IGR_HDG_REPOSITORY_ROOT}/src/numerics/hdg/device/device_core.cpp)
  target_link_libraries(dg_igr_hdg_device_core_test PRIVATE mfem)
else()
  target_link_libraries(dg_igr_hdg_device_core_test PRIVATE igr_mfem mfem)
endif()
target_compile_options(dg_igr_hdg_device_core_test PRIVATE
  $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic;-Werror>)
set_target_properties(dg_igr_hdg_device_core_test PROPERTIES
  CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  if(MFEM_USE_CUDA)
    # CUDA is runtime-validated on the SM120 resource.  The HIP branch below
    # is structurally gated here but is not claimed as validated on this host.
    add_test(NAME dg_igr_hdg_device_core_cuda
      COMMAND dg_igr_hdg_device_core_test)
    set_tests_properties(dg_igr_hdg_device_core_cuda PROPERTIES
      LABELS "hdg;gpu;cuda;device"
      RESOURCE_LOCK dg_igr_sm120
      TIMEOUT 30)
  elseif(MFEM_USE_HIP)
    add_test(NAME dg_igr_hdg_device_core_hip
      COMMAND dg_igr_hdg_device_core_test)
    set_tests_properties(dg_igr_hdg_device_core_hip PROPERTIES
      LABELS "hdg;gpu;hip;device"
      RESOURCE_LOCK dg_igr_gpu
      TIMEOUT 30)
  else()
    add_test(NAME dg_igr_hdg_device_core_cpu_fallback
      COMMAND dg_igr_hdg_device_core_test)
    set_tests_properties(dg_igr_hdg_device_core_cpu_fallback PROPERTIES
      LABELS "hdg;cpu;device;fallback" TIMEOUT 30)
  endif()
endif()
