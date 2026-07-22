get_filename_component(DG_IGR_HDG_EULER_ROOT
  "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)

add_executable(dg_igr_hdg_smooth_euler_cpu_test
  test_smooth_euler.cpp
  ${DG_IGR_HDG_EULER_ROOT}/src/numerics/hdg/euler/smooth_euler.cpp)
target_include_directories(dg_igr_hdg_smooth_euler_cpu_test PRIVATE
  ${DG_IGR_HDG_EULER_ROOT}/include)
target_compile_options(dg_igr_hdg_smooth_euler_cpu_test PRIVATE
  $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic;-Werror>)
set_target_properties(dg_igr_hdg_smooth_euler_cpu_test PROPERTIES
  CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  find_package(MFEM 4.9 CONFIG REQUIRED)
  target_sources(dg_igr_hdg_smooth_euler_cpu_test PRIVATE
    ${DG_IGR_HDG_EULER_ROOT}/src/numerics/hdg/core/host_reference.cpp)
  target_link_libraries(dg_igr_hdg_smooth_euler_cpu_test PRIVATE mfem)
else()
  target_link_libraries(dg_igr_hdg_smooth_euler_cpu_test PRIVATE igr_mfem mfem)
  add_test(NAME dg_igr_hdg_smooth_euler_serial_cpu
    COMMAND dg_igr_hdg_smooth_euler_cpu_test)
  set_tests_properties(dg_igr_hdg_smooth_euler_serial_cpu PROPERTIES
    LABELS "hdg;euler;cpu;experimental;serial" TIMEOUT 600)
endif()
