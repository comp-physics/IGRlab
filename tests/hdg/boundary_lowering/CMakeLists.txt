add_executable(dg_igr_hdg_boundary_lowering_test test_boundary_lowering.cpp)
target_include_directories(dg_igr_hdg_boundary_lowering_test PRIVATE
  ${DG_IGR_HDG_REPOSITORY_ROOT}/include)
if(TARGET igr_mfem)
  target_link_libraries(dg_igr_hdg_boundary_lowering_test PRIVATE igr_mfem mfem)
else()
  target_sources(dg_igr_hdg_boundary_lowering_test PRIVATE
    ${DG_IGR_HDG_REPOSITORY_ROOT}/src/physics/bc/boundary_state.cpp
    ${DG_IGR_HDG_REPOSITORY_ROOT}/src/numerics/hdg/generic/boundary_context.cpp
    ${DG_IGR_HDG_REPOSITORY_ROOT}/src/numerics/hdg/generic/hdg_boundary_wiring.cpp)
  target_link_libraries(dg_igr_hdg_boundary_lowering_test PRIVATE mfem)
endif()
target_compile_options(dg_igr_hdg_boundary_lowering_test PRIVATE
  $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic;-Werror;-Wno-missing-field-initializers>)
set_target_properties(dg_igr_hdg_boundary_lowering_test PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF)

add_test(NAME dg_igr_hdg_boundary_lowering
  COMMAND dg_igr_hdg_boundary_lowering_test)
set_tests_properties(dg_igr_hdg_boundary_lowering PROPERTIES
  LABELS "hdg;generic;boundary"
  TIMEOUT 30)
