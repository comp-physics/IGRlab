# Focused registration owned by the case-context foundation.  The campaign
# integration layer may add this directory to a composite test matrix.
add_executable(dg_igr_case_context_compile_test test_case_context_compile.cpp)
target_link_libraries(dg_igr_case_context_compile_test PRIVATE igr_mfem mfem)
target_include_directories(dg_igr_case_context_compile_test PRIVATE
  ${CMAKE_SOURCE_DIR}/include
)
add_test(NAME dg_igr_case_context_compile
  COMMAND dg_igr_case_context_compile_test
)
set_tests_properties(dg_igr_case_context_compile PROPERTIES
  LABELS "case-context;mesh;boundary-plan"
  TIMEOUT 30
)
