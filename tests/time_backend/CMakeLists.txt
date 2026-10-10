add_executable(dg_igr_time_backend_contract_test test_time_control.cpp)
target_include_directories(dg_igr_time_backend_contract_test PRIVATE
  ${CMAKE_SOURCE_DIR}/include)
target_compile_options(dg_igr_time_backend_contract_test PRIVATE
  $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic;-Werror>)
set_target_properties(dg_igr_time_backend_contract_test PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF)

add_test(NAME dg_igr_time_backend_contract
  COMMAND dg_igr_time_backend_contract_test)
set_tests_properties(dg_igr_time_backend_contract PROPERTIES
  LABELS "time;backend;contract"
  TIMEOUT 30)

add_executable(dg_igr_rodas3_coefficients_test test_rodas3.cpp)
target_include_directories(dg_igr_rodas3_coefficients_test PRIVATE
  ${CMAKE_SOURCE_DIR}/include)
target_compile_options(dg_igr_rodas3_coefficients_test PRIVATE
  $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic;-Werror>)
set_target_properties(dg_igr_rodas3_coefficients_test PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF)

add_test(NAME dg_igr_rodas3_coefficients
  COMMAND dg_igr_rodas3_coefficients_test)
set_tests_properties(dg_igr_rodas3_coefficients PROPERTIES
  LABELS "time;backend;rodas3;dae"
  TIMEOUT 30)
