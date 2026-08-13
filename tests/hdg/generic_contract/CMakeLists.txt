if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  cmake_minimum_required(VERSION 3.20)
  project(dg_igr_hdg_generic_contract LANGUAGES CXX)

  find_package(MFEM 4.9 CONFIG REQUIRED)
  find_package(MPI REQUIRED COMPONENTS CXX)
endif()

get_filename_component(DG_IGR_HDG_GENERIC_REPOSITORY_ROOT
  "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)

add_executable(dg_igr_hdg_generic_contract_test test_contract.cpp)
target_include_directories(dg_igr_hdg_generic_contract_test PRIVATE
  ${DG_IGR_HDG_GENERIC_REPOSITORY_ROOT}/include)
target_link_libraries(dg_igr_hdg_generic_contract_test PRIVATE mfem MPI::MPI_CXX)
target_compile_options(dg_igr_hdg_generic_contract_test PRIVATE
  $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic;-Werror>)
set_target_properties(dg_igr_hdg_generic_contract_test PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF)

if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  add_test(NAME dg_igr_hdg_generic_contract
    COMMAND dg_igr_hdg_generic_contract_test)
  set_tests_properties(dg_igr_hdg_generic_contract PROPERTIES
    LABELS "hdg;generic;contract"
    TIMEOUT 30)
endif()
