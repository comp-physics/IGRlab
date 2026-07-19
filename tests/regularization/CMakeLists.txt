cmake_minimum_required(VERSION 3.14)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(dg_igr_regularization_contract LANGUAGES CXX)
  find_package(MFEM REQUIRED CONFIG)
  if(MFEM_USE_MPI)
    find_package(MPI REQUIRED COMPONENTS CXX)
  endif()
  include(CTest)
endif()

get_filename_component(DG_IGR_REPOSITORY_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.."
                       ABSOLUTE)

add_executable(dg_igr_regularization_contract_test
  test_regularization_contract.cpp
)

if(TARGET igr_mfem)
  target_link_libraries(dg_igr_regularization_contract_test PRIVATE igr_mfem mfem)
else()
  target_sources(dg_igr_regularization_contract_test PRIVATE
    "${DG_IGR_REPOSITORY_ROOT}/src/physics/regularization/validation.cpp"
  )
  target_link_libraries(dg_igr_regularization_contract_test PRIVATE mfem)
  if(MFEM_USE_MPI)
    target_link_libraries(dg_igr_regularization_contract_test PRIVATE MPI::MPI_CXX)
  endif()
endif()

target_include_directories(dg_igr_regularization_contract_test PRIVATE
  "${DG_IGR_REPOSITORY_ROOT}/include"
)
set_target_properties(dg_igr_regularization_contract_test PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
)

add_test(NAME dg_igr_regularization_contract_cpu
         COMMAND dg_igr_regularization_contract_test)
set_tests_properties(dg_igr_regularization_contract_cpu PROPERTIES
  LABELS "regularization;contract;cpu"
  TIMEOUT 30
)
