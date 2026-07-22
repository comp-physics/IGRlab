cmake_minimum_required(VERSION 3.20)
project(dg_igr_hdg_contract_negative LANGUAGES CXX)

if(NOT DEFINED HDG_NEGATIVE_SOURCE OR NOT DEFINED DG_IGR_REPOSITORY_ROOT)
  message(FATAL_ERROR "HDG negative compile test was not configured by tests/hdg")
endif()

add_library(dg_igr_hdg_contract_negative STATIC "${HDG_NEGATIVE_SOURCE}")
target_include_directories(dg_igr_hdg_contract_negative PRIVATE
  "${DG_IGR_REPOSITORY_ROOT}/include")
target_compile_options(dg_igr_hdg_contract_negative PRIVATE
  $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic;-Werror>)
set_target_properties(dg_igr_hdg_contract_negative PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF)
