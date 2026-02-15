add_executable(inoue_hattori_stationary_shock_1dy
               inoue_hattori_stationary_shock_1dy.cpp)
target_link_libraries(inoue_hattori_stationary_shock_1dy igr_mfem)
target_include_directories(inoue_hattori_stationary_shock_1dy PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../shock_vortex_interaction)
