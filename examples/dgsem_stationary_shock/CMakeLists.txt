# Create the dgsem_stationary_shock executable
add_executable(dgsem_stationary_shock dgsem_stationary_shock.cpp)
target_link_libraries(dgsem_stationary_shock igr_mfem)
target_include_directories(dgsem_stationary_shock PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../stationary_shock)

# Create symlink to mesh file
add_custom_command(TARGET dgsem_stationary_shock POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E create_symlink
    ${CMAKE_CURRENT_SOURCE_DIR}/../stationary_shock/stationary_shock.msh
    ${CMAKE_CURRENT_BINARY_DIR}/stationary_shock.msh
    COMMENT "Creating symlink to stationary_shock.msh"
)
