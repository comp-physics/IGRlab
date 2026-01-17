# Create the stationary_shock executable
add_executable(stationary_shock stationary_shock.cpp)
target_link_libraries(stationary_shock igr_mfem)

# Create symlink to mesh file
add_custom_command(TARGET stationary_shock POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E create_symlink
    ${CMAKE_CURRENT_SOURCE_DIR}/stationary_shock.msh
    ${CMAKE_CURRENT_BINARY_DIR}/stationary_shock.msh
    COMMENT "Creating symlink to stationary_shock.msh"
)