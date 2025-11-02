# Create the forward_facing_step executable
add_executable(forward_facing_step forward_facing_step.cpp)
target_link_libraries(forward_facing_step igr_mfem)

# Create symlink to mesh file
add_custom_command(TARGET forward_facing_step POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E create_symlink
    ${CMAKE_CURRENT_SOURCE_DIR}/forward_facing.msh
    ${CMAKE_CURRENT_BINARY_DIR}/forward_facing.msh
    COMMENT "Creating symlink to supersonic_cylinder.msh"
)