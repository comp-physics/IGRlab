# Create the supersonic_cylinder_w_igr executable
add_executable(supersonic_cylinder_w_igr supersonic_cylinder_w_igr.cpp)
target_link_libraries(supersonic_cylinder_w_igr igr_mfem)

# Create symlink to mesh file
add_custom_command(TARGET supersonic_cylinder_w_igr POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E create_symlink
    ${CMAKE_CURRENT_SOURCE_DIR}/supersonic_cylinder.msh
    ${CMAKE_CURRENT_BINARY_DIR}/supersonic_cylinder.msh
    COMMENT "Creating symlink to supersonic_cylinder.msh"
)