# Create the supersonic_cylinder executable
add_executable(supersonic_cylinder supersonic_cylinder.cpp)
target_link_libraries(supersonic_cylinder igr_mfem)

# Create symlink to mesh file
add_custom_command(TARGET supersonic_cylinder POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E create_symlink
    ${CMAKE_CURRENT_SOURCE_DIR}/supersonic_cylinder.msh
    ${CMAKE_CURRENT_BINARY_DIR}/supersonic_cylinder.msh
    COMMENT "Creating symlink to supersonic_cylinder.msh"
)