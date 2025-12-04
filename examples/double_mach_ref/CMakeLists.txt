# Create the double_mach_ref executable
add_executable(double_mach_ref double_mach_ref.cpp)
target_link_libraries(double_mach_ref igr_mfem)

# Create symlink to mesh file
add_custom_command(TARGET double_mach_ref POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E create_symlink
    ${CMAKE_CURRENT_SOURCE_DIR}/double_mach_ref.msh
    ${CMAKE_CURRENT_BINARY_DIR}/double_mach_ref.msh
    COMMENT "Creating symlink to double_mach_ref.msh"
)