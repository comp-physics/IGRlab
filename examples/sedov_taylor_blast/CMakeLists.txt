# Create the sedov_taylor_blast executable
add_executable(sedov_taylor_blast sedov_taylor_blast.cpp)
target_link_libraries(sedov_taylor_blast igr_mfem)

# Create symlink to mesh file
add_custom_command(TARGET sedov_taylor_blast POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E create_symlink
    ${CMAKE_CURRENT_SOURCE_DIR}/sedov_taylor.msh
    ${CMAKE_CURRENT_BINARY_DIR}/sedov_taylor.msh
    COMMENT "Creating symlink to sedov_taylor.msh"
)