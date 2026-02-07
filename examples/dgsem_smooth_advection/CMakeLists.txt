# Create the dgsem_smooth_advection executable
add_executable(dgsem_smooth_advection dgsem_smooth_advection.cpp)
target_link_libraries(dgsem_smooth_advection igr_mfem)

# Create symlink to mesh file
add_custom_command(TARGET dgsem_smooth_advection POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E create_symlink
    ${CMAKE_CURRENT_SOURCE_DIR}/smooth_advection.msh
    ${CMAKE_CURRENT_BINARY_DIR}/smooth_advection.msh
    COMMENT "Creating symlink to smooth_advection.msh"
)
