# Create the shock_vortex_interaction executable
add_executable(shock_vortex_interaction shock_vortex_interaction.cpp)
target_link_libraries(shock_vortex_interaction igr_mfem)

# Create symlink to mesh file
add_custom_command(TARGET shock_vortex_interaction POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E create_symlink
    ${CMAKE_CURRENT_SOURCE_DIR}/shock_vortex.msh
    ${CMAKE_CURRENT_BINARY_DIR}/shock_vortex.msh
    COMMENT "Creating symlink to shock_vortex.msh"
)