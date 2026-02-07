# Create the dgsem_shock_vortex_interaction executable
add_executable(dgsem_shock_vortex_interaction dgsem_shock_vortex_interaction.cpp)
target_link_libraries(dgsem_shock_vortex_interaction igr_mfem)
target_include_directories(dgsem_shock_vortex_interaction PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../shock_vortex_interaction)

# Create symlink to mesh file
add_custom_command(TARGET dgsem_shock_vortex_interaction POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E create_symlink
    ${CMAKE_CURRENT_SOURCE_DIR}/../shock_vortex_interaction/shock_vortex.msh
    ${CMAKE_CURRENT_BINARY_DIR}/shock_vortex.msh
    COMMENT "Creating symlink to shock_vortex.msh"
)
