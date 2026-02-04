# Create the dgsem_forward_facing_step executable
add_executable(dgsem_forward_facing_step dgsem_forward_facing_step.cpp)
target_link_libraries(dgsem_forward_facing_step igr_mfem)
target_include_directories(dgsem_forward_facing_step PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../forward_facing_step)

# Create symlink to mesh file
add_custom_command(TARGET dgsem_forward_facing_step POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E create_symlink
    ${CMAKE_CURRENT_SOURCE_DIR}/../forward_facing_step/forward_facing.msh
    ${CMAKE_CURRENT_BINARY_DIR}/forward_facing.msh
    COMMENT "Creating symlink to forward_facing.msh"
)
