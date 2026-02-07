# Create the dgsem_supersonic_cylinder executable
add_executable(dgsem_supersonic_cylinder dgsem_supersonic_cylinder.cpp)
target_link_libraries(dgsem_supersonic_cylinder igr_mfem)
target_include_directories(dgsem_supersonic_cylinder PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../supersonic_example)

# Create symlink to mesh file
add_custom_command(TARGET dgsem_supersonic_cylinder POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E create_symlink
    ${CMAKE_CURRENT_SOURCE_DIR}/../supersonic_example/supersonic_cylinder.msh
    ${CMAKE_CURRENT_BINARY_DIR}/supersonic_cylinder.msh
    COMMENT "Creating symlink to supersonic_cylinder.msh"
)
