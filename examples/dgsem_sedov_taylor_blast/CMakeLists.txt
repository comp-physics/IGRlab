# Create the dgsem_sedov_taylor_blast executable
add_executable(dgsem_sedov_taylor_blast dgsem_sedov_taylor_blast.cpp)
target_link_libraries(dgsem_sedov_taylor_blast igr_mfem)
target_include_directories(dgsem_sedov_taylor_blast PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../sedov_taylor_blast)

# Create symlink to mesh file
add_custom_command(TARGET dgsem_sedov_taylor_blast POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E create_symlink
    ${CMAKE_CURRENT_SOURCE_DIR}/../sedov_taylor_blast/sedov_taylor.msh
    ${CMAKE_CURRENT_BINARY_DIR}/sedov_taylor.msh
    COMMENT "Creating symlink to sedov_taylor.msh"
)
