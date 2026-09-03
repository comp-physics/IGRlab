# Create the sedov_taylor_blast executable
add_executable(sedov_taylor_blast sedov_taylor_blast.cpp)
if(TARGET dg_igr_hdg_main_backend_cuda)
  dg_igr_configure_hdg_cuda_final_consumer(sedov_taylor_blast)
  target_link_libraries(sedov_taylor_blast PRIVATE
    dg_igr_hdg_main_backend_cuda
    igr_mfem)
  target_compile_definitions(sedov_taylor_blast PRIVATE
    DG_IGR_HAS_HDG_MAIN_BACKEND=1
    DG_IGR_HAS_HDG_MAIN_BACKEND_CUDA=1)
elseif(TARGET dg_igr_hdg_main_backend_cpu)
  target_link_libraries(sedov_taylor_blast dg_igr_hdg_main_backend_cpu)
  target_compile_definitions(sedov_taylor_blast PRIVATE
    DG_IGR_HAS_HDG_MAIN_BACKEND=1
    DG_IGR_HAS_HDG_MAIN_BACKEND_CUDA=0)
else()
  target_link_libraries(sedov_taylor_blast igr_mfem)
  target_compile_definitions(sedov_taylor_blast PRIVATE
    DG_IGR_HAS_HDG_MAIN_BACKEND=0
    DG_IGR_HAS_HDG_MAIN_BACKEND_CUDA=0)
endif()

# Create symlink to mesh file
add_custom_command(TARGET sedov_taylor_blast POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E create_symlink
    ${CMAKE_CURRENT_SOURCE_DIR}/sedov_taylor.msh
    ${CMAKE_CURRENT_BINARY_DIR}/sedov_taylor.msh
    COMMENT "Creating symlink to sedov_taylor.msh"
)
