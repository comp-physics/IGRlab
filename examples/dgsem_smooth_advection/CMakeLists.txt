# Create the dgsem_smooth_advection executable
add_executable(dgsem_smooth_advection dgsem_smooth_advection.cpp)
if(TARGET dg_igr_hdg_main_backend_cuda)
  dg_igr_configure_hdg_cuda_final_consumer(dgsem_smooth_advection)
  target_link_libraries(dgsem_smooth_advection PRIVATE
    dg_igr_hdg_main_backend_cuda
    igr_mfem)
  target_compile_definitions(dgsem_smooth_advection PRIVATE
    DG_IGR_HAS_HDG_MAIN_BACKEND=1
    DG_IGR_HAS_HDG_MAIN_BACKEND_CUDA=1)
elseif(TARGET dg_igr_hdg_main_backend_cpu)
  target_link_libraries(dgsem_smooth_advection dg_igr_hdg_main_backend_cpu)
  target_compile_definitions(dgsem_smooth_advection PRIVATE
    DG_IGR_HAS_HDG_MAIN_BACKEND=1
    DG_IGR_HAS_HDG_MAIN_BACKEND_CUDA=0)
else()
  target_link_libraries(dgsem_smooth_advection igr_mfem)
  target_compile_definitions(dgsem_smooth_advection PRIVATE
    DG_IGR_HAS_HDG_MAIN_BACKEND=0
    DG_IGR_HAS_HDG_MAIN_BACKEND_CUDA=0)
endif()

# Create symlink to mesh file
add_custom_command(TARGET dgsem_smooth_advection POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E create_symlink
    ${CMAKE_CURRENT_SOURCE_DIR}/smooth_advection.msh
    ${CMAKE_CURRENT_BINARY_DIR}/smooth_advection.msh
    COMMENT "Creating symlink to smooth_advection.msh"
)
