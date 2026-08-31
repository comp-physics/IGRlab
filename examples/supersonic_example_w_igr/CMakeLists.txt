# Create the supersonic_cylinder_w_igr executable
add_executable(supersonic_cylinder_w_igr supersonic_cylinder_w_igr.cpp)
if(TARGET dg_igr_hdg_main_backend_cuda)
  dg_igr_configure_hdg_cuda_final_consumer(supersonic_cylinder_w_igr)
  target_link_libraries(supersonic_cylinder_w_igr PRIVATE
    dg_igr_hdg_main_backend_cuda
    igr_mfem)
  target_compile_definitions(supersonic_cylinder_w_igr PRIVATE
    DG_IGR_HAS_HDG_MAIN_BACKEND=1
    DG_IGR_HAS_HDG_MAIN_BACKEND_CUDA=1)
elseif(TARGET dg_igr_hdg_main_backend_cpu)
  target_link_libraries(supersonic_cylinder_w_igr
    dg_igr_hdg_main_backend_cpu)
  target_compile_definitions(supersonic_cylinder_w_igr PRIVATE
    DG_IGR_HAS_HDG_MAIN_BACKEND=1
    DG_IGR_HAS_HDG_MAIN_BACKEND_CUDA=0)
else()
  target_link_libraries(supersonic_cylinder_w_igr igr_mfem)
  target_compile_definitions(supersonic_cylinder_w_igr PRIVATE
    DG_IGR_HAS_HDG_MAIN_BACKEND=0
    DG_IGR_HAS_HDG_MAIN_BACKEND_CUDA=0)
endif()

# Create symlink to mesh file
add_custom_command(TARGET supersonic_cylinder_w_igr POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E create_symlink
    ${CMAKE_CURRENT_SOURCE_DIR}/supersonic_cylinder.msh
    ${CMAKE_CURRENT_BINARY_DIR}/supersonic_cylinder.msh
    COMMENT "Creating symlink to supersonic_cylinder.msh"
)
