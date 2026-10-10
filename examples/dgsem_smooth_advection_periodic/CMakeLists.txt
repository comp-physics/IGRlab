# Create the dgsem_smooth_advection_periodic executable
add_executable(dgsem_smooth_advection_periodic dgsem_smooth_advection_periodic.cpp)
if(TARGET dg_igr_hdg_main_backend_cuda)
  target_link_libraries(dgsem_smooth_advection_periodic
    dg_igr_hdg_main_backend_cuda
    igr_mfem)
  target_compile_definitions(dgsem_smooth_advection_periodic PRIVATE
    DG_IGR_HAS_HDG_MAIN_BACKEND=1
    DG_IGR_HAS_HDG_MAIN_BACKEND_CUDA=1)
  # Keep device resolution at the final consumer so all CUDA archives,
  # including igr_mfem, participate in one device-link step.
  set_target_properties(dgsem_smooth_advection_periodic PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_EXTENSIONS OFF
    CUDA_ARCHITECTURES "${DG_IGR_HDG_NATIVE_CUDA_ARCH}"
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON)
elseif(TARGET dg_igr_hdg_main_backend_cpu)
  target_link_libraries(dgsem_smooth_advection_periodic
    dg_igr_hdg_main_backend_cpu)
  target_compile_definitions(dgsem_smooth_advection_periodic PRIVATE
    DG_IGR_HAS_HDG_MAIN_BACKEND=1
    DG_IGR_HAS_HDG_MAIN_BACKEND_CUDA=0)
else()
  target_link_libraries(dgsem_smooth_advection_periodic igr_mfem)
  target_compile_definitions(dgsem_smooth_advection_periodic PRIVATE
    DG_IGR_HAS_HDG_MAIN_BACKEND=0
    DG_IGR_HAS_HDG_MAIN_BACKEND_CUDA=0)
endif()
