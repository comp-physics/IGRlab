# Create the dgsem_smooth_advection_periodic executable
add_executable(dgsem_smooth_advection_periodic dgsem_smooth_advection_periodic.cpp)
if(TARGET dg_igr_hdg_main_backend_cpu)
  target_link_libraries(dgsem_smooth_advection_periodic
    dg_igr_hdg_main_backend_cpu)
  target_compile_definitions(dgsem_smooth_advection_periodic PRIVATE
    DG_IGR_HAS_HDG_MAIN_BACKEND=1)
else()
  target_link_libraries(dgsem_smooth_advection_periodic igr_mfem)
  target_compile_definitions(dgsem_smooth_advection_periodic PRIVATE
    DG_IGR_HAS_HDG_MAIN_BACKEND=0)
endif()
