if(DG_IGR_ENABLE_HDG_RUNTIME)
  add_executable(hdg_smooth_periodic hdg_smooth_periodic.cpp)
  target_include_directories(hdg_smooth_periodic PRIVATE
    "${PROJECT_SOURCE_DIR}/include")
  if(MFEM_USE_CUDA)
    target_sources(hdg_smooth_periodic PRIVATE
      "${PROJECT_SOURCE_DIR}/src/numerics/hdg/runtime/app/solver_cuda.cu")
    dg_igr_configure_hdg_cuda_final_consumer(hdg_smooth_periodic)
    target_link_libraries(hdg_smooth_periodic PRIVATE
      dg_igr_hdg_runtime_app_contract
      dg_igr_hdg_runtime_app_cpu
      dg_igr_hdg_runtime_cuda
      dg_igr_hdg_auxiliary_cpu
      mfem
      MPI::MPI_CXX)
    target_compile_definitions(hdg_smooth_periodic PRIVATE
      DG_IGR_HDG_SMOOTH_PERIODIC_HAS_CUDA=1)
  else()
    target_link_libraries(hdg_smooth_periodic PRIVATE
      dg_igr_hdg_runtime_app_cpu
      mfem
      MPI::MPI_CXX)
    target_compile_definitions(hdg_smooth_periodic PRIVATE
      DG_IGR_HDG_SMOOTH_PERIODIC_HAS_CUDA=0)
  endif()
  target_compile_options(hdg_smooth_periodic PRIVATE
    $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic;-Werror>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
  set_target_properties(hdg_smooth_periodic PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF)
endif()
