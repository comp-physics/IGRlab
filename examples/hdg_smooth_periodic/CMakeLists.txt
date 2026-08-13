if(DG_IGR_ENABLE_HDG_RUNTIME AND NOT MFEM_USE_CUDA)
  add_executable(hdg_smooth_periodic hdg_smooth_periodic.cpp)
  target_include_directories(hdg_smooth_periodic PRIVATE
    "${PROJECT_SOURCE_DIR}/include")
  target_link_libraries(hdg_smooth_periodic PRIVATE
    dg_igr_hdg_runtime_app_cpu
    mfem
    MPI::MPI_CXX)
  target_compile_options(hdg_smooth_periodic PRIVATE
    $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic;-Werror>)
  set_target_properties(hdg_smooth_periodic PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF)
endif()
