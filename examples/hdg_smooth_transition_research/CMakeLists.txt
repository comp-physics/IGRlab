# Generic-main W8C smooth-transition research smoke case.  It links only the
# profile-selected main HDG adapter; it does not depend on the legacy runtime.
add_executable(hdg_smooth_transition_research
  hdg_smooth_transition_research.cpp)

if(TARGET dg_igr_hdg_main_backend_cuda)
  target_link_libraries(hdg_smooth_transition_research PRIVATE
    dg_igr_hdg_main_backend_cuda
    igr_mfem)
  target_compile_definitions(hdg_smooth_transition_research PRIVATE
    DG_IGR_HAS_HDG_MAIN_BACKEND=1
    DG_IGR_HAS_HDG_MAIN_BACKEND_CUDA=1)
  set_target_properties(hdg_smooth_transition_research PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_EXTENSIONS OFF
    CUDA_ARCHITECTURES "${DG_IGR_HDG_NATIVE_CUDA_ARCH}"
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON)
elseif(TARGET dg_igr_hdg_main_backend_cpu)
  target_link_libraries(hdg_smooth_transition_research PRIVATE
    dg_igr_hdg_main_backend_cpu)
  target_compile_definitions(hdg_smooth_transition_research PRIVATE
    DG_IGR_HAS_HDG_MAIN_BACKEND=1
    DG_IGR_HAS_HDG_MAIN_BACKEND_CUDA=0)
else()
  target_link_libraries(hdg_smooth_transition_research PRIVATE igr_mfem)
  target_compile_definitions(hdg_smooth_transition_research PRIVATE
    DG_IGR_HAS_HDG_MAIN_BACKEND=0
    DG_IGR_HAS_HDG_MAIN_BACKEND_CUDA=0)
endif()
