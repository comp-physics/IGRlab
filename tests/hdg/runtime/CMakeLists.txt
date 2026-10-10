function(_dg_igr_require_hdg_cxx17_target target)
  if(NOT TARGET "${target}")
    message(FATAL_ERROR "Required HDG component target is missing: ${target}")
  endif()
  get_target_property(_dg_igr_hdg_type "${target}" TYPE)
  get_target_property(_dg_igr_hdg_standard "${target}" CXX_STANDARD)
  get_target_property(_dg_igr_hdg_required "${target}" CXX_STANDARD_REQUIRED)
  get_target_property(_dg_igr_hdg_extensions "${target}" CXX_EXTENSIONS)
  if(NOT _dg_igr_hdg_type STREQUAL "STATIC_LIBRARY" OR
     NOT _dg_igr_hdg_standard STREQUAL "17" OR
     NOT _dg_igr_hdg_required OR
     _dg_igr_hdg_extensions)
    message(FATAL_ERROR
            "${target} is not a strict static C++17 component library.")
  endif()
endfunction()

if(DG_IGR_ENABLE_HDG_RUNTIME)
  if(NOT TARGET dg_igr_hdg_runtime_components)
    message(FATAL_ERROR "Required HDG component aggregate target is missing.")
  endif()
  if(NOT TARGET dg_igr_hdg_runtime)
    message(FATAL_ERROR
            "The production dg_igr_hdg_runtime executable is required.")
  endif()
  _dg_igr_require_hdg_cxx17_target(dg_igr_hdg_runtime_common)
  _dg_igr_require_hdg_cxx17_target(dg_igr_hdg_runtime_app_contract)
  _dg_igr_require_hdg_cxx17_target(dg_igr_hdg_runtime_app_cpu)
  _dg_igr_require_hdg_cxx17_target(dg_igr_hdg_generic_cpu)
  _dg_igr_require_hdg_cxx17_target(dg_igr_hdg_sc25_oracle_cpu)
  get_target_property(_dg_igr_hdg_sc25_cpu_excluded
                      dg_igr_hdg_sc25_oracle_cpu EXCLUDE_FROM_ALL)
  if(NOT _dg_igr_hdg_sc25_cpu_excluded)
    message(FATAL_ERROR
            "The frozen SC25 CPU oracle must be excluded from production builds.")
  endif()
  dg_igr_hdg_require_target_link_closure_absent(
    dg_igr_hdg_generic_cpu dg_igr_hdg_sc25_oracle_cpu)
  dg_igr_hdg_require_target_link_closure_absent(
    dg_igr_hdg_runtime_components dg_igr_hdg_sc25_oracle_cpu)
  dg_igr_hdg_require_target_link_closure_absent(
    dg_igr_hdg_runtime_app_cpu dg_igr_hdg_sc25_oracle_cpu)

  if(MFEM_USE_CUDA)
    foreach(_dg_igr_hdg_cxx_archive IN ITEMS
            dg_igr_hdg_runtime_common
            dg_igr_hdg_generic_cpu
            dg_igr_hdg_sc25_oracle_cpu
            dg_igr_hdg_auxiliary_cpu
            dg_igr_hdg_runtime_app_cpu)
      get_target_property(_dg_igr_hdg_cxx_archive_resolve
                          "${_dg_igr_hdg_cxx_archive}"
                          CUDA_RESOLVE_DEVICE_SYMBOLS)
      if(NOT _dg_igr_hdg_cxx_archive_resolve STREQUAL "OFF")
        message(FATAL_ERROR
                "${_dg_igr_hdg_cxx_archive} must explicitly defer CUDA device-symbol resolution to final executables.")
      endif()
    endforeach()

    _dg_igr_require_hdg_cxx17_target(dg_igr_hdg_runtime_cuda)
    _dg_igr_require_hdg_cxx17_target(dg_igr_hdg_sc25_oracle_cuda)
    get_target_property(_dg_igr_hdg_sc25_cuda_excluded
                        dg_igr_hdg_sc25_oracle_cuda EXCLUDE_FROM_ALL)
    if(NOT _dg_igr_hdg_sc25_cuda_excluded)
      message(FATAL_ERROR
              "The frozen SC25 CUDA oracle must be excluded from production builds.")
    endif()
    dg_igr_hdg_require_target_link_closure_absent(
      dg_igr_hdg_runtime_cuda dg_igr_hdg_sc25_oracle_cuda)
    dg_igr_hdg_require_target_link_closure_absent(
      dg_igr_hdg_runtime_components dg_igr_hdg_sc25_oracle_cuda)
    get_target_property(_dg_igr_hdg_cuda_standard
                        dg_igr_hdg_runtime_cuda CUDA_STANDARD)
    get_target_property(_dg_igr_hdg_cuda_required
                        dg_igr_hdg_runtime_cuda CUDA_STANDARD_REQUIRED)
    get_target_property(_dg_igr_hdg_cuda_extensions
                        dg_igr_hdg_runtime_cuda CUDA_EXTENSIONS)
    get_target_property(_dg_igr_hdg_cuda_separable
                        dg_igr_hdg_runtime_cuda CUDA_SEPARABLE_COMPILATION)
    get_target_property(_dg_igr_hdg_cuda_resolve
                        dg_igr_hdg_runtime_cuda CUDA_RESOLVE_DEVICE_SYMBOLS)
    get_target_property(_dg_igr_hdg_cuda_architectures
                        dg_igr_hdg_runtime_cuda CUDA_ARCHITECTURES)
    if(NOT _dg_igr_hdg_cuda_standard STREQUAL "17" OR
       NOT _dg_igr_hdg_cuda_required OR
       _dg_igr_hdg_cuda_extensions OR
       NOT _dg_igr_hdg_cuda_separable OR
       NOT _dg_igr_hdg_cuda_resolve STREQUAL "OFF" OR
       NOT _dg_igr_hdg_cuda_architectures STREQUAL "${DG_IGR_HDG_NATIVE_CUDA_ARCH}")
      message(FATAL_ERROR
              "dg_igr_hdg_runtime_cuda is not the strict non-resolving native-SM${DG_IGR_HDG_NATIVE_CUDA_ARCH} separable CUDA17 component.")
    endif()
  elseif(TARGET dg_igr_hdg_runtime_cuda)
    message(FATAL_ERROR
            "dg_igr_hdg_runtime_cuda must exist only for a valid CUDA profile.")
  endif()

  add_subdirectory(common)
  add_subdirectory(app)
  add_subdirectory(cpu)
  if(MFEM_USE_CUDA)
    add_subdirectory(cuda)
  endif()
endif()

if(DG_IGR_ENABLE_HDG_AUXILIARY_EXPERIMENT)
  _dg_igr_require_hdg_cxx17_target(dg_igr_hdg_auxiliary_cpu)
endif()
