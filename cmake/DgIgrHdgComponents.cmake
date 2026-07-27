include_guard(GLOBAL)
include(CMakeParseArguments)

function(_dg_igr_hdg_require_canonical_target target allowed_targets)
  list(FIND allowed_targets "${target}" _dg_igr_hdg_target_index)
  if(_dg_igr_hdg_target_index EQUAL -1)
    message(FATAL_ERROR
            "HDG component target '${target}' is not canonical; expected one of: ${allowed_targets}")
  endif()
  if(TARGET "${target}")
    message(FATAL_ERROR "HDG component target '${target}' already exists.")
  endif()
  if(NOT TARGET igr_mfem)
    message(FATAL_ERROR
            "HDG component target '${target}' must be registered after igr_mfem.")
  endif()
endfunction()

function(_dg_igr_hdg_registration_anchor target extension language output_var)
  string(MAKE_C_IDENTIFIER "${target}" _dg_igr_hdg_identifier)
  set(_dg_igr_hdg_anchor
      "${CMAKE_CURRENT_BINARY_DIR}/${target}_registration_anchor.${extension}")
  file(GENERATE OUTPUT "${_dg_igr_hdg_anchor}" CONTENT
       "namespace dg_igr::hdg::registration { int ${_dg_igr_hdg_identifier}_anchor = 0; }\n")
  set_source_files_properties("${_dg_igr_hdg_anchor}"
                              PROPERTIES GENERATED TRUE LANGUAGE "${language}")
  set(${output_var} "${_dg_igr_hdg_anchor}" PARENT_SCOPE)
endfunction()

function(dg_igr_add_hdg_cxx17_component)
  set(_dg_igr_hdg_options "")
  set(_dg_igr_hdg_one_value TARGET)
  set(_dg_igr_hdg_multi_value
      SOURCES PUBLIC_LINK_LIBRARIES PRIVATE_LINK_LIBRARIES)
  cmake_parse_arguments(HDG
                        "${_dg_igr_hdg_options}"
                        "${_dg_igr_hdg_one_value}"
                        "${_dg_igr_hdg_multi_value}"
                        ${ARGN})
  if(HDG_UNPARSED_ARGUMENTS OR NOT HDG_TARGET)
    message(FATAL_ERROR
            "dg_igr_add_hdg_cxx17_component requires TARGET and accepts only SOURCES/PUBLIC_LINK_LIBRARIES/PRIVATE_LINK_LIBRARIES.")
  endif()

  set(_dg_igr_hdg_cxx_targets
      dg_igr_hdg_runtime_common
      dg_igr_hdg_runtime_cpu
      dg_igr_hdg_auxiliary_cpu)
  _dg_igr_hdg_require_canonical_target("${HDG_TARGET}"
                                       "${_dg_igr_hdg_cxx_targets}")

  set(_dg_igr_hdg_sources ${HDG_SOURCES})
  if(NOT _dg_igr_hdg_sources)
    _dg_igr_hdg_registration_anchor("${HDG_TARGET}" cpp CXX
                                    _dg_igr_hdg_anchor)
    list(APPEND _dg_igr_hdg_sources "${_dg_igr_hdg_anchor}")
    message(STATUS
            "${HDG_TARGET}: registration seam only; no numerical adapter sources are present.")
  endif()

  add_library("${HDG_TARGET}" STATIC ${_dg_igr_hdg_sources})
  set_target_properties("${HDG_TARGET}" PROPERTIES
                        CXX_STANDARD 17
                        CXX_STANDARD_REQUIRED ON
                        CXX_EXTENSIONS OFF)
  target_include_directories("${HDG_TARGET}" PUBLIC
                             "${PROJECT_SOURCE_DIR}/include")
  target_link_libraries("${HDG_TARGET}" PUBLIC igr_mfem)
  if(HDG_PUBLIC_LINK_LIBRARIES)
    target_link_libraries("${HDG_TARGET}" PUBLIC
                          ${HDG_PUBLIC_LINK_LIBRARIES})
  endif()
  if(HDG_PRIVATE_LINK_LIBRARIES)
    target_link_libraries("${HDG_TARGET}" PRIVATE
                          ${HDG_PRIVATE_LINK_LIBRARIES})
  endif()
endfunction()

function(dg_igr_require_hdg_cuda17_profile)
  if(NOT MFEM_USE_CUDA OR MFEM_USE_HIP)
    message(FATAL_ERROR
            "HDG CUDA components require an MFEM CUDA-only profile.")
  endif()
  if(NOT DG_IGR_ACCELERATOR_LANGUAGE STREQUAL "CUDA" OR
     NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR
            "HDG CUDA components require the local CUDA language and compiler.")
  endif()
  if(NOT MFEM_USE_MPI OR NOT TARGET MPI::MPI_CXX)
    message(FATAL_ERROR
            "HDG CUDA components require the MPI/HYPRE-enabled MFEM profile.")
  endif()
  if(NOT MFEM_USE_UMPIRE OR NOT TARGET umpire)
    message(FATAL_ERROR
            "HDG CUDA components require MFEM Umpire support and the umpire target.")
  endif()

  get_target_property(_dg_igr_hdg_mfem_links mfem INTERFACE_LINK_LIBRARIES)
  if(NOT _dg_igr_hdg_mfem_links OR
     NOT _dg_igr_hdg_mfem_links MATCHES "[Hh][Yy][Pp][Rr][Ee]")
    message(FATAL_ERROR
            "HDG CUDA components require HYPRE in mfem's exported link interface.")
  endif()

  if(NOT DG_IGR_EXPECT_CUDA_ARCH)
    message(FATAL_ERROR
            "HDG CUDA components require DG_IGR_EXPECT_CUDA_ARCH for the native application profile.")
  endif()
  set(_dg_igr_hdg_expected_arch "${DG_IGR_EXPECT_CUDA_ARCH}")
  string(REPLACE "sm_" "" _dg_igr_hdg_expected_arch
                 "${_dg_igr_hdg_expected_arch}")
  if(NOT _dg_igr_hdg_expected_arch MATCHES "^[0-9]+$")
    message(FATAL_ERROR
            "HDG CUDA expected architecture must be numeric or sm_<num>.")
  endif()
  if(NOT CMAKE_CUDA_ARCHITECTURES)
    message(FATAL_ERROR
            "HDG CUDA components require a concrete native CMAKE_CUDA_ARCHITECTURES value.")
  endif()
  foreach(_dg_igr_hdg_arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
    if(NOT _dg_igr_hdg_arch MATCHES "^([0-9]+)(-real)?$")
      message(FATAL_ERROR
              "HDG CUDA components reject non-native architecture '${_dg_igr_hdg_arch}'.")
    endif()
    if(NOT CMAKE_MATCH_1 STREQUAL "${_dg_igr_hdg_expected_arch}")
      message(FATAL_ERROR
              "HDG CUDA architecture mismatch: expected ${_dg_igr_hdg_expected_arch}, got ${_dg_igr_hdg_arch}.")
    endif()
  endforeach()

  if(NOT DG_IGR_REQUIRED_CUDA_TARGETS)
    message(FATAL_ERROR
            "HDG CUDA components require the validated CUDA toolkit target set.")
  endif()
  foreach(_dg_igr_hdg_cuda_target IN LISTS DG_IGR_REQUIRED_CUDA_TARGETS)
    if(NOT TARGET "${_dg_igr_hdg_cuda_target}")
      message(FATAL_ERROR
              "HDG CUDA prerequisite target is missing: ${_dg_igr_hdg_cuda_target}")
    endif()
  endforeach()
endfunction()

function(dg_igr_add_hdg_cuda17_component)
  set(_dg_igr_hdg_options "")
  set(_dg_igr_hdg_one_value TARGET)
  set(_dg_igr_hdg_multi_value
      SOURCES PUBLIC_LINK_LIBRARIES PRIVATE_LINK_LIBRARIES)
  cmake_parse_arguments(HDG
                        "${_dg_igr_hdg_options}"
                        "${_dg_igr_hdg_one_value}"
                        "${_dg_igr_hdg_multi_value}"
                        ${ARGN})
  if(HDG_UNPARSED_ARGUMENTS OR NOT HDG_TARGET)
    message(FATAL_ERROR
            "dg_igr_add_hdg_cuda17_component requires TARGET and accepts only SOURCES/PUBLIC_LINK_LIBRARIES/PRIVATE_LINK_LIBRARIES.")
  endif()

  set(_dg_igr_hdg_cuda_targets dg_igr_hdg_runtime_cuda)
  _dg_igr_hdg_require_canonical_target("${HDG_TARGET}"
                                       "${_dg_igr_hdg_cuda_targets}")
  dg_igr_require_hdg_cuda17_profile()

  set(_dg_igr_hdg_sources ${HDG_SOURCES})
  if(NOT _dg_igr_hdg_sources)
    _dg_igr_hdg_registration_anchor("${HDG_TARGET}" cu CUDA
                                    _dg_igr_hdg_anchor)
    list(APPEND _dg_igr_hdg_sources "${_dg_igr_hdg_anchor}")
    message(STATUS
            "${HDG_TARGET}: registration seam only; no numerical adapter sources are present.")
  else()
    set_source_files_properties(${_dg_igr_hdg_sources} PROPERTIES LANGUAGE CUDA)
  endif()

  add_library("${HDG_TARGET}" STATIC ${_dg_igr_hdg_sources})
  set_target_properties("${HDG_TARGET}" PROPERTIES
                        CXX_STANDARD 17
                        CXX_STANDARD_REQUIRED ON
                        CXX_EXTENSIONS OFF
                        CUDA_STANDARD 17
                        CUDA_STANDARD_REQUIRED ON
                        CUDA_EXTENSIONS OFF
                        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
                        CUDA_SEPARABLE_COMPILATION ON)
  target_include_directories("${HDG_TARGET}" PUBLIC
                             "${PROJECT_SOURCE_DIR}/include")
  target_compile_options("${HDG_TARGET}" PRIVATE
                         $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
                         $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
  target_link_libraries("${HDG_TARGET}" PUBLIC
                        igr_mfem ${HDG_PUBLIC_LINK_LIBRARIES}
                        PRIVATE umpire ${DG_IGR_REQUIRED_CUDA_TARGETS}
                                ${HDG_PRIVATE_LINK_LIBRARIES})
endfunction()
