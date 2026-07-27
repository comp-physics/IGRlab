include("${PROJECT_SOURCE_DIR}/cmake/DgIgrHdgComponents.cmake")

file(GLOB_RECURSE _dg_igr_hdg_runtime_common_sources CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/common/*.cpp"
     "${CMAKE_CURRENT_SOURCE_DIR}/shared/*.cpp"
     "${CMAKE_CURRENT_SOURCE_DIR}/cli/*.cpp"
     "${CMAKE_CURRENT_SOURCE_DIR}/io/*.cpp")
file(GLOB_RECURSE _dg_igr_hdg_runtime_cpu_sources CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/cpu/*.cpp")
file(GLOB_RECURSE _dg_igr_hdg_runtime_cuda_sources CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/cuda/*.cpp"
     "${CMAKE_CURRENT_SOURCE_DIR}/cuda/*.cu")
file(GLOB_RECURSE _dg_igr_hdg_runtime_app_sources CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/app/*.cpp")
file(GLOB_RECURSE _dg_igr_hdg_runtime_all_cpp CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
file(GLOB_RECURSE _dg_igr_hdg_runtime_all_cuda CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")

set(_dg_igr_hdg_runtime_unclassified_cpp
    ${_dg_igr_hdg_runtime_all_cpp})
list(REMOVE_ITEM _dg_igr_hdg_runtime_unclassified_cpp
     ${_dg_igr_hdg_runtime_common_sources}
     ${_dg_igr_hdg_runtime_cpu_sources}
     ${_dg_igr_hdg_runtime_cuda_sources}
     ${_dg_igr_hdg_runtime_app_sources})
set(_dg_igr_hdg_runtime_unclassified_cuda
    ${_dg_igr_hdg_runtime_all_cuda})
list(REMOVE_ITEM _dg_igr_hdg_runtime_unclassified_cuda
     ${_dg_igr_hdg_runtime_cuda_sources})
if(_dg_igr_hdg_runtime_unclassified_cpp OR
   _dg_igr_hdg_runtime_unclassified_cuda)
  message(FATAL_ERROR
          "Unclassified HDG runtime sources are excluded from igr_mfem but have no named component target: "
          "${_dg_igr_hdg_runtime_unclassified_cpp};${_dg_igr_hdg_runtime_unclassified_cuda}")
endif()

if(DG_IGR_ENABLE_HDG_RUNTIME)
  if(TARGET dg_igr_hdg_runtime OR TARGET dg_igr_hdg)
    message(FATAL_ERROR
            "Canonical HDG runtime or executable target name is already in use.")
  endif()

  add_library(dg_igr_hdg_runtime INTERFACE)

  dg_igr_add_hdg_cxx17_component(
    TARGET dg_igr_hdg_runtime_common
    SOURCES ${_dg_igr_hdg_runtime_common_sources})
  dg_igr_add_hdg_cxx17_component(
    TARGET dg_igr_hdg_runtime_cpu
    SOURCES ${_dg_igr_hdg_runtime_cpu_sources}
    PUBLIC_LINK_LIBRARIES dg_igr_hdg_runtime_common)

  target_link_libraries(dg_igr_hdg_runtime INTERFACE
                        dg_igr_hdg_runtime_common
                        dg_igr_hdg_runtime_cpu)

  if(MFEM_USE_CUDA)
    dg_igr_add_hdg_cuda17_component(
      TARGET dg_igr_hdg_runtime_cuda
      SOURCES ${_dg_igr_hdg_runtime_cuda_sources}
      PUBLIC_LINK_LIBRARIES dg_igr_hdg_runtime_common)
    target_link_libraries(dg_igr_hdg_runtime INTERFACE
                          dg_igr_hdg_runtime_cuda)
  elseif(_dg_igr_hdg_runtime_cuda_sources)
    message(STATUS
            "HDG CUDA sources are reserved but not built by the requested CPU profile.")
  endif()

  if(_dg_igr_hdg_runtime_app_sources)
    add_executable(dg_igr_hdg ${_dg_igr_hdg_runtime_app_sources})
    set_target_properties(dg_igr_hdg PROPERTIES
                          CXX_STANDARD "${CMAKE_CXX_STANDARD}"
                          CXX_STANDARD_REQUIRED ON
                          CXX_EXTENSIONS OFF)
    target_link_libraries(dg_igr_hdg PRIVATE dg_igr_hdg_runtime)
  else()
    message(STATUS
            "dg_igr_hdg: target name reserved; no executable sources are present.")
  endif()
endif()

if(DG_IGR_ENABLE_HDG_AUXILIARY_EXPERIMENT)
  file(GLOB_RECURSE _dg_igr_hdg_auxiliary_sources CONFIGURE_DEPENDS
       "${CMAKE_CURRENT_SOURCE_DIR}/../auxiliary/*.cpp")
  dg_igr_add_hdg_cxx17_component(
    TARGET dg_igr_hdg_auxiliary_cpu
    SOURCES ${_dg_igr_hdg_auxiliary_sources})
endif()
