# Collect all source files (including module subdirectories)
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "*.cpp")

# Create library
add_library(igr_mfem STATIC ${SOURCES})

# Include directories  
target_include_directories(igr_mfem PUBLIC 
    ${CMAKE_SOURCE_DIR}/include
)

# Link MFEM
target_link_libraries(igr_mfem PUBLIC mfem)

if(MFEM_USE_CUDA)
    # Keep CUDA TPL linkage explicit to avoid transitive-link drops when MFEM/HYPRE
    # exports are incomplete.
    target_link_libraries(igr_mfem PUBLIC ${DG_IGR_REQUIRED_CUDA_TARGETS})
endif()
