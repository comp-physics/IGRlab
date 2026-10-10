cmake_minimum_required(VERSION 3.14)

if(NOT DEFINED IGRLAB_COMPATIBILITY_HELPER OR
   NOT EXISTS "${IGRLAB_COMPATIBILITY_HELPER}")
  message(FATAL_ERROR "IGRlab alias test requires the production compatibility helper.")
endif()
if(NOT DEFINED IGRLAB_TEST_BINARY_DIR)
  message(FATAL_ERROR "IGRlab alias test requires a private fixture directory.")
endif()
include("${IGRLAB_COMPATIBILITY_HELPER}")

function(check_alias canonical legacy expected)
  foreach(name IN ITEMS "${canonical}" "${legacy}")
    get_property(cached_value CACHE "${name}" PROPERTY VALUE)
    if(NOT "${${name}}" STREQUAL "${expected}" OR
       NOT "${cached_value}" STREQUAL "${expected}")
      message(FATAL_ERROR "${name}: expected '${expected}', normal='${${name}}', cache='${cached_value}'")
    endif()
  endforeach()
endfunction()

igrlab_cache_alias(IGRLAB_TEST_DEFAULT DG_IGR_TEST_DEFAULT BOOL OFF "fixture")
check_alias(IGRLAB_TEST_DEFAULT DG_IGR_TEST_DEFAULT OFF)

# Migrating an old cache must preserve its selected value.
set(DG_IGR_TEST_MIGRATED YES CACHE BOOL "fixture")
igrlab_cache_alias(IGRLAB_TEST_MIGRATED DG_IGR_TEST_MIGRATED BOOL OFF "fixture")
check_alias(IGRLAB_TEST_MIGRATED DG_IGR_TEST_MIGRATED ON)

# On the first migration the newly supplied canonical spelling wins.
set(DG_IGR_TEST_OVERRIDE ON CACHE BOOL "fixture")
set(IGRLAB_TEST_OVERRIDE OFF CACHE BOOL "fixture")
igrlab_cache_alias(IGRLAB_TEST_OVERRIDE DG_IGR_TEST_OVERRIDE BOOL ON "fixture")
check_alias(IGRLAB_TEST_OVERRIDE DG_IGR_TEST_OVERRIDE OFF)

# Reconfiguration through either spelling remains usable. Clear the normal
# variable too when simulating a command-line cache update in this process.
unset(DG_IGR_TEST_OVERRIDE)
set(DG_IGR_TEST_OVERRIDE ON CACHE BOOL "fixture" FORCE)
igrlab_cache_alias(IGRLAB_TEST_OVERRIDE DG_IGR_TEST_OVERRIDE BOOL OFF "fixture")
check_alias(IGRLAB_TEST_OVERRIDE DG_IGR_TEST_OVERRIDE ON)
unset(IGRLAB_TEST_OVERRIDE)
set(IGRLAB_TEST_OVERRIDE OFF CACHE BOOL "fixture" FORCE)
igrlab_cache_alias(IGRLAB_TEST_OVERRIDE DG_IGR_TEST_OVERRIDE BOOL ON "fixture")
check_alias(IGRLAB_TEST_OVERRIDE DG_IGR_TEST_OVERRIDE OFF)

# Equivalent BOOL forms must not count as contradictory edits.
set(IGRLAB_TEST_OVERRIDE NO)
set(DG_IGR_TEST_OVERRIDE 0)
igrlab_cache_alias(IGRLAB_TEST_OVERRIDE DG_IGR_TEST_OVERRIDE BOOL ON "fixture")
check_alias(IGRLAB_TEST_OVERRIDE DG_IGR_TEST_OVERRIDE OFF)

igrlab_cache_alias(IGRLAB_TEST_ARCH DG_IGR_TEST_ARCH STRING "80" "fixture")
set(DG_IGR_TEST_ARCH "120")
igrlab_cache_alias(IGRLAB_TEST_ARCH DG_IGR_TEST_ARCH STRING "80" "fixture")
check_alias(IGRLAB_TEST_ARCH DG_IGR_TEST_ARCH "120")
set(IGRLAB_TEST_ARCH "80")
igrlab_cache_alias(IGRLAB_TEST_ARCH DG_IGR_TEST_ARCH STRING "120" "fixture")
check_alias(IGRLAB_TEST_ARCH DG_IGR_TEST_ARCH "80")

igrlab_cache_alias(IGRLAB_TEST_SOURCES DG_IGR_TEST_SOURCES STRING "a.cpp;b.cpp" "fixture")
set(DG_IGR_TEST_SOURCES "a.cpp;b.cpp;c.cpp")
igrlab_cache_alias(IGRLAB_TEST_SOURCES DG_IGR_TEST_SOURCES STRING "" "fixture")
check_alias(IGRLAB_TEST_SOURCES DG_IGR_TEST_SOURCES "a.cpp;b.cpp;c.cpp")

# An automatically required source update must also refresh normal variables
# that would otherwise shadow newly written cache entries.
set(IGRLAB_TEST_SOURCES "a.cpp;b.cpp;c.cpp;d.cpp")
igrlab_cache_alias(IGRLAB_TEST_SOURCES DG_IGR_TEST_SOURCES STRING "" "fixture")
check_alias(IGRLAB_TEST_SOURCES DG_IGR_TEST_SOURCES "a.cpp;b.cpp;c.cpp;d.cpp")

# Conflicting edits are tested in a child script because FATAL_ERROR must
# reject that configuration without aborting the parent test's assertions.
file(MAKE_DIRECTORY "${IGRLAB_TEST_BINARY_DIR}")
set(conflict_script "${IGRLAB_TEST_BINARY_DIR}/conflict.cmake")
file(WRITE "${conflict_script}"
  "cmake_minimum_required(VERSION 3.14)\n"
  "include(\"${IGRLAB_COMPATIBILITY_HELPER}\")\n"
  "igrlab_cache_alias(IGRLAB_TEST_CONFLICT DG_IGR_TEST_CONFLICT STRING 80 fixture)\n"
  "set(IGRLAB_TEST_CONFLICT 120)\n"
  "set(DG_IGR_TEST_CONFLICT 90)\n"
  "igrlab_cache_alias(IGRLAB_TEST_CONFLICT DG_IGR_TEST_CONFLICT STRING 80 fixture)\n")
execute_process(COMMAND "${CMAKE_COMMAND}" -P "${conflict_script}"
                RESULT_VARIABLE conflict_result
                OUTPUT_VARIABLE conflict_stdout
                ERROR_VARIABLE conflict_stderr)
if(conflict_result EQUAL 0 OR
   NOT conflict_stderr MATCHES "Conflicting changes to IGRLAB_TEST_CONFLICT")
  message(FATAL_ERROR "Conflicting alias edits did not reject with the expected reason: ${conflict_stdout}${conflict_stderr}")
endif()
file(REMOVE "${conflict_script}")
message(STATUS "IGRlab CMake alias/cache checks passed")
