# IGRlab configuration names are canonical. Legacy DG_IGR_* cache names and
# target/CTest names remain transitional interfaces for existing build drivers.
# Synchronize aliases in both directions so an existing cache can be changed
# through either spelling. On the first migration, an explicit canonical value
# takes precedence over an older legacy cache value.
function(igrlab_cache_alias canonical legacy type initial description)
  set(_igrlab_last "_IGRLAB_ALIAS_LAST_${canonical}")
  if(DEFINED ${canonical})
    set(_igrlab_canonical "${${canonical}}")
    if(type STREQUAL "BOOL")
      if(${canonical})
        set(_igrlab_canonical ON)
      else()
        set(_igrlab_canonical OFF)
      endif()
    endif()
  endif()
  if(DEFINED ${legacy})
    set(_igrlab_legacy "${${legacy}}")
    if(type STREQUAL "BOOL")
      if(${legacy})
        set(_igrlab_legacy ON)
      else()
        set(_igrlab_legacy OFF)
      endif()
    endif()
  endif()

  set(_igrlab_value "${initial}")
  if(DEFINED ${_igrlab_last})
    set(_igrlab_canonical_changed OFF)
    set(_igrlab_legacy_changed OFF)
    if(DEFINED ${canonical} AND
       NOT "${_igrlab_canonical}" STREQUAL "${${_igrlab_last}}")
      set(_igrlab_canonical_changed ON)
    endif()
    if(DEFINED ${legacy} AND
       NOT "${_igrlab_legacy}" STREQUAL "${${_igrlab_last}}")
      set(_igrlab_legacy_changed ON)
    endif()
    if(_igrlab_canonical_changed AND _igrlab_legacy_changed AND
       NOT "${_igrlab_canonical}" STREQUAL "${_igrlab_legacy}")
      message(FATAL_ERROR
        "Conflicting changes to ${canonical} and its legacy alias ${legacy}. "
        "Set one spelling or give both the same value.")
    endif()
    if(_igrlab_canonical_changed)
      set(_igrlab_value "${_igrlab_canonical}")
    elseif(_igrlab_legacy_changed)
      set(_igrlab_value "${_igrlab_legacy}")
    else()
      set(_igrlab_value "${${_igrlab_last}}")
    endif()
  elseif(DEFINED ${canonical})
    set(_igrlab_value "${_igrlab_canonical}")
  elseif(DEFINED ${legacy})
    set(_igrlab_value "${_igrlab_legacy}")
  endif()

  set(${canonical} "${_igrlab_value}" CACHE ${type} "${description}" FORCE)
  set(${legacy} "${_igrlab_value}" CACHE ${type}
      "Legacy alias for ${canonical}. ${description}" FORCE)
  set(${_igrlab_last} "${_igrlab_value}" CACHE INTERNAL
      "Last synchronized value of ${canonical} and ${legacy}." FORCE)
  # A caller's normal variable may otherwise shadow the refreshed cache entry.
  set(${canonical} "${_igrlab_value}" PARENT_SCOPE)
  set(${legacy} "${_igrlab_value}" PARENT_SCOPE)
endfunction()
