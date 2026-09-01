if(NOT DEFINED HDG_MAIN_EXECUTABLE OR NOT DEFINED HDG_MAIN_MESH OR
   NOT DEFINED HDG_MAIN_WORK)
  message(FATAL_ERROR
    "CUDA HDG checkpoint executable, mesh, and private work directory are required.")
endif()

file(REMOVE_RECURSE "${HDG_MAIN_WORK}")
file(MAKE_DIRECTORY "${HDG_MAIN_WORK}")
set(_control_checkpoint "${HDG_MAIN_WORK}/control.chk")
set(_split_checkpoint "${HDG_MAIN_WORK}/split.chk")
set(_control_vtk "${HDG_MAIN_WORK}/control-vtk")
set(_split_vtk "${HDG_MAIN_WORK}/split-vtk")
set(_control_arguments
  --mesh "${HDG_MAIN_MESH}"
  --checkpoint "${_control_checkpoint}"
  --vtk-root "${_control_vtk}")
set(_split_arguments
  --mesh "${HDG_MAIN_MESH}"
  --checkpoint "${_split_checkpoint}"
  --vtk-root "${_split_vtk}")
set(_resume_arguments
  --mesh "${HDG_MAIN_MESH}"
  --checkpoint "${_split_checkpoint}"
  --restart "${_split_checkpoint}"
  --vtk-root "${_split_vtk}")
set(_success_pattern
  "\\[hdg-main\\] status=Success.*step=4.*time=4e-05.*output_cycle=4.*cuda_engine=ordinary.*time_treatment=rodas3")

# Uninterrupted adaptive control. Output boundaries clip each accepted step so
# the checkpoint/controller cursor has four deterministic committed updates.
execute_process(
  COMMAND "${HDG_MAIN_EXECUTABLE}" ${_control_arguments}
  RESULT_VARIABLE _control_status
  OUTPUT_VARIABLE _control_stdout
  ERROR_VARIABLE _control_stderr)
if(NOT _control_status EQUAL 0 OR
   NOT _control_stdout MATCHES "${_success_pattern}" OR
   NOT _control_stderr STREQUAL "")
  message(FATAL_ERROR
    "uninterrupted CUDA HDG checkpoint control failed: ${_control_status}\n${_control_stdout}\n${_control_stderr}")
endif()

# Process one installs the headerless test callback. It enters and then aborts
# the third prepared PDE candidate, after the first two committed boundaries
# have each atomically advanced the rolling checkpoint generation.
execute_process(
  COMMAND "${HDG_MAIN_EXECUTABLE}" ${_split_arguments}
    --inject-after-two
  RESULT_VARIABLE _split_status
  OUTPUT_VARIABLE _split_stdout
  ERROR_VARIABLE _split_stderr)
if(_split_status EQUAL 0 OR
   NOT _split_stderr MATCHES
     "\\[hdg-main\\] status=Rejected reason=test-only CUDA prepare-boundary failure injected")
  message(FATAL_ERROR
    "CUDA HDG split writer did not fail at its prescribed prepared boundary: ${_split_status}\n${_split_stdout}\n${_split_stderr}")
endif()
if(NOT IS_SYMLINK "${_split_checkpoint}")
  message(FATAL_ERROR
    "CUDA HDG rolling checkpoint alias was not published: ${_split_checkpoint}")
endif()
set(_split_store
  "${HDG_MAIN_WORK}/.split.chk.dg-igr-hdg-checkpoint-store")
file(GLOB _split_generations LIST_DIRECTORIES true
  "${_split_store}/.dg-igr-generations/generation-*")
list(LENGTH _split_generations _split_generation_count)
if(NOT _split_generation_count EQUAL 2)
  message(FATAL_ERROR
    "split writer must leave exactly two committed generations, found ${_split_generation_count}")
endif()

execute_process(
  COMMAND "${HDG_MAIN_EXECUTABLE}"
    --inspect-checkpoint "${_split_checkpoint}"
  RESULT_VARIABLE _mid_inspect_status
  OUTPUT_VARIABLE _mid_summary
  ERROR_VARIABLE _mid_inspect_stderr)
string(STRIP "${_mid_summary}" _mid_summary)
if(NOT _mid_inspect_status EQUAL 0 OR
   NOT _mid_summary MATCHES
     "checkpoint-summary.*step=2.*output_cycle=2.*adaptive=1.*accepted=2.*rejected=0.*descriptor_step=2.*descriptor_generation=2" OR
   NOT _mid_inspect_stderr STREQUAL "")
  message(FATAL_ERROR
    "mid-horizon checkpoint did not decode as the second committed boundary: ${_mid_inspect_status}\n${_mid_summary}\n${_mid_inspect_stderr}")
endif()

# Both projected initial data and the projected static source participate in
# numerical identity. These changed problem definitions must be rejected
# before restoring PDE state or opening the existing output series.
foreach(_perturbation IN ITEMS --perturb-initial --perturb-source)
  execute_process(
    COMMAND "${HDG_MAIN_EXECUTABLE}" ${_resume_arguments} ${_perturbation}
    RESULT_VARIABLE _identity_status
    OUTPUT_VARIABLE _identity_stdout
    ERROR_VARIABLE _identity_stderr)
  if(_identity_status EQUAL 0 OR
     NOT _identity_stderr MATCHES
       "\\[hdg-main\\] status=Rejected reason=generic CUDA HDG checkpoint decode/identity validation failed")
    message(FATAL_ERROR
      "changed projected problem input did not fail checkpoint identity (${_perturbation}): ${_identity_status}\n${_identity_stdout}\n${_identity_stderr}")
  endif()
endforeach()

# Process two removes the injected failure, restores the same numerical
# identity/controller/output cursor, appends the same native PVD series, and
# advances to the original horizon.
execute_process(
  COMMAND "${HDG_MAIN_EXECUTABLE}" ${_resume_arguments}
  RESULT_VARIABLE _resume_status
  OUTPUT_VARIABLE _resume_stdout
  ERROR_VARIABLE _resume_stderr)
if(NOT _resume_status EQUAL 0 OR
   NOT _resume_stdout MATCHES "${_success_pattern}" OR
   NOT _resume_stderr STREQUAL "")
  message(FATAL_ERROR
    "CUDA HDG checkpoint continuation failed: ${_resume_status}\n${_resume_stdout}\n${_resume_stderr}")
endif()

file(GLOB _control_pvd "${_control_vtk}/*/*/*.pvd")
file(GLOB _resumed_pvd "${_split_vtk}/*/*/*.pvd")
list(LENGTH _control_pvd _control_pvd_count)
list(LENGTH _resumed_pvd _resumed_pvd_count)
if(NOT _control_pvd_count EQUAL 1 OR NOT _resumed_pvd_count EQUAL 1)
  message(FATAL_ERROR
    "CUDA HDG checkpoint process did not leave one native PVD series per run")
endif()
list(GET _control_pvd 0 _control_pvd_file)
list(GET _resumed_pvd 0 _resumed_pvd_file)
file(READ "${_control_pvd_file}" _control_pvd_text)
file(READ "${_resumed_pvd_file}" _resumed_pvd_text)
if(NOT _control_pvd_text STREQUAL _resumed_pvd_text)
  message(FATAL_ERROR
    "same-root resumed CUDA HDG PVD series differs from uninterrupted control\ncontrol=${_control_pvd_text}\nresumed=${_resumed_pvd_text}")
endif()
get_filename_component(_resumed_collection "${_resumed_pvd_file}" DIRECTORY)
foreach(_padded_cycle IN ITEMS 000000 000001 000002 000003 000004)
  if(NOT EXISTS "${_resumed_collection}/Cycle${_padded_cycle}/data.pvtu")
    message(FATAL_ERROR
      "same-root resumed CUDA HDG series is missing cycle ${_padded_cycle}")
  endif()
endforeach()

execute_process(
  COMMAND "${HDG_MAIN_EXECUTABLE}"
    --inspect-checkpoint "${_control_checkpoint}"
  RESULT_VARIABLE _control_inspect_status
  OUTPUT_VARIABLE _control_summary
  ERROR_VARIABLE _control_inspect_stderr)
execute_process(
  COMMAND "${HDG_MAIN_EXECUTABLE}"
    --inspect-checkpoint "${_split_checkpoint}"
  RESULT_VARIABLE _resume_inspect_status
  OUTPUT_VARIABLE _resume_summary
  ERROR_VARIABLE _resume_inspect_stderr)
string(STRIP "${_control_summary}" _control_summary)
string(STRIP "${_resume_summary}" _resume_summary)
if(NOT _control_inspect_status EQUAL 0 OR
   NOT _resume_inspect_status EQUAL 0 OR
   NOT _control_inspect_stderr STREQUAL "" OR
   NOT _resume_inspect_stderr STREQUAL "" OR
   NOT _control_summary MATCHES
     "checkpoint-summary.*step=4.*output_cycle=4.*adaptive=1.*accepted=4.*rejected=0.*descriptor_step=4.*descriptor_generation=4" OR
   NOT _control_summary STREQUAL _resume_summary)
  message(FATAL_ERROR
    "resumed CUDA HDG payload hashes/controller/output coordinates differ from control\ncontrol=${_control_summary}\nresumed=${_resume_summary}\n${_control_inspect_stderr}\n${_resume_inspect_stderr}")
endif()

# Corrupt the exact test-owned current generation. Restart must reject before
# decoded state can be published into the pristine CUDA session.
file(REAL_PATH "${_split_checkpoint}" _artifact)
if(NOT EXISTS "${_artifact}")
  message(FATAL_ERROR
    "CUDA HDG rolling checkpoint does not resolve to an immutable artifact")
endif()
file(WRITE "${_artifact}" "corrupt checkpoint\n")
execute_process(
  COMMAND "${HDG_MAIN_EXECUTABLE}" ${_resume_arguments}
  RESULT_VARIABLE _corrupt_status
  OUTPUT_VARIABLE _corrupt_stdout
  ERROR_VARIABLE _corrupt_stderr)
if(_corrupt_status EQUAL 0 OR
   NOT _corrupt_stderr MATCHES
     "\\[hdg-main\\] status=Rejected reason=generic CUDA HDG checkpoint decode/identity validation failed")
  message(FATAL_ERROR
    "corrupt CUDA HDG checkpoint did not fail closed: ${_corrupt_status}\n${_corrupt_stdout}\n${_corrupt_stderr}")
endif()
