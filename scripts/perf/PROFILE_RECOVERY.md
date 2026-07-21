# Profiling Tooling Recovery

This record preserves the auditable lineage needed to rebuild profiling support without importing a workload, a numerical policy, or a capture harness.

## Three-lineage inventory

| Lineage | Exact object | Relevant objects | Disposition |
| --- | --- | --- | --- |
| Complete Nsight toolkit | commit `790cc723bb4ba2a283d30103cc40cda1364a387c`, parent `7c07ebd6c123c3237c8f341789747dc677f8a67a`, tree `5652331b81dc6beeec5007a85b4d2f8352d6a0ec` | `profile_dg_igr_nsys.sh` blob `ddfe677f8c2b4f34eabca2e01e1ba97ddff9b614`; `profile_dg_igr_ncu.sh` blob `9a567003170c765248757883df0d323bab5d420f`; metadata writer blob `dbc1f948601d278ca1025c576c66c766a1deeade`; metadata validator blob `eb8aa9bf19731d1088bdc36c01cf738a716807bc` | Do not restore launchers, SQLite exporters, hotspot analyzers, or workload defaults. They target Inoue/DG, select refinement/order/time options, and encode old all-or-nothing checks. Their provenance intent is recovered here in tool-neutral form. |
| Accepted base | commit `4dc81619d6f2559650531956a6d1d1930fcc10e2`, parents `255d6a6a10a6f8253909e506994be30cd7078998` and `77c2d0505187249e3a02b79f7695bef8bbe05666`, tree `30ef069c8c8be758bb5468408e5a875051e034a6` | `tests/CMakeLists.txt` blob `779dd8fbc2a4534bbdf77f4dd37554e18bdbc35d` lines 762–794 register absent Nsight scripts; run metadata blob `1c4539108a1fc36b6cb21a7c43fb5cdac08b789b`; NVTX wrapper blob `e813991aff053f1c1ac071842f3d0d8a1ede359f` | Preserve existing run metadata and NVTX instrumentation untouched. The two script paths are absent from this tree (Git lookup exits 128); registrations are dangling. |
| Protected WIP | commit `1fdb07a8683e33ac0e0ed70dec037c8a0564d0d6`, parent `e2a09f47493832f94198079b87f6e3263f415073`, tree `cb41e33cfa81e1f9c87697fb57371b425549832b` | L2 timing runner blob `592773a6d890c38515e17b13302f70216e74dec0`; exact-face report blob `49db5b2db418a0031b50a8a751d3acb64aa5d446`; promotion report blob `fbeb49a5a4f17bda21770ee1e16bebc109ed38d6` | Retain as protected historical evidence only. All three parse or certify IGR/L2-specific telemetry and are not a reusable observability contract. |

## Recovered seed

`profile_provenance.py` writes a sidecar from explicit capture inputs. It never invokes Nsight, queries a GPU, scans a build tree, or supplies application arguments. `validate_profile_provenance.py` validates the sidecar and can make a requested fact mandatory with `--require`; absent requested facts fail explicitly rather than becoming a silent skip. The JSON schema documents the stable v1 boundary.

The later harness may call the writer before capture and the validator after filling mandatory facts, but this task deliberately adds no runner, test registration, build registration, or CMake change.

## Dangling-registration recommendation

`PROFILE-HARNESS-02` should remove or replace both base registrations together in its isolated CMake change. Replacement must point only to a newly reviewed generic harness with explicit executable, output location, command, and required provenance facts. Until that replacement exists, leaving the disabled registrations is misleading because their script objects are absent; this recovery does not modify them.

## Intentionally excluded

- The historical `profile_dg_igr_nsys.sh` and `profile_dg_igr_ncu.sh` launchers, their hard-coded Inoue/DG arguments, MPI defaults, and SQLite/Nsight report analyzers.
- WIP L2 timing, exact-face, promotion, VTK, and hotspot/report tooling.
- Any shared compile-away API, NVTX changes, numerical change, physical scenario, CMake registration, profiler invocation, or GPU/build-tree discovery.
