# Mapped DG checkpoint integration validation

PR #79 integrates the cumulative mapped ordinary-DG spectral correction and
throughput campaign into `dev`. The base at review time is
`e5b3ddf30a24d249b25df2d7b631c7bd9ae37850`; it is already an ancestor of this
branch. The unrelated deferred HDG PR #75 is not included.

Independent cumulative IGR and Euler/output reviews found one actionable gap:
fixed-sweep production history was not restored after a rejected RK attempt.
The checkpoint explicitly rejects effective Zhang–Shu positivity limiting and
direct fixed-mode abort/retry calls until transactional history is supported.
It retains every-stage-only and fresh-start restrictions. Converged Krylov
keeps its existing retry support. The measured no-limiter Sedov algorithm and
kernels are unchanged.

Canonical DeltaAI `build.sh` job **3446580** qualified source
`1e891cecf6471a109a4983fcca79c095b2408d54` on one GH200 (`gh077`). All **25 selected
CPU/CUDA groups passed**, including configuration/retry admission, ordinary-DG
Euler/IGR, mapped dense/tensor paths, local spectral backends, existing DGSEM
basis/flux/operator/coupling, and cached native VTK. The selected tests took
38.8145 seconds; the complete allocation including fresh application build
was 12m37s. Installed MFEM/Spack identities were unchanged. Build objects and
test output used private node `/tmp` and were cleaned by the job wrapper.

Qualified application SHA-256:
`78cfbf93d77d977c8ede6bc4b98da6b97aecabe2a597f91368851c0a5cc34edd`.
The [compact validation record](benchmarks/dg_mapped_checkpoint_validation.json)
binds exact commands, GPU, tests, source and raw receipt/log hashes. Later
commits only record this validation.

The [performance checkpoint](dg-mapped-throughput-20261023.md) remains measured
at source `bc0df2971a6de2f840d995aa45f75bb664523b85`, with its recorded binary and
same-GPU full/profile pairs. This native regression is not a new full timing
pair, a default promotion of approximate closure, or a qualification of 3D,
multi-GPU or the subsequent GLL/mapped DGSEM extension. No hosted check runs
or status results were present when the PR was opened; these native results
are reported explicitly rather than calling hosted CI passed.
