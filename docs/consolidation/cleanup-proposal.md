# Reviewable IGRlab cleanup proposal

Proposed batch: **39 local refs, 61 remote refs, three old worktrees, and closure of superseded PRs #37/#38**. Nothing has been deleted, closed, merged, or published. The original primary checkout, both stashes, six archive refs, 12 experiment refs, and primary/main/dev/consolidation refs remain retained.

Captured retained consolidation snapshot: `0e4d578b0b3bd1acf99f3b30a05cfda5d0d47e8d`. Numerical source: `0e4d578b0b3bd1acf99f3b30a05cfda5d0d47e8d`. Validation status: passed required current-source checks at 0e4d578. A later final docs-only HEAD is recorded in the external approval manifest, avoiding self-reference in the committed snapshot. Ancestry, patch-equivalence and the independent 18-patch experiment review remain pinned to `0a34f84625a52a33cc73905c0708475e0551a743`.

All 100 retirement SHAs matched the refreshed local/live-origin inventory. Both PRs remain open at the expected head/base SHAs. All four original worktree porcelain and staged/unstaged full-index binary patches exactly match the recovery metadata.

## Recovery and execution gates

Recovery root: `/home/jarias9/archives/igrlab/consolidation-20261010`. `recovery_state.json` is now **verified**; all five volumes passed member/content verification. The first archive preserves raw Git objects, refs, reflogs, stashes and unreachable history. The other four preserve original worktree files, untracked research, and ignored content represented in the manifests.

Independent restore proof: `/home/jarias9/archives/igrlab/consolidation-20261010/restore_proof.json`. Observed status: verified named refs/stashes/100 retirement roots and full historical reflog closure using the separately validated210-object supplement.

The read-only preflight verifies archive checksum metadata and bytes, exact manifest contents/modes/mtime/symlink targets, worktree HEAD/branch/porcelain/binary patches, untracked and ignored content digests, all proposed and retained ref SHAs, stashes, submodule pins, PR head/base/state, and the restore proof with all 100 retirement records plus required-object connectivity. It prints proposed actions only and contains no execution or deletion path.

Run: `python3 /tmp/igrlab-consolidation-git/cleanup_preflight.py --json`. Capture evidence in `cleanup_preflight_result.json`. `--skip-archive-bytes` performs an explicitly partial check and cannot pass the full guard. Any unexpected state stops the affected cleanup proposal. Network reads use the existing gh credential helper without changing Git config or refs.

Required current-source selected backend/numerical/communicator/empty-local-rank checks passed on `0e4d578`. Explicit user approval of the concrete batch and an immediate ref/state/writer recheck are required before execution. Final docs-only HEAD and its external approval/preflight evidence remain pending root signal. This preflight grants no destructive authorization.

## Passed numerical-source validation

Tested source: `0e4d578b0b3bd1acf99f3b30a05cfda5d0d47e8d`; MFEM pin `d9d6526cc1749980a2ba1da16e2c1ca1e07d82ec`. All configurations used canonical `build.sh` Release profiles. These are selected suites, not every registered test or historical campaign.

| Configuration | HDG | Research | Selected result |
|---|---|---|---:|
| CPU | ON | OFF | 68/68 passed |
| CPU | OFF | ON | 18/18 passed |
| CUDA SM120 | ON | OFF | 31/31 passed |
| CUDA SM120 | OFF | OFF | 18/18 passed |

CUDA modal damping memcheck passed with zero device memory errors. `--report-api-errors no` excluded CUDA-aware MPI host-pointer probing API messages; device memory checks remained enabled. The current-source bounded assembled L2/BrokenL2 M16 application pair passed. It verifies this application treatment/conservation case, not the original p39 fine-step study or universal accuracy improvement.

PR #37 current CaseContext communicator-authority checks passed (`dg_igr_bc_tests`, `dg_igr_case_context_compile`). PR #38 current zero-owned-rank collective admission check passed (`dg_igr_hdg_main_backend_collective_projection_rejected_rank2`). The screened tensor patch test belongs to the opted-in HDG subdirectory and is absent when HDG is OFF. Evidence and exact result/log identities are recorded in `cleanup_proposal.json`.

## Recovery scope and supplemented historical objects

Exact raw Git bytes and physical worktree snapshots are preserved and verified. The restore drill found 210 pre-existing zero-byte loose files whose object bodies were missing before capture. All 210 bodies have since been reconstructed exactly from archived physical research files and placed in a separate, independently validated supplement. Root completed the bounded original object-database repair, retaining every displaced original; independent checks verify both installed objects and full original physical worktree preservation.

The isolated recovery now verifies all 132 named refs/stashes, all 100 retirement records and the full named/reflog/retirement closure: 28,141 objects with zero missing. The raw archive faithfully retains the original empty files, so the supplement is required as part of complete recovery. Raw volumes remain immutable. No residual historical gap remains in the verified supplementary recovery; primary artifact retirement remains excluded.

## Primary preservation

The primary checkout `/home/jarias9/projects/dg_igr_miniapp` remains at `1c75579a94bca0d067aedd44c26aff13c5831b26`. Its status and binary patches exactly match the archive; the full read-only manifest/content check supplies byte-level evidence. No primary source, output, ignored cache, stash, or research-file retirement is proposed in this batch.

## Measured read-only preflight evidence

At `2026-10-10T22:22:19.901488+00:00`, all five archive byte checksums, metadata guards, retirement/retained ref SHAs, stashes, submodule pins and PR guards passed. All four original worktrees matched every manifest entry and archived patch. Primary: 72,102 entries and 34,804,970,073 file bytes checked, with zero mismatches.

Later independent recovery guard: successful named/ref/reflog restore proof recorded and verified using the separately validated supplement. Comprehensive final preflight awaits root repair and final candidate SHA.

## Proposed execution order

1. Complete archive member/checksum verification and successful restore demonstration.
2. Finalize current candidate validation and obtain explicit approval of the exact cleanup/closure batch.
3. Refresh all expected local/remote SHA and worktree-state guards; stop/review mismatches.
4. Close PRs #37 and #38 as superseded before deleting their head/base remote refs, with retained candidate and current gate evidence.
5. Remove the three old registered worktrees through Git after restoration/state/writer checks.
6. Delete the 39 approved local refs with expected-old-SHA guards.
7. Delete the 61 approved remote refs with per-ref expected-SHA leases, rechecking retained/default refs.
8. Record action results and preserve recovery archives, retained refs/stashes, and unresolved experiment refs.

## Retained local refs

| Ref | Expected/observed SHA |
|---|---|
| `refs/heads/main` | `b04f8e2d7dbdff8947302e387912041aa656e746` |
| `refs/heads/dev` | `30afa08f8ad19a3ad37e627463ef63c40da1ae91` |
| `refs/heads/codex/tensor-precision-investigation` | `1c75579a94bca0d067aedd44c26aff13c5831b26` |
| `refs/heads/codex/igrlab-consolidation` | `0e4d578b0b3bd1acf99f3b30a05cfda5d0d47e8d` |

## Six preserved archive refs

| Ref | Expected/observed SHA |
|---|---|
| `refs/heads/archive/worktree-cleanup-20260823/dgsem-production-baseline` | `c46b258870921af0a3e568162c211acf6ae2c6d8` |
| `refs/heads/archive/worktree-cleanup-20260823/hdg-auxiliary-cpu-e2-recovery` | `20e30aa4733a05a1aaa8baf482dbb3c0b18bac1d` |
| `refs/heads/archive/worktree-cleanup-20260823/hdg-cuda-traced-igr-private-topology-v1` | `49fbc27d1fa5a4861a8bb5b018b4fb7fef6c2f5f` |
| `refs/heads/archive/worktree-cleanup-20260823/hdg-generic-traced-igr-private-v2` | `8126b0fc4990ce8d18e758d8d4b25038b2aedbb9` |
| `refs/heads/archive/worktree-cleanup-20260823/hdg-w8c-conventional-cpu` | `6bb849f7d6aff57f265aa80c7509f8e777cd2050` |
| `refs/heads/archive/worktree-cleanup-20260823/hdg-w8c-scalar-igr-cpu` | `dbf65c76c783bf9d58458709922ea8a7a5b4580d` |

## Twelve experiment refs retained in this batch

| Ref | Expected/observed SHA |
|---|---|
| `refs/heads/codex/hdg-default-numerical-policy` | `b7f8115910172c3275da211c5b15ff067c00f2b9` |
| `refs/heads/codex/hdg-igr-combine-tf32-matrix-free` | `45822f9cba238884ccd45d9f04a26c34d6a6a47d` |
| `refs/heads/codex/hdg-igr-equilibrated-field-split` | `119d6d9196b03f4b6ebe3625465bd47d2abfd820` |
| `refs/heads/codex/hdg-igr-fixed-sweep-preconditioner` | `786c02d2370fd221502493391d8748ff5af330d6` |
| `refs/heads/codex/hdg-igr-fixed-sweep-specialized` | `2aa498470f4e22126206787dbb9768513c39d2d2` |
| `refs/heads/codex/hdg-igr-schwarz-tf32-ir` | `16b36728049a7b8606a6e1c9ccff1ee1d5d1bca3` |
| `refs/heads/codex/hdg-igr-trace-optimization-v2` | `69c78187dccbc7fb376e916bcd1939decbb32b30` |
| `refs/heads/codex/hdg-q12-three-way-control` | `19afa688ef7ab74e85140074fa942542f0609562` |
| `refs/heads/codex/hdg-tensor-campaign-doc-refresh` | `54dc928e5a98079e6b0aca7ef7644dc5a501d069` |
| `refs/heads/codex/hdg-tensor-cylinder-smoke` | `69c78187dccbc7fb376e916bcd1939decbb32b30` |
| `refs/heads/codex/hdg-tensor-cylinder-smoke-dev` | `d19e02f2eabea39b31448b103eb1ae841351b56d` |
| `refs/heads/codex/hdg-tensor-volume-assembly` | `e0193821ff04884a7070deefd30ca61bbd4ca62c` |

All 18 distinct unmatched experimental patches were inspected against `0a34f84`: preserve with exact provenance and import none. Required backend/manual tensor capabilities are retained in the consolidated source. The specific TF32-IR1 Schwarz ablation remains archived research; FP64/TF32 volume specializations remain present. Their current execution and numerical claims depend on root validation.

## 39 local retirement candidates

| Ref | Expected/observed SHA | Basis |
|---|---|---|
| `refs/heads/codex/adaptive-dae-reusable-trace` | `901c6f8bf6326b5bafb962bd48d8db9307a8872c` | Ancestry contained |
| `refs/heads/codex/hdg-alignment-armijo-state-tests` | `274cfe3ff81366be39301682da935c6090c176c7` | Patch equivalent |
| `refs/heads/codex/hdg-alignment-policy-parameters` | `73306e62eb0fc45d2fbfc7e45e238e96b8b5c826` | Patch equivalent |
| `refs/heads/codex/hdg-case-context-v1` | `e75f3cbd9428b7b044b4400186336153909f1ac6` | Patch equivalent |
| `refs/heads/codex/hdg-generic-contract-v1` | `c5444f34bd4eb1d8cc59f70f8832109b1a031b41` | Patch equivalent |
| `refs/heads/codex/hdg-hash-strength-reduction` | `e029eac85b1468885c4da0624b6387c6c9de20c1` | Patch equivalent |
| `refs/heads/codex/hdg-igr-fieldsplit-v1` | `0fda3a56e7c08811afd6377f586d8c9b15a4a9f8` | Patch equivalent |
| `refs/heads/codex/hdg-igr-linear-policy-v1` | `b2b54b9a59c292d679b0443fe70d508dfae6a8f9` | Ancestry contained |
| `refs/heads/codex/hdg-igr-matrix-free-equilibrated-field-split` | `c139d39547f0ed0ac2bea355885035093bf156f9` | Patch equivalent |
| `refs/heads/codex/hdg-igr-matrix-free-stage2` | `332dfaa992f1f46cf69a1fe1b6692be1a44e740c` | Patch equivalent |
| `refs/heads/codex/hdg-igr-symmetric-field-split` | `4a61c382b03ffce33eb44bb311e1cf92a5682dd9` | Ancestry contained |
| `refs/heads/codex/hdg-igr-warp-field-split` | `3f10eaf9667f38ca5287f8121b5025db560f52ed` | Patch equivalent |
| `refs/heads/codex/hdg-native-cgs2-experiment` | `b7e28434b63256dca5e909206fa08a344e52403f` | Patch equivalent |
| `refs/heads/codex/hdg-native-schwarz-fp64` | `9e8948f3dd77a82aa85ee9738d02e80f9f2ec004` | Patch equivalent |
| `refs/heads/codex/hdg-native-trace-fixtures` | `b379e3cebc1469420a6900236b2b12c1c09b442d` | Patch equivalent |
| `refs/heads/codex/hdg-p123-cfl-refinement-tests` | `0488918eb37924fb5c8b5d7f631a222a6bf06d03` | Patch equivalent |
| `refs/heads/codex/hdg-paraview-output-v1` | `b856c67e9a3491b77cc86fa1304df4517cff6efe` | Patch equivalent |
| `refs/heads/codex/hdg-production-trace-policy` | `a735c5bf221b1c593a97d2054a8f750a73c9e16f` | Patch equivalent |
| `refs/heads/codex/hdg-schwarz-campaign-metadata` | `77b7ae8b1f6f464b7dd6d2bd59075b33da48e4ed` | Patch equivalent |
| `refs/heads/codex/hdg-tensor-assembly-fusion` | `d8200a1ac403045e2694611bb3a4cc12e42af39a` | Patch equivalent |
| `refs/heads/codex/hdg-tensor-blocked-local-v1` | `d67afd2faa9b895d7deefaea1d1ad63d6ef2e76c` | Patch equivalent |
| `refs/heads/codex/hdg-tensor-core-backend-v1` | `df5f9c121400a4740e8bc8a4955a5c712af3a9b6` | Ancestry contained |
| `refs/heads/codex/hdg-tensor-dmma-exact` | `3307734ea4ae707e5cdc30b65bbd7d977d9adc9e` | Patch equivalent |
| `refs/heads/codex/hdg-tensor-hash-fast` | `ebe03fdb2ddcc55dc7b61859f4bb127e9ff3d54e` | Patch equivalent |
| `refs/heads/codex/hdg-tensor-hash-jump` | `8f90a2dc70c0910cbb12e7f693946a641e76556f` | Patch equivalent |
| `refs/heads/codex/hdg-tensor-igr-cfl-allocation` | `d000cc99131ff7c65d98ac772d8f3e9ab3618b49` | Patch equivalent |
| `refs/heads/codex/hdg-tensor-native-trace-v1` | `7f6f9e753750a287e4dd178485b7d5161cddba6d` | Patch equivalent |
| `refs/heads/codex/hdg-tensor-policy-ablation-v2` | `1c79c81fe4107717a490e948e9dc00e9d82dda1e` | Ancestry contained |
| `refs/heads/codex/hdg-tensor-recovery-fusion` | `1af2069819932c41a285dddc31a5d7264986a38c` | Patch equivalent |
| `refs/heads/codex/hdg-tensor-sass-profile-evidence` | `c0fe0eb58161ed9b318e66e8cd70d38b3d9c22da` | Patch equivalent |
| `refs/heads/codex/hdg-tensor-trace-routing-tests` | `80982ba8cfa6b7ae8a1a1dd7cb19f6edffc6de00` | Patch equivalent |
| `refs/heads/codex/hdg-tensor-volume-conventional-q12-parity` | `04d543128ab338fc62aae34e0d34afb1eda25ad9` | Patch equivalent |
| `refs/heads/codex/hdg-time-to-solution-v1` | `c9213c25071d4e9c3fe9e1ebd2b6469c0fd30606` | Ancestry contained |
| `refs/heads/codex/hdg-zhang-shu-limiter` | `eae4e3f77484ff046099f57a50628c352fc6ef3a` | Ancestry contained |
| `refs/heads/codex/igr-ilu-refresh-f646a03` | `29e6a9ba2305dc754c5293379037f5f57544208d` | Patch equivalent |
| `refs/heads/codex/igr-quadrature-equivalence-v1` | `8f252494c6db935789c7eb7b5cb62e362dfdffd0` | Ancestry contained |
| `refs/heads/codex/qualification-tolerance-parameterization` | `dc67aa69a2279ece53b611b15fd917f6f638bcd9` | Patch equivalent |
| `refs/heads/codex/shock-regime-explicit-imex` | `1ec72eb7ca29806055b8bbf72c8bdd9aadae32b8` | Ancestry contained |
| `refs/heads/codex/tensor-dg-igr-breakthrough` | `f95ee67e4bfdc62ccfc50890400f8b843de53797` | Ancestry contained |

## 61 remote retirement candidates

| Ref | Expected/observed SHA | Basis |
|---|---|---|
| `refs/heads/codex/adaptive-dae-reusable-trace` | `eae4e3f77484ff046099f57a50628c352fc6ef3a` | Ancestry contained |
| `refs/heads/codex/hdg-case-context-v1` | `e75f3cbd9428b7b044b4400186336153909f1ac6` | Patch equivalent |
| `refs/heads/codex/hdg-cuda-borrowed-llf-v1` | `18bec20e1b189127dfb1148f8c7bdd538c081e54` | Ancestry contained |
| `refs/heads/codex/hdg-cuda-igr-extension-v1` | `1086eff358da003ecf455d578271afd065981d55` | Ancestry contained |
| `refs/heads/codex/hdg-cuda-numerical-alignment-fp64-perf-v3` | `55d9a33e689be5a4087d6d3afde84102ea7e4870` | Ancestry contained |
| `refs/heads/codex/hdg-generic-boundary-lowering-v1` | `95c65b558a48e80a21511caf3324efb8f2ee83bc` | Ancestry contained |
| `refs/heads/codex/hdg-generic-contract-v1` | `c5444f34bd4eb1d8cc59f70f8832109b1a031b41` | Patch equivalent |
| `refs/heads/codex/hdg-generic-cpu-engine-v1` | `da077f7ebd78177821c4cb84f81c5b8257d8dd53` | Ancestry contained |
| `refs/heads/codex/hdg-generic-cuda-data-v1` | `69a1f35583f23d87c2d6a9ac3852b1dc04f1bb7c` | Ancestry contained |
| `refs/heads/codex/hdg-generic-cuda-main-adapter-v1` | `fd896f9126f151f7175d27ce4eaddaedd62008c6` | Ancestry contained |
| `refs/heads/codex/hdg-generic-cuda-session-v2` | `fdb1a2b9a831d6845f8778b826e233cefa3a4fdd` | Ancestry contained |
| `refs/heads/codex/hdg-generic-cuda-topology-v1` | `e970222e0cb3d411c3356b3b2db49ab485c70ad8` | Ancestry contained |
| `refs/heads/codex/hdg-igr-extension-boundary-v1` | `cdc1dfaacb2cc9f2c6a213b54f17b722e188813e` | Ancestry contained |
| `refs/heads/codex/hdg-igr-symmetric-field-split` | `4a61c382b03ffce33eb44bb311e1cf92a5682dd9` | Ancestry contained |
| `refs/heads/codex/hdg-main-case-session-contract-v2` | `c4d1a5d3deaf8197f042822e0f070fac0fc260a9` | Ancestry contained |
| `refs/heads/codex/hdg-main-config-adapter-v1` | `6fe286e751014d28b6b783f8283b15a1a2c383c0` | Ancestry contained |
| `refs/heads/codex/hdg-main-cpu-mfem-partition-v1` | `f5c7ab42a8808df9e9323a5c5ddf331b7f84cf96` | Ancestry contained |
| `refs/heads/codex/hdg-main-cpu-rank2-adapter-v1` | `9227b359a143d184172cade27876dad97679204e` | Ancestry contained |
| `refs/heads/codex/hdg-main-cpu-serial-adapter-v1` | `fd6d178bb508db8c64b00b697b556b9902f44a63` | Ancestry contained |
| `refs/heads/codex/hdg-main-cuda-adapter-v1` | `85ecd8c6dba1ec77b9f1c7bcc88e5a00418432dd` | Ancestry contained |
| `refs/heads/codex/hdg-paraview-output-v1` | `b856c67e9a3491b77cc86fa1304df4517cff6efe` | Patch equivalent |
| `refs/heads/codex/hdg-rank2-build-output-layout` | `f91dab85f61e136edfd91496030910bd70abe4ef` | Ancestry contained |
| `refs/heads/codex/hdg-rank2-campaign-integration-seam` | `6f4bf1f051a96411438fc41b29c91080e5495fe8` | Ancestry contained |
| `refs/heads/codex/hdg-standalone-cpu-domain-v1` | `972ad0f4ecf36fafd02fb31d909fcf703d313b61` | Ancestry contained |
| `refs/heads/codex/hdg-standalone-cuda-domain-v1` | `972ad0f4ecf36fafd02fb31d909fcf703d313b61` | Ancestry contained |
| `refs/heads/codex/hdg-standalone-manifest-retirement-v1` | `7fe32c7baa67f3ff1d7955f14ca351f01d85e87d` | Ancestry contained |
| `refs/heads/codex/hdg-standalone-sc25-oracle-seam-v1` | `e3fa7fe7e121ef1124c147b21912fd97acab7f74` | Ancestry contained |
| `refs/heads/codex/hdg-standalone-shared-seam-v1` | `924f8c96403d001998e21e083789292adf00105a` | Ancestry contained |
| `refs/heads/codex/hdg-standalone-typed-dispatch-v1` | `eb0a8e8aebed9a618a56c640fcd88e6efd9ab6fa` | Ancestry contained |
| `refs/heads/codex/hdg-tensor-core-backend-v1` | `df5f9c121400a4740e8bc8a4955a5c712af3a9b6` | Ancestry contained |
| `refs/heads/codex/hdg-time-output-v1` | `35c9e45ac036257e4c3761ec3c6694c5a1177f3d` | Ancestry contained |
| `refs/heads/codex/hdg-time-to-solution-v1` | `c9213c25071d4e9c3fe9e1ebd2b6469c0fd30606` | Ancestry contained |
| `refs/heads/codex/hdg-time-validation-v2` | `283d8ee7d573c225c40bd76490ce5a89535590c4` | Ancestry contained |
| `refs/heads/codex/hdg-traced-igr-smooth-transition-cpu-rank2` | `a393ba6920c07bd43e487d8d37d5561364605d29` | Ancestry contained |
| `refs/heads/codex/hdg-traced-igr-smooth-transition-cuda-rank2` | `c2c6f89f11bcc07e59bdadee812a321b7ec8c908` | Ancestry contained |
| `refs/heads/codex/shock-regime-explicit-imex` | `1ec72eb7ca29806055b8bbf72c8bdd9aadae32b8` | Ancestry contained |
| `refs/heads/codex/tensor-dg-igr-breakthrough` | `f95ee67e4bfdc62ccfc50890400f8b843de53797` | Ancestry contained |
| `refs/heads/contract/hdg-euler-device-build-seam` | `0d40f33f00ab0a69e4ed6f53aacbd9ae6953920e` | Ancestry contained |
| `refs/heads/contract/hdg-run-lifecycle-w8` | `ab2e08b3c4e6a9755983abd61fb53af8faf2dda9` | Ancestry contained |
| `refs/heads/contract/hdg-two-phase-w8a` | `20dd44058e04156cce8551c3e310f48d667cc96e` | Ancestry contained |
| `refs/heads/feature/hdg-euler-device-full` | `f217e46ce3a851abcdcd7eaa645eb3d63b832ef0` | Ancestry contained |
| `refs/heads/feature/hdg-euler-device-v2` | `ce91f26936808544afe8346f873431d16f820865` | Ancestry contained |
| `refs/heads/feature/hdg-euler-mpi` | `dbbd35faa0e233df394a9c67a05a1fd395a2e004` | Ancestry contained |
| `refs/heads/feature/hdg-euler-mpi-device` | `f53fc4d55e698eace0939da599a5b41f19a51b9c` | Ancestry contained |
| `refs/heads/feature/hdg-trace-dg-mpi` | `6f6e9c359d7bb9f21e82af6d25dc15ae8d841f2a` | Ancestry contained |
| `refs/heads/gpu` | `7b7a8e2c8c813672f6e20d005860c7b36765865a` | Ancestry contained |
| `refs/heads/integration/hdg-cpu-w5c` | `4f70d39ac23f9e35b0953a49d20bed86f459fb96` | Ancestry contained |
| `refs/heads/integration/hdg-functional-gpu-w7` | `ac271a038c9d26648fce8f2ebf6afb0113a92495` | Ancestry contained |
| `refs/heads/integration/hdg-generic-backend-v1` | `4f7e763dc774ba53a2e2ff4888c2a7ea65294a61` | Ancestry contained |
| `refs/heads/integration/hdg-generic-cpu-reference-closure-v1` | `4acfa881dd21d35c9d3e6e8b76cd382c1c23c30c` | Ancestry contained |
| `refs/heads/integration/hdg-generic-cuda-main-v1` | `10c5a6efa92a6ceb667f01e11602855126d1e6cb` | Ancestry contained |
| `refs/heads/integration/hdg-gpu-w6` | `88cd3e218610e8acadfe9d7a30484ae4c04ee581` | Ancestry contained |
| `refs/heads/integration/hdg-main-application-v1` | `8b5df469e427642c0c15f485720ca3a2e58d2982` | Ancestry contained |
| `refs/heads/integration/hdg-main-backend-v2` | `fdb1a2b9a831d6845f8778b826e233cefa3a4fdd` | Ancestry contained |
| `refs/heads/integration/hdg-runtime-igr-w8` | `b0e45bfc5b789879d4f19a1d04bb96e949c38261` | Ancestry contained |
| `refs/heads/integration/hdg-runtime-w8a-shared` | `7f0812cfc96c6d0288ed1e6126ba2cf223af64ab` | Ancestry contained |
| `refs/heads/integration/hdg-standalone-closure-v1` | `972ad0f4ecf36fafd02fb31d909fcf703d313b61` | Ancestry contained |
| `refs/heads/integration/hdg-traced-igr-smooth-transition-rank2-v4` | `db3da20e6ad4e6bf635efd91960b751438c1258e` | Ancestry contained |
| `refs/heads/integration/hdg-w5c-device-sm120` | `c53fa8101c27d81e4043d9484af2ead50c4ab4fb` | Ancestry contained |
| `refs/heads/new_dgsem` | `4a149ac453908e87d8fefd8ba9a75f341accd635` | Ancestry contained |
| `refs/heads/recovery/hdg-euler-cpu-03c-continuous-recovery` | `51b669f5e18148468d200d9e3afc2bea461346c6` | Ancestry contained |

## Retained remote refs

| Ref | Expected/observed SHA |
|---|---|
| `refs/heads/main` | `b04f8e2d7dbdff8947302e387912041aa656e746` |
| `refs/heads/dev` | `db4d2ae1af3f55fe87f3b0360804d4834169a691` |
| `refs/heads/dgsem` | `20b120ff3fe54ece5863793d58a7df29b3da2e87` |

## Three old worktree candidates

| Registered path | Expected HEAD | Tracked dirty / untracked / ignored |
|---|---|---:|
| `/home/jarias9/.codex/worktrees/cigmo-overnight/dg_igr_miniapp` | `06c3f0d1cd7393c5fa3bfdcb5b518842cbb22731` (detached) | 0 / 0 / 0 |
| `/home/jarias9/.codex/worktrees/pace-a100/dg_igr_miniapp` | `06c3f0d1cd7393c5fa3bfdcb5b518842cbb22731` (detached) | 38 / 3 / 0 |
| `/home/jarias9/.codex/worktrees/sedov-fp32-comparison/dg_igr_miniapp` | `06c3f0d1cd7393c5fa3bfdcb5b518842cbb22731` (detached) | 6 / 0 / 0 |

Remove through Git only after successful restoration, current state/writer checks, validation and approval. PACE and Sedov remain dirty; exact status, full-index binary patches and untracked-content hashes are in `cleanup_proposal.json` and `cleanup_worktree_states.json`. No ignored cache files were observed in these three worktrees, so no separate cache deletion is proposed.

## Superseded PR closure candidates

| PR | Expected head SHA | Expected base SHA | Retained equivalents | Current gate |
|---|---|---|---|---|
| [#37](https://github.com/comp-physics/dg_igr_miniapp/pull/37) | `e75f3cbd9428b7b044b4400186336153909f1ac6` | `72212509a2fdebb8f44eb3a72e39257866c40a01` | `c12f26a` | **Passed at `0e4d578`**: Build and test current CaseContext communicator identity on preserved source; inspect post-cherry-pick evolution. |
| [#38](https://github.com/comp-physics/dg_igr_miniapp/pull/38) | `c5444f34bd4eb1d8cc59f70f8832109b1a031b41` | `72212509a2fdebb8f44eb3a72e39257866c40a01` | `b228265`, `93539e6` | **Passed at `0e4d578`**: Confirm current two-rank empty-local-rank admission fixture is registered and passes with the initialized MFEM tuple. |

Both must still be open at these exact guarded head/base refs before closure. Close as superseded before retiring their remote head/base refs. Their retained equivalent patches already exist; no PR merge is proposed.

## Separate later decisions

Remote repository/default-branch/checkout rename, publication/merge, primary artifact retirement, six historical archive refs, 12 experiment refs, historical remote dgsem and stash retirement remain outside this batch. The current implementation and recovery remain reviewable before any further irreversible step.

Supporting evidence: `branch_disposition.md`, `branch_analysis.json`, `experiment_review.md`/`.json`, `cleanup_worktree_states.json`, `cleanup_proposal.json`, `cleanup_preflight.py`, and `cleanup_preflight_result.json` under `/tmp/igrlab-consolidation-git`.
