# DGSEM-REF-01 failed-attempt-1 evidence manifest

This tracked manifest is the integrity and provenance record for the preserved
failed-attempt-1 directory. It does not make the ignored output directory part
of the source commit and it must not be used as a passing result.

## Preserved path and command provenance

- Worktree-relative directory: `out/dgsem_ref_29ed_cpu/evidence_serial`
- Worktree: `/home/jarias9/.codex/worktrees/29ed/dg_igr_miniapp`
- Lease: nonblocking `/tmp/dg_igr-sm120.lock`
- Command family used for the bounded registered serial attempt:

  ```text
  flock --verbose -n -E 75 /tmp/dg_igr-sm120.lock env DG_IGR_DGSEM_REFERENCE_OUTPUT_DIR=/home/jarias9/.codex/worktrees/29ed/dg_igr_miniapp/out/dgsem_ref_29ed_cpu/evidence_serial ctest --test-dir build_dgsem_ref_29ed_cpu --output-on-failure -R '^dg_igr_dgsem_(basis|flux|operator|advection|igr)_tests$'
  ```

The attempt did not retain a standalone CTest output log in this directory;
its command output was captured only in the originating Codex task transcript.
The four files below are the artifacts written before/during the known operator,
convergence, and scalar-IGR failures. Later passing logs are intentionally kept
in different directories and are not substitutes for the missing attempt-1 log.

## SHA-256 file anchors

```text
857c3b96f56303eca4703656c49a7f0cd4ac0096f24f0d5d35d237c9e64c1303  out/dgsem_ref_29ed_cpu/evidence_serial/consumer_inventory.json
eae8b53b15de437ced13bcd3122c4a1cecf3b60c7f46785c174ea786d608f068  out/dgsem_ref_29ed_cpu/evidence_serial/convergence_contract.json
8a37211eb99cf32796be2cd70797f3c6c5a26e41df7cff108ae23ea28e21052a  out/dgsem_ref_29ed_cpu/evidence_serial/deterministic_flux_fixtures.json
45a014bcea5cdaade8cedb6657a188b2ce2800ec15a089a7ee7a1981e54f2b97  out/dgsem_ref_29ed_cpu/evidence_serial/p1_p4_convergence.csv
```

The SHA-256 of that ordered `sha256sum` stream is
`9c464d145132b56674649daa7a45f4f04309357b82e8565155d6045d2836dfed`.

## Known integrity caveat

The preserved directory is mode `0755` and its files are mode `0644`; it is
writable by the worktree owner. The hashes above, unique output path, and the
rule never to rerun into this directory are therefore the integrity controls.
Reviewers should recompute the hashes before relying on the artifacts. This
manifest records the caveat rather than claiming filesystem immutability.
