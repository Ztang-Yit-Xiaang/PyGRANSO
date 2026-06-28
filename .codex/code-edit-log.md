# Code Edit Log

Entries record Codex-assisted work sessions, findings, validation, conclusions, autonomous next work, human reflection, and required human action.

## Torch-OSQP dense reference migration

- Status: planned
- Start local time: 2026-06-23 21:55:52 -05:00
- End local time: 2026-06-23 21:56:06 CDT-0500
- Duration: Not recorded

### Goal

- Implement the approved dense-LU Torch-OSQP architecture, validation pipeline, support policy, research archive, and revised PDF.

### What changed

- `git status`: M README.md
- `git status`: M docs/MIXED_PRECISION.md
- `git status`: M docs/UNCONSTRAINED_AND_OSQP.md
- `git status`: M pygranso/private/bfgsHessianInverse.py
- `git status`: M pygranso/private/bfgssqp.py
- `git status`: M pygranso/private/qpSteeringStrategy.py
- `git status`: M pygranso/private/qpTerminationCondition.py
- `git status`: M pygranso/private/solveQP.py
- `git status`: M pygranso/pygransoOptions.py
- `git status`: ?? bench_osqp_runtime.py
- `git status`: ?? bench_pygranso_osqp_workloads.py
- `git status`: ?? presentations/
- `git status`: ?? pygranso/private/osqpTorchAdapter.py
- `git status`: ?? pygranso/private/torchOSQP.py
- `git status`: ?? test_osqp_torch_adapter.py

### What was found

- The PyGRANSO worktree already contains uncommitted sparse-CG, CUDA Graph, adapter, benchmark, documentation, and test work that must be preserved before refactoring.
- The requested migration spans source, tests, documentation, generated evidence, Git archive references, and a rendered PDF.

### Validation

- None

### Conclusion

- None

### Next steps

**Codex can proceed:**

- Validate and archive the current research snapshot, then implement and verify the approved migration without resetting user changes.

**Human reflection:**

- The signed archive tag depends on an available Git signing key; this will be tested and reported rather than silently downgraded.

### Human action

- None

## Torch-OSQP dense reference migration completed

- Status: completed
- Start local time: 2026-06-23T21:55:52-05:00
- End local time: 2026-06-24 11:58:45 CDT-0500
- Duration: approximately 14h 2m

### Goal

- Implement and validate the approved dense-LU Torch-OSQP architecture, backend policy, evidence pipeline, CI gates, research archive, and revised PDF.

### What changed

- Created archive branch archive/sparse-cg-cuda-graph at commit da142c1 and preserved the research snapshot; a signed tag was not created because no signing key is configured.
- Replaced the package Torch solver with reusable DenseLUSolver, dense direct ADMM, Ruiz scaling, deterministic adaptive rho, strict polishing, warm starts, validation, and optimizer-owned TorchOSQPWorkspace state.
- Reworked osqpTorchAdapter.py, solveQP.py, bfgssqp.py, qpTerminationCondition.py, and public options for the three-value backend policy, full fallback telemetry, strict input contracts, and per-run workspace invalidation.
- Replaced legacy tests with deterministic, differential, workspace, metamorphic, randomized, and B1/B2/B3 end-to-end coverage; added Linux/Windows/macOS core and nightly CI workflows.
- Added dense benchmark and stability evidence generators plus local CPU/CUDA CSV, JSON, Markdown, and performance artifacts under output/.
- Revised README and technical documentation; added a maintained engineering specification and rendered the visually checked 10-page revised PDF.
- Disabled the legacy bench_osqp_runtime.py entry point in favor of bench_osqp_dense_reference.py; the executable research version remains on the archive branch.

### What was found

- Strict polishing exposed two real defects: refinement was applied to the regularized rather than exact KKT system, and duplicate equality/bound active rows made the dual split nonunique. Exact-KKT polishing plus equality-first duplicate removal fixed both.
- PyGRANSO stationarity QPs created a float32 equality RHS under double precision, and an unindexed cuda target was compared literally with cuda:0. Canonical dtype/device handling fixed B1/B2/B3 Torch CPU and CUDA runs.
- Float32 cannot honestly support the float64 conditioning envelope at 1e-5 tolerances; it is now qualified to an estimated scaled-KKT condition near 1e2, while float64 remains authoritative through about 1e8.
- Local NVIDIA correctness passed, but B1/B2/B3 Torch CUDA medians were 14.4x to 20.0x builtin CPU runtime, so CUDA correctly remains unpromoted for auto selection.

### Validation

- Baseline legacy adapter suite: 79 passed before refactoring.
- Final deterministic/differential/metamorphic/end-to-end suite: 59 passed; one unrelated NumPy deprecation warning.
- Ruff static check passed for package, tests, benchmarks, stability generator, and PDF renderer.
- Windows CPU float64 stability: 300/300 passed; Windows CPU qualified float32: 200/200 passed.
- NVIDIA CUDA float64: 220/220 passed; NVIDIA CUDA qualified float32: 200/200 passed.
- B1/B2/B3 builtin-vs-Torch observables agreed; CUDA performance promotion gate failed at 20.0x, 14.4x, and 15.8x slowdown.
- Workflow YAML parsed successfully; revised PDF rendered to 10 pages, contained the required text, and passed page-by-page visual inspection.

### Conclusion

- The dense Torch reference pipeline is implemented and locally validated within its stated Windows CPU/CUDA and dtype-specific numerical envelopes; no accelerator is auto-promoted without its remaining release gates.

### Next steps

**Codex can proceed:**

- Run and inspect the new GitHub Actions matrix on Linux, Windows, macOS, PyTorch 2.8, and current stable.
- Optimize or replace the CUDA dense backend behind the same factorize/solve boundary before reconsidering automatic promotion.

**Human reflection:**

- The conservative float32 conditioning limit is intentional evidence-based scoping, not a tolerance relaxation; decide whether float32 should remain public or be described as experimental.
- CUDA correctness is strong, but automatic selection would currently be a severe performance regression.

### Human action

- Configure a Git signing key, then create signed tag research-sparse-cg-cuda-graph-final at da142c1.
- Review and commit the feature/torch-osqp-dense-reference worktree, then run the configured CI workflows before release.
- Provide real ROCm and Apple MPS runners before making either backend support claim.

## Torch-OSQP artifact cleanup bookkeeping

- Status: completed
- Start local time: 2026-06-24T12:00:00-05:00
- End local time: 2026-06-24 11:59:25 CDT-0500
- Duration: approximately 2m

### Goal

- Keep superseded local evidence and PDF render scratch out of the implementation handoff.

### What changed

- Updated .gitignore with exact paths for temporary PDF renders and superseded failed/smoke evidence runs; final PDF and release evidence remain visible under output/.

### What was found

- The managed workspace ACL prevented deleting generated binary artifacts, so exact ignore rules were used without hiding the final evidence directories.

### Validation

- git status no longer reports tmp/; final output/ remains available for review.

### Conclusion

- The handoff is cleanly scoped despite sandbox-owned generated files remaining on disk.

### Next steps

**Codex can proceed:**

- None

**Human reflection:**

- None

### Human action

- None

## Torch-OSQP final diff hygiene

- Status: completed
- Start local time: 2026-06-24T12:02:00-05:00
- End local time: 2026-06-24 12:00:11 CDT-0500
- Duration: under 1m

### Goal

- Remove the final whitespace defect reported by git diff --check.

### What changed

- Removed the extra trailing blank line from pygranso/private/solveQP.py.

### What was found

- git diff --check reported one new blank line at EOF.

### Validation

- git diff --check passed after the edit; only a line-ending informational warning remains for the archived benchmark stub.

### Conclusion

- The implementation diff has no whitespace errors.

### Next steps

**Codex can proceed:**

- None

**Human reflection:**

- None

### Human action

- None

## Torch-OSQP completion audit

- Status: planned
- Start local time: 2026-06-24T12:01:40-05:00
- End local time: 2026-06-24 12:01:54 CDT-0500
- Duration: Not recorded

### Goal

- Prove the active pipeline goal requirement-by-requirement, correct remaining gaps, and establish clean source/evidence provenance.

### What changed

- Planned audit only; implementation files are not yet changed in this continuation.

### What was found

- The feature branch still points at the archive commit with the complete implementation uncommitted, the signed archive tag is absent, and evidence manifests identify a dirty archive-base worktree.

### Validation

- Initial git status and branch/tag inspection completed.

### Conclusion

- Completion is not yet proven; a full audit and provenance cleanup are required.

### Next steps

**Codex can proceed:**

- Audit source, tests, evidence, PDF, CI, and signing state against every explicit requirement.

**Human reflection:**

- The requested signed tag must represent a real configured signing identity; it must not be faked with an ephemeral key.

### Human action

- None at this checkpoint.

## Torch-OSQP completion audit and final evidence

- Status: completed
- Start local time: 2026-06-24T12:01:40-05:00
- End local time: 2026-06-24 12:44:55 CDT-0500
- Duration: 42m 30s

### Goal

- Implement and validate the revised dense Torch-OSQP development and validation pipeline, correct audit gaps, produce the revised PDF and release evidence, and report any promotion blockers honestly.

### What changed

- pygranso/private/torchLinearSolve.py and torchOSQP.py: implemented finite-checked reusable LU, dense KKT ADMM, Ruiz scaling, deterministic adaptive rho, strict polishing, warm starts, and diagnostics.
- pygranso/private/osqpWorkspace.py and bfgssqp.py: added a private per-optimizer workspace with structural invalidation and compatible value-update reuse.
- pygranso/private/osqpTorchAdapter.py, solveQP.py, and pygransoOptions.py: implemented the public auto/builtin/torch policy, validated size/memory/device fallback telemetry, common defaults, migration errors, and explicit-Torch failure semantics.
- pygranso/private/qpTerminationCondition.py, qpSteeringStrategy.py, and bfgssqp.py: preserved original hard QP failure causes instead of masking them with secondary unpack or NoneType errors.
- tests/, torch_osqp_stability.py, bench_osqp_dense_reference.py, and bench_pygranso_osqp_workloads.py: added deterministic unit, differential, metamorphic, randomized, failure-reproduction, provenance, and B1/B2/B3 performance validation.
- .github/workflows/: added cross-platform deterministic/nightly matrices and a manual real-hardware CUDA promotion gate.
- README.md, docs/, and output/pdf/Full Development and Validation Pipeline - Revised.pdf: rewrote the pipeline as an executive summary plus decision-complete engineering specification with support matrix, risks, decisions, migration, and evidence.
- .gitignore and legacy files: isolated generated evidence, removed custom-CG/CUDA-Graph active-path artifacts, and retained the archived baseline on archive/sparse-cg-cuda-graph.
- `git status`: M .gitignore
- `git status`: M README.md
- `git status`: D  bench_osqp_runtime.py
- `git status`: M bench_pygranso_osqp_workloads.py
- `git status`: M docs/MIXED_PRECISION.md
- `git status`: M docs/UNCONSTRAINED_AND_OSQP.md
- `git status`: D  presentations/OSQP_Torch_Translation_Progress.pptx
- `git status`: D  presentations/OSQP_Torch_Translation_Progress_speaker_notes.md
- `git status`: M pygranso/private/bfgssqp.py
- `git status`: M pygranso/private/osqpTorchAdapter.py
- `git status`: M pygranso/private/qpSteeringStrategy.py
- `git status`: M pygranso/private/qpTerminationCondition.py
- `git status`: M pygranso/private/solveQP.py
- `git status`: M pygranso/private/torchOSQP.py
- `git status`: M pygranso/pygransoOptions.py
- `git status`: M pyproject.toml
- `git status`: D test_osqp_torch_adapter.py
- `git status`: ?? .codex/
- `git status`: ?? .github/
- `git status`: ?? bench_osqp_dense_reference.py
- `git status`: ?? docs/FULL_DEVELOPMENT_AND_VALIDATION_PIPELINE.md
- `git status`: ?? docs/TORCH_OSQP_COMPLETION_AUDIT.md
- `git status`: ?? pygranso/private/osqpWorkspace.py
- `git status`: ?? pygranso/private/torchLinearSolve.py
- `git status`: ?? scripts/
- `git status`: ?? tests/
- `git status`: ?? torch_osqp_stability.py

### What was found

- All locally supported differential buckets passed: CPU float64 200/200, CPU float32 200/200, CUDA float64 200/200, and CUDA float32 200/200; the 100 float64 stress cases per backend also passed.
- All 100 high-conditioning float32 stress cases failed on each backend while producing zero release-gate failures, validating a conservative float32 conditioning claim rather than the float64 1e8 guarantee.
- CUDA B1/B2/B3 outputs were equivalent to builtin CPU OSQP but median runtime was 12.48x, 21.33x, and 43.46x slower, so CUDA cannot be auto-promoted under the 5x ceiling.
- Every final evidence manifest identifies the same 84-file source tree SHA-256 43bb246bc3db8cb0f946dff106cd6a4c4e8c15e3a563be5418419c653bf3163f despite the implementation worktree being uncommitted.
- MPS float64 auto fallback needed an explicit CPU result device; hard steering/stationarity QP failures also needed cause-preserving propagation. Both gaps were corrected and regression-tested.
- The signed archive tag remains absent because no signing identity or secret key is configured; the feature worktree also remains uncommitted because the managed Git-write approval quota rejected staging.

### Validation

- python -B -m pytest -q: 71 passed; one pre-existing NumPy deprecation warning and two sandbox cache-write warnings.
- python -B -m ruff check .: passed; only cache-write and removed-rule configuration warnings.
- git diff --check: passed.
- All three GitHub Actions workflow YAML files parsed successfully.
- Final stability evidence: four 300-case manifests completed within the two-hour ceiling with zero supported release-gate failures; float32 stress failures were serialized.
- Final B1/B2/B3 five-repeat performance report completed and failed all three CUDA promotion gates while preserving benchmark equivalence.
- Revised PDF parsed as 10 pages and 33,568 bytes; all rendered page/contact-sheet visual checks were clean.

### Conclusion

- The local implementation, tests, evidence generators, support policy, documentation, and revised PDF are complete and internally consistent. Release completion is still blocked by the genuine signed archive tag, a clean feature commit/evidence provenance cycle, external OS/PyTorch CI runs, and real-hardware backend gates; CUDA is deliberately unpromoted on measured performance.

### Next steps

**Codex can proceed:**

- When Git-write approval becomes available, commit the feature worktree, regenerate the four manifests from the clean commit, and verify that their commit and fingerprint provenance agree.
- After appropriate runners are available, execute the configured Linux/macOS, PyTorch 2.8/current-stable, CUDA, ROCm, and MPS release gates and update the support matrix only for passing backends.

**Human reflection:**

- Decide whether float32 should remain a narrowly qualified convenience path or be excluded from any strong conditioning guarantee; the present evidence strongly favors narrow qualification.
- The dense CUDA implementation is correct but poorly matched to these small repeated QPs; optimization work should not weaken the current fallback-first promotion policy.

### Human action

- Configure or provide the authorized signing identity/key, then create signed tag research-sparse-cg-cuda-graph-final at da142c1641c6f14ba3eb564abe88ff16972d771a.
- Approve the Git-write operation needed to commit the implementation once the managed approval window permits it.
- Provide or authorize the external hosted and real-hardware runners required for Linux/macOS, PyTorch-version, CUDA, ROCm, and MPS release claims.

## Torch-OSQP evidence provenance correction

- Status: completed
- Start local time: 2026-06-25 09:09:54 -05:00
- End local time: 2026-06-25 09:55:49 Central Daylight Time-0500
- Duration: 45m 31s

### Goal

- Align the revised Torch-OSQP documentation, PDF, and generated evidence with the final CUDA promotion results and maintained-source fingerprint.

### What changed

- README.md: replaced stale CUDA 14.4x-20.0x promotion-gate wording with final B1/B2/B3 slowdowns.
- docs/UNCONSTRAINED_AND_OSQP.md: replaced stale CUDA slowdown range with final representative workload slowdowns.
- docs/FULL_DEVELOPMENT_AND_VALIDATION_PIPELINE.md: corrected CUDA support-matrix and performance-gate numbers.
- docs/TORCH_OSQP_COMPLETION_AUDIT.md: corrected CUDA stress-seed count and condition-qualified release-gate evidence counts.
- output/pdf/Full Development and Validation Pipeline - Revised.pdf: regenerated the revised 10-page PDF from corrected Markdown.
- output/stability/*-final/: regenerated CPU/CUDA float32/float64 stability CSVs, manifests, summaries, and failure reproductions against the final maintained-source fingerprint.
- output/stability/torch_osqp_release_summary.md: corrected condition-qualified counts and final source_tree_sha256.
- `git status`: M .gitignore
- `git status`: M README.md
- `git status`: D  bench_osqp_runtime.py
- `git status`: M bench_pygranso_osqp_workloads.py
- `git status`: M docs/MIXED_PRECISION.md
- `git status`: M docs/UNCONSTRAINED_AND_OSQP.md
- `git status`: D  presentations/OSQP_Torch_Translation_Progress.pptx
- `git status`: D  presentations/OSQP_Torch_Translation_Progress_speaker_notes.md
- `git status`: M pygranso/private/bfgssqp.py
- `git status`: M pygranso/private/osqpTorchAdapter.py
- `git status`: M pygranso/private/qpSteeringStrategy.py
- `git status`: M pygranso/private/qpTerminationCondition.py
- `git status`: M pygranso/private/solveQP.py
- `git status`: M pygranso/private/torchOSQP.py
- `git status`: M pygranso/pygransoOptions.py
- `git status`: M pyproject.toml
- `git status`: D test_osqp_torch_adapter.py
- `git status`: ?? .codex/
- `git status`: ?? .github/
- `git status`: ?? bench_osqp_dense_reference.py
- `git status`: ?? docs/FULL_DEVELOPMENT_AND_VALIDATION_PIPELINE.md
- `git status`: ?? docs/TORCH_OSQP_COMPLETION_AUDIT.md
- `git status`: ?? pygranso/private/osqpWorkspace.py
- `git status`: ?? pygranso/private/torchLinearSolve.py
- `git status`: ?? scripts/
- `git status`: ?? tests/
- `git status`: ?? torch_osqp_stability.py

### What was found

- The prior roll-up summary used nominal family counts as supported counts; raw CSV release_gate/support_class fields show condition-qualified counts of 175/199/175/200 for CPU64/CPU32/CUDA64/CUDA32.
- The maintained source fingerprint includes docs/*.md, README.md, scripts, tests, workflows, and solver files, so documentation corrections require regenerated stability manifests for exact provenance.
- CUDA correctness remains passing inside the local supported envelope, but performance evidence still fails the <=5x auto-promotion gate at 12.48x, 21.33x, and 43.46x builtin CPU runtime.

### Validation

- python -B -m pytest -q: 71 passed; warnings limited to pre-existing NumPy deprecation and sandbox cache-write warnings.
- python -B -m ruff check .: passed; warnings limited to removed UP038 ignore and sandbox cache-write warnings.
- git diff --check: passed.
- Workflow YAML parse for torch-osqp-core.yml, torch-osqp-cuda-promotion.yml, and torch-osqp-nightly.yml: passed.
- PDF regeneration and checks: 10 pages; TOC/support/risk/decision/migration sections present; final CUDA numbers present; stale CUDA numbers absent; PyMuPDF contact sheet visually checked.
- Stability regeneration: CPU float64 300 cases/0 numerical failures/0 release-gate failures; CPU float32 300 cases/100 non-gating stress failures/0 release-gate failures; CUDA float64 300 cases/0 numerical failures/0 release-gate failures; CUDA float32 300 cases/100 non-gating stress failures/0 release-gate failures.
- Source fingerprint consistency: current source_tree_sha256 3ca16e2225a62e3c4eecf82b9ae3443d4c75745016b45c8c324fa2cf8cca33fe matches all four final manifests and the roll-up summary.

### Conclusion

- The local implementation, revised documentation/PDF, and regenerated evidence package are internally consistent and validated; release closure is still blocked by a Git index lock/live Git processes, the missing signed-tag identity, and external runner gates.

### Next steps

**Codex can proceed:**

- After the active Git processes exit and the stale `.git/index.lock` is safely cleared, stage and commit the feature branch; after a real signing identity is configured, create the signed research archive tag.

**Human reflection:**

- Float32 support remains intentionally narrower than float64; the evidence now makes the condition-qualified versus stress distinction explicit instead of hiding it behind family totals.

### Human action

- Confirm no Git operation is active, clear the stale `.git/index.lock` if appropriate, provide or configure a real Git signing identity for the required signed archive tag, and run hosted Linux/Windows/macOS plus PyTorch-version and real-hardware backend gates before release promotion.

## Torch-OSQP clean evidence regeneration

- Status: completed
- Start local time: 2026-06-27 10:11:44 -05:00
- End local time: 2026-06-27 10:39:55 CDT-0500
- Duration: Not recorded

### Goal

- Remove unrelated npm artifacts and regenerate final Torch-OSQP stability evidence from the clean feature commit while keeping generated artifacts local.

### What changed

- node_modules/: removed untracked headroom-ai dependency tree after user approval.
- package.json: removed unrelated untracked npm manifest after user approval.
- package-lock.json: removed unrelated untracked npm lockfile after user approval.
- output/stability/windows-cpu-float64-final/: regenerated local final CPU float64 stability CSV, manifest, summary, and failure directory.
- output/stability/windows-cpu-float32-final/: regenerated local final CPU float32 stability CSV, manifest, summary, and failure reproductions.
- output/stability/nvidia-cuda-float64-final/: regenerated local final CUDA float64 stability CSV, manifest, summary, and failure directory.
- output/stability/nvidia-cuda-float32-final/: regenerated local final CUDA float32 stability CSV, manifest, summary, and failure reproductions.
- output/stability/torch_osqp_release_summary.md: updated provenance text to feature commit af052f4 and clean worktree.

### What was found

- The untracked npm artifacts were unrelated to Torch-OSQP and were the only visible worktree dirt before regeneration.
- The first sandboxed deletion attempt was denied by Windows permissions; escalated deletion succeeded after resolving and verifying all targets under the PyGRANSO repo.
- All regenerated final manifests now record git_commit af052f4bcf8e47e2b4c4cdd7654535c758534446, git_dirty=false, source_file_count=84, and source_tree_sha256=3ca16e2225a62e3c4eecf82b9ae3443d4c75745016b45c8c324fa2cf8cca33fe.
- CUDA correctness remains locally passing inside the release-gate envelope, but existing performance evidence still keeps CUDA unpromoted.

### Validation

- python -B torch_osqp_stability.py --device cpu --dtype float64 --seeds 100 --stress-seeds 100 --time-limit-seconds 7200 --output output/stability/windows-cpu-float64-final: 300 cases, 0 numerical failures, 0 release-gate failures.
- python -B torch_osqp_stability.py --device cpu --dtype float32 --seeds 100 --stress-seeds 100 --time-limit-seconds 7200 --output output/stability/windows-cpu-float32-final: 300 cases, 100 expected non-gating stress failures, 0 release-gate failures.
- python -B torch_osqp_stability.py --device cuda --dtype float64 --seeds 100 --stress-seeds 100 --time-limit-seconds 7200 --output output/stability/nvidia-cuda-float64-final: 300 cases, 0 numerical failures, 0 release-gate failures.
- python -B torch_osqp_stability.py --device cuda --dtype float32 --seeds 100 --stress-seeds 100 --time-limit-seconds 7200 --output output/stability/nvidia-cuda-float32-final: 300 cases, 100 expected non-gating stress failures, 0 release-gate failures.
- python -B -m pytest -q: 71 passed; warnings limited to existing NumPy deprecation and cache-permission warnings.
- python -B -m ruff check .: passed; warnings limited to removed UP038 ignore and ruff cache-permission warnings.
- Manifest provenance check: all four final manifests match the current maintained source hash, feature commit af052f4, and git_dirty=false.
- git status --short --branch and git diff --check before report append: branch clean against origin; npm artifacts absent.

### Conclusion

- The confirmed cleanup and clean-commit evidence regeneration are complete; generated artifacts remain local/ignored, and only the required code-edit log update is a tracked local change.

### Next steps

**Codex can proceed:**

- If asked, stage and commit the code-edit log update or rerun/inspect additional release evidence.

**Human reflection:**

- The stability evidence is now cleaner for release review because manifests point to the feature commit instead of relying only on a source-tree hash from a dirty pre-commit state.

### Human action

- Configure a real Git signing identity and create the signed archive tag research-sparse-cg-cuda-graph-final at da142c1; run hosted CPU/PyTorch-version CI and real-hardware backend gates before release promotion.

## Torch-OSQP audit provenance refresh

- Status: completed
- Start local time: 2026-06-28 17:56:40 -05:00
- End local time: 2026-06-28 18:23:14 CDT-0500
- Duration: Not recorded

### Goal

- Refresh stale completion-audit provenance wording and regenerate final stability evidence from the clean updated docs commit.

### What changed

- docs/TORCH_OSQP_COMPLETION_AUDIT.md: updated audit date and replaced stale pre-feature-commit provenance wording with clean-feature-commit/source-hash wording.
- output/stability/windows-cpu-float64-final/: regenerated local final CPU float64 stability evidence after the audit-doc change.
- output/stability/windows-cpu-float32-final/: regenerated local final CPU float32 stability evidence after the audit-doc change.
- output/stability/nvidia-cuda-float64-final/: regenerated local final CUDA float64 stability evidence after the audit-doc change.
- output/stability/nvidia-cuda-float32-final/: regenerated local final CUDA float32 stability evidence after the audit-doc change.
- output/stability/torch_osqp_release_summary.md: updated local ignored roll-up provenance to commit cdbad24 and source hash f7206b5a6c501ca1a58acfdd27686941b55e87977a2ada4db3186ee27e63e717.

### What was found

- The completion audit still claimed manifests identified a dirty source tree before a human-created feature commit; that was stale after the report and evidence regeneration commits.
- Because docs/*.md are part of the maintained-source fingerprint, refreshing the audit doc required regenerating all final evidence buckets to keep manifest provenance exact.
- All regenerated final manifests now record git_commit cdbad242fbc3649a4f9ba765d10ce63a02b7f71c, git_dirty=false, source_file_count=84, and source_tree_sha256=f7206b5a6c501ca1a58acfdd27686941b55e87977a2ada4db3186ee27e63e717.

### Validation

- python -B torch_osqp_stability.py --device cpu --dtype float64 --seeds 100 --stress-seeds 100 --time-limit-seconds 7200 --output output/stability/windows-cpu-float64-final: 300 cases, 0 numerical failures, 0 release-gate failures.
- python -B torch_osqp_stability.py --device cpu --dtype float32 --seeds 100 --stress-seeds 100 --time-limit-seconds 7200 --output output/stability/windows-cpu-float32-final: 300 cases, 100 expected non-gating stress failures, 0 release-gate failures.
- python -B torch_osqp_stability.py --device cuda --dtype float64 --seeds 100 --stress-seeds 100 --time-limit-seconds 7200 --output output/stability/nvidia-cuda-float64-final: 300 cases, 0 numerical failures, 0 release-gate failures.
- python -B torch_osqp_stability.py --device cuda --dtype float32 --seeds 100 --stress-seeds 100 --time-limit-seconds 7200 --output output/stability/nvidia-cuda-float32-final: 300 cases, 100 expected non-gating stress failures, 0 release-gate failures.
- python -B -m pytest -q: 71 passed; warnings limited to existing NumPy deprecation and cache-permission warnings.
- python -B -m ruff check .: passed; warnings limited to removed UP038 ignore and ruff cache-permission warnings.
- Manifest provenance check: all four final manifests match commit cdbad24, git_dirty=false, and current maintained-source hash f7206b5a6c501ca1a58acfdd27686941b55e87977a2ada4db3186ee27e63e717.
- git diff --check: passed.

### Conclusion

- The tracked completion audit and local ignored evidence are now consistent with the clean updated docs commit; the implementation remains locally validated.

### Next steps

**Codex can proceed:**

- If signing is configured, create the signed archive tag; if remote authorization is provided, push the local follow-up commits.

**Human reflection:**

- The evidence package is now easier to audit because the completion audit no longer describes an earlier dirty-worktree phase.

### Human action

- Configure a real Git signing identity and create signed tag research-sparse-cg-cuda-graph-final at da142c1; run hosted CPU/PyTorch-version CI and real-hardware backend gates before release promotion.

