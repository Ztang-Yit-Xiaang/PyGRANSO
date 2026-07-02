# Full Development and Validation Pipeline

Version: 2.0  
Date: 2026-06-23  
Status: Implementation specification

## Part I - Executive Summary

### 1. Decision

Build a correctness-first, Torch-native OSQP reference route for PyGRANSO.
The implementation is dense and uses PyTorch's built-in LU factorization. It
is not presented as a sparse large-scale solver. Sparse acceleration is a
future backend that must fit behind the same internal interface.

The first trustworthy milestone supports feasible convex QPs and is judged by
observable outcomes: compatible status, feasibility, stationarity, residuals,
and objective. It does not require identical ADMM trajectories or iteration
counts across devices or backends.

### 2. Primary acceptance envelope

| Item | Supported contract |
| --- | --- |
| Problem class | Feasible convex QPs in OSQP form |
| Authoritative precision | float64 |
| Qualified precision | float32 with looser tolerances and KKT conditioning approximately 1e2 |
| Automatic dense limit | KKT dimension n + m <= 2400 |
| Float64 supported conditioning | Approximately 1e8 |
| Stress-only conditioning | Approximately 1e10 |
| Linear algebra | Reusable torch.linalg.lu_factor_ex and lu_solve |
| Required features | Ruiz scaling, adaptive rho, polishing, warm starts |
| Default CPU policy | Builtin OSQP |
| Default accelerator policy | Torch only after that backend passes its gates |
| Failure budget | Zero unexplained failures inside the supported matrix |

### 3. Backend support matrix

| Backend | Precision | Initial status | Promotion evidence |
| --- | --- | --- | --- |
| Torch CPU on Linux | float32/float64 | Release-gated | Core, nightly, end-to-end |
| Torch CPU on Windows | float32/float64 | Local correctness pass; release-gated | Cross-version CI still required |
| Torch CPU on macOS | float32/float64 | Release-gated | Core, nightly, end-to-end |
| NVIDIA CUDA | float32/float64 | Unpromoted | Earlier local correctness evidence passed, but current-source full CUDA reruns on the GTX 1650 exceeded the two-hour local wrapper and 12.48x, 21.33x, and 43.46x end-to-end slowdowns failed the <=5x gate |
| AMD ROCm | float32/float64 | Unclaimed | Real-hardware runner required |
| Apple MPS | float32 only | Unclaimed | Reusable LU must pass on real Apple hardware |
| Apple MPS float64 | Unsupported | Builtin CPU result under auto | MPS does not support float64 tensors |

### 4. Outcome

The default `auto` policy follows the requested optimization device. CPU work
uses builtin OSQP. Accelerator work uses Torch only when the device, KKT size,
memory estimate, correctness suite, and performance sanity gate are all
satisfied. Any automatic Torch exception or unsolved status produces a visible
builtin fallback and a complete causal record. Explicit Torch requests never
change backend silently.

### 5. Main risks and controls

| Risk | Control |
| --- | --- |
| Repeated dense refactorization | Cache LU and refactor only when the KKT matrix changes |
| Dense memory growth | n + m limit plus conservative working-memory preflight |
| Float32 stagnation | Dtype-aware 1e-5 defaults and condition-aware classification |
| Hidden CPU fallback | Warning plus structured backend, trigger, transfer, and result fields |
| Cross-run state contamination | One private workspace per BFGS-SQP run |
| Misclassified infeasibility | Do not claim certificates in the first milestone |
| Optional polishing corrupts a valid result | Accept only nonworse KKT metrics; numerical failure raises |
| Unsupported accelerator claims | Promote each backend only after real-hardware evidence |

<!-- PAGEBREAK -->

## Part II - Decision-Complete Engineering Specification

### 6. Architecture

```text
PyGRANSO BFGS-SQP
  -> steering QP or stationarity QP
  -> solveQP.py
  -> osqpTorchAdapter.py
  -> canonical P, q, A, l, u
     -> builtin CPU OSQP
     -> dense Torch OSQP reference
        -> OSQP ADMM equations
        -> private DenseLUSolver boundary
        -> PyTorch LU factorization and repeated solves
```

The linear-solver boundary is internal. The user selects only
`osqp_algebra={auto,builtin,torch}`. No nested linear-solver selector is
exposed until a second validated Torch backend exists.

### 7. Canonical QP contract

Solve:

```text
minimize    0.5 * x' P x + q' x
subject to  l <= A x <= u
```

Requirements:

- P is square, finite, and positive semidefinite within the supported numerical envelope.
- q and A are finite.
- l and u may contain infinity but never NaN, and l <= u elementwise.
- All Torch inputs use the same device and dtype after adapter normalization.
- Only float32 and float64 are supported.
- Near-symmetric P is replaced by 0.5 * (P + P.T) only when its infinity-norm asymmetry is within a dtype-aware tolerance.
- A diagnostic eigenvalue check is available for tests and debugging; it is not paid on every solve.

### 8. Dense linear-solver lifecycle

The private solver owns the matrix and its LU factors.

```text
factorize(K)
  validate square, dtype, device, and finite values
  call torch.linalg.lu_factor_ex(K, check_errors=False)
  reject nonzero info or non-finite factors

solve(rhs)
  normalize a vector RHS to shape (n, 1)
  validate shape, dtype, device, and finite values
  call torch.linalg.lu_solve(LU, pivots, rhs)
  reject non-finite solutions
  optionally report ||Kx-b|| / max(1, ||b||)

refactorize(K)
  run only after P, A, rho, sigma, dtype, or device changes
```

Reuse the same factorization across ADMM iterations and across compatible
updates to q, l, and u. A change to P or A values refactorizes. A change to
dimensions, sparsity pattern, constraint ordering, dtype, device, or backend
invalidates the complete workspace.

### 9. Optimizer-owned workspace

Each BFGS-SQP run owns one private `TorchOSQPWorkspace`. It stores:

- original-coordinate x, z, and y warm state;
- problem and structure signatures;
- cached scaling vectors and cost scale;
- the latest adaptive rho value;
- the dense LU solver and counters;
- the builtin OSQP update workspace;
- the last structured diagnostics.

Independent, nested, or concurrent PyGRANSO runs never share warm state or
factorizations through module globals.

### 10. Preserved OSQP equations

The direct KKT system is:

```text
[ P + sigma I    A'          ] [x_tilde] = [sigma x - q]
[ A             -diag(rho)^-1] [nu     ]   [z - y/rho ]
```

Recover and update:

```text
z_tilde = z + (nu - y) / rho
x_next = alpha * x_tilde + (1-alpha) * x
z_relaxed = alpha * z_tilde + (1-alpha) * z
z_next = project_[l,u](z_relaxed + y/rho)
y_next = y + rho * (z_relaxed - z_next)
```

Stopping residuals are evaluated in original coordinates:

```text
r_primal = ||A x - z||_inf
r_dual   = ||P x + q + A' y||_inf
```

Nonunique problems are accepted by feasibility, stationarity, objective, and
compatible status rather than by matching x exactly.

### 11. Required numerical features

#### 11.1 Ruiz scaling

Use ten deterministic diagonal-equilibration passes, solve the scaled problem,
then unscale x, z, and y. Reuse cached scaling for vector-only parametric
updates. All acceptance residuals and objectives are reported in original
coordinates.

#### 11.2 Adaptive rho

Use a deterministic interval of 50 and an update tolerance of 5. Equality rows
receive the larger vector-valued rho policy. Every accepted rho change rebuilds
and refactorizes K while continuing from the current x, z, and y.

<!-- PAGEBREAK -->

#### 11.3 Polishing

Build the active-set polishing KKT system through the same LU boundary. Reuse
the factorization for refinement steps. A candidate is accepted only when its
KKT metric is no worse or it satisfies the target tolerances. A factorization,
refinement, or candidate-acceptance failure raises when polishing was requested.

#### 11.4 Warm starts

Warm starts are enabled internally. Compatible vector updates reuse x, z, y,
rho, scaling, and LU. Matrix-value updates retain x, z, and y but recompute
scaling and factors. Structural changes clear the workspace.

### 12. Defaults

| Setting | Default |
| --- | ---: |
| rho | 0.1 |
| sigma | 1e-6 |
| alpha | 1.6 |
| max_iter | 4000 |
| check_termination | 25 |
| float64 eps_abs and eps_rel | 1e-8 |
| float32 eps_abs and eps_rel | 1e-5 |
| scaling | 10 |
| adaptive_rho | true |
| rho_update_interval | 50 |
| rho_update_tolerance | 5 |
| polishing | true |
| polish_delta | 1e-6 |
| polish_refine_iter | 3 |
| warm_start | true |

### 13. Backend selection and fallback

| Request | Behavior |
| --- | --- |
| auto on CPU | Builtin CPU OSQP |
| auto on validated accelerator inside limits | Torch reference route |
| auto on unsupported accelerator | Warn and use builtin |
| auto above size or memory envelope | Warn and use builtin |
| auto Torch exception or unsolved status | Warn, retry builtin, retain causal telemetry |
| explicit builtin | Builtin CPU OSQP and return on requested device |
| explicit torch inside limits | Torch reference route |
| explicit torch above limits | Warn and attempt; never silently change backend |

Fallback diagnostics include the requested and selected backends, the original
exception or status, the fallback backend and outcome, and whether data moved
between accelerator and CPU.

### 14. Status and error semantics

- Return structured statuses for solved and maximum-iteration outcomes.
- Raise for invalid inputs, unsupported explicit operations, LU failure, NaN or Inf, and numerical polishing failure.
- Under auto, any unsolved-compatible Torch status is eligible for a warned builtin retry.
- An explicit Torch failure propagates to PyGRANSO's steering or stationarity fallback contract without being rewritten as a secondary unpacking or unbound-variable error.
- Never label a linear solve failure as infeasibility.
- Move primal and dual infeasibility certificates and nonconvex detection to a future milestone.

### 15. Validation pipeline

```text
Dense LU unit tests
  -> KKT assembly and ADMM equation tests
  -> deterministic complete-QP tests
  -> scaling, adaptive-rho, polishing, and warm-state tests
  -> matched-setting builtin-vs-Torch differential tests
  -> PyGRANSO steering and stationarity contracts
  -> metamorphic tests
  -> seeded randomized and conditioning tests
  -> B1/B2/B3 and constrained end-to-end runs
  -> backend-specific hardware gates
  -> performance sanity gate
```

Core CI runs deterministic cases on every change. Nightly validation runs 100
fixed seeds per family and backend bucket with a two-hour budget. Every failure
records the complete QP, seed, settings, environment, results, and residuals.

Inside the supported matrix, the failure budget is zero unexplained failures.
Cases near condition number 1e10 are classified as stress evidence rather than
as guaranteed support.

### 16. Evidence package

Each stability run produces:

- `torch_osqp_stability_results.csv` with case-level gates;
- `torch_osqp_stability_manifest.json` with commit, platform, hardware, Python, PyTorch, OSQP, backend, settings, and seeds;
- `torch_osqp_stability_summary.md` with family totals;
- one serialized reproduction file for every failure.

If the wall-clock budget is exceeded after a case completes, the run still
writes the CSV, manifest, summary, and any failure reproductions collected so
far, marks `timed_out=true` and `partial_results=true` in the manifest, records
the last completed case, and exits nonzero. A timed-out bucket is not passing
release evidence, but it remains reproducible telemetry instead of a silent
artifact loss.

The manifest captures Git provenance before creating its own output directory,
so `git_dirty` describes source state rather than generated evidence files. It
also records status entries and a deterministic SHA-256 over maintained source,
tests, workflows, scripts, and documentation, making dirty-worktree evidence
exactly identifiable before a release commit exists.

Stability reports estimate conditioning with a deterministic dense CPU KKT
diagnostic using the initial common `rho` and `sigma` settings. The QP solves
still execute on the requested backend, but the condition estimate itself does
not call accelerator SVD/condition kernels; this keeps nightly and hardware
evidence buckets inside the two-hour gate.

Differential tests use identical algorithm settings. They compare status,
primal and dual residuals, objective, equality violation, bound violation, and
finite values. Objective gaps use `abs(torch-reference) / max(1, abs(reference))`.

### 17. Performance gate

Performance is not a correctness criterion. It controls only automatic backend
promotion. On representative accelerator-targeted PyGRANSO workloads, the
Torch median end-to-end time, including necessary transfers, must be no worse
than five times builtin CPU OSQP. A backend that fails remains explicit or
unclaimed even when its correctness suite passes.

Current CUDA promotion evidence is intentionally insufficient. Earlier local
NVIDIA correctness buckets passed, and a current-source CUDA float64 smoke
passes, but current-source 100-seed CUDA float64 reruns on the local GTX 1650
exceeded the two-hour wrapper. Independent end-to-end B1, B2, and B3 medians
were 12.48x, 21.33x, and 43.46x the builtin CPU median respectively. CUDA
therefore remains explicit-only and `auto` records a warned
`cuda_not_promoted` builtin fallback.

### 18. Migration sequence

1. Validate and preserve the sparse-CG/CUDA Graph research snapshot.
2. Create archive branch `archive/sparse-cg-cuda-graph`.
3. Create signed tag `research-sparse-cg-cuda-graph-final` when a signing key is available.
4. Remove custom CG, Jacobi, sparse-operator, CUDA Graph, and selection code from the package path.
5. Add DenseLUSolver and optimizer-owned workspace tests.
6. Refactor direct ADMM around reusable LU.
7. Validate scaling, adaptive rho, polishing, and warm starts.
8. Implement the backend policy, migration errors, size guards, and fallback telemetry.
9. Add differential, randomized, hardware, PyGRANSO, and reporting gates.
10. Promote each backend only after its own correctness and performance evidence passes.

### 19. Decision log

| Decision | Rationale |
| --- | --- |
| Dense reference first | Correctness and maintainability are the primary objective |
| Reusable LU instead of solve per iteration | Preserves built-in numerical ownership without repeated refactorization |
| Observable OSQP agreement | Different devices and factorizations need not match trajectories |
| Float64 authoritative | Float32 degrades materially on ill-conditioned KKT systems |
| Feasible convex scope first | Prevents unvalidated certificate claims |
| Auto follows requested device | Avoids unprofitable CPU-to-accelerator transfers |
| Per-run workspace | Prevents cross-run state contamination |
| Backend-by-backend promotion | Support claims require real hardware |
| Five-times performance ceiling | Prevents severe automatic regressions without making speed the success criterion |

### 20. Future work

After the dense reference route passes all applicable gates, a sparse direct or
iterative backend may implement the same factorize/solve/refactorize contract.
Candidates include cuDSS, torch-sla, future PyTorch sparse solvers, and batched
GPU solvers. The ADMM, adapter, PyGRANSO integration, tests, and evidence schema
must not change for a solver replacement.
