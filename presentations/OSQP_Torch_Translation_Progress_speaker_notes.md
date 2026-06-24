# OSQP-to-Torch Translation Progress Speaker Notes

## Slide 1: Translating PyGRANSO OSQP QP Solves to Torch

Open by separating the achievement from the limitation. We removed mandatory data movement for the Torch path, but we have not implemented the full optimized sparse OSQP CUDA backend.

On-slide bullets:
- Goal: solve PyGRANSO QP subproblems without forced NumPy / CPU conversion
- Current milestone: Torch-native dense OSQP-style prototype
- Main concern: GPU execution does not guarantee speedup

## Slide 2: Where OSQP Appears in PyGRANSO

This slide orients collaborators who know optimization but not the PyGRANSO internals. The point is that OSQP is not called directly by users; it is reached through PyGRANSO's QP subproblem machinery.

On-slide bullets:
- pygransoOptions.py: exposes QP solver and OSQP backend policy
- bfgssqp.py: stores QPsolver and osqp_options
- qpSteeringStrategy.py: solves steering QPs
- qpTerminationCondition.py: solves stationarity QPs
- solveQP.py: dispatches QPsolver='osqp' into the adapter
- osqpTorchAdapter.py: chooses builtin vs Torch backend
- torchOSQP.py: dense Torch ADMM prototype

Callout: Call chain: options -> bfgssqp -> steering / termination QPs -> solveQP -> OSQP adapter -> backend

## Slide 3: PyGRANSO QP Form

Emphasize shape and device conventions. A technically correct solve is not enough if it returns the wrong shape or silently changes device semantics.

On-slide bullets:
- PyGRANSO builds stationarity / steering QPs as: H, f, Aeq, beq, LB, UB
- H: quadratic matrix
- f: linear objective vector
- Aeq, beq: equality constraints, sometimes absent
- LB, UB: variable lower and upper bounds
- Inputs may be Torch tensors on CPU or CUDA
- Returned solution must remain a Torch column vector with shape (nvar, 1)

Code / diagram text:
```text
H, f, Aeq, beq, LB, UB

solution shape: (nvar, 1)
```

## Slide 4: Translation to OSQP Canonical Form

This is the mathematical heart of the adapter. The correctness tests mainly verify that this mapping is preserved in both builtin and Torch paths.

On-slide bullets:
- The adapter converts PyGRANSO QP data into OSQP's P, q, A, l, u form
- Equality constraints become fixed lower and upper bounds
- Variable bounds are represented by appending an identity matrix
- If equality constraints are absent, only the identity-bound block is used

Code / diagram text:
```text
P = H
q = f
A = [Aeq; I]
l = [beq; LB]
u = [beq; UB]

No Aeq/beq:
A = I
l = LB
u = UB
```

## Slide 5: Original CPU Builtin Path

This is the path we preserve for reliability. It is not wrong, but it should not be confused with GPU execution when the original data started on CUDA.

On-slide bullets:
- algebra='builtin' keeps the existing compatibility path
- Torch tensors are converted with value.detach().cpu().numpy()
- SciPy CSC matrices are built for P and A
- Python OSQP is called as osqp.OSQP(algebra='builtin')
- The solution is converted back to a Torch column vector
- Reliable and mature, but it is a CPU solve

Callout: Returning a CUDA tensor at the end does not mean the QP was solved on GPU.

## Slide 6: New Torch Backend Path

This is the main implementation milestone. It solves the data-movement problem: CUDA QP tensors no longer have to be copied through NumPy for the Torch backend.

On-slide bullets:
- algebra='torch' keeps QP data as Torch tensors
- Autograd is detached because the QP solve is not differentiated
- CUDA tensors remain on CUDA
- Constraints are built with Torch operations
- The adapter calls solve_torch_osqp(...) instead of Python OSQP

Code / diagram text:
```text
algebra='torch'
  real Torch / CUDA tensor computation
  no NumPy
  no SciPy sparse conversion
  no Python OSQP call
```

## Slide 7: Backend Policy

The important nuance is that torch is not fake GPU if tensors are on CUDA. It is real CUDA tensor computation, but it is not the fully optimized OSQP CUDA backend.

On-slide bullets:
- CPU + auto / builtin: existing Python OSQP CPU path
- CPU + torch: dense Torch prototype on CPU
- CUDA + auto / torch: dense Torch prototype on CUDA
- CUDA + builtin: raises unless cuda_fallback=True
- Any + cuda: reserved future real OSQP CUDA interop, currently raises

Code / diagram text:
```text
builtin -> Python OSQP / SciPy / NumPy / CPU
torch   -> Torch tensor prototype, CUDA-capable, dense
cuda    -> future real OSQP CUDA interop, not implemented yet
```

## Slide 8: Torch ADMM Prototype

This mirrors the OSQP-style ADMM update at a prototype level. The limitation is that the KKT system is dense and solved directly each iteration.

On-slide bullets:
- Build a dense KKT matrix using Torch tensors
- Each iteration solves a linear system with torch.linalg.solve
- Projection is done by clamping z into [l, u]
- Stopping uses OSQP-style primal and dual infinity-norm residuals
- Supported settings include rho, sigma, alpha, max_iter, eps_abs, eps_rel

Code / diagram text:
```text
K = [[P + sigma I, A.T],
     [A,          -(1/rho) I]]

solve K [x_tilde; nu] = rhs
z_next = clamp(z_relaxed + y/rho, l, u)
y_next = y + rho * (z_relaxed - z_next)
```

## Slide 9: What Works Now

Use this as validation evidence. The tests do not prove performance, but they prove the translation and backend routing behavior.

On-slide bullets:
- CPU Torch backend solves a simple bound QP
- CPU Torch backend validates equality-plus-bounds mapping
- CUDA Torch backend preserves device, dtype, and (nvar, 1) shape
- CUDA no-copy guard proves the Torch path does not call the NumPy converter
- CPU builtin regression still passes with Python OSQP installed

Code / diagram text:
```text
pytest -q test_osqp_torch_adapter.py
16 passed
```

## Slide 10: Important Limitation: Sparsity Is Broken

This is the key concern. The prototype removes CPU copying, but it replaces sparse OSQP machinery with dense Torch linear algebra.

On-slide bullets:
- OSQP is fast largely because it exploits sparse matrices
- The Torch prototype currently builds dense matrices
- Bound constraints add an identity matrix, which is mathematically sparse but materialized dense
- The KKT matrix is assembled as one dense block matrix
- torch.linalg.solve treats the system as dense
- to_dense() explicitly destroys sparse layout if sparse tensors appear

Callout: Real CUDA execution can still be slow if the algorithm stops exploiting sparse structure.

## Slide 11: Why GPU May Not Accelerate

The honest interpretation is: algebra='torch' can run on GPU, but performance must be benchmarked. GPU memory alone is not the same as an optimized GPU solver.

On-slide bullets:
- CPU OSQP uses sparse CSC data structures
- CPU OSQP has mature factorization, scaling, adaptive rho, and polishing behavior
- Torch prototype uses dense A and dense K
- Torch prototype solves the dense KKT system repeatedly
- No sparse factorization cache, no warm start logic, no adaptive rho, no scaling, no polishing
- For small or sparse PyGRANSO QPs, CPU OSQP may still win

Code / diagram text:
```text
GPU execution != automatic speedup

Speed depends on problem size, sparsity, copy overhead, and linear algebra structure.
```

## Slide 12: Fallback Behavior and Fake-GPU Risk

This is the fake-GPU scenario. The current adapter tries to prevent silent fallback by making CPU fallback explicit.

On-slide bullets:
- Dangerous case: CUDA tensors fall back to builtin OSQP
- Data copies CUDA -> CPU through .detach().cpu().numpy()
- Python OSQP solves on CPU
- Solution copies back to CUDA at the end
- Default cuda_fallback=False raises instead of silently pretending
- If cuda_fallback=True, the adapter warns explicitly

Code / diagram text:
```text
CUDA tensors + builtin fallback
  -> copy QP data to CPU
  -> solve with Python OSQP
  -> copy solution back to CUDA
```

## Slide 13: The Right Benchmark Question

The benchmark should answer whether the implementation is useful for PyGRANSO's real workload, not only whether it avoids NumPy conversion.

On-slide bullets:
- Bad question: Does it run on GPU?
- Better question: Does algebra='torch' beat builtin CPU OSQP on real PyGRANSO QPs?
- Measure QP size: variables and constraints
- Measure sparsity density
- Compare CPU builtin time, Torch CUDA time, and CPU copy overhead
- Track ADMM iterations and memory use

Code / diagram text:
```text
Benchmark target:
  builtin CPU OSQP vs Torch CUDA prototype
  on QPs generated by PyGRANSO, not only toy QPs
```

## Slide 14: Future Work

This slide should make the next research direction obvious. The prototype is valuable because it identifies the next bottleneck: sparse-preserving GPU linear algebra.

On-slide bullets:
- Add benchmark instrumentation around solveQP
- Compare builtin vs torch on real PyGRANSO-generated QPs
- Preserve sparse structure instead of building dense A and K
- Investigate sparse Torch operations or custom CUDA sparse KKT solves
- Add warm starts and factorization reuse
- Add adaptive rho and scaling equivalents
- Eventually connect to real OSQP CUDA algebra / interop

Callout: Next challenge: keep the no-copy benefit while recovering sparse solver efficiency.

## Slide 15: Final Takeaway

End with the balanced message: this is progress, but the performance story is not finished.

On-slide bullets:
- Level 1: CPU OSQP - reliable, sparse, mature
- Level 2: Torch prototype - real CUDA tensors, no NumPy copy, dense and limited
- Level 3: real OSQP CUDA interop - future target, sparse and optimized
- Current work solves the data-movement problem first
- Next work should preserve sparse solver efficiency

Code / diagram text:
```text
The Torch path is real GPU-capable computation,
but not full OSQP GPU acceleration yet.
```
