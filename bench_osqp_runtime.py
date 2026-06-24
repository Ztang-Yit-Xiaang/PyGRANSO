import argparse
import copy
import csv
import importlib.util
import io
from pathlib import Path
import statistics
import time
import warnings

import numpy as np
import torch
from scipy import sparse

from pygranso.private.osqpTorchAdapter import solve_osqp_torch_qp
from pygranso.private.solveQP import getLastOSQPInfo, resetOSQPWarmState, solveQP

DEFAULT_CASES = [
    "bound",
    "equality",
    "random_spd",
    "active_bound",
    "ill_conditioned_spd",
]
DEFAULT_SIZES = [100, 600]
EXTERNAL_SOLVERS = ["torch_sla_pytorch_cg", "torch_sla_cudss"]
TABLE_COLUMNS = [
    ("case", 12),
    ("n", 6),
    ("backend", 14),
    ("status", 10),
    ("median_ms", 11),
    ("selected", 10),
    ("reason", 28),
    ("objective", 12),
    ("prim_res", 12),
    ("dual_res", 12),
    ("cg_iters", 9),
    ("cg_fix", 7),
    ("compile", 9),
    ("graph", 9),
    ("rho_upd", 8),
    ("scale", 7),
    ("cache", 7),
    ("polish", 9),
    ("external", 12),
    ("setup_ms", 9),
    ("update_ms", 9),
    ("solve_ms", 9),
    ("cg_ms", 9),
    ("admm_ms", 9),
    ("resid_ms", 9),
    ("graph_ms", 9),
    ("dense_mb", 10),
    ("sparse_nnz", 11),
    ("ref_err", 10),
]
ACADEMIC_COLUMNS = [
    "method",
    "case",
    "size",
    "device",
    "median_ms",
    "iqr_ms",
    "speedup_vs_cpu_osqp",
    "speedup_vs_cpu_fresh",
    "speedup_vs_cpu_warm",
    "speedup_ci_low",
    "speedup_ci_high",
    "efficiency_status",
    "selected_policy",
    "cache_hit",
    "cuda_graph",
    "primal_residual",
    "dual_residual",
    "eps_primal",
    "eps_dual",
    "relative_objective_gap",
    "ref_err",
    "cg_iterations",
    "setup_ms",
    "update_ms",
    "solve_ms",
    "graph_replay_ms",
]
EQUALITY_SWEEP_VARIANTS = [
    (
        "eq_converged",
        {
            "cg_fixed_iters": None,
            "adaptive_rho": False,
            "scaling": 0,
            "dense_memory_limit_mb": 1e-12,
        },
    ),
    (
        "eq_fixed2",
        {
            "cg_fixed_iters": "2",
            "adaptive_rho": False,
            "scaling": 0,
            "dense_memory_limit_mb": 1e-12,
        },
    ),
    (
        "eq_fixed5",
        {
            "cg_fixed_iters": "5",
            "adaptive_rho": False,
            "scaling": 0,
            "dense_memory_limit_mb": 1e-12,
        },
    ),
    (
        "eq_fixed10",
        {
            "cg_fixed_iters": "10",
            "adaptive_rho": False,
            "scaling": 0,
            "dense_memory_limit_mb": 1e-12,
        },
    ),
    (
        "eq_adaptive",
        {
            "cg_fixed_iters": None,
            "adaptive_rho": True,
            "scaling": 0,
            "dense_memory_limit_mb": 1e-12,
        },
    ),
    (
        "eq_scaling5",
        {
            "cg_fixed_iters": None,
            "adaptive_rho": False,
            "scaling": 5,
            "dense_memory_limit_mb": 1e-12,
        },
    ),
]
ABLATION_VARIANTS = [
    ("abl_cold_auto", "torch_auto", {}),
    ("abl_warm_cache", "pygranso_torch", {}),
    ("abl_fixed_auto", "pygranso_torch", {"cg_fixed_iters": "auto"}),
    ("abl_fixed1", "pygranso_torch", {"cg_fixed_iters": "1"}),
    ("abl_fixed2", "pygranso_torch", {"cg_fixed_iters": "2"}),
    ("abl_fixed3", "pygranso_torch", {"cg_fixed_iters": "3"}),
    ("abl_fixed5", "pygranso_torch", {"cg_fixed_iters": "5"}),
    ("abl_adaptive_rho", "pygranso_torch", {"adaptive_rho": True}),
    ("abl_scaling5", "pygranso_torch", {"scaling": 5}),
    ("abl_polishing", "pygranso_torch", {"polishing": True}),
]
FAIR_OPTIMIZATION_VARIANTS = [
    (
        "fair_fast_fixed1_iter10",
        "pygranso_torch",
        {
            "max_iter": 10,
            "cg_fixed_iters": "1",
            "cg_check_interval": 10,
            "check_termination": 10,
        },
    ),
    (
        "fair_fast_fixed1_iter20",
        "pygranso_torch",
        {
            "max_iter": 20,
            "cg_fixed_iters": "1",
            "cg_check_interval": 20,
            "check_termination": 20,
        },
    ),
    (
        "fair_fast_auto_iter20",
        "pygranso_torch",
        {
            "max_iter": 20,
            "cg_fixed_iters": "auto",
            "cg_check_interval": 20,
            "check_termination": 20,
        },
    ),
    (
        "fair_fast_converged_iter20",
        "pygranso_torch",
        {
            "max_iter": 20,
            "cg_fixed_iters": None,
            "cg_check_interval": 20,
            "check_termination": 20,
        },
    ),
    (
        "fair_fast_adaptive_iter20",
        "pygranso_torch",
        {
            "max_iter": 20,
            "cg_fixed_iters": "auto",
            "cg_check_interval": 20,
            "check_termination": 20,
            "adaptive_rho": True,
        },
    ),
    (
        "fair_fast_scaled_iter20",
        "pygranso_torch",
        {
            "max_iter": 20,
            "cg_fixed_iters": "auto",
            "cg_check_interval": 20,
            "check_termination": 20,
            "scaling": 5,
        },
    ),
    *[
        (
            f"fair_graph_fixed1_iter{iterations}",
            "pygranso_torch",
            {
                "max_iter": iterations,
                "cg_fixed_iters": "1",
                "cg_check_interval": iterations,
                "check_termination": iterations,
                "cuda_graph": True,
            },
        )
        for iterations in (10, 12, 15, 20)
    ],
    *[
        (
            f"fair_graph_fixed2_iter{iterations}",
            "pygranso_torch",
            {
                "max_iter": iterations,
                "cg_fixed_iters": "2",
                "cg_check_interval": iterations,
                "check_termination": iterations,
                "cuda_graph": True,
            },
        )
        for iterations in (10, 12, 15, 20)
    ],
]


def make_sparse_bound_qp(n, device="cpu", dtype=torch.float64):
    """Build a deterministic diagonal bound QP.

    The problem is:
        minimize 0.5 * ||x||^2 - 1^T x
        subject to -1 <= x <= 1

    Its solution is x = 1, and the diagonal Hessian keeps the sparse structure
    easy to inspect.
    """
    device = torch.device(device)
    indices = torch.arange(n, device=device)
    H = torch.sparse_coo_tensor(
        torch.stack((indices, indices)),
        torch.ones(n, device=device, dtype=dtype),
        (n, n),
        device=device,
        dtype=dtype,
    ).coalesce()
    f = -torch.ones((n, 1), device=device, dtype=dtype)
    LB = -torch.ones((n, 1), device=device, dtype=dtype)
    UB = torch.ones((n, 1), device=device, dtype=dtype)
    return H, f, None, None, LB, UB


def make_sparse_equality_bound_qp(n, device="cpu", dtype=torch.float64):
    """Build a sparse diagonal QP with one sparse equality plus bounds."""
    device = torch.device(device)
    diag = torch.arange(n, device=device)
    H = torch.sparse_coo_tensor(
        torch.stack((diag, diag)),
        torch.ones(n, device=device, dtype=dtype),
        (n, n),
        device=device,
        dtype=dtype,
    ).coalesce()
    f = torch.linspace(-0.5, 0.5, n, device=device, dtype=dtype).reshape(n, 1)
    A = torch.sparse_coo_tensor(
        torch.stack((torch.zeros(n, device=device, dtype=torch.long), diag)),
        torch.ones(n, device=device, dtype=dtype),
        (1, n),
        device=device,
        dtype=dtype,
    ).coalesce()
    b = torch.zeros((1, 1), device=device, dtype=dtype)
    LB = -torch.ones((n, 1), device=device, dtype=dtype)
    UB = torch.ones((n, 1), device=device, dtype=dtype)
    return H, f, A, b, LB, UB


def make_random_sparse_spd_qp(
    n,
    device="cpu",
    dtype=torch.float64,
    seed=0,
    density=0.01,
):
    """Build a seeded sparse SPD box QP for reference comparisons."""
    device = torch.device(device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + 1009 * int(n))

    offdiag_nnz = max(n, int(n * n * density / 2))
    rows = torch.randint(0, n, (offdiag_nnz,), generator=generator)
    cols = torch.randint(0, n, (offdiag_nnz,), generator=generator)
    mask = rows != cols
    rows = rows[mask]
    cols = cols[mask]
    values = (torch.rand(rows.numel(), generator=generator, dtype=dtype) - 0.5) * 0.04

    diag = torch.arange(n)
    diag_values = 2.0 + torch.rand(n, generator=generator, dtype=dtype) * 0.5
    all_rows = torch.cat((diag, rows, cols)).to(device=device)
    all_cols = torch.cat((diag, cols, rows)).to(device=device)
    all_values = torch.cat((diag_values, values, values)).to(device=device)

    H = torch.sparse_coo_tensor(
        torch.stack((all_rows, all_cols)),
        all_values,
        (n, n),
        device=device,
        dtype=dtype,
    ).coalesce()
    f = 0.1 * (
        torch.rand((n, 1), generator=generator, dtype=dtype).to(device=device) - 0.5
    )
    LB = -torch.ones((n, 1), device=device, dtype=dtype)
    UB = torch.ones((n, 1), device=device, dtype=dtype)
    return H, f, None, None, LB, UB


def make_active_bound_qp(n, device="cpu", dtype=torch.float64):
    """Build a sparse diagonal QP with predictable active lower/upper bounds."""
    device = torch.device(device)
    indices = torch.arange(n, device=device)
    H = torch.sparse_coo_tensor(
        torch.stack((indices, indices)),
        torch.ones(n, device=device, dtype=dtype),
        (n, n),
        device=device,
        dtype=dtype,
    ).coalesce()
    f = torch.linspace(-2.0, 2.0, n, device=device, dtype=dtype).reshape(n, 1)
    LB = torch.zeros((n, 1), device=device, dtype=dtype)
    UB = torch.ones((n, 1), device=device, dtype=dtype)
    return H, f, None, None, LB, UB


def make_ill_conditioned_sparse_spd_qp(n, device="cpu", dtype=torch.float64):
    """Build a sparse diagonal SPD QP with a wide eigenvalue range."""
    device = torch.device(device)
    indices = torch.arange(n, device=device)
    diag_values = torch.logspace(
        -4.0,
        4.0,
        n,
        device=device,
        dtype=dtype,
    )
    H = torch.sparse_coo_tensor(
        torch.stack((indices, indices)),
        diag_values,
        (n, n),
        device=device,
        dtype=dtype,
    ).coalesce()
    f = 0.01 * torch.sin(
        torch.linspace(0.0, 4.0, n, device=device, dtype=dtype)
    ).reshape(n, 1)
    LB = -torch.ones((n, 1), device=device, dtype=dtype)
    UB = torch.ones((n, 1), device=device, dtype=dtype)
    return H, f, None, None, LB, UB


CASE_BUILDERS = {
    "bound": make_sparse_bound_qp,
    "equality": make_sparse_equality_bound_qp,
    "random_spd": make_random_sparse_spd_qp,
    "active_bound": make_active_bound_qp,
    "ill_conditioned_spd": make_ill_conditioned_sparse_spd_qp,
}


def make_qp_case(case, n, device="cpu", dtype=torch.float64, seed=0, density=0.01):
    if case == "random_spd":
        return make_random_sparse_spd_qp(n, device, dtype, seed, density)
    return CASE_BUILDERS[case](n, device, dtype)


def estimate_dense_kkt_mb(n, dtype=torch.float64, n_eq=0):
    dtype_bytes = torch.empty((), dtype=dtype).element_size()
    kkt_dim = 2 * n + n_eq
    return (kkt_dim * kkt_dim * dtype_bytes) / (1024 * 1024)


def estimate_case_stats(case, n, dtype, args):
    H, _f, A, _b, _LB, _UB = make_qp_case(
        case,
        n,
        device="cpu",
        dtype=dtype,
        seed=args.seed,
        density=args.random_density,
    )
    n_eq = 0 if A is None else A.shape[0]
    return {
        "dense_mb": estimate_dense_kkt_mb(n, dtype, n_eq),
        "sparse_nnz": _torch_nnz(H) + _torch_nnz(A) + n,
    }


def make_parametric_qp_sequence(
    case,
    n,
    args,
    device="cpu",
    dtype=torch.float64,
    count=1,
):
    base = make_qp_case(
        case,
        n,
        device=device,
        dtype=dtype,
        seed=args.seed,
        density=args.random_density,
    )
    H, f, A, b, LB, UB = base
    grid = torch.linspace(0.0, 1.0, n, device=torch.device(device), dtype=dtype).reshape(
        n, 1
    )
    sequence = []
    for step in range(max(int(count), 1)):
        phase = float(step + 1)
        q_delta = args.parametric_delta * torch.sin((phase + 1.0) * 3.14159 * grid)
        f_step = f + q_delta
        H_step = H
        A_step = A
        b_step = b
        if getattr(args, "parametric_matrix_values", False):
            variable_scale = 1.0 + args.parametric_delta * torch.sin(
                phase * 1.61803 + 2.0 * 3.14159 * grid.reshape(-1)
            )
            H_step = _scale_sparse_rows_cols(H, variable_scale, variable_scale)
            if A is not None:
                row_grid = torch.linspace(
                    0.0,
                    1.0,
                    A.shape[0],
                    device=A.device,
                    dtype=dtype,
                )
                row_scale = 1.0 + args.parametric_delta * torch.cos(
                    phase * 0.75488 + 2.0 * 3.14159 * row_grid
                )
                A_step = _scale_sparse_rows_cols(A, row_scale, variable_scale)
                feasible_x = 0.25 * torch.sin(
                    phase * 0.5 + 2.0 * 3.14159 * grid.reshape(-1)
                )
                b_step = _matvec(A_step, feasible_x).reshape(-1, 1)
        elif b is not None:
            rhs_delta = args.parametric_delta * torch.cos(
                torch.tensor(phase, device=b.device, dtype=dtype)
            )
            b_step = b + rhs_delta.reshape(1, 1)
        sequence.append(
            (
                H_step,
                f_step,
                A_step,
                b_step,
                LB,
                UB,
            )
        )
    return sequence


def osqp_problem_matrices(qp):
    H, f, A, b, LB, UB = qp
    q = _torch_vector_to_numpy(f)
    lower = _torch_vector_to_numpy(LB).reshape(-1, 1)
    upper = _torch_vector_to_numpy(UB).reshape(-1, 1)
    nvar = q.size
    P = sparse.triu(_torch_matrix_to_scipy_csc(H), format="csc")
    if A is not None and b is not None:
        Aeq = _torch_matrix_to_scipy_csc(A)
        beq = _torch_vector_to_numpy(b).reshape(-1, 1)
        A_osqp = sparse.vstack([Aeq, sparse.eye(nvar, format="csc")], format="csc")
        l = np.vstack((beq, lower)).reshape(-1)
        u = np.vstack((beq, upper)).reshape(-1)
    else:
        A_osqp = sparse.eye(nvar, format="csc")
        l = lower.reshape(-1)
        u = upper.reshape(-1)
    return P, q, A_osqp, l, u


def benchmark_settings(linear_solver, args):
    return {
        "linear_solver": linear_solver,
        "return_info": True,
        "return_state": bool(args.warm_start),
        "max_iter": args.max_iter,
        "check_termination": args.check_termination,
        "eps_abs": args.eps_abs,
        "eps_rel": args.eps_rel,
        "cg_rtol": args.cg_rtol,
        "cg_atol": 0.0,
        "cg_max_iter": args.cg_max_iter,
        "cg_check_interval": args.cg_check_interval,
        "cg_fixed_iters": _cg_fixed_iters_arg(args.cg_fixed_iters),
        "torch_compile_admm": args.torch_compile_admm,
        "cuda_graph": args.cuda_graph,
        "cuda_event_timing": args.cuda_event_timing,
        "scaling": args.scaling,
        "adaptive_rho": args.adaptive_rho,
        "rho_update_interval": _rho_update_interval_arg(args.rho_update_interval),
        "rho_update_tolerance": args.rho_update_tolerance,
        "warm_start": bool(args.warm_start),
        "polishing": bool(args.polishing),
        "polish_delta": args.polish_delta,
        "polish_refine_iter": args.polish_refine_iter,
        "linear_solver_auto_dense_memory_limit_mb": args.dense_memory_limit_mb,
        "verbose": False,
    }


def builtin_settings(args, reference=False):
    return {
        "eps_abs": min(args.eps_abs, 1e-8) if reference else args.eps_abs,
        "eps_rel": min(args.eps_rel, 1e-8) if reference else args.eps_rel,
        "max_iter": max(args.reference_max_iter, args.max_iter)
        if reference
        else args.max_iter,
        "polishing": False,
        "verbose": False,
    }


def objective_for_qp(solution, qp):
    H, f, _A, _b, _LB, _UB = qp
    x = solution.reshape(-1)
    Hx = _matvec(H, x)
    return float((0.5 * torch.dot(x, Hx) + torch.dot(f.reshape(-1), x)).item())


def qp_constraint_violation(solution, qp):
    _H, _f, A, b, LB, UB = qp
    x = solution.reshape(-1)
    violations = [
        torch.clamp(LB.reshape(-1) - x, min=0.0),
        torch.clamp(x - UB.reshape(-1), min=0.0),
    ]
    if A is not None and b is not None:
        violations.append(torch.abs(_matvec(A, x) - b.reshape(-1)))
    return float(torch.max(torch.cat(violations)).item())


def _relative_gap(value, reference):
    if value is None or reference is None:
        return None
    return abs(float(value) - float(reference)) / max(1.0, abs(float(reference)))


def _timing_iqr(samples):
    if len(samples) < 2:
        return 0.0 if len(samples) == 1 else None
    values = np.asarray(samples, dtype=float)
    return float(np.percentile(values, 75) - np.percentile(values, 25))


def synchronize_if_needed(device):
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def solve_backend_qp(qp, backend, args, initial_state=None):
    H, f, A, b, LB, UB = qp

    if backend == "builtin_cpu":
        result = solve_osqp_torch_qp(
            H,
            f,
            A,
            b,
            LB,
            UB,
            torch.device("cpu"),
            args.dtype == "float64",
            options={"algebra": "builtin", "settings": builtin_settings(args)},
        )
        return result

    if backend == "torch_dense":
        linear_solver = "dense"
    elif backend == "torch_sparse_cg":
        linear_solver = "sparse_cg"
    else:
        linear_solver = "auto"
    settings = benchmark_settings(linear_solver, args)
    if initial_state is not None:
        settings["initial_state"] = initial_state
        settings["warm_start"] = True
        settings["return_state"] = True
    result = solve_osqp_torch_qp(
        H,
        f,
        A,
        b,
        LB,
        UB,
        torch.device(args.device),
        args.dtype == "float64",
        options={
            "algebra": "torch",
            "settings": settings,
        },
    )
    return result


def run_once(n, backend, args, case=None, initial_state=None):
    case = case or args.cases[0]
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = "cpu" if backend == "builtin_cpu" else args.device
    qp = make_qp_case(
        case,
        n,
        device=device,
        dtype=dtype,
        seed=args.seed,
        density=args.random_density,
    )
    result = solve_backend_qp(qp, backend, args, initial_state)
    return result, qp


def solve_pygranso_qp(qp, args):
    H, f, A, b, LB, UB = qp
    result = solveQP(
        H,
        f,
        A,
        b,
        LB,
        UB,
        "osqp",
        torch.device(args.device),
        args.dtype == "float64",
        osqp_options={
            "algebra": "torch",
            "settings": benchmark_settings("auto", args),
        },
    )
    info = getLastOSQPInfo() or {}
    return result, info


def run_pygranso_once(n, args, case=None):
    case = case or args.cases[0]
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    qp = make_qp_case(
        case,
        n,
        device=args.device,
        dtype=dtype,
        seed=args.seed,
        density=args.random_density,
    )
    result, info = solve_pygranso_qp(qp, args)
    return (result, info), qp


def solve_cpu_reference_qp(qp, args):
    H, f, A, b, LB, UB = qp
    return solve_osqp_torch_qp(
        H,
        f,
        A,
        b,
        LB,
        UB,
        torch.device("cpu"),
        args.dtype == "float64",
        options={"algebra": "builtin", "settings": builtin_settings(args, True)},
    )


def solve_cpu_reference(n, case, args, dtype):
    qp = make_qp_case(
        case,
        n,
        device="cpu",
        dtype=dtype,
        seed=args.seed,
        density=args.random_density,
    )
    return solve_cpu_reference_qp(qp, args)


def summarize_result(
    n,
    case,
    backend,
    result,
    qp,
    elapsed_ms,
    stats,
    reference=None,
    timing_samples=None,
):
    if isinstance(result, tuple):
        solution, info = result
    else:
        solution = result
        info = {}

    objective = info.get("objective", objective_for_qp(solution, qp))
    reference_objective = (
        objective_for_qp(reference, _cpu_qp_copy(qp))
        if reference is not None
        else None
    )
    samples = list(timing_samples or [])
    return {
        "case": case,
        "n": n,
        "backend": backend,
        "status": "ok",
        "solver_status": info.get("status", "unknown"),
        "median_ms": elapsed_ms,
        "samples_ms": samples,
        "iqr_ms": _timing_iqr(samples),
        "selected": info.get("linear_solver_selected", "-"),
        "reason": info.get("linear_solver_auto_reason", "-"),
        "objective": objective,
        "relative_objective_gap": _relative_gap(objective, reference_objective),
        "constraint_violation": qp_constraint_violation(solution, qp),
        "prim_res": info.get("primal_residual"),
        "dual_res": info.get("dual_residual"),
        "eps_prim": info.get("eps_primal"),
        "eps_dual": info.get("eps_dual"),
        "cg_iters": info.get("total_cg_iterations", 0),
        "cg_fix": info.get("cg_fixed_iters_selected"),
        "compile": _compile_summary(info),
        "graph": _graph_summary(info),
        "rho_upd": info.get("rho_updates", 0),
        "scale": info.get("scaling_passes", 0),
        "cache": "hit" if info.get("sparse_setup_cache_hit", False) else "-",
        "polish": _polish_summary(info),
        "external": "-",
        "setup_ms": info.get("timing_setup_ms"),
        "update_ms": info.get("timing_update_ms"),
        "solve_ms": info.get("timing_solve_ms"),
        "cg_ms": info.get("timing_cg_ms"),
        "admm_ms": info.get("timing_admm_update_ms"),
        "resid_ms": info.get("timing_residual_ms"),
        "graph_ms": info.get("timing_cuda_graph_replay_ms"),
        "dense_mb": info.get("estimated_dense_kkt_mb", stats["dense_mb"]),
        "sparse_nnz": info.get("estimated_sparse_nnz", stats["sparse_nnz"]),
        "ref_err": reference_error(solution, reference) if reference is not None else None,
    }


def skipped_row(n, case, backend, reason, stats):
    return {
        "case": case,
        "n": n,
        "backend": backend,
        "status": "skipped",
        "solver_status": "skipped",
        "median_ms": None,
        "samples_ms": [],
        "iqr_ms": None,
        "selected": "-",
        "reason": reason,
        "objective": None,
        "relative_objective_gap": None,
        "constraint_violation": None,
        "prim_res": None,
        "dual_res": None,
        "eps_prim": None,
        "eps_dual": None,
        "cg_iters": None,
        "cg_fix": None,
        "compile": "-",
        "graph": "-",
        "rho_upd": None,
        "scale": None,
        "cache": "-",
        "polish": "-",
        "external": "-",
        "setup_ms": None,
        "update_ms": None,
        "solve_ms": None,
        "cg_ms": None,
        "admm_ms": None,
        "resid_ms": None,
        "graph_ms": None,
        "dense_mb": stats["dense_mb"],
        "sparse_nnz": stats["sparse_nnz"],
        "ref_err": None,
    }


def error_row(n, case, backend, error, stats):
    return {
        "case": case,
        "n": n,
        "backend": backend,
        "status": "error",
        "solver_status": "error",
        "median_ms": None,
        "samples_ms": [],
        "iqr_ms": None,
        "selected": "-",
        "reason": f"{type(error).__name__}: {error}",
        "objective": None,
        "relative_objective_gap": None,
        "constraint_violation": None,
        "prim_res": None,
        "dual_res": None,
        "eps_prim": None,
        "eps_dual": None,
        "cg_iters": None,
        "cg_fix": None,
        "compile": "-",
        "graph": "-",
        "rho_upd": None,
        "scale": None,
        "cache": "-",
        "polish": "-",
        "external": "-",
        "setup_ms": None,
        "update_ms": None,
        "solve_ms": None,
        "cg_ms": None,
        "admm_ms": None,
        "resid_ms": None,
        "graph_ms": None,
        "dense_mb": stats["dense_mb"],
        "sparse_nnz": stats["sparse_nnz"],
        "ref_err": None,
    }


def time_backend(n, backend, args, case=None):
    case = case or args.cases[0]
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    stats = estimate_case_stats(case, n, dtype, args)
    if backend == "builtin_cpu" and importlib.util.find_spec("osqp") is None:
        return skipped_row(n, case, backend, "osqp_unavailable", stats)

    if backend == "torch_dense" and stats["dense_mb"] > args.dense_memory_limit_mb:
        return skipped_row(n, case, backend, "dense_kkt_memory_limit", stats)

    try:
        warm_state = None
        for _ in range(args.warmups):
            warm_result, _warm_qp = run_once(
                n, backend, args, case, initial_state=warm_state
            )
            warm_state = _state_from_result(warm_result, warm_state)
            synchronize_if_needed(args.device if backend != "builtin_cpu" else "cpu")

        timings = []
        last_result = None
        last_qp = None
        for _ in range(args.repeats):
            synchronize_if_needed(args.device if backend != "builtin_cpu" else "cpu")
            start = time.perf_counter()
            last_result, last_qp = run_once(
                n, backend, args, case, initial_state=warm_state
            )
            synchronize_if_needed(args.device if backend != "builtin_cpu" else "cpu")
            timings.append((time.perf_counter() - start) * 1000)
            warm_state = _state_from_result(last_result, warm_state)

        reference = None
        if _wants_reference(case, backend, args):
            reference = solve_cpu_reference(n, case, args, dtype)

        return summarize_result(
            n,
            case,
            backend,
            last_result,
            last_qp,
            statistics.median(timings),
            stats,
            reference,
            timings,
        )
    except Exception as exc:
        return error_row(n, case, backend, exc, stats)


def time_pygranso_repeat(n, args, case=None):
    case = case or args.cases[0]
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    stats = estimate_case_stats(case, n, dtype, args)
    backend = "pygranso_torch"
    try:
        resetOSQPWarmState()
        for _ in range(args.warmups):
            run_pygranso_once(n, args, case)
            synchronize_if_needed(args.device)

        timings = []
        last_result = None
        last_qp = None
        for _ in range(args.repeats):
            synchronize_if_needed(args.device)
            start = time.perf_counter()
            last_result, last_qp = run_pygranso_once(n, args, case)
            synchronize_if_needed(args.device)
            timings.append((time.perf_counter() - start) * 1000)

        reference = None
        if _wants_reference(case, backend, args):
            reference = solve_cpu_reference(n, case, args, dtype)

        return summarize_result(
            n,
            case,
            backend,
            last_result,
            last_qp,
            statistics.median(timings),
            stats,
            reference,
            timings,
        )
    except Exception as exc:
        return error_row(n, case, backend, exc, stats)


def time_parametric_backend(n, backend, args, case=None):
    case = case or args.cases[0]
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    stats = estimate_case_stats(case, n, dtype, args)
    if backend == "builtin_cpu" and importlib.util.find_spec("osqp") is None:
        return skipped_row(n, case, backend, "osqp_unavailable", stats)
    if backend == "torch_dense" and stats["dense_mb"] > args.dense_memory_limit_mb:
        return skipped_row(n, case, backend, "dense_kkt_memory_limit", stats)

    try:
        device = "cpu" if backend == "builtin_cpu" else args.device
        sequence = make_parametric_qp_sequence(
            case,
            n,
            args,
            device=device,
            dtype=dtype,
            count=args.warmups + args.repeats,
        )
        warm_state = None
        last_result = None
        last_qp = None
        if backend == "pygranso_torch":
            resetOSQPWarmState()
        for qp in sequence[: args.warmups]:
            if backend == "pygranso_torch":
                result, info = solve_pygranso_qp(qp, args)
                last_result = (result, info)
            else:
                last_result = solve_backend_qp(qp, backend, args, warm_state)
                warm_state = _state_from_result(last_result, warm_state)
            synchronize_if_needed(device)

        timings = []
        for qp in sequence[args.warmups :]:
            synchronize_if_needed(device)
            start = time.perf_counter()
            if backend == "pygranso_torch":
                result, info = solve_pygranso_qp(qp, args)
                last_result = (result, info)
            else:
                last_result = solve_backend_qp(qp, backend, args, warm_state)
                warm_state = _state_from_result(last_result, warm_state)
            synchronize_if_needed(device)
            timings.append((time.perf_counter() - start) * 1000)
            last_qp = qp

        reference = None
        if last_qp is not None and _wants_reference(case, backend, args):
            reference = solve_cpu_reference_qp(_cpu_qp_copy(last_qp), args)

        return summarize_result(
            n,
            case,
            backend,
            last_result,
            last_qp,
            statistics.median(timings),
            stats,
            reference,
            timings,
        )
    except Exception as exc:
        return error_row(n, case, backend, exc, stats)


def time_builtin_update_warm(n, args, case=None):
    case = case or args.cases[0]
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    stats = estimate_case_stats(case, n, dtype, args)
    backend = "builtin_update_warm"
    if importlib.util.find_spec("osqp") is None:
        return skipped_row(n, case, backend, "osqp_unavailable", stats)

    try:
        osqp = __import__("osqp")
        sequence = make_parametric_qp_sequence(
            case,
            n,
            args,
            device="cpu",
            dtype=dtype,
            count=args.warmups + args.repeats + 1,
        )
        P, q, A_osqp, l, u = osqp_problem_matrices(sequence[0])
        prob = osqp.OSQP(algebra="builtin")
        setup_start = time.perf_counter()
        prob.setup(P, q, A_osqp, l, u, **builtin_settings(args))
        setup_ms = (time.perf_counter() - setup_start) * 1000
        last_res = prob.solve()
        rebuilds = 0

        update_times = []
        solve_times = []
        total_times = []
        for step, qp in enumerate(sequence[1:]):
            P_next, q_next, A_next, l_next, u_next = osqp_problem_matrices(qp)
            total_start = time.perf_counter()
            update_start = time.perf_counter()
            same_pattern = _same_csc_pattern(P, P_next) and _same_csc_pattern(
                A_osqp, A_next
            )
            if same_pattern:
                update_values = {"q": q_next, "l": l_next, "u": u_next}
                if not np.array_equal(P.data, P_next.data):
                    update_values["Px"] = P_next.data
                if not np.array_equal(A_osqp.data, A_next.data):
                    update_values["Ax"] = A_next.data
                prob.update(**update_values)
            else:
                prob = osqp.OSQP(algebra="builtin")
                prob.setup(
                    P_next,
                    q_next,
                    A_next,
                    l_next,
                    u_next,
                    **builtin_settings(args),
                )
                rebuilds += 1
            update_ms = (time.perf_counter() - update_start) * 1000
            if getattr(last_res, "x", None) is not None and getattr(last_res, "y", None) is not None:
                prob.warm_start(x=last_res.x, y=last_res.y)
            solve_start = time.perf_counter()
            last_res = prob.solve()
            solve_ms = (time.perf_counter() - solve_start) * 1000
            total_ms = (time.perf_counter() - total_start) * 1000
            if step >= args.warmups:
                update_times.append(update_ms)
                solve_times.append(solve_ms)
                total_times.append(total_ms)
            P = P_next
            A_osqp = A_next

        solution = torch.from_numpy(np.asarray(last_res.x).reshape(-1, 1)).to(
            dtype=dtype
        )
        info = {
            "linear_solver_selected": "builtin_update_warm",
            "linear_solver_auto_reason": "osqp_update_warm",
            "timing_setup_ms": setup_ms,
            "timing_update_ms": statistics.median(update_times),
            "timing_solve_ms": statistics.median(solve_times),
            "workspace_rebuilds": rebuilds,
            "status": str(getattr(last_res.info, "status", "unknown")),
            "primal_residual": float(last_res.info.prim_res),
            "dual_residual": float(last_res.info.dual_res),
            "objective": float(last_res.info.obj_val),
        }
        return summarize_result(
            n,
            case,
            backend,
            (solution, info),
            sequence[-1],
            statistics.median(total_times),
            stats,
            None,
            total_times,
        )
    except Exception as exc:
        return error_row(n, case, backend, exc, stats)


def time_equality_sweep_variant(n, args, label, overrides):
    variant_args = _copy_args_with(args, **overrides)
    row = time_backend(n, "torch_sparse_cg", variant_args, "equality")
    row["backend"] = label
    return row


def time_ablation_variant(n, args, label, backend, overrides):
    variant_args = _copy_args_with(args, **overrides)
    if args.parametric_sequence:
        row = time_parametric_backend(n, backend, variant_args, "random_spd")
    elif backend == "pygranso_torch":
        row = time_pygranso_repeat(n, variant_args, "random_spd")
    else:
        row = time_backend(n, backend, variant_args, "random_spd")
    row["backend"] = label
    return row


def time_fair_optimization_variant(n, args, label, backend, overrides):
    variant_args = _copy_args_with(args, **overrides)
    row = time_parametric_backend(n, backend, variant_args, "random_spd")
    row["backend"] = label
    return row


def time_external_solver(n, solver, args, case=None):
    case = case or args.cases[0]
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    stats = estimate_case_stats(case, n, dtype, args)
    availability = external_solver_availability(solver)
    backend = f"external:{solver}"
    if not availability["available"]:
        return skipped_row(n, case, backend, availability["reason"], stats)

    try:
        timings = []
        last_solution = None
        last_qp = None
        for _ in range(args.warmups):
            run_external_once(n, solver, args, case, availability["module"])
            synchronize_if_needed(args.device)

        for _ in range(args.repeats):
            synchronize_if_needed(args.device)
            start = time.perf_counter()
            last_solution, last_qp = run_external_once(
                n, solver, args, case, availability["module"]
            )
            synchronize_if_needed(args.device)
            timings.append((time.perf_counter() - start) * 1000)

        return summarize_external_result(
            n,
            case,
            backend,
            solver,
            last_solution,
            last_qp,
            statistics.median(timings),
            stats,
            timings,
        )
    except Exception as exc:
        return error_row(n, case, backend, exc, stats)


def run_external_once(n, solver, args, case, torch_sla_module):
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    qp = make_qp_case(
        case,
        n,
        device=args.device,
        dtype=dtype,
        seed=args.seed,
        density=args.random_density,
    )
    H, f, _A, _b, _LB, _UB = qp
    rhs = -f.reshape(-1)
    solution = solve_linear_system_with_torch_sla(
        H, rhs, solver, torch_sla_module, args
    )
    return solution.reshape(-1, 1), qp


def summarize_external_result(
    n, case, backend, solver, solution, qp, elapsed_ms, stats, timing_samples=None
):
    residual = linear_residual(qp[0], solution.reshape(-1), -qp[1].reshape(-1))
    samples = list(timing_samples or [])
    return {
        "case": case,
        "n": n,
        "backend": backend,
        "status": "ok",
        "solver_status": "linear_system_only",
        "median_ms": elapsed_ms,
        "samples_ms": samples,
        "iqr_ms": _timing_iqr(samples),
        "selected": "linear",
        "reason": "torch_sla_optional_reference",
        "objective": objective_for_qp(solution, qp),
        "relative_objective_gap": None,
        "constraint_violation": None,
        "prim_res": residual,
        "dual_res": None,
        "eps_prim": None,
        "eps_dual": None,
        "cg_iters": None,
        "cg_fix": None,
        "compile": "-",
        "graph": "-",
        "rho_upd": None,
        "scale": None,
        "cache": "-",
        "polish": "-",
        "external": solver,
        "setup_ms": None,
        "update_ms": None,
        "solve_ms": None,
        "cg_ms": None,
        "admm_ms": None,
        "resid_ms": None,
        "graph_ms": None,
        "dense_mb": stats["dense_mb"],
        "sparse_nnz": stats["sparse_nnz"],
        "ref_err": None,
    }


def external_solver_availability(solver):
    if solver not in EXTERNAL_SOLVERS:
        return {"available": False, "reason": f"unknown_external_solver:{solver}"}
    if importlib.util.find_spec("torch_sla") is None:
        return {"available": False, "reason": "torch_sla_unavailable"}
    try:
        module = __import__("torch_sla")
    except Exception as exc:
        return {
            "available": False,
            "reason": f"torch_sla_import_failed:{type(exc).__name__}",
        }
    if solver == "torch_sla_cudss" and importlib.util.find_spec("cupy") is None:
        return {"available": False, "reason": "cupy_unavailable_for_cudss"}
    return {"available": True, "reason": "available", "module": module}


def solve_linear_system_with_torch_sla(matrix, rhs, solver, torch_sla_module, args):
    backend = "pytorch_cg" if solver == "torch_sla_pytorch_cg" else "cudss"
    for name in ("solve", "spsolve"):
        candidate = getattr(torch_sla_module, name, None)
        if callable(candidate):
            try:
                return candidate(matrix, rhs, solver=backend)
            except TypeError:
                try:
                    return candidate(matrix, rhs, backend=backend)
                except TypeError:
                    return candidate(matrix, rhs)
    raise RuntimeError(
        "torch_sla is installed, but no supported solve/spsolve API was found."
    )


def linear_residual(matrix, solution, rhs):
    return float(
        torch.linalg.vector_norm(_matvec(matrix, solution) - rhs, ord=float("inf")).item()
    )


def reference_error(solution, reference):
    diff = solution.detach().cpu().reshape(-1) - reference.detach().cpu().reshape(-1)
    return float(torch.linalg.vector_norm(diff, ord=float("inf")).item())


def format_value(value):
    if value is None:
        return "-"
    if isinstance(value, float):
        if abs(value) >= 1e4 or (value != 0 and abs(value) < 1e-3):
            return f"{value:.2e}"
        return f"{value:.3f}"
    return str(value)


def print_table(rows):
    header = " ".join(name.ljust(width) for name, width in TABLE_COLUMNS)
    print(header)
    print("-" * len(header))
    for row in rows:
        values = []
        for name, width in TABLE_COLUMNS:
            text = format_value(row[name])
            if len(text) > width:
                text = text[: width - 1] + "."
            values.append(text.ljust(width))
        print(" ".join(values))


def academic_rows(rows, args):
    baselines = cpu_baseline_times(rows)
    return [academic_row(row, args, baselines) for row in rows]


def print_academic_table(rows, args):
    academic = academic_rows(rows, args)
    print()
    print("Academic validation table")
    print(markdown_table(academic, ACADEMIC_COLUMNS))


def print_fair_summary(rows, args):
    if not args.fair_optimization_suite:
        return
    print()
    print("Best fair CUDA row")
    print(fair_summary_text(best_fair_cuda_row(rows, args)))


def markdown_table(rows, columns):
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append(
            "| " + " | ".join(format_value(row[column]) for column in columns) + " |"
        )
    return "\n".join(lines)


def academic_row(row, args, baselines=None):
    baselines = baselines or {}
    fresh_speedup = speedup_vs_cpu(row, baselines.get("fresh", {}))
    warm_speedup = speedup_vs_cpu(row, baselines.get("warm", {}))
    warm_ci = bootstrap_speedup_interval(
        baselines.get("warm_samples", {}).get((row["case"], row["n"])),
        row.get("samples_ms"),
    )
    return {
        "method": method_label(row["backend"]),
        "case": row["case"],
        "size": row["n"],
        "device": academic_device(row, args),
        "median_ms": row["median_ms"],
        "iqr_ms": row.get("iqr_ms"),
        "speedup_vs_cpu_osqp": fresh_speedup,
        "speedup_vs_cpu_fresh": fresh_speedup,
        "speedup_vs_cpu_warm": warm_speedup,
        "speedup_ci_low": None if warm_ci is None else warm_ci[0],
        "speedup_ci_high": None if warm_ci is None else warm_ci[1],
        "efficiency_status": efficiency_status(
            row, fresh_speedup, warm_speedup, warm_ci
        ),
        "selected_policy": selected_policy(row),
        "cache_hit": row["cache"],
        "cuda_graph": row.get("graph", "-"),
        "primal_residual": row["prim_res"],
        "dual_residual": row["dual_res"],
        "eps_primal": row.get("eps_prim"),
        "eps_dual": row.get("eps_dual"),
        "relative_objective_gap": row.get("relative_objective_gap"),
        "ref_err": row["ref_err"],
        "cg_iterations": row["cg_iters"],
        "setup_ms": row.get("setup_ms"),
        "update_ms": row.get("update_ms"),
        "solve_ms": row.get("solve_ms"),
        "graph_replay_ms": row.get("graph_ms"),
    }


def method_label(backend):
    labels = {
        "builtin_cpu": "CPU OSQP fresh",
        "builtin_update_warm": "CPU OSQP update/warm",
        "torch_dense": "Torch dense",
        "torch_auto": "Torch cold sparse-CG",
        "torch_sparse_cg": "Torch sparse-CG",
        "pygranso_torch": "Torch warm/cache sparse-CG",
        "abl_cold_auto": "Ablation cold auto",
        "abl_warm_cache": "Ablation warm cache",
        "abl_fixed_auto": "Ablation fixed CG auto",
        "abl_fixed1": "Ablation fixed CG 1",
        "abl_fixed2": "Ablation fixed CG 2",
        "abl_fixed3": "Ablation fixed CG 3",
        "abl_fixed5": "Ablation fixed CG 5",
        "abl_adaptive_rho": "Ablation adaptive rho",
        "abl_scaling5": "Ablation Ruiz scaling",
        "abl_polishing": "Ablation polishing",
        "fair_fast_fixed1_iter10": "Fair fast fixed CG 1 iter10",
        "fair_fast_fixed1_iter20": "Fair fast fixed CG 1 iter20",
        "fair_fast_auto_iter20": "Fair fast auto CG iter20",
        "fair_fast_converged_iter20": "Fair fast converged CG iter20",
        "fair_fast_adaptive_iter20": "Fair fast adaptive rho iter20",
        "fair_fast_scaled_iter20": "Fair fast Ruiz scaling iter20",
        "eq_converged": "Equality converged CG",
        "eq_fixed2": "Equality fixed CG 2",
        "eq_fixed5": "Equality fixed CG 5",
        "eq_fixed10": "Equality fixed CG 10",
        "eq_adaptive": "Equality adaptive rho",
        "eq_scaling5": "Equality Ruiz scaling",
    }
    if backend.startswith("external:"):
        return backend.replace("external:", "External ")
    return labels.get(backend, backend)


def academic_device(row, args):
    if row["backend"] in {"builtin_cpu", "builtin_update_warm"}:
        return "cpu"
    return args.device


def selected_policy(row):
    parts = [str(row["selected"])]
    if row["cg_fix"] not in {None, "-"}:
        parts.append(f"cg_fixed={row['cg_fix']}")
    if row["rho_upd"] not in {None, 0, "-"}:
        parts.append(f"rho_updates={row['rho_upd']}")
    if row["scale"] not in {None, 0, "-"}:
        parts.append(f"scaling={row['scale']}")
    if row["polish"] != "-":
        parts.append(f"polish={row['polish']}")
    return ", ".join(parts)


def cpu_baseline_times(rows):
    baselines = {"fresh": {}, "warm": {}, "fresh_samples": {}, "warm_samples": {}}
    for row in rows:
        if (
            row["backend"] == "builtin_cpu"
            and row["status"] == "ok"
            and row["median_ms"] is not None
        ):
            baselines["fresh"][(row["case"], row["n"])] = row["median_ms"]
            baselines["fresh_samples"][(row["case"], row["n"])] = row.get(
                "samples_ms", []
            )
        if (
            row["backend"] == "builtin_update_warm"
            and row["status"] == "ok"
            and row["median_ms"] is not None
        ):
            baselines["warm"][(row["case"], row["n"])] = row["median_ms"]
            baselines["warm_samples"][(row["case"], row["n"])] = row.get(
                "samples_ms", []
            )
    return baselines


def speedup_vs_cpu(row, baselines):
    baseline = baselines.get((row["case"], row["n"]))
    elapsed = row["median_ms"]
    if baseline is None or elapsed in {None, 0}:
        return None
    return baseline / elapsed


def bootstrap_speedup_interval(baseline_samples, candidate_samples, draws=2000):
    if not baseline_samples or not candidate_samples:
        return None
    if len(baseline_samples) < 2 or len(candidate_samples) < 2:
        return None
    baseline = np.asarray(baseline_samples, dtype=float)
    candidate = np.asarray(candidate_samples, dtype=float)
    generator = np.random.default_rng(0)
    ratios = np.empty(int(draws), dtype=float)
    for index in range(int(draws)):
        baseline_draw = generator.choice(baseline, baseline.size, replace=True)
        candidate_draw = generator.choice(candidate, candidate.size, replace=True)
        ratios[index] = np.median(baseline_draw) / np.median(candidate_draw)
    return tuple(float(value) for value in np.percentile(ratios, [2.5, 97.5]))


def row_accuracy_passes(row):
    if row.get("status", "ok") != "ok":
        return False
    checked = False
    objective_gap = row.get("relative_objective_gap")
    if objective_gap is not None:
        checked = True
        if objective_gap > 1e-5:
            return False
    for residual_name, tolerance_name in (
        ("prim_res", "eps_prim"),
        ("dual_res", "eps_dual"),
    ):
        residual = row.get(residual_name)
        tolerance = row.get(tolerance_name)
        if residual is not None and tolerance is not None:
            checked = True
            if residual > tolerance:
                return False
    if checked:
        return True
    ref_err = row.get("ref_err")
    return ref_err is not None and ref_err <= 1e-5


def efficiency_status(row, fresh_speedup, warm_speedup=None, warm_ci=None):
    status = row.get("status", "ok")
    if status != "ok":
        return status
    if row.get("backend") == "builtin_cpu":
        return "baseline_fresh"
    if row.get("backend") == "builtin_update_warm":
        return "baseline_warm"
    if not row_accuracy_passes(row):
        return "reject_accuracy"
    if fresh_speedup is None:
        return "no_cpu_baseline"
    if fresh_speedup <= 1.0:
        return "loss_cpu_fresh"
    if warm_speedup is not None and warm_speedup <= 1.0:
        return "loss_cpu_warm"
    if warm_ci is not None and warm_ci[0] <= 1.0:
        return "inconclusive_ci"
    return "win"


def best_fair_cuda_row(rows, args):
    paired_rows = list(zip(rows, academic_rows(rows, args)))
    candidates = []
    for raw, academic in paired_rows:
        if raw.get("backend") in {"builtin_cpu", "builtin_update_warm", "torch_dense"}:
            continue
        if raw.get("status") != "ok":
            continue
        if academic.get("speedup_vs_cpu_warm") is None:
            continue
        if not row_accuracy_passes(raw):
            continue
        candidates.append((raw, academic))

    winners = [
        (raw, academic)
        for raw, academic in candidates
        if academic["speedup_vs_cpu_warm"] > 1.0
        and (
            academic.get("speedup_ci_low") is None
            or academic["speedup_ci_low"] > 1.0
        )
    ]
    if winners:
        _raw, academic = max(
            winners, key=lambda item: item[1]["speedup_vs_cpu_warm"]
        )
        return _fair_summary("win", academic)

    if candidates:
        _raw, academic = max(
            candidates, key=lambda item: item[1]["speedup_vs_cpu_warm"]
        )
        status = (
            "inconclusive"
            if academic["speedup_vs_cpu_warm"] > 1.0
            and academic.get("speedup_ci_low") is not None
            and academic["speedup_ci_low"] <= 1.0
            else "no_win"
        )
        return _fair_summary(status, academic)

    return {
        "status": "no_valid_cuda_rows",
        "method": "-",
        "case": "-",
        "size": "-",
        "median_ms": None,
        "speedup_vs_cpu_warm": None,
        "needed_speedup_to_match_cpu_warm": None,
        "ref_err": None,
        "efficiency_status": "no_valid_cuda_rows",
        "selected_policy": "-",
    }


def _fair_summary(status, academic):
    warm_speedup = academic["speedup_vs_cpu_warm"]
    needed = None
    if warm_speedup is not None and warm_speedup > 0 and warm_speedup <= 1.0:
        needed = 1.0 / warm_speedup
    return {
        "status": status,
        "method": academic["method"],
        "case": academic["case"],
        "size": academic["size"],
        "median_ms": academic["median_ms"],
        "speedup_vs_cpu_warm": warm_speedup,
        "speedup_ci_low": academic.get("speedup_ci_low"),
        "speedup_ci_high": academic.get("speedup_ci_high"),
        "needed_speedup_to_match_cpu_warm": needed,
        "ref_err": academic["ref_err"],
        "efficiency_status": academic["efficiency_status"],
        "selected_policy": academic["selected_policy"],
    }


def fair_summary_text(summary):
    if summary["status"] == "win":
        return (
            f"best_fair_cuda_row: method={summary['method']}, "
            f"case={summary['case']}, size={summary['size']}, "
            f"median_ms={format_value(summary['median_ms'])}, "
            f"speedup_vs_cpu_warm={format_value(summary['speedup_vs_cpu_warm'])}, "
            f"ref_err={format_value(summary['ref_err'])}, "
            f"policy={summary['selected_policy']}"
        )
    if summary["status"] == "no_win":
        return (
            f"best_fair_cuda_row: none. closest_cuda_row={summary['method']}, "
            f"case={summary['case']}, size={summary['size']}, "
            f"speedup_vs_cpu_warm={format_value(summary['speedup_vs_cpu_warm'])}, "
            f"needed_speedup_to_match_cpu_warm="
            f"{format_value(summary['needed_speedup_to_match_cpu_warm'])}, "
            f"ref_err={format_value(summary['ref_err'])}, "
            f"policy={summary['selected_policy']}"
        )
    if summary["status"] == "inconclusive":
        return (
            f"best_fair_cuda_row: inconclusive. closest_cuda_row={summary['method']}, "
            f"case={summary['case']}, size={summary['size']}, "
            f"speedup_vs_cpu_warm={format_value(summary['speedup_vs_cpu_warm'])}, "
            f"speedup_ci=[{format_value(summary['speedup_ci_low'])}, "
            f"{format_value(summary['speedup_ci_high'])}], "
            f"ref_err={format_value(summary['ref_err'])}, "
            f"policy={summary['selected_policy']}"
        )
    return "best_fair_cuda_row: none. No CUDA row had reference error and CPU warm speedup telemetry."


def export_academic_artifacts(rows, args):
    academic = academic_rows(rows, args)
    if args.export_academic_md:
        path = Path(args.export_academic_md)
        path.parent.mkdir(parents=True, exist_ok=True)
        text = markdown_table(academic, ACADEMIC_COLUMNS) + "\n"
        if args.fair_optimization_suite:
            text += "\n" + fair_summary_text(best_fair_cuda_row(rows, args)) + "\n"
        path.write_text(text, encoding="utf-8")
    if args.export_academic_csv:
        path = Path(args.export_academic_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=ACADEMIC_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(academic)
        path.write_text(buffer.getvalue(), encoding="utf-8")
        if args.fair_optimization_suite:
            summary_path = path.with_name(
                f"{path.stem}_best_fair_cuda_row{path.suffix}"
            )
            summary = best_fair_cuda_row(rows, args)
            summary_buffer = io.StringIO()
            writer = csv.DictWriter(
                summary_buffer,
                fieldnames=list(summary),
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerow(summary)
            summary_path.write_text(summary_buffer.getvalue(), encoding="utf-8")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Compare OSQP adapter runtimes.")
    parser.add_argument("--cases", nargs="+", choices=CASE_BUILDERS, default=DEFAULT_CASES)
    parser.add_argument("--sizes", nargs="+", type=int, default=DEFAULT_SIZES)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--dtype", choices=["float64", "float32"], default="float64")
    parser.add_argument("--max-iter", type=int, default=10)
    parser.add_argument("--check-termination", type=int, default=10)
    parser.add_argument("--eps-abs", type=float, default=1e-5)
    parser.add_argument("--eps-rel", type=float, default=1e-5)
    parser.add_argument("--cg-rtol", type=float, default=1e-5)
    parser.add_argument("--cg-max-iter", type=int, default=100)
    parser.add_argument("--cg-check-interval", type=int, default=1)
    parser.add_argument("--cg-fixed-iters", default=None)
    parser.add_argument("--torch-compile-admm", action="store_true")
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--cuda-event-timing", action="store_true")
    parser.add_argument("--scaling", type=int, default=0)
    parser.add_argument("--adaptive-rho", action="store_true")
    parser.add_argument("--rho-update-interval", default="auto")
    parser.add_argument("--rho-update-tolerance", type=float, default=5.0)
    parser.add_argument("--warm-start", action="store_true")
    parser.add_argument("--polishing", action="store_true")
    parser.add_argument("--polish-delta", type=float, default=1e-6)
    parser.add_argument("--polish-refine-iter", type=int, default=3)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--academic-table",
        action="store_true",
        help="Print a Markdown table with final-report validation columns.",
    )
    parser.add_argument(
        "--include-pygranso-repeat",
        action="store_true",
        help="Add repeated solveQP/PyGRANSO-level Torch OSQP timing rows.",
    )
    parser.add_argument(
        "--include-cpu-warm",
        action="store_true",
        help="Add a CPU OSQP update/warm baseline for same-sparsity QP sequences.",
    )
    parser.add_argument(
        "--parametric-sequence",
        action="store_true",
        help="Benchmark same-sparsity QP sequences with changing vectors.",
    )
    parser.add_argument(
        "--parametric-delta",
        type=float,
        default=1e-2,
        help="Perturbation size for same-sparsity parametric QP sequences.",
    )
    parser.add_argument(
        "--parametric-matrix-values",
        action="store_true",
        help="Also change P/A values while preserving their sparse index patterns.",
    )
    parser.add_argument(
        "--equality-sweep",
        action="store_true",
        help="Add equality-case policy sweep rows for CG/rho/scaling decisions.",
    )
    parser.add_argument(
        "--cuda-win-suite",
        action="store_true",
        help="Use the large random_spd CUDA-win benchmark preset.",
    )
    parser.add_argument(
        "--fair-optimization-suite",
        action="store_true",
        help="Run low-sync CUDA candidates against CPU OSQP update/warm.",
    )
    parser.add_argument(
        "--ablation-suite",
        action="store_true",
        help="Add random_spd rows for warm cache, fixed CG, rho, scaling, and polishing.",
    )
    parser.add_argument(
        "--max-safe-size",
        type=int,
        default=None,
        help="Optional extra size appended to the cuda-win suite.",
    )
    parser.add_argument(
        "--export-academic-md",
        default=None,
        help="Write the academic validation table to a Markdown file.",
    )
    parser.add_argument(
        "--export-academic-csv",
        default=None,
        help="Write the academic validation table to a CSV file.",
    )
    parser.add_argument("--dense-memory-limit-mb", type=float, default=32.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--random-density", type=float, default=0.01)
    parser.add_argument("--reference-max-iter", type=int, default=4000)
    parser.add_argument(
        "--external-solver",
        action="append",
        choices=EXTERNAL_SOLVERS,
        default=[],
        help="Optional external sparse-linear-solver benchmark reference.",
    )
    parser.add_argument(
        "--reference",
        choices=["auto", "always", "off"],
        default="auto",
        help="Compare Torch rows to CPU OSQP; auto compares random_spd only.",
    )
    args = parser.parse_args(argv)
    apply_benchmark_preset(args)
    return args


def apply_benchmark_preset(args):
    if args.cuda_win_suite:
        args.cases = ["random_spd"]
        sizes = [1200, 1600, 2000]
        if args.max_safe_size is not None and args.max_safe_size not in sizes:
            sizes.append(args.max_safe_size)
        args.sizes = sorted(sizes)
        args.repeats = 5
        args.warmups = 2
        args.max_iter = 200
        args.check_termination = 10
        args.reference = "auto"
        args.academic_table = True
        args.include_pygranso_repeat = True
        args.include_cpu_warm = True
        args.parametric_sequence = True
    if args.fair_optimization_suite:
        args.cases = ["random_spd"]
        if args.sizes == DEFAULT_SIZES:
            args.sizes = [1200]
        args.reference = "auto"
        args.academic_table = True
        args.include_pygranso_repeat = True
        args.include_cpu_warm = True
        args.parametric_sequence = True


def main(argv=None):
    args = parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but torch.cuda.is_available() is False.")

    warnings.filterwarnings("ignore", message='"polish" is deprecated')
    warnings.filterwarnings("ignore", message="The default value of raise_error")
    warnings.filterwarnings(
        "ignore", message="Sparse invariant checks are implicitly disabled"
    )
    warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta")

    rows = []
    for case in args.cases:
        for n in args.sizes:
            if args.parametric_sequence:
                for backend in ("builtin_cpu", "torch_dense", "torch_auto"):
                    rows.append(time_parametric_backend(n, backend, args, case))
                if args.include_cpu_warm:
                    rows.append(time_builtin_update_warm(n, args, case))
                if args.include_pygranso_repeat:
                    rows.append(time_parametric_backend(n, "pygranso_torch", args, case))
            else:
                for backend in ("builtin_cpu", "torch_dense", "torch_auto"):
                    rows.append(time_backend(n, backend, args, case))
                if args.include_cpu_warm:
                    rows.append(time_builtin_update_warm(n, args, case))
                if args.include_pygranso_repeat:
                    rows.append(time_pygranso_repeat(n, args, case))
            for solver in args.external_solver:
                rows.append(time_external_solver(n, solver, args, case))
    if args.equality_sweep:
        for n in args.sizes:
            for label, overrides in EQUALITY_SWEEP_VARIANTS:
                rows.append(time_equality_sweep_variant(n, args, label, overrides))
    if args.ablation_suite:
        for n in args.sizes:
            for label, backend, overrides in ABLATION_VARIANTS:
                rows.append(time_ablation_variant(n, args, label, backend, overrides))
    if args.fair_optimization_suite:
        for n in args.sizes:
            for label, backend, overrides in FAIR_OPTIMIZATION_VARIANTS:
                rows.append(
                    time_fair_optimization_variant(n, args, label, backend, overrides)
                )
    print_table(rows)
    if args.academic_table:
        print_academic_table(rows, args)
    print_fair_summary(rows, args)
    export_academic_artifacts(rows, args)
    if args.profile:
        profile_first_torch_row(args)


def _matvec(matrix, vector):
    if matrix.layout == torch.strided:
        return matrix @ vector
    return torch.sparse.mm(matrix, vector.reshape(-1, 1)).reshape(-1)


def _scale_sparse_rows_cols(matrix, row_scale, col_scale):
    if matrix.layout == torch.strided:
        return row_scale.reshape(-1, 1) * matrix * col_scale.reshape(1, -1)
    if matrix.layout == torch.sparse_csr:
        rows = torch.repeat_interleave(
            torch.arange(matrix.shape[0], device=matrix.device),
            matrix.crow_indices()[1:] - matrix.crow_indices()[:-1],
        )
        values = matrix.values() * row_scale[rows] * col_scale[matrix.col_indices()]
        return torch.sparse_csr_tensor(
            matrix.crow_indices(),
            matrix.col_indices(),
            values,
            size=tuple(matrix.shape),
            device=matrix.device,
            dtype=matrix.dtype,
            check_invariants=False,
        )
    coalesced = matrix.coalesce()
    indices = coalesced.indices()
    values = (
        coalesced.values()
        * row_scale[indices[0]]
        * col_scale[indices[1]]
    )
    return torch.sparse_coo_tensor(
        indices,
        values,
        size=tuple(matrix.shape),
        device=matrix.device,
        dtype=matrix.dtype,
        check_invariants=False,
    ).coalesce()


def _same_csc_pattern(left, right):
    return (
        left.shape == right.shape
        and np.array_equal(left.indptr, right.indptr)
        and np.array_equal(left.indices, right.indices)
    )


def _torch_vector_to_numpy(tensor):
    return tensor.detach().cpu().numpy().reshape(-1)


def _torch_matrix_to_scipy_csc(tensor):
    if tensor.layout == torch.strided:
        return sparse.csc_matrix(tensor.detach().cpu().numpy())
    if tensor.layout == torch.sparse_csr:
        cpu = tensor.detach().cpu()
        return sparse.csr_matrix(
            (
                cpu.values().numpy(),
                cpu.col_indices().numpy(),
                cpu.crow_indices().numpy(),
            ),
            shape=tuple(cpu.shape),
        ).tocsc()
    coalesced = tensor.detach().cpu().coalesce()
    indices = coalesced.indices().numpy()
    values = coalesced.values().numpy()
    return sparse.coo_matrix(
        (values, (indices[0], indices[1])),
        shape=tuple(coalesced.shape),
    ).tocsc()


def _cpu_qp_copy(qp):
    return tuple(None if value is None else value.detach().cpu() for value in qp)


def _torch_nnz(tensor):
    if tensor is None:
        return 0
    if tensor.layout == torch.strided:
        return int(torch.count_nonzero(tensor).item())
    return int(tensor._nnz())


def _wants_reference(case, backend, args):
    if backend == "builtin_cpu" or importlib.util.find_spec("osqp") is None:
        return False
    return args.reference == "always" or (
        args.reference == "auto" and case == "random_spd"
    )


def _state_from_result(result, fallback=None):
    if not isinstance(result, tuple):
        return fallback
    _solution, info = result
    return info.get("state", fallback)


def _polish_summary(info):
    if not info.get("polishing", False):
        return "-"
    return "ok" if info.get("polishing_success", False) else info.get(
        "polishing_status", "no"
    )


def _compile_summary(info):
    if not info.get("torch_compile_admm", False):
        return "-"
    status = info.get("torch_compile_admm_status", "unknown")
    if status == "enabled":
        return "on"
    if status == "disabled":
        return "-"
    return status


def _graph_summary(info):
    if not info.get("cuda_graph", False):
        return "-"
    status = info.get("cuda_graph_status", "unknown")
    if info.get("cuda_graph_cache_hit", False):
        return "hit"
    return status


def _rho_update_interval_arg(value):
    if value == "auto":
        return value
    return int(value)


def _cg_fixed_iters_arg(value):
    if value in {None, "auto"}:
        return value
    return int(value)


def _copy_args_with(args, **overrides):
    values = copy.copy(vars(args))
    values.update(overrides)
    return argparse.Namespace(**values)


def profile_first_torch_row(args):
    profile_args = _copy_args_with(args, warm_start=True)
    case = profile_args.cases[0]
    n = profile_args.sizes[0]
    activities = [torch.profiler.ProfilerActivity.CPU]
    if profile_args.device == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    try:
        warm_state = None
        if profile_args.cuda_graph:
            warm_result, _warm_qp = run_once(n, "torch_auto", profile_args, case)
            warm_state = _state_from_result(warm_result)
            synchronize_if_needed(profile_args.device)
        with torch.profiler.profile(activities=activities, record_shapes=True) as prof:
            run_once(n, "torch_auto", profile_args, case, initial_state=warm_state)
            synchronize_if_needed(profile_args.device)
        sort_by = (
            "self_cuda_time_total"
            if profile_args.device == "cuda"
            else "self_cpu_time_total"
        )
        print()
        print(f"Profiler: case={case}, n={n}, backend=torch_auto")
        print(prof.key_averages().table(sort_by=sort_by, row_limit=15))
        print_profiler_summary(prof)
    except Exception as exc:
        print(f"Profiler unavailable: {type(exc).__name__}: {exc}")


def print_profiler_summary(prof):
    events = prof.key_averages()
    self_cpu_ms = sum(event.self_cpu_time_total for event in events) / 1000.0
    device_events = [event for event in events if event.key.startswith("aten::")]
    cuda_total = sum(
        float(getattr(event, "self_cuda_time_total", 0.0))
        for event in device_events
    )
    device_total = sum(
        float(getattr(event, "self_device_time_total", 0.0))
        for event in device_events
    )
    self_cuda_ms = (cuda_total if cuda_total > 0 else device_total) / 1000.0
    sparse_calls = sum(
        event.count
        for event in events
        if "cusparse" in event.key.lower() or "sparse" in event.key.lower()
    )
    vector_ops = {"aten::add", "aten::sub", "aten::mul", "aten::div", "aten::copy_"}
    vector_calls = sum(event.count for event in events if event.key in vector_ops)
    print(
        "Profiler summary: "
        f"self_cpu_ms={self_cpu_ms:.3f}, "
        f"self_cuda_ms={self_cuda_ms:.3f}, "
        f"sparse_calls={sparse_calls}, "
        f"vector_calls={vector_calls}"
    )


if __name__ == "__main__":
    main()
