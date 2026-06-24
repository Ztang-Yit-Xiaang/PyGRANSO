import importlib
import warnings
from numbers import Integral, Number

import numpy as np
import torch
from scipy import sparse

from pygranso.private.torchOSQP import solve_torch_osqp, solve_torch_osqp_from_qp

DEFAULT_OSQP_SETTINGS = {
    "eps_abs": 1e-12,
    "eps_rel": 1e-12,
    "polish": True,
    "verbose": False,
}

DEFAULT_TORCH_OSQP_SETTINGS = {
    "linear_solver": "auto",
    "rho": 0.1,
    "sigma": 1e-6,
    "alpha": 1.6,
    "max_iter": 4000,
    "eps_abs": 1e-12,
    "eps_rel": 1e-12,
    "check_termination": 25,
    "cg_rtol": 1e-6,
    "cg_atol": 0.0,
    "cg_max_iter": 100,
    "cg_check_interval": 1,
    "cg_fixed_iters": None,
    "torch_compile_admm": False,
    "cuda_graph": False,
    "cuda_event_timing": False,
    "scaling": 0,
    "adaptive_rho": False,
    "rho_update_interval": "auto",
    "rho_update_tolerance": 5.0,
    "warm_start": False,
    "initial_state": None,
    "return_state": False,
    "polishing": False,
    "polish_delta": 1e-6,
    "polish_refine_iter": 3,
    "linear_solver_auto_min_kkt_dim": 512,
    "linear_solver_auto_sparse_min_kkt_dim": 1024,
    "linear_solver_auto_max_density": 0.10,
    "linear_solver_auto_dense_memory_limit_mb": 256,
    "return_info": False,
    "verbose": False,
}

SUPPORTED_TORCH_SETTINGS = set(DEFAULT_TORCH_OSQP_SETTINGS)
TORCH_ONLY_SETTINGS = set(DEFAULT_TORCH_OSQP_SETTINGS) - {
    "eps_abs",
    "eps_rel",
    "max_iter",
    "polishing",
    "verbose",
}
UNSUPPORTED_TORCH_SETTINGS = set()
_BUILTIN_OSQP_WORKSPACE = None
_BUILTIN_OSQP_WORKSPACE_STATS = None


class OSQPCudaInteropUnavailableError(RuntimeError):
    """Raised when a real CUDA OSQP interop path was requested but is unavailable."""


def solve_osqp_torch_qp(
    H,
    f,
    A,
    b,
    LB,
    UB,
    torch_device,
    double_precision,
    options=None,
):
    """Solve PyGRANSO's quadprog-style QP with OSQP and return a Torch column.

    PyGRANSO builds QPs as Torch tensors.  The CPU path delegates to the OSQP
    Python package.  The Torch path is a PyGRANSO prototype that can choose a
    dense solve or sparse-CG solve without going through NumPy, SciPy, or the
    Python OSQP package.
    """

    opts = _normalize_options(options)
    target_device = torch.device(torch_device)
    torch_dtype = torch.double if double_precision else torch.float
    backend = _select_backend(opts, target_device)

    if backend["name"] == "torch":
        torch_settings = _normalize_torch_settings(
            opts["settings"], opts["user_settings"]
        )
        try:
            return _solve_torch_osqp_path(
                H,
                f,
                A,
                b,
                LB,
                UB,
                target_device,
                backend["solve_device"],
                torch_dtype,
                torch_settings,
                backend["allow_device_move"],
            )
        except Exception as exc:
            if (
                not backend["fallback_on_unsupported"]
                or _explicit_concrete_torch_solver(opts["user_settings"])
            ):
                raise
            warnings.warn(
                "CUDA Torch OSQP was selected by osqp_algebra='auto' but the "
                "Torch solve path failed for this problem/device "
                f"({type(exc).__name__}: {exc}). Falling back to builtin CPU OSQP.",
                RuntimeWarning,
                stacklevel=2,
            )

    return _solve_builtin_osqp_path(
        H,
        f,
        A,
        b,
        LB,
        UB,
        target_device,
        torch_dtype,
        _builtin_osqp_settings(opts["settings"]),
        opts["builtin_workspace_cache"],
    )


def _select_backend(opts, target_device):
    algebra = opts["algebra"]
    if algebra == "cuda":
        raise OSQPCudaInteropUnavailableError(
            "PyGRANSO OSQP CUDA tensor interop is not implemented yet. "
            "Use opts.osqp_algebra = 'torch' for the Torch prototype, "
            "or opts.osqp_algebra = 'builtin' for the CPU OSQP path."
        )

    if algebra == "auto":
        if torch.cuda.is_available():
            solve_device = target_device if target_device.type == "cuda" else torch.device("cuda")
            return {
                "name": "torch",
                "solve_device": solve_device,
                "allow_device_move": True,
                "fallback_on_unsupported": True,
            }
        return {
            "name": "builtin",
            "solve_device": torch.device("cpu"),
            "allow_device_move": False,
            "fallback_on_unsupported": False,
        }

    if algebra == "torch":
        return {
            "name": "torch",
            "solve_device": target_device,
            "allow_device_move": False,
            "fallback_on_unsupported": False,
        }

    if target_device.type == "cuda":
        if not opts["cuda_fallback"]:
            raise OSQPCudaInteropUnavailableError(
                "PyGRANSO received CUDA QP tensors but the current OSQP adapter "
                "would need to copy through CPU. Set opts.osqp_cuda_fallback = True "
                "to allow that explicit fallback, or set opts.osqp_algebra = 'torch' "
                "to use the Torch prototype."
            )
        warnings.warn(
            "Falling back to CPU OSQP for CUDA PyGRANSO QP tensors. This copies QP "
            "data to CPU and returns the solution to the requested CUDA device.",
            RuntimeWarning,
            stacklevel=2,
        )
    return {
        "name": "builtin",
        "solve_device": torch.device("cpu"),
        "allow_device_move": False,
        "fallback_on_unsupported": False,
    }


def _solve_builtin_osqp_path(
    H,
    f,
    A,
    b,
    LB,
    UB,
    target_device,
    torch_dtype,
    settings,
    workspace_cache=False,
):
    global _BUILTIN_OSQP_WORKSPACE, _BUILTIN_OSQP_WORKSPACE_STATS
    osqp = _import_osqp()
    H_np = _column_or_matrix_to_numpy(H, "H")
    f_np = _column_or_matrix_to_numpy(f, "f").reshape(-1)
    LB_np = _column_or_matrix_to_numpy(LB, "LB").reshape(-1, 1)
    UB_np = _column_or_matrix_to_numpy(UB, "UB").reshape(-1, 1)
    nvar = f_np.size

    if H_np.shape != (nvar, nvar):
        raise ValueError(f"H must have shape {(nvar, nvar)}, got {H_np.shape}.")
    if LB_np.shape != (nvar, 1) or UB_np.shape != (nvar, 1):
        raise ValueError("LB and UB must be column vectors with len(f) rows.")

    H_sparse = sparse.triu(sparse.csc_matrix(H_np), format="csc")
    A_new, LB_new, UB_new = _build_constraints(A, b, LB_np, UB_np, nvar)
    cache_hit = bool(
        workspace_cache
        and _BUILTIN_OSQP_WORKSPACE is not None
        and _same_csc_structure(_BUILTIN_OSQP_WORKSPACE["P"], H_sparse)
        and _same_csc_structure(_BUILTIN_OSQP_WORKSPACE["A"], A_new)
    )
    if cache_hit:
        prob = _BUILTIN_OSQP_WORKSPACE["prob"]
        update_values = {"q": f_np, "l": LB_new, "u": UB_new}
        if not np.array_equal(_BUILTIN_OSQP_WORKSPACE["P"].data, H_sparse.data):
            update_values["Px"] = H_sparse.data
        if not np.array_equal(_BUILTIN_OSQP_WORKSPACE["A"].data, A_new.data):
            update_values["Ax"] = A_new.data
        prob.update(**update_values)
        previous = _BUILTIN_OSQP_WORKSPACE.get("result")
        if previous is not None and previous.x is not None and previous.y is not None:
            prob.warm_start(x=previous.x, y=previous.y)
        _BUILTIN_OSQP_WORKSPACE_STATS["updates"] += 1
    else:
        prob = osqp.OSQP(algebra="builtin")
        prob.setup(H_sparse, f_np, A_new, LB_new, UB_new, **settings)
        if workspace_cache:
            rebuilds = 0
            if _BUILTIN_OSQP_WORKSPACE_STATS is not None:
                rebuilds = _BUILTIN_OSQP_WORKSPACE_STATS["rebuilds"] + 1
            _BUILTIN_OSQP_WORKSPACE_STATS = {
                "setups": 1,
                "updates": 0,
                "rebuilds": rebuilds,
                "last_cache_hit": False,
            }
    res = prob.solve()
    if workspace_cache:
        _BUILTIN_OSQP_WORKSPACE = {
            "prob": prob,
            "P": H_sparse,
            "A": A_new,
            "result": res,
        }
        _BUILTIN_OSQP_WORKSPACE_STATS["last_cache_hit"] = cache_hit

    solution = getattr(res, "x", None)
    if solution is None or solution.size == 0:
        raise RuntimeError("OSQP did not return a primal solution.")
    solution = np.asarray(solution).reshape((nvar, 1))
    if not np.all(np.isfinite(solution)):
        raise RuntimeError("OSQP returned a non-finite solution.")

    return torch.from_numpy(solution).to(device=target_device, dtype=torch_dtype)


def reset_builtin_osqp_workspace():
    global _BUILTIN_OSQP_WORKSPACE, _BUILTIN_OSQP_WORKSPACE_STATS
    _BUILTIN_OSQP_WORKSPACE = None
    _BUILTIN_OSQP_WORKSPACE_STATS = {
        "setups": 0,
        "updates": 0,
        "rebuilds": 0,
        "last_cache_hit": False,
    }


def get_builtin_osqp_workspace_stats():
    if _BUILTIN_OSQP_WORKSPACE_STATS is None:
        return {"setups": 0, "updates": 0, "rebuilds": 0, "last_cache_hit": False}
    return dict(_BUILTIN_OSQP_WORKSPACE_STATS)


def _same_csc_structure(left, right):
    return (
        left.shape == right.shape
        and np.array_equal(left.indptr, right.indptr)
        and np.array_equal(left.indices, right.indices)
    )


def _solve_torch_osqp_path(
    H,
    f,
    A,
    b,
    LB,
    UB,
    target_device,
    solve_device,
    torch_dtype,
    settings,
    allow_device_move=False,
):
    with torch.no_grad():
        requested_linear_solver = settings["linear_solver"]
        preserve_sparse = requested_linear_solver in {"auto", "sparse_cg"}
        P = _torch_qp_tensor(
            H,
            "H",
            solve_device,
            torch_dtype,
            preserve_sparse=preserve_sparse,
            allow_device_move=allow_device_move,
        )
        device = P.device

        q = _torch_qp_tensor(
            f, "f", device, torch_dtype, allow_device_move=allow_device_move
        ).reshape(-1)
        nvar = q.numel()
        if P.shape != (nvar, nvar):
            raise ValueError(f"H must have shape {(nvar, nvar)}, got {P.shape}.")

        A_eq = None
        b_eq = None
        if A is not None and b is not None:
            A_eq = _torch_qp_tensor(
                A,
                "A",
                device,
                torch_dtype,
                preserve_sparse=preserve_sparse,
                allow_device_move=allow_device_move,
            )
            if A_eq.ndim == 1:
                A_eq = A_eq.reshape(1, -1)
            if A_eq.ndim != 2:
                raise ValueError("A must be a vector or matrix.")
            if A_eq.shape[1] != nvar:
                raise ValueError(f"A must have {nvar} columns, got {A_eq.shape[1]}.")
            b_eq = _torch_rhs_tensor(
                b, "b", device, torch_dtype, allow_device_move=allow_device_move
            )

        selection = _select_torch_linear_solver(P, A_eq, nvar, torch_dtype, settings)
        solve_settings = settings.copy()
        solve_settings.update(selection)
        solve_settings["linear_solver"] = selection["linear_solver_selected"]

        if selection["linear_solver_selected"] == "sparse_cg":
            LB_t = _torch_qp_tensor(
                LB, "LB", device, torch_dtype, allow_device_move=allow_device_move
            )
            UB_t = _torch_qp_tensor(
                UB, "UB", device, torch_dtype, allow_device_move=allow_device_move
            )
            try:
                solution = solve_torch_osqp_from_qp(
                    P, q, A_eq, b_eq, LB_t, UB_t, solve_settings
                )
                return _move_torch_solution_result(solution, target_device, torch_dtype)
            except Exception as exc:
                can_retry_dense = (
                    requested_linear_solver == "auto"
                    and _is_sparse_solver_unsupported_error(exc)
                    and selection["estimated_dense_kkt_mb"]
                    <= settings["linear_solver_auto_dense_memory_limit_mb"]
                )
                if not can_retry_dense:
                    raise
                solve_settings = solve_settings.copy()
                solve_settings["linear_solver"] = "dense"
                solve_settings["linear_solver_selected"] = "dense"
                solve_settings["linear_solver_auto_reason"] = (
                    "sparse_cg_failed_retry_dense: "
                    f"{type(exc).__name__}: {exc}"
                )

        A_osqp, l_osqp, u_osqp = _build_constraints_torch(
            A,
            b,
            LB,
            UB,
            nvar,
            device,
            torch_dtype,
            allow_device_move=allow_device_move,
        )
        solution = solve_torch_osqp(P, q, A_osqp, l_osqp, u_osqp, solve_settings)
        return _move_torch_solution_result(solution, target_device, torch_dtype)


def _normalize_options(options):
    options = options or {}
    algebra = options.get("algebra", "auto")
    if algebra not in {"auto", "builtin", "cuda", "torch"}:
        raise ValueError(
            "osqp_algebra must be one of 'auto', 'builtin', 'cuda', or 'torch'."
        )
    cuda_fallback = bool(options.get("cuda_fallback", False))
    user_settings = options.get("settings") or {}
    if not isinstance(user_settings, dict):
        raise ValueError("osqp settings must be provided as a dict.")
    settings = DEFAULT_OSQP_SETTINGS.copy()
    settings.update(user_settings)
    return {
        "algebra": algebra,
        "cuda_fallback": cuda_fallback,
        "builtin_workspace_cache": bool(options.get("builtin_workspace_cache", False)),
        "settings": settings,
        "user_settings": user_settings,
    }


def _normalize_torch_settings(settings, user_settings):
    torch_settings = DEFAULT_TORCH_OSQP_SETTINGS.copy()
    explicit_settings = set(user_settings)

    for key, value in settings.items():
        if key in SUPPORTED_TORCH_SETTINGS:
            torch_settings[key] = value
            continue

        # DEFAULT_OSQP_SETTINGS contains polish=True for the builtin CPU path.
        # Torch uses the explicit paper-complete spelling "polishing"; the
        # legacy default is ignored unless the user asked for polish directly.
        if key == "polish" and key not in explicit_settings:
            continue
        if key == "polish":
            torch_settings["polishing"] = bool(value)
            continue

        if key in UNSUPPORTED_TORCH_SETTINGS:
            if _unsupported_torch_setting_enabled(value):
                raise ValueError(
                    f"The Torch OSQP prototype does not support enabled setting "
                    f"{key!r}."
                )
            continue

        raise ValueError(f"The Torch OSQP prototype does not support setting {key!r}.")

    _validate_torch_settings(torch_settings)
    return torch_settings


def _builtin_osqp_settings(settings):
    builtin_settings = {
        key: value for key, value in settings.items() if key not in TORCH_ONLY_SETTINGS
    }
    if "polishing" in builtin_settings and "polish" in builtin_settings:
        del builtin_settings["polish"]
    return builtin_settings


def _select_torch_linear_solver(P, A_eq, nvar, torch_dtype, settings):
    requested = settings["linear_solver"]
    n_eq = 0 if A_eq is None else A_eq.shape[0]
    n_bounds = nvar
    effective_m = n_eq + n_bounds
    kkt_dim = nvar + effective_m
    dtype_bytes = torch.empty((), dtype=torch_dtype).element_size()
    dense_kkt_mb = (kkt_dim * kkt_dim * dtype_bytes) / (1024 * 1024)
    sparse_nnz = _torch_nnz(P) + _torch_nnz(A_eq) + n_bounds
    structural_entries = max(nvar * nvar + n_eq * nvar + n_bounds, 1)
    effective_density = sparse_nnz / structural_entries

    metadata = {
        "linear_solver_requested": requested,
        "estimated_kkt_dim": int(kkt_dim),
        "estimated_dense_kkt_mb": float(dense_kkt_mb),
        "estimated_sparse_nnz": int(sparse_nnz),
        "estimated_sparse_density": float(effective_density),
    }

    if requested in {"dense", "sparse_cg"}:
        metadata.update(
            {
                "linear_solver_selected": requested,
                "linear_solver_auto_reason": f"explicit_{requested}",
            }
        )
        return metadata

    if kkt_dim <= settings["linear_solver_auto_min_kkt_dim"]:
        selected = "dense"
        reason = "kkt_dim_below_dense_threshold"
    elif dense_kkt_mb > settings["linear_solver_auto_dense_memory_limit_mb"]:
        selected = "sparse_cg"
        reason = "dense_kkt_memory_exceeds_limit"
    elif (
        kkt_dim >= settings["linear_solver_auto_sparse_min_kkt_dim"]
        and effective_density <= settings["linear_solver_auto_max_density"]
    ):
        selected = "sparse_cg"
        reason = "large_sparse_problem"
    else:
        selected = "dense"
        reason = "conservative_dense_default"

    metadata.update(
        {
            "linear_solver_selected": selected,
            "linear_solver_auto_reason": reason,
        }
    )
    return metadata


def _torch_nnz(tensor):
    if tensor is None:
        return 0
    if tensor.layout == torch.strided:
        return int(torch.count_nonzero(tensor).item())
    return int(tensor._nnz())


def _move_torch_solution_result(result, target_device, torch_dtype):
    if isinstance(result, tuple):
        solution, info = result
        return solution.to(device=target_device, dtype=torch_dtype), info
    return result.to(device=target_device, dtype=torch_dtype)


def _is_sparse_solver_unsupported_error(exc):
    if isinstance(exc, NotImplementedError):
        return True
    message = str(exc).lower()
    return (
        "notimplemented" in message
        or "not implemented" in message
        or "unsupported" in message
        or "not available" in message
        or "sparse" in message
    )


def _explicit_concrete_torch_solver(user_settings):
    return user_settings.get("linear_solver") in {"dense", "sparse_cg"}


def _unsupported_torch_setting_enabled(value):
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, Number):
        return value != 0
    return bool(value)


def _validate_torch_settings(settings):
    if settings["linear_solver"] not in {"auto", "dense", "sparse_cg"}:
        raise ValueError(
            "Torch OSQP setting 'linear_solver' must be 'auto', 'dense', or "
            "'sparse_cg'."
        )
    settings["rho"] = _positive_float(settings["rho"], "rho")
    settings["sigma"] = _positive_float(settings["sigma"], "sigma")
    settings["alpha"] = _positive_float(settings["alpha"], "alpha")
    if settings["alpha"] >= 2:
        raise ValueError("Torch OSQP setting 'alpha' must be in (0, 2).")
    settings["max_iter"] = _positive_int(settings["max_iter"], "max_iter")
    settings["eps_abs"] = _nonnegative_float(settings["eps_abs"], "eps_abs")
    settings["eps_rel"] = _nonnegative_float(settings["eps_rel"], "eps_rel")
    settings["check_termination"] = _positive_int(
        settings["check_termination"], "check_termination"
    )
    settings["cg_rtol"] = _nonnegative_float(settings["cg_rtol"], "cg_rtol")
    settings["cg_atol"] = _nonnegative_float(settings["cg_atol"], "cg_atol")
    settings["cg_max_iter"] = _positive_int(settings["cg_max_iter"], "cg_max_iter")
    settings["cg_check_interval"] = _positive_int(
        settings["cg_check_interval"], "cg_check_interval"
    )
    settings["cg_fixed_iters"] = _optional_positive_int(
        settings["cg_fixed_iters"], "cg_fixed_iters"
    )
    settings["torch_compile_admm"] = bool(settings["torch_compile_admm"])
    settings["cuda_graph"] = bool(settings["cuda_graph"])
    settings["cuda_event_timing"] = bool(settings["cuda_event_timing"])
    settings["scaling"] = _nonnegative_int(settings["scaling"], "scaling")
    settings["adaptive_rho"] = bool(settings["adaptive_rho"])
    settings["rho_update_interval"] = _rho_update_interval_setting(
        settings["rho_update_interval"], "rho_update_interval"
    )
    settings["rho_update_tolerance"] = _positive_float(
        settings["rho_update_tolerance"], "rho_update_tolerance"
    )
    settings["warm_start"] = bool(settings["warm_start"])
    if settings["initial_state"] is not None and not isinstance(
        settings["initial_state"], dict
    ):
        raise ValueError("Torch OSQP setting 'initial_state' must be a dict or None.")
    settings["return_state"] = bool(settings["return_state"])
    settings["polishing"] = bool(settings["polishing"])
    settings["polish_delta"] = _positive_float(
        settings["polish_delta"], "polish_delta"
    )
    settings["polish_refine_iter"] = _nonnegative_int(
        settings["polish_refine_iter"], "polish_refine_iter"
    )
    settings["linear_solver_auto_min_kkt_dim"] = _positive_int(
        settings["linear_solver_auto_min_kkt_dim"],
        "linear_solver_auto_min_kkt_dim",
    )
    settings["linear_solver_auto_sparse_min_kkt_dim"] = _positive_int(
        settings["linear_solver_auto_sparse_min_kkt_dim"],
        "linear_solver_auto_sparse_min_kkt_dim",
    )
    settings["linear_solver_auto_max_density"] = _density_float(
        settings["linear_solver_auto_max_density"],
        "linear_solver_auto_max_density",
    )
    settings["linear_solver_auto_dense_memory_limit_mb"] = _positive_float(
        settings["linear_solver_auto_dense_memory_limit_mb"],
        "linear_solver_auto_dense_memory_limit_mb",
    )
    settings["return_info"] = bool(settings["return_info"])
    settings["verbose"] = bool(settings["verbose"])


def _positive_float(value, name):
    value = _float_setting(value, name)
    if value <= 0:
        raise ValueError(f"Torch OSQP setting {name!r} must be positive.")
    return value


def _nonnegative_float(value, name):
    value = _float_setting(value, name)
    if value < 0:
        raise ValueError(f"Torch OSQP setting {name!r} must be nonnegative.")
    return value


def _density_float(value, name):
    value = _positive_float(value, name)
    if value > 1:
        raise ValueError(f"Torch OSQP setting {name!r} must be in (0, 1].")
    return value


def _float_setting(value, name):
    if isinstance(value, bool) or not isinstance(value, Number):
        raise ValueError(f"Torch OSQP setting {name!r} must be numeric.")
    return float(value)


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"Torch OSQP setting {name!r} must be a positive integer.")
    value = int(value)
    if value <= 0:
        raise ValueError(f"Torch OSQP setting {name!r} must be a positive integer.")
    return value


def _nonnegative_int(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(
            f"Torch OSQP setting {name!r} must be a nonnegative integer."
        )
    value = int(value)
    if value < 0:
        raise ValueError(
            f"Torch OSQP setting {name!r} must be a nonnegative integer."
        )
    return value


def _optional_positive_int(value, name):
    if value is None:
        return None
    if value == "auto":
        return value
    return _positive_int(value, name)


def _rho_update_interval_setting(value, name):
    if value == "auto":
        return value
    return _positive_int(value, name)


def _import_osqp():
    try:
        return importlib.import_module("osqp")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The OSQP Python package is required for QPsolver='osqp'. "
            "Install PyGRANSO with its OSQP dependency, for example `pip install -e .`."
        ) from exc


def _column_or_matrix_to_numpy(value, name):
    if torch.is_tensor(value):
        tensor = value.detach().cpu()
        if tensor.layout != torch.strided:
            tensor = tensor.to_dense()
        return tensor.numpy()
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, Number):
        return np.asarray([[value]])
    return np.asarray(value)


def _build_constraints(A, b, LB, UB, nvar):
    eye = sparse.eye(nvar, format="csc")
    if A is None or b is None:
        return eye, LB, UB

    A_np = _column_or_matrix_to_numpy(A, "A")
    if A_np.ndim == 1:
        A_np = A_np.reshape(1, -1)
    if A_np.shape[1] != nvar:
        raise ValueError(f"A must have {nvar} columns, got {A_np.shape[1]}.")

    b_np = _column_or_matrix_to_numpy(b, "b").reshape(-1, 1)
    if b_np.size == 1 and A_np.shape[0] != 1:
        b_np = np.full((A_np.shape[0], 1), float(b_np.reshape(-1)[0]))
    if b_np.shape != (A_np.shape[0], 1):
        raise ValueError("b must be scalar or have one entry per row of A.")

    Aeq = sparse.csc_matrix(A_np)
    A_new = sparse.vstack([Aeq, eye], format="csc")
    return A_new, np.vstack((b_np, LB)), np.vstack((b_np, UB))


def _build_constraints_torch(
    A, b, LB, UB, nvar, device, dtype, allow_device_move=False
):
    LB_t = _torch_qp_tensor(
        LB, "LB", device, dtype, allow_device_move=allow_device_move
    ).reshape(-1)
    UB_t = _torch_qp_tensor(
        UB, "UB", device, dtype, allow_device_move=allow_device_move
    ).reshape(-1)
    if LB_t.numel() != nvar or UB_t.numel() != nvar:
        raise ValueError("LB and UB must be column vectors with len(f) rows.")

    eye = torch.eye(nvar, device=device, dtype=dtype)
    if A is None or b is None:
        return eye, LB_t, UB_t

    A_t = _torch_qp_tensor(A, "A", device, dtype, allow_device_move=allow_device_move)
    if A_t.ndim == 1:
        A_t = A_t.reshape(1, -1)
    if A_t.ndim != 2:
        raise ValueError("A must be a vector or matrix.")
    if A_t.shape[1] != nvar:
        raise ValueError(f"A must have {nvar} columns, got {A_t.shape[1]}.")

    b_t = _torch_rhs_tensor(
        b, "b", device, dtype, allow_device_move=allow_device_move
    ).reshape(-1)
    if b_t.numel() == 1 and A_t.shape[0] != 1:
        b_t = b_t.expand(A_t.shape[0])
    if b_t.numel() != A_t.shape[0]:
        raise ValueError("b must be scalar or have one entry per row of A.")

    A_osqp = torch.cat((A_t, eye), dim=0)
    l_osqp = torch.cat((b_t, LB_t), dim=0)
    u_osqp = torch.cat((b_t, UB_t), dim=0)
    return A_osqp, l_osqp, u_osqp


def _torch_qp_tensor(
    value,
    name,
    device,
    dtype,
    preserve_sparse=False,
    allow_device_move=False,
):
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a Torch tensor for the Torch OSQP backend.")

    tensor = value.detach()
    if tensor.layout != torch.strided and not preserve_sparse:
        tensor = tensor.to_dense()
    if device is not None and not _device_matches(tensor.device, device):
        if not allow_device_move:
            raise ValueError(
                f"{name} must be on {device} for the Torch OSQP backend, "
                f"got {tensor.device}."
            )
        tensor = tensor.to(device=device)
    if tensor.dtype != dtype:
        tensor = tensor.to(dtype=dtype)
    return tensor


def _torch_rhs_tensor(value, name, device, dtype, allow_device_move=False):
    if torch.is_tensor(value):
        return _torch_qp_tensor(
            value, name, device, dtype, allow_device_move=allow_device_move
        )
    if isinstance(value, Number):
        return torch.tensor(value, device=device, dtype=dtype)
    return torch.as_tensor(value, device=device, dtype=dtype)


def _device_matches(actual_device, requested_device):
    if actual_device.type != requested_device.type:
        return False
    if requested_device.index is None:
        return True
    return actual_device.index == requested_device.index


def _ensure_requested_device(tensor, target_device, name):
    if tensor.device.type != target_device.type:
        raise ValueError(
            f"{name} is on {tensor.device}, but Torch OSQP was requested on "
            f"{target_device}."
        )
    if target_device.index is not None and tensor.device.index != target_device.index:
        raise ValueError(
            f"{name} is on {tensor.device}, but Torch OSQP was requested on "
            f"{target_device}."
        )
