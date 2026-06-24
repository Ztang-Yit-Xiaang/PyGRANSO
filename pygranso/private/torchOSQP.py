import importlib.util
import time

import torch

SPARSE_LAYOUTS = {
    torch.sparse_coo,
    torch.sparse_csr,
    torch.sparse_csc,
    torch.sparse_bsr,
    torch.sparse_bsc,
}

_COMPILED_ADMM_UPDATE = None
_COMPILED_ADMM_ERROR = None


def solve_torch_osqp(P, q, A, l, u, settings):
    """Solve an OSQP-form QP with a selectable Torch ADMM backend."""
    linear_solver = settings.get("linear_solver", "dense")
    if linear_solver == "sparse_cg":
        return solve_torch_osqp_sparse_cg(P, q, A, l, u, settings)
    if linear_solver != "dense":
        raise ValueError(f"Unknown Torch OSQP linear_solver {linear_solver!r}.")
    return solve_torch_osqp_dense(P, q, A, l, u, settings)


def solve_torch_osqp_from_qp(P, q, A_eq, b_eq, LB, UB, settings):
    """Solve PyGRANSO's QP form without materializing bound identity rows."""
    linear_solver = settings.get("linear_solver", "dense")
    if linear_solver != "sparse_cg":
        raise ValueError("solve_torch_osqp_from_qp is only for linear_solver='sparse_cg'.")

    with torch.no_grad():
        q = _detached_vector(q, "q")
        P = _as_sparse_csr(P, "P")
        device = q.device
        dtype = q.dtype
        n = q.numel()
        if P.shape != (n, n):
            raise ValueError(f"P must have shape {(n, n)}, got {P.shape}.")

        LB = _detached_vector(LB, "LB", device, dtype)
        UB = _detached_vector(UB, "UB", device, dtype)
        if LB.numel() != n or UB.numel() != n:
            raise ValueError("LB and UB must be column vectors with len(q) rows.")

        if A_eq is None or b_eq is None:
            A_eq_csr = None
            b_vec = None
            l = LB
            u = UB
        else:
            A_eq_csr = _as_sparse_csr(A_eq, "A")
            if A_eq_csr.shape[1] != n:
                raise ValueError(f"A must have {n} columns, got {A_eq_csr.shape[1]}.")
            b_vec = _detached_vector(b_eq, "b", device, dtype)
            if b_vec.numel() == 1 and A_eq_csr.shape[0] != 1:
                b_vec = b_vec.expand(A_eq_csr.shape[0])
            if b_vec.numel() != A_eq_csr.shape[0]:
                raise ValueError("b must be scalar or have one entry per row of A.")
            l = torch.cat((b_vec, LB), dim=0)
            u = torch.cat((b_vec, UB), dim=0)

        original = {
            "P": P,
            "q": q,
            "A_eq": A_eq_csr,
            "b": b_vec,
            "LB": LB,
            "UB": UB,
            "l": l,
            "u": u,
        }
        scaling = _ruiz_scale_qp_data(P, q, A_eq_csr, b_vec, LB, UB, settings)
        if scaling is not None:
            P = scaling["P"]
            q = scaling["q"]
            A_eq_csr = scaling["A_eq"]
            b_vec = scaling["b"]
            LB = scaling["LB"]
            UB = scaling["UB"]
            if b_vec is None:
                l = LB
                u = UB
            else:
                l = torch.cat((b_vec, LB), dim=0)
                u = torch.cat((b_vec, UB), dim=0)

        operator = BoundConstrainedOSQPOperator(
            P, A_eq_csr, n, device, dtype, cache=_initial_sparse_cache(settings)
        )
        solve_settings = settings
        if scaling is not None:
            solve_settings = settings.copy()
            solve_settings["_include_state_for_postprocess"] = True
        solution, info = _solve_sparse_cg_operator(operator, q, l, u, solve_settings)
        if scaling is not None:
            original_operator = BoundConstrainedOSQPOperator(
                original["P"], original["A_eq"], n, device, dtype
            )
            solution, info = _unscale_sparse_cg_result(
                solution,
                info,
                scaling,
                original_operator,
                original["q"],
                original["l"],
                original["u"],
                settings,
            )
        if settings.get("return_info", False):
            return solution, info
        return solution


def solve_torch_osqp_dense(P, q, A, l, u, settings):
    """Solve an OSQP-form QP with the original dense Torch ADMM prototype.

    This is intentionally a PyGRANSO-side dense prototype. It does not use the
    sparse C/CUDA OSQP algebra layer and should not be treated as a final OSQP
    CUDA interop implementation.
    """
    with torch.no_grad():
        P = _dense_detached(P)
        q = q.detach().reshape(-1)
        A = _dense_detached(A)
        l = l.detach().reshape(-1)
        u = u.detach().reshape(-1)

        n = q.numel()
        m = l.numel()
        if P.shape != (n, n):
            raise ValueError(f"P must have shape {(n, n)}, got {P.shape}.")
        if A.shape != (m, n):
            raise ValueError(f"A must have shape {(m, n)}, got {A.shape}.")
        if u.numel() != m:
            raise ValueError("l and u must have the same number of entries.")

        rho = settings["rho"]
        sigma = settings["sigma"]
        alpha = settings["alpha"]
        max_iter = settings["max_iter"]
        eps_abs = settings["eps_abs"]
        eps_rel = settings["eps_rel"]
        check_termination = settings["check_termination"]
        verbose = settings["verbose"]

        device = q.device
        dtype = q.dtype
        eye_n = torch.eye(n, device=device, dtype=dtype)
        eye_m = torch.eye(m, device=device, dtype=dtype)

        top = torch.cat((P + sigma * eye_n, A.T), dim=1)
        bottom = torch.cat((A, -(1.0 / rho) * eye_m), dim=1)
        K = torch.cat((top, bottom), dim=0)

        x = torch.zeros(n, device=device, dtype=dtype)
        z = torch.zeros(m, device=device, dtype=dtype)
        y = torch.zeros(m, device=device, dtype=dtype)
        status = "max_iter_reached"
        last_residuals = None

        for iteration in range(1, max_iter + 1):
            rhs = torch.cat((sigma * x - q, z - y / rho))
            solution = torch.linalg.solve(K, rhs)
            x_tilde = solution[:n]
            nu = solution[n:]

            z_tilde = z + (nu - y) / rho
            x_next = alpha * x_tilde + (1.0 - alpha) * x
            z_relaxed = alpha * z_tilde + (1.0 - alpha) * z
            z_next = torch.clamp(z_relaxed + y / rho, min=l, max=u)
            y_next = y + rho * (z_relaxed - z_next)

            x = x_next
            z = z_next
            y = y_next

            if iteration % check_termination == 0:
                last_residuals = _dense_residuals(P, q, A, x, z, y, eps_abs, eps_rel)
                primal_res, dual_res, eps_prim, eps_dual = last_residuals
                if bool((primal_res <= eps_prim).item()) and bool(
                    (dual_res <= eps_dual).item()
                ):
                    status = "solved"
                    if verbose:
                        print(f"Torch OSQP prototype converged in {iteration} iterations.")
                    break
        else:
            iteration = max_iter
            if verbose:
                if last_residuals is None:
                    last_residuals = _dense_residuals(
                        P, q, A, x, z, y, eps_abs, eps_rel
                    )
                primal_res, dual_res, eps_prim, eps_dual = last_residuals
                print(
                    "Torch OSQP prototype reached max_iter="
                    f"{max_iter} with primal={primal_res.item():.3e}/"
                    f"{eps_prim.item():.3e}, dual={dual_res.item():.3e}/"
                    f"{eps_dual.item():.3e}."
                )

        if not torch.all(torch.isfinite(x)):
            raise RuntimeError("Torch OSQP prototype returned a non-finite solution.")
        info = _dense_info(P, q, A, x, z, y, settings, iteration, status)
        if settings.get("return_info", False):
            return x.reshape(n, 1), info
        return x.reshape(n, 1)


def solve_torch_osqp_sparse_cg(P, q, A, l, u, settings):
    """Solve an OSQP-form QP with sparse matvecs and preconditioned CG."""
    with torch.no_grad():
        q = _detached_vector(q, "q")
        device = q.device
        dtype = q.dtype
        P = _as_sparse_csr(P, "P")
        A = _as_sparse_csr(A, "A")
        l = _detached_vector(l, "l", device, dtype)
        u = _detached_vector(u, "u", device, dtype)

        n = q.numel()
        m = l.numel()
        if P.shape != (n, n):
            raise ValueError(f"P must have shape {(n, n)}, got {P.shape}.")
        if A.shape != (m, n):
            raise ValueError(f"A must have shape {(m, n)}, got {A.shape}.")
        if u.numel() != m:
            raise ValueError("l and u must have the same number of entries.")

        original = {"P": P, "q": q, "A": A, "l": l, "u": u}
        scaling = _ruiz_scale_osqp_data(P, q, A, l, u, settings)
        if scaling is not None:
            P = scaling["P"]
            q = scaling["q"]
            A = scaling["A"]
            l = scaling["l"]
            u = scaling["u"]

        operator = ExplicitOSQPOperator(
            P, A, device, dtype, cache=_initial_sparse_cache(settings)
        )
        solve_settings = settings
        if scaling is not None:
            solve_settings = settings.copy()
            solve_settings["_include_state_for_postprocess"] = True
        solution, info = _solve_sparse_cg_operator(operator, q, l, u, solve_settings)
        if scaling is not None:
            original_operator = ExplicitOSQPOperator(
                original["P"], original["A"], device, dtype
            )
            solution, info = _unscale_sparse_cg_result(
                solution,
                info,
                scaling,
                original_operator,
                original["q"],
                original["l"],
                original["u"],
                settings,
            )
        if settings.get("return_info", False):
            return solution, info
        return solution


class ExplicitOSQPOperator:
    """Sparse OSQP operator for an explicitly provided constraint matrix."""

    uses_matrix_free_bounds = False

    def __init__(self, P_csr, A_csr, device, dtype, cache=None):
        self.P = P_csr
        self.A = A_csr
        self.device = device
        self.dtype = dtype
        self.n = P_csr.shape[0]
        self.m = A_csr.shape[0]
        self._cache = _compatible_sparse_cache(cache, "explicit", P_csr, A_csr)
        self.cache_hit = self._cache is not None
        transpose = _transpose_structure(A_csr, self._cache, "AT")
        self.AT = _csr_with_values(
            transpose["crow_indices"],
            transpose["col_indices"],
            A_csr.values()[transpose["value_map"]],
            (A_csr.shape[1], A_csr.shape[0]),
        )
        self._AT_structure = transpose
        self._diag_P = None
        self._A_rows = self._cache.get("A_rows") if self.cache_hit else _csr_row_indices(A_csr)
        self._A_cols = self._cache.get("A_cols") if self.cache_hit else A_csr.col_indices()
        self._A_values = A_csr.values()

    def P_mv(self, vector):
        return spmv(self.P, vector)

    def A_mv(self, vector):
        return spmv(self.A, vector)

    def AT_mv(self, vector):
        return spmv(self.AT, vector)

    def refresh_transpose_values(self):
        self.AT.values().copy_(self.A.values()[self._AT_structure["value_map"]])

    def diag_P(self):
        if self._diag_P is None:
            self._diag_P = sparse_diagonal(self.P)
        return self._diag_P

    def diag_ATRA(self, rho_vec):
        return sparse_gram_diagonal_from_indices(
            self.A.shape[1], self._A_rows, self._A_cols, self._A_values, rho_vec
        )

    def sparse_storage_nnz(self):
        return _nnz(self.P) + _nnz(self.A) + _nnz(self.AT)

    def sparse_cache(self):
        return {
            "kind": "explicit",
            "device": str(self.device),
            "dtype": str(self.dtype),
            "P_shape": tuple(self.P.shape),
            "A_shape": tuple(self.A.shape),
            "P_nnz": _nnz(self.P),
            "A_nnz": _nnz(self.A),
            **_cached_structure("P", self.P),
            **_cached_structure("A", self.A),
            "AT_crow_indices": self._AT_structure["crow_indices"].detach(),
            "AT_col_indices": self._AT_structure["col_indices"].detach(),
            "AT_value_map": self._AT_structure["value_map"].detach(),
            "A_rows": self._A_rows.detach(),
            "A_cols": self._A_cols.detach(),
        }


class BoundConstrainedOSQPOperator:
    """Sparse equality operator plus matrix-free variable-bound identity rows."""

    uses_matrix_free_bounds = True

    def __init__(self, P_csr, A_eq_csr, n, device, dtype, cache=None):
        self.P = P_csr
        self.A_eq = A_eq_csr
        self.device = device
        self.dtype = dtype
        self.n = n
        self.n_eq = 0 if A_eq_csr is None else A_eq_csr.shape[0]
        self.m = self.n_eq + n
        self._cache = _compatible_sparse_cache(cache, "bounds", P_csr, A_eq_csr)
        self.cache_hit = self._cache is not None
        self._AT_eq_structure = None
        if A_eq_csr is None:
            self.AT_eq = None
        else:
            transpose = _transpose_structure(A_eq_csr, self._cache, "AT_eq")
            self.AT_eq = _csr_with_values(
                transpose["crow_indices"],
                transpose["col_indices"],
                A_eq_csr.values()[transpose["value_map"]],
                (A_eq_csr.shape[1], A_eq_csr.shape[0]),
            )
            self._AT_eq_structure = transpose
        self._diag_P = None
        if A_eq_csr is None:
            self._A_eq_rows = None
            self._A_eq_cols = None
            self._A_eq_values = None
        else:
            self._A_eq_rows = (
                self._cache.get("A_eq_rows") if self.cache_hit else _csr_row_indices(A_eq_csr)
            )
            self._A_eq_cols = (
                self._cache.get("A_eq_cols") if self.cache_hit else A_eq_csr.col_indices()
            )
            self._A_eq_values = A_eq_csr.values()

    def P_mv(self, vector):
        return spmv(self.P, vector)

    def A_eq_mv(self, vector):
        if self.A_eq is None:
            return torch.empty(0, device=self.device, dtype=self.dtype)
        return spmv(self.A_eq, vector)

    def A_mv(self, vector):
        if self.A_eq is None:
            return vector
        return torch.cat((self.A_eq_mv(vector), vector), dim=0)

    def AT_mv(self, vector):
        if self.A_eq is None:
            return vector
        eq_part = vector[: self.n_eq]
        bound_part = vector[self.n_eq :]
        return spmv(self.AT_eq, eq_part) + bound_part

    def refresh_transpose_values(self):
        if self.A_eq is not None:
            self.AT_eq.values().copy_(
                self.A_eq.values()[self._AT_eq_structure["value_map"]]
            )

    def diag_P(self):
        if self._diag_P is None:
            self._diag_P = sparse_diagonal(self.P)
        return self._diag_P

    def diag_ATRA(self, rho_vec):
        bound_diag = rho_vec[self.n_eq :]
        if self.A_eq is None:
            return bound_diag
        eq_diag = sparse_gram_diagonal_from_indices(
            self.A_eq.shape[1],
            self._A_eq_rows,
            self._A_eq_cols,
            self._A_eq_values,
            rho_vec[: self.n_eq],
        )
        return eq_diag + bound_diag

    def sparse_storage_nnz(self):
        total = _nnz(self.P)
        if self.A_eq is not None:
            total += _nnz(self.A_eq) + _nnz(self.AT_eq)
        return total

    def sparse_cache(self):
        cache = {
            "kind": "bounds",
            "device": str(self.device),
            "dtype": str(self.dtype),
            "P_shape": tuple(self.P.shape),
            "A_shape": None if self.A_eq is None else tuple(self.A_eq.shape),
            "P_nnz": _nnz(self.P),
            "A_nnz": 0 if self.A_eq is None else _nnz(self.A_eq),
            **_cached_structure("P", self.P),
        }
        if self.A_eq is not None:
            cache.update(
                {
                    **_cached_structure("A", self.A_eq),
                    "AT_eq_crow_indices": self._AT_eq_structure[
                        "crow_indices"
                    ].detach(),
                    "AT_eq_col_indices": self._AT_eq_structure[
                        "col_indices"
                    ].detach(),
                    "AT_eq_value_map": self._AT_eq_structure["value_map"].detach(),
                    "A_eq_rows": self._A_eq_rows.detach(),
                    "A_eq_cols": self._A_eq_cols.detach(),
                }
            )
        return cache


def spmv(matrix, vector):
    """Sparse or dense matrix-vector multiply without densifying sparse matrices."""
    if matrix.layout == torch.strided:
        return matrix @ vector
    return torch.sparse.mm(matrix, vector.reshape(-1, 1)).reshape(-1)


def reduced_system_matvec(operator, sigma, rho_vec, vector):
    Av = operator.A_mv(vector)
    return operator.P_mv(vector) + sigma * vector + operator.AT_mv(rho_vec * Av)


def jacobi_preconditioner_diagonal(operator, sigma, rho_vec):
    return operator.diag_P() + sigma + operator.diag_ATRA(rho_vec)


def conjugate_gradient(
    matvec,
    b,
    x0=None,
    preconditioner=None,
    rtol=1e-6,
    atol=0.0,
    max_iter=100,
    check_interval=1,
    fixed_iters=None,
):
    """Small preconditioned CG helper for symmetric positive definite systems."""
    if x0 is None:
        x = torch.zeros_like(b)
    else:
        x = x0.detach().clone()

    r = b - matvec(x)
    b_norm = torch.linalg.vector_norm(b)
    residual_norm = torch.linalg.vector_norm(r)
    tolerance = torch.maximum(
        torch.as_tensor(float(atol), device=b.device, dtype=b.dtype),
        torch.as_tensor(float(rtol), device=b.device, dtype=b.dtype) * b_norm,
    )
    if fixed_iters is None and bool((residual_norm <= tolerance).item()):
        return x, _cg_info(True, 0, residual_norm, b_norm, "converged")

    z = preconditioner(r) if preconditioner is not None else r
    p = z.clone()
    rz_old = torch.dot(r, z)
    breakdown_eps = torch.as_tensor(torch.finfo(b.dtype).eps, device=b.device, dtype=b.dtype)
    status = "max_iter_reached"
    converged = False
    iteration = 0
    check_interval = max(1, int(check_interval))
    target_iter = int(fixed_iters) if fixed_iters is not None else int(max_iter)

    for iteration in range(1, target_iter + 1):
        Ap = matvec(p)
        denom = torch.dot(p, Ap)
        denom_safe = torch.where(torch.abs(denom) <= breakdown_eps, breakdown_eps, denom)

        alpha = rz_old / denom_safe
        x = x + alpha * p
        r = r - alpha * Ap

        z = preconditioner(r) if preconditioner is not None else r
        rz_new = torch.dot(r, z)
        rz_old_safe = torch.where(torch.abs(rz_old) <= breakdown_eps, breakdown_eps, rz_old)
        beta = rz_new / rz_old_safe
        p = z + beta * p

        should_check = fixed_iters is None and iteration % check_interval == 0
        if should_check:
            residual_norm = torch.linalg.vector_norm(r)
            if bool(
                (
                    (residual_norm <= tolerance)
                    | (torch.abs(denom) <= breakdown_eps)
                    | (torch.abs(rz_old) <= breakdown_eps)
                ).item()
            ):
                if bool((residual_norm <= tolerance).item()):
                    status = "converged"
                    converged = True
                else:
                    status = "breakdown"
                break
        rz_old = rz_new

    if iteration > 0 and (fixed_iters is not None or iteration % check_interval != 0):
        residual_norm = torch.linalg.vector_norm(r)
        if fixed_iters is not None:
            status = "fixed_iters"

    return x, _cg_info(converged, iteration, residual_norm, b_norm, status)


def sparse_diagonal(matrix):
    if matrix.layout == torch.strided:
        return torch.diagonal(matrix)
    if matrix.layout == torch.sparse_csr:
        rows = _csr_row_indices(matrix)
        cols = matrix.col_indices()
        values = matrix.values()
    elif matrix.layout == torch.sparse_csc:
        cols = _csc_col_indices(matrix)
        rows = matrix.row_indices()
        values = matrix.values()
    else:
        coo = _to_coalesced_coo(matrix)
        rows = coo.indices()[0]
        cols = coo.indices()[1]
        values = coo.values()

    n = matrix.shape[0]
    diag = torch.zeros(n, device=values.device, dtype=values.dtype)
    mask = rows == cols
    if bool(torch.any(mask).item()):
        diag.scatter_add_(0, rows[mask], values[mask])
    return diag


def sparse_gram_diagonal(matrix, rho_vec):
    if matrix.shape[0] == 0:
        return torch.zeros(matrix.shape[1], device=rho_vec.device, dtype=rho_vec.dtype)
    if matrix.layout == torch.sparse_csr:
        rows = _csr_row_indices(matrix)
        cols = matrix.col_indices()
        values = matrix.values()
    elif matrix.layout == torch.sparse_csc:
        cols = _csc_col_indices(matrix)
        rows = matrix.row_indices()
        values = matrix.values()
    else:
        coo = _to_coalesced_coo(matrix)
        rows = coo.indices()[0]
        cols = coo.indices()[1]
        values = coo.values()

    return sparse_gram_diagonal_from_indices(matrix.shape[1], rows, cols, values, rho_vec)


def sparse_gram_diagonal_from_indices(n_cols, rows, cols, values, rho_vec):
    diag = torch.zeros(n_cols, device=values.device, dtype=values.dtype)
    contrib = rho_vec[rows] * values.square()
    if contrib.numel() > 0:
        diag.scatter_add_(0, cols, contrib)
    return diag


def _admm_update_function(settings):
    if not settings.get("torch_compile_admm", False):
        settings["_torch_compile_admm_status"] = "disabled"
        return _admm_vector_update
    if not hasattr(torch, "compile"):
        settings["_torch_compile_admm_status"] = "unavailable"
        return _admm_vector_update
    if importlib.util.find_spec("triton") is None:
        settings["_torch_compile_admm_status"] = "unavailable_triton"
        return _admm_vector_update

    global _COMPILED_ADMM_UPDATE, _COMPILED_ADMM_ERROR
    if _COMPILED_ADMM_UPDATE is not None:
        settings["_torch_compile_admm_status"] = "enabled"
        return _COMPILED_ADMM_UPDATE
    if _COMPILED_ADMM_ERROR is not None:
        settings["_torch_compile_admm_status"] = _COMPILED_ADMM_ERROR
        return _admm_vector_update

    try:
        _COMPILED_ADMM_UPDATE = torch.compile(_admm_vector_update)
        settings["_torch_compile_admm_status"] = "enabled"
        return _COMPILED_ADMM_UPDATE
    except Exception as exc:
        _COMPILED_ADMM_ERROR = f"fallback_compile: {type(exc).__name__}: {exc}"
        settings["_torch_compile_admm_status"] = _COMPILED_ADMM_ERROR
        return _admm_vector_update


def _admm_vector_update(x_tilde, x, z_tilde, z, y, rho_vec, l, u, alpha):
    x_next = x.clone()
    x_next.mul_(1.0 - alpha)
    x_next.add_(x_tilde, alpha=alpha)

    z_relaxed = z.clone()
    z_relaxed.mul_(1.0 - alpha)
    z_relaxed.add_(z_tilde, alpha=alpha)

    z_next = z_relaxed + y / rho_vec
    z_next = torch.maximum(torch.minimum(z_next, u), l)

    y_next = z_relaxed - z_next
    y_next.mul_(rho_vec)
    y_next.add_(y)
    return x_next, z_next, y_next


def _solve_sparse_cg_operator(operator, q, l, u, settings):
    setup_start = time.perf_counter()
    n = q.numel()
    m = l.numel()
    if operator.n != n:
        raise ValueError(f"operator has {operator.n} variables but q has {n}.")
    if operator.m != m:
        raise ValueError(f"operator has {operator.m} constraints but l has {m}.")
    if u.numel() != m:
        raise ValueError("l and u must have the same number of entries.")

    if settings.get("cuda_graph", False):
        return _solve_sparse_cg_cuda_graph(operator, q, l, u, settings)

    n_eq = int(getattr(operator, "n_eq", 0))
    rho_bar = torch.as_tensor(float(settings["rho"]), device=q.device, dtype=q.dtype)
    rho_vec = _rho_vector(settings["rho"], m, q.device, q.dtype, n_eq=n_eq)
    sigma = settings["sigma"]
    alpha = settings["alpha"]
    max_iter = settings["max_iter"]
    eps_abs = settings["eps_abs"]
    eps_rel = settings["eps_rel"]
    check_termination = settings["check_termination"]
    cg_rtol = settings.get("cg_rtol", 1e-6)
    cg_atol = settings.get("cg_atol", 0.0)
    cg_max_iter = settings.get("cg_max_iter", max(20, min(500, 2 * n)))
    cg_check_interval = settings.get("cg_check_interval", 1)
    cg_fixed_iters_requested = settings.get("cg_fixed_iters")
    cg_fixed_iters = _resolve_cg_fixed_iters(cg_fixed_iters_requested, operator, l, u)
    settings["_cg_fixed_iters_selected"] = cg_fixed_iters
    adaptive_rho = bool(settings.get("adaptive_rho", False))
    rho_update_interval = _rho_update_interval(settings, check_termination)
    rho_update_tolerance = float(settings.get("rho_update_tolerance", 5.0))
    verbose = settings["verbose"]

    x, z, y, x_tilde_warm_start = _initial_admm_state(
        settings, n, m, q.device, q.dtype
    )

    diag_M = jacobi_preconditioner_diagonal(operator, sigma, rho_vec)
    eps = torch.finfo(q.dtype).eps
    inv_diag_M = diag_M.clamp_min(eps).reciprocal()
    admm_update = _admm_update_function(settings)

    def preconditioner(residual):
        return residual * inv_diag_M

    total_cg_iterations = 0
    last_cg_info = _cg_info(False, 0, torch.inf * torch.ones((), device=q.device), torch.ones((), device=q.device), "not_started")
    status = "max_iter_reached"
    last_residuals = None
    rho_updates = 0
    timing = {
        "setup_ms": 0.0,
        "cg_ms": 0.0,
        "admm_update_ms": 0.0,
        "residual_ms": 0.0,
    }
    phase_timer = _SolverPhaseTimer(
        q.device, bool(settings.get("cuda_event_timing", False))
    )
    timing["setup_ms"] = (time.perf_counter() - setup_start) * 1000

    for iteration in range(1, max_iter + 1):
        rhs = operator.AT_mv(rho_vec * z - y)
        rhs.add_(x, alpha=sigma)
        rhs.sub_(q)

        def matvec(vector):
            return reduced_system_matvec(operator, sigma, rho_vec, vector)

        cg_start = phase_timer.start("cg_ms")
        x_tilde, cg_info = conjugate_gradient(
            matvec,
            rhs,
            x0=x_tilde_warm_start,
            preconditioner=preconditioner,
            rtol=cg_rtol,
            atol=cg_atol,
            max_iter=cg_max_iter,
            check_interval=cg_check_interval,
            fixed_iters=cg_fixed_iters,
        )
        phase_timer.stop("cg_ms", cg_start)
        x_tilde_warm_start = x_tilde
        last_cg_info = cg_info
        total_cg_iterations += cg_info["iterations"]

        z_tilde = operator.A_mv(x_tilde)
        admm_start = phase_timer.start("admm_update_ms")
        try:
            x_next, z_next, y_next = admm_update(
                x_tilde, x, z_tilde, z, y, rho_vec, l, u, alpha
            )
        except Exception as exc:
            if not settings.get("torch_compile_admm", False):
                raise
            settings["_torch_compile_admm_status"] = (
                f"fallback_runtime: {type(exc).__name__}: {exc}"
            )
            admm_update = _admm_vector_update
            x_next, z_next, y_next = admm_update(
                x_tilde, x, z_tilde, z, y, rho_vec, l, u, alpha
            )
        phase_timer.stop("admm_update_ms", admm_start)

        x = x_next
        z = z_next
        y = y_next

        should_check_termination = iteration % check_termination == 0
        should_update_rho = adaptive_rho and iteration % rho_update_interval == 0
        if should_check_termination or should_update_rho:
            residual_start = phase_timer.start("residual_ms")
            last_residuals = _operator_residuals(
                operator, q, x, z, y, eps_abs, eps_rel
            )
            primal_res, dual_res, eps_prim, eps_dual = last_residuals
            if should_update_rho:
                updated, rho_bar, rho_vec = _adaptive_rho_update(
                    rho_bar,
                    rho_vec,
                    primal_res,
                    dual_res,
                    eps_prim,
                    eps_dual,
                    rho_update_tolerance,
                    m,
                    q.device,
                    q.dtype,
                    n_eq,
                )
                if updated:
                    diag_M = jacobi_preconditioner_diagonal(operator, sigma, rho_vec)
                    inv_diag_M = diag_M.clamp_min(eps).reciprocal()
                    rho_updates += 1
            phase_timer.stop("residual_ms", residual_start)
            if should_check_termination and bool((primal_res <= eps_prim).item()) and bool(
                (dual_res <= eps_dual).item()
            ):
                status = "solved"
                if verbose:
                    print(f"Torch sparse CG OSQP converged in {iteration} iterations.")
                break
    else:
        iteration = max_iter
        if verbose:
            if last_residuals is None:
                last_residuals = _operator_residuals(
                    operator, q, x, z, y, eps_abs, eps_rel
                )
            primal_res, dual_res, eps_prim, eps_dual = last_residuals
            print(
                "Torch sparse CG OSQP reached max_iter="
                f"{max_iter} with primal={primal_res.item():.3e}/"
                f"{eps_prim.item():.3e}, dual={dual_res.item():.3e}/"
                f"{eps_dual.item():.3e}."
            )

    if not torch.all(torch.isfinite(x)):
        raise RuntimeError("Torch sparse CG OSQP returned a non-finite solution.")

    timing.update(phase_timer.totals())

    polish_info = _default_polish_info(settings)
    if settings.get("polishing", False):
        x, z, y, polish_info = _polish_solution(operator, q, l, u, x, z, y, settings)

    info = _operator_info(
        operator,
        q,
        x,
        z,
        y,
        settings,
        iteration,
        status,
        total_cg_iterations,
        last_cg_info,
        rho_updates,
        rho_bar,
        rho_vec,
        polish_info,
        timing,
    )
    if settings.get("return_state", False) or settings.get("_include_state_for_postprocess", False):
        state = _solver_state(
            x, z, y, x_tilde_warm_start, rho_vec, operator.sparse_cache()
        )
        if settings.get("_include_state_for_postprocess", False):
            info["_state"] = state
        if settings.get("return_state", False):
            info["state"] = state
    return x.reshape(n, 1), info


class _SolverPhaseTimer:
    def __init__(self, device, use_cuda_events):
        self.use_cuda_events = bool(use_cuda_events and device.type == "cuda")
        self.events = {"cg_ms": [], "admm_update_ms": [], "residual_ms": []}
        self.cpu_totals = {name: 0.0 for name in self.events}

    def start(self, name):
        if not self.use_cuda_events:
            return time.perf_counter()
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    def stop(self, name, start):
        if not self.use_cuda_events:
            self.cpu_totals[name] += (time.perf_counter() - start) * 1000.0
            return
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        self.events[name].append((start, end))

    def totals(self):
        if not self.use_cuda_events:
            return dict(self.cpu_totals)
        pending = [pair for pairs in self.events.values() for pair in pairs]
        if pending:
            pending[-1][1].synchronize()
        return {
            name: sum(start.elapsed_time(end) for start, end in pairs)
            for name, pairs in self.events.items()
        }


def _solve_sparse_cg_cuda_graph(operator, q, l, u, settings):
    _validate_cuda_graph_settings(operator, q, settings)
    settings["_cg_fixed_iters_selected"] = int(settings["cg_fixed_iters"])
    setup_start = time.perf_counter()
    n = q.numel()
    m = l.numel()
    n_eq = int(getattr(operator, "n_eq", 0))
    rho_vec = _rho_vector(settings["rho"], m, q.device, q.dtype, n_eq=n_eq)
    x, z, y, cg_x = _initial_admm_state(settings, n, m, q.device, q.dtype)
    policy = _cuda_graph_policy(operator, settings)
    previous_state = settings.get("initial_state")
    graph_state = (
        previous_state.get("cuda_graph_state")
        if isinstance(previous_state, dict)
        else None
    )
    cache_hit = bool(
        isinstance(graph_state, dict)
        and operator.cache_hit
        and graph_state.get("policy") == policy
    )
    capture_ms = 0.0
    if not cache_hit:
        capture_start = time.perf_counter()
        graph_state = _capture_sparse_cg_cuda_graph(
            operator, q, l, u, x, z, y, cg_x, rho_vec, settings, policy
        )
        capture_ms = (time.perf_counter() - capture_start) * 1000.0

    _load_cuda_graph_inputs(
        graph_state, operator, q, l, u, x, z, y, cg_x, rho_vec
    )
    replay_start = torch.cuda.Event(enable_timing=True)
    replay_end = torch.cuda.Event(enable_timing=True)
    replay_start.record()
    graph_state["graph"].replay()
    replay_end.record()
    replay_end.synchronize()
    replay_ms = replay_start.elapsed_time(replay_end)

    x, z, y, cg_x = graph_state["outputs"]
    residuals = _operator_residuals(
        operator, q, x, z, y, settings["eps_abs"], settings["eps_rel"]
    )
    primal_res, dual_res, eps_prim, eps_dual = residuals
    solved = bool((primal_res <= eps_prim).item()) and bool(
        (dual_res <= eps_dual).item()
    )
    status = "solved" if solved else "max_iter_reached"
    if not bool(torch.all(torch.isfinite(x)).item()):
        raise RuntimeError("Torch CUDA Graph sparse-CG returned a non-finite solution.")

    fixed_cg = int(settings["cg_fixed_iters"])
    max_iter = int(settings["max_iter"])
    last_cg_info = {
        "converged": False,
        "iterations": fixed_cg,
        "residual_norm": None,
        "relative_residual": None,
        "status": "fixed_iters_cuda_graph",
    }
    timing = {
        "setup_ms": (time.perf_counter() - setup_start) * 1000.0,
        "cg_ms": 0.0,
        "admm_update_ms": 0.0,
        "residual_ms": 0.0,
        "cuda_graph_capture_ms": capture_ms,
        "cuda_graph_replay_ms": replay_ms,
    }
    info = _operator_info(
        operator,
        q,
        x,
        z,
        y,
        settings,
        max_iter,
        status,
        max_iter * fixed_cg,
        last_cg_info,
        0,
        torch.as_tensor(float(settings["rho"]), device=q.device, dtype=q.dtype),
        rho_vec,
        _default_polish_info(settings),
        timing,
    )
    info.update(
        {
            "cuda_graph": True,
            "cuda_graph_status": "replayed" if cache_hit else "captured",
            "cuda_graph_cache_hit": cache_hit,
            "cuda_graph_recaptures": 0 if cache_hit else 1,
            "cuda_graph_eligibility_reason": "fixed_work_sparse_cg",
            "timing_mode": "cuda_graph_events",
        }
    )
    if settings.get("return_state", False) or settings.get(
        "_include_state_for_postprocess", False
    ):
        state = _solver_state(
            x,
            z,
            y,
            cg_x,
            rho_vec,
            operator.sparse_cache(),
            cuda_graph_state=graph_state,
        )
        if settings.get("_include_state_for_postprocess", False):
            info["_state"] = state
        if settings.get("return_state", False):
            info["state"] = state
    return x.reshape(n, 1).clone(), info


def _validate_cuda_graph_settings(operator, q, settings):
    if q.device.type != "cuda":
        raise ValueError("Torch OSQP setting 'cuda_graph=True' requires a CUDA tensor.")
    fixed_iters = settings.get("cg_fixed_iters")
    if not isinstance(fixed_iters, int) or isinstance(fixed_iters, bool) or fixed_iters <= 0:
        raise ValueError(
            "Torch OSQP setting 'cuda_graph=True' requires a positive integer "
            "cg_fixed_iters."
        )
    if settings.get("adaptive_rho", False):
        raise ValueError("cuda_graph=True does not support adaptive_rho.")
    if int(settings.get("scaling", 0) or 0) != 0:
        raise ValueError("cuda_graph=True currently requires scaling=0.")
    if settings.get("polishing", False):
        raise ValueError("cuda_graph=True currently requires polishing=False.")
    if settings.get("torch_compile_admm", False):
        raise ValueError("cuda_graph=True and torch_compile_admm cannot be combined.")
    if int(settings["check_termination"]) < int(settings["max_iter"]):
        raise ValueError(
            "cuda_graph=True requires check_termination >= max_iter so termination "
            "is checked once after graph replay."
        )
    if not isinstance(operator, (ExplicitOSQPOperator, BoundConstrainedOSQPOperator)):
        raise ValueError("cuda_graph=True requires a supported sparse OSQP operator.")


def _cuda_graph_policy(operator, settings):
    return (
        type(operator).__name__,
        tuple(operator.P.shape),
        None
        if getattr(operator, "A_eq", None) is None
        else tuple(operator.A_eq.shape),
        None if getattr(operator, "A", None) is None else tuple(operator.A.shape),
        str(operator.device),
        str(operator.dtype),
        int(settings["max_iter"]),
        int(settings["cg_fixed_iters"]),
        float(settings["rho"]),
        float(settings["sigma"]),
        float(settings["alpha"]),
    )


def _capture_sparse_cg_cuda_graph(
    operator, q, l, u, x, z, y, cg_x, rho_vec, settings, policy
):
    static_operator = _clone_operator_for_cuda_graph(operator)
    graph_state = {
        "policy": policy,
        "operator": static_operator,
        "q": q.clone(),
        "l": l.clone(),
        "u": u.clone(),
        "x": x.clone(),
        "z": z.clone(),
        "y": y.clone(),
        "cg_x": cg_x.clone(),
        "rho_vec": rho_vec.clone(),
        "breakdown_eps": torch.full(
            (), torch.finfo(q.dtype).eps, device=q.device, dtype=q.dtype
        ),
    }
    p_rows = _csr_row_indices(static_operator.P)
    p_diag_positions = torch.nonzero(
        p_rows == static_operator.P.col_indices(), as_tuple=False
    ).reshape(-1)
    graph_state["p_diag_positions"] = p_diag_positions
    graph_state["p_diag_rows"] = p_rows[p_diag_positions]

    def workload():
        return _cuda_graph_fixed_admm(graph_state, settings)

    warmup_stream = torch.cuda.Stream(device=q.device)
    warmup_stream.wait_stream(torch.cuda.current_stream(q.device))
    with torch.cuda.stream(warmup_stream):
        workload()
    torch.cuda.current_stream(q.device).wait_stream(warmup_stream)
    torch.cuda.current_stream(q.device).synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = workload()
    graph_state["graph"] = graph
    graph_state["outputs"] = outputs
    return graph_state


def _clone_operator_for_cuda_graph(operator):
    P = _clone_csr(operator.P)
    if isinstance(operator, ExplicitOSQPOperator):
        A = _clone_csr(operator.A)
        return ExplicitOSQPOperator(
            P, A, operator.device, operator.dtype, cache=operator.sparse_cache()
        )
    A_eq = None if operator.A_eq is None else _clone_csr(operator.A_eq)
    return BoundConstrainedOSQPOperator(
        P,
        A_eq,
        operator.n,
        operator.device,
        operator.dtype,
        cache=operator.sparse_cache(),
    )


def _clone_csr(matrix):
    return _csr_with_values(
        matrix.crow_indices().detach().clone(),
        matrix.col_indices().detach().clone(),
        matrix.values().detach().clone(),
        tuple(matrix.shape),
    )


def _load_cuda_graph_inputs(
    graph_state, operator, q, l, u, x, z, y, cg_x, rho_vec
):
    static_operator = graph_state["operator"]
    static_operator.P.values().copy_(operator.P.values())
    if isinstance(static_operator, ExplicitOSQPOperator):
        static_operator.A.values().copy_(operator.A.values())
    elif static_operator.A_eq is not None:
        static_operator.A_eq.values().copy_(operator.A_eq.values())
    graph_state["q"].copy_(q)
    graph_state["l"].copy_(l)
    graph_state["u"].copy_(u)
    graph_state["x"].copy_(x)
    graph_state["z"].copy_(z)
    graph_state["y"].copy_(y)
    graph_state["cg_x"].copy_(cg_x)
    graph_state["rho_vec"].copy_(rho_vec)


def _cuda_graph_fixed_admm(graph_state, settings):
    operator = graph_state["operator"]
    q = graph_state["q"]
    l = graph_state["l"]
    u = graph_state["u"]
    rho_vec = graph_state["rho_vec"]
    x = graph_state["x"]
    z = graph_state["z"]
    y = graph_state["y"]
    cg_x = graph_state["cg_x"]
    sigma = float(settings["sigma"])
    alpha = float(settings["alpha"])
    fixed_cg = int(settings["cg_fixed_iters"])

    operator.refresh_transpose_values()
    diag_M = _graphsafe_sparse_diagonal(
        operator.P,
        graph_state["p_diag_rows"],
        graph_state["p_diag_positions"],
    ) + sigma
    diag_M = diag_M + operator.diag_ATRA(rho_vec)
    inv_diag_M = diag_M.clamp_min(graph_state["breakdown_eps"]).reciprocal()
    for _ in range(int(settings["max_iter"])):
        rhs = operator.AT_mv(rho_vec * z - y)
        rhs = rhs + sigma * x - q
        cg_x = _fixed_cg_graphsafe(
            operator,
            rhs,
            cg_x,
            inv_diag_M,
            sigma,
            rho_vec,
            fixed_cg,
            graph_state["breakdown_eps"],
        )
        z_tilde = operator.A_mv(cg_x)
        x, z, y = _admm_vector_update(
            cg_x, x, z_tilde, z, y, rho_vec, l, u, alpha
        )
    return x, z, y, cg_x


def _fixed_cg_graphsafe(
    operator, b, x, inv_diag_M, sigma, rho_vec, iterations, breakdown_eps
):
    r = b - reduced_system_matvec(operator, sigma, rho_vec, x)
    z = r * inv_diag_M
    p = z.clone()
    rz_old = torch.dot(r, z)
    for _ in range(iterations):
        Ap = reduced_system_matvec(operator, sigma, rho_vec, p)
        denom = torch.dot(p, Ap)
        denom_safe = torch.where(torch.abs(denom) <= breakdown_eps, breakdown_eps, denom)
        alpha = rz_old / denom_safe
        x = x + alpha * p
        r = r - alpha * Ap
        z = r * inv_diag_M
        rz_new = torch.dot(r, z)
        rz_old_safe = torch.where(
            torch.abs(rz_old) <= breakdown_eps, breakdown_eps, rz_old
        )
        p = z + (rz_new / rz_old_safe) * p
        rz_old = rz_new
    return x


def _graphsafe_sparse_diagonal(matrix, diagonal_rows, diagonal_positions):
    values = matrix.values()
    diag = torch.zeros(matrix.shape[0], device=values.device, dtype=values.dtype)
    return diag.scatter_add(0, diagonal_rows, values[diagonal_positions])


def _as_sparse_csr(value, name):
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a Torch tensor for the Torch OSQP backend.")
    tensor = value.detach()
    if tensor.layout == torch.sparse_csr:
        return tensor
    if tensor.layout == torch.strided:
        return tensor.to_sparse_csr()
    if tensor.layout == torch.sparse_coo:
        return tensor.coalesce().to_sparse_csr()
    if tensor.layout in SPARSE_LAYOUTS:
        return _to_coalesced_coo(tensor).to_sparse_csr()
    raise TypeError(f"{name} has unsupported tensor layout {tensor.layout}.")


def _to_coalesced_coo(matrix):
    if matrix.layout == torch.sparse_coo:
        return matrix.coalesce()
    return matrix.to_sparse_coo().coalesce()


def _transpose_to_csr(matrix):
    coo = _to_coalesced_coo(matrix)
    indices = coo.indices()
    transposed = torch.sparse_coo_tensor(
        torch.stack((indices[1], indices[0]), dim=0),
        coo.values(),
        (coo.shape[1], coo.shape[0]),
        device=coo.device,
        dtype=coo.dtype,
    ).coalesce()
    return transposed.to_sparse_csr()


def _csr_row_indices(matrix):
    counts = matrix.crow_indices()[1:] - matrix.crow_indices()[:-1]
    return torch.repeat_interleave(
        torch.arange(matrix.shape[0], device=matrix.device), counts
    )


def _csc_col_indices(matrix):
    counts = matrix.ccol_indices()[1:] - matrix.ccol_indices()[:-1]
    return torch.repeat_interleave(
        torch.arange(matrix.shape[1], device=matrix.device), counts
    )


def _rho_vector(rho, m, device, dtype, n_eq=0, equality_rho_scale=1000.0):
    if torch.is_tensor(rho):
        rho_vec = rho.detach().to(device=device, dtype=dtype).reshape(-1)
        if rho_vec.numel() == 1:
            rho_vec = rho_vec.expand(m)
        if rho_vec.numel() != m:
            raise ValueError("rho tensor must be scalar or have one entry per constraint.")
        if bool(torch.any(rho_vec <= 0).item()):
            raise ValueError("rho entries must be positive.")
        if rho_vec.numel() == 1 or rho.detach().reshape(-1).numel() == 1:
            rho_vec = rho_vec.clone()
            rho_vec[: int(n_eq)] = rho_vec[: int(n_eq)] * float(equality_rho_scale)
        return rho_vec
    rho_vec = torch.full((m,), float(rho), device=device, dtype=dtype)
    if int(n_eq) > 0:
        rho_vec[: int(n_eq)] = float(rho) * float(equality_rho_scale)
    return rho_vec


def _rho_update_interval(settings, check_termination):
    interval = settings.get("rho_update_interval", "auto")
    if interval == "auto":
        return max(1, min(int(check_termination), 10))
    return max(1, int(interval))


def _adaptive_rho_update(
    rho_bar,
    rho_vec,
    primal_res,
    dual_res,
    eps_prim,
    eps_dual,
    tolerance,
    m,
    device,
    dtype,
    n_eq,
):
    tiny = torch.as_tensor(torch.finfo(dtype).tiny, device=device, dtype=dtype)
    prim_ratio = primal_res / eps_prim.clamp_min(tiny)
    dual_ratio = dual_res / eps_dual.clamp_min(tiny)
    ratio = prim_ratio / dual_ratio.clamp_min(tiny)
    update_needed = (ratio > tolerance) | (ratio < 1.0 / tolerance)
    if not bool(update_needed.item()):
        return False, rho_bar, rho_vec
    multiplier = torch.sqrt(ratio).clamp(0.1, 10.0)
    rho_bar = (rho_bar * multiplier).clamp_min(tiny)
    rho_vec = _rho_vector(float(rho_bar.item()), m, device, dtype, n_eq=n_eq)
    return True, rho_bar, rho_vec


def _initial_admm_state(settings, n, m, device, dtype):
    x = torch.zeros(n, device=device, dtype=dtype)
    z = torch.zeros(m, device=device, dtype=dtype)
    y = torch.zeros(m, device=device, dtype=dtype)
    cg_x = torch.zeros_like(x)
    if not settings.get("warm_start", False):
        return x, z, y, cg_x

    state = settings.get("initial_state")
    if not isinstance(state, dict):
        return x, z, y, cg_x
    x = _state_vector(state.get("x"), n, x, device, dtype)
    z = _state_vector(state.get("z"), m, z, device, dtype)
    y = _state_vector(state.get("y"), m, y, device, dtype)
    cg_x = _state_vector(state.get("cg_x", state.get("x_tilde", x)), n, x, device, dtype)
    return x, z, y, cg_x


def _initial_sparse_cache(settings):
    if not settings.get("warm_start", False):
        return None
    state = settings.get("initial_state")
    if not isinstance(state, dict):
        return None
    return state.get("sparse_cache")


def _resolve_cg_fixed_iters(value, operator, l, u):
    if value != "auto":
        return value
    n_eq = int(getattr(operator, "n_eq", 0))
    if n_eq > 0:
        return None
    if not getattr(operator, "uses_matrix_free_bounds", False):
        equality_rows = torch.isclose(l, u, rtol=1e-9, atol=1e-12)
        if bool(torch.any(equality_rows).item()):
            return None
    return 1


def _state_vector(value, expected, fallback, device, dtype):
    if value is None:
        return fallback.clone()
    if not torch.is_tensor(value):
        return fallback.clone()
    vector = value.detach().to(device=device, dtype=dtype).reshape(-1)
    if vector.numel() != expected:
        return fallback.clone()
    return vector.clone()


def _solver_state(
    x, z, y, cg_x, rho_vec, sparse_cache=None, cuda_graph_state=None
):
    state = {
        "x": x.detach().clone(),
        "z": z.detach().clone(),
        "y": y.detach().clone(),
        "cg_x": cg_x.detach().clone(),
        "rho": rho_vec.detach().clone(),
        "sparse_cache": sparse_cache,
    }
    if cuda_graph_state is not None:
        state["cuda_graph_state"] = cuda_graph_state
    return state


def _compatible_sparse_cache(cache, kind, P, A):
    if not isinstance(cache, dict):
        return None
    if cache.get("kind") != kind:
        return None
    if cache.get("device") != str(P.device) or cache.get("dtype") != str(P.dtype):
        return None
    if cache.get("P_shape") != tuple(P.shape) or cache.get("P_nnz") != _nnz(P):
        return None
    a_shape = None if A is None else tuple(A.shape)
    a_nnz = 0 if A is None else _nnz(A)
    if cache.get("A_shape") != a_shape or cache.get("A_nnz") != a_nnz:
        return None
    if not _cached_structure_matches(cache, "P", P):
        return None
    if A is not None and not _cached_structure_matches(cache, "A", A):
        return None
    return cache


def _cached_structure(prefix, matrix):
    return {
        f"{prefix}_crow_indices": matrix.crow_indices().detach().clone(),
        f"{prefix}_col_indices": matrix.col_indices().detach().clone(),
    }


def _cached_structure_matches(cache, prefix, matrix):
    cached_crow = cache.get(f"{prefix}_crow_indices")
    cached_cols = cache.get(f"{prefix}_col_indices")
    if not torch.is_tensor(cached_crow) or not torch.is_tensor(cached_cols):
        return False
    return torch.equal(cached_crow, matrix.crow_indices()) and torch.equal(
        cached_cols, matrix.col_indices()
    )


def _transpose_structure(matrix, cache, prefix):
    if cache is not None:
        crow = cache.get(f"{prefix}_crow_indices")
        cols = cache.get(f"{prefix}_col_indices")
        value_map = cache.get(f"{prefix}_value_map")
        if torch.is_tensor(crow) and torch.is_tensor(cols) and torch.is_tensor(value_map):
            return {
                "crow_indices": crow,
                "col_indices": cols,
                "value_map": value_map,
            }

    transpose = _transpose_to_csr(matrix)
    source_rows = _csr_row_indices(matrix)
    source_cols = matrix.col_indices()
    transpose_rows = _csr_row_indices(transpose)
    transpose_cols = transpose.col_indices()
    source_keys = source_rows * matrix.shape[1] + source_cols
    transpose_source_keys = transpose_cols * matrix.shape[1] + transpose_rows
    sorted_keys, sorted_positions = torch.sort(source_keys)
    value_map = sorted_positions[torch.searchsorted(sorted_keys, transpose_source_keys)]
    return {
        "crow_indices": transpose.crow_indices().detach().clone(),
        "col_indices": transpose.col_indices().detach().clone(),
        "value_map": value_map.detach(),
    }


def _csr_with_values(crow_indices, col_indices, values, shape):
    return torch.sparse_csr_tensor(
        crow_indices,
        col_indices,
        values,
        size=shape,
        device=values.device,
        dtype=values.dtype,
        check_invariants=False,
    )


def _ruiz_scale_qp_data(P, q, A_eq, b, LB, UB, settings):
    passes = int(settings.get("scaling", 0) or 0)
    if passes <= 0:
        return None

    n = q.numel()
    device = q.device
    dtype = q.dtype
    tiny = torch.as_tensor(torch.finfo(dtype).tiny, device=device, dtype=dtype)
    D = torch.ones(n, device=device, dtype=dtype)
    E_eq = torch.ones(0 if A_eq is None else A_eq.shape[0], device=device, dtype=dtype)
    P_s = P
    q_s = q.clone()
    A_s = A_eq
    b_s = None if b is None else b.clone()
    LB_s = LB.clone()
    UB_s = UB.clone()

    for _ in range(passes):
        p_row = _sparse_axis_abs_sum(P_s, 0)
        p_col = _sparse_axis_abs_sum(P_s, 1)
        a_col = (
            torch.zeros(n, device=device, dtype=dtype)
            if A_s is None
            else _sparse_axis_abs_sum(A_s, 1)
        )
        var_norm = torch.maximum(torch.maximum(p_row, p_col), a_col).clamp_min(tiny)
        D_step = torch.rsqrt(var_norm).clamp(0.1, 10.0)

        if A_s is None:
            E_step = E_eq
        else:
            row_norm = _sparse_axis_abs_sum(A_s, 0).clamp_min(tiny)
            E_step = torch.rsqrt(row_norm).clamp(0.1, 10.0)

        P_s = _scale_sparse_csr_rows_cols(P_s, D_step, D_step)
        q_s = q_s * D_step
        if A_s is not None:
            A_s = _scale_sparse_csr_rows_cols(A_s, E_step, D_step)
            b_s = b_s * E_step
            E_eq = E_eq * E_step
        LB_s = LB_s / D_step
        UB_s = UB_s / D_step
        D = D * D_step

    cost_scale = _cost_scale(P_s, q_s)
    P_s = _scale_sparse_csr_values(P_s, cost_scale)
    q_s = q_s * cost_scale
    E_full = torch.cat((E_eq, 1.0 / D), dim=0)
    return {
        "kind": "modified_ruiz",
        "passes": passes,
        "D": D,
        "E": E_full,
        "cost_scale": cost_scale,
        "P": P_s,
        "q": q_s,
        "A_eq": A_s,
        "b": b_s,
        "LB": LB_s,
        "UB": UB_s,
    }


def _ruiz_scale_osqp_data(P, q, A, l, u, settings):
    passes = int(settings.get("scaling", 0) or 0)
    if passes <= 0:
        return None

    n = q.numel()
    m = l.numel()
    device = q.device
    dtype = q.dtype
    tiny = torch.as_tensor(torch.finfo(dtype).tiny, device=device, dtype=dtype)
    D = torch.ones(n, device=device, dtype=dtype)
    E = torch.ones(m, device=device, dtype=dtype)
    P_s = P
    A_s = A
    q_s = q.clone()
    l_s = l.clone()
    u_s = u.clone()

    for _ in range(passes):
        p_row = _sparse_axis_abs_sum(P_s, 0)
        p_col = _sparse_axis_abs_sum(P_s, 1)
        a_col = _sparse_axis_abs_sum(A_s, 1)
        var_norm = torch.maximum(torch.maximum(p_row, p_col), a_col).clamp_min(tiny)
        row_norm = _sparse_axis_abs_sum(A_s, 0).clamp_min(tiny)
        D_step = torch.rsqrt(var_norm).clamp(0.1, 10.0)
        E_step = torch.rsqrt(row_norm).clamp(0.1, 10.0)

        P_s = _scale_sparse_csr_rows_cols(P_s, D_step, D_step)
        A_s = _scale_sparse_csr_rows_cols(A_s, E_step, D_step)
        q_s = q_s * D_step
        l_s = l_s * E_step
        u_s = u_s * E_step
        D = D * D_step
        E = E * E_step

    cost_scale = _cost_scale(P_s, q_s)
    P_s = _scale_sparse_csr_values(P_s, cost_scale)
    q_s = q_s * cost_scale
    return {
        "kind": "modified_ruiz",
        "passes": passes,
        "D": D,
        "E": E,
        "cost_scale": cost_scale,
        "P": P_s,
        "q": q_s,
        "A": A_s,
        "l": l_s,
        "u": u_s,
    }


def _cost_scale(P, q):
    one = torch.ones((), device=q.device, dtype=q.dtype)
    norm = torch.maximum(_sparse_abs_max(P), torch.linalg.vector_norm(q, ord=float("inf")))
    return one / torch.maximum(norm, one)


def _scale_sparse_csr_values(matrix, value_scale):
    coo = _to_coalesced_coo(matrix)
    return torch.sparse_coo_tensor(
        coo.indices(),
        coo.values() * value_scale,
        coo.shape,
        device=coo.device,
        dtype=coo.dtype,
    ).coalesce().to_sparse_csr()


def _scale_sparse_csr_rows_cols(matrix, row_scale, col_scale):
    coo = _to_coalesced_coo(matrix)
    indices = coo.indices()
    values = coo.values() * row_scale[indices[0]] * col_scale[indices[1]]
    return torch.sparse_coo_tensor(
        indices,
        values,
        coo.shape,
        device=coo.device,
        dtype=coo.dtype,
    ).coalesce().to_sparse_csr()


def _sparse_axis_abs_sum(matrix, axis):
    coo = _to_coalesced_coo(matrix)
    length = matrix.shape[axis]
    out = torch.zeros(length, device=coo.device, dtype=coo.dtype)
    if coo.values().numel() == 0:
        return out
    index = coo.indices()[axis]
    out.scatter_add_(0, index, coo.values().abs())
    return out


def _sparse_abs_max(matrix):
    coo = _to_coalesced_coo(matrix)
    if coo.values().numel() == 0:
        return torch.zeros((), device=coo.device, dtype=coo.dtype)
    return torch.max(coo.values().abs())


def _unscale_sparse_cg_result(solution, info, scaling, operator, q, l, u, settings):
    x_scaled = solution.reshape(-1)
    x = scaling["D"] * x_scaled
    state = info.pop("_state", None)

    z = operator.A_mv(x)
    y = torch.zeros_like(z)
    if isinstance(state, dict):
        z = state["z"] / scaling["E"]
        y = (scaling["E"] / scaling["cost_scale"]) * state["y"]

    primal_res, dual_res, eps_prim, eps_dual = _operator_residuals(
        operator, q, x, z, y, settings["eps_abs"], settings["eps_rel"]
    )
    objective = 0.5 * torch.dot(x, operator.P_mv(x)) + torch.dot(q, x)

    info.update(
        {
            "scaled_primal_residual": info.get("primal_residual"),
            "scaled_dual_residual": info.get("dual_residual"),
            "scaled_objective": info.get("objective"),
            "primal_residual": float(primal_res.item()),
            "dual_residual": float(dual_res.item()),
            "eps_primal": float(eps_prim.item()),
            "eps_dual": float(eps_dual.item()),
            "objective": float(objective.item()),
            "scaling_applied": True,
            "scaling_kind": scaling["kind"],
            "scaling_passes": int(scaling["passes"]),
            "ruiz_cost_scale": float(scaling["cost_scale"].item()),
        }
    )
    if settings.get("return_state", False):
        cg_x = state["cg_x"] if isinstance(state, dict) else x_scaled
        info["state"] = {
            "x": x.detach().clone(),
            "z": z.detach().clone(),
            "y": y.detach().clone(),
            "cg_x": (scaling["D"] * cg_x).detach().clone(),
            "rho": state["rho"].detach().clone() if isinstance(state, dict) else None,
            "sparse_cache": state.get("sparse_cache") if isinstance(state, dict) else None,
        }
    else:
        info.pop("state", None)
    return x.reshape(-1, 1), info


def _default_polish_info(settings):
    enabled = bool(settings.get("polishing", False))
    return {
        "polishing": enabled,
        "polishing_success": False,
        "polishing_status": "disabled" if not enabled else "not_run",
        "polishing_time_ms": 0.0,
        "polishing_active_constraints": 0,
        "polishing_lower_active": 0,
        "polishing_upper_active": 0,
        "polishing_refine_iter": int(settings.get("polish_refine_iter", 0) or 0),
    }


def _polish_solution(operator, q, l, u, x, z, y, settings):
    info = _default_polish_info(settings)
    start = time.perf_counter()
    _sync_if_cuda(q.device)
    try:
        A_active, rhs_active, assignment = _active_constraint_system(operator, y, l, u)
        info["polishing_active_constraints"] = int(rhs_active.numel())
        info["polishing_lower_active"] = int(assignment["lower_count"])
        info["polishing_upper_active"] = int(assignment["upper_count"])
        if rhs_active.numel() == 0:
            info["polishing_status"] = "skipped_no_active_constraints"
            return x, z, y, _finish_polish_timing(info, q.device, start)

        delta = float(settings.get("polish_delta", 1e-6))
        refine_iter = int(settings.get("polish_refine_iter", 0) or 0)
        P_dense = operator.P.to_dense()
        eye_n = torch.eye(operator.n, device=q.device, dtype=q.dtype)
        eye_m = torch.eye(rhs_active.numel(), device=q.device, dtype=q.dtype)
        top = torch.cat((P_dense + delta * eye_n, A_active.T), dim=1)
        bottom = torch.cat(
            (A_active, -delta * eye_m),
            dim=1,
        )
        K = torch.cat((top, bottom), dim=0)
        rhs = torch.cat((-q, rhs_active), dim=0)
        polished = torch.linalg.solve(K, rhs)
        for _ in range(refine_iter):
            correction = torch.linalg.solve(K, rhs - K @ polished)
            polished = polished + correction

        x_candidate = polished[: operator.n]
        active_dual = polished[operator.n :]
        y_candidate = _scatter_active_dual(active_dual, assignment, y)
        z_candidate = torch.maximum(torch.minimum(operator.A_mv(x_candidate), u), l)

        old_metric = _kkt_metric(operator, q, x, z, y, settings)
        new_metric = _kkt_metric(
            operator, q, x_candidate, z_candidate, y_candidate, settings
        )
        improved = bool((new_metric["score"] <= old_metric["score"]).item())
        satisfies = bool(
            (
                (new_metric["primal"] <= new_metric["eps_primal"])
                & (new_metric["dual"] <= new_metric["eps_dual"])
            ).item()
        )
        if improved or satisfies:
            info["polishing_success"] = True
            info["polishing_status"] = "accepted"
            info["polishing_old_score"] = float(old_metric["score"].item())
            info["polishing_new_score"] = float(new_metric["score"].item())
            return (
                x_candidate,
                z_candidate,
                y_candidate,
                _finish_polish_timing(info, q.device, start),
            )

        info["polishing_status"] = "rejected_no_improvement"
        info["polishing_old_score"] = float(old_metric["score"].item())
        info["polishing_new_score"] = float(new_metric["score"].item())
        return x, z, y, _finish_polish_timing(info, q.device, start)
    except Exception as exc:
        info["polishing_status"] = f"failed: {type(exc).__name__}: {exc}"
        return x, z, y, _finish_polish_timing(info, q.device, start)


def _finish_polish_timing(info, device, start):
    _sync_if_cuda(device)
    info["polishing_time_ms"] = (time.perf_counter() - start) * 1000
    return info


def _sync_if_cuda(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def _active_constraint_system(operator, y, l, u):
    if isinstance(operator, BoundConstrainedOSQPOperator):
        return _bound_active_constraint_system(operator, y, l, u)
    return _explicit_active_constraint_system(operator, y, l, u)


def _bound_active_constraint_system(operator, y, l, u):
    parts = []
    rhs_parts = []
    assignments = []
    n_eq = operator.n_eq
    device = y.device
    dtype = y.dtype

    if n_eq > 0:
        eq_dense = operator.A_eq.to_dense()
        eq_idx = torch.arange(n_eq, device=device)
        parts.append(eq_dense)
        rhs_parts.append(0.5 * (l[:n_eq] + u[:n_eq]))
        assignments.append(("eq", eq_idx, n_eq))

    bound_y = y[n_eq:]
    lower_idx = torch.nonzero(bound_y < 0, as_tuple=False).reshape(-1)
    upper_idx = torch.nonzero(bound_y > 0, as_tuple=False).reshape(-1)
    if lower_idx.numel() > 0:
        lower_rows = torch.zeros((lower_idx.numel(), operator.n), device=device, dtype=dtype)
        lower_rows[torch.arange(lower_idx.numel(), device=device), lower_idx] = 1.0
        parts.append(lower_rows)
        rhs_parts.append(l[n_eq:][lower_idx])
        assignments.append(("lower", lower_idx + n_eq, lower_idx.numel()))
    if upper_idx.numel() > 0:
        upper_rows = torch.zeros((upper_idx.numel(), operator.n), device=device, dtype=dtype)
        upper_rows[torch.arange(upper_idx.numel(), device=device), upper_idx] = 1.0
        parts.append(upper_rows)
        rhs_parts.append(u[n_eq:][upper_idx])
        assignments.append(("upper", upper_idx + n_eq, upper_idx.numel()))

    return _active_result(parts, rhs_parts, assignments, operator.n, device, dtype)


def _explicit_active_constraint_system(operator, y, l, u):
    device = y.device
    dtype = y.dtype
    A_dense = operator.A.to_dense()
    equality = torch.isclose(l, u, rtol=1e-9, atol=1e-12)
    lower = (y < 0) & ~equality
    upper = (y > 0) & ~equality
    parts = []
    rhs_parts = []
    assignments = []

    eq_idx = torch.nonzero(equality, as_tuple=False).reshape(-1)
    lower_idx = torch.nonzero(lower, as_tuple=False).reshape(-1)
    upper_idx = torch.nonzero(upper, as_tuple=False).reshape(-1)
    if eq_idx.numel() > 0:
        parts.append(A_dense[eq_idx])
        rhs_parts.append(0.5 * (l[eq_idx] + u[eq_idx]))
        assignments.append(("eq", eq_idx, eq_idx.numel()))
    if lower_idx.numel() > 0:
        parts.append(A_dense[lower_idx])
        rhs_parts.append(l[lower_idx])
        assignments.append(("lower", lower_idx, lower_idx.numel()))
    if upper_idx.numel() > 0:
        parts.append(A_dense[upper_idx])
        rhs_parts.append(u[upper_idx])
        assignments.append(("upper", upper_idx, upper_idx.numel()))

    return _active_result(parts, rhs_parts, assignments, operator.n, device, dtype)


def _active_result(parts, rhs_parts, assignments, n, device, dtype):
    if not parts:
        A_active = torch.empty((0, n), device=device, dtype=dtype)
        rhs = torch.empty(0, device=device, dtype=dtype)
    else:
        A_active = torch.cat(parts, dim=0)
        rhs = torch.cat(rhs_parts, dim=0)
    lower_count = sum(int(count) for kind, _idx, count in assignments if kind == "lower")
    upper_count = sum(int(count) for kind, _idx, count in assignments if kind == "upper")
    return A_active, rhs, {
        "assignments": assignments,
        "lower_count": lower_count,
        "upper_count": upper_count,
    }


def _scatter_active_dual(active_dual, assignment, y_template):
    y = torch.zeros_like(y_template)
    offset = 0
    for _kind, indices, count in assignment["assignments"]:
        y[indices] = active_dual[offset : offset + count]
        offset += count
    return y


def _kkt_metric(operator, q, x, z, y, settings):
    primal, dual, eps_primal, eps_dual = _operator_residuals(
        operator, q, x, z, y, settings["eps_abs"], settings["eps_rel"]
    )
    tiny = torch.as_tensor(torch.finfo(q.dtype).tiny, device=q.device, dtype=q.dtype)
    score = torch.maximum(primal / eps_primal.clamp_min(tiny), dual / eps_dual.clamp_min(tiny))
    return {
        "primal": primal,
        "dual": dual,
        "eps_primal": eps_primal,
        "eps_dual": eps_dual,
        "score": score,
    }


def _detached_vector(value, name, device=None, dtype=None):
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a Torch tensor for the Torch OSQP backend.")
    tensor = value.detach()
    if device is not None and tensor.device != device:
        tensor = tensor.to(device=device)
    if dtype is not None and tensor.dtype != dtype:
        tensor = tensor.to(dtype=dtype)
    return tensor.reshape(-1)


def _dense_detached(value):
    value = value.detach()
    if value.layout != torch.strided:
        return value.to_dense()
    return value


def _dense_residuals(P, q, A, x, z, y, eps_abs, eps_rel):
    Ax = A @ x
    Px = P @ x
    ATy = A.T @ y
    return _residual_values(Px, q, Ax, ATy, z, eps_abs, eps_rel)


def _operator_residuals(operator, q, x, z, y, eps_abs, eps_rel):
    Ax = operator.A_mv(x)
    Px = operator.P_mv(x)
    ATy = operator.AT_mv(y)
    return _residual_values(Px, q, Ax, ATy, z, eps_abs, eps_rel)


def _residual_values(Px, q, Ax, ATy, z, eps_abs, eps_rel):
    primal_res = torch.linalg.vector_norm(Ax - z, ord=float("inf"))
    dual_res = torch.linalg.vector_norm(Px + q + ATy, ord=float("inf"))

    eps_prim = eps_abs + eps_rel * torch.maximum(
        torch.linalg.vector_norm(Ax, ord=float("inf")),
        torch.linalg.vector_norm(z, ord=float("inf")),
    )
    eps_dual = eps_abs + eps_rel * torch.maximum(
        torch.maximum(
            torch.linalg.vector_norm(Px, ord=float("inf")),
            torch.linalg.vector_norm(ATy, ord=float("inf")),
        ),
        torch.linalg.vector_norm(q, ord=float("inf")),
    )
    return primal_res, dual_res, eps_prim, eps_dual


def _dense_info(P, q, A, x, z, y, settings, iteration, status):
    primal_res, dual_res, eps_prim, eps_dual = _dense_residuals(
        P, q, A, x, z, y, settings["eps_abs"], settings["eps_rel"]
    )
    objective = 0.5 * torch.dot(x, P @ x) + torch.dot(q, x)
    return {
        "status": status,
        "admm_iterations": iteration,
        "primal_residual": float(primal_res.item()),
        "dual_residual": float(dual_res.item()),
        "eps_primal": float(eps_prim.item()),
        "eps_dual": float(eps_dual.item()),
        "objective": float(objective.item()),
        "linear_solver": "dense",
        "linear_solver_requested": settings.get(
            "linear_solver_requested", settings.get("linear_solver", "dense")
        ),
        "linear_solver_selected": settings.get("linear_solver_selected", "dense"),
        "linear_solver_auto_reason": settings.get(
            "linear_solver_auto_reason", "explicit_dense"
        ),
        "estimated_kkt_dim": settings.get("estimated_kkt_dim"),
        "estimated_dense_kkt_mb": settings.get("estimated_dense_kkt_mb"),
        "estimated_sparse_nnz": settings.get("estimated_sparse_nnz"),
        "total_cg_iterations": 0,
        "average_cg_iterations": 0.0,
        "last_cg_relative_residual": None,
        "last_cg_residual_norm": None,
        "last_cg_converged": None,
        "last_cg_status": None,
        "cg_check_interval": settings.get("cg_check_interval", 1),
        "cg_fixed_iters": settings.get("cg_fixed_iters"),
        "cg_fixed_iters_selected": settings.get("_cg_fixed_iters_selected"),
        "torch_compile_admm": bool(settings.get("torch_compile_admm", False)),
        "torch_compile_admm_status": settings.get(
            "_torch_compile_admm_status", "disabled"
        ),
        "cuda_graph": False,
        "cuda_graph_status": "disabled",
        "cuda_graph_cache_hit": False,
        "cuda_graph_recaptures": 0,
        "cuda_graph_eligibility_reason": "dense_solver",
        "adaptive_rho": bool(settings.get("adaptive_rho", False)),
        "rho_update_interval": settings.get("rho_update_interval", "auto"),
        "rho_update_tolerance": settings.get("rho_update_tolerance", 5.0),
        "rho_updates": 0,
        "rho_bar": float(settings["rho"]),
        "rho_min": float(settings["rho"]),
        "rho_max": float(settings["rho"]),
        "scaling_applied": False,
        "scaling_passes": 0,
        **_default_polish_info(settings),
        "device": str(q.device),
        "dtype": str(q.dtype),
        "sparse_setup_cache_hit": False,
        "timing_setup_ms": 0.0,
        "timing_cg_ms": 0.0,
        "timing_admm_update_ms": 0.0,
        "timing_residual_ms": 0.0,
        "timing_cuda_graph_capture_ms": 0.0,
        "timing_cuda_graph_replay_ms": 0.0,
    }


def _operator_info(
    operator,
    q,
    x,
    z,
    y,
    settings,
    iteration,
    status,
    total_cg_iterations,
    last_cg_info,
    rho_updates=0,
    rho_bar=None,
    rho_vec=None,
    polish_info=None,
    timing=None,
):
    primal_res, dual_res, eps_prim, eps_dual = _operator_residuals(
        operator, q, x, z, y, settings["eps_abs"], settings["eps_rel"]
    )
    objective = 0.5 * torch.dot(x, operator.P_mv(x)) + torch.dot(q, x)
    average_cg = total_cg_iterations / max(iteration, 1)
    info = {
        "status": status,
        "admm_iterations": iteration,
        "primal_residual": float(primal_res.item()),
        "dual_residual": float(dual_res.item()),
        "eps_primal": float(eps_prim.item()),
        "eps_dual": float(eps_dual.item()),
        "objective": float(objective.item()),
        "linear_solver": "sparse_cg",
        "linear_solver_requested": settings.get(
            "linear_solver_requested", settings.get("linear_solver", "sparse_cg")
        ),
        "linear_solver_selected": settings.get(
            "linear_solver_selected", "sparse_cg"
        ),
        "linear_solver_auto_reason": settings.get(
            "linear_solver_auto_reason", "explicit_sparse_cg"
        ),
        "estimated_kkt_dim": settings.get("estimated_kkt_dim"),
        "estimated_dense_kkt_mb": settings.get("estimated_dense_kkt_mb"),
        "estimated_sparse_nnz": settings.get("estimated_sparse_nnz"),
        "total_cg_iterations": int(total_cg_iterations),
        "average_cg_iterations": float(average_cg),
        "last_cg_relative_residual": last_cg_info["relative_residual"],
        "last_cg_residual_norm": last_cg_info["residual_norm"],
        "last_cg_converged": last_cg_info["converged"],
        "last_cg_status": last_cg_info["status"],
        "cg_check_interval": int(settings.get("cg_check_interval", 1)),
        "cg_fixed_iters": settings.get("cg_fixed_iters"),
        "cg_fixed_iters_selected": settings.get("_cg_fixed_iters_selected"),
        "torch_compile_admm": bool(settings.get("torch_compile_admm", False)),
        "torch_compile_admm_status": settings.get(
            "_torch_compile_admm_status", "disabled"
        ),
        "cuda_graph": bool(settings.get("cuda_graph", False)),
        "cuda_graph_status": "disabled",
        "cuda_graph_cache_hit": False,
        "cuda_graph_recaptures": 0,
        "cuda_graph_eligibility_reason": (
            "requested" if settings.get("cuda_graph", False) else "disabled"
        ),
        "adaptive_rho": bool(settings.get("adaptive_rho", False)),
        "rho_update_interval": _rho_update_interval(
            settings, settings["check_termination"]
        ),
        "rho_update_tolerance": float(settings.get("rho_update_tolerance", 5.0)),
        "rho_updates": int(rho_updates),
        "rho_bar": None if rho_bar is None else float(rho_bar.item()),
        "rho_min": None if rho_vec is None else float(torch.min(rho_vec).item()),
        "rho_max": None if rho_vec is None else float(torch.max(rho_vec).item()),
        "scaling_applied": bool(settings.get("scaling", 0)),
        "scaling_passes": int(settings.get("scaling", 0) or 0),
        "device": str(q.device),
        "dtype": str(q.dtype),
        "uses_matrix_free_bounds": operator.uses_matrix_free_bounds,
        "sparse_setup_cache_hit": bool(getattr(operator, "cache_hit", False)),
        "sparse_storage_nnz": operator.sparse_storage_nnz(),
        "dense_kkt_entries": (operator.n + operator.m) ** 2,
    }
    info.update(polish_info or _default_polish_info(settings))
    timing = timing or {}
    info.update(
        {
            "timing_setup_ms": float(timing.get("setup_ms", 0.0)),
            "timing_cg_ms": float(timing.get("cg_ms", 0.0)),
            "timing_admm_update_ms": float(timing.get("admm_update_ms", 0.0)),
            "timing_residual_ms": float(timing.get("residual_ms", 0.0)),
            "timing_cuda_graph_capture_ms": float(
                timing.get("cuda_graph_capture_ms", 0.0)
            ),
            "timing_cuda_graph_replay_ms": float(
                timing.get("cuda_graph_replay_ms", 0.0)
            ),
            "timing_mode": (
                "cuda_events"
                if settings.get("cuda_event_timing", False) and q.device.type == "cuda"
                else "host_wall"
            ),
        }
    )
    return info


def _cg_info(converged, iterations, residual_norm, b_norm, status):
    b_norm_value = float(b_norm.item())
    residual_value = float(residual_norm.item())
    relative = residual_value / max(b_norm_value, 1.0)
    return {
        "converged": bool(converged),
        "iterations": int(iterations),
        "residual_norm": residual_value,
        "relative_residual": relative,
        "status": status,
    }


def _nnz(matrix):
    if matrix is None:
        return 0
    if matrix.layout == torch.strided:
        return int(torch.count_nonzero(matrix).item())
    return int(matrix._nnz())
