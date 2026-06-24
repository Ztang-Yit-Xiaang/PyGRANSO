import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy import sparse

import bench_osqp_runtime as osqp_bench
import pygranso.private.osqpTorchAdapter as adapter
import pygranso.private.torchOSQP as torch_osqp
import pygranso.private.solveQP as solve_qp_module
from pygranso.private.osqpTorchAdapter import (
    OSQPCudaInteropUnavailableError,
    solve_osqp_torch_qp,
)
from pygranso.private.solveQP import solveQP
from pygranso.private.torchOSQP import (
    BoundConstrainedOSQPOperator,
    ExplicitOSQPOperator,
    jacobi_preconditioner_diagonal,
    reduced_system_matvec,
    solve_torch_osqp_from_qp,
)
from pygranso.pygransoOptions import pygransoOptions
from pygranso.pygransoStruct import pygransoStruct

OSQP_AVAILABLE = importlib.util.find_spec("osqp") is not None


def sparse_cg_settings(**overrides):
    settings = {
        "linear_solver": "sparse_cg",
        "rho": 0.1,
        "sigma": 1e-6,
        "alpha": 1.6,
        "max_iter": 4000,
        "eps_abs": 1e-8,
        "eps_rel": 1e-8,
        "check_termination": 25,
        "cg_rtol": 1e-8,
        "cg_atol": 0.0,
        "cg_max_iter": 100,
        "verbose": False,
    }
    settings.update(overrides)
    return settings


def auto_settings(**overrides):
    settings = {
        "linear_solver": "auto",
        "rho": 0.1,
        "sigma": 1e-6,
        "alpha": 1.6,
        "max_iter": 4000,
        "eps_abs": 1e-8,
        "eps_rel": 1e-8,
        "check_termination": 25,
        "cg_rtol": 1e-8,
        "cg_atol": 0.0,
        "cg_max_iter": 100,
        "verbose": False,
    }
    settings.update(overrides)
    return settings


def simple_bound_qp(dtype=torch.float64, device="cpu"):
    H = torch.eye(1, device=device, dtype=dtype)
    f = torch.tensor([[-2.0]], device=device, dtype=dtype)
    LB = torch.zeros((1, 1), device=device, dtype=dtype)
    UB = torch.ones((1, 1), device=device, dtype=dtype)
    return H, f, None, None, LB, UB


def simple_bounds_qp(dtype=torch.float64, device="cpu"):
    H = torch.eye(2, device=device, dtype=dtype)
    f = torch.tensor([[-1.0], [-2.0]], device=device, dtype=dtype)
    LB = torch.zeros((2, 1), device=device, dtype=dtype)
    UB = torch.ones((2, 1), device=device, dtype=dtype)
    return H, f, None, None, LB, UB


def test_cpu_builtin_adapter_canonicalizes_equality_and_bounds(monkeypatch):
    captured = {}

    class FakeProblem:
        def __init__(self, algebra):
            captured["algebra"] = algebra

        def setup(self, P, q, A, l, u, **settings):
            captured.update({"P": P, "q": q, "A": A, "l": l, "u": u, "settings": settings})

        def solve(self):
            return SimpleNamespace(x=np.array([0.5, 0.5]))

    fake_osqp = SimpleNamespace(OSQP=FakeProblem)
    monkeypatch.setattr(
        "pygranso.private.osqpTorchAdapter.importlib.import_module",
        lambda name: fake_osqp,
    )

    dtype = torch.float64
    H = torch.eye(2, dtype=dtype)
    f = torch.zeros((2, 1), dtype=dtype)
    A = torch.ones((1, 2), dtype=dtype)
    LB = torch.zeros((2, 1), dtype=dtype)
    UB = torch.ones((2, 1), dtype=dtype)

    solution = solve_osqp_torch_qp(
        H,
        f,
        A,
        1.0,
        LB,
        UB,
        torch.device("cpu"),
        True,
        options={"algebra": "builtin", "settings": {"eps_abs": 1e-8}},
    )

    assert captured["algebra"] == "builtin"
    assert sparse.isspmatrix_csc(captured["P"])
    assert sparse.isspmatrix_csc(captured["A"])
    assert captured["P"].shape == (2, 2)
    assert captured["A"].shape == (3, 2)
    np.testing.assert_allclose(captured["q"], np.array([0.0, 0.0]))
    np.testing.assert_allclose(captured["l"], np.array([[1.0], [0.0], [0.0]]))
    np.testing.assert_allclose(captured["u"], np.array([[1.0], [1.0], [1.0]]))
    assert captured["settings"]["eps_abs"] == 1e-8
    torch.testing.assert_close(solution, torch.tensor([[0.5], [0.5]], dtype=dtype))


def test_cpu_torch_backend_solves_simple_bound_qp():
    dtype = torch.float64
    H, f, A, b, LB, UB = simple_bound_qp(dtype=dtype)

    solution = solve_osqp_torch_qp(
        H,
        f,
        A,
        b,
        LB,
        UB,
        torch.device("cpu"),
        True,
        options={"algebra": "torch"},
    )

    assert solution.shape == (1, 1)
    assert solution.device.type == "cpu"
    assert solution.dtype == dtype
    torch.testing.assert_close(
        solution, torch.tensor([[1.0]], dtype=dtype), atol=1e-5, rtol=1e-5
    )


@pytest.mark.parametrize(
    "b",
    [
        1.0,
        torch.tensor(1.0, dtype=torch.float64),
        torch.tensor([1.0], dtype=torch.float64),
        torch.tensor([[1.0]], dtype=torch.float64),
    ],
)
def test_cpu_torch_backend_maps_equality_and_bounds(b):
    dtype = torch.float64
    H = torch.eye(2, dtype=dtype)
    f = torch.zeros((2, 1), dtype=dtype)
    A = torch.ones((1, 2), dtype=dtype)
    LB = torch.zeros((2, 1), dtype=dtype)
    UB = torch.ones((2, 1), dtype=dtype)

    solution = solve_osqp_torch_qp(
        H,
        f,
        A,
        b,
        LB,
        UB,
        torch.device("cpu"),
        True,
        options={"algebra": "torch"},
    )

    torch.testing.assert_close(
        solution,
        0.5 * torch.ones((2, 1), dtype=dtype),
        atol=1e-5,
        rtol=1e-5,
    )


def test_torch_backend_accepts_explicit_polishing():
    H, f, A, b, LB, UB = simple_bound_qp()

    solution, info = solve_osqp_torch_qp(
        H,
        f,
        A,
        b,
        LB,
        UB,
        torch.device("cpu"),
        True,
        options={
            "algebra": "torch",
            "settings": sparse_cg_settings(return_info=True, polish=True),
        },
    )

    torch.testing.assert_close(
        solution, torch.tensor([[1.0]], dtype=torch.float64), atol=1e-5, rtol=1e-5
    )
    assert info["polishing"] is True
    assert info["polishing_status"] != "disabled"


def test_auto_policy_selects_cuda_when_available(monkeypatch):
    captured = {}

    def fake_torch_path(
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
        captured.update(
            {
                "target_device": target_device,
                "solve_device": solve_device,
                "allow_device_move": allow_device_move,
                "settings": settings,
            }
        )
        return torch.zeros((f.numel(), 1), device=target_device, dtype=torch_dtype)

    monkeypatch.setattr(adapter.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(adapter, "_solve_torch_osqp_path", fake_torch_path)

    H, f, A, b, LB, UB = simple_bounds_qp()
    solution = solve_osqp_torch_qp(
        H,
        f,
        A,
        b,
        LB,
        UB,
        torch.device("cpu"),
        True,
        options={"algebra": "auto"},
    )

    assert captured["solve_device"].type == "cuda"
    assert captured["target_device"].type == "cpu"
    assert captured["allow_device_move"] is True
    assert captured["settings"]["linear_solver"] == "auto"
    assert solution.device.type == "cpu"


def test_auto_policy_falls_back_to_cpu_without_cuda(monkeypatch):
    captured = {}

    def fake_builtin_path(
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
        captured["target_device"] = target_device
        return torch.full((f.numel(), 1), 0.25, device=target_device, dtype=torch_dtype)

    def fail_torch_path(*args, **kwargs):
        raise AssertionError("auto without CUDA should not use the Torch path")

    monkeypatch.setattr(adapter.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(adapter, "_solve_builtin_osqp_path", fake_builtin_path)
    monkeypatch.setattr(adapter, "_solve_torch_osqp_path", fail_torch_path)

    H, f, A, b, LB, UB = simple_bounds_qp()
    solution = solve_osqp_torch_qp(
        H,
        f,
        A,
        b,
        LB,
        UB,
        torch.device("cpu"),
        True,
        options={"algebra": "auto"},
    )

    assert captured["target_device"].type == "cpu"
    torch.testing.assert_close(
        solution, torch.full((2, 1), 0.25, dtype=torch.float64)
    )


def test_auto_policy_cpu_fallback_after_cuda_sparse_not_supported(monkeypatch):
    captured = {}

    def unsupported_torch_path(*args, **kwargs):
        raise NotImplementedError("sparse CUDA kernel is not implemented")

    def fake_builtin_path(
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
        captured["used_builtin"] = True
        return torch.full((f.numel(), 1), 0.75, device=target_device, dtype=torch_dtype)

    monkeypatch.setattr(adapter.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(adapter, "_solve_torch_osqp_path", unsupported_torch_path)
    monkeypatch.setattr(adapter, "_solve_builtin_osqp_path", fake_builtin_path)

    H, f, A, b, LB, UB = simple_bounds_qp()
    with pytest.warns(RuntimeWarning, match="Falling back to builtin CPU OSQP"):
        solution = solve_osqp_torch_qp(
            H,
            f,
            A,
            b,
            LB,
            UB,
            torch.device("cpu"),
            True,
            options={
                "algebra": "auto",
                "settings": {"linear_solver": "auto"},
            },
        )

    assert captured["used_builtin"] is True
    torch.testing.assert_close(
        solution, torch.full((2, 1), 0.75, dtype=torch.float64)
    )


def test_auto_policy_explicit_sparse_cg_failure_does_not_fallback(monkeypatch):
    def unsupported_torch_path(*args, **kwargs):
        raise NotImplementedError("sparse CUDA kernel is not implemented")

    def unexpected_builtin_path(*args, **kwargs):
        raise AssertionError("explicit sparse_cg should not fall back to CPU OSQP")

    monkeypatch.setattr(adapter.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(adapter, "_solve_torch_osqp_path", unsupported_torch_path)
    monkeypatch.setattr(adapter, "_solve_builtin_osqp_path", unexpected_builtin_path)

    H, f, A, b, LB, UB = simple_bounds_qp()
    with pytest.raises(NotImplementedError, match="sparse CUDA kernel"):
        solve_osqp_torch_qp(
            H,
            f,
            A,
            b,
            LB,
            UB,
            torch.device("cpu"),
            True,
            options={
                "algebra": "auto",
                "settings": {"linear_solver": "sparse_cg"},
            },
        )


@pytest.mark.skipif(not OSQP_AVAILABLE, reason="OSQP Python package is not installed")
@pytest.mark.parametrize("double_precision", [True, False])
def test_bounds_only_qp_returns_column_on_requested_device(double_precision):
    dtype = torch.float64 if double_precision else torch.float32
    H, f, A, b, LB, UB = simple_bounds_qp(dtype=dtype)

    solution = solve_osqp_torch_qp(
        H, f, A, b, LB, UB, torch.device("cpu"), double_precision
    )

    assert solution.shape == (2, 1)
    assert solution.device.type == "cpu"
    assert solution.dtype == dtype
    torch.testing.assert_close(solution, torch.ones((2, 1), dtype=dtype), atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not OSQP_AVAILABLE, reason="OSQP Python package is not installed")
@pytest.mark.parametrize("b", [1.0, torch.tensor([[1.0]], dtype=torch.float64)])
def test_equality_plus_bounds_accepts_scalar_or_tensor_b(b):
    dtype = torch.float64
    H = torch.eye(2, dtype=dtype)
    f = torch.zeros((2, 1), dtype=dtype)
    A = torch.ones((1, 2), dtype=dtype)
    LB = torch.zeros((2, 1), dtype=dtype)
    UB = torch.ones((2, 1), dtype=dtype)

    solution = solve_osqp_torch_qp(H, f, A, b, LB, UB, torch.device("cpu"), True)

    torch.testing.assert_close(
        solution,
        0.5 * torch.ones((2, 1), dtype=dtype),
        atol=1e-5,
        rtol=1e-5,
    )


@pytest.mark.skipif(not OSQP_AVAILABLE, reason="OSQP Python package is not installed")
def test_objective_convention_uses_half_quadratic():
    dtype = torch.float64
    H = 2.0 * torch.eye(1, dtype=dtype)
    f = torch.tensor([[-4.0]], dtype=dtype)
    LB = torch.tensor([[-10.0]], dtype=dtype)
    UB = torch.tensor([[10.0]], dtype=dtype)

    solution = solve_osqp_torch_qp(H, f, None, None, LB, UB, torch.device("cpu"), True)

    torch.testing.assert_close(solution, torch.tensor([[2.0]], dtype=dtype), atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not OSQP_AVAILABLE, reason="OSQP Python package is not installed")
def test_runtime_builtin_cpu_osqp_solves_bound_qp():
    dtype = torch.float64
    H, f, A, b, LB, UB = simple_bound_qp(dtype=dtype)

    solution = solve_osqp_torch_qp(
        H,
        f,
        A,
        b,
        LB,
        UB,
        torch.device("cpu"),
        True,
        options={"algebra": "builtin"},
    )

    assert solution.shape == (1, 1)
    assert solution.device.type == "cpu"
    assert solution.dtype == dtype
    assert torch.all(torch.isfinite(solution))
    torch.testing.assert_close(
        solution, torch.tensor([[1.0]], dtype=dtype), atol=1e-5, rtol=1e-5
    )


def test_runtime_torch_dense_solves_bound_qp_with_info():
    dtype = torch.float64
    H, f, A, b, LB, UB = simple_bound_qp(dtype=dtype)

    solution, info = solve_osqp_torch_qp(
        H,
        f,
        A,
        b,
        LB,
        UB,
        torch.device("cpu"),
        True,
        options={
            "algebra": "torch",
            "settings": {"linear_solver": "dense", "return_info": True},
        },
    )

    assert solution.shape == (1, 1)
    assert solution.device.type == "cpu"
    assert solution.dtype == dtype
    assert torch.all(torch.isfinite(solution))
    torch.testing.assert_close(
        solution, torch.tensor([[1.0]], dtype=dtype), atol=1e-5, rtol=1e-5
    )
    assert info["linear_solver"] == "dense"
    assert info["linear_solver_requested"] == "dense"
    assert info["linear_solver_selected"] == "dense"
    assert info["linear_solver_auto_reason"] == "explicit_dense"


def test_runtime_torch_auto_sparse_cg_reports_governance_info():
    dtype = torch.float64
    n = 600
    idx = torch.arange(n)
    H = torch.sparse_coo_tensor(
        torch.stack((idx, idx)),
        torch.ones(n, dtype=dtype),
        (n, n),
        dtype=dtype,
    )
    f = torch.zeros((n, 1), dtype=dtype)
    LB = -torch.ones((n, 1), dtype=dtype)
    UB = torch.ones((n, 1), dtype=dtype)

    solution, info = solve_osqp_torch_qp(
        H,
        f,
        None,
        None,
        LB,
        UB,
        torch.device("cpu"),
        True,
        options={
            "algebra": "torch",
            "settings": {
                "linear_solver": "auto",
                "return_info": True,
                "max_iter": 1,
                "check_termination": 1,
            },
        },
    )

    assert solution.shape == (n, 1)
    assert torch.all(torch.isfinite(solution))
    assert info["linear_solver_requested"] == "auto"
    assert info["linear_solver_selected"] == "sparse_cg"
    assert info["linear_solver_auto_reason"] == "large_sparse_problem"
    assert info["estimated_kkt_dim"] == 2 * n
    assert info["estimated_sparse_nnz"] == 2 * n
    assert info["estimated_dense_kkt_mb"] > 0
    assert info["sparse_storage_nnz"] == n
    assert info["dense_kkt_entries"] == (2 * n) ** 2
    assert info["sparse_storage_nnz"] < info["dense_kkt_entries"] // 100


@pytest.mark.skipif(not OSQP_AVAILABLE, reason="OSQP Python package is not installed")
def test_runtime_builtin_cpu_and_torch_dense_agree_on_small_qp():
    dtype = torch.float64
    H, f, A, b, LB, UB = simple_bounds_qp(dtype=dtype)

    builtin_solution = solve_osqp_torch_qp(
        H,
        f,
        A,
        b,
        LB,
        UB,
        torch.device("cpu"),
        True,
        options={"algebra": "builtin"},
    )
    dense_solution, dense_info = solve_osqp_torch_qp(
        H,
        f,
        A,
        b,
        LB,
        UB,
        torch.device("cpu"),
        True,
        options={
            "algebra": "torch",
            "settings": {"linear_solver": "dense", "return_info": True},
        },
    )

    expected = torch.ones((2, 1), dtype=dtype)
    torch.testing.assert_close(builtin_solution, expected, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(dense_solution, expected, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(dense_solution, builtin_solution, atol=1e-5, rtol=1e-5)
    assert dense_info["linear_solver_selected"] == "dense"


def test_runtime_benchmark_qp_builder_returns_sparse_bound_problem():
    H, f, A, b, LB, UB = osqp_bench.make_sparse_bound_qp(
        10, device="cpu", dtype=torch.float64
    )

    assert H.shape == (10, 10)
    assert H.is_sparse
    assert H._nnz() == 10
    assert f.shape == (10, 1)
    assert A is None
    assert b is None
    assert LB.shape == (10, 1)
    assert UB.shape == (10, 1)
    torch.testing.assert_close(f, -torch.ones((10, 1), dtype=torch.float64))
    torch.testing.assert_close(LB, -torch.ones((10, 1), dtype=torch.float64))
    torch.testing.assert_close(UB, torch.ones((10, 1), dtype=torch.float64))


def test_runtime_benchmark_qp_builders_cover_sparse_cases():
    H, f, A, b, LB, UB = osqp_bench.make_sparse_equality_bound_qp(
        12, device="cpu", dtype=torch.float64
    )

    assert H.is_sparse
    assert H._nnz() == 12
    assert A.is_sparse
    assert A.shape == (1, 12)
    assert A._nnz() == 12
    assert b.shape == (1, 1)
    assert f.shape == (12, 1)
    assert LB.shape == (12, 1)
    assert UB.shape == (12, 1)

    H, f, A, b, LB, UB = osqp_bench.make_random_sparse_spd_qp(
        12, device="cpu", dtype=torch.float64, seed=7, density=0.15
    )

    assert H.is_sparse
    assert H.shape == (12, 12)
    assert H._nnz() >= 12
    assert A is None
    assert b is None
    assert f.shape == (12, 1)
    assert LB.shape == (12, 1)
    assert UB.shape == (12, 1)
    torch.testing.assert_close(H.to_dense(), H.to_dense().T)


def test_runtime_benchmark_timing_runner_returns_required_row_keys():
    args = osqp_bench.parse_args(
        [
            "--sizes",
            "10",
            "--repeats",
            "1",
            "--warmups",
            "0",
            "--max-iter",
            "3",
            "--check-termination",
            "1",
        ]
    )

    row = osqp_bench.time_backend(10, "torch_dense", args)

    required_keys = {
        "n",
        "backend",
        "status",
        "median_ms",
        "selected",
        "reason",
        "objective",
        "prim_res",
        "dual_res",
        "cg_iters",
        "cg_fix",
        "compile",
        "rho_upd",
        "scale",
        "cache",
        "polish",
        "external",
        "setup_ms",
        "update_ms",
        "solve_ms",
        "cg_ms",
        "admm_ms",
        "resid_ms",
        "dense_mb",
        "sparse_nnz",
        "case",
        "ref_err",
    }
    assert required_keys <= row.keys()
    assert row["case"] == "bound"
    assert row["n"] == 10
    assert row["backend"] == "torch_dense"
    assert row["status"] == "ok"
    assert row["selected"] == "dense"
    assert row["median_ms"] >= 0
    assert row["dense_mb"] > 0


def test_runtime_benchmark_ablation_settings_are_forwarded():
    args = osqp_bench.parse_args(
        [
            "--sizes",
            "10",
            "--scaling",
            "3",
            "--adaptive-rho",
            "--rho-update-interval",
            "2",
            "--cg-check-interval",
            "4",
            "--cg-fixed-iters",
            "5",
            "--torch-compile-admm",
            "--warm-start",
            "--polishing",
            "--polish-refine-iter",
            "2",
        ]
    )

    settings = osqp_bench.benchmark_settings("auto", args)

    assert settings["scaling"] == 3
    assert settings["adaptive_rho"] is True
    assert settings["rho_update_interval"] == 2
    assert settings["cg_check_interval"] == 4
    assert settings["cg_fixed_iters"] == 5
    assert settings["torch_compile_admm"] is True
    assert settings["warm_start"] is True
    assert settings["return_state"] is True
    assert settings["polishing"] is True
    assert settings["polish_refine_iter"] == 2


def test_runtime_benchmark_parametric_sequence_preserves_structure():
    args = osqp_bench.parse_args(["--parametric-sequence"])
    sequence = osqp_bench.make_parametric_qp_sequence(
        "random_spd",
        12,
        args,
        device="cpu",
        dtype=torch.float64,
        count=3,
    )

    first_H = sequence[0][0].coalesce()
    first_indices = first_H.indices()
    assert len(sequence) == 3
    for qp in sequence[1:]:
        H = qp[0].coalesce()
        torch.testing.assert_close(H.indices(), first_indices)
    assert not torch.allclose(sequence[0][1], sequence[1][1])


def test_runtime_benchmark_additional_cases_cover_active_and_ill_conditioned():
    active = osqp_bench.make_qp_case("active_bound", 10, dtype=torch.float64)
    ill = osqp_bench.make_qp_case("ill_conditioned_spd", 10, dtype=torch.float64)

    assert active[0].is_sparse
    assert active[4].min().item() == 0.0
    assert active[5].max().item() == 1.0
    assert ill[0].is_sparse
    assert ill[0].coalesce().values().max() / ill[0].coalesce().values().min() > 1e6


def test_runtime_benchmark_external_solver_skips_without_torch_sla(monkeypatch):
    real_find_spec = osqp_bench.importlib.util.find_spec

    def fake_find_spec(name):
        if name == "torch_sla":
            return None
        return real_find_spec(name)

    monkeypatch.setattr(osqp_bench.importlib.util, "find_spec", fake_find_spec)
    args = osqp_bench.parse_args(
        [
            "--sizes",
            "8",
            "--repeats",
            "1",
            "--warmups",
            "0",
            "--external-solver",
            "torch_sla_pytorch_cg",
        ]
    )

    row = osqp_bench.time_external_solver(
        8, "torch_sla_pytorch_cg", args, "bound"
    )

    assert row["status"] == "skipped"
    assert row["backend"] == "external:torch_sla_pytorch_cg"
    assert row["reason"] == "torch_sla_unavailable"
    assert row["external"] == "-"


def test_runtime_benchmark_external_solver_uses_lazy_torch_sla(monkeypatch):
    class FakeTorchSla:
        @staticmethod
        def solve(matrix, rhs, solver=None):
            dense = matrix.to_dense() if matrix.layout != torch.strided else matrix
            return torch.linalg.solve(dense, rhs.reshape(-1, 1)).reshape(-1)

    real_find_spec = osqp_bench.importlib.util.find_spec

    def fake_find_spec(name):
        if name == "torch_sla":
            return object()
        return real_find_spec(name)

    monkeypatch.setitem(sys.modules, "torch_sla", FakeTorchSla)
    monkeypatch.setattr(osqp_bench.importlib.util, "find_spec", fake_find_spec)
    args = osqp_bench.parse_args(
        [
            "--sizes",
            "8",
            "--repeats",
            "1",
            "--warmups",
            "0",
            "--external-solver",
            "torch_sla_pytorch_cg",
        ]
    )

    row = osqp_bench.time_external_solver(
        8, "torch_sla_pytorch_cg", args, "bound"
    )

    assert row["status"] == "ok"
    assert row["selected"] == "linear"
    assert row["external"] == "torch_sla_pytorch_cg"
    assert row["prim_res"] <= 1e-10


def test_runtime_benchmark_cpu_update_warm_reuses_setup(monkeypatch):
    instances = []

    class FakeResult:
        def __init__(self, n):
            self.x = np.zeros(n)
            self.y = np.zeros(n)
            self.info = SimpleNamespace(
                status="solved", prim_res=0.0, dual_res=0.0, obj_val=0.0
            )

    class FakeProblem:
        def __init__(self, algebra=None):
            self.setup_calls = 0
            self.update_calls = 0
            self.warm_start_calls = 0
            instances.append(self)

        def setup(self, P, q, A, l, u, **settings):
            self.n = q.size
            self.setup_calls += 1

        def solve(self):
            return FakeResult(self.n)

        def update(self, Px=None, Ax=None, q=None, l=None, u=None):
            self.last_Px = Px
            self.last_Ax = Ax
            self.update_calls += 1

        def warm_start(self, x=None, y=None):
            self.warm_start_calls += 1

    class FakeOSQPModule:
        OSQP = FakeProblem

    real_find_spec = osqp_bench.importlib.util.find_spec

    def fake_find_spec(name):
        if name == "osqp":
            return object()
        return real_find_spec(name)

    monkeypatch.setattr(osqp_bench.importlib.util, "find_spec", fake_find_spec)
    monkeypatch.setitem(sys.modules, "osqp", FakeOSQPModule)
    args = osqp_bench.parse_args(
        [
            "--cases",
            "random_spd",
            "--sizes",
            "6",
            "--repeats",
            "2",
            "--warmups",
            "1",
            "--reference",
            "off",
            "--parametric-sequence",
        ]
    )

    row = osqp_bench.time_builtin_update_warm(6, args, "random_spd")

    assert row["backend"] == "builtin_update_warm"
    assert row["status"] == "ok"
    assert row["update_ms"] is not None
    assert row["solve_ms"] is not None
    assert len(instances) == 1
    assert instances[0].setup_calls == 1
    assert instances[0].update_calls == 3
    assert instances[0].warm_start_calls == 3
    assert instances[0].last_Px is None
    assert instances[0].last_Ax is None


def test_runtime_benchmark_academic_row_format():
    args = osqp_bench.parse_args(["--academic-table", "--device", "cpu"])
    row = {
        "status": "ok",
        "backend": "torch_auto",
        "case": "random_spd",
        "n": 12,
        "median_ms": 3.5,
        "selected": "sparse_cg",
        "cg_fix": 2,
        "rho_upd": 1,
        "scale": 5,
        "polish": "-",
        "cache": "hit",
        "prim_res": 1e-6,
        "dual_res": 2e-6,
        "ref_err": 3e-6,
        "cg_iters": 24,
        "setup_ms": 1.0,
        "update_ms": None,
        "solve_ms": None,
    }

    academic = osqp_bench.academic_row(
        row,
        args,
        {
            "fresh": {("random_spd", 12): 7.0},
            "warm": {("random_spd", 12): 4.0},
        },
    )

    assert academic["method"] == "Torch cold sparse-CG"
    assert academic["device"] == "cpu"
    assert academic["speedup_vs_cpu_osqp"] == 2.0
    assert academic["speedup_vs_cpu_fresh"] == 2.0
    assert academic["speedup_vs_cpu_warm"] == 4.0 / 3.5
    assert academic["efficiency_status"] == "win"
    assert academic["selected_policy"] == (
        "sparse_cg, cg_fixed=2, rho_updates=1, scaling=5"
    )
    assert academic["cache_hit"] == "hit"


def test_runtime_benchmark_rejects_inaccurate_speedup():
    args = osqp_bench.parse_args(["--academic-table"])
    row = {
        "status": "ok",
        "backend": "torch_auto",
        "case": "random_spd",
        "n": 12,
        "median_ms": 1.0,
        "selected": "sparse_cg",
        "cg_fix": None,
        "rho_upd": 0,
        "scale": 0,
        "polish": "-",
        "cache": "-",
        "prim_res": 1e-6,
        "dual_res": 2e-6,
        "ref_err": 1e-3,
        "cg_iters": 4,
        "setup_ms": None,
        "update_ms": None,
        "solve_ms": None,
    }

    academic = osqp_bench.academic_row(
        row,
        args,
        {"fresh": {("random_spd", 12): 10.0}, "warm": {("random_spd", 12): 10.0}},
    )

    assert academic["speedup_vs_cpu_osqp"] == 10.0
    assert academic["efficiency_status"] == "reject_accuracy"


def test_runtime_benchmark_rejects_rows_that_do_not_beat_cpu_warm():
    args = osqp_bench.parse_args(["--academic-table"])
    row = {
        "status": "ok",
        "backend": "pygranso_torch",
        "case": "random_spd",
        "n": 12,
        "median_ms": 5.0,
        "selected": "sparse_cg",
        "cg_fix": None,
        "rho_upd": 0,
        "scale": 0,
        "polish": "-",
        "cache": "hit",
        "prim_res": 1e-6,
        "dual_res": 2e-6,
        "ref_err": 1e-7,
        "cg_iters": 4,
        "setup_ms": None,
        "update_ms": None,
        "solve_ms": None,
    }

    academic = osqp_bench.academic_row(
        row,
        args,
        {"fresh": {("random_spd", 12): 10.0}, "warm": {("random_spd", 12): 4.0}},
    )

    assert academic["speedup_vs_cpu_fresh"] == 2.0
    assert academic["speedup_vs_cpu_warm"] == 0.8
    assert academic["efficiency_status"] == "loss_cpu_warm"


def test_runtime_benchmark_cuda_win_preset():
    args = osqp_bench.parse_args(["--cuda-win-suite", "--max-safe-size", "2400"])

    assert args.cases == ["random_spd"]
    assert args.sizes == [1200, 1600, 2000, 2400]
    assert args.repeats == 5
    assert args.warmups == 2
    assert args.max_iter == 200
    assert args.reference == "auto"
    assert args.academic_table is True
    assert args.include_pygranso_repeat is True
    assert args.include_cpu_warm is True
    assert args.parametric_sequence is True


def test_runtime_benchmark_fair_optimization_suite_preset():
    args = osqp_bench.parse_args(["--fair-optimization-suite"])

    assert args.cases == ["random_spd"]
    assert args.sizes == [1200]
    assert args.reference == "auto"
    assert args.academic_table is True
    assert args.include_pygranso_repeat is True
    assert args.include_cpu_warm is True
    assert args.parametric_sequence is True


def test_runtime_benchmark_fair_candidate_settings_are_forwarded():
    variants = {
        label: overrides
        for label, _backend, overrides in osqp_bench.FAIR_OPTIMIZATION_VARIANTS
    }
    args = osqp_bench.parse_args(["--fair-optimization-suite"])

    fixed10 = osqp_bench.benchmark_settings(
        "auto",
        osqp_bench._copy_args_with(args, **variants["fair_fast_fixed1_iter10"]),
    )
    assert fixed10["max_iter"] == 10
    assert fixed10["cg_fixed_iters"] == 1
    assert fixed10["cg_check_interval"] == 10
    assert fixed10["check_termination"] == 10

    auto20 = osqp_bench.benchmark_settings(
        "auto",
        osqp_bench._copy_args_with(args, **variants["fair_fast_auto_iter20"]),
    )
    assert auto20["max_iter"] == 20
    assert auto20["cg_fixed_iters"] == "auto"
    assert auto20["cg_check_interval"] == 20
    assert auto20["check_termination"] == 20

    adaptive = osqp_bench.benchmark_settings(
        "auto",
        osqp_bench._copy_args_with(args, **variants["fair_fast_adaptive_iter20"]),
    )
    assert adaptive["adaptive_rho"] is True

    scaled = osqp_bench.benchmark_settings(
        "auto",
        osqp_bench._copy_args_with(args, **variants["fair_fast_scaled_iter20"]),
    )
    assert scaled["scaling"] == 5


def _academic_test_row(
    backend,
    median_ms,
    ref_err=None,
    status="ok",
    case="random_spd",
    n=12,
):
    return {
        "status": status,
        "backend": backend,
        "case": case,
        "n": n,
        "median_ms": median_ms,
        "selected": "sparse_cg" if backend != "builtin_cpu" else "-",
        "cg_fix": None,
        "rho_upd": 0,
        "scale": 0,
        "polish": "-",
        "cache": "hit" if backend != "builtin_cpu" else "-",
        "prim_res": 1e-7 if backend != "builtin_cpu" else None,
        "dual_res": 1e-7 if backend != "builtin_cpu" else None,
        "ref_err": ref_err,
        "cg_iters": 4 if backend != "builtin_cpu" else 0,
        "setup_ms": None,
        "update_ms": None,
        "solve_ms": None,
    }


def test_runtime_benchmark_best_fair_row_rejects_bad_ref_error():
    args = osqp_bench.parse_args(["--fair-optimization-suite"])
    rows = [
        _academic_test_row("builtin_cpu", 10.0),
        _academic_test_row("builtin_update_warm", 4.0),
        _academic_test_row("fair_fast_fixed1_iter10", 2.0, ref_err=1e-3),
    ]

    summary = osqp_bench.best_fair_cuda_row(rows, args)

    assert summary["status"] == "no_valid_cuda_rows"


def test_runtime_benchmark_best_fair_row_rejects_cpu_warm_loss():
    args = osqp_bench.parse_args(["--fair-optimization-suite"])
    rows = [
        _academic_test_row("builtin_cpu", 10.0),
        _academic_test_row("builtin_update_warm", 4.0),
        _academic_test_row("fair_fast_fixed1_iter10", 5.0, ref_err=1e-7),
    ]

    summary = osqp_bench.best_fair_cuda_row(rows, args)

    assert summary["status"] == "no_win"
    assert summary["speedup_vs_cpu_warm"] == 0.8
    assert summary["needed_speedup_to_match_cpu_warm"] == 1.25


def test_runtime_benchmark_best_fair_row_selects_winner():
    args = osqp_bench.parse_args(["--fair-optimization-suite"])
    rows = [
        _academic_test_row("builtin_cpu", 10.0),
        _academic_test_row("builtin_update_warm", 4.0),
        _academic_test_row("fair_fast_fixed1_iter10", 3.0, ref_err=1e-7),
        _academic_test_row("fair_fast_fixed1_iter20", 2.0, ref_err=1e-7),
    ]

    summary = osqp_bench.best_fair_cuda_row(rows, args)

    assert summary["status"] == "win"
    assert summary["method"] == "Fair fast fixed CG 1 iter20"
    assert summary["speedup_vs_cpu_warm"] == 2.0


def test_runtime_benchmark_exports_academic_artifacts(monkeypatch):
    writes = {}

    def fake_mkdir(self, parents=False, exist_ok=False):
        return None

    def fake_write_text(self, text, encoding=None):
        writes[str(self)] = text
        return len(text)

    monkeypatch.setattr(Path, "mkdir", fake_mkdir)
    monkeypatch.setattr(Path, "write_text", fake_write_text)
    markdown_path = Path("benchmark_export_test_output") / "table.md"
    csv_path = Path("benchmark_export_test_output") / "table.csv"
    args = osqp_bench.parse_args(
        [
            "--export-academic-md",
            str(markdown_path),
            "--export-academic-csv",
            str(csv_path),
        ]
    )
    rows = [
        {
            "status": "ok",
            "backend": "builtin_cpu",
            "case": "random_spd",
            "n": 12,
            "median_ms": 10.0,
            "selected": "-",
            "cg_fix": None,
            "rho_upd": 0,
            "scale": 0,
            "polish": "-",
            "cache": "-",
            "prim_res": None,
            "dual_res": None,
            "ref_err": None,
            "cg_iters": 0,
            "setup_ms": None,
            "update_ms": None,
            "solve_ms": None,
        },
        {
            "status": "ok",
            "backend": "pygranso_torch",
            "case": "random_spd",
            "n": 12,
            "median_ms": 2.5,
            "selected": "sparse_cg",
            "cg_fix": None,
            "rho_upd": 0,
            "scale": 0,
            "polish": "-",
            "cache": "hit",
            "prim_res": 0.0,
            "dual_res": 1e-7,
            "ref_err": 1e-7,
            "cg_iters": 5,
            "setup_ms": None,
            "update_ms": None,
            "solve_ms": None,
        },
    ]

    osqp_bench.export_academic_artifacts(rows, args)

    markdown = writes[str(markdown_path)]
    csv_text = writes[str(csv_path)]
    assert "speedup_vs_cpu_osqp" in markdown
    assert "Torch warm/cache sparse-CG" in markdown
    assert "4.0" in csv_text


def test_runtime_benchmark_pygranso_repeat_reports_last_info(monkeypatch):
    solution = torch.ones((4, 1), dtype=torch.float64)
    info = {
        "linear_solver_selected": "sparse_cg",
        "linear_solver_auto_reason": "test",
        "objective": -1.0,
        "primal_residual": 1e-7,
        "dual_residual": 2e-7,
        "total_cg_iterations": 4,
        "sparse_setup_cache_hit": True,
        "estimated_dense_kkt_mb": 1.0,
        "estimated_sparse_nnz": 12,
    }
    calls = {"reset": 0, "solve": 0}

    def fake_reset():
        calls["reset"] += 1

    def fake_solve_qp(*_args, **_kwargs):
        calls["solve"] += 1
        return solution

    monkeypatch.setattr(osqp_bench, "resetOSQPWarmState", fake_reset)
    monkeypatch.setattr(osqp_bench, "solveQP", fake_solve_qp)
    monkeypatch.setattr(osqp_bench, "getLastOSQPInfo", lambda: info)
    args = osqp_bench.parse_args(
        ["--sizes", "4", "--repeats", "1", "--warmups", "1", "--reference", "off"]
    )

    row = osqp_bench.time_pygranso_repeat(4, args, "bound")

    assert row["backend"] == "pygranso_torch"
    assert row["selected"] == "sparse_cg"
    assert row["cache"] == "hit"
    assert row["cg_iters"] == 4
    assert calls == {"reset": 1, "solve": 2}


def test_solve_qp_reuses_torch_osqp_state_between_matching_calls(monkeypatch):
    captured_settings = []

    def fake_solve_osqp(
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
        settings = dict((options or {}).get("settings", {}))
        captured_settings.append(settings)
        solution = torch.ones((f.numel(), 1), dtype=torch.float64)
        return solution, {
            "state": {
                "x": solution.reshape(-1),
                "z": solution.reshape(-1),
                "y": solution.reshape(-1),
                "cg_x": solution.reshape(-1),
                "rho": torch.ones(f.numel(), dtype=torch.float64),
                "sparse_cache": {"kind": "bounds"},
            }
        }

    monkeypatch.setattr(solve_qp_module, "OSQP_WARM_STATE", None)
    monkeypatch.setattr(solve_qp_module, "OSQP_WARM_SIGNATURE", None)
    monkeypatch.setattr(solve_qp_module, "solve_osqp_torch_qp", fake_solve_osqp)
    H, f, A, b, LB, UB = simple_bounds_qp(dtype=torch.float64)
    options = {"algebra": "torch", "settings": {"linear_solver": "sparse_cg"}}

    first = solveQP(H, f, A, b, LB, UB, "osqp", torch.device("cpu"), True, options)
    second = solveQP(H, f, A, b, LB, UB, "osqp", torch.device("cpu"), True, options)

    assert first.shape == (2, 1)
    assert second.shape == (2, 1)
    assert captured_settings[0]["return_info"] is True
    assert captured_settings[0]["return_state"] is True
    assert "initial_state" not in captured_settings[0]
    assert captured_settings[1]["warm_start"] is True
    assert captured_settings[1]["initial_state"] is not None
    assert solve_qp_module.getLastOSQPInfo()["state"] is not None
    solve_qp_module.resetOSQPWarmState()
    assert solve_qp_module.getLastOSQPInfo() is None


@pytest.mark.skipif(not OSQP_AVAILABLE, reason="OSQP Python package is not installed")
def test_runtime_benchmark_random_spd_reports_reference_error():
    args = osqp_bench.parse_args(
        [
            "--cases",
            "random_spd",
            "--sizes",
            "8",
            "--repeats",
            "1",
            "--warmups",
            "0",
            "--max-iter",
            "50",
            "--check-termination",
            "5",
            "--reference",
            "auto",
        ]
    )

    row = osqp_bench.time_backend(8, "torch_auto", args, "random_spd")

    assert row["case"] == "random_spd"
    assert row["backend"] == "torch_auto"
    assert row["status"] == "ok"
    assert row["ref_err"] is not None
    assert row["ref_err"] >= 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("double_precision", [True, False])
def test_cuda_torch_backend_keeps_device_dtype_shape_and_avoids_numpy(
    monkeypatch, double_precision
):
    def fail_numpy_conversion(value, name):
        raise AssertionError(f"unexpected NumPy conversion for {name}")

    monkeypatch.setattr(
        "pygranso.private.osqpTorchAdapter._column_or_matrix_to_numpy",
        fail_numpy_conversion,
    )

    dtype = torch.float64 if double_precision else torch.float32
    H, f, A, b, LB, UB = simple_bound_qp(dtype=dtype, device="cuda")

    solution = solve_osqp_torch_qp(
        H,
        f,
        A,
        b,
        LB,
        UB,
        torch.device("cuda"),
        double_precision,
        options={"algebra": "torch"},
    )

    assert solution.shape == (1, 1)
    assert solution.device.type == "cuda"
    assert solution.dtype == dtype
    torch.testing.assert_close(
        solution, torch.tensor([[1.0]], device="cuda", dtype=dtype), atol=1e-5, rtol=1e-5
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_builtin_without_fallback_fails_clearly():
    H, f, A, b, LB, UB = simple_bounds_qp(device="cuda")

    with pytest.raises(OSQPCudaInteropUnavailableError, match="cuda_fallback"):
        solve_osqp_torch_qp(
            H,
            f,
            A,
            b,
            LB,
            UB,
            torch.device("cuda"),
            True,
            options={"algebra": "builtin", "cuda_fallback": False},
        )


def test_default_torch_linear_solver_is_auto():
    settings = adapter._normalize_torch_settings(adapter.DEFAULT_OSQP_SETTINGS, {})

    assert settings["linear_solver"] == "auto"


def test_small_dense_auto_selects_dense_and_solves():
    dtype = torch.float64
    H, f, A, b, LB, UB = simple_bound_qp(dtype=dtype)

    solution, info = solve_osqp_torch_qp(
        H,
        f,
        A,
        b,
        LB,
        UB,
        torch.device("cpu"),
        True,
        options={"algebra": "torch", "settings": {"return_info": True}},
    )

    torch.testing.assert_close(
        solution, torch.tensor([[1.0]], dtype=dtype), atol=1e-5, rtol=1e-5
    )
    assert info["linear_solver_requested"] == "auto"
    assert info["linear_solver_selected"] == "dense"
    assert info["linear_solver_auto_reason"] == "kkt_dim_below_dense_threshold"


def test_auto_judge_dense_medium_below_sparse_threshold():
    settings = adapter._normalize_torch_settings(
        adapter.DEFAULT_OSQP_SETTINGS,
        {},
    )
    n = 300
    P = torch.ones((n, n), dtype=torch.float64)

    selection = adapter._select_torch_linear_solver(
        P, None, n, torch.float64, settings
    )

    assert selection["estimated_kkt_dim"] == 2 * n
    assert selection["linear_solver_selected"] == "dense"
    assert selection["linear_solver_auto_reason"] == "conservative_dense_default"


def test_auto_judge_explicit_dense_and_sparse_override():
    n = 600
    indices = torch.arange(n)
    P = torch.sparse_coo_tensor(
        torch.stack((indices, indices)),
        torch.ones(n, dtype=torch.float64),
        (n, n),
        dtype=torch.float64,
    ).to_sparse_csr()

    dense_settings = adapter._normalize_torch_settings(
        {"linear_solver": "dense"}, {"linear_solver": "dense"}
    )
    sparse_settings = adapter._normalize_torch_settings(
        {"linear_solver": "sparse_cg"}, {"linear_solver": "sparse_cg"}
    )

    dense_selection = adapter._select_torch_linear_solver(
        P, None, n, torch.float64, dense_settings
    )
    sparse_selection = adapter._select_torch_linear_solver(
        P, None, n, torch.float64, sparse_settings
    )

    assert dense_selection["linear_solver_selected"] == "dense"
    assert dense_selection["linear_solver_auto_reason"] == "explicit_dense"
    assert sparse_selection["linear_solver_selected"] == "sparse_cg"
    assert sparse_selection["linear_solver_auto_reason"] == "explicit_sparse_cg"


def test_large_sparse_bounds_only_auto_selects_sparse_without_dense_ops(monkeypatch):
    dtype = torch.float64
    n = 600
    indices = torch.arange(n)
    H = torch.sparse_coo_tensor(
        torch.stack((indices, indices)),
        torch.ones(n, dtype=dtype),
        (n, n),
        dtype=dtype,
    )
    f = torch.zeros((n, 1), dtype=dtype)
    LB = -torch.ones((n, 1), dtype=dtype)
    UB = torch.ones((n, 1), dtype=dtype)

    def fail_eye(*args, **kwargs):
        raise AssertionError("auto sparse path should not build a dense identity")

    def fail_solve(*args, **kwargs):
        raise AssertionError("auto sparse path should not call torch.linalg.solve")

    monkeypatch.setattr(torch, "eye", fail_eye)
    monkeypatch.setattr(torch.linalg, "solve", fail_solve)

    solution, info = solve_osqp_torch_qp(
        H,
        f,
        None,
        None,
        LB,
        UB,
        torch.device("cpu"),
        True,
        options={
            "algebra": "torch",
            "settings": auto_settings(max_iter=1, check_termination=1, return_info=True),
        },
    )

    assert solution.shape == (n, 1)
    assert info["linear_solver_requested"] == "auto"
    assert info["linear_solver_selected"] == "sparse_cg"
    assert info["linear_solver_auto_reason"] == "large_sparse_problem"


def test_sparse_equality_plus_bounds_auto_selects_sparse_cg():
    dtype = torch.float64
    n = 600
    diag = torch.arange(n)
    H = torch.sparse_coo_tensor(
        torch.stack((diag, diag)),
        torch.ones(n, dtype=dtype),
        (n, n),
        dtype=dtype,
    )
    A = torch.sparse_coo_tensor(
        torch.stack((torch.zeros(n, dtype=torch.long), diag)),
        torch.ones(n, dtype=dtype),
        (1, n),
        dtype=dtype,
    )
    f = torch.zeros((n, 1), dtype=dtype)
    LB = -torch.ones((n, 1), dtype=dtype)
    UB = torch.ones((n, 1), dtype=dtype)

    solution, info = solve_osqp_torch_qp(
        H,
        f,
        A,
        0.0,
        LB,
        UB,
        torch.device("cpu"),
        True,
        options={
            "algebra": "torch",
            "settings": auto_settings(max_iter=1, check_termination=1, return_info=True),
        },
    )

    assert solution.shape == (n, 1)
    assert info["linear_solver_requested"] == "auto"
    assert info["linear_solver_selected"] == "sparse_cg"
    assert info["linear_solver_auto_reason"] == "large_sparse_problem"


def test_auto_sparse_failure_retries_dense_when_safe(monkeypatch):
    captured = {}

    def fail_sparse(*args, **kwargs):
        raise NotImplementedError("sparse operation is not implemented")

    def fake_dense(P, q, A, l, u, settings):
        captured["settings"] = settings
        solution = torch.zeros((q.numel(), 1), device=q.device, dtype=q.dtype)
        return solution, {
            "linear_solver_requested": settings["linear_solver_requested"],
            "linear_solver_selected": settings["linear_solver_selected"],
            "linear_solver_auto_reason": settings["linear_solver_auto_reason"],
        }

    monkeypatch.setattr(adapter, "solve_torch_osqp_from_qp", fail_sparse)
    monkeypatch.setattr(adapter, "solve_torch_osqp", fake_dense)

    H, f, A, b, LB, UB = simple_bound_qp(dtype=torch.float64)
    solution, info = solve_osqp_torch_qp(
        H,
        f,
        A,
        b,
        LB,
        UB,
        torch.device("cpu"),
        True,
        options={
            "algebra": "torch",
            "settings": auto_settings(
                return_info=True,
                linear_solver_auto_min_kkt_dim=1,
                linear_solver_auto_sparse_min_kkt_dim=1,
                linear_solver_auto_max_density=1.0,
            ),
        },
    )

    assert solution.shape == (1, 1)
    assert captured["settings"]["linear_solver"] == "dense"
    assert info["linear_solver_requested"] == "auto"
    assert info["linear_solver_selected"] == "dense"
    assert info["linear_solver_auto_reason"].startswith(
        "sparse_cg_failed_retry_dense"
    )


def test_sparse_cg_accepts_sparse_csr_inputs_without_to_dense(monkeypatch):
    H, f, A, b, LB, UB = simple_bound_qp(dtype=torch.float64)
    H = H.to_sparse_csr()

    def fail_to_dense(self):
        raise AssertionError("sparse_cg path should not densify sparse tensors")

    monkeypatch.setattr(torch.Tensor, "to_dense", fail_to_dense)

    solution = solve_osqp_torch_qp(
        H,
        f,
        A,
        b,
        LB,
        UB,
        torch.device("cpu"),
        True,
        options={
            "algebra": "torch",
            "settings": sparse_cg_settings(),
        },
    )

    torch.testing.assert_close(
        solution, torch.tensor([[1.0]], dtype=torch.float64), atol=1e-5, rtol=1e-5
    )


def test_sparse_cg_does_not_call_torch_linalg_solve(monkeypatch):
    H, f, A, b, LB, UB = simple_bound_qp(dtype=torch.float64)

    def fail_solve(*args, **kwargs):
        raise AssertionError("sparse_cg path should not call torch.linalg.solve")

    monkeypatch.setattr(torch.linalg, "solve", fail_solve)

    solution = solve_osqp_torch_qp(
        H,
        f,
        A,
        b,
        LB,
        UB,
        torch.device("cpu"),
        True,
        options={
            "algebra": "torch",
            "settings": sparse_cg_settings(),
        },
    )

    torch.testing.assert_close(
        solution, torch.tensor([[1.0]], dtype=torch.float64), atol=1e-5, rtol=1e-5
    )


def test_sparse_cg_does_not_build_dense_identity_for_bounds(monkeypatch):
    H, f, A, b, LB, UB = simple_bound_qp(dtype=torch.float64)

    def fail_eye(*args, **kwargs):
        raise AssertionError("sparse_cg path should not build a dense identity")

    monkeypatch.setattr(torch, "eye", fail_eye)

    solution = solve_osqp_torch_qp(
        H,
        f,
        A,
        b,
        LB,
        UB,
        torch.device("cpu"),
        True,
        options={
            "algebra": "torch",
            "settings": sparse_cg_settings(),
        },
    )

    torch.testing.assert_close(
        solution, torch.tensor([[1.0]], dtype=torch.float64), atol=1e-5, rtol=1e-5
    )


def test_sparse_cg_keeps_bound_constraints_matrix_free():
    dtype = torch.float64
    P = torch.sparse_coo_tensor(
        torch.tensor([[0, 1], [0, 1]]),
        torch.ones(2, dtype=dtype),
        (2, 2),
        dtype=dtype,
    ).to_sparse_csr()
    operator = BoundConstrainedOSQPOperator(
        P, None, 2, torch.device("cpu"), dtype
    )

    v = torch.tensor([3.0, 4.0], dtype=dtype)
    y = torch.tensor([5.0, 6.0], dtype=dtype)

    assert operator.uses_matrix_free_bounds is True
    torch.testing.assert_close(operator.A_mv(v), v)
    torch.testing.assert_close(operator.AT_mv(y), y)


def test_sparse_cg_reduced_operator_uses_spmv_not_ata():
    class SpyOperator:
        def __init__(self):
            self.calls = {"P": 0, "A": 0, "AT": 0}

        def P_mv(self, vector):
            self.calls["P"] += 1
            return 2.0 * vector

        def A_mv(self, vector):
            self.calls["A"] += 1
            return torch.stack((vector[0] + vector[1], vector[1]))

        def AT_mv(self, vector):
            self.calls["AT"] += 1
            return torch.stack((vector[0], vector[0] + vector[1]))

    operator = SpyOperator()
    rho_vec = torch.tensor([0.5, 2.0], dtype=torch.float64)
    v = torch.tensor([1.0, 3.0], dtype=torch.float64)

    out = reduced_system_matvec(operator, 1e-6, rho_vec, v)

    expected_Av = torch.tensor([4.0, 3.0], dtype=torch.float64)
    expected = 2.0 * v + 1e-6 * v + torch.tensor([2.0, 8.0], dtype=torch.float64)
    torch.testing.assert_close(out, expected)
    assert operator.calls == {"P": 1, "A": 1, "AT": 1}
    torch.testing.assert_close(expected_Av, torch.tensor([4.0, 3.0], dtype=torch.float64))


def test_sparse_cg_preconditioner_diagonal_matches_dense_reference():
    dtype = torch.float64
    P_dense = torch.tensor([[4.0, 1.0], [1.0, 3.0]], dtype=dtype)
    A_dense = torch.tensor([[1.0, 2.0], [0.0, -3.0]], dtype=dtype)
    rho_vec = torch.tensor([0.25, 2.0], dtype=dtype)
    sigma = 1e-6
    operator = ExplicitOSQPOperator(
        P_dense.to_sparse_csr(),
        A_dense.to_sparse_csr(),
        torch.device("cpu"),
        dtype,
    )

    diag = jacobi_preconditioner_diagonal(operator, sigma, rho_vec)
    dense_reference = torch.diagonal(P_dense) + sigma + torch.diagonal(
        A_dense.T @ torch.diag(rho_vec) @ A_dense
    )

    torch.testing.assert_close(diag, dense_reference)


def test_sparse_cg_reports_inner_and_outer_residuals_separately():
    dtype = torch.float64
    P = torch.eye(2, dtype=dtype).to_sparse_csr()
    q = torch.zeros(2, dtype=dtype)
    A = torch.ones((1, 2), dtype=dtype).to_sparse_csr()
    b = torch.tensor([1.0], dtype=dtype)
    LB = torch.zeros(2, dtype=dtype)
    UB = torch.ones(2, dtype=dtype)
    settings = sparse_cg_settings(return_info=True)

    solution, info = solve_torch_osqp_from_qp(P, q, A, b, LB, UB, settings)

    torch.testing.assert_close(
        solution, 0.5 * torch.ones((2, 1), dtype=dtype), atol=1e-5, rtol=1e-5
    )
    assert info["linear_solver"] == "sparse_cg"
    assert info["primal_residual"] <= info["eps_primal"] * 10
    assert info["dual_residual"] <= 1e-5
    assert info["total_cg_iterations"] > 0
    assert info["last_cg_relative_residual"] is not None
    assert info["last_cg_converged"] in {True, False}
    assert info["torch_compile_admm"] is False
    assert info["torch_compile_admm_status"] == "disabled"
    assert info["timing_setup_ms"] >= 0
    assert info["timing_cg_ms"] >= 0
    assert info["timing_admm_update_ms"] >= 0
    assert info["timing_residual_ms"] >= 0


def test_sparse_cg_uses_larger_rho_for_equalities():
    dtype = torch.float64
    P = torch.eye(2, dtype=dtype).to_sparse_csr()
    q = torch.zeros(2, dtype=dtype)
    A = torch.ones((1, 2), dtype=dtype).to_sparse_csr()
    b = torch.tensor([1.0], dtype=dtype)
    LB = torch.zeros(2, dtype=dtype)
    UB = torch.ones(2, dtype=dtype)
    settings = sparse_cg_settings(max_iter=1, check_termination=1, return_info=True)

    _solution, info = solve_torch_osqp_from_qp(P, q, A, b, LB, UB, settings)

    assert info["rho_min"] == pytest.approx(0.1)
    assert info["rho_max"] == pytest.approx(100.0)


def test_sparse_cg_warm_start_returns_reusable_state():
    dtype = torch.float64
    P = torch.eye(2, dtype=dtype).to_sparse_csr()
    q = torch.tensor([-1.0, -2.0], dtype=dtype)
    LB = torch.zeros(2, dtype=dtype)
    UB = torch.ones(2, dtype=dtype)
    first_settings = sparse_cg_settings(
        max_iter=5,
        check_termination=1,
        return_info=True,
        return_state=True,
    )

    _first_solution, first_info = solve_torch_osqp_from_qp(
        P, q, None, None, LB, UB, first_settings
    )
    second_settings = sparse_cg_settings(
        max_iter=5,
        check_termination=1,
        return_info=True,
        return_state=True,
        warm_start=True,
        initial_state=first_info["state"],
    )
    second_solution, second_info = solve_torch_osqp_from_qp(
        P, q, None, None, LB, UB, second_settings
    )

    assert {"x", "z", "y", "cg_x", "rho", "sparse_cache"} <= second_info[
        "state"
    ].keys()
    assert first_info["state"]["sparse_cache"] is not None
    assert second_info["sparse_setup_cache_hit"] is True
    assert torch.all(torch.isfinite(second_solution))


def test_sparse_cg_auto_fixed_iters_avoids_equality_coupled_systems():
    dtype = torch.float64
    P = torch.eye(2, dtype=dtype).to_sparse_csr()
    q = torch.tensor([-1.0, -2.0], dtype=dtype)
    LB = torch.zeros(2, dtype=dtype)
    UB = torch.ones(2, dtype=dtype)
    bound_settings = sparse_cg_settings(
        max_iter=2,
        check_termination=1,
        return_info=True,
        cg_fixed_iters="auto",
    )

    _bound_solution, bound_info = solve_torch_osqp_from_qp(
        P, q, None, None, LB, UB, bound_settings
    )

    A = torch.ones((1, 2), dtype=dtype).to_sparse_csr()
    b = torch.tensor([1.0], dtype=dtype)
    equality_settings = sparse_cg_settings(
        max_iter=2,
        check_termination=1,
        return_info=True,
        cg_fixed_iters="auto",
    )
    _equality_solution, equality_info = solve_torch_osqp_from_qp(
        P, q, A, b, LB, UB, equality_settings
    )

    assert bound_info["cg_fixed_iters"] == "auto"
    assert bound_info["cg_fixed_iters_selected"] == 1
    assert equality_info["cg_fixed_iters"] == "auto"
    assert equality_info["cg_fixed_iters_selected"] is None


def test_sparse_cg_compile_admm_uses_compile_wrapper(monkeypatch):
    dtype = torch.float64
    P = torch.eye(2, dtype=dtype).to_sparse_csr()
    q = torch.tensor([-1.0, -2.0], dtype=dtype)
    LB = torch.zeros(2, dtype=dtype)
    UB = torch.ones(2, dtype=dtype)
    captured = {}

    def fake_compile(fn):
        captured["compiled"] = fn
        return fn

    real_find_spec = torch_osqp.importlib.util.find_spec

    def fake_find_spec(name):
        if name == "triton":
            return object()
        return real_find_spec(name)

    monkeypatch.setattr(torch_osqp, "_COMPILED_ADMM_UPDATE", None)
    monkeypatch.setattr(torch_osqp, "_COMPILED_ADMM_ERROR", None)
    monkeypatch.setattr(torch_osqp.importlib.util, "find_spec", fake_find_spec)
    monkeypatch.setattr(torch_osqp.torch, "compile", fake_compile, raising=False)
    settings = sparse_cg_settings(
        max_iter=2,
        check_termination=1,
        return_info=True,
        torch_compile_admm=True,
    )

    _solution, info = solve_torch_osqp_from_qp(P, q, None, None, LB, UB, settings)

    assert captured["compiled"] is not None
    assert info["torch_compile_admm"] is True
    assert info["torch_compile_admm_status"] == "enabled"


def test_sparse_cg_adaptive_rho_reports_positive_updates():
    dtype = torch.float64
    P = torch.eye(2, dtype=dtype).to_sparse_csr()
    q = torch.tensor([-8.0, -0.1], dtype=dtype)
    LB = torch.zeros(2, dtype=dtype)
    UB = torch.ones(2, dtype=dtype)
    settings = sparse_cg_settings(
        max_iter=4,
        check_termination=1,
        return_info=True,
        adaptive_rho=True,
        rho_update_interval=1,
        rho_update_tolerance=1.0,
    )

    _solution, info = solve_torch_osqp_from_qp(P, q, None, None, LB, UB, settings)

    assert info["adaptive_rho"] is True
    assert info["rho_updates"] >= 1
    assert info["rho_min"] > 0
    assert info["rho_max"] > 0


def test_sparse_cg_ruiz_scaling_unscales_solution_and_reports_original_residuals():
    dtype = torch.float64
    P = torch.diag(torch.tensor([1.0, 100.0], dtype=dtype)).to_sparse_csr()
    q = torch.tensor([-1.0, -1.0], dtype=dtype)
    LB = -torch.ones(2, dtype=dtype)
    UB = torch.ones(2, dtype=dtype)
    settings = sparse_cg_settings(
        max_iter=200,
        check_termination=5,
        return_info=True,
        scaling=2,
    )

    solution, info = solve_torch_osqp_from_qp(P, q, None, None, LB, UB, settings)

    torch.testing.assert_close(
        solution,
        torch.tensor([[1.0], [0.01]], dtype=dtype),
        atol=1e-4,
        rtol=1e-4,
    )
    assert info["scaling_applied"] is True
    assert info["scaling_passes"] == 2
    assert "scaled_primal_residual" in info
    assert info["primal_residual"] <= info["eps_primal"] * 10


def test_sparse_cg_polishing_accepts_only_nonworse_kkt_metric():
    dtype = torch.float64
    P = torch.eye(1, dtype=dtype).to_sparse_csr()
    q = torch.tensor([-2.0], dtype=dtype)
    LB = torch.zeros(1, dtype=dtype)
    UB = torch.ones(1, dtype=dtype)
    settings = sparse_cg_settings(
        max_iter=50,
        check_termination=1,
        return_info=True,
        polishing=True,
        polish_refine_iter=1,
    )

    solution, info = solve_torch_osqp_from_qp(P, q, None, None, LB, UB, settings)

    torch.testing.assert_close(
        solution, torch.tensor([[1.0]], dtype=dtype), atol=1e-5, rtol=1e-5
    )
    assert info["polishing"] is True
    if info["polishing_success"]:
        assert info["polishing_new_score"] <= info["polishing_old_score"]


def test_sparse_cg_large_sparse_setup_memory_smoke():
    dtype = torch.float64
    n = 1000
    indices = torch.arange(n)
    P = torch.sparse_coo_tensor(
        torch.stack((indices, indices)),
        torch.ones(n, dtype=dtype),
        (n, n),
        dtype=dtype,
    ).to_sparse_csr()
    q = torch.zeros(n, dtype=dtype)
    LB = -torch.ones(n, dtype=dtype)
    UB = torch.ones(n, dtype=dtype)
    settings = sparse_cg_settings(max_iter=1, check_termination=1, return_info=True)

    solution, info = solve_torch_osqp_from_qp(P, q, None, None, LB, UB, settings)

    assert solution.shape == (n, 1)
    assert info["sparse_storage_nnz"] == n
    assert info["dense_kkt_entries"] == (2 * n) ** 2
    assert info["sparse_storage_nnz"] < info["dense_kkt_entries"] // 100


@pytest.mark.skipif(not OSQP_AVAILABLE, reason="OSQP Python package is not installed")
@pytest.mark.parametrize(
    ("case", "n"),
    [("equality", 8), ("random_spd", 8), ("random_spd", 24)],
)
def test_torch_dense_and_sparse_cg_match_builtin_on_seeded_sparse_qps(case, n):
    dtype = torch.float64
    H, f, A, b, LB, UB = osqp_bench.make_qp_case(
        case, n, device="cpu", dtype=dtype, seed=11, density=0.10
    )
    builtin_settings = {
        "eps_abs": 1e-8,
        "eps_rel": 1e-8,
        "max_iter": 10000,
        "polishing": False,
        "verbose": False,
    }
    torch_settings = {
        "return_info": True,
        "max_iter": 1000,
        "check_termination": 10,
        "eps_abs": 1e-8,
        "eps_rel": 1e-8,
        "cg_rtol": 1e-10,
        "cg_max_iter": 200,
    }

    builtin_solution = solve_osqp_torch_qp(
        H,
        f,
        A,
        b,
        LB,
        UB,
        torch.device("cpu"),
        True,
        options={"algebra": "builtin", "settings": builtin_settings},
    )
    dense_solution, dense_info = solve_osqp_torch_qp(
        H,
        f,
        A,
        b,
        LB,
        UB,
        torch.device("cpu"),
        True,
        options={
            "algebra": "torch",
            "settings": {"linear_solver": "dense", **torch_settings},
        },
    )
    sparse_solution, sparse_info = solve_osqp_torch_qp(
        H,
        f,
        A,
        b,
        LB,
        UB,
        torch.device("cpu"),
        True,
        options={
            "algebra": "torch",
            "settings": {"linear_solver": "sparse_cg", **torch_settings},
        },
    )

    torch.testing.assert_close(dense_solution, builtin_solution, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(sparse_solution, builtin_solution, atol=1e-4, rtol=1e-4)
    assert dense_info["linear_solver_selected"] == "dense"
    assert sparse_info["linear_solver_selected"] == "sparse_cg"
    assert sparse_info["uses_matrix_free_bounds"] is True


def test_pygranso_options_include_osqp_backend_policy():
    user_opts = pygransoStruct()
    user_opts.osqp_algebra = "torch"
    user_opts.osqp_cuda_fallback = True
    user_opts.osqp_settings = {"eps_abs": 1e-8, "eps_rel": 1e-8, "verbose": False}

    opts = pygransoOptions(2, user_opts)

    assert opts.osqp_algebra == "torch"
    assert opts.osqp_cuda_fallback is True
    assert opts.osqp_settings["eps_abs"] == 1e-8


def test_sparse_cache_rejects_equal_nnz_with_different_indices():
    P = torch.eye(3, dtype=torch.float64).to_sparse_csr()
    A1 = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64
    ).to_sparse_csr()
    A2 = torch.tensor(
        [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64
    ).to_sparse_csr()
    first = ExplicitOSQPOperator(P, A1, P.device, P.dtype)
    second = ExplicitOSQPOperator(P, A2, P.device, P.dtype, first.sparse_cache())

    assert A1._nnz() == A2._nnz()
    assert second.cache_hit is False


def test_sparse_cache_refreshes_numeric_transpose_and_diagonal():
    P1 = torch.diag(torch.tensor([1.0, 2.0, 3.0])).to_sparse_csr()
    A1 = torch.tensor([[1.0, 0.0, 2.0]], dtype=torch.float64).to_sparse_csr()
    first = ExplicitOSQPOperator(P1, A1, P1.device, P1.dtype)
    cache = first.sparse_cache()

    P2 = torch.diag(torch.tensor([4.0, 5.0, 6.0])).to_sparse_csr()
    A2 = torch.tensor([[3.0, 0.0, 7.0]], dtype=torch.float64).to_sparse_csr()
    second = ExplicitOSQPOperator(P2, A2, P2.device, P2.dtype, cache)

    assert second.cache_hit is True
    torch.testing.assert_close(second.diag_P(), torch.tensor([4.0, 5.0, 6.0]))
    torch.testing.assert_close(second.AT.to_dense(), A2.to_dense().T)


def test_parametric_matrix_sequence_changes_values_not_structure():
    args = osqp_bench.parse_args(
        ["--parametric-sequence", "--parametric-matrix-values", "--seed", "7"]
    )
    sequence = osqp_bench.make_parametric_qp_sequence(
        "equality", 12, args, dtype=torch.float64, count=3
    )
    H0, _f0, A0, _b0, _LB0, _UB0 = sequence[0]
    H1, _f1, A1, _b1, _LB1, _UB1 = sequence[1]
    H0_csr = H0.to_sparse_csr()
    H1_csr = H1.to_sparse_csr()
    A0_csr = A0.to_sparse_csr()
    A1_csr = A1.to_sparse_csr()

    assert torch.equal(H0_csr.crow_indices(), H1_csr.crow_indices())
    assert torch.equal(H0_csr.col_indices(), H1_csr.col_indices())
    assert torch.equal(A0_csr.crow_indices(), A1_csr.crow_indices())
    assert torch.equal(A0_csr.col_indices(), A1_csr.col_indices())
    assert not torch.equal(H0_csr.values(), H1_csr.values())
    assert not torch.equal(A0_csr.values(), A1_csr.values())
    assert torch.min(torch.linalg.eigvalsh(H1.to_dense())) > 0


def test_academic_accuracy_gate_uses_residuals_and_objective_gap():
    row = {
        "status": "ok",
        "relative_objective_gap": 5e-6,
        "prim_res": 2e-6,
        "eps_prim": 1e-5,
        "dual_res": 3e-6,
        "eps_dual": 1e-5,
        "ref_err": 2e-5,
    }
    assert osqp_bench.row_accuracy_passes(row) is True
    row["dual_res"] = 2e-5
    assert osqp_bench.row_accuracy_passes(row) is False


def test_bootstrap_speedup_interval_requires_repeated_samples():
    assert osqp_bench.bootstrap_speedup_interval([1.0], [0.5]) is None
    interval = osqp_bench.bootstrap_speedup_interval(
        [10.0, 11.0, 9.0, 10.5, 9.5], [5.0, 5.5, 4.5, 5.1, 4.9]
    )
    assert interval[0] > 1.0
    assert interval[1] > interval[0]


def test_osqp_trace_accepts_scalar_rhs():
    solve_qp_module.beginOSQPTrace(capture_data=True)
    solve_qp_module._record_osqp_qp(
        torch.eye(2),
        torch.zeros((2, 1)),
        torch.ones((1, 2)),
        1,
        -torch.ones((2, 1)),
        torch.ones((2, 1)),
    )
    trace = solve_qp_module.endOSQPTrace()

    assert len(trace) == 1
    assert torch.is_tensor(trace[0]["qp"][3])


def test_cuda_graph_rejects_cpu_inputs():
    P = torch.eye(3, dtype=torch.float64).to_sparse_csr()
    settings = sparse_cg_settings(
        max_iter=10,
        check_termination=10,
        cg_fixed_iters=1,
        cuda_graph=True,
        return_info=True,
    )
    with pytest.raises(ValueError, match="requires a CUDA tensor"):
        solve_torch_osqp_from_qp(
            P,
            torch.zeros(3),
            None,
            None,
            -torch.ones(3),
            torch.ones(3),
            settings,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_graph_replays_and_matches_eager_fixed_work():
    device = torch.device("cuda")
    P = torch.eye(8, device=device, dtype=torch.float64).to_sparse_csr()
    q1 = -torch.ones(8, device=device, dtype=torch.float64)
    q2 = q1 + 0.01 * torch.arange(8, device=device, dtype=torch.float64)
    LB = -torch.ones(8, device=device, dtype=torch.float64)
    UB = torch.ones(8, device=device, dtype=torch.float64)
    common = dict(
        max_iter=12,
        check_termination=12,
        cg_fixed_iters=1,
        cg_check_interval=12,
        warm_start=True,
        return_state=True,
        return_info=True,
    )
    graph_settings = sparse_cg_settings(cuda_graph=True, **common)
    eager_settings = sparse_cg_settings(cuda_graph=False, **common)

    _graph_first, graph_first_info = solve_torch_osqp_from_qp(
        P, q1, None, None, LB, UB, graph_settings
    )
    graph_settings = dict(graph_settings, initial_state=graph_first_info["state"])
    graph_second, graph_second_info = solve_torch_osqp_from_qp(
        P, q2, None, None, LB, UB, graph_settings
    )
    _eager_first, eager_first_info = solve_torch_osqp_from_qp(
        P, q1, None, None, LB, UB, eager_settings
    )
    eager_settings = dict(eager_settings, initial_state=eager_first_info["state"])
    eager_second, _eager_second_info = solve_torch_osqp_from_qp(
        P, q2, None, None, LB, UB, eager_settings
    )

    assert graph_second_info["cuda_graph_cache_hit"] is True
    assert graph_second_info["cuda_graph_status"] == "replayed"
    torch.testing.assert_close(graph_second, eager_second, atol=1e-10, rtol=1e-10)
