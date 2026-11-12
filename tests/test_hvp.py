"""Tests 3 and 4: Hessian-vector products and power iteration."""

import math

import torch
from conftest import make_mlp_problem

from hypergrad.functional import tree_flatten, tree_unflatten
from hypergrad.hvp import hvp, power_iteration


def test_hvp_matches_explicit_hessian():
    problem, params, train, _ = make_mlp_problem(seed=0)
    hp = {"log_wd": torch.tensor([math.log(1e-2), math.log(1e-1)])}

    def f(prm):
        return problem.train_loss(prm, hp, train)

    flat = tree_flatten(params)
    hess = torch.autograd.functional.hessian(lambda v: f(tree_unflatten(v, params)), flat)
    g = torch.Generator().manual_seed(1)
    for _ in range(3):
        v = torch.randn(flat.shape, generator=g)
        got = tree_flatten(hvp(f, params, tree_unflatten(v, params)))
        torch.testing.assert_close(got, hess @ v, rtol=1e-10, atol=1e-12)


def random_spd(eigs: list[float], seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(len(eigs), len(eigs), generator=g))
    return q @ torch.diag(torch.tensor(eigs)) @ q.T


def test_power_iteration_on_known_quadratic():
    a = random_spd([10.0, 4.0, 2.0, 1.0, 0.5], seed=0)
    params = {"w": torch.zeros(5)}

    def f(prm):
        return 0.5 * prm["w"] @ a @ prm["w"]

    eig, vec = power_iteration(lambda v: hvp(f, params, v), params, iters=200, tol=1e-12)
    assert abs(eig - 10.0) / 10.0 < 1e-6
    hv = a @ vec["w"]
    torch.testing.assert_close(hv, eig * vec["w"], rtol=1e-4, atol=1e-5)


def test_power_iteration_on_hessian_block():
    """Block mode returns lambda_max of the diagonal block H_bb, not of the full Hessian."""
    a = random_spd([3.0, 1.0], seed=1)
    b = random_spd([7.0, 2.0, 0.5], seed=2)
    params = {"a": torch.zeros(2), "b": torch.zeros(3)}

    def f(prm):
        coupling = 5.0 * prm["a"][0] * prm["b"][0]  # off-diagonal block, ignored per block
        return 0.5 * prm["a"] @ a @ prm["a"] + 0.5 * prm["b"] @ b @ prm["b"] + coupling

    hvp_fn = lambda v: hvp(f, params, v)  # noqa: E731
    eig_a, _ = power_iteration(hvp_fn, params, iters=300, tol=1e-12, block=["a"])
    eig_b, _ = power_iteration(hvp_fn, params, iters=300, tol=1e-12, block=["b"])
    assert abs(eig_a - 3.0) < 1e-6
    assert abs(eig_b - 7.0) < 1e-6
