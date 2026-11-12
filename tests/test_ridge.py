"""Test 1: ridge regression, where w*(lambda) and the hypergradient have a closed form.

    L_train(w, lambda) = 1/2 ||X w - y||^2 + lambda/2 ||w||^2
    L_val(w)           = 1/2 ||X_v w - y_v||^2
    w*(lambda)         = (X^T X + lambda I)^{-1} X^T y
    dL_val/dlambda     = -(X_v^T (X_v w* - y_v))^T (X^T X + lambda I)^{-1} w*

Hyperparameters live in log space, so the reference is lambda * dL_val/dlambda.
"""

import math

import pytest
import torch

from hypergrad.inner import DiffSGD
from hypergrad.methods import implicit_hypergrad, unrolled_hypergrad
from hypergrad.problem import BilevelProblem

LAM = 2.0


def per_sample_loss(params, batch):
    return 0.5 * (batch["x"] @ params["w"] - batch["y"]) ** 2


@pytest.fixture
def ridge():
    g = torch.Generator().manual_seed(0)
    d, n, n_val = 6, 60, 40
    # Column scales spread the spectrum (condition number ~ 1e2) so that short unrolls are
    # visibly biased while long unrolls still converge.
    scales = torch.logspace(0, -1, d)
    x = torch.randn(n + n_val, d, generator=g) * scales
    w_true = torch.randn(d, generator=g)
    y = x @ w_true + 0.3 * torch.randn(n + n_val, generator=g)
    train = {"x": x[:n], "y": y[:n]}
    val = {"x": x[n:], "y": y[n:]}

    a = train["x"].T @ train["x"] + LAM * torch.eye(d)
    w_star = torch.linalg.solve(a, train["x"].T @ train["y"])
    r = val["x"].T @ (val["x"] @ w_star - val["y"])
    dlam = -(r @ torch.linalg.solve(a, w_star))
    eigs = torch.linalg.eigvalsh(a)

    problem = BilevelProblem(per_sample_loss, {"w": 0}, reduction="sum")
    return {
        "problem": problem,
        "train": train,
        "val": val,
        "w_star": w_star,
        "ref": (LAM * dlam).item(),  # d L_val / d log(lambda)
        "L": eigs.max().item(),
        "mu": eigs.min().item(),
        "d": d,
    }


def hp_for(ridge):
    return {
        "log_wd": torch.tensor([math.log(LAM)]),
        "log_lr": torch.tensor([math.log(1.0 / ridge["L"])]),
    }


def rel_err(a, b):
    return abs(a - b) / abs(b)


def test_ift_cg_matches_closed_form(ridge):
    res = implicit_hypergrad(
        ridge["problem"],
        {"w": ridge["w_star"]},
        hp_for(ridge),
        ridge["train"],
        ridge["val"],
        solver="cg",
        iters=ridge["d"],  # CG is exact in d steps on a d-dimensional quadratic
    )
    assert rel_err(res.hypergrad["log_wd"].item(), ridge["ref"]) < 1e-4
    assert res.hypergrad["log_lr"].item() == 0.0


def test_ift_neumann_converges_monotonically(ridge):
    alpha = 1.0 / ridge["L"]
    errors = []
    for iters in [0, 5, 20, 100, 300, 1000]:
        res = implicit_hypergrad(
            ridge["problem"],
            {"w": ridge["w_star"]},
            hp_for(ridge),
            ridge["train"],
            ridge["val"],
            solver="neumann",
            iters=iters,
            alpha=alpha,
        )
        errors.append(rel_err(res.hypergrad["log_wd"].item(), ridge["ref"]))
    # With alpha = 1/L every eigenvalue of (I - alpha H) lies in [0, 1): the truncated
    # series approaches H^{-1} from below in every eigendirection.
    assert all(e1 >= e2 for e1, e2 in zip(errors, errors[1:], strict=False)), errors
    assert errors[-1] < 1e-4, errors


def unrolled_error(ridge, k):
    problem = ridge["problem"]
    opt = DiffSGD(problem, momentum=0.0)
    w0 = {"w": torch.zeros(ridge["d"])}
    res = unrolled_hypergrad(
        opt, w0, opt.init_state(w0), hp_for(ridge), [ridge["train"]] * k, ridge["val"]
    )
    return rel_err(res.hypergrad["log_wd"].item(), ridge["ref"])


def test_unrolled_long_horizon_matches_closed_form(ridge):
    assert unrolled_error(ridge, 2000) < 1e-2


def test_unrolled_short_horizon_is_biased(ridge):
    errors = {k: unrolled_error(ridge, k) for k in [5, 20, 100, 500, 2000]}
    print("\nridge: relative error of the unrolled hypergradient vs K")
    for k, e in errors.items():
        print(f"  K={k:5d}  rel_err={e:.3e}")
    assert errors[5] > 0.1
    assert errors[2000] < errors[500] < errors[100] < errors[5]
