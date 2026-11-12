"""Implicit-function-theorem hypergradients.

At a stationary point grad_w L_train(w*, lambda) = 0 the implicit function theorem gives

    d L_val / d lambda = - grad^2_{lambda w} L_train  [grad^2_w L_train]^{-1}  grad_w L_val

(plus the direct term dL_val/dlambda, which is zero here). The inverse-Hessian-vector product
q = H^{-1} grad_w L_val is approximated with conjugate gradient or a truncated Neumann
series; only Hessian-vector products are needed, never the trajectory.

Hyperparameters that do not appear in L_train (learning rate, momentum) get an implicit
hypergradient of exactly zero: they change how training reaches w*, not w* itself.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from ..functional import Params, tree_add, tree_detach, tree_dot, tree_scale
from ..hvp import hvp
from ..problem import Batches, BilevelProblem, HParams

HvpFn = Callable[[Params], Params]


def inverse_hvp_cg(
    hvp_fn: HvpFn, v: Params, iters: int, damping: float = 0.0, tol: float = 1e-12
) -> Params:
    """Approximate (H + damping I)^{-1} v with ``iters`` conjugate-gradient iterations, x_0 = 0."""

    def a(u: Params) -> Params:
        hu = hvp_fn(u)
        return tree_add(hu, u, damping) if damping else hu

    x = {k: torch.zeros_like(t) for k, t in v.items()}
    r = dict(v)
    p = dict(v)
    rs = tree_dot(r, r)
    for _ in range(iters):
        if rs.sqrt() <= tol:
            break
        ap = a(p)
        alpha = rs / tree_dot(p, ap)
        x = tree_add(x, p, alpha)
        r = tree_add(r, ap, -alpha)
        rs_new = tree_dot(r, r)
        p = tree_add(r, p, rs_new / rs)
        rs = rs_new
    return x


def inverse_hvp_neumann(hvp_fn: HvpFn, v: Params, iters: int, alpha: float) -> Params:
    """H^{-1} v ~= alpha * sum_{j=0}^{iters} (I - alpha H)^j v   (Lorraine et al., 2020).

    Converges when 0 < alpha < 2 / lambda_max(H) and H is positive definite.
    """
    cur = dict(v)
    acc = dict(v)
    for _ in range(iters):
        cur = tree_add(cur, hvp_fn(cur), -alpha)
        acc = tree_add(acc, cur)
    return tree_scale(acc, alpha)


@dataclass
class ImplicitResult:
    hypergrad: HParams
    ihvp: Params  # q ~= H^{-1} grad_w L_val
    val_loss: torch.Tensor


def implicit_hypergrad(
    problem: BilevelProblem,
    params: Params,
    hp: HParams,
    train_batches: Batches,
    val_batches: Batches,
    solver: str = "cg",
    iters: int = 20,
    alpha: float | None = None,
    damping: float = 0.0,
) -> ImplicitResult:
    w = tree_detach(params, requires_grad=True)
    hp_fixed = {k: v.detach() for k, v in hp.items()}

    val = problem.val_loss(w, val_batches)
    gv = dict(zip(w, torch.autograd.grad(val, list(w.values())), strict=True))

    def hvp_fn(u: Params) -> Params:
        return hvp(lambda prm: problem.train_loss(prm, hp_fixed, train_batches), w, u)

    if solver == "cg":
        q = inverse_hvp_cg(hvp_fn, gv, iters, damping=damping)
    elif solver == "neumann":
        if alpha is None:
            raise ValueError("Neumann solver needs alpha (< 2 / lambda_max)")
        q = inverse_hvp_neumann(hvp_fn, gv, iters, alpha)
    else:
        raise ValueError(f"unknown solver {solver!r}")

    # Mixed second derivative: grad_lambda <grad_w L_train(w, lambda), q>, with q fixed.
    hp_req = {k: v.detach().clone().requires_grad_(True) for k, v in hp.items()}
    loss = problem.train_loss(w, hp_req, train_batches)
    gw = torch.autograd.grad(loss, list(w.values()), create_graph=True)
    inner = sum((g * q[k].detach()).sum() for k, g in zip(w, gw, strict=True))
    if inner.requires_grad:
        mixed = torch.autograd.grad(inner, list(hp_req.values()), allow_unused=True)
    else:
        mixed = [None] * len(hp_req)
    hypergrad = {
        k: (-m.detach() if m is not None else torch.zeros_like(v))
        for (k, v), m in zip(hp_req.items(), mixed, strict=True)
    }
    return ImplicitResult(hypergrad, {k: t.detach() for k, t in q.items()}, val.detach())
