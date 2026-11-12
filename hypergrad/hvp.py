"""Hessian-vector products and power iteration for the largest Hessian eigenvalue."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import torch
from torch import Tensor
from torch.func import grad, jvp

from .functional import Params, tree_dot, tree_norm, tree_scale


def hvp(f: Callable[[Params], Tensor], params: Params, v: Params) -> Params:
    """H v with H = d^2 f / dw^2 at ``params`` (forward-over-reverse, no explicit Hessian)."""
    params = {k: t.detach() for k, t in params.items()}
    v = {k: t.detach() for k, t in v.items()}
    _, out = jvp(grad(f), (params,), (v,))
    return out


def power_iteration(
    hvp_fn: Callable[[Params], Params],
    like: Params,
    iters: int = 100,
    tol: float = 1e-8,
    block: Iterable[str] | None = None,
    seed: int = 0,
) -> tuple[float, Params]:
    """Dominant eigenvalue (largest in magnitude) of the Hessian, or of a diagonal block of it.

    ``block`` restricts the iteration to the sub-block H_bb for the named parameters: the
    vector is zero outside the block and H v is projected back onto it. Returns the Rayleigh
    quotient and the eigenvector.
    """
    names = set(like) if block is None else set(block)
    g = torch.Generator(device="cpu").manual_seed(seed)

    def project(u: Params) -> Params:
        return {k: (t if k in names else torch.zeros_like(t)) for k, t in u.items()}

    v = project(
        {k: torch.randn(t.shape, generator=g, dtype=t.dtype).to(t.device) for k, t in like.items()}
    )
    v = tree_scale(v, 1.0 / tree_norm(v))
    eig = 0.0
    for _ in range(iters):
        hv = project(hvp_fn(v))
        new_eig = tree_dot(v, hv).item()
        norm = tree_norm(hv)
        if norm == 0:
            return 0.0, v
        v = tree_scale(hv, 1.0 / norm)
        if abs(new_eig - eig) <= tol * max(1.0, abs(new_eig)):
            eig = new_eig
            break
        eig = new_eig
    return eig, v
