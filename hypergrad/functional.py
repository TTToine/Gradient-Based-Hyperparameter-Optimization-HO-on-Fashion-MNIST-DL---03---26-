"""Functional view of a model: parameters are a ``dict[str, Tensor]`` evaluated with torch.func.

All hypergradient code works on these dictionaries ("trees") instead of on ``nn.Module``
state, so that a training trajectory w_0 -> w_1 -> ... -> w_K is an ordinary chain of
tensor operations that autograd can differentiate through.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn
from torch.func import functional_call

Params = dict[str, Tensor]


def get_params(model: nn.Module, requires_grad: bool = False) -> Params:
    """Detached copy of the model parameters as a dict."""
    return {
        name: p.detach().clone().requires_grad_(requires_grad)
        for name, p in model.named_parameters()
    }


def make_forward(model: nn.Module) -> Callable[[Params, Tensor], Tensor]:
    """Return ``forward(params, x)`` that evaluates ``model`` with the given parameters."""
    buffers = dict(model.named_buffers())

    def forward(params: Params, x: Tensor) -> Tensor:
        return functional_call(model, (params, buffers), (x,))

    return forward


# --- tree helpers -------------------------------------------------------------------------


def tree_zeros_like(a: Params) -> Params:
    return {k: torch.zeros_like(v) for k, v in a.items()}


def tree_detach(a: Params, requires_grad: bool = False) -> Params:
    return {k: v.detach().clone().requires_grad_(requires_grad) for k, v in a.items()}


def tree_add(a: Params, b: Params, alpha: float | Tensor = 1.0) -> Params:
    """a + alpha * b"""
    return {k: a[k] + alpha * b[k] for k in a}


def tree_scale(a: Params, s: float | Tensor) -> Params:
    return {k: s * v for k, v in a.items()}


def tree_dot(a: Params, b: Params) -> Tensor:
    return sum((a[k] * b[k]).sum() for k in a)


def tree_norm(a: Params) -> Tensor:
    return torch.sqrt(tree_dot(a, a))


def tree_flatten(a: Params) -> Tensor:
    return torch.cat([v.reshape(-1) for v in a.values()])


def tree_unflatten(vec: Tensor, like: Params) -> Params:
    out, i = {}, 0
    for k, v in like.items():
        n = v.numel()
        out[k] = vec[i : i + n].view_as(v)
        i += n
    return out
