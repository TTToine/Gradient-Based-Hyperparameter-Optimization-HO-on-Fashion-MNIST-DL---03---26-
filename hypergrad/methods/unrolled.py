"""Reverse-mode hypergradients through an unrolled training trajectory.

Given batches b_1..b_T, run T steps of ``DiffSGD`` from (w_0, v_0) and return
d L_val(w_T) / d lambda, back-propagated through the last K = T - n_warmup steps only.

* full unrolling:      n_warmup = 0 (the exact hypergradient of the T-step procedure)
* truncated unrolling: n_warmup > 0, or warm-started (w_0, v_0) carried over from the
  previous outer step (Shaban et al., 2019). The steps before the window are treated as
  constants, which biases the hypergradient but caps memory at K steps.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from ..functional import Params, tree_detach
from ..inner import DiffSGD
from ..problem import Batch, Batches, HParams


@dataclass
class UnrolledResult:
    hypergrad: HParams
    params: Params  # w_T, detached
    state: Params  # v_T, detached
    val_loss: Tensor


def unrolled_hypergrad(
    opt: DiffSGD,
    params: Params,
    state: Params,
    hp: HParams,
    train_batches: Sequence[Batch],
    val_batches: Batches,
    n_warmup: int = 0,
) -> UnrolledResult:
    hp_req = {k: v.detach().clone().requires_grad_(True) for k, v in hp.items()}
    p = tree_detach(params, requires_grad=True)
    s = tree_detach(state)
    for i, batch in enumerate(train_batches):
        differentiable = i >= n_warmup
        p, s, _ = opt.step(p, s, hp_req if differentiable else hp, batch, differentiable)

    val = opt.problem.val_loss(p, val_batches)
    if val.requires_grad:
        grads = torch.autograd.grad(val, list(hp_req.values()), allow_unused=True)
    else:
        grads = [None] * len(hp_req)
    hypergrad = {
        k: (g.detach() if g is not None else torch.zeros_like(v))
        for (k, v), g in zip(hp_req.items(), grads, strict=True)
    }
    return UnrolledResult(hypergrad, tree_detach(p), tree_detach(s), val.detach())
