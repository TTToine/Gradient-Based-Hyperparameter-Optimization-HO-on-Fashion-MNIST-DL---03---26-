"""Reverse-mode hypergradients through an unrolled training trajectory.

Given batches b_1..b_T, run T steps of ``DiffSGD`` from (w_0, v_0) and return
d L_val(w_T) / d lambda, back-propagated through the last K = T - n_warmup steps only.

* full unrolling:      n_warmup = 0 (the exact hypergradient of the T-step procedure)
* truncated unrolling: n_warmup > 0, or warm-started (w_0, v_0) carried over from the
  previous outer step (Shaban et al., 2019). The steps before the window are treated as
  constants, which biases the hypergradient but caps memory at K steps.

Memory. Keeping the autograd graph of all K steps costs O(K) activations. With
``checkpoint_every = c >= 1`` the forward pass instead runs without a graph and stores, every
c steps, the weights, the velocity and the RNG state (CPU and CUDA). The backward pass then
walks the window segment by segment, last segment first: it restores the checkpoint and the
RNG state, recomputes the c steps of that segment with a graph, and back-propagates the
adjoint (dL/dw, dL/dv) through them. Peak memory is O(K/c) checkpoints of (w, v) plus the
graph of c steps, at the price of one extra forward pass. Because the RNG state is replayed,
anything random inside a step (dropout, augmentation) is identical in the recomputation, and
the result equals the one obtained with the full graph (``checkpoint_every = 0``).
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


@dataclass
class RngState:
    cpu: Tensor
    cuda: list[Tensor] | None

    @classmethod
    def capture(cls) -> RngState:
        cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
        return cls(torch.get_rng_state(), cuda)

    def restore(self) -> None:
        torch.set_rng_state(self.cpu)
        if self.cuda is not None:
            torch.cuda.set_rng_state_all(self.cuda)


def unrolled_hypergrad(
    opt: DiffSGD,
    params: Params,
    state: Params,
    hp: HParams,
    train_batches: Sequence[Batch],
    val_batches: Batches,
    n_warmup: int = 0,
    checkpoint_every: int = 1,
) -> UnrolledResult:
    """Hypergradient of L_val(w_T). ``checkpoint_every = 0`` keeps the full graph; c >= 1
    recomputes the window in segments of c steps (see the module docstring)."""
    if checkpoint_every < 0:
        raise ValueError("checkpoint_every must be >= 0")
    if checkpoint_every == 0:
        return _full_graph(opt, params, state, hp, train_batches, val_batches, n_warmup)
    return _checkpointed(
        opt, params, state, hp, train_batches, val_batches, n_warmup, checkpoint_every
    )


def _hypergrad_dict(hp_req: HParams, grads) -> HParams:
    return {
        k: (g.detach() if g is not None else torch.zeros_like(v))
        for (k, v), g in zip(hp_req.items(), grads, strict=True)
    }


def _full_graph(opt, params, state, hp, train_batches, val_batches, n_warmup) -> UnrolledResult:
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
    return UnrolledResult(
        _hypergrad_dict(hp_req, grads), tree_detach(p), tree_detach(s), val.detach()
    )


def _checkpointed(
    opt, params, state, hp, train_batches, val_batches, n_warmup, every
) -> UnrolledResult:
    hp_fixed = {k: v.detach() for k, v in hp.items()}
    p, s = tree_detach(params), tree_detach(state)
    for batch in train_batches[:n_warmup]:
        p, s, _ = opt.step(p, s, hp_fixed, batch, differentiable=False)

    # Forward pass without graph; checkpoint (w, v, RNG) at the start of every segment.
    window = list(train_batches[n_warmup:])
    checkpoints = []
    for i, batch in enumerate(window):
        if i % every == 0:
            checkpoints.append((i, tree_detach(p), tree_detach(s), RngState.capture()))
        p, s, _ = opt.step(p, s, hp_fixed, batch, differentiable=False)

    w_final = tree_detach(p, requires_grad=True)
    val = opt.problem.val_loss(w_final, val_batches)
    rng_after_val = RngState.capture()
    names = list(w_final)
    adj_w = dict(zip(names, torch.autograd.grad(val, list(w_final.values())), strict=True))
    adj_v = {k: torch.zeros_like(t) for k, t in s.items()}
    hypergrad = {k: torch.zeros_like(v) for k, v in hp_fixed.items()}

    # Backward pass, one segment at a time.
    hp_req = {k: v.clone().requires_grad_(True) for k, v in hp_fixed.items()}
    hp_keys = list(hp_req)
    for start, p0, s0, rng in reversed(checkpoints):
        rng.restore()
        w_in = tree_detach(p0, requires_grad=True)
        v_in = tree_detach(s0, requires_grad=True)
        w_out, v_out = w_in, v_in
        for batch in window[start : start + every]:
            w_out, v_out, _ = opt.step(w_out, v_out, hp_req, batch, differentiable=True)
        grads = torch.autograd.grad(
            [*w_out.values(), *v_out.values()],
            [*w_in.values(), *v_in.values(), *hp_req.values()],
            grad_outputs=[*adj_w.values(), *adj_v.values()],
            allow_unused=True,
        )
        n = len(names)
        adj_w = {k: _or_zero(g, w_in[k]) for k, g in zip(names, grads[:n], strict=True)}
        adj_v = {k: _or_zero(g, v_in[k]) for k, g in zip(names, grads[n : 2 * n], strict=True)}
        for k, g in zip(hp_keys, grads[2 * n :], strict=True):
            if g is not None:
                hypergrad[k] += g.detach()
        del w_out, v_out, grads

    # Leave the global RNG where a run without recomputation would have left it.
    rng_after_val.restore()
    return UnrolledResult(hypergrad, tree_detach(p), tree_detach(s), val.detach())


def _or_zero(g: Tensor | None, like: Tensor) -> Tensor:
    return torch.zeros_like(like) if g is None else g.detach()
