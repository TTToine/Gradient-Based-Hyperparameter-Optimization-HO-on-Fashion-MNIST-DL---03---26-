"""Hand-written differentiable SGD with heavy-ball momentum.

    v_{t+1} = mu * v_t + grad_w L_train(w_t, lambda)
    w_{t+1} = w_t - eta_{g(w)} * v_{t+1}

with per-group learning rates eta_g = exp(log_lr[g]). With ``differentiable=True`` each step
keeps the autograd graph (``create_graph=True``), so the final weights are a differentiable
function of every hyperparameter and of the initial weights. With ``differentiable=False``
the step is a plain, memory-free SGD step.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor

from .functional import Params, tree_zeros_like
from .problem import Batch, BilevelProblem, HParams


class DiffSGD:
    def __init__(self, problem: BilevelProblem, momentum: float = 0.0) -> None:
        self.problem = problem
        self.momentum = momentum

    def init_state(self, params: Params) -> Params:
        return tree_zeros_like({k: v.detach() for k, v in params.items()})

    def _momentum(self, hp: HParams) -> Tensor | float:
        if "logit_momentum" in hp:
            return torch.sigmoid(hp["logit_momentum"])
        return self.momentum

    def step(
        self,
        params: Params,
        state: Params,
        hp: HParams,
        batch: Batch,
        differentiable: bool = False,
    ) -> tuple[Params, Params, Tensor]:
        """One SGD step. Returns (new_params, new_state, train_loss_before_step)."""
        if not differentiable:
            hp = {k: v.detach() for k, v in hp.items()}
            params = {k: v.detach().requires_grad_(True) for k, v in params.items()}
            state = {k: v.detach() for k, v in state.items()}
        else:
            # autograd.grad needs every parameter to be part of a graph.
            params = {
                k: v if v.requires_grad else v.detach().requires_grad_(True)
                for k, v in params.items()
            }

        loss = self.problem.train_loss(params, hp, batch)
        grads = torch.autograd.grad(loss, list(params.values()), create_graph=differentiable)

        lr = hp["log_lr"].exp()
        mu = self._momentum(hp)
        new_params, new_state = {}, {}
        for (name, p), g in zip(params.items(), grads, strict=True):
            v = mu * state[name] + g
            new_state[name] = v
            new_params[name] = p - lr[self.problem.groups[name]] * v

        if not differentiable:
            new_params = {k: v.detach() for k, v in new_params.items()}
            new_state = {k: v.detach() for k, v in new_state.items()}
        return new_params, new_state, loss.detach()

    def run(
        self,
        params: Params,
        state: Params,
        hp: HParams,
        batches: Iterable[Batch],
        differentiable: bool = False,
    ) -> tuple[Params, Params, list[float]]:
        losses = []
        for batch in batches:
            params, state, loss = self.step(params, state, hp, batch, differentiable)
            losses.append(loss.item())
        return params, state, losses
