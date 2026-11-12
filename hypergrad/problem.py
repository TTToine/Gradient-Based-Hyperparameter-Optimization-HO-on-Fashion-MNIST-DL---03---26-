"""Bilevel problem: inner objective L_train(w, lambda) and outer objective L_val(w).

Hyperparameters are a ``dict[str, Tensor]`` in unconstrained space. Two kinds exist, and
the distinction is the point of the project:

* objective hyperparameters enter L_train and therefore move its minimiser w*(lambda):
    - ``log_wd``       one per parameter group, penalty 0.5 * exp(log_wd[g]) * ||w_g||^2
    - ``sample_logit`` one per training example, weight sigmoid(sample_logit[i])
* optimizer hyperparameters enter only the update rule (see ``inner.DiffSGD``):
    - ``log_lr``         one per parameter group
    - ``logit_momentum`` scalar (optional; otherwise momentum is a fixed float)

``BilevelProblem`` only reads objective hyperparameters; any other key is ignored here.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import torch
from torch import Tensor

from .functional import Params

Batch = Mapping[str, Tensor]  # keys "x", "y" and, for sample weights, "idx"
HParams = dict[str, Tensor]
Batches = Batch | Sequence[Batch]

OBJECTIVE_KEYS = ("log_wd", "sample_logit")


def as_batch_list(batches: Batches) -> list[Batch]:
    return [batches] if isinstance(batches, Mapping) else list(batches)


class BilevelProblem:
    def __init__(
        self,
        per_sample_loss: Callable[[Params, Batch], Tensor],
        groups: Mapping[str, int],
        reduction: str = "mean",
    ) -> None:
        if reduction not in ("mean", "sum"):
            raise ValueError(f"unknown reduction {reduction!r}")
        self.per_sample_loss = per_sample_loss
        self.groups = dict(groups)
        self.n_groups = max(self.groups.values()) + 1
        self.reduction = reduction

    def _reduce(self, losses: Tensor) -> Tensor:
        return losses.mean() if self.reduction == "mean" else losses.sum()

    def data_loss(self, params: Params, batch: Batch, hp: HParams | None = None) -> Tensor:
        losses = self.per_sample_loss(params, batch)
        if hp is not None and "sample_logit" in hp:
            losses = losses * torch.sigmoid(hp["sample_logit"][batch["idx"]])
        return self._reduce(losses)

    def penalty(self, params: Params, hp: HParams) -> Tensor:
        if "log_wd" not in hp:
            return next(iter(params.values())).new_zeros(())
        wd = hp["log_wd"].exp()
        return 0.5 * sum(wd[self.groups[n]] * p.pow(2).sum() for n, p in params.items())

    def train_loss(self, params: Params, hp: HParams, batches: Batches) -> Tensor:
        """L_train(w, lambda), averaged over the given batches."""
        blist = as_batch_list(batches)
        data = sum(self.data_loss(params, b, hp) for b in blist) / len(blist)
        return data + self.penalty(params, hp)

    def val_loss(self, params: Params, batches: Batches) -> Tensor:
        """L_val(w), averaged over the given batches. Does not depend on hyperparameters."""
        blist = as_batch_list(batches)
        return sum(self._reduce(self.per_sample_loss(params, b)) for b in blist) / len(blist)


def classification_problem(
    forward: Callable[[Params, Tensor], Tensor], groups: Mapping[str, int]
) -> BilevelProblem:
    """Cross-entropy training/validation objectives for a functional classifier."""
    import torch.nn.functional as F

    def per_sample_loss(params: Params, batch: Batch) -> Tensor:
        return F.cross_entropy(forward(params, batch["x"]), batch["y"], reduction="none")

    return BilevelProblem(per_sample_loss, groups, reduction="mean")
