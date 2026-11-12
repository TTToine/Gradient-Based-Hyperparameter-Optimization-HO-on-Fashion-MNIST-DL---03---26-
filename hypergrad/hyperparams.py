"""Hyperparameter parametrisations, always in unconstrained space.

lr_g  = exp(log_lr[g])          one per parameter group
wd_g  = exp(log_wd[g])          one per parameter group (omitted entirely when wd = 0)
a_i   = sigmoid(sample_logit[i]) one per training example
"""

from __future__ import annotations

import math

import torch

from .problem import HParams


def _per_group(value: float | list[float], n_groups: int, name: str) -> list[float]:
    if isinstance(value, int | float):
        return [float(value)] * n_groups
    if len(value) != n_groups:
        raise ValueError(f"{name}: expected {n_groups} values, got {len(value)}")
    return [float(v) for v in value]


def build_hparams(
    spec: dict,
    n_groups: int,
    n_train: int | None = None,
    device: str | torch.device = "cpu",
) -> HParams:
    """Build the hyperparameter dict from a config section.

    ``spec`` keys: ``lr`` (float or list), ``wd`` (float or list, 0 disables the penalty),
    ``sample_weight`` (initial weight in (0, 1) for every example; needs ``n_train``).
    """
    hp: HParams = {}
    lrs = _per_group(spec["lr"], n_groups, "lr")
    hp["log_lr"] = torch.tensor([math.log(v) for v in lrs], device=device)

    wds = _per_group(spec.get("wd", 0.0), n_groups, "wd")
    if any(v > 0 for v in wds):
        if any(v <= 0 for v in wds):
            raise ValueError("wd must be > 0 in every group, or 0 in all of them")
        hp["log_wd"] = torch.tensor([math.log(v) for v in wds], device=device)

    if spec.get("sample_weight") is not None:
        if n_train is None:
            raise ValueError("sample weights need n_train")
        a = float(spec["sample_weight"])
        hp["sample_logit"] = torch.full((n_train,), math.log(a / (1 - a)), device=device)
    return hp


def readable(hp: HParams) -> dict[str, list[float]]:
    """Per-group hyperparameters in natural units, for logs and result files."""
    out: dict[str, list[float]] = {"lr": hp["log_lr"].exp().tolist()}
    out["wd"] = hp["log_wd"].exp().tolist() if "log_wd" in hp else [0.0] * len(out["lr"])
    return out
