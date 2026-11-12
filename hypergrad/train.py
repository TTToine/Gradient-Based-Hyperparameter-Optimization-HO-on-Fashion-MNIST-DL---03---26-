"""Final training with fixed hyperparameters: the single protocol shared by every method.

* the same update rule as the inner loop of the hypergradient methods (``DiffSGD``:
  heavy-ball momentum, per-group constant learning rates, weight decay as an L2 term in the
  objective, optional per-example weights), with no schedule and no early stopping;
* a fixed number of epochs; after every epoch the model is evaluated on the validation
  split and the epoch with the highest validation accuracy is selected;
* the test split is evaluated once, on the selected weights.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import torch

from .data import BatchStream, Data
from .functional import get_params, make_forward
from .inner import DiffSGD
from .metrics import evaluate
from .models import build_model, param_groups
from .problem import HParams, classification_problem

log = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    epochs: int = 15
    batch_size: int = 128
    momentum: float = 0.9


def train_final(
    model_name: str,
    hp: HParams,
    data: Data,
    cfg: TrainConfig,
    seed: int,
    device: str | torch.device,
) -> dict:
    torch.manual_seed(seed)
    model = build_model(model_name).to(device)
    forward = make_forward(model)
    problem = classification_problem(forward, param_groups(model))
    opt = DiffSGD(problem, momentum=cfg.momentum)
    hp = {k: v.detach().to(device) for k, v in hp.items()}

    params = get_params(model)
    state = opt.init_state(params)
    stream = BatchStream(data.train, cfg.batch_size, seed=seed)

    history = []
    best = {"val_acc": -1.0}
    best_params = params
    diverged = False
    for epoch in range(1, cfg.epochs + 1):
        params, state, losses = opt.run(params, state, hp, stream.epoch())
        train_loss = sum(losses) / len(losses)
        if not math.isfinite(train_loss):
            diverged = True
            log.warning("training diverged at epoch %d", epoch)
            break
        val = evaluate(forward, params, data.val)
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "val_acc": val["acc"], "val_nll": val["nll"]}
        )
        log.info(
            "epoch %2d/%d  train_loss %.4f  val_acc %.4f  val_nll %.4f",
            epoch,
            cfg.epochs,
            train_loss,
            val["acc"],
            val["nll"],
        )
        if val["acc"] > best["val_acc"]:
            best = {"epoch": epoch, "val_acc": val["acc"], "val_nll": val["nll"]}
            best_params = {k: v.clone() for k, v in params.items()}

    test = evaluate(forward, best_params, data.test)
    return {
        "best_epoch": best.get("epoch"),
        "val_acc": best["val_acc"] if best.get("epoch") else None,
        "val_nll": best.get("val_nll"),
        "test_acc": test["acc"],
        "test_nll": test["nll"],
        "diverged": diverged,
        "steps": len(history) * stream.steps_per_epoch,
        "history": history,
    }
