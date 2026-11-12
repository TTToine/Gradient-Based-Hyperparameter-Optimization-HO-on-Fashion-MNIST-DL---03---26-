"""Evaluation metrics: accuracy/NLL, cleaning AUROC/F1, hypergradient cosine similarity."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor

from .data import Split, eval_batches
from .functional import Params


@torch.no_grad()
def evaluate(
    forward: Callable[[Params, Tensor], Tensor], params: Params, split: Split
) -> dict[str, float]:
    """Accuracy and mean negative log-likelihood on a split."""
    correct, nll, n = 0, 0.0, 0
    for batch in eval_batches(split):
        logits = forward(params, batch["x"])
        nll += F.cross_entropy(logits, batch["y"], reduction="sum").item()
        correct += (logits.argmax(1) == batch["y"]).sum().item()
        n += batch["y"].shape[0]
    return {"acc": correct / n, "nll": nll / n}


def auroc(scores: Tensor, labels: Tensor) -> float:
    """Area under the ROC curve: P(score of a random positive > score of a random negative),
    ties counted as 1/2 (Mann-Whitney U statistic)."""
    scores = scores.detach().double().cpu()
    labels = labels.detach().bool().cpu()
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUROC needs both classes")
    # average ranks (1-based), ties share the mean rank
    sorted_scores, order = scores.sort()
    _, inverse, counts = torch.unique_consecutive(
        sorted_scores, return_inverse=True, return_counts=True
    )
    ends = counts.cumsum(0).double()
    mean_rank = ends - (counts.double() - 1) / 2
    ranks = torch.empty_like(scores)
    ranks[order] = mean_rank[inverse]
    u = ranks[labels].sum().item() - n_pos * (n_pos + 1) / 2
    return u / (n_pos * n_neg)


def precision_recall_f1(pred: Tensor, labels: Tensor) -> dict[str, float]:
    """Binary precision, recall and F1 (positive class = True)."""
    pred, labels = pred.bool().cpu(), labels.bool().cpu()
    tp = (pred & labels).sum().item()
    fp = (pred & ~labels).sum().item()
    fn = (~pred & labels).sum().item()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def cosine_similarity(a: dict[str, Tensor], b: dict[str, Tensor]) -> float:
    """Cosine similarity between two hypergradients (dicts with the same keys)."""
    va = torch.cat([a[k].reshape(-1).double() for k in sorted(a)])
    vb = torch.cat([b[k].reshape(-1).double() for k in sorted(a)])
    return (va @ vb / (va.norm() * vb.norm())).item()
