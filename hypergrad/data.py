"""Fashion-MNIST held in memory, with deterministic splits, label noise and batch order.

Splits (fixed by ``split_seed``, independent of the run seed):
  * the official 60k training set is permuted once; its last 10k images form the
    validation pool and the first 50k the training pool;
  * ``n_train`` / ``n_val`` take prefixes of these pools, so quick-mode subsets are nested
    in the full splits;
  * the official 10k test set is the test split (optionally a prefix of it).

Label noise is applied to the training split only, with its own seed; the indices of the
corrupted examples are returned so that they can be saved with the results.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import torch
from torch import Tensor

MEAN, STD = 0.2860, 0.3530
NUM_CLASSES = 10
VAL_POOL = 10_000


@dataclass
class Split:
    x: Tensor  # [N, 1, 28, 28], normalised
    y: Tensor  # labels used for training/evaluation (noisy for a corrupted train split)
    idx: Tensor  # arange(N): position of each example inside this split

    def __len__(self) -> int:
        return self.y.shape[0]

    def batch(self, rows: Tensor) -> dict[str, Tensor]:
        return {"x": self.x[rows], "y": self.y[rows], "idx": self.idx[rows]}


@dataclass
class Data:
    train: Split
    val: Split
    test: Split
    corrupted: Tensor  # sorted indices (into the train split) of corrupted labels
    y_train_clean: Tensor


def corrupt_labels(
    y: Tensor, rate: float, kind: str, seed: int, num_classes: int = NUM_CLASSES
) -> tuple[Tensor, Tensor]:
    """Corrupt exactly round(rate * N) labels.

    ``symmetric``: the new label is uniform over the other num_classes - 1 classes.
    ``shift``:     y -> (y + 1) mod num_classes (structured noise of the original exam code).
    """
    g = torch.Generator().manual_seed(seed)
    n = y.shape[0]
    k = round(rate * n)
    idx = torch.randperm(n, generator=g)[:k].sort().values
    y_noisy = y.clone()
    if kind == "symmetric":
        offset = torch.randint(1, num_classes, (k,), generator=g)
    elif kind == "shift":
        offset = torch.ones(k, dtype=torch.long)
    else:
        raise ValueError(f"unknown noise kind {kind!r}")
    y_noisy[idx] = (y[idx] + offset) % num_classes
    return y_noisy, idx


def _normalise(images: Tensor) -> Tensor:
    return ((images.float() / 255.0 - MEAN) / STD).unsqueeze(1)


def load_fashion_mnist(
    root: str = "data",
    n_train: int = 50_000,
    n_val: int = 10_000,
    n_test: int | None = None,
    split_seed: int = 0,
    noise_rate: float = 0.0,
    noise_kind: str = "symmetric",
    noise_seed: int = 1000,
    device: str | torch.device = "cpu",
) -> Data:
    from torchvision.datasets import FashionMNIST

    official_train = FashionMNIST(root, train=True, download=True)
    official_test = FashionMNIST(root, train=False, download=True)

    n_pool = official_train.data.shape[0] - VAL_POOL
    if n_train > n_pool or n_val > VAL_POOL:
        raise ValueError(f"n_train <= {n_pool} and n_val <= {VAL_POOL} required")
    perm = torch.randperm(
        official_train.data.shape[0], generator=torch.Generator().manual_seed(split_seed)
    )
    train_rows = perm[:n_pool][:n_train]
    val_rows = perm[n_pool:][:n_val]

    x_all = _normalise(official_train.data)
    y_all = official_train.targets.clone()
    y_train_clean = y_all[train_rows]
    if noise_rate > 0:
        y_train, corrupted = corrupt_labels(y_train_clean, noise_rate, noise_kind, noise_seed)
    else:
        y_train, corrupted = y_train_clean.clone(), torch.empty(0, dtype=torch.long)

    x_test = _normalise(official_test.data)
    y_test = official_test.targets.clone()
    if n_test is not None:
        x_test, y_test = x_test[:n_test], y_test[:n_test]

    def split(x: Tensor, y: Tensor) -> Split:
        return Split(x.to(device), y.to(device), torch.arange(y.shape[0], device=device))

    return Data(
        train=split(x_all[train_rows], y_train),
        val=split(x_all[val_rows], y_all[val_rows]),
        test=split(x_test, y_test),
        corrupted=corrupted,
        y_train_clean=y_train_clean,
    )


class BatchStream:
    """Infinite, seeded stream of shuffled minibatches (each epoch reshuffles, drop_last)."""

    def __init__(self, split: Split, batch_size: int, seed: int) -> None:
        if batch_size > len(split):
            raise ValueError("batch_size larger than the split")
        self.split = split
        self.batch_size = batch_size
        self.generator = torch.Generator().manual_seed(seed)
        self.steps_per_epoch = len(split) // batch_size
        self._queue: list[Tensor] = []

    def _refill(self) -> None:
        perm = torch.randperm(len(self.split), generator=self.generator)
        perm = perm.to(self.split.y.device)
        b = self.batch_size
        self._queue = [perm[i * b : (i + 1) * b] for i in range(self.steps_per_epoch)]

    def __iter__(self) -> Iterator[dict[str, Tensor]]:
        while True:
            yield self.next()

    def next(self) -> dict[str, Tensor]:
        if not self._queue:
            self._refill()
        return self.split.batch(self._queue.pop(0))

    def take(self, n: int) -> list[dict[str, Tensor]]:
        return [self.next() for _ in range(n)]

    def epoch(self) -> list[dict[str, Tensor]]:
        """The remaining batches of the current epoch (a full epoch if none are pending)."""
        if not self._queue:
            self._refill()
        out = [self.split.batch(rows) for rows in self._queue]
        self._queue = []
        return out


def eval_batches(split: Split, batch_size: int = 1000) -> Iterator[dict[str, Tensor]]:
    for start in range(0, len(split), batch_size):
        rows = torch.arange(start, min(start + batch_size, len(split)), device=split.y.device)
        yield split.batch(rows)
