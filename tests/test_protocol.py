"""Phase 2: data splits, label noise, batch order, metrics, configs (no dataset download)."""

import itertools
import math
from pathlib import Path

import pytest
import torch

from hypergrad.data import BatchStream, Split, corrupt_labels
from hypergrad.experiment import deep_merge, load_config, result_path
from hypergrad.hyperparams import build_hparams, readable
from hypergrad.metrics import auroc, cosine_similarity, precision_recall_f1

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("kind", ["symmetric", "shift"])
def test_corrupt_labels(kind):
    y = torch.arange(1000) % 10
    noisy, idx = corrupt_labels(y, 0.2, kind, seed=3)
    changed = (noisy != y).nonzero().squeeze(1)
    assert idx.numel() == 200
    assert torch.equal(changed, idx)  # every selected label really changes, no other does
    if kind == "shift":
        assert torch.equal(noisy[idx], (y[idx] + 1) % 10)
    again, idx2 = corrupt_labels(y, 0.2, kind, seed=3)
    assert torch.equal(again, noisy) and torch.equal(idx2, idx)


def test_symmetric_noise_covers_all_other_classes():
    y = torch.zeros(5000, dtype=torch.long)
    noisy, idx = corrupt_labels(y, 0.5, "symmetric", seed=0)
    assert set(noisy[idx].tolist()) == set(range(1, 10))


def make_split(n=50):
    return Split(torch.randn(n, 1, 28, 28), torch.arange(n) % 10, torch.arange(n))


def test_batch_stream_is_deterministic_and_covers_each_epoch():
    split = make_split(50)
    a = BatchStream(split, 10, seed=7)
    b = BatchStream(split, 10, seed=7)
    for _ in range(3):
        ea, eb = a.epoch(), b.epoch()
        assert all(torch.equal(x["idx"], y["idx"]) for x, y in zip(ea, eb, strict=True))
        seen = torch.cat([batch["idx"] for batch in ea]).sort().values
        assert torch.equal(seen, torch.arange(50))
    c = BatchStream(split, 10, seed=8)
    assert not torch.equal(c.epoch()[0]["idx"], BatchStream(split, 10, seed=7).epoch()[0]["idx"])


def test_batch_stream_take_continues_across_epochs():
    split = make_split(30)
    s = BatchStream(split, 10, seed=0)
    taken = s.take(7)
    ref = BatchStream(split, 10, seed=0)
    expected = list(itertools.chain(ref.epoch(), ref.epoch(), ref.epoch()))[:7]
    assert all(torch.equal(x["idx"], y["idx"]) for x, y in zip(taken, expected, strict=True))


def brute_force_auroc(scores, labels):
    pos, neg = scores[labels], scores[~labels]
    wins = (pos[:, None] > neg[None, :]).double() + 0.5 * (pos[:, None] == neg[None, :]).double()
    return wins.mean().item()


def test_auroc_matches_pair_counting_with_ties():
    g = torch.Generator().manual_seed(0)
    scores = torch.randint(0, 6, (200,), generator=g).double()  # many ties
    labels = torch.rand(200, generator=g) < 0.3
    assert math.isclose(auroc(scores, labels), brute_force_auroc(scores, labels), rel_tol=1e-12)
    assert auroc(labels.double(), labels) == 1.0
    assert auroc(-labels.double(), labels) == 0.0


def test_precision_recall_f1():
    pred = torch.tensor([1, 1, 0, 0, 1], dtype=torch.bool)
    true = torch.tensor([1, 0, 1, 0, 1], dtype=torch.bool)
    m = precision_recall_f1(pred, true)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)


def test_cosine_similarity():
    a = {"x": torch.tensor([1.0, 0.0]), "y": torch.tensor([0.0])}
    assert cosine_similarity(a, a) == pytest.approx(1.0)
    b = {"x": torch.tensor([0.0, 2.0]), "y": torch.tensor([0.0])}
    assert cosine_similarity(a, b) == pytest.approx(0.0)


def test_build_hparams_roundtrip():
    hp = build_hparams({"lr": [0.1, 0.01], "wd": 1e-4, "sample_weight": 0.9}, 2, n_train=5)
    r = readable(hp)
    assert r["lr"] == pytest.approx([0.1, 0.01])
    assert r["wd"] == pytest.approx([1e-4, 1e-4])
    assert torch.sigmoid(hp["sample_logit"]).allclose(torch.full((5,), 0.9))
    assert "log_wd" not in build_hparams({"lr": 0.1, "wd": 0.0}, 2)
    with pytest.raises(ValueError):
        build_hparams({"lr": [0.1, 0.1, 0.1]}, 2)


def test_quick_config_overrides_and_separate_results_dir():
    path = REPO / "experiments/configs/baseline_default.yaml"
    full, quick = load_config(path), load_config(path, quick=True)
    assert full["data"]["n_train"] == 50000 and quick["data"]["n_train"] == 2000
    assert quick["data"]["split_seed"] == full["data"]["split_seed"]  # untouched keys survive
    assert result_path(full, 0).parent.name == "results"
    assert result_path(quick, 0).parent.name == "quick"
    assert deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"b": 3}}) == {"a": {"b": 3, "c": 2}}
