"""Checkpointed unrolling must reproduce the full-graph hypergradient exactly.

Same seed, same batches; the models use dropout so that the test also fails if the RNG
state is not replayed during recomputation.
"""

import math

import pytest
import torch
from conftest import make_mlp_problem, minibatches

from hypergrad.inner import DiffSGD
from hypergrad.methods import unrolled_hypergrad


@pytest.fixture
def deterministic():
    old = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(old)


def rel_err(a: dict, b: dict) -> float:
    va = torch.cat([a[k].reshape(-1) for k in sorted(a)])
    vb = torch.cat([b[k].reshape(-1) for k in sorted(a)])
    return ((va - vb).norm() / vb.norm()).item()


def run(opt, params, hp, batches, val, every, n_warmup=0, seed=123):
    torch.manual_seed(seed)  # same RNG stream for dropout in both runs
    res = unrolled_hypergrad(
        opt,
        params,
        opt.init_state(params),
        hp,
        batches,
        val,
        n_warmup=n_warmup,
        checkpoint_every=every,
    )
    return res, torch.rand(1)  # also check where the global RNG is left


@pytest.mark.parametrize("k", [10, 23, 50])
@pytest.mark.parametrize("every", [1, 3, 4, 7, 14])
def test_checkpointed_equals_full_graph_with_dropout(k, every):
    """Includes K not divisible by c (e.g. K=50, c=14 and K=23, c=4): the last segment is
    shorter than c."""
    problem, params, train, val = make_mlp_problem(seed=0, sizes=(5, 16, 16, 3), dropout=0.3)
    opt = DiffSGD(problem, momentum=0.9)
    batches = minibatches(train, batch_size=8, n_steps=k)
    hp = {
        "log_lr": torch.tensor([math.log(0.1), math.log(0.05), math.log(0.08)]),
        "log_wd": torch.full((3,), math.log(1e-3)),
    }
    full, rng_full = run(opt, params, hp, batches, val, every=0)
    ckpt, rng_ckpt = run(opt, params, hp, batches, val, every=every)
    assert rel_err(ckpt.hypergrad, full.hypergrad) < 1e-5
    assert torch.equal(ckpt.val_loss, full.val_loss)
    assert all(torch.equal(ckpt.params[n], full.params[n]) for n in full.params)
    assert torch.equal(rng_ckpt, rng_full)


def test_rng_replay_is_needed():
    """Sanity check of the test itself: with dropout, different RNG streams give different
    hypergradients, so the equality above is not trivially satisfied."""
    problem, params, train, val = make_mlp_problem(seed=0, sizes=(5, 16, 16, 3), dropout=0.3)
    opt = DiffSGD(problem, momentum=0.9)
    batches = minibatches(train, batch_size=8, n_steps=10)
    hp = {"log_lr": torch.full((3,), math.log(0.1)), "log_wd": torch.full((3,), -6.0)}
    a, _ = run(opt, params, hp, batches, val, every=0, seed=1)
    b, _ = run(opt, params, hp, batches, val, every=0, seed=2)
    assert rel_err(a.hypergrad, b.hypergrad) > 1e-3


def test_checkpointed_truncated_window():
    problem, params, train, val = make_mlp_problem(seed=1, sizes=(5, 16, 3), dropout=0.2)
    opt = DiffSGD(problem, momentum=0.5)
    batches = minibatches(train, batch_size=8, n_steps=20)
    hp = {"log_lr": torch.full((2,), math.log(0.1)), "log_wd": torch.full((2,), -5.0)}
    full, _ = run(opt, params, hp, batches, val, every=0, n_warmup=8)
    ckpt, _ = run(opt, params, hp, batches, val, every=5, n_warmup=8)
    assert rel_err(ckpt.hypergrad, full.hypergrad) < 1e-5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("k", [10, 23, 50])
def test_checkpointed_equals_full_graph_small_cnn_cuda(k, deterministic):
    """The real setting: SmallCNN in float32 on GPU, batch 128, momentum 0.9.

    Needs deterministic kernels (as in every experiment run): without them two identical
    full-graph runs already differ by ~1e-3 at K = 50 (atomics in cuDNN backward)."""
    import torch.nn.functional as F

    from hypergrad.functional import get_params, make_forward
    from hypergrad.models import SmallCNN, param_groups
    from hypergrad.problem import BilevelProblem

    torch.set_default_dtype(torch.float32)
    dev = torch.device("cuda")
    torch.manual_seed(0)
    model = SmallCNN().to(dev)
    forward = make_forward(model)

    def loss(params, batch):
        return F.cross_entropy(forward(params, batch["x"]), batch["y"], reduction="none")

    problem = BilevelProblem(loss, param_groups(model))
    opt = DiffSGD(problem, momentum=0.9)
    g = torch.Generator().manual_seed(0)
    x = torch.randn(128 * k + 256, 1, 28, 28, generator=g).to(dev)
    y = torch.randint(0, 10, (128 * k + 256,), generator=g).to(dev)
    batches = [{"x": x[i * 128 : (i + 1) * 128], "y": y[i * 128 : (i + 1) * 128]} for i in range(k)]
    val = {"x": x[-256:], "y": y[-256:]}
    hp = {
        "log_lr": torch.full((4,), math.log(0.01), device=dev),
        "log_wd": torch.full((4,), math.log(1e-4), device=dev),
    }
    params = get_params(model)
    full, _ = run(opt, params, hp, batches, val, every=0)
    for every in (1, 4, 14):  # K=23, c=4 and K=50, c=14: last segment shorter
        ckpt, _ = run(opt, params, hp, batches, val, every=every)
        assert rel_err(ckpt.hypergrad, full.hypergrad) < 1e-5, every
