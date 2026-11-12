"""Test 2: unrolled hypergradient vs. central finite differences on a tiny MLP.

Same initial weights, same minibatches; only the hyperparameter is perturbed.
"""

import math

import torch
from conftest import make_mlp_problem, minibatches

from hypergrad.inner import DiffSGD
from hypergrad.methods import unrolled_hypergrad


def final_val_loss(opt, params, hp, batches, val):
    p, _, _ = opt.run(params, opt.init_state(params), hp, batches, differentiable=False)
    return opt.problem.val_loss(p, val).item()


def finite_difference(opt, params, hp, batches, val, eps=1e-6):
    fd = {}
    for key, value in hp.items():
        g = torch.zeros_like(value)
        for i in range(value.numel()):
            plus = {k: v.clone() for k, v in hp.items()}
            minus = {k: v.clone() for k, v in hp.items()}
            plus[key].view(-1)[i] += eps
            minus[key].view(-1)[i] -= eps
            g.view(-1)[i] = (
                final_val_loss(opt, params, plus, batches, val)
                - final_val_loss(opt, params, minus, batches, val)
            ) / (2 * eps)
        fd[key] = g
    return fd


def test_unrolled_matches_finite_differences():
    problem, params, train, val = make_mlp_problem(seed=0)
    opt = DiffSGD(problem, momentum=0.9)
    batches = minibatches(train, batch_size=8, n_steps=12)
    hp = {
        "log_lr": torch.tensor([math.log(0.1), math.log(0.05)]),
        "log_wd": torch.tensor([math.log(1e-2), math.log(3e-2)]),
    }
    res = unrolled_hypergrad(opt, params, opt.init_state(params), hp, batches, val)
    fd = finite_difference(opt, params, hp, batches, val)
    for key in hp:
        err = (res.hypergrad[key] - fd[key]).norm() / fd[key].norm()
        assert err < 1e-3, (key, res.hypergrad[key], fd[key])


def test_truncated_ignores_steps_before_window():
    """With n_warmup = T the window is empty and the hypergradient is exactly zero; with
    n_warmup = 0 it equals the full unrolled hypergradient."""
    problem, params, train, val = make_mlp_problem(seed=1)
    opt = DiffSGD(problem, momentum=0.5)
    batches = minibatches(train, batch_size=8, n_steps=6)
    hp = {"log_lr": torch.full((2,), math.log(0.1)), "log_wd": torch.full((2,), -4.0)}
    state = opt.init_state(params)
    full = unrolled_hypergrad(opt, params, state, hp, batches, val, n_warmup=0)
    trunc = unrolled_hypergrad(opt, params, state, hp, batches, val, n_warmup=3)
    empty = unrolled_hypergrad(opt, params, state, hp, batches, val, n_warmup=6)
    for key in hp:
        assert torch.all(empty.hypergrad[key] == 0)
        assert not torch.allclose(trunc.hypergrad[key], full.hypergrad[key])
    # All three run the same trajectory.
    assert torch.allclose(full.val_loss, trunc.val_loss)
    assert torch.allclose(full.val_loss, empty.val_loss)
