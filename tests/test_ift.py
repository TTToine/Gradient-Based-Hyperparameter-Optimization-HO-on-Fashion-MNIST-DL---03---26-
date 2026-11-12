"""Test 5: the implicit hypergradient w.r.t. optimizer hyperparameters is zero.

The learning rate does not appear in L_train, so it does not move the minimiser w*(lambda):
the implicit function theorem assigns it a hypergradient of exactly zero. The unrolled
hypergradient of the same quantity is not zero, because the learning rate changes *where
training is after K steps*. This is the central distinction of the project.
"""

import math

import torch
from conftest import make_mlp_problem

from hypergrad.functional import tree_detach
from hypergrad.inner import DiffSGD
from hypergrad.methods import implicit_hypergrad, unrolled_hypergrad


def setup(seed=0):
    problem, params, train, val = make_mlp_problem(seed=seed)
    opt = DiffSGD(problem, momentum=0.0)
    hp = {
        "log_lr": torch.tensor([math.log(0.5), math.log(0.5)]),
        "log_wd": torch.tensor([math.log(5e-2), math.log(5e-2)]),
    }
    return problem, params, train, val, opt, hp


def test_implicit_hypergrad_wrt_lr_is_zero():
    problem, params, train, val, opt, hp = setup()
    w, _, _ = opt.run(params, opt.init_state(params), hp, [train] * 500)
    ift = implicit_hypergrad(problem, w, hp, train, val, solver="cg", iters=50)
    assert torch.all(ift.hypergrad["log_lr"] == 0)
    assert ift.hypergrad["log_wd"].abs().min() > 0

    unrolled = unrolled_hypergrad(opt, params, opt.init_state(params), hp, [train] * 20, val)
    assert unrolled.hypergrad["log_lr"].abs().min() > 0


def test_implicit_matches_long_unrolled_on_mlp():
    """Full-batch GD to (near) stationarity: IFT and long unrolling agree on log_wd."""
    problem, params, train, val, opt, hp = setup(seed=2)
    unrolled = unrolled_hypergrad(opt, params, opt.init_state(params), hp, [train] * 2000, val)
    w = tree_detach(unrolled.params, requires_grad=True)
    grads = torch.autograd.grad(problem.train_loss(w, hp, train), list(w.values()))
    grad_norm = torch.cat([g.reshape(-1) for g in grads]).norm()
    assert grad_norm < 1e-5  # the IFT assumption (grad_w L_train = 0) approximately holds
    ift = implicit_hypergrad(problem, w, hp, train, val, solver="cg", iters=100)
    a, b = ift.hypergrad["log_wd"], unrolled.hypergrad["log_wd"]
    assert (a - b).norm() / b.norm() < 1e-3, (a, b)
