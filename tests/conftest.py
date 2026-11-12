import pytest
import torch
import torch.nn.functional as F

from hypergrad.functional import get_params, make_forward
from hypergrad.models import MLP, param_groups
from hypergrad.problem import BilevelProblem


@pytest.fixture(autouse=True)
def float64():
    """Hypergradient checks compare against closed forms and finite differences: use float64."""
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def make_mlp_problem(seed: int = 0, sizes=(3, 6, 2), n_train: int = 32, n_val: int = 16):
    """Tiny classification problem on random data, with a smooth (tanh) MLP."""
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    model = MLP(list(sizes))
    forward = make_forward(model)

    def per_sample_loss(params, batch):
        return F.cross_entropy(forward(params, batch["x"]), batch["y"], reduction="none")

    problem = BilevelProblem(per_sample_loss, param_groups(model))
    x = torch.randn(n_train + n_val, sizes[0], generator=g)
    y = torch.randint(0, sizes[-1], (n_train + n_val,), generator=g)
    train = {"x": x[:n_train], "y": y[:n_train], "idx": torch.arange(n_train)}
    val = {"x": x[n_train:], "y": y[n_train:]}
    return problem, get_params(model), train, val


def minibatches(data: dict, batch_size: int, n_steps: int) -> list[dict]:
    """Deterministic cyclic minibatches."""
    n = data["x"].shape[0]
    out = []
    for t in range(n_steps):
        idx = torch.arange(t * batch_size, (t + 1) * batch_size) % n
        out.append({k: v[idx] for k, v in data.items()})
    return out
