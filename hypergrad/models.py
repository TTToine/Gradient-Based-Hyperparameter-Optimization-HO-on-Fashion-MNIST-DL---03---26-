"""Models. Parameter groups (for per-layer hyperparameters) are the top-level submodules."""

from __future__ import annotations

from torch import nn


class SmallCNN(nn.Module):
    """The exam CNN for 28x28 grey images, without dropout.

    Dropout is removed so that the inner training step is a deterministic function of
    (weights, batch, hyperparameters); otherwise the unrolled hypergradient would depend on
    the dropout masks and could not be checked against finite differences or a reference.
    Groups: conv1, conv2, conv3, fc.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()

        def block(c_in: int, c_out: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
            )

        self.conv1 = block(1, 32)
        self.conv2 = block(32, 64)
        self.conv3 = block(64, 128)
        self.fc = nn.Sequential(
            nn.Flatten(), nn.Linear(128 * 3 * 3, 256), nn.ReLU(), nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.fc(self.conv3(self.conv2(self.conv1(x))))


class MLP(nn.Module):
    """Small MLP; one group per linear layer (fc0, fc1, ...). Smooth tanh by default."""

    def __init__(
        self, sizes: list[int], activation: type[nn.Module] = nn.Tanh, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.n_layers = len(sizes) - 1
        for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:], strict=True)):
            self.add_module(f"fc{i}", nn.Linear(a, b))
        self.act = activation()
        self.drop = nn.Dropout(dropout)  # used by the RNG-replay tests of checkpointing

    def forward(self, x):
        x = x.flatten(1)
        for i in range(self.n_layers):
            x = getattr(self, f"fc{i}")(x)
            if i < self.n_layers - 1:
                x = self.drop(self.act(x))
        return x


def build_model(name: str, num_classes: int = 10) -> nn.Module:
    if name == "small_cnn":
        return SmallCNN(num_classes)
    if name == "mlp":
        return MLP([28 * 28, 256, num_classes], activation=nn.ReLU)
    raise ValueError(f"unknown model {name!r}")


def param_groups(model: nn.Module) -> dict[str, int]:
    """Map each parameter name to the index of its top-level submodule."""
    order: dict[str, int] = {}
    groups = {}
    for name, _ in model.named_parameters():
        top = name.split(".")[0]
        order.setdefault(top, len(order))
        groups[name] = order[top]
    return groups
