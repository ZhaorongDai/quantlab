"""Tiny cross-section torch heads shared by the model tests.

Plain helpers, not fixtures, imported as ``from tests.dl_heads import
MeanContextHead``. Each is the smallest ``DLModel`` that exercises one part
of the base class: the network maps ``[S_t, N, F]`` to ``[S_t, L]``.
"""

import torch
from torch import nn

from quantlab.base.model import DLModel
from quantlab.dl_model.training import TargetTransform, ValLossPatience, masked_mse


class MeanContextNet(nn.Module):
    """A linear map of each symbol's flattened window plus the cross-section mean
    of a second one, so every symbol's output depends on the others."""

    def __init__(self, num_features: int, num_labels: int, window_bars: int):
        super().__init__()
        self.own = nn.Linear(window_bars * num_features, num_labels)
        self.context = nn.Linear(window_bars * num_features, num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(x.shape[0], -1)
        return self.own(flat) + self.context(flat).mean(dim=0, keepdim=True)


class MeanContextHead(DLModel):
    """``MeanContextNet`` with its declarations read from ``hyperparameters``.

    ``window_bars`` (default 3), ``transform`` (a ``TargetTransform``,
    default z-score), ``stopping`` (default ``ValLossPatience(2)``) and
    ``clip`` (default True) may be overridden per test.
    """

    @property
    def window_bars(self) -> int:
        return self.config.hyperparameters.get("window_bars", 3)

    @property
    def target_transform(self) -> TargetTransform:
        return self.config.hyperparameters.get("transform", TargetTransform("zscore"))

    @property
    def stopping(self):
        return self.config.hyperparameters.get("stopping", ValLossPatience(2))

    @property
    def clip_features(self) -> bool:
        return self.config.hyperparameters.get("clip", True)

    def _init_model(self, num_features, num_labels, hyperparameters):
        return MeanContextNet(num_features, num_labels, self.window_bars)

    def _init_optim(self, model):
        return torch.optim.Adam(model.parameters(), lr=self.config.lr)

    def _train_one_batch(self, epoch, x, y):
        self.optim.zero_grad()
        loss = masked_mse(self.model(x), y)
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.model.parameters(), 3.0)
        self.optim.step()
        return loss.detach()

    def _val_one_batch(self, epoch, x, y):
        return masked_mse(self.model(x), y)

    def _test_one_batch(self, epoch, x, y):
        return masked_mse(self.model(x), y)


class RecordingNet(MeanContextNet):
    """``MeanContextNet`` that keeps every input it was called with."""

    def __init__(self, *args):
        super().__init__(*args)
        self.inputs: list[torch.Tensor] = []

    def forward(self, x):
        self.inputs.append(x.detach().clone())
        return super().forward(x)


class RecordingHead(MeanContextHead):
    """``MeanContextHead`` whose network records its inputs."""

    def _init_model(self, num_features, num_labels, hyperparameters):
        return RecordingNet(num_features, num_labels, self.window_bars)


class OneBarHead(MeanContextHead):
    """``MeanContextHead`` on a one-bar window: no warm-up, so it runs on
    panel stand-ins that have no dataset calendar."""

    window_bars = 1
