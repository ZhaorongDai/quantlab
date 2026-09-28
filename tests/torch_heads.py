"""Tiny cross-section torch heads shared by the model tests.

Plain helpers, not fixtures, imported as ``from tests.torch_heads import
MeanContextHead``. Each is a small ``TorchModel``: the network maps
``[S_t, N, F]`` to ``[S_t, L]``, and the loss is ``masked_mse``.
"""

import torch
from torch import nn

from quantlab.base.model import TorchModel
from quantlab.torch_model.training import (
    TrainLossThreshold,
    cs_rank_norm,
    cs_zscore,
    drop_extreme,
    masked_mse,
)


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


class MeanContextHead(TorchModel):
    """``MeanContextNet`` with its choices read from ``hyperparameters``.

    ``window_bars`` (default 3); ``transform``: ``"zscore"`` (default),
    ``"rank"``, or ``(kind, drop_fraction)``; ``stopping``: ``("patience", n)``
    (default ``("patience", 2)``) or ``("threshold", threshold, max_epochs)``;
    ``clip`` (default True) keeps the default feature transform, False only
    fills NaN.
    """

    @property
    def window_bars(self) -> int:
        return self.config.hyperparameters.get("window_bars", 3)

    def _init_model(self, num_features, num_labels, hyperparameters):
        return MeanContextNet(num_features, num_labels, self.window_bars)

    def _loss(self, output, batch):
        return masked_mse(output, batch.y, batch.mask)

    def _transform_feature(self, x):
        if self.config.hyperparameters.get("clip", True):
            return super()._transform_feature(x)
        return torch.nan_to_num(x, nan=0.0)

    def _transform_target(self, y, training):
        spec = self.config.hyperparameters.get("transform", "zscore")
        kind, fraction = (spec, 0.0) if isinstance(spec, str) else spec
        keep = drop_extreme(y, fraction) if training and fraction else None
        if keep is not None:
            y = y[keep]
        return (cs_rank_norm(y) if kind == "rank" else cs_zscore(y)), keep

    def _on_fit_start(self):
        kind, *args = self.config.hyperparameters.get("stopping", ("patience", 2))
        self.threshold = TrainLossThreshold(*args) if kind == "threshold" else None
        self.patience = args[0] if kind == "patience" else None
        self.best, self.bad, self.best_state = float("inf"), 0, None

    def _should_stop(self, epoch, train_loss, val_loss):
        if self.threshold is not None:
            return self.threshold.update(train_loss)
        if val_loss is None:
            return False
        if val_loss < self.best:
            self.best, self.bad = val_loss, 0
            self.best_state = {
                k: v.detach().clone() for k, v in self.model.state_dict().items()
            }
            return False
        self.bad += 1
        return self.bad >= self.patience

    def _on_fit_end(self):
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)


class RecordingNet(MeanContextNet):
    """``MeanContextNet`` that keeps every input it was called with."""

    def __init__(self, *args):
        super().__init__(*args)
        self.inputs: list[torch.Tensor] = []

    def forward(self, x):
        self.inputs.append(x.detach().cpu().clone())
        return super().forward(x)


class RecordingHead(MeanContextHead):
    """``MeanContextHead`` whose network records its inputs."""

    def _init_model(self, num_features, num_labels, hyperparameters):
        return RecordingNet(num_features, num_labels, self.window_bars)


class OneBarHead(MeanContextHead):
    """``MeanContextHead`` on a one-bar window: no warm-up, so it runs on
    panel stand-ins that have no dataset calendar."""

    window_bars = 1
