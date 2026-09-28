"""Training pieces of a deep head, and the per-bar windows it is fed.

A ``DLModel`` trains on one *cross-section* per step: the symbols with at
least one finite feature at a bar, each carrying its last N bars of features
(ADR 0006). This module holds the pieces of that loop a head chooses or the
base class needs:

- ``CrossSectionWindows`` builds those windows lazily from a ``[T, S, F]``
  array, one bar at a time.
- ``TargetTransform`` turns a bar's raw labels into the training target (a
  per-bar rank or z-score, optionally dropping the extremes).
- ``ValLossPatience`` and ``TrainLossThreshold`` are stopping helpers a
  head calls from its stop hooks (``_should_stop``, ``_on_fit_end``).
- ``masked_mse`` is a loss for the step hooks that leaves missing labels out.

Each head picks a transform and a stopping behaviour following its reference
implementation; metrics are always computed on the raw label, never on the
transformed target.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from scipy.stats import rankdata


class CrossSectionWindows:
    """Per-bar windows over a ``[T, S, F]`` feature array.

    The cross-section of bar ``t`` is the symbols with at least one finite
    feature there, judged on the raw values. Each window holds bars
    ``t - N + 1`` to ``t``, oldest first, so its last row is bar ``t``.
    Values are clipped to ``[-clip, clip]`` when ``clip`` is set, then every
    non-finite value becomes 0; rows before the first bar of the array are
    zeros too. Only a clipped float32 copy of the array is kept, never the
    ``[T, S, N, F]`` stack of every window.

    Parameters
    ----------
    x : np.ndarray
        Features, ``[T, S, F]``.
    window_bars : int
        N, the number of bars per window; at least 1.
    clip : float or None
        Clip bound applied before NaN is replaced; None disables clipping.

    Raises
    ------
    ValueError
        If ``x`` is not three-dimensional or ``window_bars`` is below 1.

    Examples
    --------
    >>> x = np.arange(6, dtype=float).reshape(3, 2, 1)   # 3 bars, 2 symbols
    >>> x[0, 1, 0] = np.nan                              # S1 absent at bar 0
    >>> windows = CrossSectionWindows(x, window_bars=2, clip=3.0)
    >>> windows.symbols(0)
    array([0])
    >>> windows.window(1, windows.symbols(1))[:, :, 0]
    array([[0., 2.],
           [0., 3.]], dtype=float32)
    """

    def __init__(self, x: np.ndarray, window_bars: int, clip: float | None):
        """Build the windows; see the class docstring for parameters."""
        if x.ndim != 3:
            raise ValueError(f"expected a [T, S, F] array, got shape {x.shape}")
        if window_bars < 1:
            raise ValueError(f"window_bars must be at least 1, got {window_bars}")
        self.window_bars = int(window_bars)
        self.present = np.isfinite(x).any(axis=-1)
        values = np.asarray(x, dtype=np.float32).copy()
        if clip is not None:
            np.clip(values, -clip, clip, out=values)
        values[~np.isfinite(values)] = 0.0
        pad = np.zeros((self.window_bars - 1,) + values.shape[1:], dtype=np.float32)
        self._padded = np.concatenate([pad, values], axis=0)

    @property
    def num_times(self) -> int:
        """Number of bars T.

        Examples
        --------
        >>> CrossSectionWindows(np.zeros((4, 2, 1)), 2, None).num_times
        4
        """
        return self.present.shape[0]

    def symbols(self, t: int) -> np.ndarray:
        """Return the symbol positions in the cross-section of bar ``t``.

        Examples
        --------
        >>> CrossSectionWindows(np.zeros((1, 3, 1)), 1, None).symbols(0)
        array([0, 1, 2])
        """
        return np.flatnonzero(self.present[t])

    def window(self, t: int, symbols: np.ndarray) -> np.ndarray:
        """Return the ``[len(symbols), N, F]`` windows ending at bar ``t``.

        Examples
        --------
        >>> windows = CrossSectionWindows(np.ones((2, 3, 4)), 5, None)
        >>> windows.window(1, np.array([0, 2])).shape
        (2, 5, 4)
        """
        block = self._padded[t : t + self.window_bars, symbols]
        return np.ascontiguousarray(block.transpose(1, 0, 2))


@dataclass(frozen=True)
class TargetTransform:
    """How a bar's raw labels become the training target.

    Applied per bar, column by column, over the finite labels only; a NaN
    label stays NaN and is left out of the loss. ``"rank"`` is Qlib's
    ``CSRankNorm``: the percentile rank (ties averaged) minus 0.5, times
    3.46. ``"zscore"`` subtracts the bar's mean and divides by its sample
    standard deviation; a bar with fewer than two finite labels, or no
    spread, gives NaN. ``drop_extreme`` is the fraction removed from each
    tail of the bar's first label, as MASTER does: those symbols leave the
    training cross-section entirely, input included. The transform and the
    drop apply to the training loss only; the validation loss transforms
    without dropping, and metrics use the raw label.

    Parameters
    ----------
    kind : {"rank", "zscore"}
        The per-bar transform.
    drop_extreme : float, default 0.0
        Fraction dropped from each tail during training, in ``[0, 0.5)``.

    Raises
    ------
    ValueError
        If ``kind`` is unknown or ``drop_extreme`` is out of range.

    Examples
    --------
    >>> TargetTransform("rank").apply(np.array([[0.3], [np.nan], [0.1], [0.2]]))
    array([[ 1.73      ],
           [        nan],
           [-0.57666667],
           [ 0.57666667]])
    """

    kind: Literal["rank", "zscore"]
    drop_extreme: float = 0.0

    #: Qlib's ``CSRankNorm`` scale, which gives a uniform rank roughly unit std.
    RANK_SCALE = 3.46

    def __post_init__(self):
        """Validate ``kind`` and ``drop_extreme``."""
        if self.kind not in ("rank", "zscore"):
            raise ValueError(
                f"TargetTransform kind must be 'rank' or 'zscore', got {self.kind!r}"
            )
        if not 0.0 <= self.drop_extreme < 0.5:
            raise ValueError(
                f"TargetTransform drop_extreme must be in [0, 0.5), "
                f"got {self.drop_extreme}"
            )

    def kept(self, y: np.ndarray) -> np.ndarray:
        """Return a boolean mask of the symbols kept in the training cross-section.

        ``y`` is one bar's ``[S, L]`` raw labels. With ``k = int(drop_extreme
        * n)``, n the number of finite first labels, the k smallest and the k
        largest first labels are dropped (ties broken by position). Symbols
        with a NaN first label are kept: they are context, not targets.

        Examples
        --------
        >>> y = np.array([[5.0], [1.0], [np.nan], [3.0], [2.0], [4.0]])
        >>> TargetTransform("zscore", drop_extreme=0.2).kept(y)
        array([False, False,  True,  True,  True,  True])
        """
        keep = np.ones(y.shape[0], dtype=bool)
        finite = np.flatnonzero(np.isfinite(y[:, 0]))
        k = int(self.drop_extreme * len(finite))
        if k > 0:
            order = finite[np.argsort(y[finite, 0], kind="stable")]
            keep[order[:k]] = False
            keep[order[-k:]] = False
        return keep

    def apply(self, y: np.ndarray) -> np.ndarray:
        """Return one bar's ``[S, L]`` labels transformed column by column.

        Examples
        --------
        >>> TargetTransform("zscore").apply(np.array([[1.0], [3.0]]))
        array([[-0.70710678],
               [ 0.70710678]])
        """
        out = np.full(y.shape, np.nan, dtype=np.float64)
        for col in range(y.shape[1]):
            rows = np.isfinite(y[:, col])
            values = y[rows, col].astype(np.float64)
            if len(values) == 0:
                continue
            if self.kind == "rank":
                out[rows, col] = (
                    rankdata(values) / len(values) - 0.5
                ) * self.RANK_SCALE
            elif len(values) > 1:
                std = values.std(ddof=1)
                if std > 0:
                    out[rows, col] = (values - values.mean()) / std
        return out


class ValLossPatience:
    """Early stopping on the validation loss, for a head's stop hooks.

    ``update`` records one epoch's validation loss and snapshots the weights
    whenever it is the lowest so far; it returns True once ``patience``
    epochs in a row have not improved on it. ``restore`` loads the best
    snapshot back. A ``None`` or non-finite loss (no validation segment)
    never counts, so training then runs to ``config.epochs``. The object
    holds the state of one fit: build a fresh one in ``_on_fit_start``.

    Parameters
    ----------
    patience : int
        Epochs without improvement tolerated before stopping; at least 1.

    Raises
    ------
    ValueError
        If ``patience`` is below 1.

    Examples
    --------
    >>> net = torch.nn.Linear(1, 1)
    >>> rule = ValLossPatience(patience=2)
    >>> [rule.update(loss, net) for loss in (3.0, 2.0, 4.0, 5.0)]
    [False, False, False, True]
    >>> rule.best
    2.0
    """

    def __init__(self, patience: int):
        """Start a fit with no best loss; see the class docstring."""
        if patience < 1:
            raise ValueError(f"patience must be at least 1, got {patience}")
        self.patience = patience
        self.best = float("inf")
        self.bad_epochs = 0
        self.best_state: dict[str, torch.Tensor] | None = None

    def update(self, val_loss: float | None, model: torch.nn.Module) -> bool:
        """Record one epoch's validation loss; return True to stop.

        Examples
        --------
        >>> ValLossPatience(1).update(0.5, torch.nn.Linear(1, 1))
        False
        """
        if val_loss is None or not np.isfinite(val_loss):
            return False
        if val_loss < self.best:
            self.best = float(val_loss)
            self.bad_epochs = 0
            self.best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience

    def restore(self, model: torch.nn.Module) -> None:
        """Load the best epoch's weights into ``model``; no-op before any snapshot.

        Examples
        --------
        >>> net = torch.nn.Linear(1, 1)
        >>> rule = ValLossPatience(1)
        >>> _ = rule.update(0.5, net)
        >>> rule.restore(net)
        """
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


class TrainLossThreshold:
    """MASTER's stopping rule, for a head's stop hooks.

    ``update`` returns True once an epoch's training loss is at or below
    ``threshold``, or after ``max_epochs`` epochs; the head keeps the last
    weights either way. The object counts the epochs of one fit: build a
    fresh one in ``_on_fit_start``.

    Parameters
    ----------
    threshold : float
        Training loss to reach.
    max_epochs : int
        Epoch cap; at least 1. Training never runs past ``config.epochs``
        anyway.

    Raises
    ------
    ValueError
        If ``max_epochs`` is below 1.

    Examples
    --------
    >>> rule = TrainLossThreshold(threshold=0.95, max_epochs=40)
    >>> [rule.update(loss) for loss in (1.2, 1.0, 0.9)]
    [False, False, True]
    """

    def __init__(self, threshold: float, max_epochs: int):
        """Start a fit at epoch 0; see the class docstring."""
        if max_epochs < 1:
            raise ValueError(f"max_epochs must be at least 1, got {max_epochs}")
        self.threshold = threshold
        self.max_epochs = max_epochs
        self.epochs = 0

    def update(self, train_loss: float) -> bool:
        """Record one epoch's training loss; return True to stop.

        Examples
        --------
        >>> TrainLossThreshold(0.5, 1).update(1.0)
        True
        """
        self.epochs += 1
        return train_loss <= self.threshold or self.epochs >= self.max_epochs


def masked_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error over the entries where ``target`` is finite.

    The step hooks of a ``DLModel`` receive targets with NaN where a label is
    missing; this loss leaves those entries out.

    Parameters
    ----------
    pred : torch.Tensor
        Predictions, ``[S_t, L]``.
    target : torch.Tensor
        Targets of the same shape, NaN where missing.

    Returns
    -------
    torch.Tensor
        The scalar mean; NaN when no target is finite.

    Examples
    --------
    >>> masked_mse(torch.tensor([[1.0], [5.0]]), torch.tensor([[0.0], [float("nan")]]))
    tensor(1.)
    """
    mask = torch.isfinite(target)
    return ((pred[mask] - target[mask]) ** 2).mean()
