"""Training pieces of a deep head, and the per-bar windows it is fed.

A ``DLModel`` trains on one *cross-section* per step: the symbols with at
least one finite feature at a bar, each carrying its last N bars of features
(ADR 0006). This module holds the pieces of that loop:

- ``CrossSectionWindows`` builds those windows lazily from a ``[T, S, F]``
  array, one bar at a time; the base class uses it.
- ``CrossSectionBatch`` is what every step hook receives: one bar's inputs,
  targets and target mask.
- ``cs_rank_norm``, ``cs_zscore`` and ``drop_extreme`` are target transforms
  for a head's ``_transform_target``.
- ``masked_mse`` is a loss for a head's ``_loss``.
- ``TrainLossThreshold`` is MASTER's stopping rule, a helper for a head's
  stop hooks.

Everything a head picks is optional to use: the hooks may be written from
scratch.
"""

from dataclasses import dataclass

import numpy as np
import torch
from scipy.stats import rankdata


class CrossSectionWindows:
    """Per-bar windows over a ``[T, S, F]`` feature array.

    The cross-section of bar ``t`` is the symbols with at least one finite
    feature there. Each window holds bars ``t - N + 1`` to ``t``, oldest
    first, so its last row is bar ``t``. Values are returned as they are,
    NaN included; rows before the first bar of the array are NaN too. Only a
    float32 copy of the array is kept, never the ``[T, S, N, F]`` stack of
    every window.

    Parameters
    ----------
    x : np.ndarray
        Features, ``[T, S, F]``.
    window_bars : int
        N, the number of bars per window; at least 1.

    Raises
    ------
    ValueError
        If ``x`` is not three-dimensional or ``window_bars`` is below 1.

    Examples
    --------
    >>> x = np.arange(6, dtype=float).reshape(3, 2, 1)   # 3 bars, 2 symbols
    >>> x[0, 1, 0] = np.nan                              # S1 absent at bar 0
    >>> windows = CrossSectionWindows(x, window_bars=2)
    >>> windows.symbols(0)
    array([0])
    >>> windows.window(1, windows.symbols(1))[:, :, 0]
    array([[ 0.,  2.],
           [nan,  3.]], dtype=float32)
    """

    def __init__(self, x: np.ndarray, window_bars: int):
        """Build the windows; see the class docstring for parameters."""
        if x.ndim != 3:
            raise ValueError(f"expected a [T, S, F] array, got shape {x.shape}")
        if window_bars < 1:
            raise ValueError(f"window_bars must be at least 1, got {window_bars}")
        self.window_bars = int(window_bars)
        self.present = np.isfinite(x).any(axis=-1)
        values = np.asarray(x, dtype=np.float32)
        pad = np.full((self.window_bars - 1,) + values.shape[1:], np.nan, dtype=np.float32)
        self._padded = np.concatenate([pad, values], axis=0)

    @property
    def num_times(self) -> int:
        """Number of bars T.

        Examples
        --------
        >>> CrossSectionWindows(np.zeros((4, 2, 1)), 2).num_times
        4
        """
        return self.present.shape[0]

    def symbols(self, t: int) -> np.ndarray:
        """Return the symbol positions in the cross-section of bar ``t``.

        Examples
        --------
        >>> CrossSectionWindows(np.zeros((1, 3, 1)), 1).symbols(0)
        array([0, 1, 2])
        """
        return np.flatnonzero(self.present[t])

    def window(self, t: int, symbols: np.ndarray) -> np.ndarray:
        """Return the ``[len(symbols), N, F]`` windows ending at bar ``t``.

        Examples
        --------
        >>> windows = CrossSectionWindows(np.ones((2, 3, 4)), 5)
        >>> windows.window(1, np.array([0, 2])).shape
        (2, 5, 4)
        """
        block = self._padded[t : t + self.window_bars, symbols]
        return np.ascontiguousarray(block.transpose(1, 0, 2))


@dataclass
class CrossSectionBatch:
    """One bar's cross-section, as every step hook and ``_loss`` receive it.

    Attributes
    ----------
    x : torch.Tensor
        ``[S_t, N, F]`` windows after the head's ``_transform_feature``.
    y : torch.Tensor
        ``[S_t, L]`` targets after the head's ``_transform_target``, with
        every invalid entry set to 0.
    mask : torch.Tensor
        ``[S_t, L]`` booleans, True where ``y`` is a valid target: the label
        exists and its transform is finite. A loss should count only these.
    y_raw : torch.Tensor
        ``[S_t, L]`` raw labels of the same symbols, NaN where missing.
    symbols : np.ndarray
        The symbol labels of the ``S_t`` rows, in row order.
    timestamp : np.datetime64
        The bar.

    Examples
    --------
    >>> batch = CrossSectionBatch(
    ...     x=torch.zeros(2, 1, 1), y=torch.tensor([[1.0], [0.0]]),
    ...     mask=torch.tensor([[True], [False]]),
    ...     y_raw=torch.tensor([[1.0], [float("nan")]]),
    ...     symbols=np.array(["A", "B"]), timestamp=np.datetime64("2024-01-02"),
    ... )
    >>> int(batch.mask.sum())
    1
    """

    x: torch.Tensor
    y: torch.Tensor
    mask: torch.Tensor
    y_raw: torch.Tensor
    symbols: np.ndarray
    timestamp: np.datetime64


#: Qlib's ``CSRankNorm`` scale, which gives a uniform rank roughly unit std.
RANK_SCALE = 3.46


def cs_rank_norm(y: torch.Tensor) -> torch.Tensor:
    """Qlib's ``CSRankNorm`` of one bar's ``[S_t, L]`` labels, column by column.

    The percentile rank among the finite labels (ties averaged), minus 0.5,
    times 3.46. NaN stays NaN.

    Parameters
    ----------
    y : torch.Tensor
        One bar's labels, ``[S_t, L]``.

    Returns
    -------
    torch.Tensor
        The ranked labels, same shape, device and dtype.

    Examples
    --------
    >>> cs_rank_norm(torch.tensor([[0.3], [float("nan")], [0.1], [0.2]]))
    tensor([[ 1.7300],
            [    nan],
            [-0.5767],
            [ 0.5767]])
    """
    values = y.detach().cpu().numpy().astype(np.float64)
    out = np.full(values.shape, np.nan)
    for col in range(values.shape[1]):
        rows = np.isfinite(values[:, col])
        if rows.any():
            out[rows, col] = (rankdata(values[rows, col]) / rows.sum() - 0.5) * RANK_SCALE
    return torch.as_tensor(out, dtype=y.dtype, device=y.device)


def cs_zscore(y: torch.Tensor) -> torch.Tensor:
    """Z-score of one bar's ``[S_t, L]`` labels, column by column.

    Subtracts the mean of the finite labels and divides by their sample
    standard deviation. NaN stays NaN; a column with fewer than two finite
    labels, or no spread, becomes NaN.

    Parameters
    ----------
    y : torch.Tensor
        One bar's labels, ``[S_t, L]``.

    Returns
    -------
    torch.Tensor
        The standardised labels, same shape.

    Examples
    --------
    >>> cs_zscore(torch.tensor([[1.0], [3.0], [float("nan")]]))
    tensor([[-0.7071],
            [ 0.7071],
            [    nan]])
    """
    finite = torch.isfinite(y)
    count = finite.sum(dim=0, keepdim=True)
    filled = torch.where(finite, y, torch.zeros_like(y))
    mean = filled.sum(dim=0, keepdim=True) / count
    centred = torch.where(finite, y - mean, torch.zeros_like(y))
    std = torch.sqrt((centred**2).sum(dim=0, keepdim=True) / (count - 1))
    out = (y - mean) / std
    valid = finite & (count > 1) & (std > 0)
    return torch.where(valid, out, torch.full_like(y, float("nan")))


def drop_extreme(y: torch.Tensor, fraction: float, column: int = 0) -> torch.Tensor:
    """Return the rows kept after dropping both tails of one label column.

    With ``k = int(fraction * n)``, n the number of finite labels in
    ``column``, the k smallest and the k largest are dropped (ties broken by
    position), as MASTER does in training. Rows with a NaN label are kept.

    Parameters
    ----------
    y : torch.Tensor
        One bar's labels, ``[S_t, L]``.
    fraction : float
        Share dropped from each tail, in ``[0, 0.5)``.
    column : int, default 0
        The label column that decides.

    Returns
    -------
    torch.Tensor
        ``[S_t]`` booleans, True for the rows kept.

    Raises
    ------
    ValueError
        If ``fraction`` is outside ``[0, 0.5)``.

    Examples
    --------
    >>> y = torch.tensor([[5.0], [1.0], [float("nan")], [3.0], [2.0], [4.0]])
    >>> drop_extreme(y, 0.2)
    tensor([False, False,  True,  True,  True,  True])
    """
    if not 0.0 <= fraction < 0.5:
        raise ValueError(f"drop_extreme fraction must be in [0, 0.5), got {fraction}")
    values = y[:, column].detach().cpu().numpy()
    keep = np.ones(len(values), dtype=bool)
    finite = np.flatnonzero(np.isfinite(values))
    k = int(fraction * len(finite))
    if k > 0:
        order = finite[np.argsort(values[finite], kind="stable")]
        keep[order[:k]] = False
        keep[order[-k:]] = False
    return torch.as_tensor(keep, device=y.device)


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


def masked_mse(pred: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean squared error over the entries where ``mask`` is True.

    Parameters
    ----------
    pred : torch.Tensor
        Predictions, ``[S_t, L]``.
    y : torch.Tensor
        Targets of the same shape.
    mask : torch.Tensor
        Booleans of the same shape; typically ``batch.mask``.

    Returns
    -------
    torch.Tensor
        The scalar mean; NaN when no entry is valid.

    Examples
    --------
    >>> masked_mse(torch.tensor([[1.0], [5.0]]), torch.tensor([[0.0], [0.0]]),
    ...            torch.tensor([[True], [False]]))
    tensor(1.)
    """
    return ((pred[mask] - y[mask]) ** 2).mean()
