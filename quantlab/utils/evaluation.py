"""Evaluation: score a trained unit's predictions against its raw labels.

A model and an ensemble are both scored here, after training, by one rule:
``evaluate`` takes the unit's prediction panel (a model's ``predict_panel``
over its collected panel, an ensemble's combined prediction), the panel of
the labels' raw values and the unit's train / validation / test segments,
returns the metrics and writes the unit's two evaluation files,
``ic_series.csv`` and ``test_predictions.zarr``. It needs no model
instance, so predictions can be scored without training anything.

Scoring rules, one set for every unit:

- Every label is scored on every non-empty segment. The first label's keys
  are ``{split}_{metric}``, every other label's ``{split}_{label}_{metric}``.
- The IC family (``ic``, ``rank_ic``, ``icir``, ``rank_icir``, see
  ``quantlab.utils.metrics.ic_panel_metrics``) always.
- The error metrics ``mse``, ``rmse``, ``mae`` and ``r2`` only for a label
  whose prediction is on its own scale (label scale ``"raw"``): a
  standardized prediction is in other units than the label.
- ``qlike`` and ``variance_ratio`` (``volatility_level_metrics``) where
  ``scores_volatility_level`` holds: a volatility label predicted raw.
- ``member_correlation`` (``quantlab.utils.ensemble.member_correlation``)
  for a label an ensemble's combined prediction averages over several
  members, given their predictions.
"""

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from quantlab.runs.trained_run import evaluation_paths
from quantlab.utils.ensemble import member_correlation
from quantlab.utils.metrics import (
    ic_panel_metrics,
    mae,
    mse,
    r2,
    rmse,
    scores_volatility_level,
    volatility_level_metrics,
)
from quantlab.utils.split import purge_segments

#: The splits a unit is scored on, in the order they are written.
SPLITS: tuple[str, ...] = ("train", "val", "test")


@dataclass(frozen=True)
class Segments:
    """The bars of a unit's train, validation and test segments.

    Each field is an array of timestamps, empty when the unit has no such
    segment; a model's come from its purged split of the training window
    and its test dates.

    Examples
    --------
    >>> segments = Segments(train=stamps[:20], val=stamps[:0], test=stamps[25:])
    >>> [split for split, _ in segments.splits()]
    ['train', 'test']
    """

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

    def splits(self) -> list[tuple[str, np.ndarray]]:
        """Return ``(split, timestamps)`` for each non-empty segment, in ``SPLITS`` order.

        Examples
        --------
        >>> Segments(train=stamps[:2], val=stamps[:0], test=stamps[2:3]).splits()[1][0]
        'test'
        """
        return [
            (split, np.asarray(stamps))
            for split, stamps in zip(SPLITS, (self.train, self.val, self.test))
            if len(stamps)
        ]


def evaluate(
    predictions: xr.Dataset,
    truth: xr.Dataset | Mapping[str, xr.Dataset],
    *,
    labels: Mapping[str, object],
    label_scales: Mapping[str, str],
    segments: Segments | Mapping[str, Segments],
    test_bounds: tuple,
    run_dir: Path | str | None,
    member_predictions: Mapping[str, Sequence[xr.Dataset]] | None = None,
) -> dict[str, float]:
    """Score a unit's predictions of every label and write its evaluation files.

    Each label's prediction is taken on its segment's bars and on the
    symbols of ``truth``, so a predicted symbol without truth is ignored and
    a missing prediction counts as missing.

    Parameters
    ----------
    predictions : xr.Dataset
        One variable per label name on ``(timestamp, symbol)``.
    truth : xr.Dataset or Mapping[str, xr.Dataset]
        The labels' raw values, one variable per label name on
        ``(timestamp, symbol)``, covering every segment's bars; or each
        label's own panel holding it (an ensemble scores a label on the
        symbols of the first member predicting it).
    labels : Mapping[str, object]
        Label name to the label object predicting it (read for its
        ``kind``), in label order: the first key is the first label.
    label_scales : Mapping[str, str]
        Label name to its prediction scale, ``"raw"`` or ``"standardized"``.
    segments : Segments or Mapping[str, Segments]
        The unit's segments, or each label's own (an ensemble scores a label
        on the segments of the first member predicting it).
    test_bounds : tuple
        The unit's ``(test_start, test_end)``; ``test_predictions.zarr``
        keeps the test bars inside them.
    run_dir : Path or str or None
        Directory receiving ``ic_series.csv`` and ``test_predictions.zarr``;
        None writes nothing.
    member_predictions : Mapping[str, Sequence[xr.Dataset]], optional
        For an ensemble: label name to the prediction panels of the members
        predicting it. A label with at least two also gets
        ``{split}_member_correlation`` (prefixed as its other keys), how
        much the members agree on its segment's bars and the truth's
        symbols.

    Returns
    -------
    dict[str, float]
        The metrics, NaN where a metric is undefined.

    Notes
    -----
    Written into ``run_dir``:

    - ``ic_series.csv``, columns ``split, timestamp, ic, rank_ic``: the first
      label's per-bar IC and rank IC behind its ``{split}_ic`` /
      ``{split}_icir``, one row per bar of each scored split in ``SPLITS``
      order; a bar where both are undefined has no row.
    - ``test_predictions.zarr``: every label's prediction on the first
      label's test bars inside ``test_bounds``; not written when there are
      none.

    Examples
    --------
    >>> metrics = evaluate(
    ...     predictions, truth, labels={"ret": label}, label_scales={"ret": "raw"},
    ...     segments=segments, test_bounds=("2024-01-31", "2024-02-09"), run_dir=None,
    ... )
    >>> sorted(metrics)[:4]
    ['test_ic', 'test_icir', 'test_mae', 'test_mse']
    """
    names = [str(name) for name in labels]
    per_label = (
        segments if not isinstance(segments, Segments) else {name: segments for name in names}
    )
    truths = truth if not isinstance(truth, xr.Dataset) else {name: truth for name in names}
    metrics: dict[str, float] = {}
    series: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for i, name in enumerate(names):
        prefix = "" if i == 0 else f"{name}_"
        scale = label_scales.get(name)
        level = scores_volatility_level(labels[name], scale)
        for split, stamps in per_label[name].splits():
            target = truths[name][name].sel(timestamp=stamps)
            symbols = target.symbol.values
            pred = (
                predictions[name]
                .reindex(timestamp=stamps, symbol=symbols)
                .values
            )
            target = target.values
            values, per_bar = ic_panel_metrics(pred, target, return_series=True)
            if scale == "raw":
                values.update(
                    mse=mse(pred, target), rmse=rmse(pred, target),
                    mae=mae(pred, target), r2=r2(pred, target),
                )
            if level:
                values.update(volatility_level_metrics(pred, target))
            members = (member_predictions or {}).get(name, ())
            if len(members) > 1:
                values["member_correlation"], _ = member_correlation(
                    [
                        panel[name]
                        .reindex(timestamp=stamps, symbol=symbols)
                        .values
                        for panel in members
                    ]
                )
            metrics.update({f"{split}_{prefix}{key}": value for key, value in values.items()})
            if i == 0:
                series[split] = (stamps, per_bar["ic"], per_bar["rank_ic"])

    if run_dir is not None:
        ic_series_path, test_predictions_path = evaluation_paths(run_dir)
        write_ic_series(ic_series_path, series)
        test = np.sort(np.asarray(per_label[names[0]].test)) if names else np.array([])
        # Bounds resolve as the segments do: a date string covers its whole day.
        (test,) = purge_segments(test, [tuple(test_bounds)], 0)
        if len(test):
            predictions[names].reindex(timestamp=test).to_zarr(
                test_predictions_path, mode="w"
            )
    return metrics


def write_ic_series(path: Path, series: Mapping) -> None:
    """Write per-bar IC series to ``path`` as ``ic_series.csv``, atomically.

    ``series`` maps a split name to ``(timestamps, ic, rank_ic)`` arrays of
    one length. The rows follow the splits of ``SPLITS`` that ``series``
    holds, each in its given order; a bar where neither value is finite has
    no row.

    Examples
    --------
    >>> ic_series_path, _ = evaluation_paths(unit)
    >>> write_ic_series(ic_series_path, {"test": (stamps, ic, rank_ic)})
    >>> pd.read_csv(ic_series_path).columns.tolist()
    ['split', 'timestamp', 'ic', 'rank_ic']
    """
    rows = []
    for split in SPLITS:
        if split not in series:
            continue
        stamps, ic, rank_ic = series[split]
        keep = np.isfinite(ic) | np.isfinite(rank_ic)
        rows.append(
            pd.DataFrame(
                {
                    "split": split,
                    "timestamp": np.asarray(stamps)[keep],
                    "ic": ic[keep],
                    "rank_ic": rank_ic[keep],
                }
            )
        )
    frame = (
        pd.concat(rows, ignore_index=True)
        if rows
        else pd.DataFrame(columns=["split", "timestamp", "ic", "rank_ic"])
    )
    path = Path(path)
    staging = path.with_name(path.name + ".tmp")
    frame.to_csv(staging, index=False)
    os.replace(staging, path)
