"""The live prediction store: a run's predictions past its end, one bar at a time.

A backtest run's prediction panel ends where the run ends. A strategy traded
live needs the prediction of every new bar, made by the same model from the
same stores; ``quantlab.backtest.live.predict_live_bar`` makes it, and this
module holds the store it is appended to, so a reader (an executor that must
not import model code) reads it with the run layer alone.

**Layout.** A Zarr store on ``(timestamp, symbol)``:

- one ``float`` variable per label of the run's prediction panel, named as
  the panel names it (``ret_5``), NaN where a symbol has no prediction (not
  an index member at the bar, or no features);
- ``timestamp``: the bars predicted, strictly increasing, one appended per
  call; ``symbol``: the union of the symbols of every row, sorted
  (``quantlab.utils.symbol_axis.sort_symbol_axis``), widened when a row brings
  a new one (its earlier rows are NaN). Both index coordinates are kept in
  one chunk (``XrBackend.append``);
- ``attrs``: ``format_version`` and ``labels`` exactly as a prediction panel
  writes them (``PredictionPanel``), so ``PredictionPanel.read`` reads the
  store too, plus ``live_format_version`` (``FORMAT_VERSION``), ``run_dir``
  (the backtest run directory, absolute), ``checkpoint`` (the checkpoint
  file the rows were predicted with, absolute) and ``trained_run`` (its
  trained unit's directory).

**Records.** Beside the store, ``<store>.rows.json`` maps each bar's label
(``bar_label``: ``"2026-10-07"`` for a daily bar) to the record of the row:
``timestamp`` (ISO), ``written_at`` (UTC, ISO), ``run_dir``, ``checkpoint``,
``data_fingerprint`` (what the prediction read, by component path of the
run's backtester, as ``quantlab.runs.record.DataRecorder`` records it) and
whatever else the writer adds (``predict_live_bar`` adds ``stores``, the
stores it extended, and ``lagging``). The row is appended first and its
record written after it, so a crash between the two leaves a row without a
record, never a record without a row.

Examples
--------
>>> store = LivePredictionStore("live/live_predictions.zarr")
>>> store.bars()[-1]
Timestamp('2026-10-07 00:00:00')
>>> row = store.row("2026-10-07")
>>> row["ret_5"].dims
('symbol',)
>>> store.record("2026-10-07")["checkpoint"]
'/data/quantlab/pipeline/sharadar_sp500/models/xgb_mvo/XGBoostRegressor_trial_0/model.joblib'
"""

import dataclasses
import json
from collections.abc import Mapping, Sequence
from os import PathLike
from pathlib import Path

import pandas as pd
import xarray as xr

from quantlab.backend.zarr import XrBackend
from quantlab.runs.prediction_panel import LabelSpec, PredictionPanel
from quantlab.utils.atomic import write_json_atomically
from quantlab.utils.date_range import bar_label
from quantlab.utils.jsonable import to_jsonable

__all__ = ["LivePredictionStore"]

_DIMS = ("timestamp", "symbol")
#: The attributes that tie a store to the run and checkpoint predicting it.
_IDENTITY = ("run_dir", "checkpoint", "trained_run", "labels")


class LivePredictionStore:
    """An append-only store of one run's live predictions, one bar per ``append``.

    See the module docstring for the layout. The store is opened on every
    call and nothing is held, so one object serves a writer and readers.

    Parameters
    ----------
    path : str or os.PathLike
        Directory of the Zarr store; it need not exist yet.

    Attributes
    ----------
    path : Path
        The store.
    records_path : Path
        ``<store>.rows.json``, the per-row records.

    Examples
    --------
    >>> store = LivePredictionStore("live/live_predictions.zarr")
    >>> store.exists, store.records_path.name
    (False, 'live_predictions.zarr.rows.json')
    """

    #: The version of the live attributes and the records file.
    FORMAT_VERSION = 1
    #: Appended to the store's path to name its records file.
    RECORDS_SUFFIX = ".rows.json"

    def __init__(self, path: str | PathLike):
        """Initialize the store; see the class docstring for parameters."""
        self.path = Path(path)
        self.records_path = Path(f"{self.path}{self.RECORDS_SUFFIX}")

    def __repr__(self) -> str:
        """Return the path."""
        return f"LivePredictionStore({str(self.path)!r})"

    @property
    def exists(self) -> bool:
        """Whether the store has been written.

        Examples
        --------
        >>> LivePredictionStore("nowhere.zarr").exists
        False
        """
        return self.path.exists()

    def _open(self) -> xr.Dataset:
        """Open the store lazily."""
        return XrBackend().read(self.path).data

    def bars(self) -> pd.DatetimeIndex:
        """The bars predicted so far, in order; empty before the first ``append``.

        Examples
        --------
        >>> LivePredictionStore("live/live_predictions.zarr").bars()
        DatetimeIndex(['2026-10-06', '2026-10-07'], dtype='datetime64[ns]', name='timestamp', freq=None)
        """
        if not self.exists:
            return pd.DatetimeIndex([], name="timestamp")
        return pd.DatetimeIndex(self._open()["timestamp"].values, name="timestamp")

    def has(self, timestamp) -> bool:
        """Whether the bar ``timestamp`` has been predicted.

        Examples
        --------
        >>> LivePredictionStore("live/live_predictions.zarr").has("2026-10-07")
        True
        """
        return pd.Timestamp(timestamp) in self.bars()

    def attrs(self) -> dict:
        """The store's attributes (see the module docstring); empty before the first ``append``.

        Examples
        --------
        >>> sorted(LivePredictionStore("live/live_predictions.zarr").attrs())
        ['checkpoint', 'format_version', 'labels', 'live_format_version', 'run_dir', 'trained_run']
        """
        return dict(self._open().attrs) if self.exists else {}

    def labels(self) -> tuple[LabelSpec, ...]:
        """The specs of the labels the store's variables predict.

        Examples
        --------
        >>> LivePredictionStore("live/live_predictions.zarr").labels()
        (LabelSpec(name='ret_5', scale='raw', delay=1, span=5),)
        """
        return PredictionPanel.read_labels(self.path)

    def row(self, timestamp) -> xr.Dataset:
        """The predictions of one bar, loaded: one variable per label on ``symbol``.

        Parameters
        ----------
        timestamp : str or datetime-like
            A bar of the store.

        Returns
        -------
        xr.Dataset
            The row, with ``timestamp`` kept as a scalar coordinate.

        Raises
        ------
        KeyError
            If the bar has not been predicted.

        Examples
        --------
        >>> LivePredictionStore("live/live_predictions.zarr").row("2026-10-07")["ret_5"].dims
        ('symbol',)
        """
        bar = pd.Timestamp(timestamp)
        if not self.has(bar):
            raise KeyError(f"{self.path} holds no prediction of {bar_label(bar)}")
        row = self._open().sel(timestamp=bar).load()
        row.attrs = {}
        return row

    def records(self) -> dict:
        """Every row's record, keyed by bar label (see the module docstring).

        Examples
        --------
        >>> list(LivePredictionStore("live/live_predictions.zarr").records())
        ['2026-10-06', '2026-10-07']
        """
        if not self.records_path.is_file():
            return {}
        return json.loads(self.records_path.read_text(encoding="utf-8"))

    def record(self, timestamp) -> dict | None:
        """The record of one bar's row, or None when it has none.

        Examples
        --------
        >>> sorted(LivePredictionStore("live/live_predictions.zarr").record("2026-10-07"))[:3]
        ['checkpoint', 'data_fingerprint', 'lagging']
        """
        return self.records().get(bar_label(timestamp))

    @staticmethod
    def header(
        labels: Sequence[LabelSpec], *, run_dir, checkpoint, trained_run=None
    ) -> dict:
        """The attributes a store of these labels, run and checkpoint carries.

        Parameters
        ----------
        labels : sequence of LabelSpec
            The run's label specs, in the order of its prediction variables.
        run_dir, checkpoint, trained_run : str or os.PathLike
            The backtest run directory, the checkpoint file and its trained
            unit's directory; written as absolute paths.

        Returns
        -------
        dict
            JSON-ready attributes.

        Examples
        --------
        >>> LivePredictionStore.header(
        ...     [LabelSpec("ret_5", "raw", 1, 5)], run_dir="/runs/a", checkpoint="/m/model.joblib"
        ... )["labels"]
        '[{"name": "ret_5", "scale": "raw", "delay": 1, "span": 5}]'
        """
        return {
            "format_version": PredictionPanel.FORMAT_VERSION,
            "labels": json.dumps([dataclasses.asdict(spec) for spec in labels]),
            "live_format_version": LivePredictionStore.FORMAT_VERSION,
            "run_dir": str(Path(run_dir).absolute()),
            "checkpoint": str(Path(checkpoint).absolute()),
            "trained_run": None if trained_run is None else str(Path(trained_run).absolute()),
        }

    def check_header(self, header: Mapping) -> None:
        """Refuse a writer whose run, checkpoint or labels are not the store's.

        A store holds the predictions of one run and one checkpoint; a
        retrained model starts a new store. Nothing is checked before the
        first ``append``.

        Raises
        ------
        ValueError
            Naming each attribute that differs.

        Examples
        --------
        >>> store.check_header(LivePredictionStore.header(specs, run_dir=other_run, checkpoint=cp))
        Traceback (most recent call last):
        ValueError: live/live_predictions.zarr holds the predictions of another run: run_dir ...
        """
        stored = self.attrs()
        if not stored:
            return
        if stored.get("live_format_version") != self.FORMAT_VERSION:
            raise ValueError(
                f"{self.path} is not a live prediction store of live_format_version "
                f"{self.FORMAT_VERSION} (found {stored.get('live_format_version')!r})"
            )
        differ = [
            f"{name} {stored.get(name)!r} (writer: {header.get(name)!r})"
            for name in _IDENTITY
            if stored.get(name) != header.get(name)
        ]
        if differ:
            raise ValueError(
                f"{self.path} holds the predictions of another run: {'; '.join(differ)}. "
                f"Write to a new store."
            )

    def append(self, row: xr.Dataset, header: Mapping, record: Mapping) -> None:
        """Append one bar's predictions, then its record.

        Parameters
        ----------
        row : xr.Dataset
            The predictions of one bar on ``(timestamp, symbol)``, one
            timestamp, the variables exactly the label names of ``header``.
        header : mapping
            The store's attributes (``header``), checked against a store
            already written (``check_header``).
        record : mapping
            JSON-ready; written under the bar's label with ``timestamp``
            added.

        Raises
        ------
        ValueError
            If the row is not one bar, its variables are not the labels, the
            bar is not after the store's last, or ``check_header`` refuses.

        Examples
        --------
        >>> store.append(row, header, {"written_at": "2026-10-08T10:02:11+00:00"})
        >>> store.bars()[-1] == row["timestamp"].values[0]
        True
        """
        self.check_header(header)
        names = [spec["name"] for spec in json.loads(header["labels"])]
        if sorted(map(str, row.data_vars)) != sorted(names):
            raise ValueError(
                f"{self.path}: the row's variables {sorted(map(str, row.data_vars))} are not "
                f"the labels {names}"
            )
        if row.sizes.get("timestamp") != 1:
            raise ValueError(
                f"{self.path}: a row is one bar, got {row.sizes.get('timestamp')} timestamp(s)"
            )
        bar = pd.Timestamp(row["timestamp"].values[0])
        bars = self.bars()
        if len(bars) and bar <= bars[-1]:
            raise ValueError(
                f"{self.path}: bar {bar_label(bar)} is not after the store's last bar "
                f"{bar_label(bars[-1])}; the store is append-only"
            )
        data = row[names].transpose(*_DIMS)
        data = data.drop_vars([c for c in data.coords if c not in _DIMS])
        data.attrs = dict(header)
        for name in data.data_vars:
            data[name].attrs = {}
        XrBackend().to_internal(data).widen_and_append(str(self.path))
        records = self.records()
        records[bar_label(bar)] = to_jsonable({"timestamp": bar.isoformat(), **record})
        write_json_atomically(self.records_path, records, indent=1)
