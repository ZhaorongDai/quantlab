"""Abstract factor layer and its two computation backends.

A *factor* turns market data into engineered features, for example a
20-day momentum or a moving-average deviation. It reads a *panel* of
a dataset (an ``xarray.Dataset`` indexed by ``timestamp`` and ``symbol``)
and produces a panel of the same shape. Label classes use the same machinery
to produce prediction targets, such as forward returns.

``Factor`` is the backend-agnostic contract the model layer programs
against. ``FactorKunQuant`` describes a factor as a KunQuant operator graph;
KunQuant compiles that graph to native code and runs it either over the whole
history (batch mode) or one bar at a time (streaming mode, for live data).
``FactorPolars`` is a batch-only backend whose factor logic is a Polars
expression chain. Concrete factor sets live under ``quantlab/factor`` and
labels under ``quantlab/label``.

Rolling operators need history before the first bar they report. That extra
history is the *warm-up*. ``compute(start, end)`` reads ``config.warmup_bars``
bars before ``start``, counted on the input dataset's own calendar, and trims
them off again.
"""

import copy
import dataclasses
import datetime
import json
import sys
import warnings
from abc import ABC, abstractmethod
from pathlib import Path

from typing import TYPE_CHECKING, Literal, Self

import KunQuant.runner.KunRunner as kr
import numpy as np
import pandas as pd
import polars as pl
import xarray as xr
from KunQuant.Driver import KunCompilerConfig
from KunQuant.jit import cfake
from KunQuant.Stage import Function

from quantlab.base.config import (
    BaseFactorConfig,
    FactorConfig,
    PolarsFactorConfig,
)
from quantlab.backend import XrBackend
from quantlab.base.data import InsufficientHistoryError
from quantlab.utils.atomic import write_json_atomically
from quantlab.utils.date_range import (
    as_label,
    check_range,
    last_moment,
    range_text,
    resample_padding,
)
from quantlab.utils.resample import (
    assert_coarser,
    resample_store_path,
    resolve_resample_how,
    validate_resample_config,
)
from quantlab.utils.timer import Timer

if TYPE_CHECKING:
    from quantlab.analysis.factor_report import FactorAnalysis


def _as_one_dataset(dataset):
    """Return ``dataset``, or a ``MergedDataset`` of it when it is a list or tuple."""
    if isinstance(dataset, (list, tuple)):
        # Imported here so the base layer does not import the dataset layer.
        from quantlab.dataset.merged import MergedDataset

        return MergedDataset(dataset)
    return dataset


def _on_panel_axes(data: xr.Dataset) -> xr.Dataset:
    """Return ``data`` laid out on ``(timestamp, symbol)``."""
    return XrBackend().to_internal(data).get_xarray_dataset(["timestamp", "symbol"])


class Factor(ABC):
    """Backend-agnostic base class for factors and labels.

    A ``Factor`` holds a config whose ``dataset`` attribute is the dataset it
    reads from. Asked for a date range, it answers ``compute(start, end)``
    from its inputs or ``read(start, end)`` from its own store, which
    ``build(start, end)`` writes and ``extend(end)`` lengthens. None of these
    holds a panel or changes a config, the factor's or its dataset's, so one
    dataset object can feed several factors. ``get_features(panel)`` and
    ``get_labels(panel)`` turn a returned panel into what the model layer
    consumes. Subclasses implement ``_get_factor_names`` and
    ``_compute_panel`` (both backends here do) and override
    ``_get_features`` and/or ``_get_labels`` for the half they support.

    Assigning ``config`` runs the property setter, which sets ``name`` and
    resolves ``factor_names`` when they are not pinned, so factor names are
    known as soon as the object is constructed, before anything is
    computed.

    Parameters
    ----------
    config : BaseFactorConfig
        The factor config. Its ``dataset`` field is the dataset it
        reads from, or a list of datasets, which the factor merges into one
        ``MergedDataset`` (see ``quantlab.dataset.merged``).

    Attributes
    ----------
    data_backend : XrBackend
        Holds the last bar a stream-mode factor computed; empty otherwise.

    Examples
    --------
    >>> factor = MyFactor(config)
    >>> factor.get_factor_names()          # known before anything is computed
    >>> panel = factor.compute("2024-02-01", "2024-02-29")
    >>> factor.build("2024-01-01", "2024-06-30")
    >>> panel = factor.read("2024-02-01", "2024-02-29")
    """

    #: Appended to ``store_path`` to name the JSON file recording the date
    #: range ``build`` wrote and ``extend`` lengthened.
    RANGE_SUFFIX = ".range.json"

    def __init__(self, config: BaseFactorConfig):
        """Initialize the factor; see the class docstring for parameters."""
        # The config setter runs before `self.data_backend` exists, so nothing
        # the setter reaches may use the storage backend.
        self.config = config
        self.data_backend = XrBackend()

    def __repr__(self) -> str:
        """Return the class name and its config."""
        return f"{self.__class__.__name__}(config={self.config})"

    def __eq__(self, other: object) -> bool:
        """Return whether ``other`` is a factor of the same class with an equal config.

        A factor is identified by its config, so a factor rebuilt from its
        ``config.json`` equals the one that wrote it, and so does a config
        holding it. What a build call holds in memory is not compared.
        Factors are unhashable.

        Examples
        --------
        >>> MyFactor(config) == MyFactor(config)
        True
        """
        if type(other) is not type(self):
            return NotImplemented
        return self.config == other.config

    @property
    def config(self) -> BaseFactorConfig:
        """The factor's normalised config: ``name`` and ``factor_names`` filled in.

        Examples
        --------
        >>> factor.config is config
        False
        >>> factor.config.name, config.name
        ('quantlab.factor.momentum.Momentum', None)
        """
        return self._config

    @config.setter
    def config(self, config: BaseFactorConfig):
        """Install a normalised copy of ``config``.

        ``name`` is set to this class's import path, a list or tuple in
        ``dataset`` becomes a ``MergedDataset`` of it, the resample fields
        are checked, ``_check_dataset`` runs, ``factor_names`` are resolved
        if unset, and ``_validate_config`` runs. The config passed in is never edited, and
        nothing is written into the dataset's config. If any step raises, the
        factor keeps the config it had. Nothing here touches the storage
        backend, which does not exist yet when ``__init__`` assigns the
        config.

        Parameters
        ----------
        config : BaseFactorConfig
            The factor config to install.

        Examples
        --------
        >>> config = PolarsFactorConfig(
        ...     warmup_bars=5, dataset=dataset, kwargs={"n": 5},
        ...     file_path="momentum_5.zarr",
        ... )
        >>> factor.config = config
        >>> factor.config.factor_names, config.factor_names
        (('momentum_5',), None)
        >>> factor.config.dataset.config is dataset.config   # left untouched
        True
        >>> factor.config = dataclasses.replace(config, dataset=[index, etf])
        >>> type(factor.config.dataset).__name__
        'MergedDataset'
        """
        validate_resample_config(
            config.resample_freq, config.resample_how, self.class_name
        )
        # `_get_factor_names` and `_validate_config` read `self.config` (a
        # label's horizon sits in its kwargs), so the candidate is installed
        # first and the previous config restored if either raises.
        previous = self.__dict__.get("_config")
        self._config = dataclasses.replace(
            config, name=self.import_path, dataset=_as_one_dataset(config.dataset)
        )
        try:
            self._check_dataset()
            if self._config.factor_names is None:
                self._config = dataclasses.replace(
                    self._config, factor_names=tuple(self._get_factor_names())
                )
            self._validate_config()
        except Exception:
            if previous is None:
                del self._config
            else:
                self._config = previous
            raise

    def _check_dataset(self) -> None:
        """Refuse an installed ``config.dataset`` this backend cannot read.

        Does nothing here; ``FactorKunQuant`` refuses a merged input in
        stream mode. It runs before ``_validate_config``, and a raise leaves
        the factor's previous config in place.
        """

    def _validate_config(self) -> None:
        """Refuse an installed config this factor cannot compute from.

        Does nothing here. A subclass with constraints across fields (its
        ``data_columns`` against its parameters, its ``factor_names``
        against what it can produce) overrides this, reads ``self.config``
        and raises ``ValueError``. It runs on every assignment, including
        the ones ``copy()`` and ``resample()`` make, and a raise leaves the
        factor's previous config in place.
        """

    @property
    def num_factors(self) -> int:
        """Number of columns this factor produces.

        Examples
        --------
        >>> factor.num_factors
        1
        """
        return len(self.get_factor_names())

    @property
    def import_path(self) -> str:
        """Dotted ``module.QualName`` path used to rebuild this class from a config.

        Examples
        --------
        >>> factor.import_path
        'quantlab.factor.momentum.Momentum'
        """
        return f"{self.__class__.__module__}.{self.__class__.__qualname__}"

    @property
    def class_name(self) -> str:
        """Bare class name, used in log and error messages.

        Examples
        --------
        >>> factor.class_name
        'Momentum'
        """
        return self.__class__.__name__

    def read(
        self,
        start: "str | datetime.date | pd.Timestamp",
        end: "str | datetime.date | pd.Timestamp",
    ) -> xr.Dataset:
        """Return the factor store from ``start`` to ``end``, both inclusive.

        The stored panel is opened lazily; the factor holds nothing
        afterwards. The range must lie inside the one recorded beside the
        store by ``build`` and ``extend`` (see ``store_range``). A resampled
        factor reads its own store when one has been built, otherwise it
        resamples the part of the source factor's store the range needs.

        Parameters
        ----------
        start, end : str, datetime.date or pd.Timestamp
            The range to return. A date-only ``end`` includes every bar of
            that day.

        Returns
        -------
        xr.Dataset
            The panel on ``(timestamp, symbol)``.

        Raises
        ------
        ValueError
            If ``start`` is after ``end``, the store has no recorded range,
            or the recorded range does not contain the requested one.

        Examples
        --------
        >>> factor.build("2024-01-01", "2024-03-31").store_range()
        ('2024-01-01', '2024-03-31')
        >>> dict(factor.read("2024-02-01", "2024-02-10").sizes)
        {'timestamp': 10, 'symbol': 8}
        >>> factor.read("2024-03-20", "2024-04-10")
        Traceback (most recent call last):
        ValueError: Momentum.read(): the store at ... covers 2024-01-01 to 2024-03-31, ...
        """
        first, last = check_range(start, end, f"{self.class_name}.read()")
        window = slice(as_label(start), as_label(end))
        if self.config.resample_freq is None or Path(self.store_path).exists():
            self._check_covers(self.store_path, start, end)
            data = XrBackend().read(self.store_path).data.sel(timestamp=window)
        else:
            self._check_covers(self.config.file_path, start, end)
            pad = resample_padding(self.config.resample_freq)
            source = XrBackend().read(self.config.file_path).data.sel(
                timestamp=slice(first - pad, last + pad)
            )
            data = self._resample_panel(source).sel(timestamp=window)
        return _on_panel_axes(data)

    @property
    def warmup_bars(self) -> int:
        """Bars of history ``compute`` reads before the requested start.

        Counted on the input dataset's own calendar, so weekends and
        holidays are skipped, not counted. It is ``config.warmup_bars``.

        Examples
        --------
        >>> factor.warmup_bars
        20
        """
        return self.config.warmup_bars

    def compute(
        self,
        start: "str | datetime.date | pd.Timestamp",
        end: "str | datetime.date | pd.Timestamp",
    ) -> xr.Dataset:
        """Compute the factor from ``start`` to ``end``, both inclusive.

        The inputs are requested from the dataset from ``warmup_bars`` bars
        before ``start`` up to ``end``, so rolling operators are warm on the
        first requested bar; the result holds only the requested range. A
        resampled factor is computed on its dataset's own bars and returned
        on its resampled bars. Neither this factor's config nor its
        dataset's config changes, and the factor holds nothing afterwards.

        If the dataset holds fewer than ``warmup_bars`` bars before
        ``start``, a ``UserWarning`` states the shortfall in bars and the
        computation starts from the first bar there is.

        Parameters
        ----------
        start, end : str, datetime.date or pd.Timestamp
            The range to return. A date-only ``end`` includes every bar of
            that day.

        Returns
        -------
        xr.Dataset
            The factor panel on ``(timestamp, symbol)``, in memory.

        Raises
        ------
        ValueError
            If ``start`` is after ``end``.

        Examples
        --------
        >>> panel = factor.compute("2024-02-01", "2024-02-10")  # warmup_bars=20
        >>> dict(panel.sizes)
        {'timestamp': 10, 'symbol': 8}

        With only two bars before ``start``, the call warns::

            UserWarning: Momentum.compute(): 20 warm-up bar(s) are needed
            before '2024-01-03' but the dataset holds only 2; the first
            bars are short by 18 bar(s) of warm-up.
        """
        inputs = self.config.dataset.panel(*self._input_range(start, end))
        panel = self._compute_panel(inputs)
        if self.config.resample_freq is not None:
            panel = self._resample_panel(panel)
        return _on_panel_axes(
            panel.sel(timestamp=slice(as_label(start), as_label(end)))
        )

    def _input_range(self, start, end, *, warn: bool = True) -> tuple:
        """Return the ``(start, end)`` of the dataset panel ``compute`` reads.

        That is ``start`` to ``end`` plus ``warmup_bars`` bars before
        ``start``, widened by the resample padding for a resampled factor.
        The backtester fingerprints the same range with ``warn=False``.

        Raises
        ------
        ValueError
            If ``start`` is after ``end``.
        """
        first, last = check_range(start, end, f"{self.class_name}.compute()")
        source_start, source_end = self._warm_start(first, start, warn), end
        if self.config.resample_freq is not None:
            pad = resample_padding(self.config.resample_freq)
            source_start = min(source_start, first - pad)
            source_end = last + pad
        return source_start, source_end

    def _warm_start(self, first: pd.Timestamp, start, warn: bool) -> pd.Timestamp:
        """Return the bar ``warmup_bars`` bars before ``first``, or the earliest.

        Warns with the shortfall in bars when the dataset holds fewer and
        ``warn`` is true.
        """
        dataset = self.config.dataset
        needed = self.warmup_bars
        try:
            return dataset.bar_before(first, needed)
        except InsufficientHistoryError as exc:
            if not warn:
                return dataset.bar_before(first, exc.available)
            warnings.warn(
                f"{self.class_name}.compute(): {needed} warm-up bar(s) are "
                f"needed before {start!r} but {dataset.class_name} holds only "
                f"{exc.available}; the first bars are short by "
                f"{needed - exc.available} bar(s) of warm-up.",
                UserWarning,
                stacklevel=4,
            )
            return dataset.bar_before(first, exc.available)

    def _compute_panel(self, inputs: xr.Dataset) -> xr.Dataset:
        """Return the factor panel computed over the whole ``inputs`` panel.

        ``inputs`` is a panel of the dataset, warm-up included; the result
        covers the same bars. Each backend overrides this.

        Raises
        ------
        NotImplementedError
            Unless a subclass overrides it.
        """
        raise NotImplementedError(
            f"{self.class_name} does not compute from a requested panel."
        )

    def build(
        self,
        start: "str | datetime.date | pd.Timestamp",
        end: "str | datetime.date | pd.Timestamp",
    ) -> Self:
        """Compute ``start`` to ``end`` and write it as the factor store.

        The store at ``store_path`` is replaced by ``compute(start, end)``
        and the range is recorded beside it, in
        ``<store_path>.range.json``, for ``store_range``, ``read`` and
        ``extend``. The factor holds nothing afterwards and no config
        changes.

        Parameters
        ----------
        start, end : str, datetime.date or pd.Timestamp
            The range the store covers.

        Returns
        -------
        Self
            ``self``, for chaining.

        Examples
        --------
        >>> factor.build("2024-01-01", "2024-03-31").store_range()
        ('2024-01-01', '2024-03-31')
        """
        panel = self.compute(start, end)
        with Timer(f"{self.class_name}: build"):
            self._drop_range(self.store_path)
            XrBackend().to_internal(panel).write(self.store_path, mode="w")
            self._record_range(self.store_path, start, end)
        return self

    def extend(self, end: "str | datetime.date | pd.Timestamp") -> Self:
        """Append the bars after the recorded range, up to ``end``.

        The bars are computed with ``compute``, so they are warmed from the
        dataset's history, then appended to the store; its timestamp, symbol
        and variable axes widen to fit, and cells created by widening are
        filled from ``_widen_fill_values()``.
        The recorded range then ends at ``end``.

        Parameters
        ----------
        end : str, datetime.date or pd.Timestamp
            The new end of the store's range.

        Returns
        -------
        Self
            ``self``, for chaining.

        Raises
        ------
        ValueError
            If the factor is resampled, the store has no recorded range, or
            the recorded range already reaches ``end``.

        Examples
        --------
        >>> factor.build("2024-01-01", "2024-03-31").store_range()
        ('2024-01-01', '2024-03-31')
        >>> factor.extend("2024-04-30").store_range()
        ('2024-01-01', '2024-04-30')
        """
        self._refuse_if_resampled("extend")
        recorded = self._stored_range(self.store_path)
        if recorded is None:
            raise ValueError(
                f"{self.class_name}.extend(): the store at {self.store_path} "
                f"has no recorded range; write it with build(start, end) "
                f"first."
            )
        recorded_start, recorded_end = recorded
        if last_moment(end) <= last_moment(recorded_end):
            raise ValueError(
                f"{self.class_name}.extend(): the store at {self.store_path} "
                f"already covers {recorded_start} to {recorded_end}; extend() "
                f"appends only bars after {recorded_end}, got end {end!r}."
            )
        after = last_moment(recorded_end) + pd.Timedelta(1, "ns")
        panel = self.compute(after, end)
        with Timer(f"{self.class_name}: extend"):
            if panel.sizes["timestamp"]:
                XrBackend().to_internal(panel).widen_and_append(
                    self.store_path, fill_values=self._widen_fill_values()
                )
            self._record_range(self.store_path, recorded_start, end)
        return self

    def store_range(self) -> tuple[str, str] | None:
        """Return the ``(start, end)`` the store was built for, if recorded.

        ``None`` when the store was not written by ``build``.

        Examples
        --------
        >>> factor.build("2024-01-01", "2024-03-31").store_range()
        ('2024-01-01', '2024-03-31')
        """
        return self._stored_range(self.store_path)

    def _check_covers(self, path: str, start, end) -> None:
        """Raise unless the range recorded for ``path`` contains the request."""
        recorded = self._stored_range(path)
        if recorded is None:
            raise ValueError(
                f"{self.class_name}.read(): the store at {path} has no "
                f"recorded range, so it cannot answer a date-range request; "
                f"write it with build(start, end)."
            )
        recorded_start, recorded_end = recorded
        if pd.Timestamp(start) < pd.Timestamp(recorded_start) or last_moment(
            end
        ) > last_moment(recorded_end):
            raise ValueError(
                f"{self.class_name}.read(): the store at {path} covers "
                f"{recorded_start} to {recorded_end}, which does not contain "
                f"{start} to {end}. Extend it with extend(end) or rebuild it "
                f"with build(start, end)."
            )

    def _range_path(self, store: str) -> Path:
        """Return the file recording the range of the store at ``store``."""
        return Path(f"{store}{self.RANGE_SUFFIX}")

    def _stored_range(self, store: str | None) -> tuple[str, str] | None:
        """Return the range recorded for ``store``, or ``None``."""
        if store is None or not self._range_path(store).is_file():
            return None
        recorded = json.loads(self._range_path(store).read_text())
        return recorded["start"], recorded["end"]

    def _record_range(self, store: str, start, end) -> None:
        """Record ``start`` and ``end`` as the range of the store at ``store``."""
        write_json_atomically(
            self._range_path(store),
            {"start": range_text(start), "end": range_text(end)},
        )

    def _drop_range(self, store: str | None) -> None:
        """Remove the range recorded for ``store``, if any."""
        if store is not None:
            self._range_path(store).unlink(missing_ok=True)

    @property
    def store_path(self) -> str | None:
        """Return the Zarr store this factor reads and writes.

        This is ``config.file_path`` for a factor that is not resampled. A
        resampled factor uses a store beside it with ``_resample_<freq>``
        added to the name, so the resampled panel never overwrites the
        source bars. ``None`` when ``config.file_path`` is ``None``.

        Examples
        --------
        >>> factor.store_path
        'data/factors/momentum.zarr'
        >>> factor.resample("1d", "last").store_path
        'data/factors/momentum_resample_1d.zarr'
        """
        return resample_store_path(
            self.config.file_path, self.config.resample_freq
        )

    def copy(self) -> Self:
        """Return a copy with its own config, dataset and empty backend.

        The config is deep-copied with its ``dataset`` replaced by
        ``dataset.copy()``, so the copy shares no mutable state with this
        factor, not even a ``kwargs`` dict.

        Examples
        --------
        >>> other = factor.copy()
        >>> other.config.dataset is factor.config.dataset
        False
        """
        other = copy.copy(self)
        other.data_backend = XrBackend()
        config = copy.deepcopy(dataclasses.replace(self.config, dataset=None))
        other.config = dataclasses.replace(
            config, dataset=self.config.dataset.copy()
        )
        return other

    def resample(
        self, freq: str, how: dict[str, str] | str
    ) -> Self:
        """Return a copy of this factor whose panel is resampled onto ``freq``.

        The factor is still computed on its dataset's own bars; only the
        computed panel is aggregated, so a minute-bar factor becomes a
        daily one without changing what it measures. The copy's config
        carries ``resample_freq=freq`` and ``resample_how=how``;
        ``compute``, ``read`` and ``build`` on it answer on the resampled
        bars. This factor and its dataset are not changed.

        A resampled factor cannot ``extend()`` its store or stream, and
        ``build()`` writes to ``store_path``, the resampled store beside the
        source. The bars are cut the way the dataset cuts them (see
        ``BaseDataset._resample_labels``).

        Parameters
        ----------
        freq : str
            A ``ResampleFrequency`` token, coarser than the dataset's bars.
        how : dict[str, str] or str
            One ``ResampleMethod`` for every factor variable, or a
            ``{variable: method}`` dict naming every one.

        Returns
        -------
        Self
            A new factor of the same class.

        Raises
        ------
        ValueError
            If ``freq`` or a method is not a known token. Whether ``freq`` is
            coarser than the bars and ``how`` names every variable is checked
            when a panel is requested.

        Examples
        --------
        >>> daily = factor.resample("1d", "last")      # factor on minute bars
        >>> factor.compute("2024-01-02", "2024-01-03").sizes["timestamp"]
        780
        >>> daily.compute("2024-01-02", "2024-01-03").sizes["timestamp"]
        2
        """
        other = self.copy()
        other.config = dataclasses.replace(
            other.config, resample_freq=freq, resample_how=how
        )
        return other

    def _refuse_if_resampled(self, method: str) -> None:
        """Raise if ``method`` is called on a resampled factor.

        Raises
        ------
        ValueError
            If ``config.resample_freq`` is set.
        """
        if self.config.resample_freq is not None:
            raise ValueError(
                f"{self.class_name}.{method}(): a resampled factor "
                f"(resample_freq={self.config.resample_freq!r}) is a view of "
                f"its source panel and does not support {method}. Compute "
                f"or update the source factor, then resample it."
            )

    def _resample_labels(self, timestamps: np.ndarray, freq: str) -> np.ndarray:
        """Return the bar each timestamp belongs to, as the dataset cuts bars."""
        return self.config.dataset._resample_labels(timestamps, freq)

    def _resample_panel(self, data: xr.Dataset) -> xr.Dataset:
        """Return ``data`` aggregated onto ``config.resample_freq`` bars.

        ``data`` is not changed; the grouping runs in a backend of its own.
        """
        freq = self.config.resample_freq
        timestamps = data["timestamp"].values
        assert_coarser(timestamps, freq, self.class_name)
        how = resolve_resample_how(
            self.config.resample_how, list(data.data_vars), self.class_name
        )
        labels = pd.Series(self._resample_labels(timestamps, freq), index=timestamps)
        with Timer(f"{self.__class__.__name__}: resample to {freq}"):
            return XrBackend().to_internal(data).resample(labels, how).data

    def _widen_fill_values(self) -> dict:
        """Return per-variable fill values for cells that widening creates.

        The default is empty, which leaves the choice to the backend.
        """
        return {}

    def _get_xarray_dataset(self) -> xr.Dataset:
        """Return the last streamed bar as an ``xarray.Dataset``."""
        return self.data_backend.get_xarray_dataset()  # type: ignore

    def _get_features(self, data: xr.Dataset) -> xr.Dataset:
        """Turn a factor panel into features; factor classes override this.

        Raises
        ------
        NotImplementedError
            Unless a subclass overrides it.
        """
        raise NotImplementedError

    def get_features(self, panel: xr.Dataset | None = None) -> xr.Dataset:
        """Return ``panel`` as model features.

        Parameters
        ----------
        panel : xr.Dataset, optional
            A panel returned by ``read(start, end)`` or
            ``compute(start, end)``. Omit it only in stream mode, where the
            bar the last ``cal_stream()`` computed is used.

        Examples
        --------
        >>> panel = factor.get_features(factor.compute("2024-02-01", "2024-02-10"))
        >>> list(panel.data_vars), dict(panel.sizes)
        (['momentum_20'], {'timestamp': 10, 'symbol': 8})
        """
        if panel is None:
            panel = self._get_xarray_dataset()
        return self._get_features(panel)

    def _get_labels(self, data: xr.Dataset) -> xr.Dataset:
        """Turn a factor panel into labels; label classes override this.

        Raises
        ------
        NotImplementedError
            Unless a subclass overrides it.
        """
        raise NotImplementedError

    def get_labels(self, panel: xr.Dataset | None = None) -> xr.Dataset:
        """Return ``panel`` as model labels.

        Parameters
        ----------
        panel : xr.Dataset, optional
            A panel returned by ``read(start, end)`` or
            ``compute(start, end)``. Omit it only in stream mode, where the
            bar the last ``cal_stream()`` computed is used.

        Examples
        --------
        >>> panel = label.get_labels(label.compute("2024-01-01", "2024-01-21"))
        >>> list(panel.data_vars), dict(panel.sizes)     # a label class
        (['ret_1'], {'timestamp': 21, 'symbol': 16})
        """
        if panel is None:
            panel = self._get_xarray_dataset()
        return self._get_labels(panel)

    def get_factor_names(self) -> tuple[str, ...]:
        """Return the names of the columns this factor produces.

        Examples
        --------
        >>> factor.get_factor_names()
        ('momentum_20',)
        """
        return self.config.factor_names

    def get_config(self) -> dict:
        """Return a serializable dict describing this factor and its dataset.

        The dataset's own config is nested under ``"dataset"``;
        ``quantlab.utils.module.load_factor_from_config`` rebuilds the factor
        from the result.

        Examples
        --------
        >>> cfg = factor.get_config()
        >>> cfg["name"]
        'quantlab.factor.momentum.Momentum'
        >>> cfg["kwargs"], cfg["dataset"]["frequency"]
        ({'n': 20}, '1d')
        """
        ds_config = self.config.dataset.get_config()
        cfg = self.config.to_dict()
        cfg["dataset"] = ds_config  # type: ignore
        return cfg  # type: ignore

    @abstractmethod
    def _get_factor_names(self) -> tuple[str, ...]:
        """Return the names of every column this factor can produce."""
        ...

    def analyze(
        self,
        start: "str | datetime.date | pd.Timestamp",
        end: "str | datetime.date | pd.Timestamp",
        factor_names: list[str] | None = None,
        frets: list["Factor"] | None = None,
        output_dir: str | None = None,
        quantiles: int = 5,
        data_strategy: Literal["cal", "read"] = "cal",
    ) -> "FactorAnalysis":
        """Report how well this factor predicts forward returns, alphalens style.

        Every analyzed factor variable is paired with every variable of every
        forward-return label (*fret*). For each pair the factor and fret
        panels are joined on their common timestamps and symbols, then
        ``quantlab.analysis.factor_report.FactorAnalyzer`` computes the
        information analysis (per-period Spearman IC, its mean, std, IR,
        t-statistic, p-value, skew, kurtosis and monthly means), the returns
        analysis (mean forward return per factor quantile, top-minus-bottom
        spread, cumulative returns) and the turnover analysis (quantile
        turnover, lag-1 factor rank autocorrelation), and draws one
        composite matplotlib figure.

        The factor and every fret are asked for their panels from ``start``
        to ``end``: computed from their inputs with ``compute(start, end)``
        under ``data_strategy="cal"``, read from their stores with
        ``read(start, end)`` under ``"read"``.

        Parameters
        ----------
        start, end : str, datetime.date or pd.Timestamp
            The range to analyze, both inclusive.
        factor_names : list of str, optional
            Variables of this factor to analyze. All of
            ``get_factor_names()`` when None.
        frets : list of Factor
            Forward-return labels, for example ``quantlab.label.fret.Return``;
            ``get_labels()`` of each gives the forward returns. Required.
        output_dir : str, optional
            When given, the directory is created and ``summary.json``,
            ``summary.csv``, ``ic.csv``, ``monthly_ic.csv``,
            ``quantile_returns.csv``, ``turnover.csv``, one
            ``<factor>__<fret>.png`` per pair and ``config.json`` are written
            there, plus ``factor_correlation.csv``,
            ``factor_correlation_pairs.csv``, ``factor_clusters.csv`` and
            ``factor_correlation.png`` when two or more factor variables are
            analyzed. ``config.json`` holds ``{"factor": ..., "frets": [...]}``,
            each rebuildable with ``load_factor_from_config``. When None,
            nothing is written.
        quantiles : int, default 5
            Number of equal-count factor buckets per timestamp.
        data_strategy : {"cal", "read"}, default "cal"
            Whether the panels are computed or read from the stores
            ``build`` wrote, for the factor and every fret alike.

        Returns
        -------
        FactorAnalysis
            ``pairs`` (metrics per ``"<factor>__<fret>"``), ``figures``
            (matplotlib figures, same keys) and tidy tables through
            ``summary_table()``, ``ic_table()``, ``quantile_returns_table()``,
            ``turnover_table()`` and ``monthly_ic_table()``; with two or more
            factor variables also ``correlation``, their
            ``FactorCorrelation``.

        Raises
        ------
        ValueError
            If ``frets`` is empty, ``data_strategy`` is neither ``"cal"``
            nor ``"read"``, a factor name is unknown, a fret's most common
            bar spacing differs from the factor's, or the panels share no
            cells.

        Examples
        --------
        ``factor`` is a ``Momentum`` (``momentum_5``) over eight symbols, and
        ``fwd`` a ``quantlab.label.fret.Return`` with ``n_forward_periods=1``
        on the same dataset:

        >>> result = factor.analyze(
        ...     "2024-02-01", "2024-02-29", frets=[fwd], quantiles=4,
        ...     output_dir="data/analysis/momentum",
        ... )
        >>> list(result.pairs)
        ['momentum_5__ret_1']
        >>> cols = ["factor", "fret", "ic_mean", "ic_t_stat", "mean_spread"]
        >>> result.summary_table()[cols].round(4)
               factor   fret  ic_mean  ic_t_stat  mean_spread
        0  momentum_5  ret_1  -0.0494    -0.7667       -0.002
        >>> sorted(os.listdir("data/analysis/momentum"))
        ['config.json', 'ic.csv', 'momentum_5__ret_1.png', 'monthly_ic.csv', 'quantile_returns.csv', 'summary.csv', 'summary.json', 'turnover.csv']
        """
        # Imported here so the base layer does not depend on the analysis
        # layer at import time; the report is an optional terminal step.
        from quantlab.analysis.factor_report import FactorAnalyzer

        if data_strategy not in ("cal", "read"):
            raise ValueError(
                f"{self.class_name}.analyze(): data_strategy must be \"cal\" "
                f"or \"read\", got {data_strategy!r}."
            )
        frets = frets or []
        if not frets:
            raise ValueError(
                "analyze needs at least one forward-return label in `frets`"
            )

        def request(obj: "Factor") -> xr.Dataset:
            if data_strategy == "read":
                return obj.read(start, end)
            return obj.compute(start, end)

        return FactorAnalyzer(quantiles=quantiles).run(
            self,
            frets,
            features=self.get_features(request(self)),
            labels=[fret.get_labels(request(fret)) for fret in frets],
            factor_names=factor_names,
            output_dir=output_dir,
        )


class FactorKunQuant(Factor):
    """Factor backend that compiles a KunQuant op graph to native code.

    A subclass describes its factor as a KunQuant graph in
    ``_get_factor_func``: ``Input`` nodes named after ``config.data_columns``,
    operator nodes, and one ``Output`` per factor name. The same graph is
    compiled on demand in two layouts, ``TS`` for ``compute()`` (a whole date
    range in one call) and ``STREAM`` for ``cal_stream()`` (one bar at a time), so a
    factor validated in a backtest runs unchanged on live data.
    ``config.mode`` says which of the two the object is used in.

    Compilation needs a working C++ compiler and dominates run time on small
    panels; pinning ``config.factor_names`` to the columns you need keeps the
    compiled graph small. In batch mode the number of symbols must be a
    multiple of the SIMD block width KunQuant uses on the host, that is, the
    number of values the CPU processes in one vector instruction.

    Parameters
    ----------
    config : FactorConfig
        The factor config, including ``mode`` (``"batch"`` or
        ``"stream"``), ``data_columns`` and ``njobs``.

    Examples
    --------
    A factor measuring how far the close is above its 5-bar average::

        class MaDeviation(FactorKunQuant):
            def _get_factor_names(self):
                return ("ma_dev_5",)

            def _get_factor_func(self):
                builder = Builder()
                with builder:
                    close = Input("close")
                    dev = op.Div(close, op.WindowedAvg(close, 5))
                    Output(op.SubConst(dev, 1.0), "ma_dev_5")
                return Function(builder.ops)
    """

    #: The config class ``load_factor_from_config`` rebuilds this factor with.
    config_cls = FactorConfig

    def __init__(self, config: FactorConfig):
        """Initialize the factor; see the class docstring for parameters.

        No graph is compiled and no stream context exists yet.
        """
        super().__init__(config)
        self._stream_context: kr.StreamContext = None
        self._lib = None
        self._buffer_name_to_id = dict()

    def copy(self) -> Self:
        """Return a copy without the compiled library or stream state.

        See ``Factor.copy``. The compiled batch library, the stream context
        and its buffer handles belong to this object's own graph and are
        not carried over; the copy compiles again on its first ``compute()``.

        Examples
        --------
        >>> factor.copy()._lib is None
        True
        """
        other = super().copy()
        other._stream_context = None
        other._lib = None
        other._buffer_name_to_id = dict()
        return other

    def _check_dataset(self) -> None:
        """Refuse a merged input in stream mode.

        A stream is fed one bar at a time for a fixed symbol list, which a
        merge of several stores does not have.

        Raises
        ------
        ValueError
            If ``config.mode`` is ``"stream"`` and ``config.dataset`` is a
            ``MergedDataset``.
        """
        from quantlab.dataset.merged import MergedDataset

        if self.config.mode == "stream" and isinstance(
            self.config.dataset, MergedDataset
        ):
            raise ValueError(
                f"{self.class_name}: stream mode takes one dataset, got a "
                f"merge of {len(self.config.dataset.datasets)}. A stream is "
                f"fed one bar at a time for a fixed symbol list; compute a "
                f"merged input in batch mode."
            )

    def compute(
        self,
        start: "str | datetime.date | pd.Timestamp",
        end: "str | datetime.date | pd.Timestamp",
    ) -> xr.Dataset:
        """Compute ``start`` to ``end`` in batch mode; see ``Factor.compute``.

        Raises
        ------
        ValueError
            If ``config.mode`` is not ``"batch"``, or ``start`` is after
            ``end``.

        Examples
        --------
        >>> dict(factor.compute("2024-02-01", "2024-02-10").sizes)
        {'timestamp': 10, 'symbol': 16}
        """
        if self.config.mode != "batch":
            raise ValueError(
                f"{self.class_name}.compute(): a date-range computation runs "
                f"the batch graph, but config.mode is {self.config.mode!r}."
            )
        return super().compute(start, end)

    @property
    def num_symbols(self) -> int:
        """Number of symbols the streaming graph runs over.

        This is the symbol list pinned on the dataset config, since a
        stream loads no panel.

        Raises
        ------
        ValueError
            If ``config.mode`` is not ``"stream"``.

        Examples
        --------
        >>> factor.num_symbols        # stream config pinning 16 symbols
        16
        """
        return len(self.symbols)

    @property
    def symbols(self) -> list[str]:
        """Symbols the streaming graph runs over, in axis order.

        This is the symbol list pinned on the dataset config. In batch mode
        the symbols are those of the requested panel instead.

        Raises
        ------
        ValueError
            If ``config.mode`` is not ``"stream"``.

        Examples
        --------
        >>> factor.symbols[:3]
        ['AAPL', 'MSFT', 'NVDA']
        """
        if self.config.mode != "stream":
            raise ValueError(
                f"{self.class_name}.symbols: only a stream-mode factor has a "
                f"fixed symbol list; a batch computation runs over the "
                f"symbols of the requested panel. config.mode is "
                f"{self.config.mode!r}."
            )
        return list(self.config.dataset.config.symbols)

    def init_stream(self) -> Self:
        """Compile the graph in the streaming layout and bind its buffers.

        Creates a ``StreamContext`` sized to ``num_symbols`` and caches a
        buffer handle for every input column and every factor name, so
        ``cal_stream`` does not look handles up by name on the hot path.
        Every name in ``config.data_columns`` must be consumed by a reachable
        ``Output``: KunQuant prunes unused inputs and the handle lookup for a
        pruned one fails.

        Returns
        -------
        Self
            ``self``, for chaining.

        Examples
        --------
        >>> factor.init_stream() is factor    # config.mode == "stream"
        True
        >>> sorted(factor._buffer_name_to_id)  # one input, three outputs
        ['adjClose', 'ma_close', 'ma_rank', 'rank_close']
        """
        self._refuse_if_resampled("init_stream")
        with Timer(f"{self.__class__.__name__}: init stream"):
            lib = self._make_stream()
            modu = lib.getModule(f"{self.__class__.__name__}_stream")  # type: ignore

            executor = kr.createMultiThreadExecutor(self.config.njobs)
            stream = kr.StreamContext(executor, modu, self.num_symbols)

            buffer_name_to_id = {}
            for name in self.config.data_columns:
                buffer_name_to_id[name] = stream.queryBufferHandle(name)
            for name in self.config.factor_names:
                buffer_name_to_id[name] = stream.queryBufferHandle(name)

            self._stream_context = stream
            self._buffer_name_to_id = buffer_name_to_id
            return self

    def _to_xarray_dataset(
        self,
        raw_factor: dict[str, np.ndarray],
        timestamps: np.ndarray,
        symbols: np.ndarray,
    ):
        """Wrap one streamed bar's arrays in an ``xarray.Dataset`` and hold it.

        Parameters
        ----------
        raw_factor : dict[str, np.ndarray]
            Factor name to a ``[num_times, num_symbols]`` array.
        timestamps : np.ndarray
            Coordinate values for the time axis.
        symbols : np.ndarray
            Coordinate values for the symbol axis.

        Returns
        -------
        Factor
            ``self``, for chaining.
        """
        self.data_backend.to_internal(
            self._output_panel(raw_factor, timestamps, symbols)
        )
        return self

    @staticmethod
    def _output_panel(
        raw_factor: dict[str, np.ndarray],
        timestamps: np.ndarray,
        symbols: np.ndarray,
    ) -> xr.Dataset:
        """Wrap raw ``[time, symbol]`` arrays in an ``xarray.Dataset``."""
        return xr.Dataset(
            {k: (["timestamp", "symbol"], v) for k, v in raw_factor.items()},
            coords={
                "timestamp": timestamps,
                "symbol": symbols,
            },
        )

    @abstractmethod
    def _get_factor_func(self) -> Function:
        """Build and return the KunQuant graph that computes this factor.

        Input names must match ``config.data_columns``; output names are the
        factor names.
        """
        ...

    def _kunquant_inputs(
        self, inputs: xr.Dataset
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """Return ``(arrays, symbols, timestamps)`` the graph runs on.

        The default exports ``config.data_columns`` of ``inputs`` through
        the dataset's ``to_kunquant``. A factor whose graph takes inputs
        from elsewhere as well overrides this and adds them.
        """
        return self.config.dataset.to_kunquant(
            data_columns=self.config.data_columns, panel=inputs
        )

    def _compute_panel(self, inputs: xr.Dataset) -> xr.Dataset:
        """Run the compiled graph over every bar of ``inputs``.

        The graph is compiled if no library is cached, run from bar 0 on an
        executor of ``config.njobs`` threads, and dropped afterwards, so each
        call compiles again.
        """
        input_dict, symbols, timestamp = self._kunquant_inputs(inputs)
        # Every input is laid out [time, symbol]; any one gives the time count.
        num_time = next(iter(input_dict.values())).shape[0]
        input_dict = self._pad_symbols(input_dict, len(symbols))

        if self._lib is None:
            self._lib = self._make()

        modu = self._lib.getModule(f"{self.__class__.__name__}")  # type: ignore

        executor = kr.createMultiThreadExecutor(self.config.njobs)
        with Timer(f" {self.__class__.__name__}: cal"):
            out_dict = kr.runGraph(executor, modu, input_dict, 0, num_time)

        self._lib = None

        return self._output_panel(
            self._cut_symbols(out_dict, len(symbols)), timestamp, symbols
        )

    def cal_stream(
        self, data: dict[str, np.ndarray], timestamp: int, symbols: list[str]
    ) -> Self:
        """Advance the streaming graph by one bar and hold that bar's outputs.

        The stream is initialized on first use.

        Parameters
        ----------
        data : dict[str, np.ndarray]
            Column name to a 1-D array of length ``num_symbols`` for
            every name in ``config.data_columns``.
        timestamp : int
            The bar's timestamp, used as the single time
            coordinate.
        symbols : list[str]
            Symbol coordinate values, in the order the arrays are
            laid out.

        Returns
        -------
        Self
            ``self``, holding a ``(1, num_symbols)`` panel for this bar.

        Examples
        --------
        >>> for step in range(3):                  # replay three bars
        ...     bar = {"adjClose": adj_close[step]}  # float32, per symbol
        ...     row = factor.cal_stream(bar, step, symbols).get_features()
        >>> row.sizes
        Frozen({'timestamp': 1, 'symbol': 16})
        >>> list(row.data_vars)
        ['rank_close', 'ma_close', 'ma_rank']
        """
        self._refuse_if_resampled("cal_stream")
        if self._stream_context is None:
            self.init_stream()

        for name in self.config.data_columns:
            self._stream_context.pushData(
                self._buffer_name_to_id[name], data[name]
            )

        self._stream_context.run()

        out_dict = {}
        for factor in self.config.factor_names:
            alpha = self._stream_context.getCurrentBuffer(
                self._buffer_name_to_id[factor]
            )[:]
            out_dict[factor] = np.expand_dims(alpha, axis=0)

        self._to_xarray_dataset(
            out_dict, np.array([timestamp]), np.array(symbols)
        )

        return self

    #: Symbol count a batch run is padded to a multiple of on macOS. KunQuant's
    #: compiled loops process symbols in fixed-size SIMD blocks and cannot
    #: handle a remainder: four values per block on Apple silicon, eight with
    #: AVX2 on an Intel Mac; eight covers both.
    SYMBOL_BLOCK_DARWIN = 8

    @staticmethod
    def _symbol_padding(num_symbols: int) -> int:
        """Return how many all-NaN dummy symbols a batch run appends.

        Only macOS pads (``sys.platform == "darwin"``), to a multiple of
        ``SYMBOL_BLOCK_DARWIN``; elsewhere the panel is passed as it is.

        Examples
        --------
        >>> FactorKunQuant._symbol_padding(5)   # on macOS
        3
        >>> FactorKunQuant._symbol_padding(16)
        0
        """
        if sys.platform != "darwin":
            return 0
        return (-num_symbols) % FactorKunQuant.SYMBOL_BLOCK_DARWIN

    @classmethod
    def _pad_symbols(
        cls, inputs: dict[str, np.ndarray], num_symbols: int
    ) -> dict[str, np.ndarray]:
        """Append ``_symbol_padding`` all-NaN columns to every ``[time, symbol]`` input.

        NaN symbols never enter a cross-sectional statistic, and
        ``_cut_symbols`` removes their outputs again.
        """
        padding = cls._symbol_padding(num_symbols)
        if not padding:
            return inputs
        return {
            name: np.pad(
                values, ((0, 0), (0, padding)), mode="constant", constant_values=np.nan
            )
            for name, values in inputs.items()
        }

    @staticmethod
    def _cut_symbols(
        outputs: dict[str, np.ndarray], num_symbols: int
    ) -> dict[str, np.ndarray]:
        """Drop the padded columns from every ``[time, symbol]`` output."""
        return {name: values[:, :num_symbols] for name, values in outputs.items()}

    def _make(self):
        """Compile the graph for batch execution with the ``TS`` layout."""
        with Timer(f" {self.__class__.__name__}: make"):
            return cfake.compileit(
                [
                    (
                        f"{self.__class__.__name__}",
                        self._get_factor_func(),
                        KunCompilerConfig(
                            input_layout="TS",
                            output_layout="TS",
                        ),
                    )
                ],
                f"{self.__class__.__name__}",
                cfake.CppCompilerConfig(),
            )

    def _make_stream(self):
        """Compile the graph for streaming execution with the ``STREAM`` layout."""
        with Timer(f"{self.__class__.__name__}: make stream"):
            return cfake.compileit(
                [
                    (
                        f"{self.__class__.__name__}_stream",
                        self._get_factor_func(),
                        KunCompilerConfig(
                            partition_factor=8,
                            input_layout="STREAM",
                            output_layout="STREAM",
                            options={"opt_reduce": False, "fast_log": True},
                        ),
                    )
                ],
                f"{self.__class__.__name__}_stream",
                cfake.CppCompilerConfig(),
            )


class FactorPolars(Factor):
    """Batch-only factor backend whose factor logic is a Polars expression chain.

    A subclass overrides ``_get_factor_lazyframe``, which receives the dataset
    as a ``LazyFrame`` and returns a lazy frame carrying only ``timestamp``,
    ``symbol`` and the factor columns. Nothing is materialized until ``compute()``
    collects it and converts the result to an ``xarray.Dataset``. Factor
    names are not declared: they are read from the schema of the returned
    frame, so they are known at construction time (which reads a few rows
    from the store) and always match what the expression chain produces.

    Column names are whatever the underlying store holds; unlike the KunQuant
    path, no per-market renaming is applied. A merged input is the exception:
    a merge renames every input to the shared names (``close``, not
    ``Close``) before the factor sees it.

    Parameters
    ----------
    config : PolarsFactorConfig
        The factor config.

    Examples
    --------
    A factor comparing each bar's volume with its 20-bar average::

        class RelativeVolume(FactorPolars):
            def _get_factor_lazyframe(self, lf):
                volume = pl.col("Volume")
                return (
                    lf.sort(["symbol", "timestamp"])
                    .with_columns(
                        (volume / volume.rolling_mean(20).over("symbol") - 1.0)
                        .alias("rel_volume_20")
                    )
                    .select(["timestamp", "symbol", "rel_volume_20"])
                )
    """

    #: The config class ``load_factor_from_config`` rebuilds this factor with.
    config_cls = PolarsFactorConfig

    #: Index columns, never reported as factor names.
    _INDEX_COLUMNS = ("timestamp", "symbol")

    #: Rows read from the store to derive the output schema at construction.
    _SCHEMA_PROBE_ROWS = 8

    def __init__(self, config: PolarsFactorConfig):
        """Initialize the factor; see the class docstring for parameters.

        The factor names are derived at once by reading a few rows of the
        dataset store.
        """
        super().__init__(config)

    def _get_factor_names(self) -> tuple[str, ...]:
        """Derive the factor names from the schema the expression chain yields.

        A few rows are read from the dataset store so the probe carries real
        dtypes; the index columns are excluded from the result.
        """
        probe = self.config.dataset.head(self._SCHEMA_PROBE_ROWS)
        factor_lf = self._get_factor_lazyframe(probe)
        return tuple(
            name
            for name in factor_lf.collect_schema().names()
            if name not in self._INDEX_COLUMNS
        )

    @abstractmethod
    def _get_factor_lazyframe(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """Return the factor as a lazy frame; the one method a subclass writes.

        Parameters
        ----------
        lf : pl.LazyFrame
            The dataset as a ``LazyFrame`` with the store's own column
            names.

        Returns
        -------
        pl.LazyFrame
            A lazy frame with exactly ``timestamp``, ``symbol`` and the factor
            columns. Do not call ``collect`` here.
        """
        ...

    def _compute_panel(self, inputs: xr.Dataset) -> xr.Dataset:
        """Collect the expression chain over ``inputs`` as a long ``LazyFrame``."""
        lf = XrBackend().to_internal(inputs).get_lazyframe()
        return self._collect(self._get_factor_lazyframe(lf))

    def _collect(self, factor_lf: pl.LazyFrame) -> xr.Dataset:
        """Collect ``factor_lf`` into a ``(timestamp, symbol)`` panel."""
        with Timer(f"{self.__class__.__name__}: cal"):
            frame = factor_lf.collect().to_pandas()
            frame = frame.set_index(list(self._INDEX_COLUMNS))
            return xr.Dataset.from_dataframe(frame)
