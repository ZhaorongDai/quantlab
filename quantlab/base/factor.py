"""Abstract factor layer and its two computation backends.

A *factor* turns market data into engineered features, for example a
20-day momentum or a moving-average deviation. It reads the *panel* held by
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
history is the *warm-up*; a factor asks its dataset for ``config.window``
extra calendar days and trims them off again afterwards.
"""

import copy
import dataclasses
from abc import ABC, abstractmethod
from pathlib import Path

from typing import Literal, Self

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
from quantlab.enums.constant import Date
from quantlab.utils.resample import (
    assert_coarser,
    resample_store_path,
    resolve_resample_how,
    validate_resample_config,
)
from quantlab.utils.timer import Timer


class Factor(ABC):
    """Backend-agnostic base class for factors and labels.

    A ``Factor`` holds a config whose ``dataset`` attribute is the dataset it
    reads from. It computes its output with ``cal()``, persists it with
    ``save()`` or ``update()``, reads it back with ``read()``, and hands the
    model layer an ``xarray.Dataset`` through ``get_features()`` or
    ``get_labels()``. Subclasses implement ``cal`` and ``_get_factor_names``
    and override ``_get_features`` and/or ``_get_labels`` for the half they
    support.

    Assigning ``config`` runs the property setter, which fills in default
    dates, resolves ``factor_names`` when they are not pinned, and moves the
    dataset's start date ``config.window`` days earlier so rolling windows are
    warm on the first requested bar. Factor names are therefore known as soon
    as the object is constructed, before anything is computed.

    Parameters
    ----------
    config : BaseFactorConfig
        The factor config. Its ``dataset`` field is the dataset the factor
        reads from. The object normalizes the config in place.

    Attributes
    ----------
    data_backend : XrBackend
        Holds the computed or loaded factor panel.

    Examples
    --------
    >>> factor = MyFactor(config)
    >>> factor.get_factor_names()          # known before cal()
    >>> panel = factor.cal().save(mode="w").get_features()
    """

    def __init__(self, config: BaseFactorConfig):
        """Initialize the factor; see the class docstring for parameters."""
        # The config setter runs before `self.data_backend` exists, so nothing
        # the setter reaches may use the storage backend.
        self.config = config
        self.data_backend = XrBackend()

    def __repr__(self) -> str:
        """Return the class name and its config."""
        return f"{self.__class__.__name__}(config={self.config})"

    def _auto_filter(self):
        """Narrow the held panel to the configured dates and, if set, symbols."""
        self.data_backend.filter_by_date(
            col="timestamp",
            start_date=self.config.start_date,
            end_date=self.config.end_date,
        )
        if self.config.symbols is not None:
            self.data_backend.filter_by_symbol("symbol", self.config.symbols)

    @property
    def config(self) -> BaseFactorConfig:
        """The factor's config; assigning it normalizes the config in place.

        Examples
        --------
        >>> factor.config is config
        True
        >>> factor.config.name
        'quantlab.factor.momentum.Momentum'
        """
        return self._config

    @config.setter
    def config(self, config: BaseFactorConfig):
        """Take ownership of ``config`` and normalize it in place.

        In order: ``name`` is set to this class's import path, fields the
        dataset declares unusable are refused, missing dates default to the
        open-ended ``Date`` bounds, ``factor_names`` are resolved if unset,
        and the dataset's date range is widened to cover the warm-up window.
        Nothing here touches the storage backend, which does not exist yet
        when ``__init__`` assigns the config.

        Parameters
        ----------
        config : BaseFactorConfig
            The factor config to install.

        Examples
        --------
        >>> factor.config = PolarsFactorConfig(
        ...     window=5, dataset=dataset, kwargs={"n": 5},
        ...     start_date="2024-02-01", file_path="momentum_5.zarr",
        ... )
        >>> factor.get_factor_names()
        ('momentum_5',)
        >>> factor.config.dataset.config.start_date   # 5 days of warm-up
        '2024-01-27'
        """
        self._config = config
        self._config.name = self.import_path

        # Refuse before names are resolved (which may probe the store) and
        # before anything is filtered.
        self._reject_declared_config_fields()

        if self._config.start_date is None:
            self._config.start_date = Date.START_DATE
        if self._config.end_date is None:
            self._config.end_date = Date.END_DATE

        validate_resample_config(
            self._config.resample_freq, self._config.resample_how, self.class_name
        )

        self._maybe_resolve_factor_names()

        self._reset_dataset_config()

    def _reject_declared_config_fields(self) -> None:
        """Refuse config fields the dataset declares unusable.

        A dataset may publish ``REJECTED_FACTOR_CONFIG_FIELDS``, a mapping
        from a config field name to the reason it cannot select on that
        dataset's panel (for example a ticker list against an integer
        identifier axis). The check runs at assignment time, before any store
        is probed, so the error names the offending field instead of
        surfacing later as a ``KeyError`` deep inside a ``.sel`` call. Only
        the mapping is read; this module names no dataset class.

        Raises
        ------
        ValueError
            If any declared field is set on the config.
        """
        dataset = getattr(self._config, "dataset", None)
        rejected = getattr(dataset, "REJECTED_FACTOR_CONFIG_FIELDS", None) or {}
        for name, reason in rejected.items():
            value = getattr(self._config, name, None)
            if value is None:
                continue
            raise ValueError(
                f"{self.class_name}: config.{name} is not selectable on a "
                f"{type(dataset).__name__} panel; got {value!r}. {reason}"
            )

    def _maybe_resolve_factor_names(self) -> None:
        """Fill ``config.factor_names`` from ``_get_factor_names()`` when unset."""
        if self._config.factor_names is None:
            self._config.factor_names = self._get_factor_names()

    def _reset_dataset_config(self):
        """Point the dataset at the factor's window plus warm-up history.

        The dataset's start date is moved ``config.window`` calendar days
        before the factor's start date so rolling operators are warm on the
        first requested bar; ``read()`` and ``save()`` narrow the result back
        through ``_auto_filter``. The symbol axis is left alone: symbols with
        no data show up as NaN columns rather than missing ones.
        """
        start_date = pd.to_datetime(self._config.start_date)
        start_date = start_date - pd.DateOffset(days=self._config.window)
        self._config.dataset.config.start_date = start_date.strftime("%Y-%m-%d")
        self._config.dataset.config.end_date = self._config.end_date

    @property
    def num_symbols(self) -> int:
        """Number of symbols on the dataset's symbol axis.

        The dataset must already be loaded; ``cal()`` loads it.

        Examples
        --------
        >>> factor.cal().num_symbols
        8
        """
        return self.config.dataset.num_symbols

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
    def symbols(self) -> list[str]:
        """Symbols on the dataset's symbol axis.

        The dataset must already be loaded; ``cal()`` loads it.

        Examples
        --------
        >>> factor.cal().symbols[:3]
        ['S0USDT', 'S1USDT', 'S2USDT']
        """
        return self.config.dataset.symbols

    @property
    def class_name(self) -> str:
        """Bare class name, used in log and error messages.

        Examples
        --------
        >>> factor.class_name
        'Momentum'
        """
        return self.__class__.__name__

    def read(self, overwrite: bool = False) -> Self:
        """Open the factor store and narrow it to the configured window.

        Parameters
        ----------
        overwrite : bool, default False
            Re-open the store even if the backend already holds data. By
            default the cached panel is reused and only narrowed. Pass
            ``True`` after changing ``config.start_date`` or
            ``config.end_date``, because the cached panel was cut to the
            old dates.

        Returns
        -------
        Self
            ``self``, for chaining.

        Examples
        --------
        >>> # config dates 2024-02-01 to 2024-02-10, store holds 60 bars
        >>> factor.read().get_features().sizes
        Frozen({'timestamp': 10, 'symbol': 8})
        >>> factor.config.end_date = "2024-02-20"
        >>> factor.read(overwrite=True).get_features().sizes
        Frozen({'timestamp': 20, 'symbol': 8})

        A resampled factor reads its own store when one has been saved, and
        otherwise reads the source store and resamples it:

        >>> daily = factor.resample("1d", "last")
        >>> daily.read().get_features().sizes
        Frozen({'timestamp': 10, 'symbol': 8})
        """
        if self.config.resample_freq is None:
            self.data_backend.read(self.config.file_path, overwrite=overwrite)
            self._auto_filter()
            return self

        if Path(self.store_path).exists():
            self.data_backend.read(self.store_path, overwrite=overwrite)
            self._auto_filter()
            return self

        fresh = overwrite or not self._holds_data()
        self.data_backend.read(self.config.file_path, overwrite=overwrite)
        self._auto_filter()
        if fresh:
            self._apply_resample()
        return self

    def save(self, mode: Literal["a", "w"] = "a", **kwargs) -> Self:
        """Write the held panel to ``store_path`` as a Zarr store.

        ``store_path`` is ``config.file_path``, or the resampled store beside
        it when the factor is resampled.

        Parameters
        ----------
        mode : {"a", "w"}, default "a"
            ``"a"`` overwrites variables in an existing store and fails if
            that store has a different time or symbol axis. ``"w"``
            replaces the store. To extend a store with a later date range,
            use ``update()`` instead.
        **kwargs
            Passed through to the backend's ``write``.

        Returns
        -------
        Self
            ``self``, for chaining.

        Raises
        ------
        ValueError
            If ``mode="a"`` meets a store whose dimension sizes
            differ from the panel being written.

        Examples
        --------
        >>> factor.cal().save(mode="w")    # replace the store
        >>> sorted(p.name for p in Path(factor.config.file_path).iterdir())
        ['momentum_20', 'symbol', 'timestamp', 'zarr.json']
        >>> factor.save()                  # same axes: rewrite in place
        """
        with Timer(f"{self.__class__.__name__}: save"):
            self._auto_filter()
            try:
                self.data_backend.write(
                    self.store_path,
                    mode=mode,
                    **kwargs,
                )
            except ValueError as exc:
                if "already exists with different dimension sizes" not in str(
                    exc
                ):
                    raise
                raise ValueError(
                    f'{self.class_name}.save(mode="a"): cannot write this '
                    f"date range into the existing store at "
                    f'{self.store_path}. In zarr, mode "a" means '
                    f'"overwrite variables in an existing store", not "append '
                    f'along time", so a date range of a different size is '
                    f'rejected. Use save(mode="w") to replace the store, or '
                    f"delete it first. To extend the store with a later date "
                    f"range, call update() instead: it widens the timestamp, "
                    f"symbol and variable axes to fit and applies the same "
                    f"safety checks as XrBackend.append(). "
                    f"Original error: {exc}"
                ) from exc
            return self

    def update(self, **kwargs) -> Self:
        """Append the held panel to the store at ``config.file_path``.

        The store's timestamp, symbol and variable axes are widened to the
        union of what it holds and what the panel carries, then the panel is
        appended. Cells created by widening are filled from
        ``_widen_fill_values()``.

        Parameters
        ----------
        **kwargs
            Passed through to the backend's ``widen_and_append``.

        Returns
        -------
        Self
            ``self``, for chaining.

        Examples
        --------
        >>> factor.read().get_features().sizes       # the store so far
        Frozen({'timestamp': 60, 'symbol': 8})
        >>> later.cal().get_features().sizes         # same store, later dates
        Frozen({'timestamp': 30, 'symbol': 8})
        >>> later.update()
        >>> factor.read(overwrite=True).get_features().sizes
        Frozen({'timestamp': 90, 'symbol': 8})
        """
        self._refuse_if_resampled("update")
        with Timer(f"{self.__class__.__name__}: update"):
            self._auto_filter()
            self.data_backend.widen_and_append(
                self.config.file_path,
                fill_values=self._widen_fill_values(),
                **kwargs,
            )
            return self

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

        The config is deep-copied, its ``dataset`` replaced by
        ``dataset.copy()``, and re-assigned through the ``config`` setter,
        so the copy shares no mutable state with this factor. The copy
        holds no panel until it is read or computed.

        Examples
        --------
        >>> other = factor.copy()
        >>> other.config.dataset is factor.config.dataset
        False
        """
        other = copy.copy(self)
        other.data_backend = XrBackend()
        config = copy.deepcopy(dataclasses.replace(self.config, dataset=None))
        config.dataset = self.config.dataset.copy()
        other.config = config
        return other

    def resample(
        self, freq: str, how: dict[str, str] | str
    ) -> Self:
        """Return a copy of this factor whose panel is resampled onto ``freq``.

        The factor is still computed on its dataset's own bars; only the
        computed panel is aggregated, so a minute-bar factor becomes a
        daily one without changing what it measures. The copy's config
        carries ``resample_freq=freq`` and ``resample_how=how``; ``cal()``,
        ``read()``, ``get_features()`` and ``get_labels()`` on it give the
        resampled panel. If this factor already holds a panel, the copy
        holds that panel resampled, in memory of its own; otherwise the copy
        is empty. This factor and its dataset are not changed.

        A resampled factor cannot ``update()`` its store or stream, and
        ``save()`` writes to ``store_path``, the resampled store beside the
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
            If ``freq`` or a method is not a known token, or, when this
            factor holds a panel, if ``freq`` is not coarser than its bars
            or ``how`` does not name every variable.

        Examples
        --------
        >>> minute = factor.cal()                      # minute bars
        >>> daily = minute.resample("1d", "last")
        >>> daily.get_features().sizes["timestamp"]
        2
        >>> minute.get_features().sizes["timestamp"]   # unchanged
        780
        """
        other = self.copy()
        config = other.config
        config.resample_freq = freq
        config.resample_how = how
        other.config = config
        if self._holds_data():
            other.data_backend.to_internal(
                self.data_backend.get_xarray_dataset()
            )
            other._apply_resample()
        return other

    def _holds_data(self) -> bool:
        """Return whether the storage backend holds a panel."""
        try:
            self.data_backend.data
        except AttributeError:
            return False
        return True

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

    def _apply_resample(self) -> None:
        """Replace the held panel with its resample, per the config."""
        freq = self.config.resample_freq
        if freq is None:
            return
        data = self.data_backend.get_xarray_dataset()
        timestamps = data["timestamp"].values
        assert_coarser(timestamps, freq, self.class_name)
        how = resolve_resample_how(
            self.config.resample_how, list(data.data_vars), self.class_name
        )
        labels = pd.Series(self._resample_labels(timestamps, freq), index=timestamps)
        with Timer(f"{self.__class__.__name__}: resample to {freq}"):
            self.data_backend.resample(labels, how)

    def _hold_panel(self, data: xr.Dataset) -> None:
        """Hand a computed panel to the backend, narrow it and resample it."""
        self.data_backend.to_internal(data)
        self._auto_filter()
        self._apply_resample()

    def _widen_fill_values(self) -> dict:
        """Return per-variable fill values for cells that widening creates.

        The default is empty, which leaves the choice to the backend.
        """
        return {}

    def _get_lazyframe(self) -> pl.LazyFrame:
        """Return the held panel as a ``LazyFrame`` with the index as columns."""
        df = self.data_backend.get_xarray_dataset().to_pandas()  # type: ignore
        df = pl.LazyFrame(df.reset_index())
        return df

    def _get_xarray_dataset(self) -> xr.Dataset:
        """Return the held panel as an ``xarray.Dataset``."""
        return self.data_backend.get_xarray_dataset()  # type: ignore

    def _get_features(self, data: xr.Dataset) -> xr.Dataset:
        """Turn the held panel into features; factor classes override this.

        Raises
        ------
        NotImplementedError
            Unless a subclass overrides it.
        """
        raise NotImplementedError

    def get_features(self) -> xr.Dataset:
        """Return the computed panel as model features.

        Examples
        --------
        >>> panel = factor.cal().get_features()
        >>> list(panel.data_vars), dict(panel.sizes)
        (['momentum_20'], {'timestamp': 60, 'symbol': 8})
        """
        return self._get_features(self._get_xarray_dataset())

    def _get_labels(self, data: xr.Dataset) -> xr.Dataset:
        """Turn the held panel into labels; label classes override this.

        Raises
        ------
        NotImplementedError
            Unless a subclass overrides it.
        """
        raise NotImplementedError

    def get_labels(self) -> xr.Dataset:
        """Return the computed panel as model labels.

        Examples
        --------
        >>> panel = label.cal().get_labels()     # a label class
        >>> list(panel.data_vars), panel.sizes
        (['ret_1'], Frozen({'timestamp': 21, 'symbol': 16}))
        """
        return self._get_labels(self._get_xarray_dataset())

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

    @abstractmethod
    def cal(self) -> Self:
        """Compute the factor panel and hold it in the storage backend.

        Returns
        -------
        Self
            ``self``, for chaining.

        Examples
        --------
        A backend override computes a ``(timestamp, symbol)`` panel, hands
        it to the storage backend and narrows it to the configured window::

            def cal(self) -> Self:
                panel = self._compute()            # an xarray.Dataset
                self.data_backend.to_internal(panel)
                self._auto_filter()
                return self
        """
        ...


class FactorKunQuant(Factor):
    """Factor backend that compiles a KunQuant op graph to native code.

    A subclass describes its factor as a KunQuant graph in
    ``_get_factor_func``: ``Input`` nodes named after ``config.data_columns``,
    operator nodes, and one ``Output`` per factor name. The same graph is
    compiled on demand in two layouts, ``TS`` for ``cal()`` (the whole history
    in one call) and ``STREAM`` for ``cal_stream()`` (one bar at a time), so a
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
        not carried over; the copy compiles again on its first ``cal()``.

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

    def _auto_filter(self):
        """Narrow the panel in batch mode; a stream holds one bar, so skip."""
        if self.config.mode == "batch":
            super()._auto_filter()

    @property
    def num_symbols(self) -> int:
        """Number of symbols the graph runs over.

        Batch mode asks the dataset; stream mode reads the symbol list pinned
        on the dataset config, since no panel has been loaded.

        Raises
        ------
        ValueError
            If ``config.mode`` is neither ``"batch"`` nor
            ``"stream"``.

        Examples
        --------
        >>> factor.num_symbols        # stream config pinning 16 symbols
        16
        """
        if self.config.mode == "batch":
            return super().num_symbols
        elif self.config.mode == "stream":
            return len(self.config.dataset.config.symbols)
        else:
            raise ValueError(f"mode {self.config.mode} is not supported")

    @property
    def symbols(self) -> list[str]:
        """Symbols the graph runs over, in axis order.

        Batch mode asks the dataset; stream mode reads the symbol list pinned
        on the dataset config.

        Raises
        ------
        ValueError
            If ``config.mode`` is neither ``"batch"`` nor
            ``"stream"``.

        Examples
        --------
        >>> factor.symbols[:3]
        ['AAPL', 'MSFT', 'NVDA']
        """
        if self.config.mode == "batch":
            return super().symbols
        elif self.config.mode == "stream":
            return list(self.config.dataset.config.symbols)
        else:
            raise ValueError(f"mode {self.config.mode} is not supported")

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
        """Wrap raw ``[time, symbol]`` arrays in an ``xarray.Dataset`` and hold it.

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
        ds = xr.Dataset(
            {k: (["timestamp", "symbol"], v) for k, v in raw_factor.items()},
            coords={
                "timestamp": timestamps,
                "symbol": symbols,
            },
        )
        self._hold_panel(ds)
        return self

    @abstractmethod
    def _get_factor_func(self) -> Function:
        """Build and return the KunQuant graph that computes this factor.

        Input names must match ``config.data_columns``; output names are the
        factor names.
        """
        ...

    def cal(self) -> Self:
        """Run the compiled graph over the dataset's full history.

        The dataset is converted with ``to_kunquant``, the graph is compiled
        if no library is cached, run from bar 0 on an executor of
        ``config.njobs`` threads, and the outputs are wrapped as a panel. The
        compiled library is dropped afterwards, so each call compiles again.

        Returns
        -------
        Self
            ``self``, for chaining.

        Examples
        --------
        >>> factor.get_factor_names()
        ('rank_close', 'ma_close', 'ma_rank')
        >>> factor.cal().get_features().sizes   # 21 configured bars
        Frozen({'timestamp': 21, 'symbol': 16})
        """
        input_dict, symbols, timestamp = self.config.dataset.to_kunquant(
            data_columns=self.config.data_columns
        )
        # Every input is laid out [time, symbol]; any one gives the time count.
        num_time = next(iter(input_dict.values())).shape[0]

        if self._lib is None:
            self._lib = self._make()

        modu = self._lib.getModule(f"{self.__class__.__name__}")  # type: ignore

        executor = kr.createMultiThreadExecutor(self.config.njobs)
        with Timer(f" {self.__class__.__name__}: cal"):
            out_dict = kr.runGraph(executor, modu, input_dict, 0, num_time)

        self._lib = None

        self._to_xarray_dataset(out_dict, timestamp, symbols)

        return self

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
    ``symbol`` and the factor columns. Nothing is materialized until ``cal()``
    collects it and converts the result to an ``xarray.Dataset``. Factor
    names are not declared: they are read from the schema of the returned
    frame, so they are known at construction time (which reads a few rows
    from the store) and always match what the expression chain produces.

    Column names are whatever the underlying store holds; unlike the KunQuant
    path, no per-market renaming is applied.

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

    def cal(self) -> Self:
        """Collect the expression chain over the full dataset and hold the panel.

        ``config.factor_names`` is refreshed from the collected schema.

        Returns
        -------
        Self
            ``self``, for chaining.

        Examples
        --------
        >>> factor.cal().get_features().sizes
        Frozen({'timestamp': 60, 'symbol': 8})
        """
        lf = self.config.dataset.read().get_lazyframe()
        factor_lf = self._get_factor_lazyframe(lf)

        self.config.factor_names = tuple(
            name
            for name in factor_lf.collect_schema().names()
            if name not in self._INDEX_COLUMNS
        )

        with Timer(f"{self.__class__.__name__}: cal"):
            frame = factor_lf.collect().to_pandas()
            frame = frame.set_index(list(self._INDEX_COLUMNS))
            data = xr.Dataset.from_dataframe(frame)

        self._hold_panel(data)
        return self
