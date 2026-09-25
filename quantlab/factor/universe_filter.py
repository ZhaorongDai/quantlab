"""Point-in-time price and liquidity universe filter, built as a factor wrapper.

``UniverseFilteredFactor`` wraps any KunQuant factor or label and is itself a
``FactorKunQuant``, so it drops into ``MLConfig.factors`` / ``labels`` and
into a backtest config without changes to the model or backtest layers. A
symbol is in the universe at bar ``t`` when its raw close is at least
``min_price`` and the trailing ``window``-bar mean of raw ``close * volume``
is at least ``min_dollar_volume``; nothing after ``t`` affects the mask at
``t``. The wrapper feeds that mask into every cross-sectional operator of the
inner graph, so out-of-universe symbols never take part in a rank or a
cross-sectional z-score, and it blanks the outputs of both factors and labels
where the mask is NaN. Time-series operators still see full history.

Two KunQuant limits apply to every caller: batch runs always start at bar 0
(in KunQuant 0.1.11 a non-zero ``start`` gives wrong results for every
``GenericCrossSectionalOp``), and the number of symbols must be a multiple of
the SIMD block width on the host.

This filter answers "is the symbol expensive and liquid enough to trade",
which is orthogonal to the index-membership masking in
``quantlab.dataset._support.masking``; the two can be stacked. See
``docs/universe.md``.
"""

import collections
from typing import Self

import KunQuant.runner.KunRunner as kr
import numpy as np
import pandas as pd
import xarray as xr
from KunQuant.Op import Builder, CrossSectionalOp, Input, OpBase
from KunQuant.ops import Div
from KunQuant.Stage import Function

from quantlab.base.factor import FactorKunQuant
from quantlab.backend import XrBackend
from quantlab.utils.module import load_factor_from_config
from quantlab.utils.timer import Timer

#: Name of the mask ``Input`` in the rewritten graph. Module-level because
#: ``_mask_cross_sectional_inputs`` needs it before the class is defined.
_MASK_INPUT_NAME = "universe_mask"


def _mask_cross_sectional_inputs(
    ops: list[OpBase],
) -> tuple[list[OpBase], bool]:
    """Insert ``Div(v, universe_mask)`` before every cross-sectional input.

    Dividing by 1.0 leaves a value unchanged and dividing by NaN yields NaN,
    so out-of-universe symbols vanish from every cross-sectional operator
    while time-series operators still see full history. Each distinct input
    node is wrapped once and shared. ``op.inputs`` is rewritten in place, so
    callers must pass a freshly built graph; rewriting a graph twice masks it
    twice.

    Args:
        ops: The ops of a freshly built factor graph.

    Returns:
        ``(ops, uses_mask)``: the topologically sorted ops and whether a mask
        ``Input`` was added. When the graph has no cross-sectional operator
        the ops are returned unchanged and no mask input is declared, because
        KunQuant prunes inputs that no ``Output`` consumes and a later buffer
        lookup for the mask would fail.
    """
    cross_sectional = [op for op in ops if isinstance(op, CrossSectionalOp)]
    if not cross_sectional:
        return list(ops), False

    builder = Builder()
    with builder:
        mask = Input(_MASK_INPUT_NAME)
        cache: dict[OpBase, OpBase] = {}
        for op in cross_sectional:
            for index, source in enumerate(op.inputs):
                if source not in cache:
                    cache[source] = Div(source, mask)
                op.inputs[index] = cache[source]

    return Function.topo_sort_ops(list(ops) + builder.ops), True


class UniverseFilteredFactor(FactorKunQuant):
    """Wrap a KunQuant factor or label with a point-in-time universe filter.

    Wrap both the factors and the labels of a model: filtering only the
    factors leaves label rows for out-of-universe symbols, and filtering only
    the labels leaves cross-sectional operators polluted by them. The
    wrapper's ``config`` is the inner factor's own config object, not a copy,
    so dates written by the model or backtester, ``config.window``,
    ``config.kwargs`` and ``config.data_columns`` all resolve on the inner
    factor.

    Being out of the universe only blanks cells: the symbol axis of every
    output is identical to the input's, and a symbol that is out for the
    whole window stays as an all-NaN column. Stores written through
    ``save()`` hold the outputs of the rewritten graph before the output
    mask; ``read()`` recomputes the mask from the dataset and applies it, so
    a store read through the wrapper must have been written by the wrapper.

    Args:
        factor: The KunQuant factor or label to wrap. Polars factors are
            refused because their cross-sectional logic is not a KunQuant
            graph and cannot be rewritten; already-wrapped factors are
            refused because the two masks would silently compose.
        min_price: Minimum raw close for a symbol to be in the universe.
        min_dollar_volume: Minimum trailing mean of raw ``close * volume``.
        window: Number of bars the dollar-volume mean is taken over.

    Raises:
        TypeError: If ``factor`` is not a ``FactorKunQuant`` or is itself a
            ``UniverseFilteredFactor``.
        ValueError: If ``window`` is less than 1.

    Example:
        >>> factors = [UniverseFilteredFactor(Alpha101Stock(factor_config))]
        >>> labels = [UniverseFilteredFactor(Return(label_config))]
        >>> model = XGBoostRegressor(MLConfig(factors=factors, labels=labels, ...))
    """

    #: Raw columns the mask reads, never the adjusted ones: adjusted history
    #: is depressed by later splits and dividends, so it cannot say whether a
    #: stock was cheap at the time.
    PRICE_COLUMN = "close"
    VOLUME_COLUMN = "volume"

    #: Name of the mask ``Input`` in the rewritten graph.
    MASK_INPUT = _MASK_INPUT_NAME

    #: Extra calendar days ``_reset_dataset_config`` pulls before the factor's
    #: start date: days per bar times ``window``, plus a fixed pad. The
    #: backtester's warm-up counts only ``config.window`` bars; without this
    #: the first ``window - 1`` bars of a window would have no dollar-volume
    #: history and drop out of the universe.
    LOOKBACK_DAYS_PER_BAR = 2
    LOOKBACK_PAD_DAYS = 10

    def __init__(
        self,
        factor: FactorKunQuant,
        min_price: float = 5.0,
        min_dollar_volume: float = 1_000_000.0,
        window: int = 20,
    ):
        """Wrap ``factor`` and widen its dataset dates for the mask's warm-up."""
        if not isinstance(factor, FactorKunQuant):
            raise TypeError(
                f"UniverseFilteredFactor wraps a FactorKunQuant, got "
                f"{type(factor).__name__}. A Polars factor's cross-sectional "
                f"expressions are polars expressions, not a KunQuant op graph, "
                f"so they cannot be rewritten -- and masking only the OUTPUTS "
                f"would leave every out-of-universe symbol sitting inside each "
                f"rank/zscore, which is exactly what this class exists to "
                f"prevent."
            )
        if isinstance(factor, UniverseFilteredFactor):
            raise TypeError(
                f"UniverseFilteredFactor cannot wrap another "
                f"{type(factor).__name__}: the inner wrapper would mask the "
                f"cross-sections a second time, and the two masks' parameters "
                f"would silently compose. Wrap the innermost factor once, with "
                f"the parameters you want."
            )
        if window < 1:
            raise ValueError(
                f"window must be >= 1 bar, got {window}; it is the number of "
                f"bars the trailing dollar-volume mean is taken over."
            )

        # Deliberately not calling super().__init__(): Factor's config setter
        # would overwrite the inner config's `name` with this wrapper's import
        # path, and the inner factor could no longer be rebuilt as its own
        # class.
        self.factor = factor
        self.min_price = float(min_price)
        self.min_dollar_volume = float(min_dollar_volume)
        self.window = int(window)

        self.data_backend = XrBackend()
        self._stream_context: kr.StreamContext = None
        self._lib = None
        self._buffer_name_to_id: dict = {}

        # Set when `_get_factor_func` compiles: does the graph have any
        # cross-sectional operator?
        self._uses_mask = False
        # The mask from the latest `cal()` / `read()` / `cal_stream()`.
        self._universe_mask: xr.DataArray | None = None
        self._stream_dollar_volume = collections.deque(maxlen=self.window)

        self._reset_dataset_config()

    def __repr__(self) -> str:
        """Return the wrapper, its inner factor and the three thresholds."""
        return (
            f"UniverseFilteredFactor({self.factor!r}, "
            f"min_price={self.min_price}, "
            f"min_dollar_volume={self.min_dollar_volume}, "
            f"window={self.window})"
        )

    # ------------------------------------------------------------------
    # Config delegation: everything the model and backtest layers write
    # lands on the inner config.
    # ------------------------------------------------------------------

    @property
    def config(self):
        """The inner factor's config object, shared rather than copied.

        Dates the model or backtester write, ``config.window`` used for
        warm-up, ``config.kwargs["n_forward_periods"]`` and
        ``config.data_columns`` all resolve on the inner factor, which is what
        makes the wrapper a drop-in replacement.

        Example:
            >>> wrapped = UniverseFilteredFactor(inner, window=3)
            >>> wrapped.config is inner.config
            True
        """
        return self.factor.config

    @config.setter
    def config(self, value):
        """Assign the config to the inner factor, then re-widen its dates.

        Example:
            >>> wrapped.config = new_config        # start_date "2024-02-01"
            >>> wrapped.factor.config is new_config
            True
            >>> new_config.dataset.config.start_date   # widened for warm-up
            '2024-01-16'
        """
        self.factor.config = value
        self._reset_dataset_config()

    def _get_factor_names(self) -> tuple[str, ...]:
        """Return the inner factor's names; wrapping changes values, not columns."""
        return self.factor._get_factor_names()

    def _reset_dataset_config(self) -> None:
        """Let the inner factor reset its dates, then pull the start date earlier.

        The backtester's warm-up counts ``config.window`` bars and does not
        know the mask needs ``window`` bars of dollar-volume history before
        its first value. The start date is only ever moved earlier, never
        later, so an inner factor that already reaches further back keeps
        its date.
        """
        self.factor._reset_dataset_config()

        config = self.config
        widened = pd.to_datetime(config.start_date) - pd.DateOffset(
            days=self.LOOKBACK_DAYS_PER_BAR * self.window
            + self.LOOKBACK_PAD_DAYS
        )
        widened_date = widened.strftime("%Y-%m-%d")

        dataset_config = config.dataset.config
        # Both are zero-padded ISO dates, so string order is date order.
        if widened_date < dataset_config.start_date:
            dataset_config.start_date = widened_date

    # ------------------------------------------------------------------
    # Graph rewrite
    # ------------------------------------------------------------------

    def _get_factor_func(self) -> Function:
        """Return the inner graph with every cross-sectional input masked.

        A fresh graph is requested from the inner factor on every call because
        the rewrite mutates ``op.inputs`` in place.
        """
        ops = self.factor._get_factor_func().ops
        rewritten, uses_mask = _mask_cross_sectional_inputs(ops)
        self._uses_mask = uses_mask
        return Function(rewritten)

    # ------------------------------------------------------------------
    # The mask itself
    # ------------------------------------------------------------------

    def compute_universe_mask(self, panel: xr.Dataset) -> xr.DataArray:
        """Compute the ``(timestamp, symbol)`` mask from raw close and volume.

        Args:
            panel: The dataset panel; must carry the raw ``close`` and
                ``volume`` variables.

        Returns:
            A ``DataArray`` named ``universe_mask`` that is 1.0 where the
            symbol is in the universe and NaN elsewhere. A window that is not
            yet full, or that contains a NaN, counts as out of the universe.

        Raises:
            ValueError: If either raw column is missing. Failing loudly is
                preferred to returning an all-out (or all-in) mask that would
                let the pipeline quietly produce empty results.

        Example:
            >>> panel = xr.Dataset(
            ...     {"close": (("timestamp", "symbol"), [[10.0, 4.0]] * 4),
            ...      "volume": (("timestamp", "symbol"), [[2e5, 2e5]] * 4)},
            ...     coords={"timestamp": pd.bdate_range("2024-01-01", periods=4),
            ...             "symbol": ["AAA", "BBB"]},
            ... )
            >>> wrapped = UniverseFilteredFactor(inner, window=3)
            >>> wrapped.compute_universe_mask(panel).values   # BBB is under $5
            array([[nan, nan],
                   [nan, nan],
                   [ 1., nan],
                   [ 1., nan]])
        """
        for column in (self.PRICE_COLUMN, self.VOLUME_COLUMN):
            if column not in panel.data_vars:
                raise ValueError(
                    f"UniverseFilteredFactor needs the RAW column {column!r} "
                    f"to decide universe membership, and the dataset panel "
                    f"does not carry it (present: "
                    f"{sorted(map(str, panel.data_vars))}). The mask reads RAW "
                    f"close/volume, never the adjusted columns: adjusted "
                    f"history is depressed by splits and dividends, so a "
                    f"penny stock today can look like a $50 stock in 2015."
                )

        panel = panel.sortby("timestamp")
        close = (
            panel[self.PRICE_COLUMN]
            .transpose("timestamp", "symbol")
            .astype("float64")
        )
        volume = (
            panel[self.VOLUME_COLUMN]
            .transpose("timestamp", "symbol")
            .astype("float64")
        )

        dollar_volume = close * volume
        average = dollar_volume.rolling(
            timestamp=self.window, min_periods=self.window
        ).mean()

        # NaN compares False, which means out of the universe, as intended.
        in_universe = (close >= self.min_price) & (
            average >= self.min_dollar_volume
        )

        return xr.where(in_universe, 1.0, np.nan).rename(self.MASK_INPUT)

    # ------------------------------------------------------------------
    # Batch computation
    # ------------------------------------------------------------------

    def cal(self) -> Self:
        """Mirror ``FactorKunQuant.cal`` with the mask supplied as an extra input.

        The graph always runs from bar 0; see the module docstring.

        Returns:
            ``self``, for chaining.

        Example:
            >>> features = wrapped.cal().get_features()
            >>> features.sizes                     # the symbol axis is intact
            Frozen({'timestamp': 21, 'symbol': 16})
            >>> # a symbol under $5 all window is an all-NaN column, not dropped
            >>> bool(features["rank_close"].sel(symbol="PENY").isnull().all())
            True
        """
        input_dict, symbols, timestamps = self.config.dataset.to_kunquant(
            data_columns=self.config.data_columns
        )
        num_time = next(iter(input_dict.values())).shape[0]

        # `_make()` calls `_get_factor_func()`, which is where `_uses_mask`
        # is decided.
        self._lib = self._make()
        modu = self._lib.getModule(f"{self.__class__.__name__}")  # type: ignore

        # `to_kunquant` just read the dataset; this is the same cached panel.
        self._universe_mask = self.compute_universe_mask(
            self.config.dataset.get_xarray_dataset()
        )

        if self._uses_mask:
            aligned = (
                self._universe_mask.reindex(
                    timestamp=timestamps, symbol=symbols
                )
                .transpose("timestamp", "symbol")
                .values
            )
            input_dict[self.MASK_INPUT] = np.ascontiguousarray(
                aligned, dtype=np.float32
            )

        executor = kr.createMultiThreadExecutor(self.config.njobs)
        with Timer(f" {self.__class__.__name__}: cal"):
            out_dict = kr.runGraph(executor, modu, input_dict, 0, num_time)

        self._lib = None
        self._to_xarray_dataset(out_dict, timestamps, symbols)
        return self

    def read(self, overwrite: bool = False) -> Self:
        """Read the stored outputs back and recompute the mask from the dataset.

        The mask comes from the dataset's raw close and volume, not from the
        factor store, so the dataset is read first. Stored values are the
        outputs of the rewritten graph before the output mask; the mask is
        applied to what ``get_features()`` and ``get_labels()`` return. A
        store written by the unwrapped inner factor cannot be repaired here,
        because its cross-sectional values already include out-of-universe
        symbols.

        Args:
            overwrite: Re-open the stores even if cached data is held.

        Returns:
            ``self``, for chaining.

        Example:
            >>> wrapped.cal().save(mode="w")
            >>> wrapped.read().get_features().sizes
            Frozen({'timestamp': 21, 'symbol': 16})
        """
        self.config.dataset.read(overwrite=overwrite)
        self._universe_mask = self.compute_universe_mask(
            self.config.dataset.get_xarray_dataset()
        )
        super().read(overwrite=overwrite)
        return self

    # ------------------------------------------------------------------
    # Streaming computation
    # ------------------------------------------------------------------

    def init_stream(self) -> Self:
        """Compile the rewritten graph for streaming and bind the mask buffer.

        The mask buffer is bound only when the graph has a cross-sectional
        operator; otherwise KunQuant has pruned the mask input and the lookup
        would fail.

        Returns:
            ``self``, for chaining.

        Example:
            >>> wrapped.init_stream() is wrapped     # config.mode == "stream"
            True
            >>> "universe_mask" in wrapped._buffer_name_to_id
            True
        """
        super().init_stream()
        if self._uses_mask:
            self._buffer_name_to_id[self.MASK_INPUT] = (
                self._stream_context.queryBufferHandle(self.MASK_INPUT)
            )
        return self

    def cal_stream(
        self, data: dict[str, np.ndarray], timestamp: int, symbols: list[str]
    ) -> Self:
        """Advance one bar: push this bar's mask, then run the inner graph.

        The dollar-volume mean uses ``np.mean`` over a full window, so an
        unfilled window or a NaN inside it gives NaN, matching the batch
        path's ``min_periods=window`` semantics exactly; the two paths must
        agree or streaming and batch factor values would diverge.

        Args:
            data: Column name to array. Besides ``config.data_columns`` it
                must carry the raw ``close`` and ``volume`` keys, which the
                mask reads; the inherited push sends only ``data_columns``.
            timestamp: The bar's timestamp.
            symbols: Symbol coordinate values, in array order.

        Returns:
            ``self``, for chaining.

        Raises:
            ValueError: If ``close`` or ``volume`` is missing from ``data``.

        Example:
            >>> bar = {"adjClose": adj[step], "close": close[step], "volume": vol[step]}
            >>> wrapped.cal_stream(bar, step, symbols).get_features().sizes
            Frozen({'timestamp': 1, 'symbol': 16})
        """
        missing = [
            column
            for column in (self.PRICE_COLUMN, self.VOLUME_COLUMN)
            if column not in data
        ]
        if missing:
            raise ValueError(
                f"{type(self).__name__}.cal_stream: the bar dict is missing "
                f"the RAW column(s) {missing}, which decide universe "
                f"membership. They are EXTRA keys beyond config.data_columns "
                f"({tuple(self.config.data_columns)}): the inherited push only "
                f"sends data_columns, so they must be supplied explicitly."
            )

        close = np.asarray(data[self.PRICE_COLUMN], dtype=np.float64).reshape(-1)
        volume = np.asarray(
            data[self.VOLUME_COLUMN], dtype=np.float64
        ).reshape(-1)
        self._stream_dollar_volume.append(close * volume)

        if len(self._stream_dollar_volume) == self.window:
            average = np.mean(np.stack(self._stream_dollar_volume), axis=0)
        else:
            average = np.full(close.shape, np.nan)

        in_universe = (close >= self.min_price) & (
            average >= self.min_dollar_volume
        )
        row = np.where(in_universe, 1.0, np.nan).astype(np.float32)

        if self._stream_context is None:
            self.init_stream()

        # The mask must be pushed before `run()`; `super().cal_stream` runs
        # as soon as it has pushed the data columns.
        if self._uses_mask:
            self._stream_context.pushData(
                self._buffer_name_to_id[self.MASK_INPUT],
                np.ascontiguousarray(row),
            )

        super().cal_stream(data, timestamp, symbols)

        self._universe_mask = xr.DataArray(
            row.reshape(1, -1).astype("float64"),
            dims=["timestamp", "symbol"],
            coords={"timestamp": [timestamp], "symbol": list(symbols)},
            name=self.MASK_INPUT,
        )
        return self

    # ------------------------------------------------------------------
    # Output masking
    # ------------------------------------------------------------------

    def _assert_computed(self) -> None:
        """Raise if no mask has been computed yet.

        Raises:
            RuntimeError: If none of ``cal()``, ``read()`` or ``cal_stream()``
                has run.
        """
        if self._universe_mask is None:
            raise RuntimeError(
                f"{type(self).__name__}: no universe mask has been computed "
                f"yet, so the outputs cannot be masked. Call cal(), read() or "
                f"cal_stream() first."
            )

    def _get_xarray_dataset(self) -> xr.Dataset:
        """Return the held panel after checking that a mask exists.

        Without the check, calling ``get_features()`` before ``cal()`` fails
        inside the storage backend with an error that names neither the cause
        nor the remedy.
        """
        self._assert_computed()
        return super()._get_xarray_dataset()

    def _mask_panel(self, data: xr.Dataset) -> xr.Dataset:
        """Blank ``data`` where the mask is NaN, keeping the symbol axis intact.

        The mask is reindexed onto ``data``'s coordinates. No column is ever
        dropped: a symbol out of the universe for the whole window remains as
        an all-NaN column, so the symbol axis never depends on the date
        window and a model never meets a panel missing a symbol it trained
        on.
        """
        self._assert_computed()

        mask = self._universe_mask.reindex(
            timestamp=data["timestamp"], symbol=data["symbol"]
        )
        return data.where(mask.notnull())

    def _get_features(self, data: xr.Dataset) -> xr.Dataset:
        """Return the inner factor's features with the output mask applied."""
        return self._mask_panel(self.factor._get_features(data))

    def _get_labels(self, data: xr.Dataset) -> xr.Dataset:
        """Return the inner factor's labels with the output mask applied.

        The inner transform runs first: ``Return._get_labels`` shifts by
        ``-n`` bars, and masking afterwards applies the mask at the label's
        own timestamp ``t``. Masking before the shift would decide bar ``t``
        with the universe at ``t + n``, a look-ahead error.
        """
        return self._mask_panel(self.factor._get_labels(data))

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def get_config(self) -> dict:
        """Return the wrapper's config with the inner factor's under ``"factor"``.

        There is deliberately no top-level ``"dataset"`` key: the dataset
        belongs to the inner factor, and a copy would rebuild as a second
        dataset object reading the same store.

        Example:
            >>> cfg = wrapped.get_config()
            >>> sorted(cfg)
            ['factor', 'min_dollar_volume', 'min_price', 'name', 'window']
            >>> cfg["window"], cfg["min_price"]
            (3, 5.0)
        """
        return {
            "name": self.import_path,
            "factor": self.factor.get_config(),
            "min_price": float(self.min_price),
            "min_dollar_volume": float(self.min_dollar_volume),
            "window": int(self.window),
        }

    #: Keys of ``get_config()`` other than ``name`` and ``factor``.
    _PARAMETER_KEYS = frozenset({"min_price", "min_dollar_volume", "window"})

    @classmethod
    def from_config(cls, config: dict) -> "UniverseFilteredFactor":
        """Rebuild from a ``get_config()`` dict, inner factor included.

        The inner factor goes back through ``load_factor_from_config``.
        Missing and unknown keys both raise, and a missing parameter is never
        filled from the current default: a default that changes later would
        silently rebuild a stored run with a different universe.

        Args:
            config: A dict as produced by ``get_config()``.

        Returns:
            A new ``UniverseFilteredFactor``.

        Raises:
            ValueError: If the ``factor`` key is absent, or the parameter keys
                do not exactly match ``min_price``, ``min_dollar_volume`` and
                ``window``.

        Example:
            >>> rebuilt = UniverseFilteredFactor.from_config(wrapped.get_config())
            >>> rebuilt.window, rebuilt.min_price, rebuilt.min_dollar_volume
            (3, 5.0, 1000000.0)
            >>> UniverseFilteredFactor.from_config({"factor": inner_cfg, "window": 3})
            Traceback (most recent call last):
            ValueError: UniverseFilteredFactor.from_config: refusing to rebuild -- ...
        """
        config = dict(config)
        config.pop("name", None)

        inner = config.pop("factor", None)
        if inner is None:
            raise ValueError(
                f"{cls.__name__}.from_config: the config has no 'factor' key, "
                f"so there is no inner factor to rebuild."
            )

        missing = sorted(cls._PARAMETER_KEYS - set(config))
        unknown = sorted(set(config) - cls._PARAMETER_KEYS)
        if missing or unknown:
            raise ValueError(
                f"{cls.__name__}.from_config: refusing to rebuild -- "
                f"missing key(s) {missing}, unknown key(s) {unknown}. Missing "
                f"parameters are NOT filled from the current defaults, which "
                f"may differ from what the stored run used (D-25/WR-06)."
            )

        return cls(load_factor_from_config(inner), **config)
