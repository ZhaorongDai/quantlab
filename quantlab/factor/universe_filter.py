"""Point-in-time price and liquidity filter for a factor's universe.

A *universe* is the set of symbols a strategy may trade on a given bar. A
*point-in-time* universe is decided using only data available at that bar,
so a backtest never benefits from knowing the future. Here a symbol is in
the universe at bar ``t`` when its raw close is at least ``min_price`` and
the mean of its raw dollar volume (``close * volume``) over the last
``window`` bars is at least ``min_dollar_volume``. Nothing after ``t``
affects the decision at ``t``.

Most factors in this project are KunQuant factors: formulas written as a
graph of operators that KunQuant compiles to native code. Some operators in
such a graph are *cross-sectional*: they compare symbols with each other on
the same bar, such as a rank or a cross-sectional z-score. Others are
*time-series* operators that look at one symbol's own history.
``UniverseFilteredFactor`` wraps a KunQuant factor or label and rewrites its
graph so that out-of-universe symbols are hidden from every cross-sectional
operator, while time-series operators still see the full history. It then
blanks the outputs wherever the symbol is out of the universe. Because the
wrapper is itself a ``FactorKunQuant``, it can be used in
``MLConfig.factors`` and ``labels`` and in a backtest config with no change
to the model or backtest code.

Two KunQuant limits apply to every caller. Batch runs always start at bar 0,
because in KunQuant 0.1.11 a non-zero start index gives wrong results for
every ``GenericCrossSectionalOp`` (a cross-sectional operator with a
hand-written C++ body, such as ``CrossSectionalZScore``). And the number of
symbols must be a multiple of the host's SIMD block width, the number of
values the CPU's vector instructions process at once.

This filter answers "is the symbol expensive and liquid enough to trade".
That is a separate question from index membership ("was it in the S&P 500
on that day"), which ``quantlab.dataset._support.masking`` handles; the two
can be combined.
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

#: Name of the mask input in the rewritten graph. Defined at module level
#: because ``_mask_cross_sectional_inputs`` needs it before the class exists.
_MASK_INPUT_NAME = "universe_mask"


def _mask_cross_sectional_inputs(
    ops: list[OpBase],
) -> tuple[list[OpBase], bool]:
    """Insert ``Div(v, universe_mask)`` in front of every cross-sectional input.

    The mask is 1.0 for an in-universe symbol and NaN otherwise. Dividing by
    1.0 leaves a value unchanged and dividing by NaN gives NaN, which
    KunQuant's cross-sectional operators skip. So out-of-universe symbols
    drop out of every cross-sectional operator, while time-series operators
    upstream still see the full history. Each distinct input node is wrapped
    once and the wrapper is shared. ``op.inputs`` is changed in place, so
    callers must pass a freshly built graph; a graph rewritten twice is
    masked twice.

    Parameters
    ----------
    ops : list[OpBase]
        The ops of a freshly built factor graph.

    Returns
    -------
    tuple[list[OpBase], bool]
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

    See the module docstring for what the universe is and how the graph is
    rewritten. Wrap both the factors and the labels of a model: filtering
    only the factors leaves label rows for out-of-universe symbols, and
    filtering only the labels leaves those symbols inside the factors'
    ranks. The wrapper's ``config`` is the inner factor's own config object,
    not a copy, so dates written by the model or backtester,
    ``config.window``, ``config.kwargs`` and ``config.data_columns`` all
    apply to the inner factor.

    Being out of the universe only blanks cells. The symbol axis of every
    output equals the input's, and a symbol that is out for the whole window
    stays as an all-NaN column. A store written through ``save()`` holds the
    rewritten graph's outputs before the output mask is applied; ``read()``
    recomputes the mask from the dataset and applies it. A store read
    through the wrapper must therefore have been written by the wrapper.

    Parameters
    ----------
    factor : FactorKunQuant
        The KunQuant factor or label to wrap. Polars factors are refused
        because their cross-sectional logic is not a KunQuant graph and
        cannot be rewritten. Already-wrapped factors are refused because
        the two masks would combine without any warning.
    min_price : float, default 5.0
        Minimum raw close for a symbol to be in the universe.
    min_dollar_volume : float, default 1_000_000.0
        Minimum trailing mean of raw ``close * volume``.
    window : int, default 20
        Number of bars the dollar-volume mean is taken over.

    Attributes
    ----------
    factor : FactorKunQuant
        The wrapped factor.
    min_price, min_dollar_volume : float
        The two thresholds.
    window : int
        The dollar-volume window in bars.

    Raises
    ------
    TypeError
        If ``factor`` is not a ``FactorKunQuant`` or is itself a
        ``UniverseFilteredFactor``.
    ValueError
        If ``window`` is less than 1.

    Examples
    --------
    >>> factors = [UniverseFilteredFactor(Alpha101Stock(factor_config))]
    >>> labels = [UniverseFilteredFactor(Return(label_config))]
    >>> model = XGBoostRegressor(MLConfig(factors=factors, labels=labels, ...))
    """

    #: Raw columns the mask reads. Adjusted prices are rescaled by later
    #: splits and dividends, so they cannot say whether a stock was cheap at
    #: the time.
    PRICE_COLUMN = "close"
    VOLUME_COLUMN = "volume"

    #: Name of the mask input in the rewritten graph.
    MASK_INPUT = _MASK_INPUT_NAME

    #: Extra calendar days ``_reset_dataset_config`` loads before the factor's
    #: start date: ``LOOKBACK_DAYS_PER_BAR * window + LOOKBACK_PAD_DAYS``. The
    #: backtester's warm-up only covers ``config.window`` bars; without this
    #: the first ``window - 1`` bars would have no dollar-volume history and
    #: fall out of the universe.
    LOOKBACK_DAYS_PER_BAR = 2
    LOOKBACK_PAD_DAYS = 10

    def __init__(
        self,
        factor: FactorKunQuant,
        min_price: float = 5.0,
        min_dollar_volume: float = 1_000_000.0,
        window: int = 20,
    ):
        """Initialize the wrapper; see the class docstring for parameters."""
        if not isinstance(factor, FactorKunQuant):
            raise TypeError(
                f"UniverseFilteredFactor wraps a FactorKunQuant, got "
                f"{type(factor).__name__}. A Polars factor's cross-sectional "
                f"expressions are Polars expressions, not a KunQuant op graph, "
                f"so they cannot be rewritten. Masking only the outputs would "
                f"leave every out-of-universe symbol inside each rank and "
                f"z-score, which is what this class exists to prevent."
            )
        if isinstance(factor, UniverseFilteredFactor):
            raise TypeError(
                f"UniverseFilteredFactor cannot wrap another "
                f"{type(factor).__name__}: the inner wrapper would mask the "
                f"cross-sections a second time, and the two masks' thresholds "
                f"would combine without any warning. Wrap the innermost factor "
                f"once, with the thresholds you want."
            )
        if window < 1:
            raise ValueError(
                f"window must be >= 1 bar, got {window}; it is the number of "
                f"bars the trailing dollar-volume mean is taken over."
            )

        # super().__init__() is skipped on purpose: the base config setter
        # would overwrite the inner config's `name` with this wrapper's import
        # path, and the inner factor could no longer be rebuilt from it.
        self.factor = factor
        self.min_price = float(min_price)
        self.min_dollar_volume = float(min_dollar_volume)
        self.window = int(window)

        self.data_backend = XrBackend()
        self._stream_context: kr.StreamContext = None
        self._lib = None
        self._buffer_name_to_id: dict = {}

        # Whether the compiled graph has a cross-sectional operator; set by
        # `_get_factor_func`.
        self._uses_mask = False
        # The mask from the latest `cal()` / `read()` / `cal_stream()`.
        self._universe_mask: xr.DataArray | None = None
        self._stream_dollar_volume = collections.deque(maxlen=self.window)

        self._reset_dataset_config()

    def __repr__(self) -> str:
        """Return a string naming the inner factor and the three parameters."""
        return (
            f"UniverseFilteredFactor({self.factor!r}, "
            f"min_price={self.min_price}, "
            f"min_dollar_volume={self.min_dollar_volume}, "
            f"window={self.window})"
        )

    # ------------------------------------------------------------------
    # Config delegation: whatever the model and backtest layers write goes
    # to the inner factor's config.
    # ------------------------------------------------------------------

    @property
    def config(self):
        """The inner factor's config object, shared rather than copied.

        Dates the model or backtester write, ``config.window`` used for
        warm-up, ``config.kwargs["n_forward_periods"]`` and
        ``config.data_columns`` all apply to the inner factor. This is what
        lets the wrapper stand in for the factor it wraps.

        Examples
        --------
        >>> wrapped = UniverseFilteredFactor(inner, window=3)
        >>> wrapped.config is inner.config
        True
        """
        return self.factor.config

    @config.setter
    def config(self, value):
        """Assign the config to the inner factor, then widen its dataset dates.

        Examples
        --------
        >>> wrapped.config = new_config        # start_date "2024-02-01"
        >>> wrapped.factor.config is new_config
        True
        >>> new_config.dataset.config.start_date   # widened for warm-up
        '2024-01-16'
        """
        self.factor.config = value
        self._reset_dataset_config()

    def _get_factor_names(self) -> tuple[str, ...]:
        """Return the inner factor's names; the wrapper changes values, not columns."""
        return self.factor._get_factor_names()

    def _reset_dataset_config(self) -> None:
        """Let the inner factor reset its dates, then move the start date earlier.

        The backtester's warm-up covers ``config.window`` bars and does not
        know that the mask needs ``window`` bars of dollar-volume history
        before its first value. The start date only ever moves earlier, so
        an inner factor that already reaches further back keeps its date.
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

        A fresh graph is requested from the inner factor on every call,
        because the rewrite changes ``op.inputs`` in place.
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

        Parameters
        ----------
        panel : xr.Dataset
            The dataset panel; must carry the raw ``close`` and
            ``volume`` variables.

        Returns
        -------
        xr.DataArray
            A ``DataArray`` named ``universe_mask`` that is 1.0 where the
            symbol is in the universe and NaN elsewhere. A window that is not
            yet full, or that contains a NaN, counts as out of the universe.

        Raises
        ------
        ValueError
            If either raw column is missing. An error is better than an
            all-out (or all-in) mask that would let the pipeline quietly
            produce empty or unfiltered results.

        Examples
        --------
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
                    f"UniverseFilteredFactor needs the raw column {column!r} "
                    f"to decide universe membership, and the dataset panel "
                    f"does not carry it (present: "
                    f"{sorted(map(str, panel.data_vars))}). The mask reads raw "
                    f"close and volume, never the adjusted columns: adjusted "
                    f"prices are rescaled by later splits and dividends, so "
                    f"they do not show what a stock cost at the time."
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

        # A NaN comparison is False, so a missing value means "out".
        in_universe = (close >= self.min_price) & (
            average >= self.min_dollar_volume
        )

        return xr.where(in_universe, 1.0, np.nan).rename(self.MASK_INPUT)

    # ------------------------------------------------------------------
    # Batch computation
    # ------------------------------------------------------------------

    def cal(self) -> Self:
        """Compute the wrapped factor in batch mode, feeding the mask as an input.

        Works like ``FactorKunQuant.cal``, plus the mask. The graph always
        runs from bar 0; see the module docstring for why.

        Returns
        -------
        Self
            ``self``, for chaining.

        Examples
        --------
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

        # `_make()` calls `_get_factor_func()`, which sets `_uses_mask`.
        self._lib = self._make()
        modu = self._lib.getModule(f"{self.__class__.__name__}")  # type: ignore

        # The same panel `to_kunquant` just loaded, served from cache.
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
        rewritten graph's outputs before the output mask; the mask is applied
        to what ``get_features()`` and ``get_labels()`` return. A store
        written by the unwrapped inner factor cannot be repaired here,
        because its cross-sectional values already include out-of-universe
        symbols.

        Parameters
        ----------
        overwrite : bool, default False
            Re-open the stores even if data is already held in memory.

        Returns
        -------
        Self
            ``self``, for chaining.

        Examples
        --------
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

        Returns
        -------
        Self
            ``self``, for chaining.

        Examples
        --------
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

        The dollar-volume mean is ``np.mean`` over a full window, so a window
        that is not yet full, or holds a NaN, gives NaN. This matches the
        batch path, which uses ``min_periods=window``; if the two differed,
        streaming and batch factor values would disagree.

        Parameters
        ----------
        data : dict[str, np.ndarray]
            Column name to array of per-symbol values. Besides
            ``config.data_columns`` it must carry the raw ``close`` and
            ``volume`` keys, which the mask reads; the inherited code only
            pushes ``data_columns`` into the graph.
        timestamp : int
            The bar's timestamp.
        symbols : list[str]
            Symbol coordinate values, in array order.

        Returns
        -------
        Self
            ``self``, for chaining.

        Raises
        ------
        ValueError
            If ``close`` or ``volume`` is missing from ``data``.

        Examples
        --------
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
                f"the raw column(s) {missing}, which decide universe "
                f"membership. They are extra keys beyond config.data_columns "
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

        # Push the mask first: `super().cal_stream` runs the graph as soon as
        # it has pushed the data columns.
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

        Raises
        ------
        RuntimeError
            If none of ``cal()``, ``read()`` or ``cal_stream()``
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
        inside the storage backend with an error that names neither the
        cause nor the fix.
        """
        self._assert_computed()
        return super()._get_xarray_dataset()

    def _mask_panel(self, data: xr.Dataset) -> xr.Dataset:
        """Blank ``data`` where the mask is NaN, keeping the symbol axis intact.

        The mask is aligned to ``data``'s coordinates. No column is dropped:
        a symbol out of the universe for the whole window remains as an
        all-NaN column. The symbol axis therefore does not depend on the
        date window, and a model never meets a panel missing a symbol it was
        trained on.
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

        The inner transform runs first. ``Return._get_labels`` shifts values
        earlier in time, and masking afterwards applies the mask at the
        label's own timestamp ``t``. Masking before the shift would decide
        bar ``t`` using the universe of a later bar, which leaks future
        information into the label.
        """
        return self._mask_panel(self.factor._get_labels(data))

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def get_config(self) -> dict:
        """Return the wrapper's config with the inner factor's under ``"factor"``.

        There is no top-level ``"dataset"`` key on purpose. The dataset
        belongs to the inner factor, and a second copy would be rebuilt as a
        second dataset object reading the same store.

        Examples
        --------
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

        The inner factor is rebuilt through ``load_factor_from_config``.
        Missing and unknown keys both raise. A missing parameter is never
        filled from the current default, because a default that changes
        later would rebuild a stored run with a different universe without
        any warning.

        Parameters
        ----------
        config : dict
            A dict as produced by ``get_config()``.

        Returns
        -------
        UniverseFilteredFactor
            A new ``UniverseFilteredFactor``.

        Raises
        ------
        ValueError
            If the ``factor`` key is absent, or the parameter keys
            do not exactly match ``min_price``, ``min_dollar_volume`` and
            ``window``.

        Examples
        --------
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
                f"parameters are not filled from the current defaults, which "
                f"may differ from what the stored run used, and unknown keys "
                f"are not ignored, because they may change what the run meant."
            )

        return cls(load_factor_from_config(inner), **config)
