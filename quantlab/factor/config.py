"""The configs of the factor layer.

A factor is constructed from one of these and exposes it as ``self.config``. They
are frozen; the factor's config setter normalises the config it is given into a
new one (dates, factor names, the ``name`` field the factor is rebuilt from; see
``quantlab.core.component``). A factor's ``dataset`` (and a market-feature
factor's ``series``) is declared with ``component()`` and written as its own config.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from quantlab.core.component import component
from quantlab.core.config import FrozenConfig
from quantlab.enums.data import ResampleFrequency

if TYPE_CHECKING:
    from quantlab.dataset.base import MarketDataset


@dataclass(kw_only=True, frozen=True)
class BaseFactorConfig(FrozenConfig):
    """Fields every factor (and label) shares, whichever backend computes it.

    The config says what is computed, not when: the date range is an
    argument of ``compute``, ``read`` and ``build``. ``compute(start, end)``
    requests ``warmup_bars`` bars of the dataset before ``start``, counted
    on the dataset's own calendar, so rolling computations are warm at the
    first requested bar.

    Examples
    --------
    With ``dataset`` a market dataset built earlier:

    >>> cfg = BaseFactorConfig(
    ...     warmup_bars=20,
    ...     dataset=dataset,
    ...     file_path="/data/factor/momentum.zarr",
    ... )
    >>> cfg.factor_names, cfg.warmup_bars
    (None, 20)
    """

    #: Bars of history read before the requested start to warm up rolling
    #: computations, counted on the dataset's own calendar.
    warmup_bars: int
    #: The market dataset the factor is computed from, or a list of them,
    #: which the factor merges into one ``MergedDataset``.
    dataset: "MarketDataset | tuple" = component()
    #: Path of the Zarr store ``build`` writes the factor values to and
    #: ``read`` reads them back from.
    file_path: str | None = None
    #: Names of the factor variables this factor produces; filled from the
    #: factor definition when left ``None``.
    factor_names: tuple[str, ...] | None = None
    #: Free-form options a specific factor class may read.
    kwargs: dict | None = None
    #: Bar size the computed factor panel is resampled onto; ``None`` keeps
    #: the dataset's bars. The factor is computed on the dataset's own bars
    #: first and aggregated afterwards. Set by ``resample()``.
    resample_freq: ResampleFrequency | None = None
    #: How each factor variable is aggregated into a resampled bar: one
    #: ``ResampleMethod`` for every variable, or a ``{variable: method}``
    #: dict naming every variable. Required when ``resample_freq`` is set.
    resample_how: dict[str, str] | str | None = None

    #: Dotted import path of the factor class; filled by the config setter.
    name: str | None = None


@dataclass(kw_only=True, frozen=True)
class FactorConfig(BaseFactorConfig):
    """Config of a KunQuant-computed factor.

    Examples
    --------
    >>> cfg = FactorConfig(
    ...     warmup_bars=128,
    ...     dataset=dataset,
    ...     file_path="/data/factor/alpha101.zarr",
    ...     mode="batch",
    ...     data_columns=("open", "high", "low", "close", "volume", "amount"),
    ...     factor_names=("alpha001", "alpha002"),
    ... )
    >>> cfg.mode, cfg.njobs
    ('batch', 128)
    """

    #: ``"batch"`` compiles the graph for a whole date range;
    #: ``"stream"`` compiles it for incremental per-bar updates.
    mode: Literal["stream", "batch"]
    #: The dataset variables fed into the compiled graph as inputs.
    data_columns: tuple[str, ...]
    #: Threads used by the KunQuant executor.
    njobs: int = 128


@dataclass(kw_only=True, frozen=True)
class PolarsFactorConfig(BaseFactorConfig):
    """Config of a Polars-computed factor.

    Adds no fields to ``BaseFactorConfig``. The Polars backend is batch-only,
    so there is no ``mode``.

    Examples
    --------
    >>> cfg = PolarsFactorConfig(
    ...     warmup_bars=20,
    ...     dataset=dataset,
    ...     file_path="/data/factor/momentum.zarr",
    ...     kwargs={"n": 20},
    ... )
    >>> hasattr(cfg, "mode")
    False
    """


@dataclass(kw_only=True, frozen=True)
class MarketFeatureConfig(BaseFactorConfig):
    """Config of ``quantlab.factor.predefined.market.MarketFeatures``.

    ``dataset`` is the *target*: the panel whose symbols receive the market
    features, and the calendar ``warmup_bars`` is counted on. ``series``
    names the index or ETF datasets the features are computed from, one
    single-symbol dataset per name; the name prefixes the features, as in
    ``spy_ret_mean_20``. ``series`` is a component field, so ``to_dict()``
    nests each series dataset's config dict under it and
    ``MarketFeatures.from_config`` rebuilds them.

    Examples
    --------
    With ``stocks`` the target dataset and ``spy`` a dataset over one ETF:

    >>> cfg = MarketFeatureConfig(dataset=stocks, series={"spy": spy})
    >>> cfg.warmup_bars, list(cfg.series)
    (60, ['spy'])
    """

    #: Bars read before the requested start; 60 fills the longest window.
    warmup_bars: int = 60
    #: Series name to the single-symbol dataset it is computed from, in the
    #: order the features are listed.
    series: "dict[str, MarketDataset]" = component(many=True)
