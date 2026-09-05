from typing import NoReturn

import polars as pl
import xarray as xr

from base.config import PolarsFactorConfig
from base.factor_polars import FactorPolars

_DEFAULT_HORIZON = 20


class Momentum(FactorPolars):
    """N-day per-symbol price momentum: `Close_t / Close_{t-n} - 1`.

    **The D-08 worked example of the `FactorPolars` contract.** Read this file
    to learn how to write a new factor against the Polars backend: subclass
    `FactorPolars`, override the single `_get_factor_lazyframe(lf)` hook,
    return a lazyframe carrying only `timestamp`, `symbol` and the computed
    factor column, and let the inherited `cal()` do the rest. Nothing here
    materializes -- the whole expression chain is deferred until `cal()`
    collects it (D-04).

    **Column names: written against crypto-spot kline data's RAW Title-Case
    `Close`.** `Dataset.get_lazyframe()` returns whatever column names the
    underlying store holds and performs no per-market normalization -- unlike
    `_to_kunquant()`, which each `Dataset` subclass overrides to rename its
    columns before feeding the compiled-graph backend. So this factor is not
    portable to a market whose store spells the close price differently
    (US equities expose lowercase names plus an `adj*` group) without a
    per-market branch. Giving `get_lazyframe()` a market-agnostic naming
    contract is 03-RESEARCH.md Open Question 2, deliberately out of scope for
    Phase 3 and the natural follow-up once more Polars factors exist.

    **Horizon is config-driven.** `n` is read from `config.kwargs["n"]`
    (default 20), mirroring how `label/spot.py:SpotReturn` reads
    `n_forward_periods`, so the same class reproduces a different signal from
    a different config file and nothing about the horizon is hardcoded in
    source. The emitted factor column is named `momentum_{n}`.
    """

    def __init__(self, factor_config: PolarsFactorConfig):
        super().__init__(factor_config)

    @property
    def horizon(self) -> int:
        kwargs = self.config.kwargs or {}
        return kwargs.get("n", _DEFAULT_HORIZON)

    def _get_factor_lazyframe(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        n = self.horizon
        factor_name = f"momentum_{n}"
        close = pl.col("Close")
        return (
            # Sorting first is what makes the row-wise shift below mean
            # "n bars earlier for THIS symbol"; .over("symbol") keeps the
            # window per-symbol rather than running across the whole panel.
            lf.sort(["symbol", "timestamp"])
            .with_columns(
                (close / close.shift(n).over("symbol") - 1.0).alias(
                    factor_name
                )
            )
            # D-04: only timestamp/symbol/factor columns may leave the hook --
            # a surviving raw price or volume column would be persisted as if
            # it were a factor.
            .select(["timestamp", "symbol", factor_name])
        )

    def _get_labels(self, data: xr.Dataset) -> NoReturn:
        raise RuntimeError(f"{__class__.__name__} does not support get_label()")

    def _get_features(self, data: xr.Dataset) -> xr.Dataset:
        return data
