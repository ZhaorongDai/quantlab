"""N-bar price momentum, the reference factor for the Polars backend.

``Momentum`` shows the whole ``FactorPolars`` contract in one class: a single
``_get_factor_lazyframe`` override that returns a lazy frame of
``timestamp``, ``symbol`` and the factor column, with the horizon read from
config. Copy it to write a new Polars factor.
"""

from typing import NoReturn

import polars as pl
import xarray as xr

from quantlab.base.config import PolarsFactorConfig
from quantlab.base.factor import FactorPolars

_DEFAULT_HORIZON = 20


class Momentum(FactorPolars):
    """N-bar per-symbol price momentum, ``Close_t / Close_{t-n} - 1``.

    Overrides ``_get_factor_lazyframe`` only, returns a lazy frame with just
    ``timestamp``, ``symbol`` and the factor column, and lets the inherited
    ``cal()`` collect it. The horizon ``n`` is read from
    ``config.kwargs["n"]`` (default 20) and the output column is named
    ``momentum_{n}``, so one class reproduces different signals from
    different config files.

    The expression reads the store's raw ``Close`` column, as the crypto spot
    kline store spells it. ``get_lazyframe()`` performs no column renaming,
    so a store that names its close price differently needs its own subclass.

    Example:
        >>> factor = Momentum(PolarsFactorConfig(
        ...     window=20, dataset=dataset, kwargs={"n": 20},
        ...     file_path="momentum.zarr",
        ... ))
        >>> factor.get_factor_names()
        ('momentum_20',)
        >>> panel = factor.cal().get_features()
    """

    def __init__(self, factor_config: PolarsFactorConfig):
        """Create the factor from a Polars factor config."""
        super().__init__(factor_config)

    @property
    def horizon(self) -> int:
        """Momentum horizon in bars, from ``config.kwargs["n"]`` (default 20).

        Example:
            >>> factor.horizon
            20
        """
        kwargs = self.config.kwargs or {}
        return kwargs.get("n", _DEFAULT_HORIZON)

    def _get_factor_lazyframe(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """Return a lazy frame carrying the ``momentum_{n}`` column."""
        n = self.horizon
        factor_name = f"momentum_{n}"
        close = pl.col("Close")
        return (
            # Sorting first makes the row shift below mean "n bars earlier for
            # this symbol"; .over("symbol") keeps the window per symbol.
            lf.sort(["symbol", "timestamp"])
            .with_columns(
                (close / close.shift(n).over("symbol") - 1.0).alias(factor_name)
            )
            # Only timestamp/symbol/factor columns may leave the hook: a
            # surviving price or volume column would be stored as a factor.
            .select(["timestamp", "symbol", factor_name])
        )

    def _get_labels(self, data: xr.Dataset) -> NoReturn:
        """Raise; this factor produces features only."""
        raise RuntimeError(f"{__class__.__name__} does not support get_label()")

    def _get_features(self, data: xr.Dataset) -> xr.Dataset:
        """Return the computed panel unchanged."""
        return data
