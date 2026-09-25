"""N-bar price momentum, the reference factor for the Polars backend.

Most factors in this project are computed with KunQuant, a library that
compiles factor formulas to native code. Polars, a DataFrame library with a
lazy query engine, is a second, batch-only way to write a factor, for
formulas that are easier to express as table operations. ``Momentum`` shows
the whole ``FactorPolars`` contract in one class: a single
``_get_factor_lazyframe`` override that returns a lazy frame with the
``timestamp``, ``symbol`` and factor columns, with the horizon read from the
config. Copy it as a starting point for a new Polars factor.
"""

from typing import NoReturn

import polars as pl
import xarray as xr

from quantlab.base.config import PolarsFactorConfig
from quantlab.base.factor import FactorPolars

_DEFAULT_HORIZON = 20


class Momentum(FactorPolars):
    """N-bar per-symbol price momentum, ``Close_t / Close_{t-n} - 1``.

    Momentum is the return a symbol earned over the last ``n`` bars; it is
    the classic "recent winners keep winning" signal. The class overrides
    only ``_get_factor_lazyframe``, which returns a lazy frame with just
    ``timestamp``, ``symbol`` and the factor column, and lets the inherited
    ``cal()`` collect it into an ``xarray.Dataset``. The horizon ``n`` is
    read from ``config.kwargs["n"]`` (default 20) and the output column is
    named ``momentum_{n}``, so different config files give different signals
    from one class.

    The expression reads the store's ``Close`` column, spelled as the crypto
    spot kline store spells it. ``get_lazyframe()`` does not rename columns,
    so a store that names its close price differently needs its own
    subclass.

    Parameters
    ----------
    factor_config : PolarsFactorConfig
        The Polars factor config. ``kwargs["n"]`` sets the horizon in bars.

    Examples
    --------
    >>> factor = Momentum(PolarsFactorConfig(
    ...     window=20, dataset=dataset, kwargs={"n": 20},
    ...     file_path="momentum.zarr",
    ... ))
    >>> factor.get_factor_names()
    ('momentum_20',)
    >>> panel = factor.cal().get_features()
    """

    def __init__(self, factor_config: PolarsFactorConfig):
        """Initialize the factor; see the class docstring for parameters."""
        super().__init__(factor_config)

    @property
    def horizon(self) -> int:
        """Momentum horizon in bars, from ``config.kwargs["n"]`` (default 20).

        Examples
        --------
        >>> factor.horizon
        20
        """
        kwargs = self.config.kwargs or {}
        return kwargs.get("n", _DEFAULT_HORIZON)

    def _get_factor_lazyframe(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """Return a lazy frame with ``timestamp``, ``symbol`` and ``momentum_{n}``.

        Parameters
        ----------
        lf : pl.LazyFrame
            The dataset's rows, one per ``(timestamp, symbol)``.
        """
        n = self.horizon
        factor_name = f"momentum_{n}"
        close = pl.col("Close")
        return (
            # After sorting, a shift of n rows within each symbol means
            # "n bars earlier for this symbol".
            lf.sort(["symbol", "timestamp"])
            .with_columns(
                (close / close.shift(n).over("symbol") - 1.0).alias(factor_name)
            )
            # Any other column left here would be stored as a factor.
            .select(["timestamp", "symbol", factor_name])
        )

    def _get_labels(self, data: xr.Dataset) -> NoReturn:
        """Raise ``RuntimeError``: this factor produces features, not labels."""
        raise RuntimeError(f"{__class__.__name__} does not support get_label()")

    def _get_features(self, data: xr.Dataset) -> xr.Dataset:
        """Return the computed panel unchanged; no post-processing is needed."""
        return data
