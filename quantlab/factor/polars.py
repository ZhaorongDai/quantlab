"""Polars backend of the factor layer: ``FactorPolars``.

A ``FactorPolars`` is a batch-only factor whose logic is a Polars expression chain. Subclass
it and implement the expressions. Shipped factors are in ``quantlab/factor/predefined``.
"""

from abc import abstractmethod

import polars as pl
import xarray as xr

from quantlab.backend.zarr import XrBackend
from quantlab.factor.config import PolarsFactorConfig
from quantlab.factor.base import Factor
from quantlab.utils.timer import Timer


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

    #: The config class ``from_config`` rebuilds this factor with.
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
