"""A factor store's panel read as a dataset's columns.

``FactorStoreDataset`` reads a stored factor panel, such as daily session
statistics of minute bars, by date range and exposes its outputs as panel
variables, optionally under a prefix, so a ``MergedDataset`` puts them
beside a price dataset's columns.
"""

import dataclasses

import numpy as np
import polars as pl
import xarray as xr

from quantlab.core.store_uses import READS, StoreUse
from quantlab.dataset.base import MarketDataset
from quantlab.dataset.config import FactorStoreDatasetConfig


class FactorStoreDataset(MarketDataset):
    """Read a factor store's Zarr panel as panel variables.

    ``panel(start, end)`` opens the store lazily and returns its outputs
    over those dates, each named ``prefix + output``; ``symbols`` and
    ``variables`` select by the panel's (prefixed) names, as on any dataset.
    Merged with a price dataset through ``MergedDataset``, the outputs become
    columns beside the prices; a name both inputs hold is the merge's error,
    so a prefix keeps two stores' outputs of the same name apart.

    The store is read as it is. The dataset never rebuilds or asks the
    factor that wrote it, so it serves any panel on ``(timestamp, symbol)``
    whatever built it, and it is not resampled. A missing value stays NaN.

    **Look-ahead contract.** The dataset shifts nothing: the value stored at
    day t is read as day t's value, beside day t's prices. A store read
    through it must therefore hold, at day t, only what is known at day t's
    close. Aligning the values is the store builder's responsibility.

    Parameters
    ----------
    config : FactorStoreDatasetConfig
        The store's Zarr path and the optional prefix.

    Examples
    --------
    ``stats.zarr`` holds two outputs, ``rvol`` and ``skew``, for ``AAA``
    and ``BBB`` on 2024-01-02 to 2024-01-04; BBB has no ``rvol`` on
    2024-01-03:

    >>> stats = FactorStoreDataset(
    ...     FactorStoreDatasetConfig(zarr_file_path="stats.zarr", prefix="hf_")
    ... )
    >>> panel = stats.panel("2024-01-02", "2024-01-03")
    >>> list(panel.data_vars)
    ['hf_rvol', 'hf_skew']
    >>> panel["hf_rvol"].to_pandas()
    symbol        AAA   BBB
    timestamp
    2024-01-02  0.011  0.02
    2024-01-03  0.013   NaN

    Merged with ``prices``, a ``FrameDataset`` holding ``close`` for the
    same days and symbols, the missing statistic leaves the price alone:

    >>> merged = MergedDataset([prices, stats]).panel("2024-01-02", "2024-01-04")
    >>> list(merged.data_vars)
    ['close', 'hf_rvol', 'hf_skew']
    >>> merged["hf_rvol"].to_pandas()
    symbol        AAA    BBB
    timestamp
    2024-01-02  0.011  0.020
    2024-01-03  0.013    NaN
    2024-01-04  0.012  0.018
    >>> merged["close"].sel(timestamp="2024-01-03", symbol="BBB").item()
    21.0
    """

    # Narrower type annotation for readers and type checkers only.
    config: FactorStoreDatasetConfig

    #: The config class used to rebuild this dataset from a saved config.
    config_cls = FactorStoreDatasetConfig

    def _normalize_config(self, config: FactorStoreDatasetConfig) -> FactorStoreDatasetConfig:
        """Return ``config`` with ``name`` set to this class's import path."""
        return dataclasses.replace(config, name=self.import_path)

    @property
    def store_path(self) -> str:
        """Return the factor store's Zarr path.

        Examples
        --------
        >>> stats.store_path
        'stats.zarr'
        """
        return self.config.zarr_file_path

    def store_uses(self) -> tuple:
        """Return the one store the panel reads: the factor store.

        Returns
        -------
        tuple of StoreUse

        Examples
        --------
        >>> stats.store_uses()
        (StoreUse(path='stats.zarr', relation='reads', handle=None),)
        """
        return (StoreUse(self.store_path, READS),)

    def resample(self, freq: str, how=None):
        """Refuse: a factor store is read on the bars it was written on.

        Raises
        ------
        ValueError
            Always; write a store on the bars wanted instead.

        Examples
        --------
        >>> stats.resample("1w", "last")
        Traceback (most recent call last):
        ValueError: FactorStoreDataset.resample('1w'): a factor store is not resampled; ...
        """
        raise ValueError(
            f"{self.class_name}.resample({freq!r}): a factor store is not "
            f"resampled; it is read on the bars it was written on. Write a "
            f"store on the bars wanted instead."
        )

    def head(self, n: int) -> pl.LazyFrame:
        """Return at most ``n`` rows of the store, its outputs under the prefix.

        Parameters
        ----------
        n : int
            Maximum number of rows.

        Examples
        --------
        >>> stats.head(1).collect_schema().names()
        ['timestamp', 'symbol', 'hf_rvol', 'hf_skew']
        """
        frame = super().head(n)
        names = [c for c in frame.collect_schema().names() if c not in ("timestamp", "symbol")]
        return frame.rename(self._prefixed(names))

    def _resamples_on_read(self) -> bool:
        """Return ``False``: the store is read on its own bars."""
        return False

    def _prefixed(self, names) -> dict[str, str]:
        """Return ``{stored name: panel name}`` for the store's ``names``."""
        return {name: f"{self.config.prefix}{name}" for name in names}

    def _open_bars(self, start, end) -> xr.Dataset:
        """Return the store's bars from ``start`` to ``end`` lazily, under the prefix."""
        data = super()._open_bars(start, end)
        return data.rename(self._prefixed(data.data_vars))

    def _to_kunquant(
        self, data: xr.Dataset, data_columns: tuple[str, ...]
    ) -> tuple[dict, np.ndarray, np.ndarray]:
        """Export the columns under their panel names."""
        return self._kunquant_arrays(data, data_columns)
