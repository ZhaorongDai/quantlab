"""Point-in-time Fama-French 48 industry of each Sharadar security, keyed by permaticker.

TICKERS gives each security's *current* SIC code (``siccode``); its
``famaindustry``, ``sicindustry`` and ``sector`` are current snapshots too,
and none of them is used here. The SIC history is in ACTIONS: a change is a
pair of rows on one date, ``sicchangefrom`` (the old code) and
``sicchangeto`` (the new one). The history is rebuilt by walking back from
the current code: on a day, a security's SIC is the ``sicchangefrom`` of its
first change dated after that day, or the current code when no change is
later. A change dated on a weekend or holiday therefore shows from the next
trading day. Every change in ACTIONS is walked through, even one dated after
the days the panel holds: TICKERS is pulled whole, so its current code
already reflects it. ``sicchangeto`` is not read. On the 2026-10-05 pull it disagrees
with the next change's ``sicchangefrom`` for 19 of 2,927 changes, and the
last change's ``sicchangeto`` differs from the current ``siccode`` for 36 of
2,575 securities. In both cases the walk back keeps the code that held
afterwards, the current one.

``SharadarIndustryDataset`` classifies each day's SIC into the Fama-French
48 industries (``quantlab.dataset._support.ff48``; a SIC inside no range is
48, "Other"), merges the thin industries named in
``config.industry_merge``, and stores the code as one float variable,
``industry``, on SEP's trading days. A security's industry is shown from its
TICKERS ``firstpricedate`` to its ``lastpricedate`` and is NaN outside them,
or where its SIC is unknown. The ``industry`` variable's ``names`` attribute maps
each code that can appear (as a string) to French's short name.

ACTIONS is keyed by the same current ticker as SEP, so its rows are mapped
through SEP's TICKERS rows; a row of a ticker outside SEP is another security
and is dropped. Two changes of one security on one date are refused. The
store grows with ``update()``: new days are appended and a stored day is
never rewritten.

Examples
--------
Build the store and read a year::

    config = SharadarIndustryConfig(
        zarr_file_path="/data/quantlab/zarrs/sharadar_industry_1d.zarr",
        raw_data_dir_path="/data/quantlab/downloads/sharadar",
    )
    SharadarIndustryDataset(config).update()
    panel = SharadarIndustryDataset(config).panel("2024-01-02", "2024-12-31")
    panel["industry"].attrs["names"]["44"]  # 'Banks'
"""

from __future__ import annotations

import dataclasses
from datetime import date

import numpy as np
import pandas as pd
import polars as pl
import xarray as xr

from quantlab.dataset._support.ff48 import (
    apply_merge,
    check_merge,
    ff48_codes,
    short_names,
)
from quantlab.dataset.base import BaseDataset
from quantlab.dataset.config import DatasetConfig, SharadarIndustryConfig
from quantlab.dataset.sharadar.tables import (
    permaticker_mapping,
    raw_through,
    scan_raw_table,
    table,
    trading_days,
)
from quantlab.dataset.sharadar.universe import normalize_universe, universe
from quantlab.utils.symbol_axis import sort_symbol_axis
from quantlab.utils.timer import Timer

#: The ACTIONS type holding a SIC change's old code.
SIC_CHANGE_FROM = "sicchangefrom"

#: Name of the panel's one variable.
VARIABLE = "industry"

#: Number of offending keys an error message lists.
_ERROR_SAMPLE = 5

#: Stand-ins for an open end of a spell, as numpy days.
_EARLIEST = np.datetime64("0001-01-01", "D")
_LATEST = np.datetime64("9999-12-31", "D")
_ONE_DAY = np.timedelta64(1, "D")


class SharadarIndustryDataset(BaseDataset):
    """Daily panel of each security's point-in-time Fama-French 48 industry code.

    Parameters
    ----------
    dataset_config : SharadarIndustryConfig
        ``raw_data_dir_path`` is ``<download-dir>/sharadar``, holding the
        ``tickers``, ``actions`` and ``sep`` (the calendar) raw tables.

    Examples
    --------
    With ``config`` as in the module example::

        ds = SharadarIndustryDataset(config).update()
        list(ds.panel("2024-01-02", "2024-01-05").data_vars)  # ['industry']
    """

    #: The config class used to rebuild this dataset from a saved config.
    config_cls = SharadarIndustryConfig

    def _normalize_config(self, config: DatasetConfig) -> SharadarIndustryConfig:
        """Normalise as the base class does, then check the table, universe and merge fields.

        Raises
        ------
        TypeError
            If ``config`` is not a ``SharadarIndustryConfig``.
        ValueError
            If ``table`` is not ``"sep"``, a universe field is invalid
            (``quantlab.dataset.sharadar.universe.normalize_universe``) or
            ``industry_merge`` is refused (``quantlab.dataset._support.ff48.check_merge``).

        Examples
        --------
        A merge naming a code outside 1..48 is refused::

            SharadarIndustryDataset(dataclasses.replace(config, industry_merge=((27, 49),)))
            # ValueError: SharadarIndustryDataset: the industry merge mapping names unknown code(s) [49]; ...
        """
        config = super()._normalize_config(config)
        if not isinstance(config, SharadarIndustryConfig):
            raise TypeError(
                f"{self.class_name} needs a SharadarIndustryConfig, got "
                f"{type(config).__name__}."
            )
        if config.table != "sep":
            raise ValueError(f"{self.class_name}: table must be 'sep', got {config.table!r}.")
        config = normalize_universe(config, self.class_name)
        return dataclasses.replace(
            config, industry_merge=check_merge(config.industry_merge, self.class_name)
        )

    def _on_config_installed(self) -> None:
        """Drop what was cached for the previous config."""
        self._spells_cache: pl.DataFrame | None = None
        self._days_cache: pl.Series | None = None

    # -- inputs ---------------------------------------------------------------

    def _raw_through(self) -> date:
        """Return the last day ACTIONS and the SEP calendar are complete through."""
        return raw_through(self.config.raw_data_dir_path, ("sep", "actions"))

    def _trading_days(self) -> pl.Series:
        """Return SEP's trading days up to ``_raw_through()``, sorted, as ns datetimes (cached)."""
        if self._days_cache is None:
            self._days_cache = trading_days(self.config.raw_data_dir_path, self._raw_through())
        return self._days_cache

    def _days_in(self, start_date, end_date) -> pl.Series:
        """Return the trading days in the configured window and ``start_date``..``end_date``."""
        first = max(pd.Timestamp(start_date), pd.Timestamp(self.config.start_date))
        last = min(pd.Timestamp(end_date), pd.Timestamp(self.config.end_date))
        days = self._trading_days()
        return days.filter(
            days.is_between(pl.lit(first.to_datetime64()), pl.lit(last.to_datetime64()))
        )

    def _securities(self) -> pl.DataFrame:
        """Return each kept security's current SIC and price dates, from SEP's TICKERS rows."""
        return (
            scan_raw_table(self.config.raw_data_dir_path, "tickers")
            .filter(
                pl.col("table").is_in(table("sep").tickers_labels)
                & pl.col("permaticker").is_in(universe(self.config))
            )
            .select("permaticker", "siccode", "firstpricedate", "lastpricedate")
            .unique()
            .collect()
        )

    def _changes(self, securities: pl.DataFrame) -> pl.DataFrame:
        """Return each kept security's SIC changes: ``permaticker``, ``date`` and the old ``sic``.

        Raises
        ------
        ValueError
            If a security has two changes on one date.
        """
        root = self.config.raw_data_dir_path
        changes = (
            scan_raw_table(root, "actions")
            # Every change, however late: the current code walked back from
            # is TICKERS', which already holds them all.
            .filter(pl.col("action") == SIC_CHANGE_FROM)
            .select("date", "ticker", pl.col("value").alias("sic"))
            .collect()
            .join(permaticker_mapping(root, "sep"), on="ticker", how="inner")
            .join(securities.select("permaticker"), on="permaticker", how="semi")
        )
        repeated = (
            changes.group_by("permaticker", "date").len().filter(pl.col("len") > 1)
            .sort("permaticker", "date")
        )
        if repeated.height:
            sample = [
                f"{row['permaticker']}@{row['date']}"
                for row in repeated.head(_ERROR_SAMPLE).to_dicts()
            ]
            raise ValueError(
                f"{self.class_name}: {repeated.height} (permaticker, date) "
                f"pair(s) have several SIC changes in ACTIONS, first {sample}; "
                f"refusing rather than choosing one."
            )
        return changes.select("permaticker", "date", "sic")

    def _spells(self) -> pl.DataFrame:
        """Return every security's industry spells (cached).

        Returns
        -------
        pl.DataFrame
            ``symbol`` (the permaticker), ``start`` (inclusive, null when
            open), ``end`` (exclusive, null when open), ``industry`` (the
            merged code, NaN for an unknown SIC), ``first`` and ``last``
            (the security's first and last price dates).

        Raises
        ------
        ValueError
            If a permaticker has two SEP TICKERS rows, or two SIC changes
            on one date.
        """
        if self._spells_cache is not None:
            return self._spells_cache
        with Timer(f"{self.class_name}: rebuild SIC history"):
            securities = self._securities()
            twice = securities.filter(pl.col("permaticker").is_duplicated())
            if twice.height:
                raise ValueError(
                    f"{self.class_name}: permaticker(s) "
                    f"{twice.get_column('permaticker').unique().sort().head(_ERROR_SAMPLE).to_list()} "
                    f"have several SEP rows in TICKERS; re-pull TICKERS."
                )
            changes = self._changes(securities).sort("permaticker", "date")
            # Before each change its old code held, back to the previous change.
            before = changes.select(
                "permaticker",
                pl.col("date").shift(1).over("permaticker").alias("start"),
                pl.col("date").alias("end"),
                pl.col("sic").cast(pl.Float64),
            )
            # From the last change on (or always, without one) the current code holds.
            current = securities.join(
                changes.group_by("permaticker").agg(pl.col("date").max().alias("start")),
                on="permaticker",
                how="left",
            ).select(
                "permaticker",
                "start",
                pl.lit(None, dtype=pl.Date).alias("end"),
                pl.col("siccode").cast(pl.Float64).alias("sic"),
            )
            spells = pl.concat([before, current]).join(
                securities.select(
                    "permaticker",
                    pl.col("firstpricedate").alias("first"),
                    pl.col("lastpricedate").alias("last"),
                ),
                on="permaticker",
            )
            sic = spells.get_column("sic").fill_null(np.nan).to_numpy()
            industry = apply_merge(ff48_codes(sic), self.config.industry_merge)
            spells = spells.select(
                pl.col("permaticker").alias("symbol"), "start", "end", "first", "last"
            ).with_columns(pl.Series(VARIABLE, industry))
        self._spells_cache = spells
        return spells

    # -- axes and windows -----------------------------------------------------

    def _listed_in(self, days: pl.Series) -> list[int]:
        """Return the permatickers priced at some point in ``days``' span, sorted."""
        securities = self._spells().select("symbol", "first", "last").unique()
        if days.len():
            securities = securities.filter(
                (pl.col("first").is_null() | (pl.col("first") <= days.max().date()))
                & (pl.col("last").is_null() | (pl.col("last") >= days.min().date()))
            )
        return sort_symbol_axis(securities.get_column("symbol").to_list())

    def _raw_axes_in_range(self) -> tuple[list, pd.DatetimeIndex]:
        """Return the permatickers priced in the window, and its trading days.

        Raises
        ------
        ValueError
            If the window holds no trading day or no security is priced in it.
        """
        days = self._days_in(self.config.start_date, self.config.end_date)
        symbols = self._listed_in(days) if days.len() else []
        if not symbols:
            raise ValueError(
                f"{self.class_name}: no security priced in "
                f"[{self.config.start_date}, {self.config.end_date}] for the "
                f"configured universe."
            )
        return symbols, pd.DatetimeIndex(days.to_list())

    def _raw_data_to_xr_window(
        self, start_date, end_date, symbols: list[int] | None = None
    ) -> xr.Dataset:
        """Return one window of the panel: every trading day by every symbol.

        Parameters
        ----------
        start_date, end_date : date-like
            The window, both inclusive.
        symbols : list of int, optional
            Permatickers to build; ``None`` is every one priced in the window.
        """
        days = self._days_in(start_date, end_date)
        if symbols is None:
            symbols = self._listed_in(days)
        symbols = [int(s) for s in symbols]
        stamps = days.to_numpy().astype("datetime64[D]")
        spells = self._spells().filter(pl.col("symbol").is_in(symbols))
        values = np.full((len(stamps), len(symbols)), np.nan)
        if len(stamps) and spells.height:
            column = {symbol: index for index, symbol in enumerate(symbols)}
            low = np.maximum(
                _days(spells.get_column("start"), _EARLIEST),
                _days(spells.get_column("first"), _EARLIEST),
            )
            # ``last`` is a priced day, so the spell runs through it.
            high = np.minimum(
                _days(spells.get_column("end"), _LATEST),
                _days(spells.get_column("last"), _LATEST - _ONE_DAY) + _ONE_DAY,
            )
            rows_from = np.searchsorted(stamps, low, side="left")
            rows_to = np.searchsorted(stamps, high, side="left")
            for symbol, first, stop, code in zip(
                spells.get_column("symbol").to_list(),
                rows_from,
                rows_to,
                spells.get_column(VARIABLE).to_numpy(),
                strict=True,
            ):
                if first < stop:
                    values[first:stop, column[symbol]] = code
        return xr.Dataset(
            {
                VARIABLE: (
                    ("timestamp", "symbol"),
                    values,
                    {"scheme": "Fama-French 48", "names": short_names(self.config.industry_merge)},
                )
            },
            coords={
                "timestamp": pd.DatetimeIndex(days.to_list()),
                "symbol": pd.Index(symbols, dtype="int64"),
            },
        )

    def _raw_data_to_xr(self) -> xr.Dataset:
        """Return the panel for the whole configured window."""
        symbols, _ = self._raw_axes_in_range()
        return self._raw_data_to_xr_window(
            self.config.start_date, self.config.end_date, symbols=symbols
        )

    def _added_symbols_with_raw_history(self, added: list, start, end) -> dict[str, int]:
        """Count each added permaticker's days with an industry between ``start`` and ``end``.

        A security first priced after the store's last day has none, so it
        widens the store.
        """
        if not added:
            return {}
        window = self._raw_data_to_xr_window(start, end, symbols=[int(s) for s in added])
        counts = np.isfinite(window[VARIABLE].values).sum(axis=0)
        return {
            str(symbol): int(count)
            for symbol, count in zip(window["symbol"].values.tolist(), counts, strict=True)
            if count
        }

    def _clean(self, data: xr.Dataset) -> xr.Dataset:
        """Return the panel unchanged: industry codes are not OHLCV market data."""
        return data


def _days(column: pl.Series, missing: np.datetime64) -> np.ndarray:
    """Return a date column as numpy days, ``missing`` where it is null."""
    days = column.cast(pl.Date).to_numpy().astype("datetime64[D]")
    return np.where(np.isnat(days), missing, days)
