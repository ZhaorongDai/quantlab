"""The Trade bar panel built from Massive's daily trade files, on the permaticker axis.

``MassiveTradeBarDataset`` reads the raw tier the Massive client writes
(``quantlab.dataset.massive.raw``) one trading day at a time, maps each
raw ``(date, ticker)`` to its Sharadar permaticker, resamples the trades
with ``quantlab.dataset.massive.resample.TradeBarResampler``, and returns
the dense ``(timestamp, symbol)`` panel the store holds (ADR 0030).

- **Sessions** are the XNYS calendar's, half days included; the window is
  ``session_start`` to ``session_end``, the regular session by default. A
  raw day that is not a session is refused.
- **The symbol axis is the permaticker** (ADR 0023). A ticker is mapped as
  traded on its day (``PermatickerResolver.resolve_on``) through the
  Sharadar raw tier in ``sharadar_dir``: SEP's tickers first, then SFP's,
  so stocks and funds are both kept. A ticker that maps to no permaticker
  is logged, counted and dropped. If two tickers of one day map to one
  permaticker, the one with more trades is kept and the other is counted
  as unmapped.
- **A cell without an eligible trade** has NaN prices and zero ``volume``
  and ``n_trades``; a permaticker on the axis that did not trade on a day
  is zero volume on every bar of it.
- **Per-day statistics** (trades in, dropped by each rule, outside the
  session, unmapped tickers) are written to the JSON sidecar
  ``<store>.massive_stats.json`` (``stats_path``), merged day by day.

Convert one day per window (``granularity="day"``): a day of the whole
market is tens of millions of trades.

Examples
--------
Needs a trade file and a condition table under ``raw_data_dir_path``, and
Sharadar's TICKERS and ACTIONS under ``sharadar_dir``::

    config = MassiveTradeBarsDatasetConfig(
        zarr_file_path="/data/zarrs/massive_trade_bars_1m.zarr",
        raw_data_dir_path="/data/downloads/massive",
        sharadar_dir="/data/downloads/sharadar",
        start_date="2024-11-29",
        end_date="2024-11-29",
    )
    MassiveTradeBarDataset(config).from_raw_data_chunked(granularity="day")
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import xarray as xr
from loguru import logger

from quantlab.dataset._support.session_calendar import XnysSessionCalendar
from quantlab.dataset.config import DatasetConfig, MassiveTradeBarsDatasetConfig
from quantlab.dataset.massive.raw import (
    latest_conditions,
    raw_days,
    raw_file,
    read_conditions,
    read_trades,
)
from quantlab.dataset.massive.resample import (
    TRADE_BAR_VARIABLES,
    TRADE_STATS_COUNTS,
    TradeBarResampler,
)
from quantlab.dataset.sharadar.permatickers import PermatickerResolver
from quantlab.dataset.stock import StockDataset
from quantlab.utils.atomic import write_json_atomically
from quantlab.utils.timer import Timer

#: Appended to the store path to name the per-day statistics sidecar.
STATS_SUFFIX = ".massive_stats.json"

#: The Sharadar tables whose tickers map a raw ticker, in order: stocks, then funds.
MAPPING_TABLES = ("sep", "sfp")

#: Variables a cell without a trade holds as 0 rather than NaN.
_COUNT_VARIABLES = ("volume", "n_trades")


@dataclass(frozen=True)
class _Day:
    """One converted trading day: its bars on the permaticker axis and its statistics."""

    bars: pl.DataFrame
    stats: dict


class MassiveTradeBarDataset(StockDataset):
    """Dense Trade bar panel on the permaticker axis; see the module docstring.

    Parameters
    ----------
    dataset_config : MassiveTradeBarsDatasetConfig
        The raw tier, the Sharadar raw tier, the store, the date range, the
        bar size and the session window.

    Attributes
    ----------
    last_stats : dict or None
        The statistics sidecar content after the last day converted.

    Examples
    --------
    >>> ds = MassiveTradeBarDataset(config).from_raw_data_chunked(granularity="day")
    >>> panel = ds.panel("2024-11-29", "2024-11-30")
    >>> sorted(panel.data_vars)
    ['close', 'high', 'low', 'n_trades', 'open', 'volume']
    """

    #: The config class used to rebuild the dataset from a saved ``config.json``.
    config_cls = MassiveTradeBarsDatasetConfig

    #: The data type the panel is built from.
    DATA_TYPE = "trades"

    last_stats: dict | None = None

    def _normalize_config(self, config: DatasetConfig) -> MassiveTradeBarsDatasetConfig:
        """Normalise as the base class does, then refuse ``symbols`` and check the session window.

        Raises
        ------
        TypeError
            If ``config`` is not a ``MassiveTradeBarsDatasetConfig``.
        ValueError
            If ``symbols`` is set, or the session window is malformed.
        """
        config = super()._normalize_config(config)
        if not isinstance(config, self.config_cls):
            raise TypeError(f"{self.class_name} needs a {self.config_cls.__name__}, got {type(config).__name__}.")
        if config.symbols is not None:
            raise ValueError(
                f"{self.class_name}: config.symbols is not selectable; got {config.symbols!r}. "
                f"The panel's symbol axis is the Sharadar permaticker, and raw tickers are "
                f"mapped to it day by day."
            )
        TradeBarResampler(config.bar_interval)
        XnysSessionCalendar(config.session_start, config.session_end)
        return config

    def _on_config_installed(self) -> None:
        """Create the calendar and resampler, and clear the per-day cache and the resolvers."""
        self._calendar = XnysSessionCalendar(self.config.session_start, self.config.session_end)
        self._resampler = TradeBarResampler(self.config.bar_interval)
        self._days: dict[date, _Day] = {}
        self._resolvers: list[PermatickerResolver] | None = None

    @property
    def _raw_data_type(self) -> str:
        """Return ``DATA_TYPE``."""
        return self.DATA_TYPE

    @property
    def stats_path(self) -> Path:
        """Return the path of the per-day statistics sidecar beside the store.

        Examples
        --------
        >>> ds.stats_path.name
        'massive_trade_bars_1m.zarr.massive_stats.json'
        """
        return Path(f"{self.config.zarr_file_path}{STATS_SUFFIX}")

    def has_raw_data(self) -> bool:
        """Return whether the raw tier holds at least one trade file.

        Examples
        --------
        >>> MassiveTradeBarDataset(config).has_raw_data()
        True
        """
        return bool(raw_days(self.config.raw_data_dir_path, self.DATA_TYPE))

    # -- days -------------------------------------------------------------------

    def _days_in_range(self) -> list[date]:
        """Return the raw trade days inside the configured range, ascending."""
        start = None if self.config.start_date is None else date.fromisoformat(self.config.start_date[:10])
        end = None if self.config.end_date is None else date.fromisoformat(self.config.end_date[:10])
        return [
            day
            for day in raw_days(self.config.raw_data_dir_path, self.DATA_TYPE)
            if (start is None or start <= day) and (end is None or day <= end)
        ]

    def _resolver_chain(self) -> list[PermatickerResolver]:
        """Return the SEP and SFP resolvers over ``sharadar_dir``, built once."""
        if self._resolvers is None:
            self._resolvers = [PermatickerResolver(self.config.sharadar_dir, code) for code in MAPPING_TABLES]
        return self._resolvers

    def _map(self, day: date, tickers: list[str]) -> tuple[dict[str, int], dict[str, str]]:
        """Map ``tickers`` as traded on ``day``: SEP first, then SFP for the rest."""
        mapped: dict[str, int] = {}
        left = list(tickers)
        reasons: dict[str, str] = {}
        for resolver in self._resolver_chain():
            if not left:
                break
            found, unresolved = resolver.resolve_on(day, left)
            mapped.update(found)
            reasons = {ticker: reason for ticker, reason in unresolved.items() if ticker not in mapped}
            left = [ticker for ticker in left if ticker not in mapped]
        return mapped, {ticker: reasons.get(ticker, "no permaticker") for ticker in left}

    def _day(self, day: date) -> _Day:
        """Read, map and resample one trading day, once per instance."""
        if day in self._days:
            return self._days[day]
        with Timer(f" {self.class_name}: {day}"):
            trades = read_trades(raw_file(self.config.raw_data_dir_path, self.DATA_TYPE, day))
            counts = trades.group_by("ticker").len().rename({"len": "trades"})
            mapped, reasons = self._map(day, counts["ticker"].to_list())
            mapping = pl.DataFrame(
                {"ticker": list(mapped), "permaticker": list(mapped.values())},
                schema={"ticker": pl.String, "permaticker": pl.Int64},
            )
            # Two tickers of one security on one day: keep the busier one.
            kept = (
                counts.join(mapping, on="ticker", how="inner")
                .sort(["permaticker", "trades", "ticker"], descending=[False, True, False])
                .unique("permaticker", keep="first", maintain_order=True)
            )
            for ticker in mapping.filter(~pl.col("ticker").is_in(kept["ticker"].implode()))["ticker"]:
                reasons[ticker] = f"another ticker of permaticker {mapped[ticker]} traded more that day"
            sessions = self._calendar.session_bounds([day])
            records = (
                trades.join(kept.select("ticker", "permaticker"), on="ticker", how="inner")
                .drop("ticker")
                .rename({"permaticker": "symbol"})
                .with_columns(pl.col("symbol").cast(pl.String), pl.lit(day).alias("date"))
            )
            bars, stats = self._resampler.resample_with_stats(
                records, read_conditions(latest_conditions(self.config.raw_data_dir_path)), sessions
            )
        trade_counts = dict(counts.iter_rows())
        unmapped = {
            ticker: {"trades": int(trade_counts[ticker]), "reason": reason} for ticker, reason in sorted(reasons.items())
        }
        if unmapped:
            logger.warning(
                f"{self.class_name}: {day}: {len(unmapped)} ticker(s) map to no permaticker and are "
                f"dropped ({sum(v['trades'] for v in unmapped.values())} trade(s)), first "
                f"{list(unmapped)[:5]}; see {str(self.stats_path)!r}."
            )
        day_stats = {
            **{name: int(stats[name].sum()) for name in TRADE_STATS_COUNTS},
            "unmapped_tickers": len(unmapped),
            "unmapped_trades": sum(v["trades"] for v in unmapped.values()),
            "unmapped": unmapped,
        }
        day_stats["trades_in"] += day_stats["unmapped_trades"]
        result = _Day(bars=bars.with_columns(pl.col("symbol").cast(pl.Int64)), stats=day_stats)
        self._days[day] = result
        return result

    def _write_stats(self, days: dict[date, dict]) -> None:
        """Merge the statistics of ``days`` into the sidecar."""
        payload = {"days": {}}
        if self.stats_path.exists():
            payload = json.loads(self.stats_path.read_text(encoding="utf-8"))
        payload["settings"] = {
            "bar_interval": self.config.bar_interval,
            "session_start": self.config.session_start,
            "session_end": self.config.session_end,
        }
        payload["days"].update({day.isoformat(): stats for day, stats in days.items()})
        payload["days"] = dict(sorted(payload["days"].items()))
        write_json_atomically(self.stats_path, payload)
        self.last_stats = payload

    # -- the panel -------------------------------------------------------------

    def _raw_axes_in_range(self) -> tuple[list[int], pd.DatetimeIndex]:
        """Return the permatickers and bar labels of the configured range.

        Every day of the range is converted (and kept for the windows), so
        the axis is every permaticker that traded on one of them.

        Raises
        ------
        ValueError
            If the range has no raw trade day, or no ticker maps.
        """
        days = self._days_in_range()
        if not days:
            raise ValueError(
                f"{self.class_name}: no Massive trade file under "
                f"{str(Path(self.config.raw_data_dir_path) / self.DATA_TYPE)!r} in "
                f"[{self.config.start_date}, {self.config.end_date}]. Download it first."
            )
        symbols: set[int] = set()
        for day in days:
            symbols.update(self._day(day).bars["symbol"].unique().to_list())
        if not symbols:
            raise ValueError(
                f"{self.class_name}: no raw ticker of {days[0]}..{days[-1]} maps to a permaticker "
                f"through {self.config.sharadar_dir!r}; refusing to write an empty panel."
            )
        labels = self._resampler.labels(self._calendar.session_bounds(days))
        return sorted(symbols), pd.DatetimeIndex(labels["timestamp"].to_list())

    def _raw_data_to_xr_window(self, start_date, end_date, symbols: list[int] | None = None) -> xr.Dataset:
        """Build the bars of the days whose labels fall in the window, densely, on ``symbols``.

        Parameters
        ----------
        start_date, end_date : date-like
            First and last bar label to include, inclusive.
        symbols : list of int, optional
            The permaticker axis; ``None`` uses ``_raw_axes_in_range``'s.

        Returns
        -------
        xr.Dataset
            ``TRADE_BAR_VARIABLES`` as float64 on ``(timestamp, symbol)``,
            ``symbol`` being int64.
        """
        if symbols is None:
            symbols, _ = self._raw_axes_in_range()
        symbols = [int(symbol) for symbol in symbols]
        start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
        labels = self._resampler.labels(self._calendar.session_bounds(self._days_in_range())).filter(
            pl.col("timestamp").is_between(start, end, closed="both")
        )
        days = labels["date"].unique().sort().to_list()
        frames = [self._day(day).bars.drop("date") for day in days]
        self._write_stats({day: self._day(day).stats for day in days})

        grid = labels.select("timestamp").join(
            pl.DataFrame({"symbol": symbols, "_position": range(len(symbols))}, schema={"symbol": pl.Int64, "_position": pl.Int64}),
            how="cross",
        )
        if frames:
            grid = grid.join(pl.concat(frames), on=["timestamp", "symbol"], how="left")
        else:
            grid = grid.with_columns(pl.lit(None, dtype=pl.Float64).alias(name) for name in TRADE_BAR_VARIABLES)
        grid = grid.with_columns(pl.col(name).fill_null(0) for name in _COUNT_VARIABLES).sort("timestamp", "_position")

        timestamps = labels["timestamp"].to_list()
        shape = (len(timestamps), len(symbols))
        variables = {
            name: (
                ("timestamp", "symbol"),
                grid[name].cast(pl.Float64).fill_null(np.nan).to_numpy().reshape(shape),
            )
            for name in TRADE_BAR_VARIABLES
        }
        return xr.Dataset(
            variables,
            coords={"timestamp": pd.DatetimeIndex(timestamps).values, "symbol": np.array(symbols, dtype=np.int64)},
        )

    def _raw_data_to_xr(self) -> xr.Dataset:
        """Build the whole configured range as one window."""
        symbols, timestamps = self._raw_axes_in_range()
        return self._raw_data_to_xr_window(timestamps[0], timestamps[-1], symbols=symbols)

    def _clean(self, data: xr.Dataset) -> xr.Dataset:
        """Return the panel as built: the resampler's rules are its cleaning."""
        return data

    def _widen_fill_values(self) -> dict:
        """Return 0 for the counts of a permaticker added to an existing store; prices stay NaN."""
        return {name: 0.0 for name in _COUNT_VARIABLES}
