"""StockFit's corporate events as daily panels on the CIK axis.

Two panels read StockFit's raw tier (``quantlab.dataset.stockfit.raw``) and
show on NYSE's trading calendar, one column per company (its CIK):

- ``StockFitCorporateEventsDataset``, events each shown on the day they
  became public and NaN on every other day:

  - ``split_ratio`` and ``split_date``: a split from the company's XBRL
    split history (``/api/company/stock-splits``), on the trading day of its
    ``firstDisclosedAt``, the acceptance day of the first filing that tagged
    it, never on its split ``date``. Filers tag a split in the next 10-Q or
    10-K, weeks to a year after it, so the split is old news on the price by
    then: this is when it became machine-readable from EDGAR. Several splits
    first tagged on one day multiply; ``split_date`` is the latest split's
    own date.
  - ``predecessor_cik`` and ``successor_cik``: a succession from the
    company's profile (``/api/company/details``, Form 8-K12B), on the
    trading day of its ``effectiveDate``: the successor's column shows the
    predecessor's CIK, the predecessor's column the successor's.

- ``StockFitDeclaredEventsDataset``, the splits and dividends declared for
  the coming days, from the forward snapshots each refresh keeps
  (``/api/company/upcoming-stock-splits`` and
  ``/api/earnings/upcoming-dividends``, 28 days ahead). On every trading day
  each company shows the soonest split and the soonest dividend in the
  newest snapshot available that day whose date is that day or later:
  ``declared_split_ratio`` and ``declared_split_date``;
  ``declared_dividend`` (the cash per share going ex that day, summed over
  the company's dividends going ex then), ``declared_dividend_annual`` (the
  indicated annual dividend) and ``declared_dividend_ex_date``. A snapshot
  is available on the day it was pulled when pulled before that session's
  close (13:00 New York time on a half day), otherwise on the next trading
  day. Its rows name tickers, not CIKs: a row's tickers are looked up in the
  operating company list the raw tier held when the snapshot was pulled,
  and a row whose tickers name no company, or more than one, is left out.
  The panel has a history only from the first snapshot on: the vendor keeps
  no archive of past declarations.

No event here adjusts any price, and quantlab takes no price from StockFit:
the ratios and amounts are what the vendor says was declared or tagged.

Each panel is built through the day before the endpoints it reads were
last pulled (``quantlab.dataset.stockfit.available.nyse_calendar``).

Examples
--------
Build the corporate event store and read a company's split disclosures::

    config = StockFitCorporateEventsConfig(
        zarr_file_path="/data/market/stockfit/stockfit_corporate_events_1d/stockfit_corporate_events_1d.zarr",
        raw_data_dir_path="/data/downloads/stockfit",
        start_date="2010-01-04",
    )
    panel = StockFitCorporateEventsDataset(config).update().panel("2020-01-02", "2024-12-31")
    panel["split_ratio"].sel(symbol=1652044).dropna("timestamp")
"""

from __future__ import annotations

from typing import ClassVar

import polars as pl
import xarray as xr

from quantlab.dataset._support.point_in_time import (
    ReleasePanelDataset,
    ReleaseSpec,
    TradingCalendar,
    on_grid,
    to_panel,
)
from quantlab.dataset.config import (
    StockFitCorporateEventsConfig,
    StockFitDeclaredEventsConfig,
    StockFitPanelConfig,
)
from quantlab.dataset.stockfit.available import nyse_calendar, pull_day
from quantlab.dataset.stockfit.raw import (
    COMPANY_DETAILS,
    STOCK_SPLITS,
    SYMBOLS,
    UPCOMING_DIVIDENDS,
    UPCOMING_STOCK_SPLITS,
    WHOLE_LIST,
    entities,
    pulls,
    read_pull,
    read_pull_as_of,
)

__all__ = ["StockFitCorporateEventsDataset", "StockFitDeclaredEventsDataset"]

_DAY = pl.Datetime("ns")


class _StockFitEventPanel(ReleasePanelDataset):
    """A StockFit panel on the CIK axis, on NYSE's calendar through the day before its endpoints were last pulled."""

    #: The raw-tier endpoints read; the calendar ends on the last day all were pulled.
    ENDPOINTS: ClassVar[tuple[str, ...]] = ()
    #: The panel's variables and their attributes, in panel order.
    VARIABLES: ClassVar[dict[str, dict]] = {}
    #: The datetime variables among them.
    DATES: ClassVar[tuple[str, ...]] = ()

    config: StockFitPanelConfig

    def _normalize_config(self, config: StockFitPanelConfig) -> StockFitPanelConfig:
        """Normalise as the base class does, then check the config class.

        Raises
        ------
        TypeError
            If ``config`` is not an instance of the panel's ``config_cls``.
        """
        config = super()._normalize_config(config)
        if not isinstance(config, self.config_cls):
            raise TypeError(f"{self.class_name} needs a {self.config_cls.__name__}, got {type(config).__name__}.")
        return config

    def _trading_calendar(self) -> TradingCalendar:
        """Return NYSE's sessions through the day the endpoints read are complete through (``nyse_calendar``).

        Raises
        ------
        ValueError
            If an endpoint was never pulled into the raw tier.
        """
        if self._calendar_cache is None:
            self._calendar_cache = nyse_calendar(
                self.config.raw_data_dir_path, self.ENDPOINTS,
                start=self.config.start_date, end=self.config.end_date, owner=self.class_name,
            )
        return self._calendar_cache

    def _spec(self) -> ReleaseSpec:
        """Return the variables and their attributes; no staleness."""
        return ReleaseSpec(variables=self.VARIABLES, dates=self.DATES, stamp=None)


class StockFitCorporateEventsDataset(_StockFitEventPanel):
    """Daily panel of StockFit's XBRL splits and successions by CIK, each on the day it became public.

    See the module docstring for the variables and when each event shows.

    Examples
    --------
    With ``config`` a ``StockFitCorporateEventsConfig``::

        StockFitCorporateEventsDataset(config).update().panel("2024-01-02", "2024-12-31")["split_ratio"]
    """

    config_cls = StockFitCorporateEventsConfig

    ENDPOINTS = (COMPANY_DETAILS, STOCK_SPLITS)
    VARIABLES = {
        "split_ratio": {
            "unit": "new shares per old share",
            "description": "Ratio of the XBRL-tagged splits first disclosed on the day, multiplied "
                           "(4.0 a four-for-one split, 0.05 a one-for-twenty reverse split). Not used "
                           "to adjust any price.",
        },
        "split_date": {
            "description": "The split's own date as tagged (usually the ex-date), shown on its first "
                           "disclosure day.",
        },
        "predecessor_cik": {
            "description": "CIK of the entity this company succeeded on the day (Form 8-K12B).",
        },
        "successor_cik": {
            "description": "CIK of the entity that succeeded this company on the day (Form 8-K12B).",
        },
    }
    DATES = ("split_date",)

    def _nothing_shown(self) -> str:
        """Return what an empty window lacks."""
        return "no split or succession is shown"

    def _released(self) -> pl.DataFrame:
        """Return every company's events per trading day with one, placed and combined.

        Returns
        -------
        pl.DataFrame
            ``symbol`` (the CIK), ``available`` and the four variables (null
            where the day has none of one), one row per pair.
        """
        calendar = self._trading_calendar()
        root = self.config.raw_data_dir_path
        splits, links = [], []
        for cik in entities(root, STOCK_SPLITS):
            splits += [
                (int(cik), row.get("firstDisclosedAt"), row.get("date"), row.get("ratio"))
                for row in read_pull(root, STOCK_SPLITS, cik) or ()
            ]
        for cik in entities(root, COMPANY_DETAILS):
            details = read_pull(root, COMPANY_DETAILS, cik) or {}
            for key, variable in (("predecessors", "predecessor_cik"), ("successors", "successor_cik")):
                links += [
                    (int(cik), link.get("effectiveDate"), variable, link.get("cik"))
                    for link in details.get(key) or () if link.get("cik") is not None
                ]
        split_rows = (
            pl.DataFrame(splits, schema={"symbol": pl.Int64, "date": pl.String, "split_date": pl.String,
                                         "split_ratio": pl.Float64}, orient="row")
            .with_columns(pl.col("date", "split_date").str.to_date(strict=False))
            .drop_nulls(["date", "split_date"])
        )
        split_rows = (
            calendar.place(split_rows, drop_earlier=True)
            .group_by("symbol", "available")
            .agg(pl.col("split_ratio").fill_null(float("nan")).product(), pl.col("split_date").max().cast(_DAY))
        )
        link_rows = (
            pl.DataFrame(links, schema={"symbol": pl.Int64, "date": pl.String, "variable": pl.String,
                                        "cik": pl.Float64}, orient="row")
            .with_columns(pl.col("date").str.to_date(strict=False))
            .drop_nulls("date")
        )
        link_rows = (
            calendar.place(link_rows, drop_earlier=True)
            .group_by("symbol", "available")
            .agg(
                pl.col("cik").filter(pl.col("variable") == variable).max().alias(variable)
                for variable in ("predecessor_cik", "successor_cik")
            )
        )
        return (
            split_rows.join(link_rows, on=["symbol", "available"], how="full", coalesce=True)
            .select("symbol", "available", *self.VARIABLES)
            .sort("available", "symbol")
        )

    def _raw_data_to_xr_window(self, start_date, end_date, symbols: list | None = None) -> xr.Dataset:
        """Return one window: each day's events on that day, NaN elsewhere."""
        days, symbols = self._window_axes(start_date, end_date, symbols)
        cells = (
            on_grid(days, symbols)
            .join(self._released_rows(), left_on=["timestamp", "symbol"], right_on=["available", "symbol"],
                  how="left")
            .sort("_day", "_column")
        )
        return to_panel(cells, days, symbols, self.VARIABLES)


def _ticker_ciks(companies: object) -> dict[str, int]:
    """Return each ticker of a company list that names exactly one CIK, and that CIK."""
    found: dict[str, set[int]] = {}
    for row in companies or ():
        if row.get("cik"):
            for ticker in row.get("symbols") or ():
                found.setdefault(ticker, set()).add(int(row["cik"]))
    return {ticker: next(iter(ciks)) for ticker, ciks in found.items() if len(ciks) == 1}


class StockFitDeclaredEventsDataset(_StockFitEventPanel):
    """Daily panel of the splits and dividends StockFit lists as declared, by CIK, from its snapshots.

    See the module docstring for the variables, when a snapshot becomes
    available and how its tickers become CIKs.

    Examples
    --------
    With ``config`` a ``StockFitDeclaredEventsConfig``::

        StockFitDeclaredEventsDataset(config).update().panel("2026-11-02", "2026-11-30")["declared_dividend"]
    """

    config_cls = StockFitDeclaredEventsConfig

    ENDPOINTS = (SYMBOLS, UPCOMING_STOCK_SPLITS, UPCOMING_DIVIDENDS)
    VARIABLES = {
        "declared_split_ratio": {
            "unit": "new shares per old share",
            "description": "Ratio of the soonest declared split not yet executed, as listed that day "
                           "(splits on one date multiplied). Not used to adjust any price.",
        },
        "declared_split_date": {"description": "Execution date of that declared split."},
        "declared_dividend": {
            "unit": "USD per share",
            "description": "Cash per share of the declared dividends going ex soonest, as listed that day, summed.",
        },
        "declared_dividend_annual": {
            "unit": "USD per share",
            "description": "The indicated annual dividend of that declaration.",
        },
        "declared_dividend_ex_date": {"description": "Ex-dividend date of that declared dividend."},
    }
    DATES = ("declared_split_date", "declared_dividend_ex_date")

    #: Each calendar read and the field holding an event's date.
    _CALENDARS: ClassVar[dict[str, str]] = {UPCOMING_STOCK_SPLITS: "splitDate", UPCOMING_DIVIDENDS: "exDate"}

    def _on_config_installed(self) -> None:
        """Drop what was cached for the previous config."""
        super()._on_config_installed()
        self._snapshots_cache: pl.DataFrame | None = None

    def _nothing_shown(self) -> str:
        """Return what an empty window lacks."""
        return "no declared split or dividend is shown"

    def _snapshots(self) -> pl.DataFrame:
        """Return every snapshot shown: ``endpoint``, ``pulled`` and ``available``, the newest per day.

        An empty snapshot is one too: it says nothing is declared.
        """
        if self._snapshots_cache is None:
            root = self.config.raw_data_dir_path
            found = [
                (endpoint, pulled_at, pull_day(pulled_at))
                for endpoint in self._CALENDARS
                for pulled_at in pulls(root, endpoint, WHOLE_LIST)
            ]
            frame = pl.DataFrame(
                found, schema={"endpoint": pl.String, "pulled": pl.Datetime("us", "UTC"), "_day": pl.Date},
                orient="row",
            )
            self._snapshots_cache = (
                self._trading_calendar().place(frame, on="_day", drop_earlier=True)
                .sort("pulled")
                .unique(["endpoint", "available"], keep="last")
                .drop("_day")
                .sort("available")
            )
        return self._snapshots_cache

    def _released(self) -> pl.DataFrame:
        """Return every row of the snapshots shown, on its CIK.

        Returns
        -------
        pl.DataFrame
            ``endpoint``, ``pulled``, ``symbol`` (the CIK), ``available``
            (the snapshot's), ``event_date``, ``value`` (the ratio, or the
            cash per share) and ``annual`` (the indicated annual dividend),
            sorted by ``available``.
        """
        root = self.config.raw_data_dir_path
        records = []
        for endpoint, pulled_at in self._snapshots().select("endpoint", "pulled").iter_rows():
            tickers = _ticker_ciks(read_pull_as_of(root, SYMBOLS, "operating", pulled_at))
            for row in read_pull(root, endpoint, WHOLE_LIST, pulled_at):
                ciks = {tickers[t] for t in row.get("symbols") or () if t in tickers}
                if len(ciks) != 1 or row.get("currency", "USD") != "USD":
                    continue
                value = row.get("ratio") if endpoint == UPCOMING_STOCK_SPLITS else row.get("dividendPerShare")
                records.append((endpoint, pulled_at, ciks.pop(), row.get(self._CALENDARS[endpoint]), value,
                                row.get("indicatedAnnualDividend")))
        rows = pl.DataFrame(
            records,
            schema={"endpoint": pl.String, "pulled": pl.Datetime("us", "UTC"), "symbol": pl.Int64,
                    "event_date": pl.String, "value": pl.Float64, "annual": pl.Float64},
            orient="row",
        ).with_columns(pl.col("event_date").str.to_date(strict=False)).drop_nulls("event_date")
        return rows.join(self._snapshots(), on=["endpoint", "pulled"]).sort("available")

    def _declared(self, days: pl.Series, endpoint: str) -> pl.DataFrame:
        """Return, per day and CIK, the rows of the newest snapshot by then that go soonest, that day or later."""
        snapshots = self._snapshots().filter(pl.col("endpoint") == endpoint).select("available", "pulled")
        shown = (
            days.to_frame()
            .sort("timestamp")
            .join_asof(snapshots, left_on="timestamp", right_on="available", strategy="backward")
            .drop_nulls("pulled")
            .drop("available")
            .join(self._released_rows().filter(pl.col("endpoint") == endpoint).drop("available"), on="pulled")
            .filter(pl.col("event_date") >= pl.col("timestamp").dt.date())
        )
        return shown.filter(pl.col("event_date") == pl.col("event_date").min().over("timestamp", "symbol"))

    def _raw_data_to_xr_window(self, start_date, end_date, symbols: list | None = None) -> xr.Dataset:
        """Return one window: each day's soonest declared split and dividend, NaN where none is."""
        days, symbols = self._window_axes(start_date, end_date, symbols)
        splits = self._declared(days, UPCOMING_STOCK_SPLITS).group_by("timestamp", "symbol").agg(
            pl.col("value").fill_null(float("nan")).product().alias("declared_split_ratio"),
            pl.col("event_date").first().cast(_DAY).alias("declared_split_date"),
        )
        dividends = self._declared(days, UPCOMING_DIVIDENDS).group_by("timestamp", "symbol").agg(
            pl.col("value").sum().alias("declared_dividend"),
            pl.col("annual").max().alias("declared_dividend_annual"),
            pl.col("event_date").first().cast(_DAY).alias("declared_dividend_ex_date"),
        )
        cells = (
            on_grid(days, symbols)
            .join(splits, on=["timestamp", "symbol"], how="left")
            .join(dividends, on=["timestamp", "symbol"], how="left")
            .sort("_day", "_column")
        )
        return to_panel(cells, days, symbols, self.VARIABLES)
