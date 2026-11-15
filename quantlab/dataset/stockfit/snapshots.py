"""StockFit panels that accumulate from snapshots: listing, earnings consensus, IPOs, S-1 offerings, filing calendar, news counts.

StockFit keeps no history of some of what it serves: the company lists say
what each company's status and SFIC sector and industry are today, the
earnings calendar what is expected next, the IPO calendar where each deal
stands. The refresh pulls each of these on every run
(``quantlab.acquisition.stockfit.endpoints``), and the raw tier keeps every
pull that changed something (``quantlab.dataset.stockfit.raw``). These panels
read those pulls back in order:

- each snapshot is placed at the time it was pulled, through the rule every
  StockFit panel uses (``quantlab.dataset.stockfit.available``): pulled
  before a session's close, it is available that trading day, otherwise on
  the next one;
- a cell shows the newest snapshot available that day, so a value shows
  only from the first snapshot that carried it, and a change from the day
  of the snapshot that carried the change;
- a company a market-wide snapshot no longer lists (an earnings date that
  passed, an IPO deal that aged out) shows nothing from that snapshot on.

So the panels start on the day of the first refresh: nothing is backfilled,
and the first snapshot of the company lists gives each company's *current*
status and industry from that day on, not their history.

The panels, all on the CIK axis and NYSE's trading days:

``StockFitListingDataset``
    ``operating`` (1 operating, 0 delisted), ``delisting_day`` and the SFIC
    ``sector`` and ``industry`` (coded; ``names`` maps each code to its
    name), from the company lists (``symbols``, ``company_delisted``).
``StockFitEarningsConsensusDataset``
    ``earnings_date``, ``filing_date``, ``eps_estimate``, ``analyst_count``
    and ``confirmed``, from the earnings calendar (``earnings_upcoming``).
``StockFitIpoCalendarDataset``
    The deal's ``stage`` and terms and dates, from the IPO calendar
    (``ipo_calendar``).
``StockFitOfferingsDataset``
    The company's latest S-1/S-1A terms, from ``recent_offerings``.
``StockFitFilingCalendarDataset``
    The predicted next 10-K, 10-Q and DEF 14A and their confidence, from
    each company's ``filings_calendar``.
``StockFitNewsCountsDataset``
    Headlines per company and day, from each company's ``news``; no text.

Examples
--------
Build the listing store and read a week::

    config = StockFitListingConfig(
        zarr_file_path="/data/stores/stockfit_listing_1d.zarr",
        raw_data_dir_path="/data/downloads/stockfit",
    )
    StockFitListingDataset(config).update()
    panel = StockFitListingDataset(config).panel("2026-11-16", "2026-11-20")
    panel["industry"].attrs["names"]  # code -> SFIC industry name
"""

from __future__ import annotations

import zlib
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import ClassVar
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
import xarray as xr

from quantlab.dataset._support.point_in_time import (
    FilingPanelDataset,
    ReleasePanelDataset,
    ReleaseSpec,
    TradingCalendar,
)
from quantlab.dataset.config import (
    DatasetConfig,
    StockFitEarningsConsensusConfig,
    StockFitFilingCalendarConfig,
    StockFitIpoCalendarConfig,
    StockFitListingConfig,
    StockFitNewsCountsConfig,
    StockFitOfferingsConfig,
)
from quantlab.dataset.stockfit.available import NEW_YORK, nyse_calendar, release_day
from quantlab.dataset.stockfit.congress import TickerHolders, company_rows
from quantlab.dataset.stockfit.raw import (
    COMPANY_DELISTED,
    EARNINGS_UPCOMING,
    FILINGS_CALENDAR,
    IPO_CALENDAR,
    NEWS,
    RECENT_OFFERINGS,
    SYMBOLS,
    WHOLE_LIST,
    entities,
    pulls,
    read_pull,
)
from quantlab.utils.timer import Timer

#: The column holding each snapshot's pull time (UTC).
PULLED = "pulled_at"

#: A float variable, and a date variable (shown as a ns datetime).
NUMBER = pl.Float64
DAY = pl.Datetime("ns")

#: The IPO deal stages, coded in the order a deal moves through them.
IPO_STAGES: dict[str, int] = {"filed": 1, "upcoming": 2, "priced": 3, "withdrawn": 4}

#: The confidence of a predicted filing date, coded.
CONFIDENCE: dict[str, int] = {"low": 1, "medium": 2, "high": 3}

#: The filing forms the filing calendar predicts, by variable suffix.
CALENDAR_FORMS: dict[str, str] = {"10k": "10-K", "10q": "10-Q", "def14a": "DEF 14A"}


def name_code(name: str) -> float:
    """Return the fixed code of a category name: its CRC-32, so a code never depends on the day it was built.

    Examples
    --------
    >>> name_code("Industrials") == name_code("Industrials"), name_code("Industrials") == name_code("Energy")
    (True, False)
    """
    return float(zlib.crc32(name.encode("utf-8")))


def _day(value) -> date | None:
    """Return a ``YYYY-MM-DD`` (or longer ISO) string's day; ``None`` for nothing."""
    return date.fromisoformat(str(value)[:10]) if value else None


def _number(value) -> float | None:
    return None if value is None else float(value)


def _frame(rows: list[dict], schema: Mapping[str, pl.DataType]) -> pl.DataFrame:
    """Return rows (``symbol`` and the variables) as a frame of the given variable schema."""
    full = {"symbol": pl.Int64, **{name: (pl.Date if dtype == DAY else dtype) for name, dtype in schema.items()}}
    return pl.DataFrame(rows, schema=full).with_columns(
        pl.col(name).cast(DAY) for name, dtype in schema.items() if dtype == DAY
    )


def market_states(
    vendor_root: str | Path,
    sources: Sequence[tuple[str, str]],
    combine: Callable[[dict[tuple[str, str], object], datetime], pl.DataFrame],
    schema: Mapping[str, pl.DataType],
) -> pl.DataFrame:
    """Return the state of market-wide snapshots at each of their pull times, with vanished rows nulled.

    The state at a pull time is ``combine`` of the newest pull of every
    source made by then: one row per symbol. A symbol a state lists and the
    next state does not gets a row of nulls at the next state's time, so it
    shows nothing from then on.

    Parameters
    ----------
    vendor_root : str or Path
        ``<download-dir>/stockfit``.
    sources : sequence of (endpoint, entity)
        The snapshots read together.
    combine : callable
        Takes ``{source: records}`` (a source never pulled by then left out)
        and the pull time; returns ``symbol`` and the variables of ``schema``,
        one row per symbol.
    schema : Mapping
        The variables and their types.

    Returns
    -------
    pl.DataFrame
        ``pulled_at`` (UTC), ``symbol`` and the variables, sorted by
        ``pulled_at``.

    Examples
    --------
    >>> market_states("/nonexistent/stockfit", [("symbols", "operating")], lambda found, at: None, {}).height
    0
    """
    stamps = {source: pulls(vendor_root, *source) for source in sources}
    times = sorted({stamp for found in stamps.values() for stamp in found})
    cache: dict[tuple[tuple[str, str], datetime], object] = {}
    states = []
    for index, at in enumerate(times):
        found = {}
        for source, made in stamps.items():
            newest = max((stamp for stamp in made if stamp <= at), default=None)
            if newest is not None:
                if (source, newest) not in cache:
                    cache[(source, newest)] = read_pull(vendor_root, *source, newest)
                found[source] = cache[(source, newest)]
        states.append(combine(found, at).with_columns(pl.lit(index, pl.Int64).alias("_state")))
    if not states:
        return _frame([], schema).with_columns(pl.lit(None, pl.Datetime("us", "UTC")).alias(PULLED)).select(
            PULLED, "symbol", *schema
        )
    shown = pl.concat(states, how="vertical")
    vanished = (
        shown.select("symbol", (pl.col("_state") + 1).alias("_state"))
        .filter(pl.col("_state") < len(times))
        .join(shown.select("symbol", "_state"), on=["symbol", "_state"], how="anti")
    )
    nulls = vanished.with_columns(
        pl.lit(None, dtype).alias(name) for name, dtype in shown.select(*schema).schema.items()
    ).select(shown.columns)
    stamps_frame = pl.DataFrame(
        {"_state": range(len(times)), PULLED: times}, schema={"_state": pl.Int64, PULLED: pl.Datetime("us", "UTC")}
    )
    return (
        pl.concat([shown, nulls], how="vertical")
        .join(stamps_frame, on="_state")
        .sort(PULLED, "symbol")
        .select(PULLED, "symbol", *schema)
    )


def entity_states(
    vendor_root: str | Path,
    endpoint: str,
    rows: Callable[[object, int, datetime], dict],
    schema: Mapping[str, pl.DataType],
) -> pl.DataFrame:
    """Return every pull of a per-company snapshot as one row: ``pulled_at``, ``symbol`` (the CIK) and the variables.

    Examples
    --------
    >>> entity_states("/nonexistent/stockfit", "news", lambda body, cik, at: {}, {"news_count": NUMBER}).height
    0
    """
    found = []
    for entity in entities(vendor_root, endpoint):
        cik = int(entity)
        for at in pulls(vendor_root, endpoint, entity):
            found.append({PULLED: at, "symbol": cik, **rows(read_pull(vendor_root, endpoint, entity, at), cik, at)})
    frame = _frame([{k: v for k, v in row.items() if k != PULLED} for row in found], schema)
    return frame.with_columns(
        pl.Series(PULLED, [row[PULLED] for row in found], dtype=pl.Datetime("us", "UTC"))
    ).select(PULLED, "symbol", *schema)


def place_pulls(states: pl.DataFrame, calendar: TradingCalendar) -> pl.DataFrame:
    """Return snapshot rows at their available trading day, the newest per symbol and day.

    A pull made before a session's close is available that day, at or after
    it on the next trading day (``available.release_day``). A pull past the
    calendar's last day is not available yet and is dropped; one before its
    first day still holds on it.

    Examples
    --------
    Two pulls on Monday 2026-11-16, the second after the close:

    >>> from datetime import UTC, date, datetime
    >>> states = pl.DataFrame({
    ...     PULLED: [datetime(2026, 11, 16, 14, tzinfo=UTC), datetime(2026, 11, 16, 22, tzinfo=UTC)],
    ...     "symbol": [1, 1], "value": [1.0, 2.0],
    ... })
    >>> calendar = TradingCalendar.of_nyse(date(2026, 11, 17), start="2026-11-16", end="2026-11-17")
    >>> place_pulls(states, calendar).select(pl.col("available").dt.date(), "value").rows()
    [(datetime.date(2026, 11, 16), 1.0), (datetime.date(2026, 11, 17), 2.0)]
    """
    dated = release_day(
        states.with_columns(pl.lit(None, pl.Date).alias("_filed")), accepted=PULLED, filed="_filed"
    ).drop("_filed")
    return (
        calendar.place(dated, on="release_day")
        .sort(PULLED)
        .unique(["symbol", "available"], keep="last", maintain_order=True)
        .drop(PULLED, "release_day")
        .sort("available", "symbol")
    )


class StockFitSnapshotPanelDataset(ReleasePanelDataset):
    """A release panel whose rows are snapshots, each placed at its pull time.

    A subclass names ``CONFIG``, ``ENDPOINT`` (the endpoint the calendar is
    dated by), ``SCHEMA`` (each variable's type) and ``ATTRS`` (each
    variable's attributes), and implements ``_states`` (``market_states`` or
    ``entity_states``).
    """

    CONFIG: type[DatasetConfig]
    ENDPOINT: str
    SCHEMA: Mapping[str, pl.DataType]
    ATTRS: Mapping[str, Mapping]

    def _normalize_config(self, config: DatasetConfig) -> DatasetConfig:
        """Normalise as the base class does, then check the config class.

        Raises
        ------
        TypeError
            If ``config`` is not the panel's config class.
        """
        config = super()._normalize_config(config)
        if not isinstance(config, self.CONFIG):
            raise TypeError(f"{self.class_name} needs a {self.CONFIG.__name__}, got {type(config).__name__}.")
        return config

    def _states(self) -> pl.DataFrame:
        """Return ``pulled_at``, ``symbol`` and the variables, one row per snapshot and symbol."""
        raise NotImplementedError

    def _attrs(self) -> dict[str, dict]:
        """Return each variable's attributes."""
        return {name: dict(self.ATTRS.get(name, {})) for name in self.SCHEMA}

    def _spec(self) -> ReleaseSpec:
        """Return the variables; a snapshot is shown until a newer one replaces it."""
        return ReleaseSpec(
            variables=self._attrs(),
            dates=tuple(name for name, dtype in self.SCHEMA.items() if dtype == DAY),
            stamp=None,
        )

    def _nothing_shown(self) -> str:
        """Return what an empty window lacks."""
        return f"no StockFit {self.ENDPOINT!r} snapshot is shown"

    def _trading_calendar(self) -> TradingCalendar:
        """Return NYSE's trading days through the day the endpoint's raw tier is complete through (cached)."""
        if self._calendar_cache is None:
            self._calendar_cache = nyse_calendar(
                self.config.raw_data_dir_path, self.ENDPOINT,
                start=self.config.start_date, end=self.config.end_date, owner=self.class_name,
            )
        return self._calendar_cache

    def _released(self) -> pl.DataFrame:
        """Return each snapshot's rows at the trading day its pull became available."""
        calendar = self._trading_calendar()
        with Timer(f"{self.class_name}: place snapshots"):
            return place_pulls(self._states(), calendar)


def _ticker_cik(holders: TickerHolders, tickers, at: datetime) -> int | None:
    """Return the CIK that held the first resolvable ticker on the pull's New York day."""
    day = at.astimezone(ZoneInfo(NEW_YORK)).date()
    for ticker in tickers:
        if ticker:
            cik = holders.on(ticker, day)
            if cik is not None:
                return cik
    return None


# -- listing ------------------------------------------------------------------


class StockFitListingDataset(StockFitSnapshotPanelDataset):
    """Daily panel of each company's listing status and SFIC sector and industry, from the company-list snapshots.

    Read from the operating and the delisted company lists (``symbols``) and
    the delisted list with its delisting day (``company_delisted``):

    - ``operating``: 1 while StockFit lists the company as operating, 0 once
      it lists it as delisted;
    - ``delisting_day``: the delisting day StockFit gives; NaT before a
      snapshot gave one, or when the day is after the pull (the vendor's
      placeholder ``2201-12-31``);
    - ``sector``, ``industry``: the SFIC sector and industry, coded by
      ``name_code``; ``names`` maps each code to its name.

    The company lists give today's values only: the panel starts at the
    first refresh with each company's status and industry as of then, and a
    change shows from the day of the snapshot that carried it.

    Examples
    --------
    With ``config`` as in the module example::

        StockFitListingDataset(config).panel("2026-11-16", "2026-11-20")["operating"]
    """

    config_cls = StockFitListingConfig
    CONFIG = StockFitListingConfig
    ENDPOINT = SYMBOLS
    SCHEMA: ClassVar[dict[str, pl.DataType]] = {"operating": NUMBER, "delisting_day": DAY, "sector": NUMBER, "industry": NUMBER}
    ATTRS: ClassVar[dict[str, dict]] = {
        "operating": {"description": "1 when StockFit lists the company as operating, 0 when delisted"},
        "delisting_day": {"description": "delisting day StockFit gives"},
        "sector": {"description": "SFIC sector, coded"},
        "industry": {"description": "SFIC industry, coded"},
    }
    SOURCES = ((SYMBOLS, "operating"), (SYMBOLS, "delisted"), (COMPANY_DELISTED, WHOLE_LIST))

    def _on_config_installed(self) -> None:
        """Drop what was cached for the previous config."""
        super()._on_config_installed()
        self._names: dict[str, dict[str, str]] = {"sector": {}, "industry": {}}

    def _attrs(self) -> dict[str, dict]:
        """Return each variable's attributes, the coded ones with their ``names``."""
        self._released_rows()  # the names are collected while the snapshots are read
        attrs = super()._attrs()
        for name, names in self._names.items():
            attrs[name]["names"] = dict(sorted(names.items()))
        return attrs

    def _code(self, variable: str, name) -> float | None:
        if not name:
            return None
        code = name_code(str(name))
        self._names[variable][str(int(code))] = str(name)
        return code

    def _combine(self, found: dict, at: datetime) -> pl.DataFrame:
        listed: dict[int, dict] = {}
        for source in ((SYMBOLS, "operating"), (SYMBOLS, "delisted")):
            for row in found.get(source, ()):
                if row.get("cik"):
                    listed[int(row["cik"])] = {
                        "symbol": int(row["cik"]),
                        "operating": 1.0 if row.get("status") == "operating" else 0.0,
                        "delisting_day": None,
                        "sector": self._code("sector", row.get("sector")),
                        "industry": self._code("industry", row.get("industry")),
                    }
        for row in found.get((COMPANY_DELISTED, WHOLE_LIST), ()):
            day = _day(row.get("delistedAt"))
            if row.get("cik") and int(row["cik"]) in listed and day is not None and day <= at.date():
                listed[int(row["cik"])]["delisting_day"] = day
        return _frame(list(listed.values()), self.SCHEMA)

    def _states(self) -> pl.DataFrame:
        """Return the company lists' state at each of their pulls."""
        return market_states(self.config.raw_data_dir_path, self.SOURCES, self._combine, self.SCHEMA)


# -- market-wide snapshots keyed by ticker ---------------------------------------------


class StockFitTickerSnapshotDataset(StockFitSnapshotPanelDataset):
    """A market-wide snapshot whose rows name a company by ticker, resolved to its CIK on the pull's day.

    Tickers are reused, so a ticker resolves through the current and prior
    tickers of the newest company lists to the one company that may have
    held it on the day of the pull (``congress.TickerHolders``). A row whose
    ticker resolves to no company, or to several, stays in the raw tier only.
    """

    SOURCES: tuple[tuple[str, str], ...]

    def _on_config_installed(self) -> None:
        """Drop what was cached for the previous config."""
        super()._on_config_installed()
        self._holders_cache: TickerHolders | None = None

    def _holders(self) -> TickerHolders:
        if self._holders_cache is None:
            self._holders_cache = TickerHolders(company_rows(self.config.raw_data_dir_path))
        return self._holders_cache

    def _row(self, record: dict, stage: str) -> dict:
        """Return the variables of one record."""
        raise NotImplementedError

    def _tickers(self, record: dict) -> list:
        """Return the record's tickers, the one to resolve first first."""
        raise NotImplementedError

    def _combine(self, found: dict, at: datetime) -> pl.DataFrame:
        rows: dict[int, dict] = {}
        for (_, stage), records in found.items():
            for record in records:
                cik = _ticker_cik(self._holders(), self._tickers(record), at)
                if cik is not None:
                    row = {"symbol": cik, **self._row(record, stage)}
                    if cik not in rows or self._replaces(row, rows[cik]):
                        rows[cik] = row
        return _frame(list(rows.values()), self.SCHEMA)

    def _replaces(self, row: dict, kept: dict) -> bool:
        """Return whether ``row`` replaces the row already kept for its company; the first is kept by default."""
        return False

    def _states(self) -> pl.DataFrame:
        """Return the snapshot's state at each of its pulls."""
        return market_states(self.config.raw_data_dir_path, self.SOURCES, self._combine, self.SCHEMA)


class StockFitEarningsConsensusDataset(StockFitTickerSnapshotDataset):
    """Daily panel of each company's next earnings date and consensus EPS, from the earnings-calendar snapshots.

    Read from ``/api/earnings/upcoming`` (companies reporting in the next
    28 days), pulled on every refresh. A row names the company by its
    tickers, resolved to its CIK on the day of the pull. Variables:

    - ``earnings_date``, ``filing_date``: the next earnings date and the
      projected day its 10-Q or 10-K is available;
    - ``eps_estimate``, ``analyst_count``: the analyst consensus EPS and the
      number of estimates behind it; NaN when StockFit has none;
    - ``confirmed``: 1 when the date is confirmed by exchange data (StockFit
      gives an expected report time), 0 when it is StockFit's prediction from
      the company's filing cadence.

    StockFit keeps no consensus history: the panel starts at the first
    refresh. A company leaves the calendar once its date passes, and then
    shows nothing until it is listed again.

    Examples
    --------
    With a config like the module example's::

        StockFitEarningsConsensusDataset(config).panel("2026-11-16", "2026-11-20")["eps_estimate"]
    """

    config_cls = StockFitEarningsConsensusConfig
    CONFIG = StockFitEarningsConsensusConfig
    ENDPOINT = EARNINGS_UPCOMING
    SOURCES = ((EARNINGS_UPCOMING, WHOLE_LIST),)
    SCHEMA: ClassVar[dict[str, pl.DataType]] = {
        "earnings_date": DAY, "filing_date": DAY, "eps_estimate": NUMBER, "analyst_count": NUMBER, "confirmed": NUMBER,
    }
    ATTRS: ClassVar[dict[str, dict]] = {
        "earnings_date": {"description": "next earnings date"},
        "filing_date": {"description": "projected day the 10-Q or 10-K is available"},
        "eps_estimate": {"unit": "currency per share", "description": "analyst consensus EPS estimate"},
        "analyst_count": {"unit": "estimates", "description": "analyst estimates behind the consensus"},
        "confirmed": {"description": "1 when the date is exchange-confirmed, 0 when predicted"},
    }

    def _tickers(self, record: dict) -> list:
        """Return every active ticker of the company, the primary first."""
        return list(record.get("symbols") or ())

    def _row(self, record: dict, stage: str) -> dict:
        """Return the date, the consensus and whether the date is confirmed."""
        return {
            "earnings_date": _day(record.get("earningsDate")),
            "filing_date": _day(record.get("filingDate")),
            "eps_estimate": _number(record.get("epsEstimate")),
            "analyst_count": _number(record.get("numEstimates")),
            "confirmed": 1.0 if record.get("reportTime") else 0.0,
        }


class StockFitIpoCalendarDataset(StockFitTickerSnapshotDataset):
    """Daily panel of each company's IPO deal, from the IPO-calendar snapshots.

    Read from ``/api/company/ipo-calendar``, one pull per stage (``filed``,
    ``upcoming``, ``priced``, ``withdrawn``) on every refresh. A deal names
    its company only by the ticker it intends to list under
    (``proposedTicker``), resolved to its CIK on the day of the pull; a deal
    with no ticker, or one no company of the company lists holds, stays in
    the raw tier only, so a deal shows on its company mostly once the
    company is listed. Variables:

    - ``stage``: 1 filed, 2 upcoming, 3 priced, 4 withdrawn (``names``); a
      company with several deals shows its latest stage;
    - ``price_range_low``, ``price_range_high``, ``price``,
      ``shares_offered``, ``dollar_value``: the deal's terms, NaN where the
      stage has none;
    - ``filed_date``, ``expected_price_date``, ``priced_date``,
      ``withdrawn_date``: the stage's dates.

    Examples
    --------
    With a config like the module example's::

        StockFitIpoCalendarDataset(config).panel("2026-11-16", "2026-11-20")["stage"]
    """

    config_cls = StockFitIpoCalendarConfig
    CONFIG = StockFitIpoCalendarConfig
    ENDPOINT = IPO_CALENDAR
    SOURCES = tuple((IPO_CALENDAR, stage) for stage in IPO_STAGES)
    SCHEMA: ClassVar[dict[str, pl.DataType]] = {
        "stage": NUMBER, "price_range_low": NUMBER, "price_range_high": NUMBER, "price": NUMBER,
        "shares_offered": NUMBER, "dollar_value": NUMBER,
        "filed_date": DAY, "expected_price_date": DAY, "priced_date": DAY, "withdrawn_date": DAY,
    }
    ATTRS: ClassVar[dict[str, dict]] = {
        "stage": {"description": "deal stage", "names": {str(code): stage for stage, code in IPO_STAGES.items()}},
        "price_range_low": {"unit": "USD", "description": "low end of the proposed price range"},
        "price_range_high": {"unit": "USD", "description": "high end of the proposed price range"},
        "price": {"unit": "USD", "description": "final offering price"},
        "shares_offered": {"unit": "shares", "description": "shares offered"},
        "dollar_value": {"unit": "USD", "description": "dollar value of the shares offered"},
        "filed_date": {"description": "registration filing date"},
        "expected_price_date": {"description": "expected pricing date"},
        "priced_date": {"description": "pricing date"},
        "withdrawn_date": {"description": "withdrawal date"},
    }

    def _tickers(self, record: dict) -> list:
        """Return the proposed ticker."""
        return [record.get("proposedTicker")]

    def _row(self, record: dict, stage: str) -> dict:
        """Return the stage, the terms and the dates."""
        return {
            "stage": float(IPO_STAGES.get(record.get("status") or stage, IPO_STAGES[stage])),
            "price_range_low": _number(record.get("priceRangeLow")),
            "price_range_high": _number(record.get("priceRangeHigh")),
            "price": _number(record.get("price")),
            "shares_offered": _number(record.get("sharesOffered")),
            "dollar_value": _number(record.get("dollarValue")),
            "filed_date": _day(record.get("filedDate")),
            "expected_price_date": _day(record.get("expectedPriceDate")),
            "priced_date": _day(record.get("pricedDate")),
            "withdrawn_date": _day(record.get("withdrawnDate")),
        }

    def _replaces(self, row: dict, kept: dict) -> bool:
        """Return whether ``row``'s deal is at a later stage."""
        return row["stage"] > kept["stage"]


# -- S-1 offerings ------------------------------------------------------------------


class StockFitOfferingsDataset(StockFitSnapshotPanelDataset):
    """Daily panel of each company's latest S-1/S-1A registration, from the recent-offerings snapshots.

    Read from ``/api/filings/recent-offerings`` (the latest S-1 or S-1/A of
    each company filed in the last 365 days, keyed by CIK), pulled on every
    refresh. Variables: ``filed_date``, ``priced`` (1 when a price range is
    set), ``shares_offered``, ``price_range_low``, ``price_range_high`` and
    ``proposed_max_offering``; NaN where the filing reports none. A company
    whose latest offering ages out of the 365 days shows nothing from then.

    Examples
    --------
    With a config like the module example's::

        StockFitOfferingsDataset(config).panel("2026-11-16", "2026-11-20")["filed_date"]
    """

    config_cls = StockFitOfferingsConfig
    CONFIG = StockFitOfferingsConfig
    ENDPOINT = RECENT_OFFERINGS
    SCHEMA: ClassVar[dict[str, pl.DataType]] = {
        "filed_date": DAY, "priced": NUMBER, "shares_offered": NUMBER,
        "price_range_low": NUMBER, "price_range_high": NUMBER, "proposed_max_offering": NUMBER,
    }
    ATTRS: ClassVar[dict[str, dict]] = {
        "filed_date": {"description": "filing date of the latest S-1 or S-1/A"},
        "priced": {"description": "1 when the offering has a price range"},
        "shares_offered": {"unit": "shares", "description": "shares offered by the company"},
        "price_range_low": {"unit": "USD", "description": "low end of the proposed price range"},
        "price_range_high": {"unit": "USD", "description": "high end of the proposed price range"},
        "proposed_max_offering": {"unit": "USD", "description": "proposed maximum aggregate offering price"},
    }

    def _combine(self, found: dict, at: datetime) -> pl.DataFrame:
        rows: dict[int, dict] = {}
        for record in found.get((RECENT_OFFERINGS, WHOLE_LIST), ()):
            if not record.get("cik"):
                continue
            row = {
                "symbol": int(record["cik"]),
                "filed_date": _day(record.get("filedDate")),
                "priced": 1.0 if record.get("isPriced") else 0.0,
                "shares_offered": _number(record.get("sharesOffered")),
                "price_range_low": _number(record.get("priceRangeLow")),
                "price_range_high": _number(record.get("priceRangeHigh")),
                "proposed_max_offering": _number(record.get("proposedMaxOffering")),
            }
            kept = rows.get(row["symbol"])
            if kept is None or (row["filed_date"] or date.min) > (kept["filed_date"] or date.min):
                rows[row["symbol"]] = row
        return _frame(list(rows.values()), self.SCHEMA)

    def _states(self) -> pl.DataFrame:
        """Return the offerings list's state at each of its pulls."""
        return market_states(
            self.config.raw_data_dir_path, ((RECENT_OFFERINGS, WHOLE_LIST),), self._combine, self.SCHEMA
        )


# -- per-company snapshots ------------------------------------------------------------


class StockFitFilingCalendarDataset(StockFitSnapshotPanelDataset):
    """Daily panel of each company's predicted next 10-K, 10-Q and DEF 14A, from its filing-calendar snapshots.

    Read from ``/api/filings/calendar``, one pull per operating company,
    re-pulled once a day as the day's requests allow. StockFit predicts the
    dates from the company's last five years of filings. Variables, per form
    (``10k``, ``10q``, ``def14a``): ``expected_<form>``, the predicted filing
    date, and ``confidence_<form>``, 1 low, 2 medium, 3 high (``names``).

    Examples
    --------
    With a config like the module example's::

        StockFitFilingCalendarDataset(config).panel("2026-11-16", "2026-11-20")["expected_10q"]
    """

    config_cls = StockFitFilingCalendarConfig
    CONFIG = StockFitFilingCalendarConfig
    ENDPOINT = FILINGS_CALENDAR
    SCHEMA: ClassVar[dict[str, pl.DataType]] = {
        **{f"expected_{form}": DAY for form in CALENDAR_FORMS},
        **{f"confidence_{form}": NUMBER for form in CALENDAR_FORMS},
    }
    ATTRS: ClassVar[dict[str, dict]] = {
        **{f"expected_{form}": {"description": f"predicted next {name} filing date"} for form, name in CALENDAR_FORMS.items()},
        **{
            f"confidence_{form}": {
                "description": f"confidence of the predicted {name} date",
                "names": {str(code): level for level, code in CONFIDENCE.items()},
            }
            for form, name in CALENDAR_FORMS.items()
        },
    }

    @staticmethod
    def _row(body, cik: int, at: datetime) -> dict:
        row: dict = {name: None for name in StockFitFilingCalendarDataset.SCHEMA}
        for form, name in CALENDAR_FORMS.items():
            expected = sorted(
                (entry for entry in (body or {}).get("nextExpected") or () if entry.get("type") == name and entry.get("expectedDate")),
                key=lambda entry: entry["expectedDate"],
            )
            if expected:
                row[f"expected_{form}"] = _day(expected[0]["expectedDate"])
                confidence = CONFIDENCE.get(expected[0].get("confidence") or "")
                row[f"confidence_{form}"] = None if confidence is None else float(confidence)
        return row

    def _states(self) -> pl.DataFrame:
        """Return every pull of every company's filing calendar."""
        return entity_states(self.config.raw_data_dir_path, FILINGS_CALENDAR, self._row, self.SCHEMA)


class StockFitNewsCountsDataset(FilingPanelDataset):
    """Daily panel of the headlines StockFit lists for each company, counted per trading day; no text.

    Read from ``/api/lookup/news`` (a company's latest 50 headlines, from
    third-party feeds), one pull per operating company, re-pulled once a day
    as the day's requests allow. A headline is counted once (by its link,
    or its title and time), on the trading day it became known: the later
    of its publication time and the pull that first carried it, through the
    rule every StockFit panel uses (before a session's close that day,
    otherwise the next trading day). So the first pull's headlines all land
    on the first pull's day.

    ``news_count`` is 0 on a day without a headline and NaN before the
    company's first pull. A company with more than 50 headlines between two
    pulls loses the oldest of them, so a count is a lower bound. Titles,
    summaries and links stay in the raw tier only.

    Examples
    --------
    With a config like the module example's::

        StockFitNewsCountsDataset(config).panel("2026-11-16", "2026-11-20")["news_count"]
    """

    config_cls = StockFitNewsCountsConfig

    def _normalize_config(self, config: DatasetConfig) -> StockFitNewsCountsConfig:
        """Normalise as the base class does, then check the config class.

        Raises
        ------
        TypeError
            If ``config`` is not a ``StockFitNewsCountsConfig``.
        """
        config = super()._normalize_config(config)
        if not isinstance(config, StockFitNewsCountsConfig):
            raise TypeError(f"{self.class_name} needs a StockFitNewsCountsConfig, got {type(config).__name__}.")
        return config

    def _on_config_installed(self) -> None:
        """Drop what was cached for the previous config."""
        super()._on_config_installed()
        self._first_cache: dict[int, np.datetime64] | None = None

    def _empty(self) -> dict[str, object]:
        """Return 0 headlines."""
        return {"news_count": 0.0}

    def _attrs(self) -> dict[str, dict]:
        """Return the variable's unit and meaning."""
        return {"news_count": {"unit": "headlines", "description": "headlines first known that trading day"}}

    def _nothing_shown(self) -> str:
        """Return what an empty window lacks."""
        return "no StockFit news pull is shown"

    def _trading_calendar(self) -> TradingCalendar:
        """Return NYSE's trading days through the day the news raw tier is complete through (cached)."""
        if self._calendar_cache is None:
            self._calendar_cache = nyse_calendar(
                self.config.raw_data_dir_path, NEWS,
                start=self.config.start_date, end=self.config.end_date, owner=self.class_name,
            )
        return self._calendar_cache

    def _known(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Return each headline's company and the time it became known, and each company's first pull."""
        root = self.config.raw_data_dir_path
        headlines: dict[tuple[int, object], datetime] = {}
        first: list[dict] = []
        for entity in entities(root, NEWS):
            cik = int(entity)
            made = pulls(root, NEWS, entity)
            first.append({"symbol": cik, PULLED: made[0]})
            for at in made:  # oldest first, so a headline keeps its first pull
                for item in (read_pull(root, NEWS, entity, at) or {}).get("news") or ():
                    if not item.get("publishedAt"):
                        continue
                    published = datetime.fromisoformat(str(item["publishedAt"])).astimezone(UTC)
                    key = (cik, item.get("link") or (item.get("title"), item["publishedAt"]))
                    headlines.setdefault(key, max(published, at))
        utc = pl.Datetime("us", "UTC")
        known = pl.DataFrame(
            {"symbol": [cik for cik, _ in headlines], PULLED: list(headlines.values())},
            schema={"symbol": pl.Int64, PULLED: utc},
        )
        return known, pl.DataFrame(first, schema={"symbol": pl.Int64, PULLED: utc})

    def _placed(self, rows: pl.DataFrame, *, drop_earlier: bool) -> pl.DataFrame:
        dated = release_day(rows.with_columns(pl.lit(None, pl.Date).alias("_filed")), accepted=PULLED, filed="_filed")
        return self._trading_calendar().place(dated, on="release_day", drop_earlier=drop_earlier)

    def _released(self) -> pl.DataFrame:
        """Return each company's headline count per trading day they became known."""
        with Timer(f"{self.class_name}: count headlines"):
            known, first = self._known()
            self._first_cache = {
                symbol: np.datetime64(day, "ns")
                for symbol, day in self._placed(first, drop_earlier=False).select("symbol", "available").iter_rows()
            }
            return (
                self._placed(known, drop_earlier=True)
                .group_by("symbol", "available")
                .agg(pl.len().cast(pl.Float64).alias("news_count"))
                .sort("available", "symbol")
            )

    def _raw_data_to_xr_window(self, start_date, end_date, symbols: list | None = None) -> xr.Dataset:
        """Return one window: each day's count, 0 without a headline, NaN before the company's first pull."""
        window = super()._raw_data_to_xr_window(start_date, end_date, symbols)
        self._released_rows()
        first = xr.DataArray(
            [self._first_cache.get(int(symbol), np.datetime64("NaT", "ns")) for symbol in window["symbol"].values],
            coords={"symbol": window["symbol"]}, dims="symbol",
        )
        return window.where(window["timestamp"] >= first)

    def _widen_fill_values(self) -> dict:
        """Fill a new company's stored days with NaN: it was not pulled then."""
        return {"news_count": np.nan}
