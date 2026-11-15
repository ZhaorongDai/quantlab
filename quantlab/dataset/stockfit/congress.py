"""Congress members' stock trades (STOCK Act reports) from StockFit, as a daily panel on the CIK axis.

StockFit parses the periodic transaction reports House members file with
the Clerk (since 2014) and senators with the Senate eFD (since 2012). The
raw tier keeps the two market-wide feeds (``congress_house_trades`` and
``congress_senate_trades``, ``quantlab.dataset.stockfit.raw``), each pull the
reports filed since a day; a report that appears in several pulls is read
from the newest. A report lists its transactions, each with the asset's
ticker as disclosed, the ticker StockFit resolved it to (``resolvedTicker``)
and how sure that link is (``linkConfidence``: ``high``, ``medium``,
``low``, ``none``). A paper report (``unparsed_paper``) has no transactions
and stays in the raw tier only.

A transaction reaches the panel only when

- its ``linkConfidence`` is ``high``;
- its ``resolvedTicker`` names one company on its transaction date
  (``TickerHolders``): tickers are reused, so a ticker is resolved through
  the current and prior tickers of StockFit's company lists, not through
  whoever holds it today;
- it is a ``purchase``, or a ``sale`` or ``sale_partial`` (an
  ``exchange`` or ``other`` counts as neither).

Every other transaction stays raw. Each counted transaction is placed at
its report's filing date (a report has no acceptance time), on that NYSE
trading day or the next one, and a day's transactions of one company are
summed, as the Sharadar insider panel sums Form 4 lines:

- ``purchase_min``, ``purchase_max``, ``sale_min``, ``sale_max``: the sums of
  the disclosed amount bands' bounds, in USD. An open band (``Over
  $50,000,000``) adds its lower bound to both.
- ``purchases``, ``sales``: the number of transactions.
- ``members_buying``, ``members_selling``: the distinct members (Bioguide ID,
  or name when the vendor matched none) with a purchase or a sale.

A day without a counted transaction is 0. A transaction that a later report
(an amendment) repeats exactly (same chamber, member, owner, ticker, type,
transaction date and amount band) is counted on its first report only.
Every asset type the vendor resolved to a listing counts, options among
them. Reports are often filed weeks after the trade (the 45-day deadline
is often missed), and the transaction date is never used to place a trade.

The calendar is NYSE's trading days through the day before the older of
the two feeds' last pulls.

Examples
--------
Build the store and read a quarter::

    config = StockFitCongressTradesConfig(
        zarr_file_path="/data/market/stockfit/stockfit_congress_trades_1d/stockfit_congress_trades_1d.zarr",
        raw_data_dir_path="/data/downloads/stockfit",
    )
    StockFitCongressTradesDataset(config).update()
    panel = StockFitCongressTradesDataset(config).panel("2024-01-02", "2024-03-28")
    panel["purchase_max"].attrs["unit"]  # 'USD'
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from datetime import date, timedelta
from pathlib import Path

import polars as pl

from quantlab.dataset._support.point_in_time import FilingPanelDataset, TradingCalendar
from quantlab.dataset.config import DatasetConfig, StockFitCongressTradesConfig
from quantlab.dataset.stockfit.available import nyse_calendar
from quantlab.dataset.stockfit.raw import (
    COMPANY_LISTS,
    CONGRESS_HOUSE_TRADES,
    CONGRESS_SENATE_TRADES,
    pulls,
    read_pull,
)
from quantlab.utils.timer import Timer

#: The raw-tier feeds read, by chamber.
FEEDS: dict[str, str] = {"house": CONGRESS_HOUSE_TRADES, "senate": CONGRESS_SENATE_TRADES}

#: The only ``linkConfidence`` placed on a company.
HIGH = "high"

#: ``transactionType`` values counted as a purchase, and as a sale.
PURCHASES: tuple[str, ...] = ("purchase",)
SALES: tuple[str, ...] = ("sale", "sale_partial")

#: How long after a prior ticker was last seen its holder may still have held
#: it: StockFit's ``lastSeen`` is a lower bound, often a period end months
#: before the real change.
HANDOVER_DAYS = 182

#: The columns that identify one transaction across the reports showing it.
TRADE_KEY: tuple[str, ...] = (
    "chamber", "member", "owner", "ticker", "kind", "transaction_date", "amount_min", "amount_max",
)


def normalize_ticker(ticker: str) -> str:
    """Return a ticker in one spelling: upper case, share-class separators as ``.``.

    Examples
    --------
    >>> normalize_ticker("brk-b"), normalize_ticker("BRK/B"), normalize_ticker("VCXB WS")
    ('BRK.B', 'BRK.B', 'VCXB.WS')
    """
    return ticker.strip().upper().replace("-", ".").replace("/", ".").replace(" ", ".")


class TickerHolders:
    """Which company held a ticker on a day, from StockFit's current and prior tickers.

    A company holds each of its current tickers (``symbols``) up to today,
    and each prior ticker (``priorTickers``) at least until its ``lastSeen``
    and perhaps up to ``HANDOVER_DAYS`` after it. Holders of one ticker
    follow each other in the order they stopped holding it, so a company
    cannot have held it on a day an earlier holder still held it. A ticker
    resolves on a day when exactly one company may have held it then.

    Parameters
    ----------
    companies : iterable of dict
        Company rows of ``/api/symbols`` or ``/api/company/delisted``
        (``cik``, ``symbols``, ``priorTickers``).

    Examples
    --------
    >>> holders = TickerHolders([
    ...     {"cik": 1, "symbols": ["META"], "priorTickers": [{"symbol": "FB", "lastSeen": "2022-03-31"}]},
    ...     {"cik": 2, "symbols": ["FB"], "priorTickers": []},
    ... ])
    >>> holders.on("FB", date(2021, 6, 1)), holders.on("FB", date(2023, 6, 1)), holders.on("FB", date(2022, 5, 2))
    (1, 2, None)
    """

    def __init__(self, companies: Iterable[dict]):
        # ticker -> cik -> last day held (None: holds it today)
        held: dict[str, dict[int, date | None]] = defaultdict(dict)
        for row in companies:
            if not row.get("cik"):
                continue
            cik = int(row["cik"])
            for symbol in row.get("symbols") or ():
                held[normalize_ticker(symbol)][cik] = None
            for prior in row.get("priorTickers") or ():
                if not prior.get("symbol") or not prior.get("lastSeen"):
                    continue
                ticker, seen = normalize_ticker(prior["symbol"]), date.fromisoformat(prior["lastSeen"][:10])
                ends = held[ticker]
                if cik not in ends or (ends[cik] is not None and ends[cik] < seen):
                    ends[cik] = seen
        self._held = {ticker: sorted(ends.items(), key=lambda item: (item[1] is None, item[1])) for ticker, ends in held.items()}

    def on(self, ticker: str, day: date) -> int | None:
        """Return the CIK of the one company that may have held ``ticker`` on ``day``; ``None`` if none or several.

        Examples
        --------
        >>> TickerHolders([{"cik": 7, "symbols": ["ZTS"]}]).on("zts", date(2020, 1, 2))
        7
        """
        holders = self._held.get(normalize_ticker(ticker), [])
        possible = set()
        for index, (cik, last) in enumerate(holders):
            # An earlier holder (ordered by when it stopped) still held it that day.
            blocked = any(other != cik and day <= ended for other, ended in holders[:index] if ended is not None)
            if not blocked and (last is None or day <= last + timedelta(days=HANDOVER_DAYS)):
                possible.add(cik)
        return possible.pop() if len(possible) == 1 else None


def company_rows(vendor_root: str | Path) -> list[dict]:
    """Return every row of the newest company lists (``COMPANY_LISTS``).

    Examples
    --------
    >>> company_rows("/nonexistent/stockfit")
    []
    """
    rows: list[dict] = []
    for endpoint, entity in COMPANY_LISTS:
        if pulls(vendor_root, endpoint, entity):
            rows.extend(read_pull(vendor_root, endpoint, entity))
    return rows


def congress_reports(vendor_root: str | Path) -> list[tuple[str, dict]]:
    """Return every report of both feeds as ``(chamber, report)``, each from the newest pull showing it.

    Examples
    --------
    >>> congress_reports("/nonexistent/stockfit")
    []
    """
    reports: dict[tuple[str, str], dict] = {}
    for chamber, endpoint in FEEDS.items():
        for pulled_at in pulls(vendor_root, endpoint, "all"):
            for report in read_pull(vendor_root, endpoint, "all", pulled_at):
                reports[(chamber, str(report["filingId"]))] = report
    return [(chamber, report) for (chamber, _), report in reports.items()]


class StockFitCongressTradesDataset(FilingPanelDataset):
    """Daily panel of Congress members' trades per company, on each report's filing day.

    Parameters
    ----------
    config : StockFitCongressTradesConfig
        ``raw_data_dir_path`` is ``<root>/downloads/stockfit``.

    Examples
    --------
    With ``config`` as in the module example::

        ds = StockFitCongressTradesDataset(config).update()
        list(ds.panel("2024-01-02", "2024-01-05").data_vars)[:2]  # ['purchase_min', 'purchase_max']
    """

    #: The config class used to rebuild this dataset from a saved config.
    config_cls = StockFitCongressTradesConfig

    def _normalize_config(self, config: DatasetConfig) -> StockFitCongressTradesConfig:
        """Normalise as the base class does, then check the config class.

        Raises
        ------
        TypeError
            If ``config`` is not a ``StockFitCongressTradesConfig``.
        """
        config = super()._normalize_config(config)
        if not isinstance(config, StockFitCongressTradesConfig):
            raise TypeError(f"{self.class_name} needs a StockFitCongressTradesConfig, got {type(config).__name__}.")
        return config

    def _empty(self) -> dict[str, object]:
        """Return 0 for every variable: no trade."""
        return {name: 0.0 for name in (
            "purchase_min", "purchase_max", "sale_min", "sale_max",
            "purchases", "sales", "members_buying", "members_selling",
        )}

    def _attrs(self) -> dict[str, dict]:
        """Return each variable's unit and meaning."""
        return {
            "purchase_min": {"unit": "USD", "description": "sum of the purchases' amount-band lower bounds"},
            "purchase_max": {"unit": "USD", "description": "sum of the purchases' amount-band upper bounds"},
            "sale_min": {"unit": "USD", "description": "sum of the sales' amount-band lower bounds"},
            "sale_max": {"unit": "USD", "description": "sum of the sales' amount-band upper bounds"},
            "purchases": {"unit": "transactions", "description": "purchases disclosed"},
            "sales": {"unit": "transactions", "description": "sales (full or partial) disclosed"},
            "members_buying": {"unit": "members", "description": "distinct members with a purchase"},
            "members_selling": {"unit": "members", "description": "distinct members with a sale"},
        }

    def _nothing_shown(self) -> str:
        """Return what an empty window lacks."""
        return "no Congress trade resolved to a company is shown"

    def _trading_calendar(self) -> TradingCalendar:
        """Return NYSE's trading days through the day before the older feed's last pull (cached).

        Raises
        ------
        ValueError
            If a feed was never pulled.
        """
        if self._calendar_cache is None:
            self._calendar_cache = nyse_calendar(
                self.config.raw_data_dir_path, tuple(FEEDS.values()),
                start=self.config.start_date, end=self.config.end_date, owner=self.class_name,
            )
        return self._calendar_cache

    def _transactions(self) -> pl.DataFrame:
        """Return every counted transaction with its company, each from the first report showing it."""
        root = self.config.raw_data_dir_path
        holders = TickerHolders(company_rows(root))
        lines = []
        for chamber, report in congress_reports(root):
            member = report.get("member") or {}
            who = member.get("bioguideId") or member.get("name")
            for trade in report.get("transactions") or ():
                kind = trade.get("transactionType")
                ticker = trade.get("resolvedTicker")
                if trade.get("linkConfidence") != HIGH or not ticker or kind not in (*PURCHASES, *SALES):
                    continue
                traded = date.fromisoformat(trade["transactionDate"][:10])
                cik = holders.on(ticker, traded)
                if cik is None:
                    continue
                low = float(trade.get("amountMin") or 0.0)
                high = trade.get("amountMax")
                lines.append({
                    "cik": cik, "date": date.fromisoformat(report["filingDate"][:10]),
                    "chamber": chamber, "member": who, "owner": trade.get("owner"),
                    "ticker": normalize_ticker(ticker), "kind": kind, "transaction_date": traded,
                    "amount_min": low, "amount_max": low if high is None else float(high),
                })
        schema = {
            "cik": pl.Int64, "date": pl.Date, "chamber": pl.String, "member": pl.String, "owner": pl.String,
            "ticker": pl.String, "kind": pl.String, "transaction_date": pl.Date,
            "amount_min": pl.Float64, "amount_max": pl.Float64,
        }
        frame = pl.DataFrame(lines, schema=schema)
        first_shown = pl.col("date") == pl.col("date").min().over(TRADE_KEY)
        return frame.filter(first_shown)

    def _released(self) -> pl.DataFrame:
        """Return every company's counted trades per filing day, summed.

        Returns
        -------
        pl.DataFrame
            ``symbol`` (the CIK), ``available`` (the trading day the report
            lands on) and one column per variable, sorted by both.
        """
        calendar = self._trading_calendar()
        with Timer(f"{self.class_name}: place trades"):
            frame = self._transactions().filter(pl.col("date") <= pl.lit(calendar.through))
            bought = pl.col("kind").is_in(PURCHASES)
            sold = pl.col("kind").is_in(SALES)

            def only(condition: pl.Expr, amount: pl.Expr) -> pl.Expr:
                return pl.when(condition).then(amount).otherwise(0.0)

            return (
                calendar.place(frame, drop_earlier=True)
                .group_by(pl.col("cik").alias("symbol"), "available")
                .agg(
                    only(bought, pl.col("amount_min")).sum().alias("purchase_min"),
                    only(bought, pl.col("amount_max")).sum().alias("purchase_max"),
                    only(sold, pl.col("amount_min")).sum().alias("sale_min"),
                    only(sold, pl.col("amount_max")).sum().alias("sale_max"),
                    bought.sum().cast(pl.Float64).alias("purchases"),
                    sold.sum().cast(pl.Float64).alias("sales"),
                    pl.col("member").filter(bought).drop_nulls().n_unique().cast(pl.Float64).alias("members_buying"),
                    pl.col("member").filter(sold).drop_nulls().n_unique().cast(pl.Float64).alias("members_selling"),
                )
                .sort("available", "symbol")
            )
