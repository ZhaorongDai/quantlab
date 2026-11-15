"""StockFit's derived metrics and share classes, at the filing dates of the statements, keyed by CIK.

StockFit computes growth rates (``/api/financials/growth``), key metrics
(``/api/financials/key-metrics``), the EPS history
(``/api/earnings/eps-history``) and the dividend history
(``/api/earnings/dividend-history``) from the filings, and parses each
company's share classes (``/api/ownership/share-classes``, the latest period
only). None of them carries a filing date. Each row therefore takes the
filing date of the same CIK and fiscal period end from the statements in the
same raw tier (``statement_filings``: the original filing of the period, by
its acceptance time when known); a row with no matching statement period is
dropped and counted in the dataset's ``build_report``.

``StockFitDerivedDataset`` holds one table of ``DERIVED_TABLES``: the
quarterly growth rates, the key metrics' ``general`` block (the sector and
industry blocks vary by company and stay in the raw tier), the EPS and
dividend histories (per-share values as filed, not split-adjusted), or the
share classes summed per period (``class_count``, ``shares_outstanding``,
``shares_authorized``; the per-class detail stays in the raw tier). The
vendor serves today's values without a trail of earlier ones, so the values
of a company's first pull may include later restatements and are flagged
(``may_include_restatement``); from then on each pull is compared with the
one before (``quantlab.dataset.stockfit.period_panel``). The share classes
gain a history only from the pulls, one period per pull.

Examples
--------
Build the growth store from a StockFit raw tier holding the statements::

    config = StockFitDerivedConfig(
        zarr_file_path="/data/quantlab/market/stockfit/stockfit_growth_quarter/stockfit_growth_quarter.zarr",
        raw_data_dir_path="/data/quantlab/downloads/stockfit",
        table="growth",
    )
    dataset = StockFitDerivedDataset(config).update()
    dataset.panel("2024-01-02", "2024-03-28")["revenueGrowth"]
    dataset.build_report.unmatched
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from loguru import logger

from quantlab.dataset.config import DatasetConfig, StockFitDerivedConfig
from quantlab.dataset.stockfit import raw
from quantlab.dataset.stockfit.period_panel import (
    PeriodRow,
    VersionedPeriodDataset,
    as_float,
    parse_accepted,
)
from quantlab.dataset.stockfit.statements import STATEMENTS

#: The growth rates, in the vendor's order (decimals: 0.15 is 15%).
GROWTH_FIELDS: tuple[str, ...] = (
    "revenueGrowth", "grossProfitGrowth", "operatingIncomeGrowth", "netIncomeGrowth", "epsGrowth",
    "ebitdaGrowth", "freeCashFlowGrowth", "totalAssetsGrowth", "stockholdersEquityGrowth",
    "operatingCashFlowGrowth",
)

#: The key metrics every company has (the ``general`` block), in the vendor's order.
KEY_METRIC_FIELDS: tuple[str, ...] = (
    "revenue", "netIncome", "grossMargin", "operatingMargin", "profitMargin", "debtToEquity", "quickRatio",
    "currentRatio", "roa", "roe", "roic", "eps", "ebitda", "freeCashFlow", "fcfMargin", "fcfNetIncome",
    "assetTurnover", "interestCoverage", "netDebt", "netDebtEbitda", "investedCapital", "sbcRevenue",
    "dividendPayout", "netShareChange", "cashRatio", "operatingCashFlowRatio", "receivablesTurnover",
    "payablesTurnover", "inventoryTurnover", "fixedAssetTurnover", "financialLeverageRatio",
    "longTermDebtToCapitalRatio", "freeCashFlowOperatingCashFlowRatio", "capitalExpenditureCoverageRatio",
    "pretaxProfitMargin", "effectiveTaxRate", "revenuePerShare", "netIncomePerShare",
    "operatingCashFlowPerShare", "freeCashFlowPerShare", "cashPerShare", "bookValuePerShare",
)

#: The EPS history's values, in the vendor's order.
EPS_FIELDS: tuple[str, ...] = (
    "eps", "epsDiluted", "netIncome", "sharesOutstanding", "sharesOutstandingDiluted", "epsGrowth",
)

#: The dividend history's values, in the vendor's order.
DIVIDEND_FIELDS: tuple[str, ...] = (
    "dividendPerShare", "totalDividendsPaid", "payoutRatio", "dividendCoverage", "dividendGrowth",
)

#: The share-class totals per period.
SHARE_CLASS_FIELDS: tuple[str, ...] = ("class_count", "shares_outstanding", "shares_authorized")


def _periods(fields: tuple[str, ...], block: str | None = None) -> Callable[[object], dict[str, dict[str, float]]]:
    """Return a reader of a per-period array: each ``period``'s ``fields`` (from ``block`` when set)."""

    def read(body) -> dict[str, dict[str, float]]:
        rows = {}
        for entry in body or ():
            if not isinstance(entry, Mapping) or not entry.get("period"):
                continue
            source = (entry.get(block) or {}) if block else entry
            rows[str(entry["period"])[:10]] = {name: as_float(source.get(name)) for name in fields}
        return rows

    return read


def _sum(values: list) -> float:
    found = [as_float(value) for value in values]
    found = [value for value in found if not math.isnan(value)]
    return float(sum(found)) if found else float("nan")


def _share_classes(body) -> dict[str, dict[str, float]]:
    """Return per period end the number of classes and their summed outstanding and authorized shares."""
    by_period: dict[str, list[Mapping]] = {}
    for row in body or ():
        if isinstance(row, Mapping) and row.get("periodEnd"):
            by_period.setdefault(str(row["periodEnd"])[:10], []).append(row)
    return {
        period: {
            "class_count": float(len(rows)),
            "shares_outstanding": _sum([row.get("sharesOutstanding") for row in rows]),
            "shares_authorized": _sum([row.get("sharesAuthorized") for row in rows]),
        }
        for period, rows in by_period.items()
    }


@dataclass(frozen=True)
class DerivedTable:
    """One derived table: the endpoint it is pulled from, its variables and how a pull becomes rows.

    Parameters
    ----------
    endpoint : str
        The raw-tier endpoint.
    variables : tuple of str
        The panel's value variables.
    rows : Callable
        Turns one pull's body into values by fiscal period end (ISO date).

    Examples
    --------
    >>> DERIVED_TABLES["growth"].endpoint
    'growth_quarter'
    """

    endpoint: str
    variables: tuple[str, ...]
    rows: Callable[[object], dict[str, dict[str, float]]]


#: The tables a store may hold.
DERIVED_TABLES: dict[str, DerivedTable] = {
    "growth": DerivedTable("growth_quarter", GROWTH_FIELDS, _periods(GROWTH_FIELDS)),
    "key_metrics": DerivedTable("key_metrics_quarter", KEY_METRIC_FIELDS, _periods(KEY_METRIC_FIELDS, "general")),
    "eps_history": DerivedTable("eps_history_quarter", EPS_FIELDS, _periods(EPS_FIELDS)),
    "dividend_history": DerivedTable("dividend_history_quarter", DIVIDEND_FIELDS, _periods(DIVIDEND_FIELDS)),
    "share_classes": DerivedTable("share_classes", SHARE_CLASS_FIELDS, _share_classes),
}

#: The statement endpoints whose periods lend their filing dates.
STATEMENT_ENDPOINTS: tuple[str, ...] = tuple(
    f"{name}_{period}" for period in ("quarter", "annual") for name in STATEMENTS
)


def _order(filing: tuple[date, datetime | None]) -> tuple[date, float]:
    """Sort key of a filing: its date, then its acceptance time (unknown last)."""
    return filing[0], filing[1].timestamp() if filing[1] is not None else float("inf")


def statement_filings(vendor_root: str | Path, cik: int | str) -> dict[str, tuple[date, datetime | None]]:
    """Return, per fiscal period end, the date and acceptance time of the period's original filing.

    Read from the newest pull of each quarterly and annual statement of the
    company; when the statements disagree, the earliest filing wins.

    Parameters
    ----------
    vendor_root : str or Path
        ``<download-dir>/stockfit``.
    cik : int or str
        The company.

    Returns
    -------
    dict
        ISO period end to ``(filing date, acceptance time or None)``.

    Examples
    --------
    >>> statement_filings("/nonexistent/stockfit", 1652044)
    {}
    """
    found: dict[str, tuple[date, datetime | None]] = {}
    for endpoint in STATEMENT_ENDPOINTS:
        if not raw.pulls(vendor_root, endpoint, str(cik)):
            continue
        for entry in raw.read_pull(vendor_root, endpoint, str(cik)) or ():
            if not entry.get("period") or not entry.get("dateFiled"):
                continue
            filing = (date.fromisoformat(entry["dateFiled"][:10]), parse_accepted(entry.get("acceptedAt")))
            key = str(entry["period"])[:10]
            known = found.get(key)
            if known is None or _order(filing) < _order(known):
                found[key] = filing
    return found


@dataclass(frozen=True)
class DerivedBuildReport:
    """What one build of a derived panel kept and dropped.

    Parameters
    ----------
    rows : int
        Distinct ``(CIK, period end)`` rows kept.
    unmatched : int
        Distinct ``(CIK, period end)`` rows dropped because the statements
        hold no filing of that period.

    Examples
    --------
    >>> DerivedBuildReport(rows=10, unmatched=2).unmatched
    2
    """

    rows: int
    unmatched: int


class StockFitDerivedDataset(VersionedPeriodDataset):
    """Daily point-in-time panel of one StockFit derived table, at the statements' filing dates, keyed by CIK.

    See the module docstring for what a cell shows and when. After a build,
    ``build_report`` (a ``DerivedBuildReport``) counts the rows kept and the
    rows dropped for want of a matching statement period.

    Parameters
    ----------
    config : StockFitDerivedConfig
        ``table`` is a key of ``DERIVED_TABLES``.

    Examples
    --------
    With ``config`` as in the module example::

        ds = StockFitDerivedDataset(config).update()
        ds.panel("2024-01-02", "2024-01-05")["revenueGrowth"], ds.build_report
    """

    #: The config class used to rebuild this dataset from a saved config.
    config_cls = StockFitDerivedConfig

    #: The counts of the last build; ``None`` before one.
    build_report: DerivedBuildReport | None = None

    def _normalize_config(self, config: DatasetConfig) -> StockFitDerivedConfig:
        """Normalise as the base class does, then check the table.

        Raises
        ------
        TypeError
            If ``config`` is not a ``StockFitDerivedConfig``.
        ValueError
            If ``table`` or ``stale_after_days`` is invalid.

        Examples
        --------
        ::

            StockFitDerivedDataset(dataclasses.replace(config, table="scores"))
            # ValueError: StockFitDerivedDataset: table must be one of ...
        """
        config = super()._normalize_config(config)
        if not isinstance(config, StockFitDerivedConfig):
            raise TypeError(f"{self.class_name} needs a StockFitDerivedConfig, got {type(config).__name__}.")
        if config.table not in DERIVED_TABLES:
            raise ValueError(f"{self.class_name}: table must be one of {tuple(DERIVED_TABLES)}, got {config.table!r}.")
        if config.stale_after_days is not None and config.stale_after_days < 1:
            raise ValueError(
                f"{self.class_name}: stale_after_days must be at least 1 or None, got {config.stale_after_days}."
            )
        return config

    def _table(self) -> DerivedTable:
        return DERIVED_TABLES[self.config.table]

    def _endpoint(self) -> str:
        """Return the table's endpoint."""
        return self._table().endpoint

    def _variables(self) -> dict[str, dict]:
        """Return the table's variables, each with its table."""
        return {name: {"table": self.config.table} for name in self._table().variables}

    def _begin(self) -> None:
        """Start counting the kept and the unmatched rows."""
        self._kept: set[tuple[int, str]] = set()
        self._unmatched: set[tuple[int, str]] = set()
        self._filings: dict[int, dict[str, tuple[date, datetime | None]]] = {}

    def _end(self) -> None:
        """Record the counts in ``build_report`` and log the unmatched rows."""
        self.build_report = DerivedBuildReport(rows=len(self._kept), unmatched=len(self._unmatched))
        if self._unmatched:
            logger.warning(
                f"{self.class_name}: dropped {len(self._unmatched)} {self.config.table} row(s) whose CIK and "
                f"period end match no statement period (kept {len(self._kept)})."
            )

    def _period_rows(self, body, cik: int, pulled_at: datetime) -> dict[str, PeriodRow]:
        """Return each period with a matching statement period, at that period's original filing."""
        filings = self._filings.get(cik)
        if filings is None:
            filings = self._filings[cik] = statement_filings(self.config.raw_data_dir_path, cik)
        rows: dict[str, PeriodRow] = {}
        for period, values in self._table().rows(body).items():
            filing = filings.get(period)
            if filing is None:
                self._unmatched.add((cik, period))
                continue
            self._kept.add((cik, period))
            rows[period] = PeriodRow(filed=filing[0], accepted_at=filing[1], values=values)
        return rows
