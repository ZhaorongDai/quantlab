"""StockFit's quarterly and TTM statement panels, keyed by CIK, as known on each day.

StockFit's three normalized statements (``/api/financials/income-statement``,
``/balance-sheet``, ``/cash-flow-statement``; 48, 78 and 52 facts) carry,
for every fiscal period, the original filing's ``dateFiled`` and
``acceptedAt`` and a restatement chain: ``sources`` maps each filing that
supplied a value of the period to its own dates and, per fact, ``{}`` (it
supplied the value) or ``{"before": x}`` (it changed the value; ``x`` is
what the filing before it had). ``fact_versions`` walks that chain: it
returns each value a fact of one period ever had, with the dates of the
filing that made it public. It hides what the chain is made of:

- amendments (``10-K/A``, ``10-Q/A``) are filings like any other, each
  value at its own filing's dates;
- a pruned filing (one whose values were all replaced later) left only its
  value in the first ``before``: that value is dated at the period's
  original filing;
- reconstructed quarters (``Q4-Recon``: Q4 = fiscal year - nine months;
  ``Q2-Recon``/``Q3-Recon``: cash-flow quarters from year-to-date values)
  have no dates of their own: their values are dated at the period's
  original filing, the one that made the reconstruction possible;
- a fact the vendor computes from others (listed in ``derived``) has no
  chain: its latest value is dated at the period's original filing.

``StockFitStatementsDataset`` places every value at its available date
(``quantlab.dataset.stockfit.available``: accepted before the NYSE close,
that day; at or after it, the next trading day; with only ``dateFiled``,
that day) on NYSE's calendar, and shows on every trading day the latest
fiscal quarter filed by then, each fact as known that day (``reported=
"as_known"``) or as first reported (``reported="original"``). A
restatement is shown from its own available date, never earlier; a later
filing of an older quarter never replaces a newer quarter. A TTM store
(``period="ttm"``) shows the four latest quarters as known that day, once
they are four consecutive fiscal quarters: flows summed over the four
quarters (missing when any of them lacks the flow, where the vendor's own
TTM sums the quarters that report it), weighted share counts averaged over
the quarters that report them (rounded half up to whole
shares), balance-sheet items and ``endCash`` from the latest quarter and
``beginningCash`` from the earliest. Annual statements stay in the raw tier;
the vendor's own TTM response (its latest period only, the same for the
three statements) is not pulled.

Every fact is a float64 variable named as the vendor names it; the five the
cash-flow statement shares with the income statement are prefixed
(``cashFlowNetIncome``, ``cashFlowDepreciation``, ...). ``release_date``
(the shown quarter's original filing date) and ``period_end`` (its fiscal
period end) say which quarter a cell shows; a quarter no newer one has
replaced stops being shown ``stale_after_days`` after its release date.

Look-ahead caveats: a derived fact of a restated period shows its latest
value from the period's first filing; the raw tier is taken to be complete
through the day before its newest pull of a quarterly statement.

Examples
--------
Build the quarterly store from a StockFit raw tier and read a quarter::

    config = StockFitStatementsConfig(
        zarr_file_path="/data/market/stockfit/stockfit_statements_quarter/stockfit_statements_quarter.zarr",
        raw_data_dir_path="/data/downloads/stockfit",
    )
    panel = StockFitStatementsDataset(config).update().panel("2024-01-02", "2024-03-28")
    panel["revenue"].sel(symbol=320193)
"""

from __future__ import annotations

import bisect
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date

import numpy as np
import polars as pl

from quantlab.dataset._support.point_in_time import (
    ReleasePanelDataset,
    ReleaseSpec,
    TradingCalendar,
)
from quantlab.dataset.config import DatasetConfig, StockFitStatementsConfig
from quantlab.dataset.stockfit import raw
from quantlab.dataset.stockfit.available import (
    nyse_calendar,
    parse_accepted,
    place_released,
)
from quantlab.utils.timer import Timer

#: The income statement's facts, in the vendor's order.
INCOME_FACTS: tuple[str, ...] = (
    "revenue", "costOfRevenue", "grossProfit", "researchAndDevelopment", "generalAndAdministrativeExpenses",
    "sellingAndMarketingExpenses", "sellingGeneralAndAdministrativeExpenses", "otherExpenses",
    "restructuringCharges", "restructuringAndImpairmentTotal", "impairments", "operatingExpenses",
    "costAndExpenses", "interestIncome", "interestExpense", "interestIncomeNet",
    "depreciationAndAmortization", "depreciation", "amortization", "operatingIncome", "nonOperatingIncome",
    "incomeBeforeTax", "incomeTaxExpense", "netIncome", "netIncomeCommonStockholders",
    "netIncomeCommonStockholdersDiluted", "weightedAverageSharesOutstanding",
    "weightedAverageSharesOutstandingDiluted", "eps", "epsDiluted", "ebit", "ebitda", "premiumsEarned",
    "premiumsWritten", "policyholderBenefitsAndClaims", "policyAcquisitionCosts",
    "otherUnderwritingExpense", "underwritingIncome", "netInvestmentIncome", "noninterestIncome",
    "noninterestExpense", "provisionForCreditLosses", "gainLossOnDisposition",
    "gainLossOnDebtExtinguishment", "acquisitionRelatedCosts", "goodwillImpairmentLoss", "unusualItemsNet",
    "unusualItemsExcludingGoodwill",
)

#: The balance sheet's facts, in the vendor's order.
BALANCE_SHEET_FACTS: tuple[str, ...] = (
    "assets", "currentAssets", "cash", "netReceivables", "accountsReceivable", "inventory", "finishedGoods",
    "otherCurrentAssets", "accumulatedDepreciation", "goodwill", "otherIntangibleAssets",
    "otherNonCurrentAssets", "nonCurrentAssets", "liabilities", "currentLiabilities", "accountsPayable",
    "payablesToBrokerDealers", "payablesToCustomers", "taxesPayable", "totalPayable",
    "otherCurrentAccruedLiabilities", "currentCapitalLeaseObligations", "operatingLeaseLiabilityCurrent",
    "financeLeaseLiabilityCurrent", "currentDeferredLiabilities", "otherCurrentLiabilities",
    "totalNonCurrentLiabilities", "longTermDebt", "longTermCapitalLeaseObligations",
    "operatingLeaseLiabilityNoncurrent", "financeLeaseLiabilityNoncurrent", "otherNonCurrentLiabilities",
    "stockholdersEquity", "minorityInterest", "totalEquity", "commonStock", "preferredStock",
    "retainedEarnings", "otherEquityAdjustments", "additionalPaidInCapital", "capitalLeaseObligations",
    "operatingLeaseLiability", "financeLeaseLiability", "workingCapital", "totalDebt", "totalBorrowings",
    "sharesIssued", "storeCount", "sharesOutstanding", "treasuryShares", "currentSharesOutstanding",
    "preferredSharesOutstanding", "publicFloat", "policyReserves", "deferredAcquisitionCosts",
    "propertyPlantAndEquipmentGross", "propertyPlantAndEquipmentNet", "constructionInProgress",
    "costsInExcessOfBillings", "billingsInExcessOfCosts", "contractReceivableRetainage", "debtCurrent",
    "longTermDebtCurrent", "deposits", "loans", "allowanceForDoubtfulAccounts",
    "allowanceForCreditLossesLoans", "assetRetirementObligation", "assetRetirementObligationCurrent",
    "assetRetirementObligationNoncurrent", "warrantyLiability", "contractLiabilityNoncurrent",
    "remainingPerformanceObligations", "investments", "shortTermInvestments", "cashAndShortTermInvestments",
    "longTermInvestments", "lettersOfCreditOutstanding",
)

#: The cash-flow statement's facts, in the vendor's order.
CASH_FLOW_FACTS: tuple[str, ...] = (
    "operatingCashFlow", "netIncome", "operatingGainsLosses", "depreciationAndAmortization", "depreciation",
    "amortization", "stockBasedCompensation", "deferredIncomeTax", "impairments", "otherNonCashItems",
    "changeInWorkingCapital", "changeInInventory", "changeInAccountsReceivable", "changeInPrepaidAssets",
    "changeInPayablesAndAccruedExpenses", "changeInAccountsPayable", "changeInAccruedExpenses",
    "changeInOtherNonCurrentAssets", "changeInOtherCurrentLiabilities", "changeInOtherWorkingCapital",
    "investingCashFlow", "purchaseOfIntangibleAssets", "purchasesOfInvestments",
    "proceedsFromSaleOfInvestments", "netOtherInvestingChanges", "financingCashFlow",
    "commonStockRepurchased", "proceedsFromCommonStockIssuance", "netCommonStockIssuance",
    "proceedsFromStockOptionsExercised", "endCash", "changeInCash", "effectOfExchangeRateChanges",
    "netChangeInCashIncludingFx", "beginningCash", "incomeTaxPaid", "interestPaid", "capitalExpenditure",
    "purchaseOfPropertyPlantAndEquipment", "issuanceOfDebt", "repaymentOfDebt", "issuanceOfShortTermDebt",
    "repaymentOfShortTermDebt", "issuanceOfLongTermDebt", "repaymentOfLongTermDebt",
    "netIssuancePaymentsOfDebt", "netIssuanceOfLongTermDebt", "cashDividendsPaid", "freeCashFlow",
    "acquisitionsNet", "netOtherFinancingChanges", "taxWithholdingShareBasedComp",
)

#: Each statement: its raw-tier name (``<name>_quarter`` is its quarterly endpoint) and its facts.
STATEMENTS: dict[str, tuple[str, ...]] = {
    "income_statement": INCOME_FACTS,
    "balance_sheet": BALANCE_SHEET_FACTS,
    "cash_flow_statement": CASH_FLOW_FACTS,
}

#: The quarterly endpoints the panels are built from.
QUARTER_ENDPOINTS: tuple[str, ...] = tuple(f"{name}_quarter" for name in STATEMENTS)


def _variable_names() -> dict[str, dict[str, str]]:
    """Return, per statement, each fact's variable name: its own, or ``cashFlow<Fact>`` when the income statement has it."""
    return {
        name: {
            fact: (
                f"cashFlow{fact[0].upper()}{fact[1:]}"
                if name == "cash_flow_statement" and fact in INCOME_FACTS
                else fact
            )
            for fact in facts
        }
        for name, facts in STATEMENTS.items()
    }


#: Per statement, each fact's variable name in the panel.
VARIABLE_NAMES: dict[str, dict[str, str]] = _variable_names()

#: Every fact variable, in statement then vendor order: 178.
FACT_VARIABLES: tuple[str, ...] = tuple(var for names in VARIABLE_NAMES.values() for var in names.values())

#: The variables saying which quarter a cell shows.
DATE_VARIABLES: tuple[str, ...] = ("release_date", "period_end")

#: TTM: the facts carried from the latest quarter (balance-sheet items and the closing cash).
TTM_LATEST: frozenset[str] = frozenset(BALANCE_SHEET_FACTS) | {"endCash"}

#: TTM: the facts taken from the earliest quarter (the opening cash).
TTM_EARLIEST: frozenset[str] = frozenset({"beginningCash"})

#: TTM: the facts averaged over the quarters (weighted share counts); every other fact is summed.
TTM_AVERAGED: frozenset[str] = frozenset(
    {"weightedAverageSharesOutstanding", "weightedAverageSharesOutstandingDiluted"}
)

#: The fiscal periods of a quarterly statement, by number.
_QUARTERS = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}

#: The values a store may hold (``StockFitStatementsConfig.REPORTED``).
REPORTED: tuple[str, ...] = StockFitStatementsConfig.REPORTED

#: The periods a store may hold (``StockFitStatementsConfig.PERIODS``).
PERIODS: tuple[str, ...] = StockFitStatementsConfig.PERIODS


@dataclass(frozen=True)
class FactVersion:
    """One value a fact of one fiscal period had, with the dates of the filing that made it public.

    Parameters
    ----------
    fact : str
        The vendor's fact name.
    value : float
        The value.
    accepted_at : str or None
        The filing's EDGAR acceptance time (ISO 8601, UTC), when known.
    date_filed : str or None
        The filing's date (ISO), when known.

    Examples
    --------
    >>> FactVersion("revenue", 1.0, None, "2024-05-01").date_filed
    '2024-05-01'
    """

    fact: str
    value: float
    accepted_at: str | None
    date_filed: str | None


def fact_versions(period: Mapping) -> list[FactVersion]:
    """Return every value each fact of one statement period had, oldest first per fact.

    The period's facts are walked through its ``sources`` chain, filings
    sorted by filing date and acceptance time: the first value is dated at
    the filing that first supplied it (at the period's original filing when
    that filing was pruned, or when the value comes from a reconstruction or
    is ``derived``), and each filing that changed it adds the value it
    changed it to, at its own dates.

    Parameters
    ----------
    period : Mapping
        One entry of a statement response: ``facts``, ``sources``,
        ``dateFiled``, ``acceptedAt``.

    Returns
    -------
    list of FactVersion
        Fact by fact, in ``facts`` order, each fact's versions in the order
        they became public. A null fact has none.

    Examples
    --------
    Plug Power's fiscal 2018 diluted EPS, -0.36 in the 10-K of 2019-03-13
    and restated to -0.39 by the 10-K of 2021-05-14:

    >>> period = {
    ...     "dateFiled": "2019-03-13", "facts": {"epsDiluted": -0.39},
    ...     "sources": {
    ...         "0001558370-19-001929": {"type": "10-K", "dateFiled": "2019-03-13", "facts": {"epsDiluted": {}}},
    ...         "0001558370-21-007147": {"type": "10-K", "dateFiled": "2021-05-14",
    ...                                  "facts": {"epsDiluted": {"before": -0.36}}},
    ...     },
    ... }
    >>> [(v.value, v.date_filed) for v in fact_versions(period)]
    [(-0.36, '2019-03-13'), (-0.39, '2021-05-14')]
    """
    own = (period.get("acceptedAt"), period.get("dateFiled"))
    filings = sorted(
        (
            (
                (source.get("acceptedAt"), source.get("dateFiled")) if source.get("dateFiled") else own,
                source.get("facts") or {},
            )
            for source in (period.get("sources") or {}).values()
        ),
        key=lambda filing: (filing[0][1] or "", filing[0][0] or ""),
    )
    versions: list[FactVersion] = []
    for fact, current in (period.get("facts") or {}).items():
        if current is None:
            continue
        listings = [(dates, facts[fact] or {}) for dates, facts in filings if fact in facts]
        changes = [(dates, entry["before"]) for dates, entry in listings if "before" in entry]
        first = listings[0][0] if listings and "before" not in listings[0][1] else own
        values = [before for _, before in changes] + [current]
        dated = [first] + [dates for dates, _ in changes]
        versions.extend(
            FactVersion(fact, float(value), accepted, filed)
            for (accepted, filed), value in zip(dated, values)
            if value is not None
        )
    return versions


@dataclass
class _Quarter:
    """One fiscal quarter of one company: when it was first available and every placed version of its variables."""

    period_end: date
    fiscal: int
    release: date
    base: np.datetime64
    versions: dict[str, list[tuple[np.datetime64, float]]]

    def value(self, name: str, at: np.datetime64) -> float | None:
        """Return the variable's value as known at ``at``, or ``None``."""
        found = self.versions.get(name)
        if not found:
            return None
        index = bisect.bisect_right([when for when, _ in found], at)
        return found[index - 1][1] if index else None


def _parse_filed(text: str | None) -> date | None:
    return None if not text else date.fromisoformat(text[:10])


class _Placer:
    """Places filing dates on a calendar by StockFit's rule, remembering each pair it placed."""

    def __init__(self, calendar: TradingCalendar):
        self.calendar = calendar
        self.placed: dict[tuple[str | None, str | None], np.datetime64 | None] = {}

    def add(self, pairs: Iterable[tuple[str | None, str | None]]) -> None:
        """Place every ``(accepted_at, date_filed)`` pair not placed yet."""
        new = sorted({pair for pair in pairs if pair not in self.placed}, key=lambda pair: (pair[1] or "", pair[0] or ""))
        if not new:
            return
        rows = pl.DataFrame(
            {
                "accepted_at": [parse_accepted(accepted) for accepted, _ in new],
                "date_filed": [_parse_filed(filed) for _, filed in new],
                "_pair": list(range(len(new))),
            },
            schema={"accepted_at": pl.Datetime("us", "UTC"), "date_filed": pl.Date, "_pair": pl.Int64},
        )
        found = dict(
            place_released(rows, self.calendar, drop_earlier=False).select("_pair", "available").iter_rows()
        )
        for index, pair in enumerate(new):
            available = found.get(index)
            self.placed[pair] = None if available is None else np.datetime64(available, "ns")

    def __call__(self, accepted: str | None, filed: str | None) -> np.datetime64 | None:
        """Return the available date of a placed pair; ``None`` when not public yet."""
        return self.placed[(accepted, filed)]


def _quarters(statements: Mapping[str, list], place: _Placer, reported: str) -> list[_Quarter]:
    """Return one company's fiscal quarters, each version of their variables placed, by first availability."""
    walked = [
        (name, entry, fact_versions(entry))
        for name, body in statements.items()
        for entry in body or ()
        if entry.get("fiscalPeriod") in _QUARTERS and entry.get("period")
    ]
    place.add(
        pair
        for _, entry, versions in walked
        for pair in [(entry.get("acceptedAt"), entry.get("dateFiled"))]
        + [(version.accepted_at, version.date_filed) for version in versions]
    )
    quarters: dict[str, _Quarter] = {}
    for name, entry, versions in walked:
        base = place(entry.get("acceptedAt"), entry.get("dateFiled"))
        if base is None:
            continue  # not public by the day the raw tier is complete through
        end = date.fromisoformat(entry["period"])
        quarter = quarters.setdefault(entry["period"], _Quarter(
            period_end=end,
            fiscal=int(entry.get("fiscalYear") or end.year) * 4 + _QUARTERS[entry["fiscalPeriod"]],
            release=_parse_filed(entry.get("dateFiled")) or end,
            base=base,
            versions={},
        ))
        quarter.base = min(quarter.base, base)
        names = VARIABLE_NAMES[name]
        for version in versions:
            variable = names.get(version.fact)
            available = place(version.accepted_at, version.date_filed)
            if variable is None or available is None:
                continue
            quarter.versions.setdefault(variable, []).append((available, version.value))
    for quarter in quarters.values():
        for variable, found in quarter.versions.items():
            # A value is never public before its quarter; the chain order breaks ties.
            found = sorted(((max(when, quarter.base), value) for when, value in found), key=lambda item: item[0])
            quarter.versions[variable] = found[:1] if reported == "original" else found
    return sorted(quarters.values(), key=lambda quarter: (quarter.base, quarter.period_end))


def _segments(quarters: list[_Quarter]) -> list[tuple[np.datetime64, np.datetime64 | None, _Quarter]]:
    """Return when each quarter is the latest one filed: ``(from, until, quarter)``, ``until`` exclusive."""
    starts: list[tuple[np.datetime64, _Quarter]] = []
    for quarter in quarters:
        if starts and quarter.period_end <= starts[-1][1].period_end:
            continue  # a later filing of an older quarter never replaces a newer one
        if starts and starts[-1][0] == quarter.base:
            starts.pop()
        starts.append((quarter.base, quarter))
    return [
        (start, starts[index + 1][0] if index + 1 < len(starts) else None, quarter)
        for index, (start, quarter) in enumerate(starts)
    ]


def _between(when: np.datetime64, start: np.datetime64, until: np.datetime64 | None) -> bool:
    return start < when and (until is None or when < until)


def _ttm(window: list[_Quarter], at: np.datetime64) -> dict[str, float | None]:
    """Return the trailing-twelve-month values of four consecutive quarters as known at ``at``."""
    values: dict[str, float | None] = {}
    for name in FACT_VARIABLES:
        if name in TTM_LATEST:
            values[name] = window[-1].value(name, at)
        elif name in TTM_EARLIEST:
            values[name] = window[0].value(name, at)
        else:
            quarters = [quarter.value(name, at) for quarter in window]
            found = [value for value in quarters if value is not None]
            if not found:
                values[name] = None
            elif name in TTM_AVERAGED:
                values[name] = float(np.floor(sum(found) / len(found) + 0.5))
            elif len(found) < len(quarters):
                values[name] = None  # a flow one quarter lacks: a partial sum would understate the year
            else:
                values[name] = float(sum(found))
    return values


def _company_rows(quarters: list[_Quarter], period: str) -> list[tuple[np.datetime64, _Quarter, dict]]:
    """Return one company's rows: ``(available, shown quarter, values)``, one per day anything shown may change."""
    rows = []
    for start, until, quarter in _segments(quarters):
        if period == "quarter":
            changes = {when for found in quarter.versions.values() for when, _ in found if _between(when, start, until)}
            for at in sorted({start} | changes):
                rows.append((at, quarter, {name: quarter.value(name, at) for name in FACT_VARIABLES}))
            continue
        older = [other for other in quarters if other.period_end <= quarter.period_end]
        changes = {other.base for other in older if _between(other.base, start, until)} | {
            when
            for other in older
            for found in other.versions.values()
            for when, _ in found
            if _between(when, start, until)
        }
        for at in sorted({start} | changes):
            known = sorted((other for other in older if other.base <= at), key=lambda other: other.period_end)[-4:]
            fiscal = [other.fiscal for other in known]
            consecutive = len(known) == 4 and fiscal == list(range(fiscal[0], fiscal[0] + 4))
            rows.append((at, quarter, _ttm(known, at) if consecutive else dict.fromkeys(FACT_VARIABLES)))
    return rows


class StockFitStatementsDataset(ReleasePanelDataset):
    """Daily point-in-time panel of StockFit's statements, quarterly or TTM, keyed by CIK.

    See the module docstring for what a cell shows and when.

    Parameters
    ----------
    config : StockFitStatementsConfig
        ``raw_data_dir_path`` is ``<root>/downloads/stockfit``, holding the
        quarterly statement pulls; ``period`` is ``"quarter"`` or ``"ttm"``,
        ``reported`` is ``"as_known"`` or ``"original"``.

    Examples
    --------
    With ``config`` as in the module example::

        ds = StockFitStatementsDataset(config).update()
        ds.panel("2024-01-02", "2024-01-05")["period_end"]
    """

    #: The config class used to rebuild this dataset from a saved config.
    config_cls = StockFitStatementsConfig

    def _normalize_config(self, config: DatasetConfig) -> StockFitStatementsConfig:
        """Normalise as the base class does, then check the statement fields.

        Raises
        ------
        TypeError
            If ``config`` is not a ``StockFitStatementsConfig``.
        ValueError
            If ``period``, ``reported`` or ``stale_after_days`` is invalid.

        Examples
        --------
        Annual statements stay in the raw tier::

            StockFitStatementsDataset(dataclasses.replace(config, period="annual"))
            # ValueError: StockFitStatementsDataset: period 'annual' is not built ...
        """
        config = super()._normalize_config(config)
        if not isinstance(config, StockFitStatementsConfig):
            raise TypeError(f"{self.class_name} needs a StockFitStatementsConfig, got {type(config).__name__}.")
        if config.period not in PERIODS:
            raise ValueError(
                f"{self.class_name}: period {config.period!r} is not built; a store holds "
                f"{' or '.join(map(repr, PERIODS))}. Annual statements stay in the raw tier."
            )
        if config.reported not in REPORTED:
            raise ValueError(f"{self.class_name}: reported must be one of {REPORTED}, got {config.reported!r}.")
        if config.stale_after_days is not None and config.stale_after_days < 1:
            raise ValueError(
                f"{self.class_name}: stale_after_days must be at least 1 or None, got {config.stale_after_days}."
            )
        return config

    # -- the release table ------------------------------------------------------

    def _ciks(self) -> list[str]:
        """Return the companies with a quarterly statement pulled, only those of ``symbols`` when set."""
        root = self.config.raw_data_dir_path
        found = sorted({cik for endpoint in QUARTER_ENDPOINTS for cik in raw.entities(root, endpoint)}, key=int)
        if self.config.symbols is not None:
            wanted = {int(symbol) for symbol in self.config.symbols}
            found = [cik for cik in found if int(cik) in wanted]
        return found

    def _trading_calendar(self) -> TradingCalendar:
        """Return NYSE's sessions through the day the quarterly statements are complete through.

        Raises
        ------
        ValueError
            If a quarterly statement endpoint was never pulled.
        """
        if self._calendar_cache is None:
            self._calendar_cache = nyse_calendar(
                self.config.raw_data_dir_path, QUARTER_ENDPOINTS,
                start=self.config.start_date, end=self.config.end_date, owner=self.class_name,
            )
        return self._calendar_cache

    def _released(self) -> pl.DataFrame:
        """Return the rows the panel ever shows, each at its available date.

        Returns
        -------
        pl.DataFrame
            ``symbol`` (the CIK), ``available``, the ``DATE_VARIABLES`` and
            ``FACT_VARIABLES``, sorted by ``available``.
        """
        root = self.config.raw_data_dir_path
        place = _Placer(self._trading_calendar())
        names = ("symbol", "available", *DATE_VARIABLES, *FACT_VARIABLES)
        columns: dict[str, list] = {name: [] for name in names}
        with Timer(f"{self.class_name}: shown rows"):
            for cik in self._ciks():
                statements = {
                    name: raw.read_pull(root, endpoint, cik) if raw.pulls(root, endpoint, cik) else []
                    for name, endpoint in zip(STATEMENTS, QUARTER_ENDPOINTS)
                }
                quarters = _quarters(statements, place, self.config.reported)
                for at, quarter, values in _company_rows(quarters, self.config.period):
                    columns["symbol"].append(int(cik))
                    columns["available"].append(at)
                    columns["release_date"].append(np.datetime64(quarter.release, "ns"))
                    columns["period_end"].append(np.datetime64(quarter.period_end, "ns"))
                    for name in FACT_VARIABLES:
                        columns[name].append(values[name])
        for name in ("available", *DATE_VARIABLES):
            columns[name] = np.asarray(columns[name], dtype="datetime64[ns]")
        schema = {
            "symbol": pl.Int64,
            "available": pl.Datetime("ns"),
            **dict.fromkeys(DATE_VARIABLES, pl.Datetime("ns")),
            **dict.fromkeys(FACT_VARIABLES, pl.Float64),
        }
        return pl.DataFrame(columns, schema=schema).sort("available", "symbol")

    def _spec(self) -> ReleaseSpec:
        """Return the fact variables (each with its statement and vendor name) and the date variables."""
        variables: dict[str, dict] = {
            variable: {"statement": statement, "fact": fact}
            for statement, names in VARIABLE_NAMES.items()
            for fact, variable in names.items()
        }
        variables["release_date"] = {"description": "original filing date of the quarter shown"}
        variables["period_end"] = {"description": "fiscal period end of the quarter shown"}
        return ReleaseSpec(variables=variables, dates=DATE_VARIABLES)

    def _nothing_shown(self) -> str:
        """Return what an empty window lacks."""
        return f"no {self.config.period} statement row is shown"
