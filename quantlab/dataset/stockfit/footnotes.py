"""StockFit's 17 footnote tables: one point-in-time panel per footnote, from one class and 17 specs.

StockFit parses the footnotes of 10-K, 10-Q and 20-F filings into 17 tables
(``/api/footnotes/<table>``), pulled per company (CIK), annual and quarterly,
into the raw tier (``quantlab.dataset.stockfit.raw``): the pull of a company's
debt footnote for its quarters is ``stockfit/footnote_debt_structure_quarter/<cik>/``.
``StockFitFootnoteDataset`` turns one footnote's pulls for one period type
into a daily panel on the CIK axis. What the panel shows is data, a
``FootnoteSpec`` per footnote in ``FOOTNOTES``: adding a field is one line of
its spec.

A spec names each variable's place in the response body by a *path*: steps
joined by dots, where a step is a key (``operating``), a key with a selector
(``measures[measure=assets]``: the first element of the list whose fields
match; ``null`` matches a missing or null field), or ``[]``, the list of
periods. Every path holds one ``[]``: each element there is one period
(``period``, ``fiscalYear``, ``dateFiled``) and the value the rest of the path
reaches is that period's. ``"[].cashSpent"`` is the cash spent of each period
of a period-major body; ``"costTypes[member=us-gaap:EmployeeSeveranceMember].history[].charges"``
is the severance charges of each period of a series-major one.

The panels hold a footnote's fixed fields and the vendor's own rollups (the
debt-type totals ``types[]``, the fair-value ``level3Share``, the
concentration aggregates, the whole-plan view of stock compensation).
Per-member lists (segments, tranches, facilities, counterparties, programs,
subsidiaries, properties) stay in the raw tier; quantlab computes no summary
statistic of them (member counts, largest shares, concentration indices):
those are factors. A footnote with no fixed field (business segments, credit
facilities, REIT properties) has a panel of its period and filing date only,
saying which raw period is the latest known each day.

Point in time
-------------
A period's values are available from its ``dateFiled``, the date of the
newest filing that reported the period, on that day (StockFit serves no
acceptance time for footnotes), on NYSE's trading calendar through the last
day the footnote was pulled. StockFit keeps no restatement trail for
footnotes: it serves each period's latest values. So values backfilled from
a company's first pull may include restatements made after the filing date
they are placed at, and the panel flags them (``may_include_restatement``).
From that first pull on, each pull is compared with the one before, and a
changed or new period is shown from its filing date when the filing came
after the previous pull, and from the pull otherwise
(``quantlab.dataset.stockfit.vintages``): that part of the history is point
in time.

On each day a company shows its latest fiscal period whose values are
available by then, in its latest known version; a restated older period
never replaces a newer one. A period released more than
``stale_after_days`` before the day stops being shown.

Examples
--------
Needs a raw tier with the footnote pulls::

    from quantlab.dataset.config import StockFitFootnoteConfig

    config = StockFitFootnoteConfig(
        zarr_file_path="/data/market/stockfit/stockfit_leases_quarter/stockfit_leases_quarter.zarr",
        raw_data_dir_path="/data/downloads/stockfit",
        footnote="leases",
        period="quarter",
    )
    dataset = StockFitFootnoteDataset(config)
    dataset.update()
    dataset.panel("2024-01-01", "2024-12-31")["operating_liability_total"]
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import ClassVar

import polars as pl

from quantlab.dataset._support.point_in_time import (
    ReleasePanelDataset,
    ReleaseSpec,
    TradingCalendar,
)
from quantlab.dataset.config import StockFitFootnoteConfig
from quantlab.dataset.stockfit import raw
from quantlab.dataset.stockfit.available import nyse_calendar, pull_day
from quantlab.dataset.stockfit.raw import as_number
from quantlab.dataset.stockfit.vintages import ServedRow, observed_versions

#: The period types a footnote is pulled for, the ``period`` query parameter.
PERIODS: tuple[str, ...] = StockFitFootnoteConfig.PERIODS

#: The variables every footnote panel has, before its spec's.
COMMON_VARIABLES: dict[str, dict] = {
    "period_end": {"description": "The fiscal period end of the shown values."},
    "filing_date": {"description": "The date of the newest filing StockFit says reported the shown values."},
    "fiscal_year": {"description": "The company's fiscal year of the shown values."},
    "may_include_restatement": {
        "description": (
            "1 when the shown values come from the company's first pull, so they may include "
            "a restatement made after their filing date; 0 when a later pull first served them."
        ),
    },
}

#: The date variables among ``COMMON_VARIABLES``.
DATE_VARIABLES = ("period_end", "filing_date")

#: Splits a path into steps at the dots outside selectors.
_STEP_SPLIT = re.compile(r"\.(?![^\[]*\])")

#: One step: a key, a selector, or both.
_STEP = re.compile(r"^(?P<key>[A-Za-z0-9_]*)(?:\[(?P<selector>[^\]]*)\])?$")


@dataclass(frozen=True)
class FootnoteSpec:
    """What one footnote's panel shows: the API path and a path per variable.

    Parameters
    ----------
    path : str
        The API path, ``/api/footnotes/<table>``.
    fields : Mapping[str, str]
        Each variable, in panel order, and its path in the response body
        (see the module docstring). A value that is missing or not a number
        is NaN.
    periods : str or None, default ``"[]"``
        The path of a list whose every element is a period of the panel even
        when no field has a value in it; ``None`` when the periods are only
        those the fields reach (a series-major body).
    description : str, default ""
        What the footnote is, for the panel's attributes.

    Examples
    --------
    >>> FOOTNOTES["backlog"].fields["remaining_performance_obligation"]
    '[].remainingPerformanceObligation'
    """

    path: str
    fields: Mapping[str, str] = field(default_factory=dict)
    periods: str | None = "[]"
    description: str = ""


def footnote_endpoint(footnote: str, period: str) -> str:
    """Return the raw tier's endpoint directory of one footnote and period type.

    Examples
    --------
    >>> footnote_endpoint("debt_structure", "quarter")
    'footnote_debt_structure_quarter'
    """
    if footnote not in FOOTNOTES:
        raise ValueError(f"unknown StockFit footnote {footnote!r}; expected one of {sorted(FOOTNOTES)}")
    if period not in PERIODS:
        raise ValueError(f"StockFit footnote period must be one of {PERIODS}, got {period!r}")
    return f"footnote_{footnote}_{period}"


# -- the paths -------------------------------------------------------------------


def _steps(path: str) -> list[tuple[str, dict[str, str | None] | None]]:
    """Return a path's steps: (key, selector), the selector ``{}`` for ``[]`` and ``None`` for none."""
    steps = []
    for text in _STEP_SPLIT.split(path):
        match = _STEP.match(text)
        if match is None:
            raise ValueError(f"StockFit footnote path {path!r} has a malformed step {text!r}")
        selector = match.group("selector")
        if selector is None:
            steps.append((match.group("key"), None))
            continue
        conditions: dict[str, str | None] = {}
        for condition in filter(None, selector.split(",")):
            name, _, value = condition.partition("=")
            conditions[name.strip()] = None if value.strip() == "null" else value.strip()
        steps.append((match.group("key"), conditions))
    if sum(1 for _, selector in steps if selector == {}) != 1:
        raise ValueError(f"StockFit footnote path {path!r} must hold one '[]', the list of periods")
    return steps


def _walk(node, steps) -> Iterator[tuple[dict, object]]:
    """Yield (period element, value) for each period the steps reach from ``node``."""
    point = None
    nodes = [(None, node)]
    for key, selector in steps:
        reached = []
        for point, here in nodes:
            if key:
                here = here.get(key) if isinstance(here, dict) else None
            if selector is None:
                reached.append((point, here))
            elif selector == {}:
                reached.extend((element, element) for element in here or () if isinstance(element, dict))
            else:
                chosen = next((
                    element for element in here or ()
                    if isinstance(element, dict)
                    and all(element.get(name) == value for name, value in selector.items())
                ), None)
                reached.append((point, chosen))
        nodes = reached
    for point, value in nodes:
        if point is not None:
            yield point, value


def served_rows(body, spec: FootnoteSpec) -> dict[str, tuple[ServedRow, int | None]]:
    """Return a footnote body's periods: (row, fiscal year) by fiscal period end.

    A period's filing date is the latest ``dateFiled`` among the elements that
    gave it a value (or listed it, by ``spec.periods``).

    Examples
    --------
    >>> body = [{"period": "2025-12-31", "fiscalYear": 2025, "dateFiled": "2026-02-05", "cashSpent": 3}]
    >>> row, year = served_rows(body, FOOTNOTES["buybacks"])["2025-12-31"]
    >>> row.filed, year, row.values["cash_spent"]
    (datetime.date(2026, 2, 5), 2025, 3.0)
    """
    filed: dict[str, date] = {}
    years: dict[str, int | None] = {}
    values: dict[str, dict[str, float]] = {}

    def note(point: dict) -> str | None:
        period, day = point.get("period"), point.get("dateFiled")
        if not period or not day:
            return None
        day = date.fromisoformat(str(day)[:10])
        filed[period] = max(filed.get(period, day), day)
        if years.get(period) is None:
            years[period] = point.get("fiscalYear")
        values.setdefault(period, {})
        return period

    if spec.periods is not None:
        for point, _ in _walk(body, _steps(spec.periods)):
            note(point)
    for name, path in spec.fields.items():
        for point, value in _walk(body, _steps(path)):
            number = as_number(value)
            if math.isnan(number):
                continue
            period = note(point)
            if period is not None:
                values[period][name] = number
    return {
        period: (ServedRow(filed[period], {name: values[period].get(name, math.nan) for name in spec.fields}),
                 years[period])
        for period in filed
    }


# -- the specs --------------------------------------------------------------------


def _each(prefix: str, path: str, names: Mapping[str, str]) -> dict[str, str]:
    """Return one field per (variable suffix, body key) in ``names``, under ``prefix`` and ``path``."""
    return {f"{prefix}{suffix}": f"{path}.{key}" for suffix, key in names.items()}


_LEASE = {
    "right_of_use_asset": "rightOfUseAsset",
    "liability_current": "liabilityCurrent",
    "liability_noncurrent": "liabilityNoncurrent",
    "liability_total": "liabilityTotal",
    "discount_rate": "discountRate",
    "payments_remainder_of_year": "payments.remainderOfYear",
    "payments_year1": "payments.year1",
    "payments_year2": "payments.year2",
    "payments_year3": "payments.year3",
    "payments_year4": "payments.year4",
    "payments_year5": "payments.year5",
    "payments_thereafter": "payments.thereafter",
    "payments_total": "payments.total",
    "payments_discount_excess": "payments.discountExcess",
}

_DEBT_TYPES = {
    "senior_notes": "us-gaap:SeniorNotesMember",
    "commercial_paper": "us-gaap:CommercialPaperMember",
    "convertible_debt": "us-gaap:ConvertibleDebtMember",
    "secured_debt": "us-gaap:SecuredDebtMember",
    "unsecured_debt": "us-gaap:UnsecuredDebtMember",
    "line_of_credit": "us-gaap:LineOfCreditMember",
    "subordinated_debt": "us-gaap:SubordinatedDebtMember",
    "medium_term_notes": "us-gaap:MediumTermNotesMember",
}

_FAIR_VALUE_MEASURES = {
    "assets": "assets",
    "liabilities": "liabilities",
    "cash_equivalents": "cashEquivalents",
    "investments": "investments",
    "afs_securities": "afsSecurities",
    "htm_securities": "htmSecurities",
    "derivative_assets": "derivativeAssets",
    "derivative_liabilities": "derivativeLiabilities",
}

_FAIR_VALUE_LEVELS = {
    "_level1": "level1", "_level2": "level2", "_level3": "level3",
    "_nav": "nav", "_total": "total", "_level3_share": "level3Share",
}

_CONCENTRATIONS = {
    "customer_revenue": ("customer", "revenue"),
    "customer_receivables": ("customer", "receivables"),
    "supplier_cost_of_revenue": ("supplier", "costOfRevenue"),
    "supplier_payables": ("supplier", "payables"),
    "geographic_revenue": ("geographic", "revenue"),
    "product_revenue": ("product", "revenue"),
    "government_contracts_revenue": ("governmentContracts", "revenue"),
}

_RETIREMENT = {
    "_benefit_obligation": "benefitObligation",
    "_funded_status": "fundedStatus",
    "_plan_assets": "planAssets.total",
    "_employer_contributions": "employerContributions",
    "_net_periodic_cost": "netPeriodicCost",
    "_actual_return": "actualReturn",
    "_discount_rate": "assumptions.discountRateBenefitObligation",
    "_expected_return_rate": "assumptions.expectedReturnRate",
}

_RESTRUCTURING_COSTS = {
    "severance": "us-gaap:EmployeeSeveranceMember",
    "facility_closing": "us-gaap:FacilityClosingMember",
    "contract_termination": "us-gaap:ContractTerminationMember",
    "other_restructuring": "us-gaap:OtherRestructuringMember",
}

_LEGAL_ENTITY = {
    "_revenue": "revenue",
    "_operating_income": "operatingIncome",
    "_net_income": "netIncome",
    "_assets": "assets",
    "_liabilities": "liabilities",
    "_equity": "equity",
    "_cash": "cash",
    "_long_term_debt": "longTermDebt",
}

#: Every footnote by name, in the vendor's order: its API path and the path of each variable.
FOOTNOTES: dict[str, FootnoteSpec] = {
    "revenue_segmentation": FootnoteSpec(
        "/api/footnotes/revenue-segmentation",
        {
            "us_revenue": "[].geography.countries[code=US].value",
            "non_us_revenue": "[].geography.regions[member=us-gaap:NonUsMember].value",
        },
        description="Revenue by geography and product; per-country, region and product members stay raw.",
    ),
    "business_segmentation": FootnoteSpec(
        "/api/footnotes/business-segmentation",
        description="Metrics per business segment; the segments stay raw.",
    ),
    "debt_structure": FootnoteSpec(
        "/api/footnotes/debt-structure",
        {
            f"{name}_{suffix}": f"[].types[member={member}].{key}"
            for name, member in _DEBT_TYPES.items()
            for suffix, key in (("face_amount", "faceAmount"), ("carrying_amount", "carryingAmount"))
        },
        description="Debt totals by debt type (the vendor's types[] rollups); the tranches stay raw.",
    ),
    "credit_facilities": FootnoteSpec(
        "/api/footnotes/credit-facilities",
        description="Credit facilities; each facility stays raw.",
    ),
    "leases": FootnoteSpec(
        "/api/footnotes/leases",
        {
            **_each("operating_", "[].operating", _LEASE),
            "operating_cost": "[].operating.cost.operating",
            "operating_variable_cost": "[].operating.cost.variable",
            "operating_short_term_cost": "[].operating.cost.shortTerm",
            "operating_sublease_income": "[].operating.cost.subleaseIncome",
            "operating_cash_paid": "[].operating.cashPaid",
            **_each("finance_", "[].finance", _LEASE),
            "finance_amortization": "[].finance.cost.amortization",
            "finance_interest": "[].finance.cost.interest",
            "finance_principal_paid": "[].finance.principalPaid",
            "finance_interest_paid": "[].finance.interestPaid",
            "not_yet_commenced_total": "[].leaseNotYetCommenced.total",
            "not_yet_commenced_operating": "[].leaseNotYetCommenced.operating",
            "not_yet_commenced_finance": "[].leaseNotYetCommenced.finance",
        },
        description="The ASC 842 lease footnote, operating and finance leases.",
    ),
    "stock_compensation": FootnoteSpec(
        "/api/footnotes/stock-compensation",
        {
            "expense": "[].planTotals.expense",
            "unrecognized_cost": "[].planTotals.unrecognizedCost",
            "tax_withholding_paid": "[].planTotals.taxWithholdingPaid",
            "granted_shares": "[].planTotals.rollForward.granted.shares",
            "granted_fair_value": "[].planTotals.rollForward.granted.waGrantDateFairValue",
            "vested_shares": "[].planTotals.rollForward.vested.shares",
            "vested_fair_value": "[].planTotals.rollForward.vested.waGrantDateFairValue",
            "vested_total_fair_value": "[].planTotals.rollForward.vested.totalFairValue",
            "forfeited_shares": "[].planTotals.rollForward.forfeited.shares",
            "nonvested_shares": "[].planTotals.rollForward.nonvested.shares",
            "nonvested_fair_value": "[].planTotals.rollForward.nonvested.waGrantDateFairValue",
            "options_outstanding": "[].planTotals.options.outstanding",
            "options_exercise_price": "[].planTotals.options.waExercisePrice",
            "options_exercisable": "[].planTotals.options.exercisable",
            "options_granted": "[].planTotals.options.granted",
            "options_exercised": "[].planTotals.options.exercised",
            "shares_available_for_grant": "[].planTotals.plan.sharesAvailableForGrant",
            "shares_authorized": "[].planTotals.plan.sharesAuthorized",
            "volatility": "[].planTotals.assumptions.volatility",
            "risk_free_rate": "[].planTotals.assumptions.riskFreeRate",
            "dividend_yield": "[].planTotals.assumptions.dividendYield",
        },
        description="Share-based compensation, the whole-plan view (planTotals); per-award entries stay raw.",
    ),
    "concentration": FootnoteSpec(
        "/api/footnotes/concentration",
        {
            f"{name}_share": f"groups[riskType={risk},benchmark={benchmark}].aggregate.history[].share"
            for name, (risk, benchmark) in _CONCENTRATIONS.items()
        },
        periods=None,
        description="Concentration shares tagged without a named counterparty; counterparties stay raw.",
    ),
    "retirement_plans": FootnoteSpec(
        "/api/footnotes/retirement-plans",
        {
            f"{kind}{suffix}": f"[].plans[type={kind},location=null].{key}"
            for kind in ("pension", "postretirement")
            for suffix, key in _RETIREMENT.items()
        },
        description="Defined-benefit plans, all locations of each plan type; other buckets stay raw.",
    ),
    "restructuring": FootnoteSpec(
        "/api/footnotes/restructuring",
        {
            f"{name}_{suffix}": f"costTypes[member={member}].history[].{key}"
            for name, member in _RESTRUCTURING_COSTS.items()
            for suffix, key in (("charges", "charges"), ("reserve", "reserve"), ("cash_payments", "cashPayments"))
        },
        periods=None,
        description="Restructuring by standard cost type; named programs stay raw.",
    ),
    "asset_composition": FootnoteSpec(
        "/api/footnotes/asset-composition",
        {
            "land_gross": "[].classes[member=us-gaap:LandMember].gross",
            "construction_in_progress_gross": "[].classes[member=us-gaap:ConstructionInProgressMember].gross",
            "us_long_lived_assets": "[].geography.countries[code=US].value",
            "non_us_long_lived_assets": "[].geography.regions[member=us-gaap:NonUsMember].value",
        },
        description="PP&E by standard class and long-lived assets in and outside the US; other members stay raw.",
    ),
    "buybacks": FootnoteSpec(
        "/api/footnotes/buybacks",
        {
            "cash_spent": "[].cashSpent",
            "repurchased_shares": "[].repurchased.shares",
            "repurchased_value": "[].repurchased.value",
            "average_price": "[].repurchased.avgPricePerShare",
            "program_authorized": "[].program.authorized",
            "program_remaining": "[].program.remaining",
            "program_authorized_shares": "[].program.authorizedShares",
            "program_remaining_shares": "[].program.remainingShares",
            "program_consumed": "[].program.consumed",
            "asr_initial_price": "[].asr.initialPricePaid",
        },
        description="Share repurchases and the board authorization.",
    ),
    "subsidiary_financials": FootnoteSpec(
        "/api/footnotes/subsidiary-financials",
        {
            **_each("parent", "[].subsidiaries[member=srt:ParentCompanyMember]", _LEGAL_ENTITY),
            **_each("subsidiaries", "[].subsidiaries[member=us-gaap:SubsidiariesMember]", _LEGAL_ENTITY),
        },
        description="The parent-only and all-subsidiaries columns; each named legal entity stays raw.",
    ),
    "fair_value_hierarchy": FootnoteSpec(
        "/api/footnotes/fair-value-hierarchy",
        {
            f"{name}{suffix}": f"[].measures[measure={measure}].{key}"
            for name, measure in _FAIR_VALUE_MEASURES.items()
            for suffix, key in _FAIR_VALUE_LEVELS.items()
        },
        description="Recurring fair-value measurements by level, with the vendor's level3Share.",
    ),
    "reit_properties": FootnoteSpec(
        "/api/footnotes/reit-properties",
        description="SEC Schedule III; each property stays raw.",
    ),
    "backlog": FootnoteSpec(
        "/api/footnotes/backlog",
        {
            "remaining_performance_obligation": "[].remainingPerformanceObligation",
            "expected_share_in_window": "[].expectedShareInWindow",
        },
        description="Remaining performance obligations (contracted revenue not yet recognized).",
    ),
    "supplier_finance": FootnoteSpec(
        "/api/footnotes/supplier-finance",
        {
            "obligation": "[].totals.obligation",
            "obligation_current": "[].totals.obligationCurrent",
            "additions": "[].totals.additions",
            "settlements": "[].totals.settlements",
        },
        description="Supplier-finance programs, whole-company totals; per-program entries stay raw.",
    ),
    "workforce_cost": FootnoteSpec(
        "/api/footnotes/workforce-cost",
        {
            "labor_expense": "periods[].laborExpense",
            "employee_compensation": "periods[].employeeCompensation",
            "direct_labor": "periods[].directLabor",
            "other_labor_expense": "periods[].otherLaborExpense",
            "defined_contribution_plan_cost": "periods[].definedContributionPlanCost",
            "accrued_compensation": "periods[].accruedCompensation",
            "accrued_salaries": "periods[].accruedSalaries",
            "accrued_bonuses": "periods[].accruedBonuses",
            "accrued_vacation": "periods[].accruedVacation",
            "accrued_payroll_taxes": "periods[].accruedPayrollTaxes",
            "accrued_employee_benefits": "periods[].accruedEmployeeBenefits",
            "workers_compensation_liability": "periods[].workersCompensationLiability",
            "number_of_employees": "periods[].numberOfEmployees",
        },
        periods="periods[]",
        description="Company-level workforce cost; per-segment and per-caption series stay raw.",
    ),
}


# -- the dataset -------------------------------------------------------------------


class StockFitFootnoteDataset(ReleasePanelDataset):
    """Daily point-in-time panel of one StockFit footnote for one period type, by CIK.

    The config (``StockFitFootnoteConfig``) names the footnote (a key of
    ``FOOTNOTES``) and the period type (``"annual"`` or ``"quarter"``). The
    variables are ``COMMON_VARIABLES`` and the spec's fields; see the module
    docstring for what a day shows.

    Examples
    --------
    Needs a raw tier with the footnote pulls::

        config = StockFitFootnoteConfig(
            zarr_file_path="/data/market/stockfit/stockfit_backlog_annual/stockfit_backlog_annual.zarr",
            raw_data_dir_path="/data/downloads/stockfit",
            footnote="backlog",
            period="annual",
        )
        StockFitFootnoteDataset(config).update()
    """

    config_cls: ClassVar = StockFitFootnoteConfig
    config: StockFitFootnoteConfig

    def _normalize_config(self, config: StockFitFootnoteConfig) -> StockFitFootnoteConfig:
        """Refuse an unknown footnote or period type; see ``RawDataset`` for the rest."""
        config = super()._normalize_config(config)
        footnote_endpoint(config.footnote, config.period)
        return config

    @property
    def footnote(self) -> FootnoteSpec:
        """The spec of the configured footnote.

        Examples
        --------
        With ``config`` naming the leases footnote::

            StockFitFootnoteDataset(config).footnote.path
            # '/api/footnotes/leases'
        """
        return FOOTNOTES[self.config.footnote]

    @property
    def endpoint(self) -> str:
        """The raw tier's endpoint directory the panel reads.

        Examples
        --------
        With ``config`` naming the leases footnote, quarterly::

            StockFitFootnoteDataset(config).endpoint
            # 'footnote_leases_quarter'
        """
        return footnote_endpoint(self.config.footnote, self.config.period)

    def _trading_calendar(self) -> TradingCalendar:
        """Return NYSE's sessions through the day the footnote's raw tier is complete through."""
        if self._calendar_cache is None:
            self._calendar_cache = nyse_calendar(
                self.config.raw_data_dir_path, self.endpoint,
                start=self.config.start_date, end=self.config.end_date, owner=self.class_name,
            )
        return self._calendar_cache

    def _spec(self) -> ReleaseSpec:
        """Return the common variables and the footnote's fields; staleness from the release day."""
        description = self.footnote.description
        variables = {name: dict(attrs) for name, attrs in COMMON_VARIABLES.items()}
        variables.update({name: {"description": f"{description} Path: {path}"}
                          for name, path in self.footnote.fields.items()})
        return ReleaseSpec(variables=variables, dates=DATE_VARIABLES, stamp="release_date")

    def _nothing_shown(self) -> str:
        """Return what an empty window lacks."""
        return f"no {self.config.period} {self.config.footnote} footnote row is shown"

    def _versions(self) -> pl.DataFrame:
        """Return every version of every company's periods, at the day it counts from."""
        root, spec = self.config.raw_data_dir_path, self.footnote
        records = []
        for entity in raw.entities(root, self.endpoint):
            pulls = []
            years: dict[str, int | None] = {}
            for pulled_at in raw.pulls(root, self.endpoint, entity):
                served = served_rows(raw.read_pull(root, self.endpoint, entity, pulled_at), spec)
                years.update({period: year for period, (_, year) in served.items() if year is not None})
                pulls.append((pulled_at, {period: row for period, (row, _) in served.items()}))
            for order, version in enumerate(observed_versions(pulls)):
                day = version.filed if version.seen_at is None else pull_day(version.seen_at)
                records.append({
                    "symbol": int(entity),
                    "date": day,
                    "_order": order,
                    "period_end": date.fromisoformat(version.key),
                    "filing_date": version.filed,
                    "fiscal_year": years.get(version.key),
                    "may_include_restatement": float(version.may_include_restatement),
                    **version.values,
                })
        schema = {
            "symbol": pl.Int64, "date": pl.Date, "_order": pl.Int64, "period_end": pl.Date,
            "filing_date": pl.Date, "fiscal_year": pl.Float64, "may_include_restatement": pl.Float64,
            **{name: pl.Float64 for name in spec.fields},
        }
        return pl.DataFrame(records, schema=schema, orient="row") if records else pl.DataFrame(schema=schema)

    def _released(self) -> pl.DataFrame:
        """Return each company's shown row from each day a version became available.

        At every available day of a company, the row is its latest fiscal
        period available by then, in that period's latest version.
        """
        placed = self._trading_calendar().place(self._versions()).sort("symbol", "available", "_order")
        names = list(self._spec().variables)
        shown = []
        for (symbol,), rows in placed.group_by(["symbol"], maintain_order=True):
            latest: dict[date, dict] = {}
            by_day: dict[datetime, dict] = {}
            for row in rows.iter_rows(named=True):
                latest[row["period_end"]] = row
                newest = latest[max(latest)]
                by_day[row["available"]] = {**{name: newest[name] for name in names},
                                            "symbol": symbol, "available": row["available"],
                                            "release_date": row["available"]}
            shown.extend(by_day.values())
        schema = {
            "symbol": pl.Int64, "available": pl.Datetime("ns"), "release_date": pl.Datetime("ns"),
            **{name: placed.schema[name] for name in names},
        }
        frame = pl.DataFrame(shown, schema=schema, orient="row") if shown else pl.DataFrame(schema=schema)
        return frame.with_columns(pl.col(DATE_VARIABLES).cast(pl.Datetime("ns"))).sort("available")


__all__ = [
    "COMMON_VARIABLES",
    "FOOTNOTES",
    "PERIODS",
    "FootnoteSpec",
    "StockFitFootnoteDataset",
    "footnote_endpoint",
    "served_rows",
]
