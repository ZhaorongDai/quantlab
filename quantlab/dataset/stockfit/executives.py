"""StockFit's proxy (DEF 14A) executive data as a daily panel on the CIK axis.

Since 2022 a company's proxy statement carries a pay-versus-performance
table, tagged in XBRL: per fiscal year, the CEO's total and "actually paid"
pay, the other named officers' averages, the company's and its peer group's
total shareholder return, and the measure the company itself chose as most
important for setting pay. It also flags four practices around insider
trading and award timing. StockFit serves them per company by CIK, and the
raw tier keeps each endpoint's pulls (``executives_compensation/<cik>/``,
``executives_governance/<cik>/``, ``executives_performance_measures/<cik>/``).

None of them carries a filing date, so a fiscal year's values are placed
``LAG_DAYS`` (120) days after the fiscal year's end, the SEC's deadline for
the proxy, on that NYSE trading day or the next. The end of fiscal year
``Y`` is the period end of the company's annual income statement for
``Y`` (``income_statement_annual/<cik>/``); a year the statements do not
date is left out. The governance flags carry no fiscal year: a pull of
them belongs to the newest fiscal year of the company's compensation as of
that pull, and each fiscal year takes the first pull that belongs to it.
Flags the vendor served before the first pull are not known, so the flags
gain a history only from the first pull on: a year's flags show from the
later of its available date and the day of the pull that served them.

``StockFitExecutivesDataset`` shows, per company and day, the newest fiscal
year available by then:

- ``ceo_total_comp``, ``ceo_actually_paid_comp``, ``avg_officer_total_comp``,
  ``avg_officer_actually_paid_comp``: USD;
- ``company_tsr``, ``peer_group_tsr``: total shareholder return, $100
  invested at the start of the measurement period;
- ``company_selected_measure``: the value of the company's own measure
  (its first, when several are served). Its name and unit stay raw, so the
  value means something only within a company;
- ``insider_trading_policy``, ``mnpi_considered_in_award_timing``,
  ``award_timing_predetermined``, ``mnpi_disclosure_timed_for_comp``: 1.0
  or 0.0, carried forward to later fiscal years until a newer pull
  changes them.

A fiscal year's values stop being shown ``stale_after_days`` after they
became available when no newer year replaced them. Values are as StockFit
serves them today: a proxy that restated an earlier year's figures shows
the restated figures for that year.

Examples
--------
Build the store and read a year::

    config = StockFitExecutivesConfig(
        zarr_file_path="/data/stockfit_executives_1d.zarr",
        raw_data_dir_path="/data/downloads/stockfit",
        start_date="2022-01-01",
    )
    StockFitExecutivesDataset(config).update()
    panel = StockFitExecutivesDataset(config).panel("2025-01-02", "2025-12-31")
    panel["ceo_actually_paid_comp"].sel(symbol=1652044)  # Alphabet's CIK
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import polars as pl

from quantlab.dataset._support.point_in_time import ReleasePanelDataset, ReleaseSpec, TradingCalendar
from quantlab.dataset.config import DatasetConfig, StockFitExecutivesConfig
from quantlab.dataset.stockfit.available import nyse_calendar
from quantlab.dataset.stockfit.period_panel import as_float as as_number
from quantlab.dataset.stockfit.raw import entities, pulls, read_pull, read_pull_as_of
from quantlab.dataset.stockfit.vintages import pull_day

#: The raw tier's endpoints of the per-company proxy data.
EXECUTIVES_COMPENSATION = "executives_compensation"
EXECUTIVES_GOVERNANCE = "executives_governance"
EXECUTIVES_PERFORMANCE_MEASURES = "executives_performance_measures"

#: The raw tier's endpoint of the annual income statements, which date each fiscal year's end.
INCOME_STATEMENT_ANNUAL = "income_statement_annual"

#: Days after a fiscal year's end its proxy values become available.
LAG_DAYS = 120

#: Each pay-versus-performance variable, from its vendor field.
COMPENSATION: dict[str, str] = {
    "ceo_total_comp": "ceoTotalComp",
    "ceo_actually_paid_comp": "ceoActuallyPaidComp",
    "avg_officer_total_comp": "avgOfficerTotalComp",
    "avg_officer_actually_paid_comp": "avgOfficerActuallyPaidComp",
    "company_tsr": "companyTsr",
    "peer_group_tsr": "peerGroupTsr",
}

#: Each governance flag variable, from its vendor field.
GOVERNANCE: dict[str, str] = {
    "insider_trading_policy": "insiderTradingPolicy",
    "mnpi_considered_in_award_timing": "mnpiConsideredInAwardTiming",
    "award_timing_predetermined": "awardTimingPredetermined",
    "mnpi_disclosure_timed_for_comp": "mnpiDisclosureTimedForComp",
}

#: The variable of the company-selected measure's value.
MEASURE = "company_selected_measure"

#: Each variable's unit and meaning, in panel order.
VARIABLES: dict[str, dict] = {
    "ceo_total_comp": {"unit": "USD", "description": "CEO total pay (Summary Compensation Table)"},
    "ceo_actually_paid_comp": {"unit": "USD", "description": "CEO compensation actually paid"},
    "avg_officer_total_comp": {"unit": "USD", "description": "average total pay of the other named officers"},
    "avg_officer_actually_paid_comp": {
        "unit": "USD", "description": "average compensation actually paid to the other named officers",
    },
    "company_tsr": {"unit": "USD per 100 invested", "description": "company total shareholder return"},
    "peer_group_tsr": {"unit": "USD per 100 invested", "description": "peer group total shareholder return"},
    MEASURE: {"description": "value of the company-selected performance measure; its unit varies by company"},
    "insider_trading_policy": {"description": "1.0 when an insider trading policy is adopted"},
    "mnpi_considered_in_award_timing": {"description": "1.0 when MNPI is considered in timing equity awards"},
    "award_timing_predetermined": {"description": "1.0 when equity award dates are predetermined"},
    "mnpi_disclosure_timed_for_comp": {"description": "1.0 when MNPI disclosure is timed for compensation"},
}

#: The endpoints the panel reads; its calendar ends at the oldest one's last complete day.
ENDPOINTS: tuple[str, ...] = (
    EXECUTIVES_COMPENSATION, EXECUTIVES_GOVERNANCE, EXECUTIVES_PERFORMANCE_MEASURES, INCOME_STATEMENT_ANNUAL,
)


def _fiscal_year_ends(vendor_root, cik: str) -> dict[int, date]:
    """Return each fiscal year's end from the company's newest annual income statement."""
    if cik not in entities(vendor_root, INCOME_STATEMENT_ANNUAL):
        return {}
    return {
        int(row["fiscalYear"]): date.fromisoformat(str(row["period"])[:10])
        for row in read_pull(vendor_root, INCOME_STATEMENT_ANNUAL, cik)
        if row.get("fiscalYear") is not None and row.get("period")
    }


def _governance(vendor_root, cik: str) -> dict[int, tuple[datetime, dict]]:
    """Return the governance flags per fiscal year: each year's first pull and its time, by the compensation then."""
    flags: dict[int, tuple[datetime, dict]] = {}
    for pulled_at in pulls(vendor_root, EXECUTIVES_GOVERNANCE, cik):
        compensation = read_pull_as_of(vendor_root, EXECUTIVES_COMPENSATION, cik, pulled_at) or []
        years = [int(row["fiscalYear"]) for row in compensation if row.get("fiscalYear") is not None]
        body = read_pull(vendor_root, EXECUTIVES_GOVERNANCE, cik, pulled_at)
        if years and isinstance(body, dict):
            flags.setdefault(max(years), (pulled_at, {
                name: None if body.get(field) is None else float(bool(body[field]))
                for name, field in GOVERNANCE.items()
            }))
    return flags


def _years(vendor_root) -> pl.DataFrame:
    """Every company's proxy values per fiscal year, each year with its own flags and their pull time."""
    ciks = sorted(
        {cik for endpoint in ENDPOINTS[:3] for cik in entities(vendor_root, endpoint)}, key=int
    )
    records = []
    for cik in ciks:
        years: dict[int, dict] = {}
        if cik in entities(vendor_root, EXECUTIVES_COMPENSATION):
            for row in read_pull(vendor_root, EXECUTIVES_COMPENSATION, cik):
                years.setdefault(int(row["fiscalYear"]), {}).update(
                    {name: as_number(row.get(field)) for name, field in COMPENSATION.items()}
                )
        if cik in entities(vendor_root, EXECUTIVES_PERFORMANCE_MEASURES):
            for row in read_pull(vendor_root, EXECUTIVES_PERFORMANCE_MEASURES, cik):
                year = years.setdefault(int(row["fiscalYear"]), {})
                year.setdefault(MEASURE, as_number(row.get("measureValue")))
        for fiscal_year, (pulled_at, flags) in _governance(vendor_root, cik).items():
            years.setdefault(fiscal_year, {}).update(flags, governance_pulled=pulled_at)
        ends = _fiscal_year_ends(vendor_root, cik)
        records.extend(
            {"cik": int(cik), "fiscal_year": fiscal_year, "fiscal_year_end": ends.get(fiscal_year), **values}
            for fiscal_year, values in years.items()
        )
    schema = {"cik": pl.Int64, "fiscal_year": pl.Int64, "fiscal_year_end": pl.Date}
    schema.update({name: pl.Float64 for name in VARIABLES})
    schema["governance_pulled"] = pl.Datetime("us", "UTC")
    return pl.DataFrame(records, schema=schema).sort("cik", "fiscal_year")


def executive_years(vendor_root) -> pl.DataFrame:
    """Return every company's proxy values per fiscal year, with the fiscal year's end.

    Parameters
    ----------
    vendor_root : str or Path
        ``<download-dir>/stockfit``.

    Returns
    -------
    pl.DataFrame
        ``cik``, ``fiscal_year``, ``fiscal_year_end`` (null when the annual
        statements do not date it) and one column per variable of
        ``VARIABLES``; the governance flags carried forward to later years.

    Examples
    --------
    >>> executive_years("/nonexistent/stockfit").columns[:3]
    ['cik', 'fiscal_year', 'fiscal_year_end']
    """
    return (
        _years(vendor_root)
        .drop("governance_pulled")
        .with_columns(pl.col(name).forward_fill().over("cik") for name in GOVERNANCE)
    )


def _shown(years: list[dict], flags: list[tuple[datetime, int, dict]]) -> list[dict]:
    """Return one company's rows, one per day what it shows may change.

    ``years`` are its fiscal years at their available dates, in order of
    availability; ``flags`` each year's own governance flags at theirs. On
    each day the newest year available by then is shown, with each flag as
    the newest flags available by then set it (a flag a pull leaves unset
    keeps its earlier value).
    """
    flags = sorted(flags, key=lambda flag: (flag[0], flag[1]))
    rows = []
    for day in sorted({year["available"] for year in years} | {at for at, _, _ in flags}):
        shown = [year for year in years if year["available"] <= day]
        if not shown:
            continue
        known = dict.fromkeys(GOVERNANCE)
        for at, _, values in flags:
            if at <= day:
                known.update({name: value for name, value in values.items() if value is not None})
        rows.append({**shown[-1], **known, "available": day})
    return rows


class StockFitExecutivesDataset(ReleasePanelDataset):
    """Daily panel of proxy pay, governance flags and company-selected measures, by CIK.

    Each fiscal year's values are shown from ``LAG_DAYS`` days after its end
    until a newer year's replace them; see the module docstring.

    Parameters
    ----------
    config : StockFitExecutivesConfig
        ``raw_data_dir_path`` is ``<download-dir>/stockfit``. The calendar
        is NYSE's trading days through the oldest last complete day of the
        three executive endpoints and the annual income statements.

    Examples
    --------
    With ``config`` as in the module example::

        ds = StockFitExecutivesDataset(config).update()
        ds.panel("2026-04-29", "2026-04-30")["ceo_total_comp"].sel(symbol=1652044).values
        # array([10800694., 10906079.])
    """

    #: The config class used to rebuild this dataset from a saved config.
    config_cls = StockFitExecutivesConfig

    def _normalize_config(self, config: DatasetConfig) -> StockFitExecutivesConfig:
        """Normalise as the base class does, then check the config class.

        Raises
        ------
        TypeError
            If ``config`` is not a ``StockFitExecutivesConfig``.
        """
        config = super()._normalize_config(config)
        if not isinstance(config, StockFitExecutivesConfig):
            raise TypeError(f"{self.class_name} needs a StockFitExecutivesConfig, got {type(config).__name__}.")
        return config

    def _trading_calendar(self) -> TradingCalendar:
        """Return NYSE's trading days through the oldest endpoint's last complete day (cached)."""
        if self._calendar_cache is None:
            calendars = [
                nyse_calendar(
                    self.config.raw_data_dir_path, endpoint,
                    start=self.config.start_date, end=self.config.end_date, owner=self.class_name,
                )
                for endpoint in ENDPOINTS
            ]
            self._calendar_cache = min(calendars, key=lambda calendar: calendar.through)
        return self._calendar_cache

    def _spec(self) -> ReleaseSpec:
        """Return the variables, stale from each fiscal year's available date."""
        return ReleaseSpec(variables=VARIABLES)

    def _nothing_shown(self) -> str:
        """Return what an empty window lacks."""
        return "no StockFit executive fiscal year is available"

    def _released(self) -> pl.DataFrame:
        """Return every company's fiscal years and flag changes, each at its available date.

        A fiscal year is available ``LAG_DAYS`` after its end; its own
        governance flags at the later of that and the day of the pull that
        first served them (``available.pull_day``): a pull's flags never show
        before the pull.

        Returns
        -------
        pl.DataFrame
            ``symbol`` (the CIK), ``available``, ``release_date`` (the shown
            fiscal year's end plus ``LAG_DAYS``, which staleness counts
            from) and one column per variable, sorted by ``available``; one
            row per company and day anything shown may change, the later
            fiscal year when two land on one day.
        """
        calendar = self._trading_calendar()
        years = _years(self.config.raw_data_dir_path).filter(pl.col("fiscal_year_end").is_not_null())
        placed = calendar.place(years, on="fiscal_year_end", lag=timedelta(days=LAG_DAYS))
        pulled = placed.filter(pl.col("governance_pulled").is_not_null()).select(
            "cik", "fiscal_year", pl.col("available").alias("_year_available"),
            pl.col("governance_pulled").map_elements(pull_day, return_dtype=pl.Date).alias("_pull_day"),
            *GOVERNANCE,
        )
        flag_days = calendar.place(pulled, on="_pull_day").with_columns(
            pl.max_horizontal("available", "_year_available").alias("available")
        )
        flags: dict[int, list] = {}
        for row in flag_days.iter_rows(named=True):
            flags.setdefault(row["cik"], []).append(
                (row["available"], row["fiscal_year"], {name: row[name] for name in GOVERNANCE})
            )
        rows = []
        ordered = placed.drop("governance_pulled").sort("cik", "available", "fiscal_year")
        for (cik,), company in ordered.group_by("cik", maintain_order=True):
            rows += _shown(company.to_dicts(), flags.get(cik, []))
        frame = pl.DataFrame(rows, schema={**ordered.schema}) if rows else ordered.clear()
        return frame.select(
            pl.col("cik").alias("symbol"),
            "available",
            (pl.col("fiscal_year_end") + timedelta(days=LAG_DAYS)).cast(pl.Datetime("ns")).alias("release_date"),
            *VARIABLES,
        ).sort("available", "symbol")
