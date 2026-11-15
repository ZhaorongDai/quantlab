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
gain a history only from the first pull on.

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

from datetime import date, timedelta

import polars as pl

from quantlab.dataset._support.point_in_time import ReleasePanelDataset, ReleaseSpec, TradingCalendar
from quantlab.dataset.config import DatasetConfig, StockFitExecutivesConfig
from quantlab.dataset.stockfit.available import nyse_calendar
from quantlab.dataset.stockfit.raw import entities, pulls, read_pull, read_pull_as_of

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


def _number(value) -> float | None:
    return None if value is None else float(value)


def _fiscal_year_ends(vendor_root, cik: str) -> dict[int, date]:
    """Return each fiscal year's end from the company's newest annual income statement."""
    if cik not in entities(vendor_root, INCOME_STATEMENT_ANNUAL):
        return {}
    return {
        int(row["fiscalYear"]): date.fromisoformat(str(row["period"])[:10])
        for row in read_pull(vendor_root, INCOME_STATEMENT_ANNUAL, cik)
        if row.get("fiscalYear") is not None and row.get("period")
    }


def _governance(vendor_root, cik: str) -> dict[int, dict]:
    """Return the governance flags per fiscal year: each year's first pull, by the compensation then."""
    flags: dict[int, dict] = {}
    for pulled_at in pulls(vendor_root, EXECUTIVES_GOVERNANCE, cik):
        compensation = read_pull_as_of(vendor_root, EXECUTIVES_COMPENSATION, cik, pulled_at) or []
        years = [int(row["fiscalYear"]) for row in compensation if row.get("fiscalYear") is not None]
        body = read_pull(vendor_root, EXECUTIVES_GOVERNANCE, cik, pulled_at)
        if years and isinstance(body, dict):
            flags.setdefault(max(years), {
                name: None if body.get(field) is None else float(bool(body[field]))
                for name, field in GOVERNANCE.items()
            })
    return flags


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
    ciks = sorted(
        {cik for endpoint in ENDPOINTS[:3] for cik in entities(vendor_root, endpoint)}, key=int
    )
    records = []
    for cik in ciks:
        years: dict[int, dict] = {}
        if cik in entities(vendor_root, EXECUTIVES_COMPENSATION):
            for row in read_pull(vendor_root, EXECUTIVES_COMPENSATION, cik):
                years.setdefault(int(row["fiscalYear"]), {}).update(
                    {name: _number(row.get(field)) for name, field in COMPENSATION.items()}
                )
        if cik in entities(vendor_root, EXECUTIVES_PERFORMANCE_MEASURES):
            for row in read_pull(vendor_root, EXECUTIVES_PERFORMANCE_MEASURES, cik):
                year = years.setdefault(int(row["fiscalYear"]), {})
                year.setdefault(MEASURE, _number(row.get("measureValue")))
        for fiscal_year, flags in _governance(vendor_root, cik).items():
            years.setdefault(fiscal_year, {}).update(flags)
        ends = _fiscal_year_ends(vendor_root, cik)
        records.extend(
            {"cik": int(cik), "fiscal_year": fiscal_year, "fiscal_year_end": ends.get(fiscal_year), **values}
            for fiscal_year, values in years.items()
        )
    schema = {"cik": pl.Int64, "fiscal_year": pl.Int64, "fiscal_year_end": pl.Date}
    schema.update({name: pl.Float64 for name in VARIABLES})
    return (
        pl.DataFrame(records, schema=schema)
        .sort("cik", "fiscal_year")
        .with_columns(pl.col(name).forward_fill().over("cik") for name in GOVERNANCE)
    )


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
        """Return every company's fiscal years, each at its available date.

        Returns
        -------
        pl.DataFrame
            ``symbol`` (the CIK), ``available``, ``release_date`` (the
            fiscal year's end plus ``LAG_DAYS``, which staleness counts
            from) and one column per variable, sorted by ``available``; one
            row per company and day, the later fiscal year when two land on
            one day.
        """
        years = executive_years(self.config.raw_data_dir_path).filter(pl.col("fiscal_year_end").is_not_null())
        placed = self._trading_calendar().place(years, on="fiscal_year_end", lag=timedelta(days=LAG_DAYS))
        return (
            placed.sort("cik", "available", "fiscal_year")
            .unique(["cik", "available"], keep="last", maintain_order=True)
            .select(
                pl.col("cik").alias("symbol"),
                "available",
                (pl.col("fiscal_year_end") + timedelta(days=LAG_DAYS)).cast(pl.Datetime("ns")).alias("release_date"),
                *VARIABLES,
            )
            .sort("available", "symbol")
        )
