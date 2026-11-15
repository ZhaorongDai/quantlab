"""StockFit's SEC filing lists, and the filing panel built from them.

Each company's filing list (``/api/filings``, every form type, newest first)
is pulled into the raw tier by CIK (``filings/<cik>/<stamp>/``). It holds the
metadata of every filing the company made: form type, accession number,
filing date, the SEC's acceptance time, the period, the amendment flag and,
for an 8-K, its items and event names. ``read_filings`` returns it as one
frame, the table accession lookups read. Item text stays raw.

``StockFitFilingsDataset`` is the filing panel: per company (by CIK) and
NYSE trading day, a filing counts on its available date only
(``quantlab.dataset.stockfit.available``: accepted before the session's close
that day, after the close the next trading day, a weekend or holiday filing
the next trading day). Its variables:

- ``form_<type>``: how many filings of each form StockFit names (``FORMS``;
  ``form_10_k``, ``form_def_14a``, ``form_13f_hr``, ...), amendments
  included, and ``form_other`` for every other form (Form 144, S-1, ...);
- ``event_<name>``: True when an 8-K of the day lists the event
  (``EVENTS``; ``event_earnings`` is Item 2.02);
- ``amendment``: True when any filing of the day is an amendment.

A day without a filing holds 0 and False, never NaN.

Examples
--------
Build the store and see which companies reported earnings on a day::

    config = StockFitFilingsConfig(
        zarr_file_path="/data/quantlab/market/stockfit/stockfit_filings_1d/stockfit_filings_1d.zarr",
        raw_data_dir_path="/data/quantlab/downloads/stockfit",
        start_date="2000-01-01",
    )
    StockFitFilingsDataset(config).update()
    panel = StockFitFilingsDataset(config).panel("2026-10-01", "2026-10-31")
    panel["event_earnings"].sel(symbol=1652044)
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from quantlab.dataset._support.point_in_time import FilingPanelDataset, TradingCalendar
from quantlab.dataset.config import StockFitFilingsConfig
from quantlab.dataset.stockfit.available import nyse_calendar, place_released
from quantlab.dataset.stockfit.raw import entities, read_pull

#: The raw tier's endpoint of the per-company filing lists.
FILINGS = "filings"

#: The form types StockFit names (its ``FilingType``), each a count variable.
FORMS: tuple[str, ...] = (
    "10-K", "10-Q", "20-F", "8-K", "DEF 14A", "DEFA14A", "13D", "13G", "3", "4", "5",
    "13F-HR", "NPORT-P", "N-CEN", "EFFECT", "25-NSE", "ARS",
)

#: The 8-K events StockFit names (its ``FilingEvent``), each a flag variable.
EVENTS: tuple[str, ...] = (
    "material_agreement", "agreement_termination", "bankruptcy", "mine_safety",
    "acquisition_disposition", "earnings", "debt_obligation", "debt_acceleration",
    "restructuring", "impairment", "delisting", "unregistered_sale", "rights_modification",
    "auditor_change", "restatement", "control_change", "officer_change", "bylaws_amendment",
    "trading_suspension", "ethics_amendment", "shell_status_change", "shareholder_vote",
    "shareholder_nomination", "abs_material", "servicer_change", "credit_enhancement",
    "distribution_failure", "securities_act_update", "reg_fd", "other", "financial_exhibits",
)

#: The count variable of the forms not in ``FORMS``.
OTHER_FORMS = "form_other"


def form_variable(form: str) -> str:
    """Return the count variable of a form type.

    Examples
    --------
    >>> form_variable("DEF 14A"), form_variable("13F-HR")
    ('form_def_14a', 'form_13f_hr')
    """
    return "form_" + "".join(c if c.isalnum() else "_" for c in form.lower())


#: The columns of ``read_filings``.
FILING_SCHEMA: dict[str, pl.DataType] = {
    "cik": pl.Int64,
    "accession_number": pl.String,
    "type": pl.String,
    "date_filed": pl.Date,
    "accepted_at": pl.Datetime("us", "UTC"),
    "period_end_date": pl.Date,
    "amendment": pl.Boolean,
    "events": pl.List(pl.String),
}


def read_filings(vendor_root: str | Path, ciks=None) -> pl.DataFrame:
    """Return the filings of the newest filing list of each company, one row per filing.

    Parameters
    ----------
    vendor_root : str or Path
        ``<download-dir>/stockfit``.
    ciks : iterable of int, optional
        The companies to read; every company with a pulled filing list
        when omitted.

    Returns
    -------
    pl.DataFrame
        ``FILING_SCHEMA``'s columns: the filer's ``cik``, the
        ``accession_number``, the form ``type`` (an amendment keeps its
        form's type and has ``amendment`` set), ``date_filed`` (the EDGAR
        filing date), ``accepted_at`` (the SEC's acceptance time, UTC; null
        when the vendor has none), ``period_end_date`` and an 8-K's
        ``events`` (empty for other forms). One row per company and
        accession number, the vendor's first.

    Examples
    --------
    >>> read_filings("/nonexistent/stockfit").columns[:3]
    ['cik', 'accession_number', 'type']

    The acceptance time of one filing, with a raw tier on disk::

        rows = read_filings("/data/quantlab/downloads/stockfit", ciks=[1652044])
        rows.filter(pl.col("accession_number") == "0001652044-26-000084")["accepted_at"]
    """
    wanted = entities(vendor_root, FILINGS) if ciks is None else [str(int(cik)) for cik in ciks]
    present = set(entities(vendor_root, FILINGS))
    records = [
        {
            "cik": int(entity),
            "accession_number": row.get("accessionNumber"),
            "type": row.get("type"),
            "date_filed": row.get("dateFiled"),
            "accepted_at": row.get("acceptedAt"),
            "period_end_date": row.get("periodEndDate"),
            "amendment": bool(row.get("amendment")),
            "events": list(row.get("events") or ()),
        }
        for entity in wanted
        if entity in present
        for row in read_pull(vendor_root, FILINGS, entity)
    ]
    text = {"date_filed": pl.String, "accepted_at": pl.String, "period_end_date": pl.String}
    frame = pl.DataFrame(records, schema={**FILING_SCHEMA, **text})
    return (
        frame.with_columns(
            pl.col("date_filed").str.slice(0, 10).str.to_date(strict=False),
            pl.col("period_end_date").str.slice(0, 10).str.to_date(strict=False),
            pl.col("accepted_at").str.to_datetime(time_zone="UTC", time_unit="us", strict=False),
        )
        .unique(["cik", "accession_number"], keep="first", maintain_order=True)
        .select(list(FILING_SCHEMA))
    )


class StockFitFilingsDataset(FilingPanelDataset):
    """Daily panel of each company's SEC filings, by CIK, each filing on its available date.

    See the module docstring for the variables and the placement.

    Parameters
    ----------
    config : StockFitFilingsConfig
        ``raw_data_dir_path`` is ``<download-dir>/stockfit``. The calendar
        is NYSE's trading days through the day before the newest pull of
        the filing lists.

    Examples
    --------
    With ``config`` as in the module example::

        ds = StockFitFilingsDataset(config).update()
        ds.panel("2026-10-28", "2026-10-29")["form_10_q"].sel(symbol=1652044).values
        # array([0, 1])
    """

    #: The config class used to rebuild this dataset from a saved config.
    config_cls = StockFitFilingsConfig

    def _trading_calendar(self) -> TradingCalendar:
        """Return NYSE's trading days through the filing lists' last complete day (cached)."""
        if self._calendar_cache is None:
            self._calendar_cache = nyse_calendar(
                self.config.raw_data_dir_path, FILINGS,
                start=self.config.start_date, end=self.config.end_date, owner=self.class_name,
            )
        return self._calendar_cache

    def _empty(self) -> dict[str, object]:
        """Return 0 for each count and False for each flag: nothing was filed."""
        return {
            **{form_variable(form): 0 for form in FORMS},
            OTHER_FORMS: 0,
            **{f"event_{event}": False for event in EVENTS},
            "amendment": False,
        }

    def _attrs(self) -> dict[str, dict]:
        """Return each variable's description."""
        return {
            **{form_variable(form): {"unit": "filings", "description": f"Filings of form {form}, amendments included."}
               for form in FORMS},
            OTHER_FORMS: {"unit": "filings", "description": "Filings of a form not counted on its own."},
            **{f"event_{event}": {"description": f"An 8-K of the day lists the {event!r} event."} for event in EVENTS},
            "amendment": {"description": "A filing of the day is an amendment."},
        }

    def _nothing_shown(self) -> str:
        """Return what an empty window lacks."""
        return "no StockFit filing is available"

    def _released(self) -> pl.DataFrame:
        """Return every company's filings per available day, counted and flagged.

        Returns
        -------
        pl.DataFrame
            ``symbol`` (the CIK), ``available`` and one column per variable,
            one row per pair, sorted by both.
        """
        filings = read_filings(self.config.raw_data_dir_path).with_columns(
            pl.col("type").str.strip_suffix("/A").alias("_form"),
            (pl.col("amendment") | pl.col("type").str.ends_with("/A")).alias("amendment"),
        )
        placed = place_released(filings, self._trading_calendar())
        named = pl.col("_form").is_in(list(FORMS))
        return (
            placed.group_by(pl.col("cik").alias("symbol"), "available")
            .agg(
                *((pl.col("_form") == form).sum().cast(pl.Int64).alias(form_variable(form)) for form in FORMS),
                (~named).sum().cast(pl.Int64).alias(OTHER_FORMS),
                *(pl.col("events").list.contains(event).any().alias(f"event_{event}") for event in EVENTS),
                pl.col("amendment").any(),
            )
            .sort("available", "symbol")
        )
