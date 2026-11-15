"""StockFit's Schedule 13D/13G filings as a daily panel on the CIK axis.

A holder of more than 5% of a company's stock files a Schedule 13D (an
active holder, who may seek control) or a 13G (a passive one), and amends it
when the stake changes. Each company's filings
(``/api/ownership/beneficial-owners/history``, every filing) are pulled into
the raw tier by the company's CIK (``beneficial_owners_history/<cik>/``).
A row is one reporting person of one filing: joint filers (a fund and its
general partner) share an accession number and usually report the same
stake. StockFit serves the structured filings the SEC has required since
December 2024, so the history starts in 2025.

The row's ``reportDate`` is not used: it is neither the event date nor the
EDGAR filing date, but a time in UTC of varying precision, sometimes hours
after the SEC accepted the filing. A filing is placed by its accession
number in the company's filing list (``quantlab.dataset.stockfit.filings``),
which gives its filing date and SEC acceptance time, on its available date
(``quantlab.dataset.stockfit.available``). A filing the list does not hold
(yet) is left out until a refresh of the list brings it.

``StockFitBeneficialOwnersDataset`` shows each filing on its available day
only:

- ``filings_13d``, ``filings_13g``: the 13D and 13G filings of the day,
  amendments included, one per accession number;
- ``activist``: True when a filing of the day is flagged activist;
- ``percent_of_class``, ``shares_owned``: the largest stake a filing of
  the day reports, in percent of the class and in shares. NaN on a day
  without a filing; 0 is a holder reporting it fell to nothing.

Examples
--------
Build the store and read a year::

    config = StockFitBeneficialOwnersConfig(
        zarr_file_path="/data/stockfit_beneficial_owners_1d.zarr",
        raw_data_dir_path="/data/downloads/stockfit",
        start_date="2025-01-01",
    )
    StockFitBeneficialOwnersDataset(config).update()
    panel = StockFitBeneficialOwnersDataset(config).panel("2025-01-02", "2025-12-31")
    panel["filings_13d"].sel(symbol=1093691)  # Plug Power's CIK
"""

from __future__ import annotations

import math

import polars as pl

from quantlab.dataset._support.point_in_time import FilingPanelDataset, TradingCalendar
from quantlab.dataset.config import DatasetConfig, StockFitBeneficialOwnersConfig
from quantlab.dataset.stockfit.available import nyse_calendar, place_released
from quantlab.dataset.stockfit.filings import FILINGS, read_filings
from quantlab.dataset.stockfit.raw import entities, read_pull

#: The raw tier's endpoint of the per-company 13D/13G filings.
BENEFICIAL_OWNERS = "beneficial_owners_history"

#: The columns of ``read_beneficial_owners``.
OWNER_SCHEMA: dict[str, pl.DataType] = {
    "cik": pl.Int64,
    "accession_number": pl.String,
    "form": pl.String,
    "reporting_person": pl.String,
    "activist": pl.Boolean,
    "percent_of_class": pl.Float64,
    "shares_owned": pl.Float64,
}


def read_beneficial_owners(vendor_root, ciks=None) -> pl.DataFrame:
    """Return the newest pull of each company's 13D/13G filings, one row per reporting person.

    Parameters
    ----------
    vendor_root : str or Path
        ``<download-dir>/stockfit``.
    ciks : iterable of int, optional
        The companies to read; every one with a pull when omitted.

    Returns
    -------
    pl.DataFrame
        ``OWNER_SCHEMA``'s columns: the company's ``cik``, the
        ``accession_number``, the ``form`` (``13D`` or ``13G``, an
        amendment's ``/A`` dropped), the ``reporting_person``, the
        ``activist`` flag, ``percent_of_class`` and ``shares_owned``.

    Examples
    --------
    >>> read_beneficial_owners("/nonexistent/stockfit").columns[:3]
    ['cik', 'accession_number', 'form']
    """
    present = entities(vendor_root, BENEFICIAL_OWNERS)
    wanted = present if ciks is None else [str(int(cik)) for cik in ciks if str(int(cik)) in present]

    def number(value) -> float | None:
        return None if value is None else float(value)

    records = [
        {
            "cik": int(entity),
            "accession_number": row.get("accessionNumber"),
            "form": str(row.get("filingType") or "").removesuffix("/A") or None,
            "reporting_person": row.get("reportingPersonName"),
            "activist": bool(row.get("activist")),
            "percent_of_class": number(row.get("percentOfClass")),
            "shares_owned": number(row.get("aggregateAmountOwned")),
        }
        for entity in wanted
        for row in read_pull(vendor_root, BENEFICIAL_OWNERS, entity)
    ]
    return pl.DataFrame(records, schema=OWNER_SCHEMA)


class StockFitBeneficialOwnersDataset(FilingPanelDataset):
    """Daily panel of 13D/13G filings, by CIK, each on its filing's available date.

    See the module docstring for the variables and the placement.

    Parameters
    ----------
    config : StockFitBeneficialOwnersConfig
        ``raw_data_dir_path`` is ``<download-dir>/stockfit``. The calendar
        is NYSE's trading days through the older of the two endpoints'
        last complete days (the 13D/13G filings and the filing lists).

    Examples
    --------
    With ``config`` as in the module example::

        ds = StockFitBeneficialOwnersDataset(config).update()
        ds.panel("2026-07-29", "2026-07-30")["percent_of_class"].sel(symbol=1093691).values
        # array([nan, 12.8])
    """

    #: The config class used to rebuild this dataset from a saved config.
    config_cls = StockFitBeneficialOwnersConfig

    def _normalize_config(self, config: DatasetConfig) -> StockFitBeneficialOwnersConfig:
        """Normalise as the base class does, then check the config class.

        Raises
        ------
        TypeError
            If ``config`` is not a ``StockFitBeneficialOwnersConfig``.
        """
        config = super()._normalize_config(config)
        if not isinstance(config, StockFitBeneficialOwnersConfig):
            raise TypeError(f"{self.class_name} needs a StockFitBeneficialOwnersConfig, got {type(config).__name__}.")
        return config

    def _trading_calendar(self) -> TradingCalendar:
        """Return NYSE's trading days through the older endpoint's last complete day (cached)."""
        if self._calendar_cache is None:
            self._calendar_cache = nyse_calendar(
                self.config.raw_data_dir_path, (BENEFICIAL_OWNERS, FILINGS),
                start=self.config.start_date, end=self.config.end_date, owner=self.class_name,
            )
        return self._calendar_cache

    def _empty(self) -> dict[str, object]:
        """Return no filing: 0 counts, no activist, no stake."""
        return {
            "filings_13d": 0.0, "filings_13g": 0.0, "activist": False,
            "percent_of_class": math.nan, "shares_owned": math.nan,
        }

    def _attrs(self) -> dict[str, dict]:
        """Return each variable's unit and meaning."""
        return {
            "filings_13d": {"unit": "filings", "description": "Schedule 13D filings, amendments included"},
            "filings_13g": {"unit": "filings", "description": "Schedule 13G filings, amendments included"},
            "activist": {"description": "a 13D/13G filing of the day is flagged activist"},
            "percent_of_class": {"unit": "percent", "description": "largest stake a filing of the day reports"},
            "shares_owned": {"unit": "shares", "description": "largest share count a filing of the day reports"},
        }

    def _nothing_shown(self) -> str:
        """Return what an empty window lacks."""
        return "no StockFit 13D/13G filing is available"

    def _released(self) -> pl.DataFrame:
        """Return every company's 13D/13G filings per available day.

        Returns
        -------
        pl.DataFrame
            ``symbol`` (the CIK), ``available`` and one column per variable,
            one row per pair, sorted by both.
        """
        root = self.config.raw_data_dir_path
        owners = read_beneficial_owners(root)
        filed = read_filings(root, ciks=owners.get_column("cik").unique().to_list()).select(
            "cik", "accession_number", "date_filed", "accepted_at"
        )
        filings = (
            owners.group_by("cik", "accession_number")
            .agg(
                pl.col("form").first(),
                pl.col("activist").any(),
                pl.col("percent_of_class").max(),
                pl.col("shares_owned").max(),
            )
            .join(filed, on=["cik", "accession_number"], how="inner")
        )
        placed = place_released(filings, self._trading_calendar())
        return (
            placed.group_by(pl.col("cik").alias("symbol"), "available")
            .agg(
                (pl.col("form") == "13D").sum().cast(pl.Float64).alias("filings_13d"),
                (pl.col("form") == "13G").sum().cast(pl.Float64).alias("filings_13g"),
                pl.col("activist").any(),
                pl.col("percent_of_class").max(),
                pl.col("shares_owned").max(),
            )
            .sort("available", "symbol")
        )
