"""StockFit 13F institutional ownership as a daily panel, keyed by CUSIP.

Institutional managers file their 13F holdings (form 13F-HR) up to 45 days
after each calendar quarter's end. StockFit serves them as holder rows, one
per manager and security, each naming the security's 9-character CUSIP and
the accession number of the filing it came from. The raw tier holds them per
company and quarter (``institutional_holders``, every share class of the
company) and, per manager the rows name, its 13F filings with their filing
dates (``manager_13f_filings``); ``quantlab.dataset.stockfit.raw`` lays them
out. The managers' own portfolios (``manager_portfolio``) stay raw files:
this panel does not read them.

``StockFitHoldingsDataset`` sums the rows by security and quarter. The
symbol is the CUSIP, so the share classes of one company are separate
symbols (ADR 0054), and the axis is text even for a CUSIP of digits only.
Only share holdings count: a row of a principal amount (``shareType``
``PRN``) or an option (``putCall`` set) is left out. Variables:

- ``holders``: managers holding the security;
- ``shares_held``: the shares they hold;
- ``value_held``: the value of those shares in USD, as reported for the
  quarter end;
- ``quarter_end``: the quarter shown.

Available dates. A quarter is shown from the first NYSE trading day on or
after its end + 45 days (``AVAILABILITY_LAG``) until the next quarter's, the
same quarter for every security, so a partial quarter is never shown and a
security with no row in the shown quarter shows nothing. Within the shown
quarter a row counts from the later of that day and its filing's available
date: StockFit merges an amendment (13F-HR/A) into the quarter it amends and
gives the amendment's accession on the row, so a position first disclosed in
a late amendment counts only from the amendment's filing. A filing is
available on the day it was accepted when accepted before that NYSE
session's close (13:00 on a half day), and on the next trading day
otherwise; a filing without an acceptance time is available on its filing
date. A row whose accession is in no pulled filing list counts from the day
after its holder rows were pulled, never earlier.

Look-ahead caveats. StockFit serves each quarter as it holds it now: an
amendment replaces the original row, so the original value is never shown
and the position is missing from the quarter until the amendment's date; a
late amendment of a quarter no longer shown is not shown at all.

Examples
--------
Build the store and read the holders of Alphabet's class A shares::

    config = StockFitHoldingsConfig(
        zarr_file_path="/data/market/stockfit/stockfit_holdings_1d/stockfit_holdings_1d.zarr",
        raw_data_dir_path="/data/downloads/stockfit",
    )
    StockFitHoldingsDataset(config).update()
    panel = StockFitHoldingsDataset(config).panel("2024-01-02", "2024-12-31")
    panel["holders"].sel(symbol="02079K305")
"""

from __future__ import annotations

from datetime import timedelta

import pandas as pd
import polars as pl
import xarray as xr

from quantlab.dataset._support.point_in_time import (
    ReleasePanelDataset,
    ReleaseSpec,
    TradingCalendar,
    on_grid,
    to_panel,
)
from quantlab.dataset.config import DatasetConfig, StockFitHoldingsConfig
from quantlab.dataset.stockfit.available import nyse_calendar, release_day
from quantlab.dataset.stockfit.raw import (
    INSTITUTIONAL_HOLDERS,
    MANAGER_13F_FILINGS,
    entities,
    pulls,
    read_pull,
    split_quarter_entity,
)

#: Days after a quarter's end until its 13F filings are all due.
AVAILABILITY_LAG = timedelta(days=45)

#: The panel's variables and their attributes.
VARIABLES: dict[str, dict] = {
    "holders": {"unit": "managers", "description": "13F managers holding the security"},
    "shares_held": {"unit": "shares", "description": "shares 13F managers hold"},
    "value_held": {
        "unit": "USD",
        "description": "value of the shares 13F managers hold, as reported for the quarter end",
    },
    "quarter_end": {"description": "calendar quarter end of the 13F holdings shown"},
}

_HOLDER_SCHEMA = {
    "symbol": pl.String,
    "manager": pl.Int64,
    "quarter_end": pl.Date,
    "accession": pl.String,
    "shares": pl.Float64,
    "value": pl.Float64,
    "pulled": pl.Date,
}


class StockFitHoldingsDataset(ReleasePanelDataset):
    """Daily panel of 13F holders and shares held per CUSIP, each quarter from quarter end + 45 days.

    See the module docstring for the variables and the available dates.

    Parameters
    ----------
    config : StockFitHoldingsConfig
        ``raw_data_dir_path`` is ``<root>/downloads/stockfit``.

    Examples
    --------
    With ``config`` as in the module example::

        ds = StockFitHoldingsDataset(config).update()
        sorted(ds.panel("2024-01-02", "2024-01-05").data_vars)
        # ['holders', 'quarter_end', 'shares_held', 'value_held']
    """

    #: The config class used to rebuild this dataset from a saved config.
    config_cls = StockFitHoldingsConfig

    #: A CUSIP is text, even when it is all digits.
    TEXT_SYMBOLS = True

    def _normalize_config(self, config: DatasetConfig) -> StockFitHoldingsConfig:
        """Normalise as the base class does, then check the config class.

        Raises
        ------
        TypeError
            If ``config`` is not a ``StockFitHoldingsConfig``.
        """
        config = super()._normalize_config(config)
        if not isinstance(config, StockFitHoldingsConfig):
            raise TypeError(f"{self.class_name} needs a StockFitHoldingsConfig, got {type(config).__name__}.")
        return config

    # -- the raw tier ------------------------------------------------------------

    def _holder_rows(self) -> pl.DataFrame:
        """Return every share holding of the newest holder pulls, with the day each pull was made."""
        root = self.config.raw_data_dir_path
        frames = []
        for entity in entities(root, INSTITUTIONAL_HOLDERS):
            _, quarter = split_quarter_entity(entity)
            pulled = pulls(root, INSTITUTIONAL_HOLDERS, entity)[-1].date()
            rows = [
                (row["cusip"], int(row["managerCik"]), quarter, row.get("accessionNumber"),
                 float(row.get("shares") or 0.0), float(row.get("value") or 0.0), pulled)
                for row in read_pull(root, INSTITUTIONAL_HOLDERS, entity)
                if row.get("cusip") and row.get("managerCik") is not None
                and row.get("shareType") == "SH" and row.get("putCall") is None
            ]
            if rows:
                frames.append(pl.DataFrame(rows, schema=_HOLDER_SCHEMA, orient="row"))
        frame = pl.concat(frames) if frames else pl.DataFrame(schema=_HOLDER_SCHEMA)
        if self.config.symbols is not None:
            frame = frame.filter(pl.col("symbol").is_in([str(s) for s in self.config.symbols]))
        return frame

    def _filings(self) -> pl.DataFrame:
        """Return the day each pulled 13F filing counts from: ``accession``, ``public``."""
        root = self.config.raw_data_dir_path
        rows = [
            (row["accessionNumber"], row.get("dateFiled"), row.get("acceptedAt"))
            for manager in entities(root, MANAGER_13F_FILINGS)
            for row in read_pull(root, MANAGER_13F_FILINGS, manager)
            if row.get("accessionNumber")
        ]
        schema = {"accession": pl.String, "dateFiled": pl.String, "acceptedAt": pl.String}
        filings = pl.DataFrame(rows, schema=schema, orient="row").select(
            "accession",
            pl.col("acceptedAt").str.to_datetime(time_zone="UTC", strict=False, time_unit="us").alias("accepted_at"),
            pl.col("dateFiled").str.slice(0, 10).str.to_date(strict=False).alias("date_filed"),
        )
        return release_day(filings).group_by("accession").agg(pl.col("release_day").min().alias("public"))

    def _trading_calendar(self) -> TradingCalendar:
        """Return NYSE's sessions through the day the holder rows are complete through (``nyse_calendar``)."""
        if self._calendar_cache is None:
            self._calendar_cache = nyse_calendar(
                self.config.raw_data_dir_path, INSTITUTIONAL_HOLDERS,
                start=self.config.start_date, end=self.config.end_date, owner=self.class_name,
            )
        return self._calendar_cache

    # -- the release table ---------------------------------------------------------

    def _released(self) -> pl.DataFrame:
        """Return each security's running quarter totals, one row per day they change.

        Returns
        -------
        pl.DataFrame
            ``symbol``, ``quarter_end`` (ns datetime), ``available`` (a
            trading day), and ``holders``, ``shares_held`` and
            ``value_held`` summed over the quarter's rows available by then;
            sorted by ``available``.
        """
        calendar = self._trading_calendar()
        holdings = self._holder_rows().join(self._filings(), on="accession", how="left")
        # A row counts from the later of quarter end + 45 days and its filing's day.
        holdings = holdings.with_columns(
            pl.max_horizontal(
                pl.col("quarter_end") + AVAILABILITY_LAG,
                pl.col("public").fill_null(pl.col("pulled") + pl.duration(days=1)),
            ).alias("due")
        )
        holdings = calendar.place(holdings, "due")
        keys = ["symbol", "quarter_end"]
        first_held = holdings.group_by(*keys, "manager").agg(pl.col("available").min())
        daily = (
            holdings.group_by(*keys, "available")
            .agg(pl.col("shares").sum(), pl.col("value").sum())
            .join(
                first_held.group_by(*keys, "available").agg(pl.len().alias("_new")),
                on=[*keys, "available"], how="left",
            )
            .sort(*keys, "available")
        )
        return (
            daily.with_columns(
                pl.col("_new").fill_null(0).cum_sum().over(keys).cast(pl.Float64).alias("holders"),
                pl.col("shares").cum_sum().over(keys).alias("shares_held"),
                pl.col("value").cum_sum().over(keys).alias("value_held"),
            )
            .select(
                "symbol", pl.col("quarter_end").cast(pl.Datetime("ns")), "available",
                "holders", "shares_held", "value_held",
            )
            .sort("available")
        )

    def _quarter_starts(self) -> pl.DataFrame:
        """Return each quarter's first shown day: ``available``, ``quarter_end``.

        The quarters are the calendar's, from the oldest with a row: a
        quarter nothing was pulled for still replaces the one before it.
        """
        first = self._released_rows().get_column("quarter_end").min()
        ends = pd.date_range(first, self._trading_calendar().through, freq="QE") if first is not None else []
        quarters = pl.DataFrame({"quarter_end": [end.date() for end in ends]}, schema={"quarter_end": pl.Date})
        return (
            self._trading_calendar()
            .place(quarters, "quarter_end", lag=AVAILABILITY_LAG)
            .select("available", pl.col("quarter_end").cast(pl.Datetime("ns")))
            .group_by("available").agg(pl.col("quarter_end").max())
            .sort("available")
        )

    def _spec(self) -> ReleaseSpec:
        """Return the variables; a quarter is replaced by the next, never stale."""
        return ReleaseSpec(variables=VARIABLES, dates=("quarter_end",), stamp=None)

    def _nothing_shown(self) -> str:
        """Return what an empty window lacks."""
        return "no 13F quarter is shown"

    # -- windows: one quarter a day, its rows as filed by then -------------------------

    def _raw_data_to_xr_window(self, start_date, end_date, symbols: list | None = None) -> xr.Dataset:
        """Return one window of the panel: each day shows the latest quarter shown by then, as filed by then.

        Parameters
        ----------
        start_date, end_date : date-like
            The window, both inclusive.
        symbols : list of str, optional
            CUSIPs to build; ``None`` is every one with a row available by
            the window's end.
        """
        days, symbols = self._window_axes(start_date, end_date, symbols)
        current = days.to_frame().join_asof(
            self._quarter_starts(), left_on="timestamp", right_on="available", strategy="backward"
        ).select("timestamp", "quarter_end")
        cells = (
            on_grid(days, symbols, text=self.TEXT_SYMBOLS)
            .join(current, on="timestamp", how="left")
            .join_asof(
                self._released_rows(),
                left_on="timestamp",
                right_on="available",
                by=["symbol", "quarter_end"],
                strategy="backward",
                check_sortedness=False,
            )
            .sort("_day", "_column")
        )
        shown = cells.get_column("holders").is_not_null()
        cells = cells.with_columns(pl.when(shown).then(pl.col("quarter_end")).alias("quarter_end"))
        return to_panel(cells, days, symbols, self._spec().variables, text=self.TEXT_SYMBOLS)
