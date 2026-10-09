"""Sharadar insider transactions (SF2) as a daily panel of net open-market buying, keyed by permaticker.

SF2 holds one row per security line of an insider's Form 3, 4 or 5. Its
``date`` is the SEC filing date, the day the transaction became public;
``transactiondate`` is earlier (two days at the median). The panel counts
only *open-market* trades in the stock itself: transaction code ``P``
(purchase) and ``S`` (sale) on a non-derivative line (``securityadcode``
``NA`` or ``ND``). Grants, option exercises, tax withholding, gifts and
derivative lines are left out: they are compensation or bookkeeping, not
an insider's decision to buy or sell.

``SharadarInsidersDataset`` places each trade at the first SEP trading day
on or after its filing date (never its transaction date) and sums a day's
trades per security (``quantlab.dataset.sharadar.filings.FilingPanelDataset``):

- ``net_shares``: shares bought minus shares sold;
- ``net_value``: USD bought minus USD sold. The vendor's
  ``transactionvalue`` is unsigned USD (INDICATORS calls it USD millions,
  but it equals shares times price), so a sale is counted negative.

A day without a trade is 0. A trade is counted on the first filing that
shows it. When an amendment (4/A) restates a filing, it repeats the trades;
a later line repeating one exactly (same insider, transaction date, code,
shares and price, ``TRADE_KEY``) is not counted again. The vendor also
relabels the superseded original ``RESTATED - <form>``, but only once the
amendment arrives, so the label is ignored: a live update, which saw the
original as a plain form on its filing date, and a rebuild of the same days
give the same panel. An amendment that changes a trade's shares or price
does not repeat it, and is counted again on its own filing date.

TICKERS has rows of SF2 (``table`` ``SF2``), so its tickers are mapped
through them. SF2 has no ``lastupdated``; it is refreshed by trailing date
windows (``SharadarClient.window_table``), and ``update()`` appends the new
trading days.

Examples
--------
Build the store and read the net buying of a quarter::

    config = SharadarInsidersConfig(
        zarr_file_path="/data/quantlab/market/sharadar/sharadar_insiders_1d/sharadar_insiders_1d.zarr",
        raw_data_dir_path="/data/quantlab/downloads/sharadar",
    )
    SharadarInsidersDataset(config).update()
    panel = SharadarInsidersDataset(config).panel("2024-01-02", "2024-03-28")
    panel["net_value"].attrs["unit"]  # 'USD'
"""

from __future__ import annotations

import polars as pl

from quantlab.dataset.config import SharadarInsidersConfig
from quantlab.dataset.sharadar.filings import FilingPanelDataset

#: Transaction codes of open-market trades, and the sign each counts with.
OPEN_MARKET: dict[str, int] = {"P": 1, "S": -1}

#: ``securityadcode`` values of a non-derivative line: acquired, disposed.
NON_DERIVATIVE: tuple[str, ...] = ("NA", "ND")

#: The columns that identify one trade across the filings that show it.
TRADE_KEY: tuple[str, ...] = (
    "permaticker", "ownername", "transactiondate", "transactioncode",
    "transactionshares", "transactionpricepershare",
)


class SharadarInsidersDataset(FilingPanelDataset):
    """Daily panel of insiders' net open-market buying, on each filing's available date.

    Parameters
    ----------
    dataset_config : SharadarInsidersConfig
        ``raw_data_dir_path`` is ``<download-dir>/sharadar``, holding the
        ``sf2``, ``sep`` (the calendar) and ``tickers`` raw tables.

    Examples
    --------
    With ``config`` as in the module example::

        ds = SharadarInsidersDataset(config).update()
        sorted(ds.panel("2024-01-02", "2024-01-05").data_vars)
        # ['net_shares', 'net_value']
    """

    #: The config class used to rebuild this dataset from a saved config.
    config_cls = SharadarInsidersConfig

    TABLE = "sf2"

    def _empty(self) -> dict[str, object]:
        """Return 0 for both variables: no trade."""
        return {"net_shares": 0.0, "net_value": 0.0}

    def _attrs(self) -> dict[str, dict]:
        """Return each variable's unit and meaning."""
        return {
            "net_shares": {"unit": "shares", "description": "open-market shares bought minus sold"},
            "net_value": {"unit": "USD", "description": "open-market USD bought minus sold"},
        }

    def _filings(self, raw: pl.DataFrame) -> pl.DataFrame:
        """Return the open-market trades of the stock, signed, each from the first filing showing it.

        Several identical lines of one filing are several trades and all
        count; the same line in a later filing is a repeat and does not.
        """
        sign = pl.col("transactioncode").replace_strict(OPEN_MARKET, default=None)
        trades = raw.filter(
            pl.col("transactioncode").is_in(list(OPEN_MARKET))
            & pl.col("securityadcode").is_in(NON_DERIVATIVE)
        )
        first_shown = pl.col("date") == pl.col("date").min().over(TRADE_KEY)
        return trades.filter(first_shown).select(
            "permaticker",
            "date",
            (sign * pl.col("transactionshares").abs()).cast(pl.Float64).alias("net_shares"),
            (sign * pl.col("transactionvalue").abs()).cast(pl.Float64).alias("net_value"),
        )

    def _combine(self) -> list[pl.Expr]:
        """Sum a day's trades; a trade without a value adds nothing to ``net_value``."""
        return [pl.col("net_shares").sum(), pl.col("net_value").sum()]
