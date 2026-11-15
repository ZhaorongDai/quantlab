"""StockFit's insider transactions (Forms 4 and 5) as a daily panel on the CIK axis.

Each company's insider transactions (``/api/insider-transactions``, every
page) are pulled into the raw tier by the issuer's CIK
(``insider_transactions/<cik>/<stamp>/``). A row is one transaction line of
an insider's filing: the owner, the transaction code, shares, price, whether
shares were acquired (``A``) or disposed (``D``), whether the line is a
derivative (an option, a unit), and the filing's date and SEC acceptance
time.

StockFit cleans the rows on its side before serving them:

- only officers' and directors' transactions are served; a holder of 10%
  who is neither is left out (Sharadar's SF2 keeps them);
- a trade reported by several co-owners appears once, under the most
  senior filer (officer, then director, then 10% owner);
- a line with a price of $100,000 or more per share (an SEC data-entry
  error) is dropped;
- a Form 5 line is dropped when a Form 4 reports the same trade.

So the two vendors' panels differ where these rules bite.

``StockFitInsidersDataset`` has the variables of the Sharadar insider panel
(``quantlab.dataset.sharadar.insiders``), with the same names and signs, on
the CIK axis. Each line is placed on its filing's available date
(``quantlab.dataset.stockfit.available``: accepted before the session's close
that day, otherwise the next trading day), never on its transaction date,
and a day's lines of one company are summed:

- ``net_shares``, ``net_value``: open-market trades in the stock, shares and
  USD bought (``P``) minus sold (``S``) on a non-derivative line;
- ``net_shares_<code>``, ``net_value_<code>``: the non-derivative lines of
  one code, ``P``, ``S``, ``A``, ``M``, ``F``, ``G``, ``D``, and ``other``
  for every remaining code. ``P`` and ``S`` keep their own sign; every other
  code counts positive when acquired and negative when disposed. A line
  without a value adds nothing to the value;
- ``net_shares_derivative``: derivative units acquired minus disposed,
  every code together;
- ``insiders_buying``, ``insiders_selling``: the distinct owners with a
  non-derivative ``P`` or ``S`` line on the day.

A day without a transaction is 0. A line that a later filing repeats
exactly (same owner, transaction date, code, shares and price, on the same
side of the derivative split) is counted on its first filing only.

Examples
--------
Build the store and read the net buying of a quarter::

    config = StockFitInsidersConfig(
        zarr_file_path="/data/market/stockfit/stockfit_insiders_1d/stockfit_insiders_1d.zarr",
        raw_data_dir_path="/data/downloads/stockfit",
        start_date="2004-01-01",
    )
    StockFitInsidersDataset(config).update()
    panel = StockFitInsidersDataset(config).panel("2024-01-02", "2024-03-28")
    panel["net_value"].sel(symbol=1652044)  # Alphabet's CIK
"""

from __future__ import annotations

import polars as pl

from quantlab.dataset._support.point_in_time import FilingPanelDataset, TradingCalendar
from quantlab.dataset.config import DatasetConfig, StockFitInsidersConfig
from quantlab.dataset.stockfit.available import nyse_calendar, place_released
from quantlab.dataset.stockfit.raw import entities, read_pull

#: The raw tier's endpoint of the per-company insider transactions.
INSIDER_TRANSACTIONS = "insider_transactions"

#: Transaction codes of open-market trades, and the sign each counts with.
OPEN_MARKET: dict[str, int] = {"P": 1, "S": -1}

#: The sign a line counts with by ``acquiredDisposed``.
SIDE_SIGN: dict[str, int] = {"A": 1, "D": -1}

#: Transaction codes with variables of their own and what each means (SEC
#: Form 4 instructions); every other code is ``other``.
CODES: dict[str, str] = {
    "P": "open-market or private purchase",
    "S": "open-market or private sale",
    "A": "grant or award from the issuer",
    "M": "exercise or conversion of a derivative",
    "F": "exercise price or tax paid by delivering stock",
    "G": "gift",
    "D": "disposition to the issuer",
}

#: The per-code variable suffixes, in variable order.
BUCKETS: tuple[str, ...] = (*CODES, "other")

#: The columns that identify one trade across the filings that show it.
TRADE_KEY: tuple[str, ...] = ("cik", "owner", "transaction_date", "code", "shares", "price", "derivative")

#: The columns of ``read_insider_transactions``.
TRANSACTION_SCHEMA: dict[str, pl.DataType] = {
    "cik": pl.Int64,
    "owner": pl.String,
    "transaction_date": pl.String,
    "code": pl.String,
    "side": pl.String,
    "derivative": pl.Boolean,
    "shares": pl.Float64,
    "price": pl.Float64,
    "value": pl.Float64,
    "date_filed": pl.Date,
    "accepted_at": pl.Datetime("us", "UTC"),
}


def read_insider_transactions(vendor_root, ciks=None) -> pl.DataFrame:
    """Return the newest pull of each company's insider transactions, one row per line.

    Parameters
    ----------
    vendor_root : str or Path
        ``<root>/downloads/stockfit``.
    ciks : iterable of int, optional
        The issuers to read; every one with a pull when omitted.

    Returns
    -------
    pl.DataFrame
        ``TRANSACTION_SCHEMA``'s columns: the issuer's ``cik``, the
        ``owner`` (the owner's CIK, or name when it has none), the
        ``transaction_date``, the transaction ``code``, the ``side``
        (``A`` or ``D``), ``derivative``, ``shares``, ``price`` and
        ``value`` (USD) as served, ``date_filed`` and ``accepted_at``.

    Examples
    --------
    >>> read_insider_transactions("/nonexistent/stockfit").columns[:3]
    ['cik', 'owner', 'transaction_date']
    """
    present = entities(vendor_root, INSIDER_TRANSACTIONS)
    wanted = present if ciks is None else [str(int(cik)) for cik in ciks if str(int(cik)) in present]

    def number(value) -> float | None:
        return None if value is None else float(value)

    records = [
        {
            "cik": int(entity),
            "owner": str(row["ownerCik"]) if row.get("ownerCik") is not None else row.get("ownerName"),
            "transaction_date": row.get("transactionDate"),
            "code": row.get("transactionCode"),
            "side": row.get("acquiredDisposed"),
            "derivative": bool(row.get("isDerivative")),
            "shares": number(row.get("shares")),
            "price": number(row.get("pricePerShare")),
            "value": number(row.get("transactionValue")),
            "date_filed": row.get("filingDate"),
            "accepted_at": row.get("acceptedAt"),
        }
        for entity in wanted
        for row in read_pull(vendor_root, INSIDER_TRANSACTIONS, entity)
    ]
    text = {"date_filed": pl.String, "accepted_at": pl.String}
    return pl.DataFrame(records, schema={**TRANSACTION_SCHEMA, **text}).with_columns(
        pl.col("date_filed").str.slice(0, 10).str.to_date(strict=False),
        pl.col("accepted_at").str.to_datetime(time_zone="UTC", time_unit="us", strict=False),
    )


class StockFitInsidersDataset(FilingPanelDataset):
    """Daily panel of insiders' transactions per code, by CIK, on each filing's available date.

    See the module docstring for the variables, the vendor's cleaning and
    the placement.

    Parameters
    ----------
    config : StockFitInsidersConfig
        ``raw_data_dir_path`` is ``<root>/downloads/stockfit``. The calendar
        is NYSE's trading days through the day before the newest pull of
        the insider transactions.

    Examples
    --------
    With ``config`` as in the module example::

        ds = StockFitInsidersDataset(config).update()
        list(ds.panel("2024-01-02", "2024-01-05").data_vars)[:4]
        # ['net_shares', 'net_value', 'net_shares_P', 'net_value_P']
    """

    #: The config class used to rebuild this dataset from a saved config.
    config_cls = StockFitInsidersConfig

    def _normalize_config(self, config: DatasetConfig) -> StockFitInsidersConfig:
        """Normalise as the base class does, then check the config class.

        Raises
        ------
        TypeError
            If ``config`` is not a ``StockFitInsidersConfig``.
        """
        config = super()._normalize_config(config)
        if not isinstance(config, StockFitInsidersConfig):
            raise TypeError(f"{self.class_name} needs a StockFitInsidersConfig, got {type(config).__name__}.")
        return config

    def _trading_calendar(self) -> TradingCalendar:
        """Return NYSE's trading days through the insider transactions' last complete day (cached)."""
        if self._calendar_cache is None:
            self._calendar_cache = nyse_calendar(
                self.config.raw_data_dir_path, INSIDER_TRANSACTIONS,
                start=self.config.start_date, end=self.config.end_date, owner=self.class_name,
            )
        return self._calendar_cache

    def _empty(self) -> dict[str, object]:
        """Return 0 for every variable: no transaction."""
        empty: dict[str, object] = {"net_shares": 0.0, "net_value": 0.0}
        for bucket in BUCKETS:
            empty[f"net_shares_{bucket}"] = 0.0
            empty[f"net_value_{bucket}"] = 0.0
        empty["net_shares_derivative"] = 0.0
        empty["insiders_buying"] = 0.0
        empty["insiders_selling"] = 0.0
        return empty

    def _attrs(self) -> dict[str, dict]:
        """Return each variable's unit and meaning."""
        attrs = {
            "net_shares": {"unit": "shares", "description": "open-market shares bought minus sold"},
            "net_value": {"unit": "USD", "description": "open-market USD bought minus sold"},
        }
        others = "codes other than " + ", ".join(CODES)
        for bucket, meaning in {**CODES, "other": "every other code"}.items():
            codes = others if bucket == "other" else f"code {bucket}"
            attrs[f"net_shares_{bucket}"] = {
                "unit": "shares",
                "description": f"non-derivative shares acquired minus disposed, {codes} ({meaning})",
            }
            attrs[f"net_value_{bucket}"] = {
                "unit": "USD",
                "description": f"non-derivative USD acquired minus disposed, {codes} ({meaning})",
            }
        attrs["net_shares_derivative"] = {
            "unit": "shares", "description": "derivative units acquired minus disposed, every transaction code",
        }
        attrs["insiders_buying"] = {
            "unit": "insiders", "description": "distinct insiders with an open-market purchase (P) of the stock",
        }
        attrs["insiders_selling"] = {
            "unit": "insiders", "description": "distinct insiders with an open-market sale (S) of the stock",
        }
        return attrs

    def _nothing_shown(self) -> str:
        """Return what an empty window lacks."""
        return "no StockFit insider transaction is available"

    def _lines(self) -> pl.DataFrame:
        """Return every transaction line, signed and placed, each from the first filing showing it."""
        code = pl.col("code")
        lines = read_insider_transactions(self.config.raw_data_dir_path).filter(
            code.is_not_null() & pl.col("side").is_in(list(SIDE_SIGN))
        )
        placed = place_released(lines, self._trading_calendar())
        first_shown = pl.col("available") == pl.col("available").min().over(TRADE_KEY)
        return placed.filter(first_shown).with_columns(
            pl.when(pl.col("derivative"))
            .then(pl.lit(None, dtype=pl.String))
            .when(code.is_in(list(CODES)))
            .then(code)
            .otherwise(pl.lit("other"))
            .alias("_bucket"),
            # P and S count by their own sign, every other code by its side.
            pl.coalesce(
                code.replace_strict(OPEN_MARKET, default=None, return_dtype=pl.Int64),
                pl.col("side").replace_strict(SIDE_SIGN, default=None, return_dtype=pl.Int64),
            ).alias("_sign"),
        )

    def _released(self) -> pl.DataFrame:
        """Return every company's lines per available day, summed.

        Returns
        -------
        pl.DataFrame
            ``symbol`` (the CIK), ``available`` and one column per variable,
            one row per pair, sorted by both.
        """
        shares = (pl.col("_sign") * pl.col("shares").abs()).fill_null(0.0)
        value = (pl.col("_sign") * pl.col("value").abs()).fill_null(0.0)
        open_market = pl.col("_bucket").is_in(list(OPEN_MARKET))

        def total(condition: pl.Expr, amount: pl.Expr, name: str) -> pl.Expr:
            return pl.when(condition).then(amount).otherwise(0.0).sum().alias(name)

        per_code = []
        for bucket in BUCKETS:
            per_code.append(total(pl.col("_bucket") == bucket, shares, f"net_shares_{bucket}"))
            per_code.append(total(pl.col("_bucket") == bucket, value, f"net_value_{bucket}"))
        return (
            self._lines()
            .group_by(pl.col("cik").alias("symbol"), "available")
            .agg(
                total(open_market, shares, "net_shares"),
                total(open_market, value, "net_value"),
                *per_code,
                total(pl.col("derivative"), shares, "net_shares_derivative"),
                pl.col("owner").filter(pl.col("_bucket") == "P").drop_nulls().n_unique()
                .cast(pl.Float64).alias("insiders_buying"),
                pl.col("owner").filter(pl.col("_bucket") == "S").drop_nulls().n_unique()
                .cast(pl.Float64).alias("insiders_selling"),
            )
            .select("symbol", "available", *self._empty())
            .sort("available", "symbol")
        )
