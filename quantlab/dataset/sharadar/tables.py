"""The Sharadar raw tier: which tables exist, their columns and where they live.

The raw tier is Sharadar's own rows, kept as parquet under
``<download-dir>/sharadar/<code>/``, one directory per table. The
acquisition layer writes it and the datasets read it; both take the table
names, the column schemas and the directory layout from this module, so the
writer and the reader cannot disagree.

Each table has a short *code*, the lower-cased legacy Sharadar code (``sep``
for stock prices), which names its directory, and an *API name*
(``stocks``), which is what ``api.sharadar.com/v1.0`` calls it. TICKERS'
``table`` column uses the upper-case code in the bulk file (``SEP``) and the
API name over REST (``stocks``). Column names and order are Sharadar's published
schema, verbatim (``GET api.sharadar.com/v1.0/schema/<api name>``, as of
2026-08-18), with the PostgreSQL types mapped to polars: ``text`` to
``String``, ``date`` to ``Date``, ``double precision`` to ``Float64`` and
``bigint`` to ``Int64``. A ticker is always ``String``, so ``NA`` or ``1234``
is never read as a null or a number.

Examples
--------
>>> TABLES["sep"].api_name
'stocks'
>>> raw_table_dir("/data/downloads/sharadar", "sep")
PosixPath('/data/downloads/sharadar/sep')
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

#: Directory under ``--download-dir`` holding every Sharadar raw table.
VENDOR_DIR = "sharadar"


@dataclass(frozen=True)
class SharadarTable:
    """One Sharadar table of the raw tier.

    Attributes
    ----------
    code : str
        Lower-cased legacy code; the name of the table's raw directory.
    api_name : str
        The table's name on ``api.sharadar.com/v1.0``.
    schema : dict of str to polars.DataType
        The columns in the vendor's order, with their types.
    """

    code: str
    api_name: str
    schema: dict[str, type[pl.DataType]]

    @property
    def tickers_labels(self) -> tuple[str, str]:
        """Return the values TICKERS' ``table`` column gives this table's rows.

        The bulk TICKERS file uses the upper-case legacy code (``SEP``), the
        REST API the API name (``stocks``); both are accepted.

        Examples
        --------
        >>> TABLES["sep"].tickers_labels
        ('SEP', 'stocks')
        """
        return (self.code.upper(), self.api_name)


#: The tables the raw tier holds so far, by code.
TABLES: dict[str, SharadarTable] = {
    table.code: table
    for table in (
        SharadarTable(
            code="sep",
            api_name="stocks",
            schema={
                "ticker": pl.String,
                "date": pl.Date,
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Float64,
                "closeadj": pl.Float64,
                "closeunadj": pl.Float64,
                "lastupdated": pl.Date,
            },
        ),
        SharadarTable(
            code="tickers",
            api_name="tickers",
            schema={
                "table": pl.String,
                "permaticker": pl.Int64,
                "ticker": pl.String,
                "name": pl.String,
                "exchange": pl.String,
                "isdelisted": pl.String,
                "category": pl.String,
                "cusips": pl.String,
                "siccode": pl.Int64,
                "sicsector": pl.String,
                "sicindustry": pl.String,
                "figi": pl.String,
                "famaindustry": pl.String,
                "sector": pl.String,
                "industry": pl.String,
                "scalemarketcap": pl.String,
                "scalerevenue": pl.String,
                "relatedtickers": pl.String,
                "currency": pl.String,
                "location": pl.String,
                "lastupdated": pl.Date,
                "firstadded": pl.Date,
                "firstpricedate": pl.Date,
                "lastpricedate": pl.Date,
                "firstquarter": pl.String,
                "lastquarter": pl.String,
                "secfilings": pl.String,
                "companysite": pl.String,
            },
        ),
        SharadarTable(
            code="sp500",
            api_name="sp500",
            schema={
                "date": pl.Date,
                "action": pl.String,
                "ticker": pl.String,
                "name": pl.String,
                "contraticker": pl.String,
                "contraname": pl.String,
                "note": pl.String,
            },
        ),
        SharadarTable(
            code="indicators",
            api_name="descriptions",
            schema={
                "table": pl.String,
                "indicator": pl.String,
                "isfilter": pl.String,
                "isprimarykey": pl.String,
                "title": pl.String,
                "description": pl.String,
                "unittype": pl.String,
            },
        ),
    )
}


def table(code: str) -> SharadarTable:
    """Return the table with this code.

    Raises
    ------
    KeyError
        If no table has this code, naming the known ones.

    Examples
    --------
    >>> table("tickers").api_name
    'tickers'
    """
    try:
        return TABLES[code]
    except KeyError:
        raise KeyError(
            f"no Sharadar table {code!r} in the raw tier; known: {sorted(TABLES)}"
        ) from None


def raw_table_dir(vendor_root: str | Path, code: str) -> Path:
    """Return the directory holding one table's raw parquet.

    Parameters
    ----------
    vendor_root : str or Path
        ``<download-dir>/sharadar``.
    code : str
        The table's code.
    """
    return Path(vendor_root) / table(code).code


def scan_raw_table(vendor_root: str | Path, code: str) -> pl.LazyFrame:
    """Scan every parquet file of one raw table.

    Raises
    ------
    FileNotFoundError
        If the table's directory holds no parquet file.

    Examples
    --------
    >>> scan_raw_table("/data/downloads/sharadar", "tickers").collect_schema().names()[:3]
    ['table', 'permaticker', 'ticker']
    """
    directory = raw_table_dir(vendor_root, code)
    files = sorted(directory.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"no raw Sharadar {code!r} table under {directory}; pull it first "
            f"(SharadarClient.bulk_table({code!r}, <download-dir>))"
        )
    return pl.scan_parquet(files)


#: Number of offending keys an error message lists.
_ERROR_SAMPLE = 5


def permaticker_mapping(vendor_root: str | Path, code: str) -> pl.DataFrame:
    """Return the ``(ticker, permaticker)`` pairs TICKERS gives one table's rows.

    Only the TICKERS rows labelled with the table (``SEP`` or ``stocks`` for
    ``sep``) count: the same ticker under another table can be another
    security.

    Parameters
    ----------
    vendor_root : str or Path
        ``<download-dir>/sharadar``.
    code : str
        The code of the table whose tickers are mapped.

    Returns
    -------
    pl.DataFrame
        Distinct ``ticker``/``permaticker`` pairs.

    Examples
    --------
    >>> permaticker_mapping("/data/downloads/sharadar", "sep").columns
    ['ticker', 'permaticker']
    """
    return (
        scan_raw_table(vendor_root, "tickers")
        .filter(pl.col("table").is_in(table(code).tickers_labels))
        .select("ticker", "permaticker")
        .unique()
        .collect()
    )


def map_permatickers(
    frame: pl.DataFrame, mapping: pl.DataFrame, *, owner: str, code: str
) -> pl.DataFrame:
    """Add each row's ``permaticker``, refusing a missing or ambiguous one.

    Parameters
    ----------
    frame : pl.DataFrame
        Rows with a ``ticker`` column.
    mapping : pl.DataFrame
        ``permaticker_mapping`` of the rows' table.
    owner : str
        Named in error messages.
    code : str
        The rows' table, named in error messages.

    Returns
    -------
    pl.DataFrame
        ``frame`` with a ``permaticker`` column.

    Raises
    ------
    ValueError
        If a ticker maps to several permatickers, or to none.

    Examples
    --------
    >>> rows = pl.DataFrame({"ticker": ["AAA"]})
    >>> pairs = pl.DataFrame({"ticker": ["AAA"], "permaticker": [101]})
    >>> map_permatickers(rows, pairs, owner="demo", code="sep")["permaticker"].to_list()
    [101]
    """
    used = mapping.join(frame.select("ticker").unique(), on="ticker")
    ambiguous = (
        used.group_by("ticker")
        .agg(pl.col("permaticker").sort())
        .filter(pl.col("permaticker").list.len() > 1)
        .sort("ticker")
    )
    if ambiguous.height:
        sample = dict(ambiguous.head(_ERROR_SAMPLE).iter_rows())
        raise ValueError(
            f"{owner}: {ambiguous.height} {code!r} ticker(s) map to several "
            f"permatickers in TICKERS, first {sample}. Refusing rather than "
            f"guessing which company a row belongs to; re-pull TICKERS."
        )
    joined = frame.join(used, on="ticker", how="left")
    unmapped = (
        joined.filter(pl.col("permaticker").is_null())
        .get_column("ticker")
        .unique()
        .sort()
    )
    if unmapped.len():
        raise ValueError(
            f"{owner}: {unmapped.len()} {code!r} ticker(s) have no permaticker "
            f"in TICKERS, first {unmapped.head(_ERROR_SAMPLE).to_list()}. "
            f"TICKERS is probably older than the {code!r} table (a ticker "
            f"changed between the two pulls); re-pull TICKERS."
        )
    return joined
