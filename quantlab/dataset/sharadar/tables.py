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

import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl

from quantlab.utils.atomic import write_json_atomically

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
    categories : tuple of str or None
        For a table converted to a panel, the TICKERS ``category`` values an
        unrostered universe keeps by default; ``None`` keeps every category.
    primary_key : tuple of str or None
        The vendor's primary key, for a table refreshed by ``lastupdated``
        (see ``updated_file``): an updated row replaces the earlier row with
        the same key. ``None`` for a table that is never refreshed that way.
    """

    code: str
    api_name: str
    schema: dict[str, type[pl.DataType]]
    categories: tuple[str, ...] | None = None
    primary_key: tuple[str, ...] | None = None

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


#: The columns of both price tables, SEP (stocks) and SFP (funds).
_PRICE_SCHEMA: dict[str, type[pl.DataType]] = {
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
}

#: SEP's default universe: domestic common stock, every share class of it;
#: ADRs, Canadian filers and preferreds are dropped. SF1 shares it.
_DOMESTIC_COMMON: tuple[str, ...] = (
    "Domestic Common Stock",
    "Domestic Common Stock Primary Class",
    "Domestic Common Stock Secondary Class",
)

#: The identity and date columns of SF1 (``fundamentals``), in the vendor's order.
SF1_KEY_COLUMNS: dict[str, type[pl.DataType]] = {
    "ticker": pl.String,
    "dimension": pl.String,
    "calendardate": pl.Date,
    "date": pl.Date,
    "reportperiod": pl.Date,
    "fiscalperiod": pl.String,
    "lastupdated": pl.Date,
}

#: SF1's indicator columns in the vendor's order: ``bigint`` ones are money,
#: share counts and ``evebit``; ``double precision`` ones are ratios and
#: per-share values.
_SF1_BIGINT = frozenset(
    "accoci assets assetsavg assetsc assetsnc capex cashneq cashnequsd cor "
    "consolinc debt debtc debtnc debtusd deferredrev depamor deposits ebit "
    "ebitda ebitdausd ebitusd ebt equity equityavg equityusd ev evebit fcf gp "
    "intangibles intexp invcap invcapavg inventory investments investmentsc "
    "investmentsnc liabilities liabilitiesc liabilitiesnc marketcap ncf ncfbus "
    "ncfcommon ncfdebt ncfdiv ncff ncfi ncfinv ncfo ncfx netinc netinccmn "
    "netinccmnusd netincdis netincnci opex opinc payables ppnenet prefdivis "
    "receivables retearn revenue revenueusd rnd sbcomp sgna sharesbas shareswa "
    "shareswadil tangibles taxassets taxexp taxliabilities workingcapital".split()
)
SF1_INDICATORS: tuple[str, ...] = tuple(
    "accoci assets assetsavg assetsc assetsnc assetturnover bvps capex cashneq "
    "cashnequsd cor consolinc currentratio de debt debtc debtnc debtusd "
    "deferredrev depamor deposits divyield dps ebit ebitda ebitdamargin "
    "ebitdausd ebitusd ebt eps epsdil epsusd equity equityavg equityusd ev "
    "evebit evebitda fcf fcfps fxusd gp grossmargin intangibles intexp invcap "
    "invcapavg inventory investments investmentsc investmentsnc liabilities "
    "liabilitiesc liabilitiesnc marketcap ncf ncfbus ncfcommon ncfdebt ncfdiv "
    "ncff ncfi ncfinv ncfo ncfx netinc netinccmn netinccmnusd netincdis "
    "netincnci netmargin opex opinc payables payoutratio pb pe pe1 ppnenet "
    "prefdivis price ps ps1 receivables retearn revenue revenueusd rnd roa roe "
    "roic ros sbcomp sgna sharefactor sharesbas shareswa shareswadil sps "
    "tangibles taxassets taxexp taxliabilities tbvps workingcapital".split()
)

#: The tables the raw tier holds so far, by code.
TABLES: dict[str, SharadarTable] = {
    table.code: table
    for table in (
        SharadarTable(
            code="sep", api_name="stocks", schema=_PRICE_SCHEMA, categories=_DOMESTIC_COMMON
        ),
        # SFP holds funds (ETF, CEF, ETN, ETD, ...): no category is dropped.
        SharadarTable(code="sfp", api_name="funds", schema=_PRICE_SCHEMA),
        # Fundamentals: every dimension is kept raw; only the as-reported
        # ones (ARQ, ART) ever reach a store.
        SharadarTable(
            code="sf1",
            api_name="fundamentals",
            schema={
                **SF1_KEY_COLUMNS,
                **{
                    name: pl.Int64 if name in _SF1_BIGINT else pl.Float64
                    for name in SF1_INDICATORS
                },
            },
            categories=_DOMESTIC_COMMON,
            primary_key=("ticker", "dimension", "date", "reportperiod"),
        ),
        SharadarTable(
            code="actions",
            api_name="actions",
            schema={
                "date": pl.Date,
                "action": pl.String,
                "ticker": pl.String,
                "name": pl.String,
                "value": pl.Float64,
                "contraticker": pl.String,
                "contraname": pl.String,
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


def bulk_file(vendor_root: str | Path, code: str) -> Path:
    """Return the path of a table's bulk pull, ``<code>/<code>.parquet``."""
    return raw_table_dir(vendor_root, code) / f"{table(code).code}.parquet"


def window_file(
    vendor_root: str | Path, code: str, pulled_at: datetime, start: date, end: date
) -> Path:
    """Return the path a date-window pull of a table is written to.

    A window pull is a complete copy of the table over ``start``..``end``
    (both inclusive), so its rows replace every earlier row of those dates.
    The name records when it was pulled, so names sort in pull order, and
    the dates it covers.

    Examples
    --------
    >>> window_file("/d/sharadar", "sep", datetime(2024, 1, 11, 8, 30),
    ...             date(2024, 1, 2), date(2024, 1, 11)).name
    'window_20240111T083000000000_2024-01-02_2024-01-11.parquet'
    """
    return raw_table_dir(vendor_root, code) / (
        f"{WINDOW_PREFIX}{pulled_at:%Y%m%dT%H%M%S%f}_{start}_{end}.parquet"
    )


#: Filename prefix of a date-window pull (see ``window_file``).
WINDOW_PREFIX = "window_"

#: Filename prefix of a ``lastupdated`` pull (see ``updated_file``).
UPDATED_PREFIX = "updated_"


def updated_file(
    vendor_root: str | Path, code: str, pulled_at: datetime, since: date
) -> Path:
    """Return the path a ``lastupdated`` pull of a table is written to.

    An updated pull holds every row the vendor changed on or after
    ``since``; each replaces the earlier row with the same primary key
    (``SharadarTable.primary_key``). The name records when it was pulled, so
    names sort in pull order.

    Examples
    --------
    >>> updated_file("/d/sharadar", "sf1", datetime(2024, 1, 11, 8, 30),
    ...              date(2024, 1, 10)).name
    'updated_20240111T083000000000_2024-01-10.parquet'
    """
    return raw_table_dir(vendor_root, code) / (
        f"{UPDATED_PREFIX}{pulled_at:%Y%m%dT%H%M%S%f}_{since}.parquet"
    )


def _window_dates(path: Path) -> tuple[date, date]:
    """Return the ``(start, end)`` dates a window file's name covers."""
    _, _, start, end = path.stem.split("_")
    return date.fromisoformat(start), date.fromisoformat(end)


def scan_raw_table(vendor_root: str | Path, code: str) -> pl.LazyFrame:
    """Scan one raw table: its bulk pull overlaid with its later pulls.

    A window pull is a complete copy of the table over its dates, so a row
    is kept only from the newest file covering its date: the bulk file's
    rows of a window's dates, and an older window's rows of a newer
    window's dates, are left out. An updated pull (``updated_file``) holds
    changed rows, so each of its rows replaces the earlier row with the same
    primary key, and the newest pull wins.

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
    windows = sorted(directory.glob(f"{WINDOW_PREFIX}*.parquet"))
    updates = sorted(directory.glob(f"{UPDATED_PREFIX}*.parquet"))
    files = [p for p in [bulk_file(vendor_root, code)] if p.exists()] + windows
    if not files:
        raise FileNotFoundError(
            f"no raw Sharadar {code!r} table under {directory}; pull it first "
            f"(SharadarClient.bulk_table({code!r}, <download-dir>))"
        )
    base = _overlay_windows(files, windows)
    if not updates:
        return base
    key = table(code).primary_key
    if key is None:
        raise ValueError(
            f"Sharadar table {code!r} has updated pulls under {directory} but "
            f"no primary key to apply them by."
        )
    pulls = [base, *(pl.scan_parquet(path) for path in updates)]
    return (
        pl.concat([frame.with_columns(pl.lit(i).alias("_pull")) for i, frame in enumerate(pulls)])
        .sort("_pull", maintain_order=True)
        .unique(subset=list(key), keep="last", maintain_order=True)
        .drop("_pull")
    )


def _overlay_windows(files: list[Path], windows: list[Path]) -> pl.LazyFrame:
    """Scan the bulk file and the window files, each date from the newest file covering it."""
    if not windows:
        return pl.scan_parquet(files)
    covered = [_window_dates(path) for path in windows]
    # The bulk file sits before every window (position -1 among them).
    first_window = len(files) - len(windows)
    frames = []
    for index, path in enumerate(files):
        frame = pl.scan_parquet(path)
        for start, end in covered[max(index - first_window + 1, 0) :]:
            frame = frame.filter(~pl.col("date").is_between(pl.lit(start), pl.lit(end)))
        frames.append(frame)
    return pl.concat(frames)


#: Sharadar's time zone: its tables are updated on US/Eastern evenings, so a
#: pull's "today" is the Eastern date.
VENDOR_TZ = ZoneInfo("America/New_York")


def vendor_today() -> date:
    """Return today's date in Sharadar's time zone (``VENDOR_TZ``)."""
    return datetime.now(VENDOR_TZ).date()


#: Name of a table's watermark file, beside its parquet.
WATERMARK_FILE = "_watermark.json"


def read_watermark(vendor_root: str | Path, code: str) -> date | None:
    """Return the day a table's raw tier is complete through, or ``None``.

    The watermark is written after a bulk pull (the day of the pull) and
    after each date-window pull (the window's last day), once its rows are
    on disk; an interrupted pull leaves the previous one.

    Examples
    --------
    >>> write_watermark(root, "sep", date(2024, 1, 11))
    >>> read_watermark(root, "sep")
    datetime.date(2024, 1, 11)
    """
    path = raw_table_dir(vendor_root, code) / WATERMARK_FILE
    if not path.exists():
        return None
    return date.fromisoformat(json.loads(path.read_text())["through"])


def write_watermark(vendor_root: str | Path, code: str, through: date) -> None:
    """Record that a table's raw tier is complete through ``through``."""
    write_json_atomically(
        raw_table_dir(vendor_root, code) / WATERMARK_FILE,
        {"table": table(code).code, "through": through.isoformat()},
        indent=2,
        sort_keys=True,
    )


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
