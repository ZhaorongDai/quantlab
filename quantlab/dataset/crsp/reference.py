"""Readers for the CRSP reference tables, six whole tables stored next to the raw daily data.

CRSP (the Center for Research in Security Prices) is a US stock database
sold through WRDS (Wharton Research Data Services). A PERMNO is CRSP's
permanent integer id for one security. Compustat is S&P's fundamentals
database, also on WRDS, which identifies a company by ``gvkey``.

CRSP's daily table says what happened to a security on a given day. It does
not say what the security was called at the time, when it was delisted,
whether it was in the S&P 500, or which Compustat security it matches.
Those facts live in small tables that are downloaded whole, once, and read
many times; together they are the *reference tier* (a tier is one layer of
files on disk, next to the *raw tier* of daily price files). This module
declares which tables make up the reference tier, with their columns and
types (``ReferenceTableSpec``), and provides ``CrspReference``, a read-only
accessor for one reference directory.

The reference tier lives in ``.../wrds_crsp/_reference/``, next to the raw
root ``.../wrds_crsp/wrds/`` rather than inside it. The raw scan reads every
parquet file below the raw root, so it never picks up a reference table or
the manifest.

The module imports only polars and the standard library. Anything that reads
the reference tables (the CRSP dataset, the constituent datasets, the roster
helpers) therefore works on a machine with no WRDS login and no database
driver. Downloading the tables is a separate acquisition step.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import polars as pl

#: The manifest written next to the tables: CRSP product end date (last date
#: of the data version), download time and row counts.
MANIFEST_NAME = "manifest.json"

#: The polars types every server column type maps to. A CRSP ``numeric`` is
#: read as ``Float64`` rather than ``Decimal`` because the panel is float64
#: throughout; decimal precision would be lost at the first multiplication.
_DATE = pl.Date
_INT = pl.Int64
_NUM = pl.Float64
_STR = pl.String


@dataclass(frozen=True)
class ReferenceTableSpec:
    """One reference table: where it lives on the WRDS server and its schema.

    Attributes
    ----------
    name : str
        Local name of the table; also the parquet file stem.
    schema : str
        Server schema (database namespace), e.g. ``"crsp_a_stock"``.
    table : str
        Table name on the server.
    columns : tuple of str
        Column names in server order, so a ``SELECT`` built from them can be
        compared one by one with the server's ``information_schema``.
    dtypes : dict of str to pl.DataType
        The polars type each column is read as.

    Examples
    --------
    >>> from quantlab.dataset.crsp.reference import REFERENCE_TABLES_BY_NAME
    >>> spec = REFERENCE_TABLES_BY_NAME["dsp500list_v2"]
    >>> spec.schema, spec.table
    ('crsp_a_indexes', 'dsp500list_v2')
    >>> spec.columns
    ('permno', 'indno', 'mbrstartdt', 'mbrenddt', 'mbrflg', 'indfam')
    """

    name: str
    schema: str
    table: str
    columns: tuple[str, ...]
    dtypes: dict[str, pl.DataType]

    def cast(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Cast an all-string frame, as a database ``COPY`` produces, to this spec's types.

        Dates are parsed with ``strict=False``, so a malformed or empty field
        becomes null instead of failing the whole table; the live tables
        have real nulls in every date column. Columns missing from ``frame``
        are added as all-null, so a table downloaded before a column existed
        still reads with the current schema.

        Parameters
        ----------
        frame : pl.DataFrame
            A frame whose columns are all ``String``.

        Returns
        -------
        pl.DataFrame
            A frame holding exactly ``columns``, in order, typed by ``dtypes``.

        Examples
        --------
        >>> import polars as pl
        >>> raw = pl.DataFrame(
        ...     {"permno": ["14593"], "indno": ["1000500"],
        ...      "mbrstartdt": ["1982-11-18"], "mbrenddt": [None],
        ...      "mbrflg": ["NORM"]},
        ... )
        >>> spec.cast(raw).schema
        Schema({'permno': Int64, 'indno': Int64, 'mbrstartdt': Date,
                'mbrenddt': Date, 'mbrflg': String, 'indfam': Int64})
        """
        missing = [name for name in self.columns if name not in frame.columns]
        if missing:
            frame = frame.with_columns(
                pl.lit(None, dtype=pl.String).alias(name) for name in missing
            )
        expressions = []
        for name in self.columns:
            dtype = self.dtypes[name]
            column = pl.col(name).cast(pl.String)
            if dtype == pl.Date:
                expressions.append(column.str.to_date(strict=False).alias(name))
            elif dtype == pl.String:
                expressions.append(column.alias(name))
            else:
                # Integers go through Float64 first: the server writes
                # `lpermno` as `90319.0`, which a direct string-to-Int64 cast
                # rejects. Float64 reads both `90319` and `90319.0`.
                expressions.append(
                    column.cast(pl.Float64, strict=False).cast(dtype).alias(name)
                )
        return frame.select(expressions)


#: ``crsp_a_stock.stksecurityinfohist``: each security's descriptive history
#: (names, tickers, share class, exchange, delisting codes). Tickers come
#: from here because the daily table has neither ``shareclass`` nor
#: ``tradingsymbol``.
_SECINFO = ReferenceTableSpec(
    name="stksecurityinfohist",
    schema="crsp_a_stock",
    table="stksecurityinfohist",
    columns=(
        "permno", "secinfostartdt", "secinfoenddt", "securitybegdt",
        "securityenddt", "securityhdrflg", "hdrcusip", "hdrcusip9", "cusip",
        "cusip9", "primaryexch", "conditionaltype", "exchangetier",
        "tradingstatusflg", "securitynm", "shareclass", "usincflg",
        "issuertype", "securitytype", "securitysubtype", "sharetype",
        "securityactiveflg", "delactiontype", "delstatustype", "delreasontype",
        "delpaymenttype", "ticker", "tradingsymbol", "permco", "siccd",
        "naics", "icbindustry", "uesindustry", "nasdcompno", "nasdissuno",
        "issuernm",
    ),
    dtypes={
        "permno": _INT, "secinfostartdt": _DATE, "secinfoenddt": _DATE,
        "securitybegdt": _DATE, "securityenddt": _DATE,
        "securityhdrflg": _STR, "hdrcusip": _STR, "hdrcusip9": _STR,
        "cusip": _STR, "cusip9": _STR, "primaryexch": _STR,
        "conditionaltype": _STR, "exchangetier": _STR,
        "tradingstatusflg": _STR, "securitynm": _STR, "shareclass": _STR,
        "usincflg": _STR, "issuertype": _STR, "securitytype": _STR,
        "securitysubtype": _STR, "sharetype": _STR, "securityactiveflg": _STR,
        "delactiontype": _STR, "delstatustype": _STR, "delreasontype": _STR,
        "delpaymenttype": _STR, "ticker": _STR, "tradingsymbol": _STR,
        "permco": _INT, "siccd": _INT, "naics": _STR, "icbindustry": _STR,
        "uesindustry": _STR, "nasdcompno": _INT, "nasdissuno": _INT,
        "issuernm": _STR,
    },
)

#: ``crsp_a_stock.stkdelists``: delisting events, kept as raw event data. The
#: delisting return is not taken from here: the daily table already includes
#: it on the delisting row (``dlydelflg='Y'``), and applying ``delret`` as
#: well would count the loss twice.
_DELISTS = ReferenceTableSpec(
    name="stkdelists",
    schema="crsp_a_stock",
    table="stkdelists",
    columns=(
        "permno", "delistingdt", "deldtprc", "deldtprcflg", "delactiontype",
        "delstatustype", "delreasontype", "delpaymenttype", "delpermno",
        "delpermco", "delret", "delretmisstype", "delnextdt", "delnextprc",
        "delnextprcflg", "delamtdt", "deldivamt", "deldistype", "deldlydt",
    ),
    dtypes={
        "permno": _INT, "delistingdt": _DATE, "deldtprc": _NUM,
        "deldtprcflg": _STR, "delactiontype": _STR, "delstatustype": _STR,
        "delreasontype": _STR, "delpaymenttype": _STR, "delpermno": _INT,
        "delpermco": _INT, "delret": _NUM, "delretmisstype": _STR,
        "delnextdt": _DATE, "delnextprc": _NUM, "delnextprcflg": _STR,
        "delamtdt": _DATE, "deldivamt": _NUM, "deldistype": _STR,
        "deldlydt": _DATE,
    },
)

#: ``crsp_a_stock.stkdistributions``: dividends, splits and other
#: distributions, one row per event. The panel's ``divCash`` and
#: ``splitFactor`` come from fields of the daily table; this table is kept
#: to audit them.
_DISTRIBUTIONS = ReferenceTableSpec(
    name="stkdistributions",
    schema="crsp_a_stock",
    table="stkdistributions",
    columns=(
        "permno", "disexdt", "disseqnbr", "disordinaryflg", "distype",
        "disfreqtype", "dispaymenttype", "disdetailtype", "distaxtype",
        "disorigcurtype", "disdivamt", "disfacpr", "disfacshr",
        "disdeclaredt", "disrecorddt", "dispaydt", "dispermno", "dispermco",
        "disamountsourcetype",
    ),
    dtypes={
        "permno": _INT, "disexdt": _DATE, "disseqnbr": _INT,
        "disordinaryflg": _STR, "distype": _STR, "disfreqtype": _STR,
        "dispaymenttype": _STR, "disdetailtype": _STR, "distaxtype": _STR,
        "disorigcurtype": _STR, "disdivamt": _NUM, "disfacpr": _NUM,
        "disfacshr": _NUM, "disdeclaredt": _DATE, "disrecorddt": _DATE,
        "dispaydt": _DATE, "dispermno": _INT, "dispermco": _INT,
        "disamountsourcetype": _STR,
    },
)

#: ``crsp_a_indexes.dsp500list_v2``: S&P 500 membership intervals by PERMNO.
#: It is in a different schema from the stock tables, which is why each spec
#: names its ``schema``.
_DSP500 = ReferenceTableSpec(
    name="dsp500list_v2",
    schema="crsp_a_indexes",
    table="dsp500list_v2",
    columns=("permno", "indno", "mbrstartdt", "mbrenddt", "mbrflg", "indfam"),
    dtypes={
        "permno": _INT, "indno": _INT, "mbrstartdt": _DATE,
        "mbrenddt": _DATE, "mbrflg": _STR, "indfam": _INT,
    },
)

#: ``comp.idxcst_his``: Compustat index membership history, the source for
#: the Nasdaq-100. ``from`` and ``thru`` are reserved SQL words, so every
#: query that names them must quote them.
_IDXCST = ReferenceTableSpec(
    name="idxcst_his",
    schema="comp",
    table="idxcst_his",
    columns=("gvkey", "iid", "gvkeyx", "from", "thru"),
    dtypes={
        "gvkey": _STR, "iid": _STR, "gvkeyx": _STR, "from": _DATE,
        "thru": _DATE,
    },
)

#: ``crsp_a_ccm.ccmxpf_lnkhist``: the CRSP/Compustat Merged (CCM) link table,
#: which maps a Compustat ``gvkey`` to a PERMNO over date ranges. ``lpermno``
#: and ``lpermco`` are floats on the server and stay floats here; an integer
#: cast on read would turn the null on a ``linktype='NR'`` row into a false 0.
_CCM = ReferenceTableSpec(
    name="ccmxpf_lnkhist",
    schema="crsp_a_ccm",
    table="ccmxpf_lnkhist",
    columns=(
        "gvkey", "linkprim", "liid", "linktype", "lpermno", "lpermco",
        "linkdt", "linkenddt",
    ),
    dtypes={
        "gvkey": _STR, "linkprim": _STR, "liid": _STR, "linktype": _STR,
        "lpermno": _NUM, "lpermco": _NUM, "linkdt": _DATE,
        "linkenddt": _DATE,
    },
)

#: Every reference table, in download order. The downloader and the reader
#: both use this tuple, so they always agree on the set of tables.
REFERENCE_TABLES: tuple[ReferenceTableSpec, ...] = (
    _SECINFO,
    _DELISTS,
    _DISTRIBUTIONS,
    _DSP500,
    _IDXCST,
    _CCM,
)

#: Every spec keyed by its ``name``.
REFERENCE_TABLES_BY_NAME: dict[str, ReferenceTableSpec] = {
    spec.name: spec for spec in REFERENCE_TABLES
}


class CrspReference:
    """Read-only access to one CRSP reference directory.

    It holds no database connection and needs no login: the tables are
    parquet files on disk. Each table is read on first request and cached on
    the instance, so repeated requests cost one file read.

    Parameters
    ----------
    reference_dir : str or Path
        The directory holding the reference parquet files and
        ``manifest.json``.

    Attributes
    ----------
    reference_dir : Path
        The reference directory.

    Examples
    --------
    >>> from quantlab.dataset.crsp.reference import CrspReference
    >>> ref = CrspReference("data/downloads/us_equity/1d/wrds_crsp/_reference")
    >>> ref.table("dsp500list_v2").columns
    ['permno', 'indno', 'mbrstartdt', 'mbrenddt', 'mbrflg', 'indfam']
    >>> ref.product_end
    datetime.date(2025, 12, 31)
    """

    def __init__(self, reference_dir) -> None:
        """Initialize without reading anything; see the class docstring."""
        self.reference_dir = Path(reference_dir)
        self._cache: dict[str, pl.DataFrame] = {}
        self._manifest: dict | None = None

    def path_for(self, name: str) -> Path:
        """Return the parquet path where the table ``name`` is stored.

        Parameters
        ----------
        name : str
            Table name, e.g. ``"stkdelists"``.

        Returns
        -------
        Path
            ``<reference_dir>/<name>.parquet``.

        Examples
        --------
        >>> ref.path_for("stkdelists").name
        'stkdelists.parquet'
        """
        return self.reference_dir / f"{name}.parquet"

    def table(self, name: str) -> pl.DataFrame:
        """Return one reference table, read from parquet and cached.

        Parameters
        ----------
        name : str
            A key of ``REFERENCE_TABLES_BY_NAME``.

        Returns
        -------
        pl.DataFrame
            The table as stored (already typed by its spec when written).

        Raises
        ------
        KeyError
            If ``name`` is not a CRSP reference table.
        FileNotFoundError
            If the parquet file is missing. The message names the directory
            and the command that fills it, because this is the error a user
            sees after converting before downloading the reference tables.

        Examples
        --------
        >>> ref.table("stksecurityinfohist").select("permno", "ticker").head(2)
        shape: (2, 2)
        ┌────────┬────────┐
        │ permno ┆ ticker │
        │ ---    ┆ ---    │
        │ i64    ┆ str    │
        ╞════════╪════════╡
        │ 14593  ┆ AAPL   │
        │ 14593  ┆ AAPL   │
        └────────┴────────┘
        """
        if name not in REFERENCE_TABLES_BY_NAME:
            raise KeyError(
                f"{type(self).__name__}: {name!r} is not a CRSP reference "
                f"table; this tier holds "
                f"{sorted(REFERENCE_TABLES_BY_NAME)}."
            )
        cached = self._cache.get(name)
        if cached is not None:
            return cached
        path = self.path_for(name)
        if not path.exists():
            raise FileNotFoundError(
                f"{type(self).__name__}: no {name}.parquet under "
                f"{str(self.reference_dir)!r}. The CRSP reference tables are "
                f"downloaded separately from the daily data; run "
                f"`scripts/ingest_wrds_crsp.py` with the reference step before "
                f"converting, or point `reference_dir` at a directory that "
                f"already holds it."
            )
        frame = pl.read_parquet(path)
        self._cache[name] = frame
        return frame

    @property
    def manifest(self) -> dict:
        """Return ``manifest.json`` as a dict, read at most once.

        Raises
        ------
        FileNotFoundError
            If the directory has no manifest. Without one, the CRSP data
            version is unknown, and a panel built from an unknown version
            cannot be checked against a later download.

        Examples
        --------
        >>> ref.manifest["product_end"]
        '2025-12-31'
        """
        if self._manifest is None:
            path = self.reference_dir / MANIFEST_NAME
            if not path.exists():
                raise FileNotFoundError(
                    f"{type(self).__name__}: no {MANIFEST_NAME} under "
                    f"{str(self.reference_dir)!r}; the reference download records "
                    f"its CRSP data version there. Re-run the reference "
                    f"download."
                )
            self._manifest = json.loads(path.read_text(encoding="utf-8"))
        return self._manifest

    @property
    def product_end(self) -> date:
        """Return the CRSP product end date, the last date of the downloaded data version.

        Examples
        --------
        >>> ref.product_end
        datetime.date(2025, 12, 31)
        """
        return date.fromisoformat(str(self.manifest["product_end"])[:10])
