"""The CRSP reference tier: six whole tables, on disk beside the raw tier.

CRSP's daily table answers "what happened to this security today". It cannot
answer "what was this security CALLED then", "when was it delisted", "was it in
the S&P 500" or "which Compustat issue is it" -- those live in small, whole
tables that are pulled once and read many times. This module states WHICH
tables, WITH WHAT columns and types, and how to read them back.

**Why they are not an `Acquisition`** (RESEARCH Pattern 3). An `Acquisition`
downloads a per-symbol time series, batched, watermarked and resumable. These
are whole tables of a few thousand rows with no symbol axis, so the whole
machinery would be ceremony over a single `COPY`. Plan 04 writes them; this
module only declares them and reads them.

**Where they live.** `{downloads}/us_equity/1d/wrds_crsp/_reference/`, a
SIBLING of the raw root `.../wrds_crsp/wrds/`, never inside it. The raw tier is
read by `StockDataset._scan_raw`, which globs `**/*.pqt` below the raw root; a
`.parquet` -- or the `manifest.json` -- sitting in that tree would be walked by
the same scan. The sibling placement is the same reasoning that already puts
the watermark sidecars outside the raw root.

**A LEAF module by intent.** It imports polars and stdlib only -- no psycopg2,
no acquisition module. A reader (`CrspStockDataset`, a constituent dataset, a
universe helper) must be able to consume the reference tier on a machine with
no WRDS credential and no database driver at all.

Every column list and every PostgreSQL type below is VERBATIM from
`03.10-LIVE-CHECK.json` key `C3_columns` (and `03.10-LIVE-CHECK-2.json` keys
`L7_2` / `L8_1` for the two Compustat-side tables), not guessed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import polars as pl

#: The manifest written beside the tables: product end, pull time, row counts.
MANIFEST_NAME = "manifest.json"

#: The three type families the live `data_type` column collapses to. A CRSP
#: `numeric` is read as `Float64` rather than `Decimal`: the panel is float64
#: end to end (D-07), and carrying decimals only as far as the first
#: multiplication would buy exactness nothing downstream preserves.
_DATE = pl.Date
_INT = pl.Int64
_NUM = pl.Float64
_STR = pl.String


@dataclass(frozen=True)
class ReferenceTableSpec:
    """ONE reference table: where it lives on the server, and its schema.

    `columns` is the SERVER order, so a `SELECT` built from it and the
    `information_schema` answer can be compared element by element. `dtypes`
    maps every one of those columns; a spec whose two fields disagree is a
    definition error and `cast()` says so rather than silently dropping a
    column.
    """

    name: str
    schema: str
    table: str
    columns: tuple[str, ...]
    dtypes: dict[str, pl.DataType]

    def cast(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Cast an all-String frame (a COPY's output) onto this spec's types.

        Dates parse with `strict=False`, so a malformed or empty field becomes
        null rather than failing the whole table -- the live tables carry
        genuine NULLs in every date column (`linkenddt`, `thru`,
        `secinfoenddt` on an open interval).

        Columns missing from `frame` are added as all-null: a spec is the
        contract the READER holds, and a table pulled before a column was
        added must still read with the current schema rather than raise on
        the first `.select()`.
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
                # Int64 via Float64: the live rows render `lpermno` as
                # `90319.0`, and a direct String -> Int64 cast of that text
                # fails. Going through Float64 first is the one path that
                # reads both `90319` and `90319.0`.
                expressions.append(
                    column.cast(pl.Float64, strict=False).cast(dtype).alias(name)
                )
        return frame.select(expressions)


#: `crsp_a_stock.stksecurityinfohist` -- the security's own history: names,
#: tickers, share class, exchange, delisting codes. THE symbology source
#: (D-04), because `dsf_v2` carries no `shareclass` and no `tradingsymbol`.
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

#: `crsp_a_stock.stkdelists` -- delisting EVENTS, kept as raw event data only.
#: The delisting RETURN is deliberately not read from here: CIZ already puts it
#: on its own daily row (`dlydelflg='Y'`), so chaining `delret` on top would
#: apply the loss twice (D-10).
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

#: `crsp_a_stock.stkdistributions` -- dividends, splits and other
#: distributions, one row per event. The panel's `divCash`/`splitFactor` come
#: from the daily table's own fields; this table is the audit trail behind
#: them.
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

#: `crsp_a_indexes.dsp500list_v2` -- S&P 500 membership spells, by PERMNO
#: (D-05). A different SCHEMA from the stock tables, which is why
#: `ReferenceTableSpec` carries `schema` rather than assuming one.
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

#: `comp.idxcst_his` -- Compustat index membership, the Nasdaq-100 source
#: (D-14). `from` and `thru` are RESERVED SQL words; every statement that
#: names them must quote them, which is why the builders in
#: `quantlab/acquisition/wrds/crsp.py` emit `sql.Identifier` and never text.
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

#: `crsp_a_ccm.ccmxpf_lnkhist` -- the CRSP/Compustat link, which is how a
#: Compustat `gvkey` becomes a PERMNO. `lpermno`/`lpermco` are `double
#: precision` on the server (`L7_2`), NOT integers, and are typed that way
#: here: casting them to Int64 at read time would turn the NULL on a
#: `linktype='NR'` row into a spurious 0.
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

#: Every reference table the phase pulls, in pull order. A TUPLE, walked by
#: both the writer (plan 04) and the reader below, so neither can know about a
#: table the other does not.
REFERENCE_TABLES: tuple[ReferenceTableSpec, ...] = (
    _SECINFO,
    _DELISTS,
    _DISTRIBUTIONS,
    _DSP500,
    _IDXCST,
    _CCM,
)

#: `name -> spec`, for the reader's lookup.
REFERENCE_TABLES_BY_NAME: dict[str, ReferenceTableSpec] = {
    spec.name: spec for spec in REFERENCE_TABLES
}


class CrspReference:
    """Read-only access to one reference directory.

    Holds no connection and needs no credential: the tier is parquet on disk.
    Tables are read lazily and cached per instance, because a conversion asks
    for `stksecurityinfohist` once and a universe build asks for two more.
    """

    def __init__(self, reference_dir) -> None:
        self.reference_dir = Path(reference_dir)
        self._cache: dict[str, pl.DataFrame] = {}
        self._manifest: dict | None = None

    def path_for(self, name: str) -> Path:
        return self.reference_dir / f"{name}.parquet"

    def table(self, name: str) -> pl.DataFrame:
        """One reference table, typed by its spec.

        A missing file raises `FileNotFoundError` naming the directory AND the
        command that fills it -- the failure a user meets when they converted
        before pulling the reference tier, which is otherwise a bare "no such
        file" about a path they have never heard of.
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
                f"{str(self.reference_dir)!r}. The CRSP reference tier is "
                f"pulled separately from the daily tier -- run "
                f"`scripts/ingest_wrds_crsp.py` with the reference step before "
                f"converting, or point `reference_dir` at a directory that "
                f"already holds it."
            )
        frame = pl.read_parquet(path)
        self._cache[name] = frame
        return frame

    @property
    def manifest(self) -> dict:
        """`manifest.json` as a dict.

        Raises the same shaped `FileNotFoundError` as `table()`: a reference
        directory with tables and no manifest has no recorded vintage, and a
        panel built against an unknown vintage cannot be checked for the
        anchor drift D-08 warns about.
        """
        if self._manifest is None:
            path = self.reference_dir / MANIFEST_NAME
            if not path.exists():
                raise FileNotFoundError(
                    f"{type(self).__name__}: no {MANIFEST_NAME} under "
                    f"{str(self.reference_dir)!r}; the reference tier records "
                    f"its CRSP vintage there. Re-run the reference pull."
                )
            self._manifest = json.loads(path.read_text(encoding="utf-8"))
        return self._manifest

    @property
    def product_end(self) -> date:
        """The CRSP product end this reference tier was pulled against."""
        return date.fromisoformat(str(self.manifest["product_end"])[:10])
