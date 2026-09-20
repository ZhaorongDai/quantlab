"""WRDS CRSP Stock v2 daily acquisition (phase 03.10).

Live-verified facts this module is built on (`03.10-LIVE-CHECK{,-2}.json`,
CONTEXT D-19):

- **Table (D-02).** `crsp_a_stock.dsf_v2`, a BASE TABLE of 50 columns covering
  1925-12-31..2025-12-31, UNIQUE on `(permno, dlycaldt)`. The sibling
  `wrds_dsfv2_query` view was rejected: its 98 columns fold one distribution
  and one shares interval into each daily row, so a day with two distributions
  repeats its `(permno, dlycaldt)`. The table name is the class constant
  `CrspQueries.DAILY_TABLE`, so a fallback is a ONE-LINE switch.
- **Order and shape (D-03).** The data pull is a bare
  `COPY (SELECT ... WHERE permno = ANY(...) AND dlycaldt BETWEEN ...)` -- no
  ORDER BY, no GROUP BY, no DISTINCT. Nothing is aggregated, ordered or
  de-duplicated on the server. A page that repeats a `(permno, dlycaldt)` FAILS
  the batch; it is never silently deduped.
- **Prices (D-08/D-09).** CIZ prices are positive and `dlyprcflg='BA'` marks a
  bid/ask midpoint, so the sign carries no information. A delisting return is
  its own daily row (`dlydelflg='Y'`), which is why the raw tier keeps every
  row exactly as CRSP serves it and all derivation happens in
  `quantlab/dataset/crsp.py`.
- **Symbology (D-04).** `dsf_v2` has no `shareclass` and no `tradingsymbol`, so
  it cannot spell `BRK.B`. The raw tier is therefore keyed by PERMNO -- the
  stable security id -- and the ticker is derived at CONVERSION time. A rename
  never touches a watermark or a resume point.
- **Annual vintage (D-01).** `crsp_a_stock` is the ANNUAL-update product; its
  last day is a hard edge. A window past it is refused, naming both dates,
  unless `kwargs["clip_to_product_end"]` is set.

**The session is reached through the MODULE ATTRIBUTE** (RESEARCH Pattern 1).
This module does `from quantlab.acquisition import wrds_taq as _wrds` and calls
`_wrds.WrdsSession.shared()` at call time; it imports NO name from `wrds_taq`.
`tests/conftest.py` patches `"quantlab.acquisition.wrds_taq.WrdsSession"`, and
a by-name binding would capture the real class at import time and escape the
patch -- in the dangerous direction, because a real run would still work while
only the test suite failed. `tests/test_crsp_tracer.py` asserts the rule with
an `ast` scan of this file.

**This module REGISTERS NOTHING**, for the same reason `wrds_taq.py` does not:
the one `wrds` descriptor lives in the neutral `quantlab/acquisition/wrds.py`,
which imports both providers (03.10 D-12, plan 01).
"""

from __future__ import annotations

import dataclasses
import io
import json
from datetime import date, datetime
from pathlib import Path

import polars as pl
import psycopg2
from loguru import logger
from psycopg2 import sql

from quantlab.acquisition import wrds_taq as _wrds
from quantlab.base.acquisition import Acquisition
from quantlab.base.config import AcquisitionConfig
from quantlab.config import get_data_root
from quantlab.utils.atomic import write_json_atomically


class CrspProductEndError(ValueError):
    """The requested window lies past the CRSP product's last day.

    A `ValueError` rather than a session error: nothing is wrong with the
    connection or the subscription -- the data simply does not exist yet,
    because `crsp_a_stock` is the ANNUAL-update product (D-01). It is raised
    BEFORE any COPY, so the message can promise that nothing was downloaded.
    """


class CrspVintageError(ValueError):
    """The raw tier already holds a DIFFERENT CRSP annual vintage.

    A `ValueError` for the same reason `CrspProductEndError` is one: nothing
    is wrong with the connection or the subscription. The account simply now
    holds a later annual release than the one this raw tier was built from,
    and CRSP REVISES history between releases -- a restated delisting return,
    a corrected price, a re-used PERMNO. Two vintages sharing one raw root
    would therefore produce a panel that is neither, with nothing on disk
    recording the seam. Raised BEFORE any COPY, so the message can promise
    that nothing was downloaded.
    """


class CrspQueries:
    """The CRSP statements, built with `psycopg2.sql` and nothing else.

    Two kinds of member, kept apart on purpose. The BUILDERS are pure
    classmethods that return a composable and touch no connection, so a test
    can assert on the exact SQL without a server. The NETWORK calls take a
    session and do nothing but hand a built statement to plan 01's generic
    `schema_usable` / `fetch_rows` / `copy_csv`.

    Every VALUE goes in as `sql.Literal` and every IDENTIFIER as
    `sql.Identifier`. No statement text is ever assembled by string
    formatting (T-03.10-02), which also means `comp.idxcst_his`'s reserved
    column names `from` and `thru` are quoted for free.
    """

    STOCK_SCHEMA = "crsp_a_stock"
    INDEX_SCHEMA = "crsp_a_indexes"
    COMPUSTAT_SCHEMA = "comp"
    CCM_SCHEMA = "crsp_a_ccm"

    #: The daily table. THE one-line switch point of D-19: if `dsf_v2` ever
    #: fails its uniqueness check, `stkdlysecuritydata` is the fallback and
    #: this constant is the only edit.
    DAILY_TABLE = "dsf_v2"

    # -- pure builders ------------------------------------------------------

    @classmethod
    def daily_where(cls, permnos, start, end) -> sql.Composed:
        """`"permno" = ANY(ARRAY[...]) AND "dlycaldt" BETWEEN <start> AND <end>`.

        An EMPTY PERMNO list is refused, the same refusal
        `WrdsSession.where_clause` makes one product over: without the PERMNO
        predicate this is a query over a 110-million-row table, which is the
        one statement this class must never be able to build.

        PERMNOs are coerced to `int` here -- after the caller's digit check --
        so a value that is not a PERMNO cannot reach the statement even as a
        literal.
        """
        values = [int(permno) for permno in permnos]
        if not values:
            raise ValueError(
                "CrspQueries.daily_where: no PERMNOs; refusing to build a "
                "query over the whole daily table."
            )
        return sql.SQL(
            "{permno} = ANY({permnos}) AND {caldt} BETWEEN {start} AND {end}"
        ).format(
            permno=sql.Identifier("permno"),
            permnos=sql.Literal(values),
            caldt=sql.Identifier("dlycaldt"),
            start=sql.Literal(cls._as_date(start)),
            end=sql.Literal(cls._as_date(end)),
        )

    @classmethod
    def copy_query(cls, schema, table, columns, where=None) -> sql.Composed:
        """`COPY (SELECT <columns> FROM <schema>.<table>[ WHERE <where>]) TO
        STDOUT WITH (FORMAT csv, HEADER true)`.

        No ordering, grouping or de-duplication clause -- D-03, asserted on
        the rendered text by `tests/test_crsp_tracer.py`.
        """
        projection = sql.SQL(", ").join(
            sql.Identifier(name) for name in columns
        )
        body = sql.SQL("SELECT {columns} FROM {table}").format(
            columns=projection, table=sql.Identifier(schema, table)
        )
        if where is not None:
            body = sql.SQL("{body} WHERE {where}").format(body=body, where=where)
        return sql.SQL(
            "COPY ({body}) TO STDOUT WITH (FORMAT csv, HEADER true)"
        ).format(body=body)

    @classmethod
    def count_query(cls, schema, table, where=None) -> sql.Composed:
        """`SELECT count(*) FROM <schema>.<table>[ WHERE <where>]`.

        Takes the SAME `where` object the COPY does, so a count and the pull
        it checks cannot select different rows.
        """
        query = sql.SQL("SELECT count(*) FROM {table}").format(
            table=sql.Identifier(schema, table)
        )
        if where is not None:
            query = sql.SQL("{query} WHERE {where}").format(
                query=query, where=where
            )
        return query

    @classmethod
    def columns_query(cls, schema, table) -> sql.Composed:
        """The `information_schema.columns` read for one table.

        Schema and table travel as LITERALS (they are values in this
        statement, not identifiers), and the ordering is done locally on
        `ordinal_position` rather than with an ORDER BY -- D-03 is about the
        DATA pull, but keeping the whole module free of ordering clauses
        means the assertion that forbids them needs no exception list.
        """
        return sql.SQL(
            "SELECT {name}, {position} FROM {catalog} "
            "WHERE {schema_col} = {schema} AND {table_col} = {table}"
        ).format(
            name=sql.Identifier("column_name"),
            position=sql.Identifier("ordinal_position"),
            catalog=sql.Identifier("information_schema", "columns"),
            schema_col=sql.Identifier("table_schema"),
            schema=sql.Literal(str(schema)),
            table_col=sql.Identifier("table_name"),
            table=sql.Literal(str(table)),
        )

    @classmethod
    def product_end_query(cls) -> sql.Composed:
        """`SELECT max("dlycaldt") FROM "crsp_a_stock"."dsf_v2"` -- the
        product's last day, i.e. which annual vintage this account holds."""
        return sql.SQL("SELECT max({column}) FROM {table}").format(
            column=sql.Identifier("dlycaldt"),
            table=sql.Identifier(cls.STOCK_SCHEMA, cls.DAILY_TABLE),
        )

    # -- network calls ------------------------------------------------------

    @classmethod
    def assert_entitled(cls, session, schemas) -> None:
        """Raise `WrdsEntitlementError` naming EVERY unusable schema.

        Run before the first data query of a pull, so an unsubscribed product
        stops the run with zero COPY calls rather than failing every batch one
        by one (the D-21 rule the TAQ provider already follows).
        """
        missing = [
            schema for schema in schemas if not session.schema_usable(schema)
        ]
        if missing:
            raise _wrds.WrdsEntitlementError(
                f"The WRDS account cannot read {', '.join(sorted(missing))}: "
                f"its subscription does not cover "
                f"{'that schema' if len(missing) == 1 else 'those schemas'}. "
                f"CRSP Stock v2 needs the CRSP annual-update subscription "
                f"(and the Compustat/CCM ones for the index universes). "
                f"Nothing was downloaded."
            )

    @classmethod
    def product_end(cls, session) -> date:
        rows = session.fetch_rows(cls.product_end_query())
        if not rows or rows[0][0] is None:
            raise CrspProductEndError(
                f"{cls.STOCK_SCHEMA}.{cls.DAILY_TABLE} reported no maximum "
                f"dlycaldt, so the CRSP product end is unknown and no window "
                f"can be checked against it. Nothing was downloaded."
            )
        return cls._as_date(rows[0][0])

    @classmethod
    def table_columns(cls, session, schema, table) -> tuple[str, ...]:
        """The table's columns in SERVER order, sorted locally."""
        rows = session.fetch_rows(cls.columns_query(schema, table))
        return tuple(
            name for name, _ in sorted(rows, key=lambda row: int(row[1]))
        )

    @classmethod
    def count(cls, session, schema, table, where) -> int:
        rows = session.fetch_rows(cls.count_query(schema, table, where))
        return int(rows[0][0])

    @classmethod
    def copy(cls, session, schema, table, columns, where) -> bytes:
        return session.copy_csv(cls.copy_query(schema, table, columns, where))

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _as_date(value) -> date:
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value
        return date.fromisoformat(str(value)[:10])


def year_pages(start, end) -> list[tuple[date, date]]:
    """Each CALENDAR YEAR of `[start, end]`, clipped to the window, ascending.

    `[(2018-06-01, 2018-12-31), (2019-01-01, 2019-12-31), (2020-01-01,
    2020-03-31)]` for `2018-06-01..2020-03-31`; `[]` for an inverted window.

    MODULE-LEVEL, and that is the point of it existing at all. Two callers
    need "which rows does this page cover": `_fetch_page`, which pulls them,
    and `CrspVolumeProbe`, which prices them BEFORE the pull. Two inline
    copies of the arithmetic would be two definitions that can drift, and the
    drift is invisible in the dangerous direction -- an estimate quoted
    against slightly different bounds than the pull uses is an estimate that
    silently understates the disk it is about to consume.

    Why a YEAR and not a month or the whole window: the window is the unit of
    resume (a failed page re-runs whole), so a year bounds a retry at roughly
    250 trading days per PERMNO while keeping the page count for a 26-year
    S&P backfill to ~26 per batch rather than ~312.
    """
    start = CrspQueries._as_date(start)
    end = CrspQueries._as_date(end)
    if start > end:
        return []
    return [
        (max(start, date(year, 1, 1)), min(end, date(year, 12, 31)))
        for year in range(start.year, end.year + 1)
    ]


class WrdsCrspDailyAcquisition(Acquisition):
    """CRSP Stock v2 daily bars behind the shared `Acquisition` base.

    One page = one CALENDAR YEAR of one PERMNO batch: `_fetch_page` reads the
    year named by the page token (the window's first year when there is none)
    and hands back the next year as the next token. The base `_fetch_batch`
    owns the loop, the shard writes, the page ledger and resume.

    The raw tier lands under `.../wrds_crsp/wrds/month=YYYY-MM/` with one row
    per `(permno, dlycaldt)`, EXACTLY as CRSP serves it: no derived price, no
    adjusted series, no filter. Everything derived is `dataset/crsp.py`'s job,
    because the adjustment anchor is a property of the whole window and not of
    a page (RESEARCH Pattern 4).

    Raw `symbol` is the PERMNO as a string, and a typed `permno` Int64 column
    rides along. `"14593"` satisfies `TRADEABLE_TICKER_PATTERN`, so the base's
    own `_validate_symbols` path guard applies unchanged.
    """

    VENDOR = "wrds"

    #: The one data type this class serves. NOT a tuple like TAQ's
    #: `TICK_DATA_TYPES`: `1d` has no second CRSP shape to choose between.
    DATA_TYPE = "crsp_daily"

    #: PERMNOs per COPY. A working value (RESEARCH Q7): an S&P-scale roster is
    #: ~1,100 distinct PERMNOs, so 200 gives ~6 batches x ~26 yearly pages.
    DEFAULT_BATCH_SIZE = 200

    #: One shared connection, so one worker (D-20/D-03). A different value is
    #: refused in `__init__` -- every extra connection can push a Duo prompt.
    DEFAULT_MAX_WORKERS = 1

    #: Count each page with the COPY's own WHERE before pulling it, and fail
    #: the page on a mismatch. Overridable via `kwargs["verify_page_counts"]`.
    DEFAULT_VERIFY_PAGE_COUNTS = True

    #: Rough bytes per raw row, for the caller-side volume arithmetic. 50
    #: columns of mostly short numerics.
    DEFAULT_BYTES_PER_ROW = 150

    CREDENTIAL_ENV_VARS = (_wrds.USERNAME_ENV,)
    REDACTION = "<WRDS CREDENTIAL REDACTED>"

    #: `downloads/us_equity/1d/{DEFAULT_SUBDIR}/wrds`.
    DEFAULT_SUBDIR = "wrds_crsp"

    #: The reference tier's directory name, a SIBLING of the raw root.
    REFERENCE_DIR_NAME = "_reference"

    #: The vintage stamp's directory name, a SIBLING of BOTH roots (D-01).
    VINTAGE_DIR_NAME = "_vintage"

    #: How many offending values a page refusal names. BOUNDED because a
    #: malformed page can be malformed in every row: an unbounded list would
    #: put a 250,000-entry repr into the failure manifest, which is a JSON
    #: file an operator has to read.
    SAMPLE_LIMIT = 10

    #: The 50 `dsf_v2` columns this class reads, in server order (live check
    #: `C3_columns`). PINNED: the SELECT, and therefore every shard, is
    #: identical for every page of every era. A column the server stops
    #: reporting fails the page loudly; nothing is null-filled.
    CRSP_COLUMNS = (
        "permno", "hdrcusip", "permco", "siccd", "nasdissuno", "yyyymmdd",
        "sharetype", "securitytype", "securitysubtype", "usincflg",
        "issuertype", "primaryexch", "conditionaltype", "tradingstatusflg",
        "dlycaldt", "dlydelflg", "dlyprc", "dlyprcflg", "dlycap", "dlycapflg",
        "dlyprevprc", "dlyprevprcflg", "dlyprevdt", "dlyprevcap",
        "dlyprevcapflg", "dlyret", "dlyretx", "dlyreti", "dlyretmissflg",
        "dlyretdurflg", "dlyorddivamt", "dlynonorddivamt", "dlyfacprc",
        "dlydistretflg", "dlyvol", "dlyclose", "dlylow", "dlyhigh", "dlybid",
        "dlyask", "dlyopen", "dlynumtrd", "dlymmcnt", "dlyprcvol",
        "dlycumfacpr", "dlycumfacshr", "cusip", "ticker", "exchangetier",
        "shrout",
    )

    #: The pinned shard projection and order.
    RAW_COLUMNS = ("timestamp", "symbol", "vendor", *CRSP_COLUMNS)

    #: `timestamp` is `Datetime("us")` at midnight, matching Tiingo's daily
    #: tier exactly (`acquisition/tiingo.py`), so the two daily vendors'
    #: shards carry the same time type and a panel built from either indexes
    #: identically.
    RAW_SCHEMA = {
        "timestamp": pl.Datetime("us"),
        "symbol": pl.String,
        "vendor": pl.String,
        "permno": pl.Int64,
        "hdrcusip": pl.String,
        "permco": pl.Int64,
        "siccd": pl.Int64,
        "nasdissuno": pl.Int64,
        "yyyymmdd": pl.Int64,
        "sharetype": pl.String,
        "securitytype": pl.String,
        "securitysubtype": pl.String,
        "usincflg": pl.String,
        "issuertype": pl.String,
        "primaryexch": pl.String,
        "conditionaltype": pl.String,
        "tradingstatusflg": pl.String,
        "dlycaldt": pl.Date,
        "dlydelflg": pl.String,
        "dlyprc": pl.Float64,
        "dlyprcflg": pl.String,
        "dlycap": pl.Float64,
        "dlycapflg": pl.String,
        "dlyprevprc": pl.Float64,
        "dlyprevprcflg": pl.String,
        "dlyprevdt": pl.Date,
        "dlyprevcap": pl.Float64,
        "dlyprevcapflg": pl.String,
        "dlyret": pl.Float64,
        "dlyretx": pl.Float64,
        "dlyreti": pl.Float64,
        "dlyretmissflg": pl.String,
        "dlyretdurflg": pl.String,
        "dlyorddivamt": pl.Float64,
        "dlynonorddivamt": pl.Float64,
        "dlyfacprc": pl.Float64,
        "dlydistretflg": pl.String,
        "dlyvol": pl.Float64,
        "dlyclose": pl.Float64,
        "dlylow": pl.Float64,
        "dlyhigh": pl.Float64,
        "dlybid": pl.Float64,
        "dlyask": pl.Float64,
        "dlyopen": pl.Float64,
        "dlynumtrd": pl.Int64,
        "dlymmcnt": pl.Int64,
        "dlyprcvol": pl.Float64,
        "dlycumfacpr": pl.Float64,
        "dlycumfacshr": pl.Float64,
        "cusip": pl.String,
        "ticker": pl.String,
        "exchangetier": pl.String,
        "shrout": pl.Int64,
    }

    def __init__(self, config: AcquisitionConfig):
        super().__init__(config)
        # Validated EAGERLY, before any session exists, so a bad config raises
        # at construction rather than inside a worker.
        self._data_type  # resolves and validates, or raises
        max_workers = self._knob("max_workers", self.DEFAULT_MAX_WORKERS)
        if max_workers != 1:
            raise ValueError(
                f"{self.class_name}: kwargs['max_workers']={max_workers!r} is "
                f"refused; WRDS acquisition runs on ONE shared connection "
                f"(D-20), because every extra connection can push a Duo prompt "
                f"to your phone and the WRDS role allows only 7."
            )
        # Through the MODULE ATTRIBUTE at call time, so the test fixture's
        # patch of `wrds_taq.WrdsSession` takes effect (RESEARCH Pattern 1).
        self._session = _wrds.WrdsSession.shared()
        self._server_columns_cache: dict[tuple[str, str], tuple[str, ...]] = {}

    @property
    def _data_type(self) -> str:
        """Always `crsp_daily`, and only under `frequency="1d"`.

        There is deliberately no default: the value names the watermark
        namespace and is the capability key `registry.run()` resolves this
        class through, so a config that does not say it is a config that could
        have meant the Tiingo daily tier.
        """
        frequency = self.config.frequency
        data_type = self._knob("data_type", None)
        if frequency != "1d" or data_type != self.DATA_TYPE:
            raise ValueError(
                f"{self.class_name}: needs frequency '1d' with "
                f"kwargs['data_type'] set to {self.DATA_TYPE!r}; got frequency "
                f"{frequency!r} and data_type {data_type!r}. There is "
                f"deliberately NO default -- the data type names the watermark "
                f"namespace and is the capability key the registry resolves "
                f"this class through."
            )
        return data_type

    # -- failure policy (D-21) ----------------------------------------------

    #: Exceptions that mean "the one session, or the account, is unusable".
    GLOBAL_STOP_ERRORS = (
        _wrds.WrdsSessionError,
        _wrds.WrdsEntitlementError,
        psycopg2.OperationalError,
        psycopg2.InterfaceError,
    )

    def _classify_error(self, exc: BaseException) -> str:
        """In this vendor `"quota"` means GLOBAL STOP, not an allocation.

        Identical to the TAQ provider's policy and for the identical reason: a
        dead single session or a missing entitlement is never one PERMNO's
        fault, so recording it per symbol would defame every remaining one,
        and retrying batch by batch would reconnect -- one Duo push per batch.
        """
        if isinstance(exc, self.GLOBAL_STOP_ERRORS):
            return "quota"
        return super()._classify_error(exc)

    # -- the annual vintage edge (D-01) -------------------------------------

    @classmethod
    def resolve_window(
        cls, session, start, end, *, clip: bool
    ) -> tuple[date, date, date | None]:
        """`(start, effective_end, clipped_product_end_or_None)`.

        Probes the product end and hands it to `window_for_product_end` below.
        `_run` does NOT call this: it probes ONCE and reuses the answer for
        both the window and the vintage stamp, because a second `max(dlycaldt)`
        per run would be a second full-table aggregate for a value that cannot
        change mid-run.
        """
        return cls.window_for_product_end(
            CrspQueries.product_end(session), start, end, clip=clip
        )

    @classmethod
    def window_for_product_end(
        cls, product_end, start, end, *, clip: bool
    ) -> tuple[date, date, date | None]:
        """The same answer as `resolve_window`, from an ALREADY-probed end.

        `crsp_a_stock` is the annual update product, so its last day is a hard
        edge rather than "data not in yet". Both arms below refuse rather than
        return an empty result: a silent empty pull over a 2026 window looks
        exactly like a roster with no members, and the operator would go
        looking for the wrong bug.

        PURE -- it touches no session, which is what lets `_run` reuse one
        probe and lets a test pin the arithmetic without a server.
        """
        start = CrspQueries._as_date(start)
        end = CrspQueries._as_date(end)
        product_end = CrspQueries._as_date(product_end)

        if start > product_end:
            raise CrspProductEndError(
                f"start_date {start.isoformat()} is past the CRSP product end "
                f"{product_end.isoformat()}. {CrspQueries.STOCK_SCHEMA} is the "
                f"ANNUAL UPDATE product, so its last day moves once a year at "
                f"the WRDS refresh, not daily. Clipping cannot help: the whole "
                f"window is past the edge, so there is nothing to clip it to. "
                f"Choose a start inside the covered range; nothing was "
                f"downloaded."
            )
        if end > product_end:
            if not clip:
                raise CrspProductEndError(
                    f"end_date {end.isoformat()} is past the CRSP product end "
                    f"{product_end.isoformat()}. {CrspQueries.STOCK_SCHEMA} is "
                    f"the ANNUAL UPDATE product and gains a year at the WRDS "
                    f"refresh. Lower --end-date to "
                    f"{product_end.isoformat()}, or pass "
                    f"kwargs['clip_to_product_end']=True to have the window "
                    f"clipped for you; nothing was downloaded."
                )
            return start, product_end, product_end
        return start, end, None

    @classmethod
    def vintage_path_for(cls, config: AcquisitionConfig) -> Path:
        """`.../{subdir}/_vintage/wrds.json`, a SIBLING of both roots.

        Not under the WATERMARK root, and that is the load-bearing half:
        `CoverageLedger.iter_watermark_symbols` lists every `*.json` there and
        reads its stem as a symbol, so a `wrds.json` parked beside the
        watermarks would become a phantom PERMNO in every coverage report --
        and, worse, one whose "watermark" has no `last_date`. Not under the RAW
        root either, for the reason `reference_dir_for` gives: `_scan_raw`
        walks every file below it.
        """
        return (
            Path(config.raw_data_dir_path).parent
            / cls.VINTAGE_DIR_NAME
            / f"{cls.VENDOR}.json"
        )

    def _assert_one_vintage(self, product_end: date) -> None:
        """Stamp the probed vintage, or refuse a raw tier built from another.

        Read-then-write rather than write-always: the stamp is the raw tier's
        PROVENANCE, so overwriting it with whatever this run happened to probe
        would destroy the only record that the shards on disk came from an
        earlier release.
        """
        path = self.vintage_path_for(self.config)
        if path.exists():
            try:
                stamped = json.loads(path.read_text(encoding="utf-8")).get(
                    "product_end"
                )
            except (OSError, ValueError) as exc:
                raise CrspVintageError(
                    f"{self.class_name}: the vintage stamp {path} could not be "
                    f"read ({exc}). It records which CRSP annual release this "
                    f"raw tier was built from, so a run cannot proceed without "
                    f"it; nothing was downloaded."
                ) from exc
            if stamped and CrspQueries._as_date(stamped) != product_end:
                raise CrspVintageError(
                    f"{self.class_name}: this raw tier was built from the CRSP "
                    f"vintage ending {CrspQueries._as_date(stamped).isoformat()}"
                    f", but the account now reads the vintage ending "
                    f"{product_end.isoformat()}. CRSP REVISES history between "
                    f"annual releases -- restated delisting returns, corrected "
                    f"prices -- so two vintages must never share one raw tier: "
                    f"the panel built from it would be neither, with nothing on "
                    f"disk recording the seam. Start a FRESH raw tier by "
                    f"passing a new subdir to build_config (e.g. "
                    f"subdir='wrds_crsp_{product_end:%Y}'), or delete the raw "
                    f"root {self.config.raw_data_dir_path} together with its "
                    f"_watermarks/{self.VENDOR} and {self.VINTAGE_DIR_NAME} "
                    f"siblings and pull again. Nothing was downloaded."
                )
            return
        write_json_atomically(
            path, {"product_end": product_end.isoformat()}, indent=2, sort_keys=True
        )

    def _run(self, symbols: list[str] | None, from_watermark: bool):
        """Check the roster, the entitlement, the product end and the vintage,
        THEN run.

        All four happen before the base runner dispatches a single batch, so a
        ticker-shaped roster, an unsubscribed account, a window past the
        vintage or a second vintage over one raw tier raises out of
        `download()`/`refresh()` with zero COPY calls.

        The ORDER is not incidental. The PERMNO check is first because it
        costs nothing; entitlement is next because the product-end probe is
        itself a query against the schema the account may not read, so probing
        first would report a missing subscription as a broken session; the
        vintage check is last because it needs the probed end.
        """
        self._assert_permnos(
            self._validate_symbols(list(symbols or self.config.symbols))
        )
        CrspQueries.assert_entitled(self._session, (CrspQueries.STOCK_SCHEMA,))

        product_end = CrspQueries.product_end(self._session)
        clip = bool(self._knob("clip_to_product_end", False))
        start, end, clipped = self.window_for_product_end(
            product_end,
            self.config.start_date,
            self.config.end_date,
            clip=clip,
        )
        if clipped is not None:
            logger.warning(
                f"{self.class_name}: end_date {self.config.end_date} is past "
                f"the {CrspQueries.STOCK_SCHEMA} product end "
                f"{clipped.isoformat()}; the window was clipped to it "
                f"(kwargs['clip_to_product_end'])."
            )
            self.config = dataclasses.replace(
                self.config, end_date=end.isoformat()
            )

        self._assert_one_vintage(product_end)
        return super()._run(symbols, from_watermark)

    def _assert_permnos(self, symbols) -> None:
        """Every raw symbol is a PERMNO (a digit string), or the run refuses.

        Checked at the TOP of `_run` as well as inside `_fetch_page`, because
        a ticker roster is an operator mistake about the whole run, not one
        batch's bad luck: recorded per batch it would land in the failure
        manifest as if WRDS had rejected those securities.
        """
        for symbol in symbols:
            if not str(symbol).isdigit():
                raise ValueError(
                    f"{self.class_name}: symbol {symbol!r} is not a PERMNO. "
                    f"The CRSP raw tier is keyed by PERMNO (a digit string), "
                    f"not by ticker -- a ticker is derived at conversion time, "
                    f"so that a rename never invalidates a watermark. Resolve "
                    f"the roster to PERMNOs first; nothing was downloaded."
                )

    # -- one page = one calendar year ---------------------------------------

    def _empty_page(self) -> pl.DataFrame:
        return pl.DataFrame(schema=self.RAW_SCHEMA).select(self.RAW_COLUMNS)

    def _server_columns(self, schema: str, table: str) -> tuple[str, ...]:
        key = (schema, table)
        cached = self._server_columns_cache.get(key)
        if cached is None:
            cached = CrspQueries.table_columns(self._session, schema, table)
            self._server_columns_cache[key] = cached
        return cached

    def _fetch_page(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str,
        page_token: str | None = None,
    ) -> tuple[pl.DataFrame, str | None]:
        """One calendar year of `dsf_v2` rows for one PERMNO batch.

        Returns `(frame, next_token)`, `next_token` being the next year as a
        string or `None` after the window's last year.

        **No sort, no dedup, no filter of any row.** The page is checked and
        either accepted whole or refused: a duplicate `(permno, dlycaldt)`
        raises, a row for a PERMNO outside the batch raises, and a row dated
        outside the page bounds raises. Filtering instead would hide a WHERE
        clause that stopped doing what it says.
        """
        symbols = self._validate_symbols(symbols)
        # Before ANY query: raw symbols are PERMNOs, and a non-digit value
        # would become a SQL literal and a shard path segment. `_run` checks
        # the same thing for the whole roster; this is the guard for the
        # direct `_fetch_page` call, which no roster check precedes.
        self._assert_permnos(symbols)

        # The SHARED page definition (`year_pages`), never a second inline
        # copy: `CrspVolumeProbe` prices exactly these bounds.
        pages = year_pages(start_date, end_date)
        if not pages:
            return self._empty_page(), None
        years = [page_start.year for page_start, _ in pages]

        year = int(page_token) if page_token else years[0]
        if year not in years:
            raise ValueError(
                f"{self.class_name}: page token {page_token!r} names no year "
                f"inside [{start_date}, {end_date}]. A token is the ISO year of "
                f"the next page to read; refusing rather than reading a year "
                f"outside the requested window."
            )
        position = years.index(year)
        next_token = (
            str(years[position + 1]) if position + 1 < len(years) else None
        )
        page_start, page_end = pages[position]

        schema, table = CrspQueries.STOCK_SCHEMA, CrspQueries.DAILY_TABLE
        server_columns = set(self._server_columns(schema, table))
        missing = [
            name for name in self.CRSP_COLUMNS if name not in server_columns
        ]
        if missing:
            # A drift the live-check evidence did not show fails loudly; it is
            # never null-filled, because a silently absent `dlycumfacshr`
            # would make every adjusted volume wrong without any error.
            raise ValueError(
                f"{self.class_name}: {schema}.{table} reports no {missing} "
                f"column(s); the table layout no longer matches the one this "
                f"class was verified against (D-02). Columns seen: "
                f"{sorted(server_columns)}."
            )

        where = CrspQueries.daily_where(symbols, page_start, page_end)
        expected_rows = (
            CrspQueries.count(self._session, schema, table, where)
            if self._knob("verify_page_counts", self.DEFAULT_VERIFY_PAGE_COUNTS)
            else None
        )

        raw = CrspQueries.copy(
            self._session, schema, table, self.CRSP_COLUMNS, where
        )
        frame = pl.read_csv(io.BytesIO(raw), infer_schema=False)
        if tuple(frame.columns) != tuple(self.CRSP_COLUMNS):
            raise ValueError(
                f"{self.class_name}: {schema}.{table} COPY returned columns "
                f"{frame.columns}, not the requested "
                f"{list(self.CRSP_COLUMNS)}."
            )
        if expected_rows is not None and frame.height != expected_rows:
            raise ValueError(
                f"{self.class_name}: {schema}.{table} page {year} "
                f"({page_start.isoformat()}..{page_end.isoformat()}) COPY "
                f"returned {frame.height} row(s) but count(*) with the same "
                f"WHERE reported {expected_rows}; the page is incomplete and "
                f"is not recorded, so the next run re-fetches year {year}."
            )
        if frame.height == 0:
            return self._empty_page(), next_token

        # DATE columns are parsed with `str.to_date`, not cast: a String ->
        # Date cast is deprecated in polars 1.44 and removed in 2.0, and the
        # explicit parse is also what makes a malformed field a null rather
        # than a whole-page failure.
        frame = frame.with_columns(
            pl.col(name).str.to_date(strict=False).alias(name)
            for name, dtype in self.RAW_SCHEMA.items()
            if dtype == pl.Date and name in self.CRSP_COLUMNS
        )
        frame = frame.cast(
            {
                name: dtype
                for name, dtype in self.RAW_SCHEMA.items()
                if name in self.CRSP_COLUMNS and dtype != pl.Date
            }
        )
        frame = frame.with_columns(
            pl.col("dlycaldt").cast(pl.Datetime("us")).alias("timestamp"),
            pl.col("permno").cast(pl.String).alias("symbol"),
            pl.lit(self.VENDOR).alias("vendor"),
        )

        self._assert_unique_keys(frame, schema, table)
        self._assert_page_belongs(frame, symbols, page_start, page_end)
        return frame.cast(self.RAW_SCHEMA).select(self.RAW_COLUMNS), next_token

    def _assert_unique_keys(
        self, frame: pl.DataFrame, schema: str, table: str
    ) -> None:
        """`(permno, dlycaldt)` is unique on the page, or the batch fails.

        D-19's live check established the uniqueness on the real table; this
        asserts it on every page anyway, because the consequence of being
        wrong is invisible. A duplicate reaches `dedup_raw_frame(keep="last")`
        downstream and is collapsed ARBITRARILY -- one of two prices survives,
        with nothing recording that a choice was made.
        """
        duplicates = (
            frame.group_by(["permno", "dlycaldt"])
            .agg(pl.len().alias("rows"))
            .filter(pl.col("rows") > 1)
            .sort(["permno", "dlycaldt"])
        )
        if duplicates.height:
            sample = [
                f"{record['permno']}@{record['dlycaldt']}x{record['rows']}"
                for record in duplicates.head(self.SAMPLE_LIMIT).to_dicts()
            ]
            raise ValueError(
                f"{self.class_name}: {schema}.{table} returned "
                f"{duplicates.height} duplicated (permno, dlycaldt) key(s) "
                f"across {frame.height} row(s), first "
                f"{len(sample)} of them {sample}. The table is unique on that "
                f"pair (D-19), so this page is refused rather than "
                f"de-duplicated -- a silent dedup would drop one of two prices "
                f"with no record."
            )

    def _assert_page_belongs(
        self,
        frame: pl.DataFrame,
        symbols: list[str],
        page_start: date,
        page_end: date,
    ) -> None:
        """Every row is for a REQUESTED PERMNO and inside the page bounds.

        Checks, never filters: a stranger PERMNO means the WHERE stopped doing
        what it says, and a row dated outside the page would land under a
        `month=` partition this page does not own -- where the next run's
        deterministic overwrite would not reach it.
        """
        wanted = {int(symbol) for symbol in symbols}
        seen = {
            int(value)
            for value in frame.get_column("permno").unique().to_list()
            if value is not None
        }
        strangers = sorted(seen - wanted)
        if strangers:
            raise ValueError(
                f"{self.class_name}: the daily COPY returned rows for "
                f"{len(strangers)} PERMNO(s) that were not requested, first "
                f"{strangers[: self.SAMPLE_LIMIT]} (requested "
                f"{len(wanted)}: {sorted(wanted)[: self.SAMPLE_LIMIT]}). "
                f"Refusing the page rather than filing another security's "
                f"prices under a requested PERMNO."
            )
        off_page = frame.filter(
            (pl.col("dlycaldt") < pl.lit(page_start))
            | (pl.col("dlycaldt") > pl.lit(page_end))
        )
        if off_page.height:
            dates = sorted(
                {str(value) for value in off_page["dlycaldt"].to_list()}
            )
            raise ValueError(
                f"{self.class_name}: the daily COPY returned "
                f"{off_page.height} row(s) dated outside the page bounds "
                f"[{page_start.isoformat()}, {page_end.isoformat()}], on "
                f"{len(dates)} date(s), first {dates[: self.SAMPLE_LIMIT]}. "
                f"Such a row would land under a month= partition this page "
                f"does not own, where the next run's deterministic overwrite "
                f"would never reach it."
            )

    # -- config -------------------------------------------------------------

    @classmethod
    def build_config(
        cls,
        symbols,
        start_date: str | None = None,
        end_date: str | None = None,
        kwargs: dict | None = None,
        subdir: str = DEFAULT_SUBDIR,
    ) -> AcquisitionConfig:
        """The `AcquisitionConfig` for a CRSP daily pull, built directly.

        This is the CRSP capability's own `config_factory` (03.10 D-12): a
        classmethod here rather than a function in `quantlab/config`, the same
        shape `WrdsTaqNbboAcquisition.build_config` already has. The raw root
        TERMINATES at the vendor segment (`.../{subdir}/wrds`) and the
        watermarks live in the sibling `.../{subdir}/_watermarks/wrds`. No
        credential goes into the config.
        """
        merged = dict(kwargs or {})
        data_type = merged.get("data_type", cls.DATA_TYPE)
        if data_type != cls.DATA_TYPE:
            raise ValueError(
                f"{cls.__name__}.build_config: kwargs['data_type']="
                f"{data_type!r} conflicts with this source, which serves only "
                f"{cls.DATA_TYPE!r}."
            )
        merged["data_type"] = cls.DATA_TYPE
        merged.setdefault("bytes_per_row", cls.DEFAULT_BYTES_PER_ROW)
        downloads = get_data_root() / "downloads" / "us_equity" / "1d" / subdir
        return AcquisitionConfig(
            market="us_equity",
            frequency="1d",
            vendor=cls.VENDOR,
            raw_data_dir_path=str(downloads / cls.VENDOR),
            watermark_path=str(downloads / "_watermarks" / cls.VENDOR),
            symbols=tuple(str(symbol) for symbol in symbols),
            start_date=start_date,
            end_date=end_date,
            kwargs=merged,
        )

    @classmethod
    def reference_dir_for(cls, config: AcquisitionConfig) -> Path:
        """`.../{subdir}/_reference`, the SIBLING of the raw root.

        Never under it: `StockDataset._scan_raw` globs `**/*.pqt` below the
        raw root, and a reference `.parquet` -- or its `manifest.json` -- in
        that tree would be walked by the same scan. Derived from the config
        rather than rebuilt from `get_data_root()` so a config pointed at a
        custom root keeps its reference tier beside its own raw tier.
        """
        return Path(config.raw_data_dir_path).parent / cls.REFERENCE_DIR_NAME
