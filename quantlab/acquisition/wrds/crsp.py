"""WRDS CRSP Stock v2 daily acquisition, keyed by PERMNO.

``WrdsCrspDailyAcquisition`` pulls ``crsp_a_stock.dsf_v2``, the CRSP daily
stock table, through the shared WRDS session in ``quantlab.acquisition.wrds
.taq``. One page is one calendar year of one PERMNO batch, fetched with a bare
``COPY (SELECT ... WHERE permno = ANY(...) AND dlycaldt BETWEEN ...)`` and
stored exactly as CRSP serves it: no ordering, no de-duplication, no derived
or adjusted series. Rows are keyed by PERMNO rather than ticker because
``dsf_v2`` cannot spell share classes and a ticker can change; the ticker is
derived later, at conversion time, in ``quantlab.dataset.crsp``.

``CrspQueries`` holds the SQL builders (pure, testable without a server) and
the thin network calls that hand them to the session. ``year_pages`` defines
the page boundaries shared by the acquisition and ``CrspVolumeProbe``, which
counts the rows a pull would move so the volume guard can price it first.

``crsp_a_stock`` is CRSP's annual-update product, so its last day is a hard
edge: a window past it is refused unless ``kwargs["clip_to_product_end"]`` is
set, and a raw tier records the vintage it was built from so two annual
releases are never mixed. The session is reached through the ``taq`` module
attribute at call time, never bound by name, so a test can replace it.
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

from quantlab.acquisition.wrds import taq as _wrds
from quantlab.base.acquisition import Acquisition
from quantlab.base.config import AcquisitionConfig
from quantlab.config import get_data_root
from quantlab.utils.atomic import write_json_atomically


class CrspProductEndError(ValueError):
    """The requested window lies past the CRSP product's last day.

    A ``ValueError`` rather than a session error: nothing is wrong with the
    connection or the subscription, the data simply does not exist yet,
    because ``crsp_a_stock`` is updated once a year. It is raised before any
    COPY, so the message can promise that nothing was downloaded.

    Examples
    --------
    >>> try:
    ...     WrdsCrspDailyAcquisition.window_for_product_end(
    ...         "2025-12-31", "2020-01-01", "2026-06-30", clip=False
    ...     )
    ... except CrspProductEndError as exc:
    ...     print(str(exc)[:59])
    end_date 2026-06-30 is past the CRSP product end 2025-12-31
    """


class CrspVintageError(ValueError):
    """The raw tier was built from a different CRSP annual release.

    CRSP revises history between releases (restated delisting returns,
    corrected prices, re-used PERMNOs), so two vintages sharing one raw root
    would produce a panel that is neither, with nothing on disk recording the
    seam. Raised before any COPY.

    Examples
    --------
    >>> try:
    ...     acq.download()
    ... except CrspVintageError as exc:
    ...     print("start a fresh raw tier:", exc)
    """


class CrspQueries:
    """SQL statements for the CRSP tables, built with ``psycopg2.sql`` only.

    The builders are pure classmethods that return a composable and touch no
    connection, so their exact text can be checked without a server. The
    network calls take a session and only hand a built statement to the
    session's ``schema_usable`` / ``fetch_rows`` / ``copy_csv``. Every value
    is a ``sql.Literal`` and every identifier a ``sql.Identifier``; no
    statement text is assembled by string formatting, which also quotes
    reserved column names such as ``comp.idxcst_his``'s ``from`` and ``thru``.

    Examples
    --------
    >>> where = CrspQueries.daily_where([14593], "2020-08-01", "2020-08-31")
    >>> query = CrspQueries.copy_query(
    ...     "crsp_a_stock", "dsf_v2", ("permno", "dlycaldt", "dlyprc"), where
    ... )
    >>> raw = CrspQueries.copy(
    ...     session, "crsp_a_stock", "dsf_v2", ("permno", "dlycaldt"), where
    ... )
    """

    STOCK_SCHEMA = "crsp_a_stock"
    INDEX_SCHEMA = "crsp_a_indexes"
    COMPUSTAT_SCHEMA = "comp"
    CCM_SCHEMA = "crsp_a_ccm"

    #: The daily table. ``dsf_v2`` is unique on ``(permno, dlycaldt)``; if
    #: that ever stopped holding, ``stkdlysecuritydata`` is the fallback and
    #: this constant is the only edit.
    DAILY_TABLE = "dsf_v2"

    # -- pure builders ------------------------------------------------------

    @classmethod
    def daily_where(cls, permnos, start, end) -> sql.Composed:
        """Build the PERMNO-and-date predicate for the daily table.

        PERMNOs are coerced to ``int`` so a value that is not a PERMNO cannot
        reach the statement even as a literal.

        Raises
        ------
        ValueError
            If ``permnos`` is empty. Without the PERMNO predicate
            this would be a query over the whole 110-million-row table.

        Examples
        --------
        >>> where = CrspQueries.daily_where(
        ...     ["14593", "10107"], "2020-01-01", "2020-12-31"
        ... )

        which renders as::

            "permno" = ANY(ARRAY[14593, 10107])
            AND "dlycaldt" BETWEEN '2020-01-01' AND '2020-12-31'
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
        """Build a ``COPY (SELECT ...) TO STDOUT`` statement in CSV format.

        The statement carries no ORDER BY, GROUP BY or DISTINCT.

        Parameters
        ----------
        schema
            Schema name, quoted as an identifier.
        table
            Table name, quoted as an identifier.
        columns
            Column names to select, in this order.
        where
            An optional ``psycopg2.sql`` predicate.

        Examples
        --------
        >>> query = CrspQueries.copy_query(
        ...     "crsp_a_stock", "dsf_v2", ("permno", "dlycaldt", "dlyprc"),
        ...     CrspQueries.daily_where([14593], "2020-08-01", "2020-08-31"),
        ... )

        which renders as::

            COPY (SELECT "permno", "dlycaldt", "dlyprc"
                  FROM "crsp_a_stock"."dsf_v2"
                  WHERE "permno" = ANY(ARRAY[14593])
                    AND "dlycaldt" BETWEEN '2020-08-01' AND '2020-08-31')
            TO STDOUT WITH (FORMAT csv, HEADER true)
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
        """Build ``SELECT count(*)`` over the same ``where`` a COPY uses.

        Sharing the predicate object means a count and the pull it checks
        cannot select different rows.

        Examples
        --------
        >>> query = CrspQueries.count_query("crsp_a_stock", "stkdelists")

        which renders as::

            SELECT count(*) FROM "crsp_a_stock"."stkdelists" 
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
        """Build the ``information_schema.columns`` read for one table.

        Schema and table are values in this statement, so they travel as
        literals. Ordering by ``ordinal_position`` is done locally rather
        than with ORDER BY, keeping the module free of ordering clauses.

        Examples
        --------
        >>> query = CrspQueries.columns_query("crsp_a_stock", "dsf_v2")

        which renders as::

            SELECT "column_name", "ordinal_position"
            FROM "information_schema"."columns"
            WHERE "table_schema" = 'crsp_a_stock' AND "table_name" = 'dsf_v2'
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
        """Build the query for the daily table's last day.

        The answer says which annual vintage the account currently holds.

        Examples
        --------
        >>> query = CrspQueries.product_end_query()

        which renders as::

            SELECT max("dlycaldt") FROM "crsp_a_stock"."dsf_v2" 
        """
        return sql.SQL("SELECT max({column}) FROM {table}").format(
            column=sql.Identifier("dlycaldt"),
            table=sql.Identifier(cls.STOCK_SCHEMA, cls.DAILY_TABLE),
        )

    # -- network calls ------------------------------------------------------

    @classmethod
    def assert_entitled(cls, session, schemas) -> None:
        """Raise ``WrdsEntitlementError`` naming every schema the role cannot read.

        Run before the first data query of a pull, so an unsubscribed product
        stops the run with zero COPY calls rather than failing every batch.

        Examples
        --------
        >>> CrspQueries.assert_entitled(session, (CrspQueries.STOCK_SCHEMA,))
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
        """Return the daily table's last day, the current annual vintage.

        Raises
        ------
        CrspProductEndError
            If the table reports no maximum date.

        Examples
        --------
        >>> product_end = CrspQueries.product_end(session)
        """
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
        """Return the table's column names in server order.

        Examples
        --------
        >>> columns = CrspQueries.table_columns(session, "crsp_a_stock", "dsf_v2")
        """
        rows = session.fetch_rows(cls.columns_query(schema, table))
        return tuple(
            name for name, _ in sorted(rows, key=lambda row: int(row[1]))
        )

    @classmethod
    def count(cls, session, schema, table, where) -> int:
        """Return ``count(*)`` for the table under ``where``.

        Examples
        --------
        >>> rows = CrspQueries.count(session, "crsp_a_stock", "stkdelists", None)
        """
        rows = session.fetch_rows(cls.count_query(schema, table, where))
        return int(rows[0][0])

    @classmethod
    def copy(cls, session, schema, table, columns, where) -> bytes:
        """Run ``copy_query`` through the session and return the CSV bytes.

        Examples
        --------
        >>> raw = CrspQueries.copy(
        ...     session, "crsp_a_stock", "stkdelists", spec.columns, None
        ... )
        """
        return session.copy_csv(cls.copy_query(schema, table, columns, where))

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _as_date(value) -> date:
        """Coerce a ``datetime``, ``date`` or ISO string to a ``date``."""
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value
        return date.fromisoformat(str(value)[:10])


def year_pages(start, end) -> list[tuple[date, date]]:
    """Split ``[start, end]`` into calendar-year pages, clipped to the window.

    The acquisition pulls one page per year and ``CrspVolumeProbe`` prices
    the same pages, so both use this one definition. A year is the unit of
    resume (a failed page re-runs whole); it bounds a retry at roughly 250
    trading days per PERMNO while keeping a long backfill to a few dozen
    pages per batch.

    Returns
    -------
    list[tuple[date, date]]
        ``[(page_start, page_end), ...]`` ascending, or ``[]`` for an
        inverted window.

    Examples
    --------
    >>> year_pages("2018-06-01", "2020-03-31")
    [(datetime.date(2018, 6, 1), datetime.date(2018, 12, 31)),
     (datetime.date(2019, 1, 1), datetime.date(2019, 12, 31)),
     (datetime.date(2020, 1, 1), datetime.date(2020, 3, 31))]
    >>> year_pages("2020-03-31", "2018-06-01")
    []
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
    """Acquisition of CRSP Stock v2 daily bars from ``crsp_a_stock.dsf_v2``.

    One page is one calendar year of one PERMNO batch: ``_fetch_page`` reads
    the year named by the page token (the window's first year when there is
    none) and returns the next year as the next token. The shared
    ``Acquisition`` base owns the batch loop, shard writes, the page ledger
    and resume.

    The raw tier lands under ``.../wrds_crsp/wrds/month=YYYY-MM/`` with one
    row per ``(permno, dlycaldt)`` exactly as CRSP serves it: no derived
    price, no adjusted series, no filter. Raw ``symbol`` is the PERMNO as a
    string and a typed ``permno`` Int64 column rides along; a digit string
    satisfies the base class's ticker check unchanged. A page is refused, not
    repaired, when it repeats a key, contains an unrequested PERMNO, or holds
    a row outside its bounds.

    Before any batch is dispatched the run checks that every symbol is a
    PERMNO, that the account can read ``crsp_a_stock``, that the window does
    not extend past the product end (or clips it when
    ``kwargs["clip_to_product_end"]`` is set), and that the raw tier's
    recorded vintage matches the one the account now serves. ``max_workers``
    other than 1 is refused because every connection can push a Duo prompt.

    Examples
    --------
    Needs ``WRDS_USERNAME`` and a ``~/.pgpass`` entry; the first query
    opens the connection and may push a Duo prompt.

    >>> cfg = WrdsCrspDailyAcquisition.build_config(
    ...     ("14593", "10107"), start_date="2020-08-01", end_date="2020-08-31"
    ... )
    >>> acq = WrdsCrspDailyAcquisition(cfg).download()
    >>> report = acq.coverage_report()

    Shards land under ``.../wrds_crsp/wrds/month=2020-08/``, watermarks
    under ``.../wrds_crsp/_watermarks/wrds/`` and the vintage stamp at
    ``.../wrds_crsp/_vintage/wrds.json``.
    """

    VENDOR = "wrds"

    #: The one data type this class serves. A single value rather than a
    #: tuple: ``1d`` has no second CRSP shape to choose between.
    DATA_TYPE = "crsp_daily"

    #: PERMNOs per COPY. A working value: an S&P-scale roster is about 1,100
    #: distinct PERMNOs, so 200 gives about six batches of yearly pages.
    DEFAULT_BATCH_SIZE = 200

    #: One shared connection, so one worker. Any other value is refused in
    #: ``__init__``.
    DEFAULT_MAX_WORKERS = 1

    #: Count each page with the COPY's own WHERE before pulling it, and fail
    #: the page on a mismatch. Overridable through
    #: ``kwargs["verify_page_counts"]``.
    DEFAULT_VERIFY_PAGE_COUNTS = True

    #: Rough bytes per raw row for the caller-side volume arithmetic: 50
    #: columns of mostly short numerics.
    DEFAULT_BYTES_PER_ROW = 150

    CREDENTIAL_ENV_VARS = (_wrds.USERNAME_ENV,)
    REDACTION = "<WRDS CREDENTIAL REDACTED>"

    #: The raw tier lives under ``downloads/us_equity/1d/{DEFAULT_SUBDIR}/wrds``.
    DEFAULT_SUBDIR = "wrds_crsp"

    #: The reference tier's directory name, a sibling of the raw root.
    REFERENCE_DIR_NAME = "_reference"

    #: The vintage stamp's directory name, a sibling of both roots.
    VINTAGE_DIR_NAME = "_vintage"

    #: How many offending values a page refusal names. Bounded because a
    #: malformed page can be malformed in every row, and the message ends up
    #: in a JSON failure manifest an operator has to read.
    SAMPLE_LIMIT = 10

    #: The 50 ``dsf_v2`` columns this class reads, in server order. Pinned so
    #: the SELECT, and therefore every shard, is identical for every page. A
    #: column the server stops reporting fails the page; nothing is
    #: null-filled.
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

    #: The shard column projection and order.
    RAW_COLUMNS = ("timestamp", "symbol", "vendor", *CRSP_COLUMNS)

    #: ``timestamp`` is ``Datetime("us")`` at midnight, the same time type as
    #: the Tiingo daily tier, so a panel built from either vendor's shards
    #: indexes identically.
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
        """Validate the config and bind the shared session."""
        super().__init__(config)
        # Validated eagerly, before any session exists, so a bad config raises
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
        # Through the module attribute at call time, so a test's patch of
        # `wrds.taq.WrdsSession` takes effect.
        self._session = _wrds.WrdsSession.shared()
        self._server_columns_cache: dict[tuple[str, str], tuple[str, ...]] = {}

    @property
    def _data_type(self) -> str:
        """Return ``"crsp_daily"`` or raise on a config that does not say so.

        There is deliberately no default: the value names the watermark
        namespace and is the capability key the registry resolves this class
        through, so a config without it could have meant another daily
        vendor.
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

    # -- failure policy -----------------------------------------------------

    #: Exceptions that mean the one session, or the account, is unusable.
    GLOBAL_STOP_ERRORS = (
        _wrds.WrdsSessionError,
        _wrds.WrdsEntitlementError,
        psycopg2.OperationalError,
        psycopg2.InterfaceError,
    )

    def _classify_error(self, exc: BaseException) -> str:
        """Map session and entitlement errors to a run-wide stop.

        The same policy as the TAQ provider: ``"quota"`` here means "stop the
        whole run". A dead session or a missing entitlement is never one
        PERMNO's fault, so it stays out of the failure manifest, and a
        batch-by-batch retry would reconnect and push Duo each time.
        """
        if isinstance(exc, self.GLOBAL_STOP_ERRORS):
            return "quota"
        return super()._classify_error(exc)

    # -- the annual vintage edge --------------------------------------------

    @classmethod
    def resolve_window(
        cls, session, start, end, *, clip: bool
    ) -> tuple[date, date, date | None]:
        """Probe the product end and resolve the window against it.

        Returns ``(start, effective_end, clipped_product_end_or_None)``; see
        ``window_for_product_end`` for the rules. ``_run`` does not call this:
        it probes once and reuses the answer for both the window and the
        vintage stamp.

        Examples
        --------
        >>> WrdsCrspDailyAcquisition.resolve_window(
        ...     session, "2020-01-01", "2026-06-30", clip=True
        ... )
        """
        return cls.window_for_product_end(
            CrspQueries.product_end(session), start, end, clip=clip
        )

    @classmethod
    def window_for_product_end(
        cls, product_end, start, end, *, clip: bool
    ) -> tuple[date, date, date | None]:
        """Resolve a window against an already known product end.

        ``crsp_a_stock`` is the annual-update product, so its last day is a
        hard edge. A ``start`` past it is always refused. An ``end`` past it
        is refused unless ``clip`` is set, in which case the window ends at
        the product end and the third element reports that date. Refusing is
        preferred to returning an empty result, which would look exactly like
        a roster with no members. Pure: it touches no session.

        Returns
        -------
        tuple[date, date, date | None]
            ``(start, effective_end, clipped_product_end_or_None)``.

        Raises
        ------
        CrspProductEndError
            If ``start`` is past the product end, or
            ``end`` is and ``clip`` is false.

        Examples
        --------
        >>> WrdsCrspDailyAcquisition.window_for_product_end(
        ...     "2025-12-31", "2020-01-01", "2026-06-30", clip=True
        ... )
        (datetime.date(2020, 1, 1), datetime.date(2025, 12, 31),
         datetime.date(2025, 12, 31))
        >>> WrdsCrspDailyAcquisition.window_for_product_end(
        ...     "2025-12-31", "2020-01-01", "2024-12-31", clip=False
        ... )
        (datetime.date(2020, 1, 1), datetime.date(2024, 12, 31), None)
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
        """Return the vintage stamp path, ``.../{subdir}/_vintage/wrds.json``.

        The stamp sits beside both the raw root and the watermark root, never
        inside either: the coverage ledger reads every ``*.json`` under the
        watermark root as a symbol, so a stamp there would become a phantom
        PERMNO, and the dataset's raw scan walks every file below the raw
        root.

        Examples
        --------
        >>> WrdsCrspDailyAcquisition.vintage_path_for(cfg)
        PosixPath('<data root>/downloads/us_equity/1d/wrds_crsp/_vintage/wrds.json')
        """
        return (
            Path(config.raw_data_dir_path).parent
            / cls.VINTAGE_DIR_NAME
            / f"{cls.VENDOR}.json"
        )

    def _assert_one_vintage(self, product_end: date) -> None:
        """Stamp the probed vintage, or refuse a raw tier built from another.

        Read-then-write rather than write-always: the stamp is the raw tier's
        provenance, and overwriting it with whatever this run probed would
        destroy the only record that the shards came from an earlier release.
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
        """Check roster, entitlement, product end and vintage, then run.

        All four checks happen before the base runner dispatches a single
        batch, so a ticker-shaped roster, an unsubscribed account, a window
        past the vintage or a second vintage over one raw tier raises out of
        ``download()`` or ``refresh()`` with zero COPY calls. The order
        matters: the PERMNO check costs nothing; entitlement comes before the
        product-end probe because that probe queries the very schema the
        account may not read; the vintage check needs the probed end.
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
        """Refuse the run unless every symbol is a PERMNO (a digit string).

        Checked at the top of ``_run`` as well as inside ``_fetch_page``: a
        ticker roster is an operator mistake about the whole run, and
        recorded per batch it would land in the failure manifest as if WRDS
        had rejected those securities.
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
        """Return an empty frame with the raw schema and column order."""
        return pl.DataFrame(schema=self.RAW_SCHEMA).select(self.RAW_COLUMNS)

    def _server_columns(self, schema: str, table: str) -> tuple[str, ...]:
        """Return the table's server columns, cached per table."""
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
        """Fetch one calendar year of ``dsf_v2`` rows for one PERMNO batch.

        Returns ``(frame, next_token)``, where ``next_token`` is the next year
        as a string or ``None`` after the window's last year. No row is
        sorted, de-duplicated or filtered: the page is counted with the same
        WHERE before the COPY and refused on a mismatch, and refused again on
        a duplicate ``(permno, dlycaldt)``, a PERMNO outside the batch, or a
        row dated outside the page bounds. Filtering instead would hide a
        WHERE clause that stopped doing what it says.
        """
        symbols = self._validate_symbols(symbols)
        # Before any query: raw symbols are PERMNOs, and a non-digit value
        # would become a SQL literal and a shard path segment. `_run` checks
        # the whole roster; this guards a direct `_fetch_page` call.
        self._assert_permnos(symbols)

        # The shared page definition, never a second inline copy:
        # `CrspVolumeProbe` prices exactly these bounds.
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
            # A layout drift fails loudly; it is never null-filled, because a
            # silently absent `dlycumfacshr` would make every adjusted volume
            # wrong without any error.
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

        # Date columns are parsed with `str.to_date`, not cast: a String to
        # Date cast is deprecated in polars 1.44 and removed in 2.0, and the
        # explicit parse makes a malformed field a null rather than a
        # whole-page failure.
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
        """Refuse the page unless ``(permno, dlycaldt)`` is unique on it.

        The table is unique on that pair, but the consequence of a duplicate
        slipping through is invisible: a downstream ``keep="last"`` dedup
        would collapse it arbitrarily, keeping one of two prices with nothing
        recording that a choice was made.
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
        """Refuse the page unless every row is a requested PERMNO in bounds.

        This checks and never filters: an unrequested PERMNO means the WHERE
        stopped doing what it says, and a row dated outside the page would
        land under a ``month=`` partition this page does not own, where the
        next run's deterministic overwrite would not reach it.
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
        """Build the ``AcquisitionConfig`` for a CRSP daily pull.

        This is the ``crsp_daily`` capability's ``config_factory``. The raw
        root ends at the vendor segment (``.../{subdir}/wrds``) and the
        watermarks live in the sibling ``.../{subdir}/_watermarks/wrds``,
        both under ``get_data_root() / "downloads" / "us_equity" / "1d"``.
        Symbols are stored as strings; ``bytes_per_row`` defaults to
        ``DEFAULT_BYTES_PER_ROW``. No credential goes into the config.

        Raises
        ------
        ValueError
            If ``kwargs["data_type"]`` is set to anything but
            ``"crsp_daily"``.

        Examples
        --------
        >>> cfg = WrdsCrspDailyAcquisition.build_config(
        ...     ("14593", "10107"), start_date="2020-08-01", end_date="2020-08-31"
        ... )
        >>> cfg.raw_data_dir_path
        '<data root>/downloads/us_equity/1d/wrds_crsp/wrds'
        >>> cfg.kwargs
        {'data_type': 'crsp_daily', 'bytes_per_row': 150}
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
        """Return the reference tier directory, ``.../{subdir}/_reference``.

        A sibling of the raw root, never inside it: the dataset's raw scan
        globs every parquet file below the raw root, and a reference table or
        its manifest in that tree would be picked up by the same scan. Derived
        from the config rather than from ``get_data_root()`` so a config
        pointed at a custom root keeps its reference tier beside its raw tier.

        Examples
        --------
        >>> WrdsCrspDailyAcquisition.reference_dir_for(cfg)
        PosixPath('<data root>/downloads/us_equity/1d/wrds_crsp/_reference')
        """
        return Path(config.raw_data_dir_path).parent / cls.REFERENCE_DIR_NAME


class CrspVolumeProbe:
    """Count the ``dsf_v2`` rows a pull would fetch, per calendar-year page.

    The counts feed the SQL volume guard, which prices a pull before it runs.
    The TAQ probe counts per trading day because a TAQ page is a day table; a
    CRSP page is a calendar year, so this counts per year, and a refusal from
    the guard therefore names a boundary a re-run can actually be given. One
    ``count(*)`` is issued per (year page, PERMNO batch), with pages from
    ``year_pages`` and the WHERE from ``CrspQueries.daily_where``, the same
    two the acquisition uses, so the rows priced are the rows that will move.
    ``daily_where`` refuses an empty batch, so a whole-table count is never
    issued.

    The guard's ``rows_by_day`` and ``trading_days`` fields are reused
    unchanged; against this dict they mean per-year buckets and a year count.
    Counts are not cached on disk: the pull re-counts each page anyway when
    ``verify_page_counts`` is on.

    Examples
    --------
    Needs a live ``WrdsSession``.

    >>> probe = CrspVolumeProbe(WrdsSession.shared(), batch_size=200)
    >>> rows_by_year = probe.count_rows_by_year(
    ...     ["14593", "10107"], "2018-06-01", "2020-03-31"
    ... )

    The keys are the page ends from ``year_pages``: ``2018-12-31``,
    ``2019-12-31`` and ``2020-03-31``.
    """

    #: Log progress every this many year pages.
    LOG_EVERY_PAGES = 5

    def __init__(
        self,
        session,
        batch_size: int = WrdsCrspDailyAcquisition.DEFAULT_BATCH_SIZE,
    ) -> None:
        """Bind the session and the PERMNOs-per-count batch size."""
        self.session = session
        self.batch_size = max(1, int(batch_size))

    def _batches(self, permnos: list[str]) -> list[list[str]]:
        """Chunk ``permnos`` into batches of ``batch_size`` in input order.

        The same chunking as ``Acquisition._batches``, restated because that
        is an instance method reading ``config.kwargs`` and this probe has no
        config: it is asked its question before any pull exists.
        """
        return [
            permnos[index : index + self.batch_size]
            for index in range(0, len(permnos), self.batch_size)
        ]

    def count_rows_by_year(
        self, permnos, start_date: str, end_date: str
    ) -> dict[str, int]:
        """Return ``{ISO page end: rows}`` over ``[start_date, end_date]``.

        Keys are year buckets (``2019-12-31``, ``2020-12-31``, and the
        window's own end for the last, partial year), each summed over the
        PERMNO batches. Entitlement is checked first, so an unsubscribed
        account raises ``WrdsEntitlementError`` before any count is issued.

        Raises
        ------
        ValueError
            If ``permnos`` is empty or contains a value that is
            not a digit string.

        Examples
        --------
        >>> counts = probe.count_rows_by_year(["14593"], "2020-01-01", "2020-12-31")
        """
        permnos = [str(permno) for permno in permnos]
        if not permnos:
            raise ValueError(
                "CrspVolumeProbe.count_rows_by_year: no PERMNOs; refusing to "
                "count the daily table without a PERMNO predicate."
            )
        for permno in permnos:
            if not permno.isdigit():
                raise ValueError(
                    f"CrspVolumeProbe: {permno!r} is not a PERMNO. The CRSP "
                    f"raw tier is keyed by PERMNO (a digit string), not by "
                    f"ticker; resolve the roster to PERMNOs first."
                )

        batches = self._batches(permnos)
        pages = year_pages(start_date, end_date)

        CrspQueries.assert_entitled(self.session, (CrspQueries.STOCK_SCHEMA,))

        counts: dict[str, int] = {}
        for position, (page_start, page_end) in enumerate(pages, start=1):
            counts[page_end.isoformat()] = sum(
                CrspQueries.count(
                    self.session,
                    CrspQueries.STOCK_SCHEMA,
                    CrspQueries.DAILY_TABLE,
                    CrspQueries.daily_where(batch, page_start, page_end),
                )
                for batch in batches
            )
            if position % self.LOG_EVERY_PAGES == 0 or position == len(pages):
                logger.info(
                    f"CRSP daily volume probe: {position}/{len(pages)} year "
                    f"page(s) counted ({len(batches)} batch(es) per page, "
                    f"{sum(counts.values()):,} rows so far)."
                )
        return counts
