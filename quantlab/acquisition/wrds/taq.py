"""WRDS NYSE TAQ millisecond NBBO acquisition and the shared WRDS session.

This module holds two things. ``WrdsSession`` is the one read-only PostgreSQL
connection every WRDS product in this package goes through: it reads the
username from ``WRDS_USERNAME``, leaves the password to libpq's ``~/.pgpass``,
connects to the pinned WRDS host once per process, and offers product-neutral
``schema_usable`` / ``fetch_rows`` / ``copy_csv`` helpers beside the TAQ-shaped
ones. ``WrdsTaqNbboAcquisition`` is the acquisition class for the
``taqm_{YYYY}.complete_nbbo_{YYYYMMDD}`` day tables: one page is one trading
day for one symbol batch, pulled with a bare ``COPY (SELECT ... WHERE ...)``
and stored one row per NBBO record, unfiltered and unsorted, with the server's
arrival order recorded as ``wrds_row_ord``. ``WrdsNbboVolumeProbe`` counts the
rows such a pull would move so the volume guard can price it first.

Sorting, de-duplication and resampling of the raw records happen in
``quantlab.dataset.nbbo``, not here. The ``wrds`` PyPI package is not used:
its connection object prompts interactively, and every extra connection can
push a Duo prompt to the account holder's phone. The vendor descriptor is
registered by the package entry module, not by this file.
"""

from __future__ import annotations

import io
import os
import stat
import tempfile
from datetime import date
from pathlib import Path

import polars as pl
import psycopg2
from loguru import logger
from psycopg2 import sql

from quantlab.base.acquisition import Acquisition
from quantlab.base.config import AcquisitionConfig
from quantlab.config import get_data_root
from quantlab.enums.data import TRADEABLE_TICKER_PATTERN

#: The environment variable the WRDS username is read from.
#:
#: Defined at module level rather than on ``WrdsSession`` because the session
#: class is replaced wholesale by a fake in tests; the credential-redaction
#: list ``CREDENTIAL_ENV_VARS`` is built from this constant so a stub that does
#: not define the name cannot disable redaction.
USERNAME_ENV = "WRDS_USERNAME"


class WrdsSessionError(RuntimeError):
    """The shared WRDS session cannot be used.

    Raised when the session could not be opened safely (a missing or unsafe
    password file, a routing environment variable), when the driver reported
    an error on it, or when it broke earlier in the same run. It is treated as
    a run-wide stop: ``WrdsTaqNbboAcquisition`` maps it to the ``"quota"``
    outcome, which halts dispatch and keeps every symbol out of the failure
    manifest, because a batch-by-batch retry would reconnect and each
    connection can push a Duo prompt.

    Example:
        >>> try:
        ...     WrdsSession.shared().trading_days(2024)
        ... except WrdsSessionError as exc:
        ...     print("session unusable:", exc)
    """


class WrdsEntitlementError(RuntimeError):
    """The account's subscription does not cover a requested schema.

    For TAQ this means a ``taqm_YYYY`` schema without ``USAGE``. Like
    ``WrdsSessionError`` it stops the whole run rather than one batch.

    Example:
        >>> try:
        ...     WrdsSession.shared().assert_entitled([2012, 2024])
        ... except WrdsEntitlementError as exc:
        ...     print(exc)
    """


def _pgpass_fields(line: str) -> list[str]:
    """Return the first four colon-separated fields of one pgpass line.

    Backslash escapes are resolved. Parsing stops at the fourth unescaped
    colon, so the fifth field, the password, is never read into any name.
    """
    fields: list[str] = []
    current: list[str] = []
    escaped = False
    for char in line:
        if escaped:
            current.append(char)
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == ":":
            fields.append("".join(current))
            current = []
            if len(fields) == 4:
                break
        else:
            current.append(char)
    return fields


class WrdsSession:
    """One lazily opened, read-only PostgreSQL connection to WRDS.

    Constructing a session makes no network call. The connection opens on the
    first query and is reused for the rest of the process; obtain it through
    ``shared()`` rather than constructing one per batch, since each connection
    can push a Duo prompt and the WRDS role allows only a handful at once. The
    host, port and database are class constants passed explicitly to
    ``psycopg2.connect`` and are never read from a config or the environment,
    so no config file can redirect the credentials elsewhere. The password is
    never handled by this code: libpq reads it from ``~/.pgpass`` (or
    ``$PGPASSFILE``), and the session checks that file for a matching entry
    before it ever tries to connect.

    The methods fall into three groups: pure SQL builders that need no
    connection (``table_identifier``, ``where_clause``, ``copy_query``,
    ``count_query``), product-neutral query helpers used by every WRDS
    provider (``schema_usable``, ``fetch_rows``, ``copy_csv``), and TAQ-shaped
    network methods built on those helpers.

    Example:
        Needs ``WRDS_USERNAME`` and a matching ``~/.pgpass`` line; the first
        query opens the connection and may push a Duo prompt.

        >>> session = WrdsSession.shared()
        >>> days = session.trading_days(2024)
        >>> WrdsSession.close_shared()
    """

    HOST = "wrds-pgdata.wharton.upenn.edu"
    PORT = 9737
    DBNAME = "wrds"
    SSLMODE = "require"
    APPLICATION_NAME = "quantlab-wrds-taq"
    CONNECT_TIMEOUT_SECONDS = 30

    #: Up to this many bytes of one COPY stay in memory before spilling to a
    #: temporary file.
    COPY_SPOOL_BYTES = 256 * 2**20

    #: libpq variables that could route the connection or its parameters
    #: around the pinned host: ``PGHOSTADDR`` overrides the address ``host``
    #: resolves to, and a service file can supply host, port and user. If any
    #: is set the session refuses to connect. ``PGHOST`` is harmless because
    #: ``host`` is always passed explicitly.
    REFUSED_ENV = ("PGHOSTADDR", "PGSERVICE", "PGSERVICEFILE")

    #: The pgpass line shape quoted in error messages. ``<password>`` is a
    #: placeholder; the real password is never read by this code.
    PGPASS_LINE_HINT = (
        "wrds-pgdata.wharton.upenn.edu:9737:wrds:$WRDS_USERNAME:<password>"
    )

    #: Process-level cache, username to session.
    _shared: dict[str, "WrdsSession"] = {}

    def __init__(self, username: str) -> None:
        """Record the username; no connection is opened here."""
        self.username = username
        self._conn = None
        # Set before `psycopg2.connect` is called, so a failed connect is
        # never retried by this object (each attempt can push Duo).
        self._connect_attempted = False
        self._broken = False

    @classmethod
    def shared(cls) -> "WrdsSession":
        """Return the process-wide session for the current ``WRDS_USERNAME``.

        The variable is read on every call, so a changed username gets its
        own session rather than a stale one.

        Raises:
            RuntimeError: If ``WRDS_USERNAME`` is unset or empty.

        Example:
            >>> session = WrdsSession.shared()
            >>> session is WrdsSession.shared()
            True
        """
        username = os.environ.get(USERNAME_ENV)
        if not username:
            raise RuntimeError(
                f"{USERNAME_ENV} environment variable must be set to your WRDS "
                f"username. The password is never read from config or from "
                f"this code: libpq reads it from ~/.pgpass (chmod 600), so "
                f"store it there before running a WRDS acquisition."
            )
        session = cls._shared.get(username)
        if session is None:
            session = cls(username)
            cls._shared[username] = session
        return session

    @classmethod
    def close_shared(cls) -> None:
        """Close and forget every shared session.

        Example:
            >>> WrdsSession.close_shared()
        """
        sessions = list(cls._shared.values())
        cls._shared.clear()
        for session in sessions:
            session.close()

    def close(self) -> None:
        """Close the underlying connection if one was opened.

        Example:
            >>> session.close()
        """
        conn, self._conn = self._conn, None
        if conn is not None:
            conn.close()

    # -- credential pre-checks -------------------------------------------------

    @staticmethod
    def _pgpass_path() -> Path:
        """Return ``$PGPASSFILE`` when set, else ``~/.pgpass``."""
        override = os.environ.get("PGPASSFILE")
        return Path(override) if override else Path.home() / ".pgpass"

    def _assert_refused_env_unset(self) -> None:
        """Raise ``WrdsSessionError`` if any ``REFUSED_ENV`` variable is set."""
        for name in self.REFUSED_ENV:
            if os.environ.get(name):
                raise WrdsSessionError(
                    f"{name} is set in the environment. libpq would use it to "
                    f"route the WRDS connection or its parameters away from "
                    f"the pinned host {self.HOST}:{self.PORT}; unset {name} "
                    f"before running a WRDS acquisition."
                )

    def _assert_pgpass_entry(self) -> None:
        """Fail before any connection attempt if libpq would find no password.

        Without this check libpq would connect without a password, the server
        would refuse it, and the operator would see a bare authentication
        error after a Duo push. The file must exist, be a regular file, not be
        group- or world-accessible (libpq ignores such a file), and contain a
        line whose host, port, database and user fields match, allowing ``*``
        wildcards and backslash escapes. Only the first four fields are
        parsed; no message quotes a line, and the username is written as
        ``$WRDS_USERNAME`` rather than its value.
        """
        path = self._pgpass_path()
        fix = (
            f"Create it with the single line `{self.PGPASS_LINE_HINT}` and run "
            f"`chmod 600 {path}`."
        )
        if not path.exists():
            raise WrdsSessionError(f"The password file {path} does not exist. {fix}")
        mode = path.stat().st_mode
        if not stat.S_ISREG(mode):
            raise WrdsSessionError(f"The password file {path} is not a regular file. {fix}")
        if mode & 0o077:
            raise WrdsSessionError(
                f"The password file {path} is group/world accessible (mode "
                f"{stat.S_IMODE(mode):o}); libpq ignores it. Run "
                f"`chmod 600 {path}`."
            )
        wanted = (self.HOST, str(self.PORT), self.DBNAME, self.username)
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.rstrip("\r\n")
                if not line or line.lstrip().startswith("#"):
                    continue
                fields = _pgpass_fields(line)
                if len(fields) == 4 and all(
                    field in ("*", want) for field, want in zip(fields, wanted)
                ):
                    return
        raise WrdsSessionError(
            f"The password file {path} has no line for "
            f"{self.HOST}:{self.PORT}:{self.DBNAME} and $WRDS_USERNAME. "
            f"Add `{self.PGPASS_LINE_HINT}` to it (mode 600)."
        )

    def _connection(self):
        """Open the read-only connection once and return it.

        The refused-environment and pgpass checks run first, and the attempt
        flag is set before ``psycopg2.connect`` is called, so neither a failed
        connect nor a broken session is ever followed by a second attempt
        from this object. No password argument is passed; libpq resolves it
        from the pgpass file. The call goes through the module attribute
        ``psycopg2.connect`` so a test can intercept every attempt.
        """
        if self._conn is not None and not self._broken:
            return self._conn
        if self._broken or self._connect_attempted:
            raise WrdsSessionError(
                "the WRDS session broke earlier in this run and is not reopened "
                "(every new connection can push Duo); re-run to resume from "
                "the recorded pages."
            )
        self._assert_refused_env_unset()
        self._assert_pgpass_entry()
        self._connect_attempted = True
        conn = psycopg2.connect(
            host=self.HOST,
            port=self.PORT,
            dbname=self.DBNAME,
            user=self.username,
            sslmode=self.SSLMODE,
            application_name=self.APPLICATION_NAME,
            connect_timeout=self.CONNECT_TIMEOUT_SECONDS,
        )
        conn.set_session(readonly=True, autocommit=True)
        self._conn = conn
        return conn

    def _query(self, work):
        """Run ``work(connection)``, converting driver errors.

        Any ``psycopg2.Error`` marks the session broken and is re-raised as a
        ``WrdsSessionError`` whose message carries the driver's text with the
        username replaced by ``$WRDS_USERNAME``. libpq errors never carry the
        password.
        """
        try:
            return work(self._connection())
        except psycopg2.Error as exc:
            self._broken = True
            detail = f"{type(exc).__name__}: {exc}".strip()
            if self.username:
                detail = detail.replace(self.username, "$WRDS_USERNAME")
            raise WrdsSessionError(
                f"the WRDS session failed ({detail}); it is not reopened in "
                f"this run -- re-run to resume from the recorded pages."
            ) from exc

    # -- pure SQL builders (no connection needed) ----------------------------

    @staticmethod
    def table_identifier(day: date) -> sql.Identifier:
        """Return the ``taqm_{YYYY}.complete_nbbo_{YYYYMMDD}`` identifier.

        Example:
            >>> from datetime import date
            >>> WrdsSession.table_identifier(date(2024, 1, 24))
            Identifier('taqm_2024', 'complete_nbbo_20240124')
        """
        return sql.Identifier(f"taqm_{day:%Y}", f"complete_nbbo_{day:%Y%m%d}")

    @staticmethod
    def where_clause(pairs) -> sql.Composed:
        """Build the WHERE clause selecting exact ``(sym_root, sym_suffix)`` pairs.

        The first conjunct, ``sym_root = ANY(roots)``, is what the server's
        chunk-group filters prune on; the second picks the exact share
        classes. Every value is a ``sql.Literal``, never interpolated text.

        Args:
            pairs: ``(root, suffix)`` tuples; a ``None`` or empty suffix means
                the plain root.

        Raises:
            ValueError: If ``pairs`` is empty. Without a ``sym_root``
                predicate the query would scan a whole day table, which this
                class never issues.

        Example:
            >>> clause = WrdsSession.where_clause([("AAPL", None), ("BRK", "B")])

            which renders as::

                sym_root = ANY(ARRAY['AAPL', 'BRK']) AND
                (sym_root, coalesce(sym_suffix, '')) IN (('AAPL', ''), ('BRK', 'B'))
        """
        pairs = [(str(root), str(suffix or "")) for root, suffix in pairs]
        if not pairs:
            # Without pairs there is no `sym_root` predicate, and a query over
            # a whole day table is a full scan of a multi-GB table.
            raise ValueError(
                "WrdsSession.where_clause: no (sym_root, sym_suffix) pairs; "
                "refusing to build a query over a whole complete_nbbo table."
            )
        roots = sorted({root for root, _ in pairs})
        pair_list = sql.SQL(", ").join(
            sql.SQL("({}, {})").format(sql.Literal(root), sql.Literal(suffix))
            for root, suffix in pairs
        )
        return sql.SQL(
            "sym_root = ANY({roots}) AND "
            "(sym_root, coalesce(sym_suffix, '')) IN ({pairs})"
        ).format(roots=sql.Literal(roots), pairs=pair_list)

    @classmethod
    def copy_query(cls, day: date, pairs, columns) -> sql.Composed:
        """Build the COPY statement for one day table and one symbol batch.

        The statement has no ORDER BY, GROUP BY or DISTINCT and no time
        predicate: the whole day is collected, and before 2018 the server's
        physical row order is the only tie-breaker between records that share
        a microsecond.

        Example:
            >>> query = WrdsSession.copy_query(
            ...     date(2024, 1, 24), [("AAPL", None)], ("date", "time_m", "best_bid")
            ... )

            which renders as::

                COPY (SELECT "date", "time_m", "best_bid"
                      FROM "taqm_2024"."complete_nbbo_20240124"
                      WHERE sym_root = ANY(ARRAY['AAPL']) AND
                            (sym_root, coalesce(sym_suffix, '')) IN (('AAPL', '')))
                TO STDOUT WITH (FORMAT csv, HEADER true)
        """
        return sql.SQL(
            "COPY (SELECT {columns} FROM {table} WHERE {where}) "
            "TO STDOUT WITH (FORMAT csv, HEADER true)"
        ).format(
            columns=sql.SQL(", ").join(sql.Identifier(name) for name in columns),
            table=cls.table_identifier(day),
            where=cls.where_clause(pairs),
        )

    @classmethod
    def count_query(cls, day: date, pairs) -> sql.Composed:
        """Build ``SELECT count(*)`` over the same WHERE as ``copy_query``.

        Sharing the WHERE means a count and the pull it checks cannot select
        different rows.

        Example:
            >>> query = WrdsSession.count_query(date(2024, 1, 24), [("AAPL", None)])

            which renders as::

                SELECT count(*) FROM "taqm_2024"."complete_nbbo_20240124"
                WHERE sym_root = ANY(ARRAY['AAPL']) AND
                      (sym_root, coalesce(sym_suffix, '')) IN (('AAPL', ''))
        """
        return sql.SQL("SELECT count(*) FROM {table} WHERE {where}").format(
            table=cls.table_identifier(day),
            where=cls.where_clause(pairs),
        )

    # -- generic query helpers -----------------------------------------------
    #
    # Product-neutral by design: one WRDS account serves several products,
    # and they all reach the server through this one session. A second
    # provider asks for a schema by name, hands over a `psycopg2.sql`
    # composable, and gets rows or CSV bytes back. All three go through
    # `self._query`, which applies the username scrub and the broken-session
    # rule; a helper reaching `self._connection()` directly would leave a
    # driver-errored session reusable and reconnect (and push Duo) next call.

    def schema_usable(self, schema: str) -> bool:
        """Return whether this role has ``USAGE`` on ``schema``.

        Asked through ``pg_namespace`` rather than by name, because the
        by-name form of ``has_schema_privilege`` raises for a schema that does
        not exist and a driver error would break the session. A missing
        schema returns no row and reads as ``False``, the same answer as "not
        subscribed". The schema travels as a query parameter.

        Example:
            >>> usable = session.schema_usable("crsp_a_stock")
        """

        def work(conn):
            """Fetch the privilege row for the schema.

            Example:
                >>> row = self._query(work)
            """
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT has_schema_privilege(oid, 'USAGE') "
                    "FROM pg_namespace WHERE nspname = %s",
                    (str(schema),),
                )
                return cursor.fetchone()

        row = self._query(work)
        return bool(row and row[0])

    def fetch_rows(self, query) -> list[tuple]:
        """Execute ``query`` and return every row.

        Meant for catalogue and metadata reads that fit in memory. A data pull
        uses ``copy_csv`` instead, which streams a full day of records
        without materialising Python tuples.

        Args:
            query: A ``psycopg2.sql`` composable or plain SQL text.

        Example:
            >>> from psycopg2 import sql
            >>> rows = session.fetch_rows(sql.SQL("SELECT 1"))
        """

        def work(conn):
            """Execute the query and fetch all rows.

            Example:
                >>> rows = self._query(work)
            """
            with conn.cursor() as cursor:
                cursor.execute(query)
                return cursor.fetchall()

        return self._query(work)

    def copy_csv(self, query) -> bytes:
        """Run a ``COPY ... TO STDOUT`` statement and return the CSV bytes.

        The composed statement goes to ``copy_expert`` as-is, so no SQL text
        is assembled outside ``psycopg2.sql``. Up to ``COPY_SPOOL_BYTES`` stay
        in memory before the buffer spills to a temporary file.

        Example:
            >>> raw = session.copy_csv(
            ...     WrdsSession.copy_query(day, [("AAPL", None)], columns)
            ... )
        """

        def work(conn):
            """Stream the COPY output through a spooled buffer.

            Example:
                >>> raw = self._query(work)
            """
            with tempfile.SpooledTemporaryFile(
                max_size=self.COPY_SPOOL_BYTES
            ) as buffer, conn.cursor() as cursor:
                cursor.copy_expert(query, buffer)
                buffer.seek(0)
                return buffer.read()

        return self._query(work)

    # -- network methods -------------------------------------------------------

    def has_schema_usage(self, year: int) -> bool:
        """Return whether this role may read ``taqm_{year}``.

        A TAQ-shaped name over ``schema_usable``; only the schema naming
        differs.

        Example:
            >>> entitled = session.has_schema_usage(2024)
        """
        return self.schema_usable(f"taqm_{int(year)}")

    def assert_entitled(self, years) -> None:
        """Raise ``WrdsEntitlementError`` naming every unreadable ``taqm_YYYY``.

        Run before the first data query of a pull or a probe, so an
        unentitled year stops the run with zero COPY calls instead of failing
        every batch of every day.

        Example:
            >>> session.assert_entitled(range(2020, 2025))
        """
        missing = [
            f"taqm_{int(year)}"
            for year in sorted({int(year) for year in years})
            if not self.has_schema_usage(year)
        ]
        if missing:
            raise WrdsEntitlementError(
                f"The WRDS account has no access to {', '.join(missing)}: its "
                f"WRDS NYSE TAQ millisecond subscription does not cover "
                f"{'that year' if len(missing) == 1 else 'those years'}. "
                f"Narrow the window to entitled years or extend the "
                f"subscription; nothing was downloaded."
            )

    def count_rows(self, day: date, pairs) -> int:
        """Return how many rows ``copy_query(day, pairs, ...)`` would return.

        Example:
            >>> session.count_rows(date(2024, 1, 24), [("AAPL", None)])
        """
        query = self.count_query(day, pairs)

        def work(conn):
            """Execute the count and fetch its single row.

            Example:
                >>> row = self._query(work)
            """
            with conn.cursor() as cursor:
                cursor.execute(query)
                return cursor.fetchone()

        row = self._query(work)
        return int(row[0])

    def trading_days(self, year: int) -> list[date]:
        """Return every day with a ``complete_nbbo`` table in ``taqm_{year}``.

        Listed ascending from ``information_schema`` rather than probed per
        calendar day, so a missing table cannot be mistaken for a holiday.

        Example:
            >>> days = session.trading_days(2024)
        """

        def work(conn):
            """List the day-table names in the year's schema.

            Example:
                >>> names = self._query(work)
            """
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema = %s "
                    "AND table_name ~ '^complete_nbbo_[0-9]{8}$'",
                    (f"taqm_{year}",),
                )
                return [row[0] for row in cursor.fetchall()]

        names = self._query(work)
        days = [
            date(int(name[-8:-4]), int(name[-4:-2]), int(name[-2:]))
            for name in names
        ]
        return sorted(days)

    def table_columns(self, day: date) -> tuple[str, ...]:
        """Return the day table's column names in server order.

        Example:
            >>> columns = session.table_columns(date(2024, 1, 24))
        """

        def work(conn):
            """Read the column names and positions from the catalogue.

            Example:
                >>> rows = self._query(work)
            """
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT column_name, ordinal_position "
                    "FROM information_schema.columns "
                    "WHERE table_schema = %s AND table_name = %s",
                    (f"taqm_{day:%Y}", f"complete_nbbo_{day:%Y%m%d}"),
                )
                return cursor.fetchall()

        rows = self._query(work)
        return tuple(name for name, _ in sorted(rows, key=lambda row: row[1]))

    def copy_nbbo_csv(self, day: date, pairs, columns) -> bytes:
        """Run ``copy_query`` for one day and batch and return the CSV bytes.

        The header line is included. The statement is exactly the one
        ``copy_query`` builds, with no ordering or time predicate added.

        Example:
            >>> raw = session.copy_nbbo_csv(
            ...     date(2024, 1, 24), [("AAPL", None)], session.table_columns(day)
            ... )
        """
        return self.copy_csv(self.copy_query(day, pairs, columns))


def trading_days_between(session, start: date, end: date) -> list[date]:
    """Return the trading days in ``[start, end]``, ascending.

    A trading day is a day with a ``complete_nbbo`` table, listed per year
    through ``session.trading_days``. The acquisition and the volume probe
    both use this function so they walk exactly the same days.

    Example:
        Needs a live ``WrdsSession``.

        >>> from datetime import date
        >>> days = trading_days_between(session, date(2024, 1, 1), date(2024, 3, 31))
    """
    days: set[date] = set()
    for year in range(start.year, end.year + 1):
        days.update(session.trading_days(year))
    return sorted(day for day in days if start <= day <= end)


class WrdsTaqNbboAcquisition(Acquisition):
    """Acquisition of WRDS TAQ ``complete_nbbo`` day tables.

    One page is one trading day's table for one symbol batch: ``_fetch_page``
    reads the day named by the page token (the window's first trading day
    when there is none) and returns the next trading day's ISO date as the
    next token. The shared ``Acquisition`` base owns the batch loop, shard
    writes, the page ledger and resume.

    The raw tier lands under ``.../wrds/data_type=nbbo/date=/symbol=/`` with
    one row per NBBO record for the full session day, unfiltered. Each row's
    arrival position within its query is stored as ``wrds_row_ord`` before
    anything else touches the frame, because before 2018 the tables carry no
    sequence number and the server's order is the only tie-breaker. TAQ's
    ``time_m`` is US/Eastern wall clock; ``timestamp`` is the same instant as
    naive UTC nanoseconds, and the ``date=`` hive key is the Eastern session
    date.

    The config must say ``frequency="tick"`` and ``kwargs["data_type"] =
    "nbbo"``; ``max_workers`` other than 1 is refused because every
    connection can push a Duo prompt. Symbols use dot notation for share
    classes (``BRK.B``); a hyphenated form is refused.

    Example:
        Needs ``WRDS_USERNAME`` and a ``~/.pgpass`` entry; the first query
        opens the connection and may push a Duo prompt.

        >>> cfg = WrdsTaqNbboAcquisition.build_config(
        ...     ("AAPL", "MSFT"), start_date="2024-01-24", end_date="2024-01-25"
        ... )
        >>> acq = WrdsTaqNbboAcquisition(cfg).download()
        >>> report = acq.coverage_report()

        Shards land under ``.../wrds_taq/wrds/data_type=nbbo/date=2024-01-24/
        symbol=AAPL/`` and the watermarks under ``.../wrds_taq/_watermarks/wrds/``.
    """

    VENDOR = "wrds"

    #: TAQ ``time_m`` is US/Eastern wall clock; the ``date=`` hive key is the
    #: Eastern session date.
    SESSION_TIME_ZONE = "America/New_York"

    #: The only data type this vendor serves under ``frequency="tick"``.
    TICK_DATA_TYPES = ("nbbo",)

    #: Symbols per day-table query. A working value, overridable through
    #: ``kwargs["batch_size"]``.
    DEFAULT_BATCH_SIZE = 25

    #: One shared connection, so one worker. Any other value is refused in
    #: ``__init__``.
    DEFAULT_MAX_WORKERS = 1

    #: Count every page with the COPY's own WHERE before pulling it, and fail
    #: the page on a mismatch. Overridable through
    #: ``kwargs["verify_page_counts"]``; the count is well under a second per
    #: (day, batch) on the server's columnar chunk filters.
    DEFAULT_VERIFY_PAGE_COUNTS = True

    CREDENTIAL_ENV_VARS = (USERNAME_ENV,)
    REDACTION = "<WRDS CREDENTIAL REDACTED>"

    SCHEMA_PATTERN = "taqm_{year}"
    TABLE_PATTERN = "complete_nbbo_{ymd}"

    #: ``BRK.B`` is root ``BRK`` with suffix ``B``: the constituent universes'
    #: notation.
    SUFFIX_DELIMITER = "."

    #: The raw tier lives under ``downloads/us_equity/tick/{DEFAULT_SUBDIR}/wrds``.
    DEFAULT_SUBDIR = "wrds_taq"

    #: The TAQ columns this class reads. ``time_m_nano`` exists only from
    #: 2018-01-02 and is optional; every other column is required.
    TAQ_COLUMNS = (
        "date",
        "time_m",
        "time_m_nano",
        "sym_root",
        "sym_suffix",
        "qu_cond",
        "natbbo_ind",
        "qu_source",
        "nbbo_qu_cond",
        "best_bid",
        "best_bidsizeshares",
        "best_ask",
        "best_asksizeshares",
    )
    OPTIONAL_TAQ_COLUMNS = ("time_m_nano",)

    #: The shard column projection and order. TAQ's ``date`` is stored as
    #: ``taq_date`` because the shard writer derives its own ``date`` hive
    #: key and would otherwise overwrite and drop the raw column.
    RAW_COLUMNS = (
        "timestamp",
        "symbol",
        "vendor",
        "taq_date",
        "time_m",
        "time_m_nano",
        "sym_root",
        "sym_suffix",
        "qu_cond",
        "natbbo_ind",
        "qu_source",
        "nbbo_qu_cond",
        "best_bid",
        "best_bidsizeshares",
        "best_ask",
        "best_asksizeshares",
        "wrds_row_ord",
    )

    RAW_SCHEMA = {
        "timestamp": pl.Datetime("ns"),
        "symbol": pl.String,
        "vendor": pl.String,
        "taq_date": pl.Date,
        "time_m": pl.Time,
        "time_m_nano": pl.Int16,
        "sym_root": pl.String,
        "sym_suffix": pl.String,
        "qu_cond": pl.String,
        "natbbo_ind": pl.String,
        "qu_source": pl.String,
        "nbbo_qu_cond": pl.String,
        "best_bid": pl.Float64,
        "best_bidsizeshares": pl.Int64,
        "best_ask": pl.Float64,
        "best_asksizeshares": pl.Int64,
        "wrds_row_ord": pl.Int64,
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
        # Resolved through the module global at call time, so a test's patch
        # of `WrdsSession` takes effect.
        self._session = WrdsSession.shared()
        self._trading_days_cache: dict[tuple[str, str], list[date]] = {}

    @property
    def _data_type(self) -> str:
        """Return ``"nbbo"``, the only data type, or raise on a bad config.

        There is deliberately no default: the value names the ``data_type=``
        hive key and the watermark namespace.
        """
        frequency = self.config.frequency
        data_type = self._knob("data_type", None)
        if frequency != "tick" or data_type not in self.TICK_DATA_TYPES:
            raise ValueError(
                f"{self.class_name}: needs frequency 'tick' with "
                f"kwargs['data_type'] set to one of "
                f"{sorted(self.TICK_DATA_TYPES)}; got frequency {frequency!r} "
                f"and data_type {data_type!r}. There is deliberately NO "
                f"default -- the data type names the raw tier's `data_type=` "
                f"hive key and the watermark namespace."
            )
        return data_type

    # -- failure policy ----------------------------------------------------------

    #: Exceptions that mean the one session, or the account, is unusable.
    GLOBAL_STOP_ERRORS = (
        WrdsSessionError,
        WrdsEntitlementError,
        psycopg2.OperationalError,
        psycopg2.InterfaceError,
    )

    def _classify_error(self, exc: BaseException) -> str:
        """Map session and entitlement errors to a run-wide stop.

        For this vendor ``"quota"`` means "stop the whole run", not an
        allocation: a dead session or a missing entitlement is never one
        symbol's fault, so it stays out of the failure manifest, and a
        batch-by-batch retry would reconnect and push Duo each time.
        Everything else (a malformed page, an unrequested symbol, a count
        mismatch) is a per-batch failure retried on the next run.
        """
        if isinstance(exc, self.GLOBAL_STOP_ERRORS):
            return "quota"
        return super()._classify_error(exc)

    # -- symbols ---------------------------------------------------------------

    @classmethod
    def symbol_to_pair(cls, symbol: str) -> tuple[str, str | None]:
        """Split a dotted symbol into its TAQ ``(sym_root, sym_suffix)`` pair.

        A hyphenated symbol such as ``BRK-B`` is refused rather than guessed
        at: ``TRADEABLE_TICKER_PATTERN`` admits both delimiters, and querying
        ``sym_root = 'BRK-B'`` would silently return nothing. More than one
        dot, or an empty root or suffix, is refused too.

        Example:
            >>> WrdsTaqNbboAcquisition.symbol_to_pair("BRK.B")
            ('BRK', 'B')
            >>> WrdsTaqNbboAcquisition.symbol_to_pair("AAPL")
            ('AAPL', None)
        """
        text = str(symbol)
        if "-" in text:
            raise ValueError(
                f"WRDS/TAQ symbol {text!r} uses a hyphen. WRDS TAQ queries use "
                f"the constituent universes' dot notation (e.g. BRK.B for "
                f"root BRK, suffix B); pass the dotted form."
            )
        root, delimiter, suffix = text.partition(cls.SUFFIX_DELIMITER)
        if not root or cls.SUFFIX_DELIMITER in suffix or (delimiter and not suffix):
            raise ValueError(
                f"WRDS/TAQ symbol {text!r} is not ROOT or ROOT.SUFFIX in dot "
                f"notation (exactly one dot, both parts non-empty)."
            )
        return root, (suffix or None)

    @classmethod
    def pair_to_symbol(cls, root: str, suffix: str | None) -> str:
        """Join a TAQ ``(sym_root, sym_suffix)`` pair back into a dotted symbol.

        Example:
            >>> WrdsTaqNbboAcquisition.pair_to_symbol("BRK", "B")
            'BRK.B'
            >>> WrdsTaqNbboAcquisition.pair_to_symbol("AAPL", None)
            'AAPL'
        """
        return str(root) if not suffix else f"{root}{cls.SUFFIX_DELIMITER}{suffix}"

    # -- trading days ------------------------------------------------------------

    @staticmethod
    def _as_date(value) -> date:
        """Coerce an ISO string or date-like value to a ``date``."""
        return date.fromisoformat(str(value)[:10])

    def _trading_days(self, start_date, end_date) -> list[date]:
        """Return the trading days in the window, cached per window."""
        start = self._as_date(start_date)
        end = self._as_date(end_date)
        key = (start.isoformat(), end.isoformat())
        cached = self._trading_days_cache.get(key)
        if cached is not None:
            return cached
        result = trading_days_between(self._session, start, end)
        self._trading_days_cache[key] = result
        return result

    # -- entitlement preflight -------------------------------------------------

    def _run(self, symbols: list[str] | None, from_watermark: bool):
        """Check the TAQ entitlement for every year of the window, then run.

        The check happens before the base runner dispatches a single batch, so
        an unentitled year raises ``WrdsEntitlementError`` out of
        ``download()`` or ``refresh()`` with zero COPY calls and no failure
        manifest write. The window checked is the config's; a refresh's
        per-symbol start is never earlier than it.
        """
        start = self._as_date(self.config.start_date)
        end = self._as_date(self.config.end_date)
        self._session.assert_entitled(range(start.year, end.year + 1))
        return super()._run(symbols, from_watermark)

    # -- one page = one day-table query --------------------------------------

    def _empty_page(self) -> pl.DataFrame:
        """Return an empty frame with the raw schema and column order."""
        return pl.DataFrame(schema=self.RAW_SCHEMA).select(self.RAW_COLUMNS)

    def _fetch_page(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str,
        page_token: str | None = None,
    ) -> tuple[pl.DataFrame, str | None]:
        """Fetch one trading day's ``complete_nbbo`` rows for one symbol batch.

        Returns ``(frame, next_token)``, where ``next_token`` is the next
        trading day's ISO date or ``None`` after the window's last day. No
        record is sorted, de-duplicated or filtered; ``wrds_row_ord`` is
        assigned from arrival order before any other operation. The page is
        counted with the same WHERE before the COPY and refused on a
        mismatch, and refused again if it contains an unrequested share class
        or a row dated off the table's day.
        """
        symbols = self._validate_symbols(symbols)
        # Before any query: a hyphenated or malformed symbol is refused here.
        pairs = [self.symbol_to_pair(symbol) for symbol in symbols]
        days = self._trading_days(start_date, end_date)
        if not days:
            return self._empty_page(), None

        day = date.fromisoformat(page_token) if page_token else days[0]
        if day not in days:
            raise ValueError(
                f"{self.class_name}: page token {page_token!r} names no trading "
                f"day inside [{start_date}, {end_date}]. A token is the ISO "
                f"date of the next day table to read; refusing rather than "
                f"reading a day outside the requested window."
            )
        position = days.index(day)
        next_token = (
            days[position + 1].isoformat() if position + 1 < len(days) else None
        )

        server_columns = set(self._session.table_columns(day))
        table = f"taqm_{day:%Y}.{self.TABLE_PATTERN.format(ymd=f'{day:%Y%m%d}')}"
        # TAQ_COLUMNS order, not server order, so the SELECT and the frame are
        # identical for every day of every era.
        columns = tuple(name for name in self.TAQ_COLUMNS if name in server_columns)
        missing = [
            name
            for name in self.TAQ_COLUMNS
            if name not in server_columns and name not in self.OPTIONAL_TAQ_COLUMNS
        ]
        if missing:
            # A layout drift fails loudly; it is never null-filled.
            raise ValueError(
                f"{self.class_name}: {table} reports no {missing} column(s); "
                f"the table layout no longer matches the one this class was "
                f"verified against (D-18). Columns seen: "
                f"{sorted(server_columns)}."
            )

        # Completeness check: the same WHERE, counted before the COPY. A page
        # that parses to a different number of rows fails as a per-batch
        # failure and is re-fetched by the next run.
        expected_rows = (
            self._session.count_rows(day, pairs)
            if self._knob("verify_page_counts", self.DEFAULT_VERIFY_PAGE_COUNTS)
            else None
        )

        raw = self._session.copy_nbbo_csv(day, pairs, columns)

        frame = pl.read_csv(io.BytesIO(raw), infer_schema=False)
        # First, before anything can reorder the rows.
        frame = frame.with_columns(
            pl.int_range(pl.len(), dtype=pl.Int64).alias("wrds_row_ord")
        )
        if expected_rows is not None and frame.height != expected_rows:
            raise ValueError(
                f"{self.class_name}: {table} (trading day {day.isoformat()}) "
                f"COPY returned {frame.height} row(s) but count(*) with the "
                f"same WHERE reported {expected_rows}; the page is incomplete "
                f"and is not recorded, so the next run re-fetches this day."
            )
        if frame.height == 0:
            return self._empty_page(), next_token
        if tuple(frame.columns[:-1]) != columns:
            raise ValueError(
                f"{self.class_name}: {table} COPY returned columns "
                f"{frame.columns[:-1]}, not the requested {list(columns)}."
            )

        if "time_m_nano" not in frame.columns:
            frame = frame.with_columns(
                pl.lit(None, dtype=pl.String).alias("time_m_nano")
            )
        frame = frame.rename({"date": "taq_date"})
        frame = frame.with_columns(
            pl.col("taq_date").str.to_date("%Y-%m-%d"),
            # `%.f` accepts values with and without a fractional part.
            pl.col("time_m").str.to_time("%H:%M:%S%.f"),
            pl.col("time_m_nano").cast(pl.Int16),
            pl.col("best_bid").cast(pl.Float64),
            pl.col("best_ask").cast(pl.Float64),
            pl.col("best_bidsizeshares").cast(pl.Int64),
            pl.col("best_asksizeshares").cast(pl.Int64),
        )
        frame = frame.with_columns(
            (
                pl.col("taq_date").dt.combine(pl.col("time_m"), time_unit="ns")
                + pl.duration(
                    nanoseconds=pl.col("time_m_nano").fill_null(0).cast(pl.Int64)
                )
            )
            .dt.replace_time_zone(
                self.SESSION_TIME_ZONE, ambiguous="raise", non_existent="raise"
            )
            .dt.convert_time_zone("UTC")
            .dt.replace_time_zone(None)
            .alias("timestamp"),
            pl.when(
                pl.col("sym_suffix").is_null() | (pl.col("sym_suffix") == "")
            )
            .then(pl.col("sym_root"))
            .otherwise(
                pl.col("sym_root")
                + pl.lit(self.SUFFIX_DELIMITER)
                + pl.col("sym_suffix")
            )
            .alias("symbol"),
            pl.lit(self.VENDOR).alias("vendor"),
        )
        self._assert_page_belongs(frame, day, pairs, table)
        frame = frame.cast(self.RAW_SCHEMA)
        return frame.select(self.RAW_COLUMNS), next_token

    def _assert_page_belongs(
        self, frame: pl.DataFrame, day: date, pairs, table: str
    ) -> None:
        """Refuse the page unless every row is for the day and a requested pair.

        This checks and never filters: dropping an unrequested row would hide
        a WHERE clause that stopped doing what it says, and a row dated off
        the table's day would land under the wrong ``date=`` partition. The
        Eastern session date of the reconstructed ``timestamp`` is checked as
        well as the raw ``date`` field, so the hive key derived by the shard
        writer cannot disagree with the table the row came from.
        """
        wanted = {(root, suffix or "") for root, suffix in pairs}
        seen = frame.select(
            pl.col("sym_root"), pl.col("sym_suffix").fill_null("")
        ).unique()
        strangers = sorted(
            f"{root}/{suffix}" if suffix else root
            for root, suffix in seen.iter_rows()
            if (root, suffix) not in wanted
        )
        if strangers:
            raise ValueError(
                f"{self.class_name}: {table} returned rows for {strangers}, "
                f"which were not requested (requested pairs: "
                f"{sorted(wanted)}). Refusing the page rather than filing "
                f"another share class under a requested symbol (D-15)."
            )
        off_day = frame.filter(
            (pl.col("taq_date") != day)
            | (self._session_date(pl.col("timestamp")) != day)
        )
        if off_day.height:
            dates = sorted({str(value) for value in off_day["taq_date"].to_list()})
            raise ValueError(
                f"{self.class_name}: {table} (trading day {day.isoformat()}) "
                f"returned {off_day.height} row(s) dated {dates}; a day table "
                f"must only hold its own day."
            )

    # -- config ----------------------------------------------------------------

    @classmethod
    def build_config(
        cls,
        symbols,
        start_date: str | None = None,
        end_date: str | None = None,
        kwargs: dict | None = None,
        subdir: str = DEFAULT_SUBDIR,
    ) -> AcquisitionConfig:
        """Build the ``AcquisitionConfig`` for a WRDS NBBO pull.

        This is the capability's ``config_factory``. The raw root ends at the
        vendor segment (``.../{subdir}/wrds``) and the watermarks live in the
        sibling ``.../{subdir}/_watermarks/wrds``, both under
        ``get_data_root() / "downloads" / "us_equity" / "tick"``. No
        credential goes into the config.

        Raises:
            ValueError: If ``kwargs["data_type"]`` is set to anything but
                ``"nbbo"``.

        Example:
            >>> cfg = WrdsTaqNbboAcquisition.build_config(
            ...     ("AAPL", "MSFT"), start_date="2024-01-24", end_date="2024-01-25"
            ... )
            >>> cfg.raw_data_dir_path
            '<data root>/downloads/us_equity/tick/wrds_taq/wrds'
            >>> cfg.kwargs
            {'data_type': 'nbbo'}
        """
        merged = dict(kwargs or {})
        data_type = merged.get("data_type", "nbbo")
        if data_type != "nbbo":
            raise ValueError(
                f"{cls.__name__}.build_config: kwargs['data_type']="
                f"{data_type!r} conflicts with this source, which serves only "
                f"'nbbo'."
            )
        merged["data_type"] = "nbbo"
        downloads = get_data_root() / "downloads" / "us_equity" / "tick" / subdir
        return AcquisitionConfig(
            market="us_equity",
            frequency="tick",
            vendor=cls.VENDOR,
            raw_data_dir_path=str(downloads / cls.VENDOR),
            watermark_path=str(downloads / "_watermarks" / cls.VENDOR),
            symbols=tuple(symbols),
            start_date=start_date,
            end_date=end_date,
            kwargs=merged,
        )


class WrdsNbboVolumeProbe:
    """Count the ``complete_nbbo`` rows a pull would fetch, per trading day.

    The counts feed the SQL volume guard, which prices a pull before it runs.
    One ``count(*)`` is issued per (trading day, symbol batch), with batches
    chunked exactly like ``Acquisition._batches`` and the WHERE built by the
    same ``WrdsSession.where_clause`` the COPY uses, so the rows priced are
    the rows the pull will move. Each count is fast on the server's columnar
    chunk filters; a whole-table count is never issued because
    ``where_clause`` refuses an empty batch. Counts are not cached on disk:
    the pull re-counts each page anyway when ``verify_page_counts`` is on.

    Example:
        Needs a live ``WrdsSession``.

        >>> probe = WrdsNbboVolumeProbe(WrdsSession.shared(), batch_size=25)
        >>> rows_by_day = probe.count_rows_by_day(
        ...     ["AAPL", "MSFT"], "2024-01-24", "2024-01-25"
        ... )
    """

    #: Log progress every this many trading days.
    LOG_EVERY_DAYS = 20

    def __init__(
        self,
        session,
        batch_size: int = WrdsTaqNbboAcquisition.DEFAULT_BATCH_SIZE,
    ) -> None:
        """Bind the session and the symbols-per-query batch size."""
        self.session = session
        self.batch_size = max(1, int(batch_size))

    def _batches(self, symbols: list[str]) -> list[list[str]]:
        """Chunk ``symbols`` into batches of ``batch_size`` in input order."""
        return [
            symbols[index : index + self.batch_size]
            for index in range(0, len(symbols), self.batch_size)
        ]

    def count_rows_by_day(
        self, symbols, start_date: str, end_date: str
    ) -> dict[str, int]:
        """Return ``{ISO trading day: rows}`` over ``[start_date, end_date]``.

        Rows are summed over the symbol batches. Entitlement is checked
        first, so an unentitled year raises ``WrdsEntitlementError`` before
        any count is issued.

        Raises:
            ValueError: If ``symbols`` is empty or contains a value that is
                not a tradeable ticker.

        Example:
            >>> counts = probe.count_rows_by_day(["AAPL"], "2024-01-24", "2024-01-24")
        """
        symbols = [str(symbol) for symbol in symbols]
        if not symbols:
            raise ValueError(
                "WrdsNbboVolumeProbe.count_rows_by_day: no symbols; refusing to "
                "count a table without a sym_root predicate."
            )
        for symbol in symbols:
            if not TRADEABLE_TICKER_PATTERN.fullmatch(symbol):
                raise ValueError(
                    f"WrdsNbboVolumeProbe: {symbol!r} is not a tradeable ticker."
                )
        batches = [
            [WrdsTaqNbboAcquisition.symbol_to_pair(symbol) for symbol in batch]
            for batch in self._batches(symbols)
        ]
        start = date.fromisoformat(str(start_date)[:10])
        end = date.fromisoformat(str(end_date)[:10])
        self.session.assert_entitled(range(start.year, end.year + 1))

        days = trading_days_between(self.session, start, end)
        counts: dict[str, int] = {}
        for position, day in enumerate(days, start=1):
            counts[day.isoformat()] = sum(
                self.session.count_rows(day, pairs) for pairs in batches
            )
            if position % self.LOG_EVERY_DAYS == 0 or position == len(days):
                logger.info(
                    f"WRDS NBBO volume probe: {position}/{len(days)} trading "
                    f"day(s) counted ({len(batches)} batch(es) per day, "
                    f"{sum(counts.values()):,} rows so far)."
                )
        return counts
