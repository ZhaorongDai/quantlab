"""WRDS NYSE TAQ millisecond NBBO acquisition (phase 03.9).

Live-verified facts this module is built on (`03.9-LIVE-CHECK-{1,2}.json`):

- **Table (D-18).** One base table per trading day,
  `taqm_{YYYY}.complete_nbbo_{YYYYMMDD}`. The `taqmsec.*` names are views of
  the same data. Columns, identical 2012/2016/2024 except the nanosecond part:
  `date, time_m, [time_m_nano], sym_root, sym_suffix, qu_cond, natbbo_ind,
  qu_source, nbbo_qu_cond, best_bid, best_bidsizeshares, best_ask,
  best_asksizeshares`. `time_m_nano` exists only from 2018-01-02. There is NO
  `qu_seqnum`.
- **Order (D-19).** Before 2018 about 5% of records share a microsecond with a
  differently-valued record, and no sequence field exists, so the physical
  order the server hands rows over in is the only tie-breaker. The pull is a
  bare `COPY (SELECT ... WHERE ...) TO STDOUT` -- no ORDER BY, no GROUP BY, no
  DISTINCT, no time predicate (D-03/D-04) -- and each row's arrival ordinal
  within its (day, symbol-batch) query is recorded as `wrds_row_ord` BEFORE any
  other frame operation. Raw is one row per record, the full day, unfiltered
  (D-02/D-05); sorting, de-duplication and resampling happen only in
  `dataset/nbbo_resample.py`.
- **Connection (D-20).** Not through the `wrds` package (its `Connection`
  prompts interactively and its `raw_sql` breaks under pandas 3) -- this
  module does not import it at all. `psycopg2` connects to the pinned WRDS
  host with `sslmode=require`; the username comes from `WRDS_USERNAME` and the
  password never passes through this code (libpq reads `~/.pgpass`). One
  connection per process run, read-only, because every connection can push a
  Duo prompt to the user's phone and the role's connection limit is 7.

**This module REGISTERS NOTHING** (03.10 D-12). The `wrds` descriptor lives in
`quantlab/acquisition/wrds.py`, a neutral module that imports the WRDS provider
modules, and this one imports nothing from the registry. One WRDS account
serves several products, so a registration placed inside a provider would have
to name the OTHER provider's classes -- and then importing either provider
first would cycle through the registry. `WrdsSession` accordingly offers
GENERIC `schema_usable` / `fetch_rows` / `copy_csv`, which the other providers
build on; the TAQ-named methods here are thin statements of "which SQL",
delegating "how to run it" to those three.
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

#: The one environment variable the WRDS username is read from.
#:
#: MODULE-LEVEL on purpose, the alpaca.py rule: `WrdsSession` is a patch target
#: (`tests/conftest.py:mock_wrds_session` replaces it wholesale), so a
#: redaction control reached THROUGH it could be silently disabled by a stub
#: that does not define the name. `CREDENTIAL_ENV_VARS` below is built from
#: this constant, never from the session class.
USERNAME_ENV = "WRDS_USERNAME"


class WrdsSessionError(RuntimeError):
    """The single WRDS session cannot be used: it could not be opened safely,
    the driver reported an error on it, or it broke earlier in this run.

    A GLOBAL condition (D-21): `WrdsTaqNbboAcquisition._classify_error` maps it
    to "quota", which stops dispatch for the whole run and keeps every symbol
    out of the failure manifest. It is never retried batch by batch, because a
    retry would mean a reconnect and every connection can push Duo (D-20).
    """


class WrdsEntitlementError(RuntimeError):
    """The WRDS account's TAQ subscription does not cover a requested year
    (`taqm_YYYY` without USAGE). Global, like `WrdsSessionError` (D-21)."""


def _pgpass_fields(line: str) -> list[str]:
    """The first FOUR `:`-separated fields of one pgpass line, `\\`-escapes
    resolved.

    Parsing stops at the fourth unescaped colon: the fifth field -- the
    password -- is never collected into any name (D-07, T-03.9-10).
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
    """One lazily-opened, read-only PostgreSQL connection to WRDS.

    Constructing it makes NO network call; the connection opens on the first
    query and is then reused for the life of the process run (D-20). Obtain it
    through `shared()`, never by constructing one per batch or per worker --
    each connection can push Duo.

    The host, port and database are CLASS CONSTANTS passed explicitly to
    `psycopg2.connect`, never read from config or from the environment: a
    config-overridable host would turn a config file into a way to send the
    user's credentials somewhere else (T-03.9-04).
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

    #: libpq environment variables that would route the connection, or its
    #: parameters, around the pinned host: `PGHOSTADDR` overrides the address
    #: `host` resolves to, and a service file can supply host/port/user
    #: (T-03.9-11). Any of them set -> refuse before connecting. `PGHOST` is
    #: harmless because `host` is always passed explicitly.
    REFUSED_ENV = ("PGHOSTADDR", "PGSERVICE", "PGSERVICEFILE")

    #: The pgpass line shape a fix message shows. `<password>` is a
    #: placeholder; the real password is never read into this code.
    PGPASS_LINE_HINT = (
        "wrds-pgdata.wharton.upenn.edu:9737:wrds:$WRDS_USERNAME:<password>"
    )

    #: Process-level cache, username -> session.
    _shared: dict[str, "WrdsSession"] = {}

    def __init__(self, username: str) -> None:
        self.username = username
        self._conn = None
        # Set BEFORE `psycopg2.connect` is called, so a failed connect is
        # never retried by this object (each attempt can push Duo, D-20).
        self._connect_attempted = False
        self._broken = False

    @classmethod
    def shared(cls) -> "WrdsSession":
        """The process-wide session for the current `WRDS_USERNAME`.

        The variable is read on EVERY call, so a changed username gets its own
        session rather than a stale one.
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
        """Close and forget every shared session."""
        sessions = list(cls._shared.values())
        cls._shared.clear()
        for session in sessions:
            session.close()

    def close(self) -> None:
        conn, self._conn = self._conn, None
        if conn is not None:
            conn.close()

    # -- credential pre-checks (D-07, D-20) ---------------------------------------

    @staticmethod
    def _pgpass_path() -> Path:
        """`$PGPASSFILE` when set, else `~/.pgpass` -- the file libpq reads."""
        override = os.environ.get("PGPASSFILE")
        return Path(override) if override else Path.home() / ".pgpass"

    def _assert_refused_env_unset(self) -> None:
        for name in self.REFUSED_ENV:
            if os.environ.get(name):
                raise WrdsSessionError(
                    f"{name} is set in the environment. libpq would use it to "
                    f"route the WRDS connection or its parameters away from "
                    f"the pinned host {self.HOST}:{self.PORT}; unset {name} "
                    f"before running a WRDS acquisition."
                )

    def _assert_pgpass_entry(self) -> None:
        """Fail fast, BEFORE any connection attempt, when libpq would not find
        a password for this connection.

        Without this, libpq would connect with no password, the server would
        refuse it, and the operator would see a bare authentication error --
        after a Duo push. Checked: the file exists, is a regular file, is not
        group/world accessible (libpq ignores such a file), and has a line
        whose host, port, database and user fields match (`*` wildcards,
        `\\:` escapes). Only fields 1-4 are parsed; the password field is
        never bound to a name, and no message quotes a line. The username is
        written as `$WRDS_USERNAME`, never its value (T-03.9-10).
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
        """Open the connection ONCE, read-only, and return it.

        Called through the module attribute `psycopg2.connect` so the test
        suite's D-28 tripwire sees every attempt. No password argument: libpq
        resolves it from the pgpass file checked above. The refused-env and
        pgpass checks run before the attempt; the attempt flag is set before
        the call, so neither a failed connect nor a broken session is ever
        followed by a second connect from this object.
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
        """Run `work(connection)`, turning any driver error into a
        `WrdsSessionError` and marking the session broken.

        The message carries the driver's text with the username replaced by
        `$WRDS_USERNAME`; libpq errors never carry the password.
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
        """`taqm_{YYYY}.complete_nbbo_{YYYYMMDD}`, built from a `date` only."""
        return sql.Identifier(f"taqm_{day:%Y}", f"complete_nbbo_{day:%Y%m%d}")

    @staticmethod
    def where_clause(pairs) -> sql.Composed:
        """`sym_root = ANY(roots) AND (sym_root, coalesce(sym_suffix, '')) IN
        (pairs)`.

        The first conjunct is what the server's `sym_root` chunk-group filters
        prune on; the second picks the exact share classes. Every value is a
        `sql.Literal`, never interpolated text (T-03.9-03).
        """
        pairs = [(str(root), str(suffix or "")) for root, suffix in pairs]
        if not pairs:
            # Without pairs there is no `sym_root` predicate, and a query over
            # a whole day table is exactly the ~76 s full scan (or the
            # multi-GB COPY) this class must never issue (D-24).
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
        """The COPY statement for one day and one symbol batch.

        No ordering, grouping or de-duplication clause and no time predicate:
        the physical order is the only tie-breaker before 2018 (D-19), and the
        full day is collected (D-04).
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
        """`SELECT count(*) FROM <day table> WHERE <where_clause(pairs)>`.

        The SAME WHERE as `copy_query`, so a count and the pull it prices (or
        checks) cannot select different rows. `count(*)` is an aggregate
        without GROUP BY; no ORDER BY / DISTINCT / LIMIT exists here either.
        """
        return sql.SQL("SELECT count(*) FROM {table} WHERE {where}").format(
            table=cls.table_identifier(day),
            where=cls.where_clause(pairs),
        )

    # -- generic query helpers (03.10 D-03) ----------------------------------
    #
    # PROVIDER-NEUTRAL by design. One WRDS account serves several products
    # (NYSE TAQ millisecond, CRSP daily, Compustat), and they all reach the
    # server through the ONE session this class is -- a second connection can
    # push Duo and the role allows seven. So a second provider must not have to
    # add its own CRSP-shaped methods to a TAQ module, and does not: it asks
    # for a schema by NAME, hands over a `psycopg2.sql` composable, and gets
    # rows or CSV bytes back.
    #
    # All three go through `self._query`, which is what applies the username
    # scrub and the broken-session rule. A helper that reached
    # `self._connection()` directly would put the role name into an exception
    # message (T-03.10-43) and would leave a driver-errored session reusable,
    # so the next call reconnects and pushes Duo.

    def schema_usable(self, schema: str) -> bool:
        """Whether this role may read `schema` (USAGE on it).

        Asked through `pg_namespace` rather than as
        `has_schema_privilege('<name>', 'USAGE')` on the name: the name form
        raises for a schema that does not exist, and a driver error breaks the
        session. A missing schema returns no row, i.e. not entitled (D-21) --
        so "no such schema" and "not subscribed" both read as `False`, which is
        the answer a caller can act on either way.

        The schema travels as a QUERY PARAMETER, never interpolated into the
        statement text.
        """

        def work(conn):
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
        """Execute `query` (a `psycopg2.sql` composable, or text) and return
        every row.

        For catalogue and metadata reads -- the shapes that fit in memory. A
        data pull uses `copy_csv` below instead, because `COPY ... TO STDOUT`
        moves a full day of records without materialising them as Python
        tuples.
        """

        def work(conn):
            with conn.cursor() as cursor:
                cursor.execute(query)
                return cursor.fetchall()

        return self._query(work)

    def copy_csv(self, query) -> bytes:
        """Run a `COPY ... TO STDOUT` statement and return the CSV bytes.

        The composed statement goes to `copy_expert` as-is (psycopg2 renders a
        `Composable` against the cursor's own connection), so no SQL text is
        ever assembled outside `psycopg2.sql`. Up to `COPY_SPOOL_BYTES` stay in
        memory before the buffer spills to a temporary file.
        """

        def work(conn):
            with tempfile.SpooledTemporaryFile(
                max_size=self.COPY_SPOOL_BYTES
            ) as buffer, conn.cursor() as cursor:
                cursor.copy_expert(query, buffer)
                buffer.seek(0)
                return buffer.read()

        return self._query(work)

    # -- network methods -------------------------------------------------------

    def has_schema_usage(self, year: int) -> bool:
        """Whether this role may read `taqm_{year}` (USAGE on the schema).

        The TAQ-shaped name over the generic `schema_usable` above: the
        statement, its parameter and its answer are unchanged, and the only
        thing stated here is which schema a TAQ year lives in.
        """
        return self.schema_usable(f"taqm_{int(year)}")

    def assert_entitled(self, years) -> None:
        """Raise `WrdsEntitlementError` naming EVERY requested `taqm_YYYY` this
        account cannot read.

        Run before the first data query of a pull or a probe, so an
        unentitled year stops the run with zero COPY calls instead of failing
        every batch of every day (D-21).
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
        """Rows `copy_query(day, pairs, ...)` would return."""
        query = self.count_query(day, pairs)

        def work(conn):
            with conn.cursor() as cursor:
                cursor.execute(query)
                return cursor.fetchone()

        row = self._query(work)
        return int(row[0])

    def trading_days(self, year: int) -> list[date]:
        """Every day that has a `complete_nbbo_YYYYMMDD` table in `taqm_{year}`,
        ascending.

        Listed from `information_schema` rather than probed per calendar day:
        catching a missing-table error would make a missing table look the
        same as a holiday.
        """

        def work(conn):
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
        """The day table's columns in server order (sorted locally on
        `ordinal_position`)."""

        def work(conn):
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
        """Run `copy_query` and return the CSV bytes (header included).

        The TAQ-shaped name over the generic `copy_csv` above: this states
        which statement to run, `copy_csv` states how to run it. The composed
        statement is unchanged, which matters more here than anywhere -- the
        live-verified NBBO pull depends on the COPY carrying no ORDER BY, no
        GROUP BY, no DISTINCT and no time predicate (D-19).
        """
        return self.copy_csv(self.copy_query(day, pairs, columns))


def trading_days_between(session, start: date, end: date) -> list[date]:
    """The trading days in `[start, end]`, ascending: the days that have a
    `complete_nbbo` table, listed per year through `session.trading_days`.

    Shared by the acquisition and the volume probe so both walk exactly the
    same days.
    """
    days: set[date] = set()
    for year in range(start.year, end.year + 1):
        days.update(session.trading_days(year))
    return sorted(day for day in days if start <= day <= end)


class WrdsTaqNbboAcquisition(Acquisition):
    """WRDS TAQ `complete_nbbo` acquisition behind the shared `Acquisition` base.

    One page = one trading day's table for one symbol batch: `_fetch_page`
    reads the day named by the page token (the first trading day of the window
    when there is none) and hands back the next trading day's ISO date as the
    next token. The base `_fetch_batch` owns the loop, the shard writes, the
    page ledger and resume (D-06).

    The raw tier lands under `.../wrds/data_type=nbbo/date=/symbol=/` with one
    row per NBBO record, the full day, unfiltered (D-02/D-04/D-05).
    """

    VENDOR = "wrds"

    #: TAQ `time_m` is US/Eastern wall clock; the `date=` hive key is the ET
    #: session date.
    SESSION_TIME_ZONE = "America/New_York"

    #: The only data type this vendor serves under `frequency="tick"`.
    TICK_DATA_TYPES = ("nbbo",)

    #: Symbols per day-table query. A working value, overridable via
    #: `kwargs["batch_size"]`.
    DEFAULT_BATCH_SIZE = 25

    #: One shared connection, so one worker (D-20). A different value is
    #: refused in `__init__`.
    DEFAULT_MAX_WORKERS = 1

    #: Count every page with the COPY's own WHERE before pulling it, and fail
    #: the page on a mismatch. Overridable via `kwargs["verify_page_counts"]`;
    #: the count is < 1 s per (day, batch) on the columnar chunk filters.
    DEFAULT_VERIFY_PAGE_COUNTS = True

    CREDENTIAL_ENV_VARS = (USERNAME_ENV,)
    REDACTION = "<WRDS CREDENTIAL REDACTED>"

    SCHEMA_PATTERN = "taqm_{year}"
    TABLE_PATTERN = "complete_nbbo_{ymd}"

    #: `BRK.B` <-> (`BRK`, `B`): the constituent universes' notation (D-15).
    SUFFIX_DELIMITER = "."

    #: `downloads/us_equity/tick/{DEFAULT_SUBDIR}/wrds`.
    DEFAULT_SUBDIR = "wrds_taq"

    #: The TAQ columns this class reads. `time_m_nano` is optional (absent
    #: before 2018-01-02); every other one is required.
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

    #: The pinned shard projection and order. TAQ's `date` is renamed
    #: `taq_date`: `_write_shard` derives a `date` hive key and would overwrite
    #: and then drop a raw column of that name.
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
        # Resolved through the module global at call time, so the test
        # fixture's patch of `WrdsSession` takes effect.
        self._session = WrdsSession.shared()
        self._trading_days_cache: dict[tuple[str, str], list[date]] = {}

    @property
    def _data_type(self) -> str:
        """Always `nbbo`, and only under `frequency="tick"`.

        There is deliberately no default: the `data_type=` hive key and the
        watermark namespace both come from this value.
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

    # -- failure policy (D-21) ------------------------------------------------------

    #: Exceptions that mean "the one session, or the account, is unusable".
    GLOBAL_STOP_ERRORS = (
        WrdsSessionError,
        WrdsEntitlementError,
        psycopg2.OperationalError,
        psycopg2.InterfaceError,
    )

    def _classify_error(self, exc: BaseException) -> str:
        """In this vendor `"quota"` means GLOBAL STOP, not an allocation.

        A dead single session or a missing entitlement is never one symbol's
        fault (D-21): recording it in the failure manifest would defame every
        symbol of every remaining batch, and retrying batch by batch would
        reconnect -- one Duo push per batch (D-20). So the session and
        entitlement errors, and the raw driver errors that mean the
        connection is gone, stop dispatch for the whole run and stay out of
        the manifest. Everything else (a malformed page, a stranger symbol, a
        count mismatch) is a per-batch "failed", retried next run.
        """
        if isinstance(exc, self.GLOBAL_STOP_ERRORS):
            return "quota"
        return super()._classify_error(exc)

    # -- symbols (D-15) --------------------------------------------------------

    @classmethod
    def symbol_to_pair(cls, symbol: str) -> tuple[str, str | None]:
        """`"BRK.B"` -> `("BRK", "B")`; `"AAPL"` -> `("AAPL", None)`.

        A hyphenated symbol (`BRK-B`, the Tiingo roster's form) is REFUSED
        rather than guessed at: `TRADEABLE_TICKER_PATTERN` admits both
        delimiters, and silently querying `sym_root = 'BRK-B'` would return
        nothing and look like a symbol with no data (D-15). More than one dot,
        or an empty root/suffix, is refused too.
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
        """`("BRK", "B")` -> `"BRK.B"`; an empty or `None` suffix -> the root."""
        return str(root) if not suffix else f"{root}{cls.SUFFIX_DELIMITER}{suffix}"

    # -- trading days ------------------------------------------------------------

    @staticmethod
    def _as_date(value) -> date:
        return date.fromisoformat(str(value)[:10])

    def _trading_days(self, start_date, end_date) -> list[date]:
        """The trading days in `[start_date, end_date]`, ascending, from the
        tables that exist -- never by probing calendar days."""
        start = self._as_date(start_date)
        end = self._as_date(end_date)
        key = (start.isoformat(), end.isoformat())
        cached = self._trading_days_cache.get(key)
        if cached is not None:
            return cached
        result = trading_days_between(self._session, start, end)
        self._trading_days_cache[key] = result
        return result

    # -- entitlement preflight (D-21) ---------------------------------------------

    def _run(self, symbols: list[str] | None, from_watermark: bool):
        """Check the TAQ entitlement for every year of the window, THEN run.

        The check happens before the base runner dispatches a single batch, so
        an unentitled year (the live account has no `taqm_2012`) raises
        `WrdsEntitlementError` out of `download()`/`refresh()` with zero COPY
        calls and no failure-manifest write, instead of failing every batch of
        every day of that year one by one (D-21). The window checked is the
        config's: a refresh's per-symbol start is never earlier than it.
        """
        start = self._as_date(self.config.start_date)
        end = self._as_date(self.config.end_date)
        self._session.assert_entitled(range(start.year, end.year + 1))
        return super()._run(symbols, from_watermark)

    # -- one page = one day-table query --------------------------------------

    def _empty_page(self) -> pl.DataFrame:
        return pl.DataFrame(schema=self.RAW_SCHEMA).select(self.RAW_COLUMNS)

    def _fetch_page(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str,
        page_token: str | None = None,
    ) -> tuple[pl.DataFrame, str | None]:
        """One trading day's `complete_nbbo` rows for one symbol batch.

        Returns `(frame, next_token)`, `next_token` being the next trading
        day's ISO date or `None` after the window's last day.

        **No sort, no dedup, no filter of any record** (D-02/D-04/D-05).
        `wrds_row_ord` is assigned from the rows' arrival order before any
        other operation touches the frame (D-19).
        """
        symbols = self._validate_symbols(symbols)
        # Before ANY query: a hyphenated or malformed symbol is refused here.
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
        # TAQ_COLUMNS order, NOT server order: the SELECT, and so the frame,
        # is identical for every day of every era (D-18).
        columns = tuple(name for name in self.TAQ_COLUMNS if name in server_columns)
        missing = [
            name
            for name in self.TAQ_COLUMNS
            if name not in server_columns and name not in self.OPTIONAL_TAQ_COLUMNS
        ]
        if missing:
            # A drift the D-18 evidence did not show fails loudly; it is never
            # null-filled.
            raise ValueError(
                f"{self.class_name}: {table} reports no {missing} column(s); "
                f"the table layout no longer matches the one this class was "
                f"verified against (D-18). Columns seen: "
                f"{sorted(server_columns)}."
            )

        # Completeness check (RESEARCH Pattern 6): the same WHERE, counted
        # before the COPY. A page that parses to a different number of rows
        # fails as a per-batch "failed" and is re-fetched by the next run.
        expected_rows = (
            self._session.count_rows(day, pairs)
            if self._knob("verify_page_counts", self.DEFAULT_VERIFY_PAGE_COUNTS)
            else None
        )

        raw = self._session.copy_nbbo_csv(day, pairs, columns)

        frame = pl.read_csv(io.BytesIO(raw), infer_schema=False)
        # FIRST, before anything can reorder the rows (D-19).
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
        """Every returned row is for the table's day and a REQUESTED share
        class; otherwise the page (and so the batch) fails.

        Checks, never filters: dropping a stranger row would hide a WHERE
        clause that stopped doing what it says (D-15), and a row dated off the
        table day would land under the wrong `date=` partition. The ET session
        date of the reconstructed `timestamp` is checked as well as the raw
        `date` field, so the hive key `_write_shard` derives cannot disagree
        with the table the row came from.
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

    # -- config (D-27) -----------------------------------------------------------

    @classmethod
    def build_config(
        cls,
        symbols,
        start_date: str | None = None,
        end_date: str | None = None,
        kwargs: dict | None = None,
        subdir: str = DEFAULT_SUBDIR,
    ) -> AcquisitionConfig:
        """The `AcquisitionConfig` for a WRDS NBBO pull, built directly.

        This is the descriptor's `config_factory` (D-27): a classmethod here
        rather than a function in `quantlab/config`. The raw root TERMINATES at
        the vendor segment (`.../{subdir}/wrds`) and the watermarks live in the
        sibling `.../{subdir}/_watermarks/wrds`. No credential goes into the
        config.
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
    """Counts the `complete_nbbo` rows a pull WOULD fetch, per trading day,
    for the SQL volume guard (D-16, D-24).

    One `count(*)` per (trading day, symbol batch), with the batches chunked
    exactly like `Acquisition._batches` and the WHERE clause built by the same
    `WrdsSession.where_clause` the COPY uses -- so the numbers the guard prices
    are the rows the pull will move. Each count is < 1 s on the server's
    columnar chunk-group filters (the D-24 calibration); a whole-table count
    takes ~76 s and is never issued (`where_clause` refuses an empty batch).

    Not adopted from RESEARCH Pattern 6: a JSON cache of these counts under the
    watermark root. It would only save the re-count seconds on a re-run, and it
    would be a second state file beside the watermarks that can disagree with
    them. The pull re-counts each page anyway (`verify_page_counts`).
    """

    #: Log progress every this many trading days.
    LOG_EVERY_DAYS = 20

    def __init__(
        self,
        session,
        batch_size: int = WrdsTaqNbboAcquisition.DEFAULT_BATCH_SIZE,
    ) -> None:
        self.session = session
        self.batch_size = max(1, int(batch_size))

    def _batches(self, symbols: list[str]) -> list[list[str]]:
        return [
            symbols[index : index + self.batch_size]
            for index in range(0, len(symbols), self.batch_size)
        ]

    def count_rows_by_day(
        self, symbols, start_date: str, end_date: str
    ) -> dict[str, int]:
        """`{ISO trading day: rows}` over `[start_date, end_date]`, summed over
        the symbol batches. Entitlement is checked first, so an unentitled year
        raises `WrdsEntitlementError` before any count is issued (D-21)."""
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
