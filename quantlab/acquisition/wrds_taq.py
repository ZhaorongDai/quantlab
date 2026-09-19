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
"""

from __future__ import annotations

import io
import os
import tempfile
from datetime import date

import polars as pl
import psycopg2
from psycopg2 import sql

from quantlab.acquisition.registry import (
    Capability,
    SourceDescriptor,
    register_source,
)
from quantlab.base.acquisition import Acquisition
from quantlab.base.config import AcquisitionConfig
from quantlab.config import get_data_root
from quantlab.dataset.nbbo import NbboPanelDataset

#: The one environment variable the WRDS username is read from.
#:
#: MODULE-LEVEL on purpose, the alpaca.py rule: `WrdsSession` is a patch target
#: (`tests/conftest.py:mock_wrds_session` replaces it wholesale), so a
#: redaction control reached THROUGH it could be silently disabled by a stub
#: that does not define the name. `CREDENTIAL_ENV_VARS` below is built from
#: this constant, never from the session class.
USERNAME_ENV = "WRDS_USERNAME"


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

    #: Process-level cache, username -> session.
    _shared: dict[str, "WrdsSession"] = {}

    def __init__(self, username: str) -> None:
        self.username = username
        self._conn = None

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

    def _connection(self):
        """Open the connection ONCE, read-only, and return it.

        Called through the module attribute `psycopg2.connect` so the test
        suite's D-28 tripwire sees every attempt. No password argument: libpq
        resolves it from `~/.pgpass`.
        """
        if self._conn is None:
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
        return self._conn

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

    # -- network methods -------------------------------------------------------

    def trading_days(self, year: int) -> list[date]:
        """Every day that has a `complete_nbbo_YYYYMMDD` table in `taqm_{year}`,
        ascending.

        Listed from `information_schema` rather than probed per calendar day:
        catching a missing-table error would make a missing table look the
        same as a holiday.
        """
        with self._connection().cursor() as cursor:
            cursor.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = %s "
                "AND table_name ~ '^complete_nbbo_[0-9]{8}$'",
                (f"taqm_{year}",),
            )
            names = [row[0] for row in cursor.fetchall()]
        days = [
            date(int(name[-8:-4]), int(name[-4:-2]), int(name[-2:]))
            for name in names
        ]
        return sorted(days)

    def table_columns(self, day: date) -> tuple[str, ...]:
        """The day table's columns in server order (sorted locally on
        `ordinal_position`)."""
        with self._connection().cursor() as cursor:
            cursor.execute(
                "SELECT column_name, ordinal_position "
                "FROM information_schema.columns "
                "WHERE table_schema = %s AND table_name = %s",
                (f"taqm_{day:%Y}", f"complete_nbbo_{day:%Y%m%d}"),
            )
            rows = cursor.fetchall()
        return tuple(name for name, _ in sorted(rows, key=lambda row: row[1]))

    def copy_nbbo_csv(self, day: date, pairs, columns) -> bytes:
        """Run `copy_query` and return the CSV bytes (header included)."""
        conn = self._connection()
        query = self.copy_query(day, pairs, columns)
        with tempfile.SpooledTemporaryFile(
            max_size=self.COPY_SPOOL_BYTES
        ) as buffer, conn.cursor() as cursor:
            cursor.copy_expert(query.as_string(conn), buffer)
            buffer.seek(0)
            return buffer.read()


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

    # -- symbols (D-15) --------------------------------------------------------

    @classmethod
    def symbol_to_pair(cls, symbol: str) -> tuple[str, str | None]:
        """`"BRK.B"` -> `("BRK", "B")`; `"AAPL"` -> `("AAPL", None)`."""
        root, delimiter, suffix = str(symbol).partition(cls.SUFFIX_DELIMITER)
        return root, (suffix if delimiter and suffix else None)

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
        days: set[date] = set()
        for year in range(start.year, end.year + 1):
            days.update(self._session.trading_days(year))
        result = sorted(day for day in days if start <= day <= end)
        self._trading_days_cache[key] = result
        return result

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

        server_columns = self._session.table_columns(day)
        columns = [name for name in server_columns if name in self.TAQ_COLUMNS]
        missing = [
            name
            for name in self.TAQ_COLUMNS
            if name not in columns and name not in self.OPTIONAL_TAQ_COLUMNS
        ]
        if missing:
            raise ValueError(
                f"{self.class_name}: {self.TABLE_PATTERN.format(ymd=f'{day:%Y%m%d}')} "
                f"reports no {missing} column(s); the table layout no longer "
                f"matches the one this class was verified against (D-18). "
                f"Columns seen: {list(server_columns)}."
            )

        pairs = [self.symbol_to_pair(symbol) for symbol in symbols]
        raw = self._session.copy_nbbo_csv(day, pairs, tuple(columns))

        frame = pl.read_csv(io.BytesIO(raw), infer_schema=False)
        # FIRST, before anything can reorder the rows (D-19).
        frame = frame.with_columns(
            pl.int_range(pl.len(), dtype=pl.Int64).alias("wrds_row_ord")
        )
        if frame.height == 0:
            return self._empty_page(), next_token

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
        frame = frame.cast(self.RAW_SCHEMA)
        return frame.select(self.RAW_COLUMNS), next_token

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


#: The registry descriptor for WRDS -- "who I am", beside the class that is
#: "how I download" (03.4 D-05, 03.9 D-17).
WRDS_SOURCE = register_source(
    SourceDescriptor(
        vendor="wrds",
        display_name="WRDS NYSE TAQ millisecond NBBO",
        acquisition_cls=WrdsTaqNbboAcquisition,
        config_factory=WrdsTaqNbboAcquisition.build_config,
        capabilities=(
            Capability(
                market="us_equity",
                frequency="tick",
                data_type="nbbo",
                dataset_cls=NbboPanelDataset,
                earliest_available="2003-09-10",
                entitlement="WRDS NYSE TAQ millisecond subscription",
            ),
        ),
        #: A LITERAL, restated rather than derived from `CREDENTIAL_ENV_VARS`
        #: (the D-04 pinning test would otherwise be `x == x`).
        required_env=("WRDS_USERNAME",),
        universe_categories=("sp500_constituent", "nasdaq100_constituent"),
    )
)
