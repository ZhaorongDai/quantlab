"""Offline stand-ins for WRDS NYSE TAQ `complete_nbbo` data (phase 03.9).

Nothing here touches the network. `FakeWrdsSession` mirrors the public surface
of `quantlab.acquisition.wrds.taq.WrdsSession` and is patched over it by the
`mock_wrds_session` fixture in `tests/conftest.py`; the autouse
`_forbid_wrds_network` tripwire stays live underneath, so a fake that failed to
install would fail the test instead of reaching WRDS (D-28).

The column names, their ORDER and the sample rows are the live-verified ones
from `03.9-LIVE-CHECK-1.json` (keys `L2_columns` and `L6_copy`), not guesses.
"""

from __future__ import annotations

import csv
import io
import os
from datetime import date

import polars as pl

# The REAL session class, captured at import time: `mock_wrds_session` patches
# `quantlab.acquisition.wrds.taq.WrdsSession` with `FakeWrdsSession` AFTER this
# module is imported, and the fake builds its SQL through the real static
# builders so the shape tests cover what the acquisition actually requests.
from quantlab.acquisition.wrds.taq import WrdsSession as RealWrdsSession

#: `taqm_{YYYY}.complete_nbbo_{YYYYMMDD}` columns, in server order, for tables
#: from 2018-01-02 on (LIVE-CHECK-1 L2).
TAQ_COLUMNS_2018_ON: tuple[str, ...] = (
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

#: The same table before 2018-01-02: no `time_m_nano` (LIVE-CHECK-1 L2/L3).
TAQ_COLUMNS_PRE_2018: tuple[str, ...] = tuple(
    name for name in TAQ_COLUMNS_2018_ON if name != "time_m_nano"
)

#: The first day whose table carries `time_m_nano`.
NANO_FIRST_DAY = date(2018, 1, 2)

#: The real CSV `COPY` returned for AAPL on 2024-01-24 (LIVE-CHECK-1 L6).
_L6_SAMPLE_CSV = (
    "date,time_m,time_m_nano,sym_root,sym_suffix,qu_cond,natbbo_ind,qu_source,"
    "nbbo_qu_cond,best_bid,best_bidsizeshares,best_ask,best_asksizeshares\n"
    "2024-01-24,04:00:00.005984,226,AAPL,,R,4,N,,180,100,,\n"
    "2024-01-24,04:00:00.006053,463,AAPL,,R,4,N,,,,,\n"
    "2024-01-24,04:00:00.006286,979,AAPL,,R,4,N,,187.1,100,,\n"
    "2024-01-24,04:00:00.006393,385,AAPL,,R,4,N,,189.1,100,,\n"
    "2024-01-24,04:00:00.006417,816,AAPL,,R,4,N,,193.1,100,195.2,300\n"
)

#: The five L6 rows as dicts, with `None` where COPY wrote an empty field.
L6_SAMPLE_ROWS: list[dict[str, str | None]] = [
    {key: (value if value != "" else None) for key, value in row.items()}
    for row in csv.DictReader(io.StringIO(_L6_SAMPLE_CSV))
]


def _field(value) -> str | None:
    """A Python value as the string COPY CSV would carry, `None` for NULL."""
    if value is None:
        return None
    return str(value)


def taq_row(
    time_m: str,
    bid,
    bid_size,
    ask,
    ask_size,
    *,
    nano=None,
    day: str = "2024-01-24",
    root: str = "AAPL",
    suffix: str | None = None,
    qu_cond: str = "R",
    natbbo_ind: str = "4",
    qu_source: str = "N",
    nbbo_qu_cond: str | None = None,
) -> dict[str, str | None]:
    """One `complete_nbbo` record as COPY CSV would carry it.

    Every value is a string or `None` (NULL). `time_m` is ET wall clock, e.g.
    `"09:30:30.000000"`. The dict carries the 2018+ column set; a pre-2018 table
    simply does not select `time_m_nano`.
    """
    return {
        "date": day,
        "time_m": time_m,
        "time_m_nano": _field(nano),
        "sym_root": root,
        "sym_suffix": _field(suffix),
        "qu_cond": _field(qu_cond),
        "natbbo_ind": _field(natbbo_ind),
        "qu_source": _field(qu_source),
        "nbbo_qu_cond": _field(nbbo_qu_cond),
        "best_bid": _field(bid),
        "best_bidsizeshares": _field(bid_size),
        "best_ask": _field(ask),
        "best_asksizeshares": _field(ask_size),
    }


def _pair_to_symbol(root: str, suffix: str | None) -> str:
    return root if not suffix else f"{root}.{suffix}"


def _default_rows() -> dict[tuple[date, str], list[dict[str, str | None]]]:
    """A few valid two-sided rows per (day, symbol) for AAPL and MSFT."""
    rows: dict[tuple[date, str], list[dict[str, str | None]]] = {}
    for day in (date(2024, 1, 24), date(2024, 1, 25)):
        iso = day.isoformat()
        rows[(day, "AAPL")] = [
            taq_row("09:29:00.000000", 194.00, 100, 194.02, 100, nano=0, day=iso),
            taq_row("09:45:00.000000", 194.05, 200, 194.07, 300, nano=0, day=iso),
            taq_row("15:30:00.000000", 194.10, 100, 194.12, 100, nano=0, day=iso),
        ]
        rows[(day, "MSFT")] = [
            taq_row(
                "09:29:30.000000", 400.00, 100, 400.05, 100,
                nano=0, day=iso, root="MSFT",
            ),
            taq_row(
                "10:00:00.000000", 400.10, 100, 400.12, 200,
                nano=0, day=iso, root="MSFT",
            ),
        ]
    return rows


class FakeWrdsSession:
    """The offline double of `quantlab.acquisition.wrds.taq.WrdsSession`.

    Class-level state, reset by `reset()` (the fixture calls it):

    - `trading_days_by_year` -- what `trading_days(year)` returns;
    - `rows` -- `(day, universe symbol) -> [record, ...]` in PHYSICAL order,
      which is the order `copy_nbbo_csv` emits them in;
    - `copy_calls` -- every `copy_nbbo_csv` call, recorded as a dict whose
      `sql` is the REAL `WrdsSession.copy_query`, rendered;
    - `count_calls` -- every `count_rows` call, likewise with the real
      `WrdsSession.count_query` rendered;
    - `schema_checks` -- every year `has_schema_usage` was asked about;
    - `entitled_years` -- the years the fake account may read (`None` = all);
    - `raise_on` -- `{copy call index: exception}`; that COPY call is recorded
      and then raises, consuming no data;
    - `count_adjust` -- `{day: delta}` added to that day's `count_rows`
      answer, to fake a page whose COPY disagrees with its count;
    - `connections` -- how many sessions were constructed (a real one would be
      a connection, and each connection can push Duo);
    - `instance` -- the shared instance, or `None`.
    """

    trading_days_by_year: dict[int, list[date]] = {}
    rows: dict[tuple[date, str], list[dict[str, str | None]]] = {}
    copy_calls: list[dict] = []
    count_calls: list[dict] = []
    schema_checks: list[int] = []
    entitled_years: set[int] | None = None
    raise_on: dict[int, BaseException] = {}
    count_adjust: dict[date, int] = {}
    connections: int = 0
    instance: "FakeWrdsSession | None" = None

    def __init__(self, username: str) -> None:
        self.username = username
        FakeWrdsSession.connections += 1

    @classmethod
    def reset(cls) -> None:
        cls.trading_days_by_year = {2024: [date(2024, 1, 24), date(2024, 1, 25)]}
        cls.rows = _default_rows()
        cls.copy_calls = []
        cls.count_calls = []
        cls.schema_checks = []
        cls.entitled_years = None
        cls.raise_on = {}
        cls.count_adjust = {}
        cls.connections = 0
        cls.instance = None

    @classmethod
    def shared(cls) -> "FakeWrdsSession":
        username = os.environ.get("WRDS_USERNAME")
        if not username:
            raise RuntimeError(
                "WRDS_USERNAME must be set (FakeWrdsSession mirrors the real "
                "session's refusal)."
            )
        if cls.instance is None or cls.instance.username != username:
            cls.instance = cls(username)
        return cls.instance

    @classmethod
    def close_shared(cls) -> None:
        cls.instance = None

    def trading_days(self, year: int) -> list[date]:
        return sorted(self.trading_days_by_year.get(year, []))

    def table_columns(self, day: date) -> tuple[str, ...]:
        if day >= NANO_FIRST_DAY:
            return TAQ_COLUMNS_2018_ON
        return TAQ_COLUMNS_PRE_2018

    def has_schema_usage(self, year: int) -> bool:
        FakeWrdsSession.schema_checks.append(int(year))
        return self.entitled_years is None or int(year) in self.entitled_years

    def assert_entitled(self, years) -> None:
        # The REAL method, run against this fake's `has_schema_usage`: one
        # implementation of the refusal and its message.
        RealWrdsSession.assert_entitled(self, years)

    def _stored(self, day: date, pairs) -> list[dict[str, str | None]]:
        wanted = {_pair_to_symbol(root, suffix) for root, suffix in pairs}
        return [
            record
            for (row_day, symbol), stored in self.rows.items()
            if row_day == day and symbol in wanted
            for record in stored
        ]

    def count_rows(self, day: date, pairs) -> int:
        FakeWrdsSession.count_calls.append(
            {
                "day": day,
                "pairs": list(pairs),
                "sql": render_composed(RealWrdsSession.count_query(day, pairs)),
            }
        )
        return len(self._stored(day, pairs)) + self.count_adjust.get(day, 0)

    def copy_nbbo_csv(
        self,
        day: date,
        pairs: list[tuple[str, str | None]],
        columns: tuple[str, ...] | list[str],
    ) -> bytes:
        columns = tuple(columns)
        index = len(FakeWrdsSession.copy_calls)
        FakeWrdsSession.copy_calls.append(
            {
                "day": day,
                "pairs": list(pairs),
                "columns": columns,
                "sql": render_composed(
                    RealWrdsSession.copy_query(day, pairs, columns)
                ),
            }
        )
        failure = self.raise_on.get(index)
        if failure is not None:
            raise failure
        records = [
            {name: record.get(name) for name in columns}
            for record in self._stored(day, pairs)
        ]
        schema = {name: pl.String for name in columns}
        frame = (
            pl.DataFrame(records, schema=schema)
            if records
            else pl.DataFrame(schema=schema)
        )
        return frame.write_csv(null_value="").encode()


# -- psycopg2 doubles for the REAL `WrdsSession` (plan 03.9-04) ---------------


def _render_literal(value) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, tuple)):
        return "ARRAY[" + ", ".join(_render_literal(item) for item in value) + "]"
    text = str(value).replace("'", "''")
    return f"'{text}'"


def render_composed(obj) -> str:
    """Render a `psycopg2.sql` object to text WITHOUT a connection.

    `Composed.as_string()` needs a live connection for its quoting context;
    this walks `SQL` / `Identifier` / `Literal` / `Composed` itself. The output
    is for shape assertions only (which tokens appear, which table is named),
    never for execution. A plain `str` is returned unchanged.
    """
    from psycopg2 import sql

    if isinstance(obj, str):
        return obj
    if isinstance(obj, sql.Composed):
        return "".join(render_composed(part) for part in obj.seq)
    if isinstance(obj, sql.SQL):
        return obj.string
    if isinstance(obj, sql.Identifier):
        return ".".join(
            '"' + name.replace('"', '""') + '"' for name in obj.strings
        )
    if isinstance(obj, sql.Literal):
        return _render_literal(obj.wrapped)
    if isinstance(obj, sql.Placeholder):
        return "%s" if obj.name is None else f"%({obj.name})s"
    raise TypeError(f"render_composed: unsupported {type(obj).__name__}")


class FakeCursor:
    """The cursor half of `FakeConnection`: records SQL, replays answers."""

    def __init__(self, connection: "FakeConnection") -> None:
        self.connection = connection
        self._rows: list[tuple] = []

    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(self, *exc) -> bool:
        return False

    def _record(self, query, params) -> str:
        text = render_composed(query)
        self.connection.executed.append((text, params))
        failure = self.connection.fail_with
        if failure is not None:
            raise failure
        return text

    def execute(self, query, params=None) -> None:
        text = self._record(query, params)
        self._rows = list(self.connection.respond(text, params))

    def fetchall(self) -> list[tuple]:
        return list(self._rows)

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def copy_expert(self, query, buffer) -> None:
        self._record(query, None)
        buffer.write(self.connection.copy_payload)


class FakeConnection:
    """A stand-in for a psycopg2 connection, built by `fake_connect`.

    Records the connect kwargs, every `set_session` call, every executed
    statement (rendered text, params) and whether `close()` ran. Answers the
    catalog queries `WrdsSession` issues from `tables` / `columns`, answers
    `has_schema_privilege` from `entitled`, `count(*)` from `count_value`, and
    feeds `copy_payload` to `copy_expert`. Setting `fail_with` to an exception
    makes every subsequent statement raise it.
    """

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.sessions: list[dict] = []
        self.executed: list[tuple[str, object]] = []
        self.closed = False
        self.fail_with: BaseException | None = None
        self.tables: list[str] = ["complete_nbbo_20240124", "complete_nbbo_20240125"]
        self.columns: tuple[str, ...] = TAQ_COLUMNS_2018_ON
        self.entitled = True
        self.count_value = 0
        self.copy_payload = b""

    def set_session(self, **kwargs) -> None:
        self.sessions.append(kwargs)

    def cursor(self) -> FakeCursor:
        return FakeCursor(self)

    def close(self) -> None:
        self.closed = True

    def respond(self, text: str, params) -> list[tuple]:
        if "information_schema.tables" in text:
            return [(name,) for name in self.tables]
        if "information_schema.columns" in text:
            return [(name, index + 1) for index, name in enumerate(self.columns)]
        if "has_schema_privilege" in text:
            return [(True,)] if self.entitled else []
        if "count(*)" in text:
            return [(self.count_value,)]
        return []


def fake_connect(sink: list):
    """A `psycopg2.connect` replacement appending each `FakeConnection` it
    builds to `sink`. Install it INSIDE a test with
    `monkeypatch.setattr("psycopg2.connect", fake_connect(sink))`; that local
    override is the only sanctioned way past the autouse tripwire."""

    def _connect(*args, **kwargs):
        assert not args, "WrdsSession must pass connect parameters by keyword"
        connection = FakeConnection(**kwargs)
        sink.append(connection)
        return connection

    return _connect
