"""Offline stand-ins for WRDS NYSE TAQ `complete_nbbo` data (phase 03.9).

Nothing here touches the network. `FakeWrdsSession` mirrors the public surface
of `quantlab.acquisition.wrds_taq.WrdsSession` and is patched over it by the
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
    """The offline double of `quantlab.acquisition.wrds_taq.WrdsSession`.

    Class-level state, reset by `reset()` (the fixture calls it):

    - `trading_days_by_year` -- what `trading_days(year)` returns;
    - `rows` -- `(day, universe symbol) -> [record, ...]` in PHYSICAL order,
      which is the order `copy_nbbo_csv` emits them in;
    - `copy_calls` -- every `copy_nbbo_csv` call, recorded as a dict;
    - `connections` -- how many sessions were constructed (a real one would be
      a connection, and each connection can push Duo);
    - `instance` -- the shared instance, or `None`.
    """

    trading_days_by_year: dict[int, list[date]] = {}
    rows: dict[tuple[date, str], list[dict[str, str | None]]] = {}
    copy_calls: list[dict] = []
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

    def copy_nbbo_csv(
        self,
        day: date,
        pairs: list[tuple[str, str | None]],
        columns: tuple[str, ...] | list[str],
    ) -> bytes:
        columns = tuple(columns)
        FakeWrdsSession.copy_calls.append(
            {"day": day, "pairs": list(pairs), "columns": columns}
        )
        wanted = {_pair_to_symbol(root, suffix) for root, suffix in pairs}
        records = [
            {name: record.get(name) for name in columns}
            for (row_day, symbol), stored in self.rows.items()
            if row_day == day and symbol in wanted
            for record in stored
        ]
        schema = {name: pl.String for name in columns}
        frame = (
            pl.DataFrame(records, schema=schema)
            if records
            else pl.DataFrame(schema=schema)
        )
        return frame.write_csv(null_value="").encode()
