"""Time-window chunk planning and the chunk resume ledger (260906-13w D-01/D-04).

Two pure collaborators for the chunked densify-and-append ingestion path:

- `TimeChunkPlanner` turns either an OBSERVED timestamp axis or a calendar
  date range into a list of time windows, both driven by one shared period
  rule so their definitions of "a year" cannot drift apart.
- `ChunkLedger` is the JSON sidecar recording which windows a previous run
  already appended, so an interrupted multi-hour conversion resumes at the
  first unwritten window instead of at the top.

Neither depends on a Dataset, a backend or a config -- they are functions of a
timestamp axis and a file path -- so this module is a LEAF: stdlib + pandas
only, zero project-internal imports. That is the same rule `dataset/cleaning.py`
follows, and it is what keeps this module unit-testable without touching a
store and structurally incapable of introducing an import cycle.
"""

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Iterable, Optional, Sequence

import pandas as pd


class TimeChunkPlanner:
    """Split a time range into windows at year, quarter or month boundaries.

    The class exposes TWO planners because the pipeline needs windows at two
    different moments, with two different amounts of knowledge:

    - `plan_from_timestamps()` runs during the WRITE, when the real observed
      timestamp axis exists. Every window edge it returns is an observed
      timestamp, so a window handed to a densifier can never name a day on
      which nothing traded.
    - `plan_calendar()` runs during SIZING, before the download, when no
      timestamp axis exists yet and calendar arithmetic is the only option.

    Both are built on the same private `_period_key()` and the same private
    `_group_by_period()`, so "what counts as one window" is defined exactly
    once. A future granularity is added to `GRANULARITIES` plus `_period_key`
    and both planners inherit it.
    """

    #: Accepted granularity tokens, in coarse-to-fine order. Also the values
    #: `ingest_us_equity.py --chunk` offers.
    GRANULARITIES: tuple[str, ...] = ("year", "quarter", "month")

    def __init__(self, granularity: str = "year") -> None:
        if granularity not in self.GRANULARITIES:
            raise ValueError(
                f"TimeChunkPlanner: unknown granularity {granularity!r}; "
                f"accepted values are {list(self.GRANULARITIES)}."
            )
        self.granularity = granularity

    def __repr__(self) -> str:
        return f"TimeChunkPlanner(granularity={self.granularity!r})"

    def _period_key(self, timestamp) -> tuple[int, int]:
        """Map one timestamp to the label of the period that contains it.

        The single definition of "one window" in this module. Both planners
        group by this and nothing else, which is what makes drift between the
        sizing windows and the write windows structurally impossible rather
        than merely unlikely.
        """
        ts = pd.Timestamp(timestamp)
        if self.granularity == "year":
            return (ts.year, 0)
        if self.granularity == "quarter":
            return (ts.year, ts.quarter)
        return (ts.year, ts.month)

    def _group_by_period(
        self, index: pd.DatetimeIndex
    ) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        """Collapse a sorted, de-duplicated index into (first, last) pairs,
        one per contiguous run of equal period keys.
        """
        windows: list[list[pd.Timestamp]] = []
        current_key: Optional[tuple[int, int]] = None
        for timestamp in index:
            key = self._period_key(timestamp)
            if key != current_key:
                windows.append([timestamp, timestamp])
                current_key = key
            else:
                windows[-1][1] = timestamp
        return [(start, end) for start, end in windows]

    def plan_from_timestamps(
        self, timestamps: Iterable
    ) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        """Windows whose edges are OBSERVED timestamps. This is the planner
        the write loop uses.

        Trading days are not calendar days. A window running to a calendar
        period end would name a date with no row behind it, and handing that
        to a densifier fabricates a row for a day the market was shut. Every
        edge returned here is a timestamp that actually appears in the axis.

        The returned windows are time-ordered, non-overlapping, and together
        cover the whole de-duplicated axis exactly once.
        """
        index = pd.DatetimeIndex(pd.unique(pd.DatetimeIndex(timestamps))).sort_values()
        if len(index) == 0:
            raise ValueError(
                "TimeChunkPlanner: cannot plan windows over an empty "
                "timestamp axis. An empty axis means the raw source produced "
                "no rows in the configured date range -- planning zero "
                "windows would silently write an empty store instead."
            )
        return self._group_by_period(index)

    def plan_calendar(
        self, start_date: str, end_date: str
    ) -> list[tuple[str, str]]:
        """SIZING ONLY -- never hand these windows to a densifier.

        These edges come from CALENDAR arithmetic, so they routinely name
        dates on which nothing traded (a 31 December that fell on a Sunday,
        a 1 January holiday). That is acceptable here and only here: sizing
        runs BEFORE the download, when no timestamp axis exists to plan
        against, and the estimator it feeds
        (`UniverseCatalog.estimate_dense_panel`) is already a 252/365.25
        approximation by design. The write loop uses
        `plan_from_timestamps()` instead.

        Returns ISO `YYYY-MM-DD` string pairs, clipped to `[start_date,
        end_date]` at the two outer edges, grouped by the SAME
        `_period_key()` the observed planner uses.
        """
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        if end < start:
            raise ValueError(
                f"TimeChunkPlanner: end_date {end_date} precedes start_date "
                f"{start_date}; there is no window to plan."
            )
        # Grouping the day-by-day range rather than using `pd.period_range`
        # is deliberate: it routes through `_period_key()`, the one place a
        # granularity is defined, so this planner cannot drift from
        # `plan_from_timestamps()`. Twenty years is ~7,600 iterations.
        days = pd.date_range(start, end, freq="D")
        return [
            (window_start.date().isoformat(), window_end.date().isoformat())
            for window_start, window_end in self._group_by_period(days)
        ]


class ChunkLedger:
    """A JSON sidecar recording which chunk windows have been appended.

    Written BESIDE the Zarr store, never inside it: `XrBackend.write()`
    rewrites a store with `mode="w"`, which replaces the store DIRECTORY, so a
    ledger kept inside would be destroyed by exactly the operation that makes
    a resume necessary.

    It stores a sha256 fingerprint of the pinned symbol list rather than the
    list itself -- 15,000 tickers would dominate the sidecar, and the only
    question a resume asks is "is this the same roster", which a fingerprint
    answers exactly.

    A missing file is an EMPTY ledger, not an error: that is the normal state
    of a first run.
    """

    #: Appended to the store path to derive the default sidecar location.
    SUFFIX = ".chunks.json"

    def __init__(self, path: str, append_dim: str = "timestamp") -> None:
        self.path = str(path)
        self.append_dim = append_dim
        self._payload = self._load()

    def __repr__(self) -> str:
        return f"ChunkLedger(path={self.path!r}, windows={len(self.windows)})"

    @classmethod
    def default_path(cls, zarr_file_path: str) -> str:
        """`<store>.chunks.json`, a SIBLING of the store directory."""
        return f"{zarr_file_path}{cls.SUFFIX}"

    @staticmethod
    def fingerprint(symbols: Sequence[str]) -> str:
        """sha256 over the newline-joined symbol list, in the given order.

        Order-sensitive on purpose: the pinned axis is an ORDERED coordinate,
        and two identical symbol sets in different orders would produce two
        differently-aligned stores.
        """
        joined = "\n".join(str(symbol) for symbol in symbols)
        return hashlib.sha256(joined.encode("utf-8")).hexdigest()

    @staticmethod
    def _key(value) -> str:
        """Canonical, JSON-safe window edge. Every comparison goes through
        this, so a `pd.Timestamp`, a `numpy.datetime64` and an ISO string all
        match the same recorded window.
        """
        return pd.Timestamp(value).isoformat()

    def _empty(self) -> dict:
        return {
            "append_dim": self.append_dim,
            "symbol_count": None,
            "symbol_fingerprint": None,
            "windows": [],
        }

    def _load(self) -> dict:
        if not Path(self.path).exists():
            return self._empty()
        with open(self.path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        payload.setdefault("append_dim", self.append_dim)
        payload.setdefault("symbol_count", None)
        payload.setdefault("symbol_fingerprint", None)
        payload.setdefault("windows", [])
        return payload

    @property
    def windows(self) -> list[dict]:
        return list(self._payload["windows"])

    @property
    def symbol_count(self) -> Optional[int]:
        return self._payload["symbol_count"]

    @property
    def symbol_fingerprint(self) -> Optional[str]:
        return self._payload["symbol_fingerprint"]

    @property
    def last_end(self) -> Optional[str]:
        """The `end` of the last recorded window, or None for an empty ledger."""
        if not self._payload["windows"]:
            return None
        return self._payload["windows"][-1]["end"]

    def is_written(self, start, end) -> bool:
        key = (self._key(start), self._key(end))
        return any(
            (window["start"], window["end"]) == key
            for window in self._payload["windows"]
        )

    def record(self, start, end, rows: int, symbols: Sequence[str]) -> None:
        """Append one window and rewrite the sidecar ATOMICALLY.

        Written to a temp file in the same directory and then `os.replace`d,
        so a crash mid-write leaves either the previous valid ledger or the
        new one -- never a half-written file that cannot be parsed, which
        would make the next run unable to resume at all.
        """
        self._payload["append_dim"] = self.append_dim
        self._payload["symbol_count"] = len(symbols)
        self._payload["symbol_fingerprint"] = self.fingerprint(symbols)
        self._payload["windows"].append(
            {
                "start": self._key(start),
                "end": self._key(end),
                "rows": int(rows),
            }
        )
        self._flush()

    def _flush(self) -> None:
        directory = Path(self.path).parent
        directory.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=str(directory),
            prefix=Path(self.path).name + ".",
            suffix=".tmp",
            delete=False,
        )
        try:
            with handle:
                json.dump(self._payload, handle, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(handle.name, self.path)
        except BaseException:
            Path(handle.name).unlink(missing_ok=True)
            raise
