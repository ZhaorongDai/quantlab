"""Time-window chunk planning and the chunk resume ledger (260906-13w D-01/D-04).

Two pure collaborators for the chunked densify-and-append ingestion path:

- `TimeChunkPlanner` turns an OBSERVED timestamp axis into a list of time
  windows, so every edge it returns is a timestamp that actually appears in
  the data.
- `ChunkLedger` is the JSON sidecar recording which windows a previous run
  already appended, so an interrupted multi-hour conversion resumes at the
  first unwritten window instead of at the top.

Neither depends on a Dataset, a backend or a config -- they are functions of a
timestamp axis and a file path -- so this module is a LEAF: stdlib plus
pandas/xarray plus the one stdlib-only leaf `quantlab.utils.atomic`, and no
other project-internal import. That is the same rule `dataset/cleaning.py`
follows, and it is what keeps this module unit-testable without touching a
Dataset and structurally incapable of introducing an import cycle --
`utils.atomic` imports nothing from this project at all, so depending on it
cannot close a cycle. (`base.data` imports this module, never the other way
round.)
"""

import hashlib
import json
from pathlib import Path
from typing import Iterable, Optional, Sequence

import pandas as pd
import xarray as xr

from quantlab.utils.atomic import write_json_atomically


class TimeChunkPlanner:
    """Split an observed timestamp axis into windows at a period boundary.

    The accepted granularity tokens are the members of `GRANULARITIES`, in
    coarse-to-fine order. This docstring deliberately does not re-enumerate
    them: the enumeration that used to stand here went stale the moment the
    ladder grew, and a summary line that can go stale is a summary line that
    will.

    There is ONE planner. `plan_from_timestamps()` runs during the WRITE,
    when the real observed timestamp axis exists, and every window edge it
    returns is an observed timestamp -- so a window handed to a densifier can
    never name a day on which nothing traded. The class has no other mode.

    A second, SIZING-time planner stood beside it until phase 03.6. It ran
    BEFORE the download, when no timestamp axis existed yet, so its edges came
    from calendar arithmetic; its only production consumer was the
    pre-download dense-panel RAM estimator that phase 03.6 deleted. Calendar
    edges routinely name days on which nothing traded, which is why that
    method had to open its own docstring with a warning never to hand its
    windows to a densifier -- and that warning was the only thing enforcing
    it. One entry point removes the warning by removing the hazard.

    The drift this class does have to defend against is between
    `GRANULARITIES` and `_period_key`. A new granularity is added in two
    places, the tuple and a `_period_key` branch, and adding it to only the
    first used to fall through to month-sized windows under another name --
    a wrong window size is a wrong memory bound, and it failed open.
    `_period_key` now ends in a `raise ValueError` naming BOTH lists. That
    raise is unreachable while the two agree, because `__init__` already
    refuses any token absent from `GRANULARITIES`; being unreachable is
    exactly what makes it a drift detector rather than dead code.
    """

    #: Accepted granularity tokens, in coarse-to-fine order. Also the values
    #: `ingest_us_equity.py --chunk` offers.
    GRANULARITIES: tuple[str, ...] = ("year", "quarter", "month", "day", "hour")

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

        The single definition of "one window" in this module.
        `_group_by_period()` groups by this and nothing else, so "what counts
        as one window" is defined in exactly one place rather than restated
        wherever a window is needed.

        The SECOND component is not a natural calendar number -- the `year`
        rung already returns a literal `0`. It is a within-year monotonically
        increasing discriminant that `_group_by_period()` only ever compares
        for EQUALITY: it is never ordered and never has arithmetic done on it.
        That is what lets the finest rung stay inside the declared
        `tuple[int, int]`: `hour`'s widest value is `366 * 24 + 23 == 8807`,
        on a leap year's last hour.
        """
        ts = pd.Timestamp(timestamp)
        if self.granularity == "year":
            return (ts.year, 0)
        if self.granularity == "quarter":
            return (ts.year, ts.quarter)
        if self.granularity == "month":
            return (ts.year, ts.month)
        if self.granularity == "day":
            return (ts.year, ts.dayofyear)
        if self.granularity == "hour":
            return (ts.year, ts.dayofyear * 24 + ts.hour)
        # Unreachable while `GRANULARITIES` and the branches above agree, and
        # that is the point: `__init__` already refuses a token absent from
        # `GRANULARITIES`, so only DRIFT between the two lists reaches here.
        # Before this raise existed the method ended in a bare `return
        # (ts.year, ts.month)`, so a rung added to `GRANULARITIES` without a
        # branch here silently produced MONTH-sized windows under another
        # name -- a wrong window size is a wrong memory bound, and it failed
        # open. The class docstring's claim that a new granularity is added in
        # `GRANULARITIES` plus a branch here was enforced by nothing before
        # this raise existed; it is now true by construction.
        raise ValueError(
            f"TimeChunkPlanner: granularity {self.granularity!r} is accepted by "
            f"GRANULARITIES but _period_key defines no period for it; accepted "
            f"values are {list(self.GRANULARITIES)}. Add a branch to "
            f"_period_key or remove the token from GRANULARITIES -- the two "
            f"lists have drifted apart."
        )

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
        """Windows whose edges are OBSERVED timestamps. This is the planner.

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

        Written to a temp file in the same directory and then renamed over
        the destination, so a crash mid-write leaves either the previous
        valid ledger or the new one -- never a half-written file that cannot
        be parsed, which would make the next run unable to resume at all.
        See `quantlab.utils.atomic.write_json_atomically` for the mechanism.
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

    def rebase(self, symbols: Sequence[str]) -> None:
        """Re-fingerprint the ledger against a NEW pinned symbol axis.

        Valid ONLY immediately after a successful widen of the store onto that
        same axis (`XrBackend.widen_symbol_axis`). Called without one it
        re-fingerprints a store whose columns are still on the OLD axis, which
        is precisely the misalignment `assert_consistent` exists to catch --
        the check would then pass and the next append would silently write onto
        a store whose columns mean something else.

        `windows` is deliberately left untouched. The ledger records WHICH
        WINDOWS have been written; a widen changes the AXIS, not the windows.
        Clearing them would make an already-complete store re-densify from the
        top and append every window a second time.
        """
        self._payload["append_dim"] = self.append_dim
        self._payload["symbol_count"] = len(symbols)
        self._payload["symbol_fingerprint"] = self.fingerprint(symbols)
        self._flush()

    def assert_consistent(self, symbols: Sequence[str], store_path: str) -> None:
        """Cross-check the ledger against the store before the first append.

        The two are independent records of the same truth, written at
        different instants, and an append is irreversible: it cannot be
        validated after the fact from the store alone (T-13w-02). So a resume
        trusts neither on its own and refuses on any disagreement rather than
        appending a window that would misalign the whole store.

        Four cases, in order:

        - store absent, ledger empty -> the normal first run;
        - a recorded fingerprint that differs from the current pinned symbol
          list -> the roster changed between runs, so every stored column is
          on a different axis than the next window would be;
        - a store with no ledger -> there is no record of WHICH windows are
          already in it, so appending would blindly duplicate or skip;
        - a store whose last append-dim value differs from the last recorded
          window's end -> a crash landed between a successful `to_zarr` and
          the ledger write, and re-running the window would duplicate it.

        Only the append-dim COORDINATE is read from the store; the data
        variables are never loaded.
        """
        store_exists = Path(store_path).exists()
        recorded = self._payload["windows"]

        if self.symbol_fingerprint is not None:
            current = self.fingerprint(symbols)
            if current != self.symbol_fingerprint:
                raise ValueError(
                    f"ChunkLedger: refusing to resume {store_path} -- the "
                    f"pinned symbol axis has {len(symbols)} symbol(s) but the "
                    f"ledger at {self.path} was written against "
                    f"{self.symbol_count}. A roster refresh between runs is "
                    f"the usual cause. Every window in the store is "
                    f"materialised on the OLD axis, so appending one on the "
                    f"new axis would silently misalign every column. Delete "
                    f"the store and the ledger to rebuild from scratch."
                )

        if store_exists and not recorded:
            raise ValueError(
                f"ChunkLedger: a store exists at {store_path} but there is no "
                f"chunk ledger at {self.path}, so there is no record of which "
                f"windows it already holds. Appending blind would duplicate "
                f"or skip windows with no way to tell afterwards. Delete the "
                f"store to rebuild it, or restore the ledger."
            )

        if not store_exists and recorded:
            raise ValueError(
                f"ChunkLedger: the ledger at {self.path} records "
                f"{len(recorded)} written window(s) but no store exists at "
                f"{store_path}. Delete the ledger to start over."
            )

        if store_exists and recorded:
            tail = self._store_tail(store_path)
            expected = recorded[-1]["end"]
            if tail is not None and self._key(tail) != expected:
                raise ValueError(
                    f"ChunkLedger: refusing to resume {store_path} -- the "
                    f"store's last {self.append_dim} is "
                    f"{pd.Timestamp(tail).date()} but the ledger's last "
                    f"recorded window ends "
                    f"{pd.Timestamp(expected).date()}. The two disagree, "
                    f"which means a crash landed between a successful write "
                    f"and the ledger update; re-running would duplicate or "
                    f"skip a window rather than resume cleanly."
                )

    def _store_tail(self, store_path: str):
        """The store's last append-dim coordinate value, or None.

        Opened lazily and only the coordinate is touched -- reading the data
        variables to answer a one-value question would defeat the whole point
        of chunking.
        """
        store = xr.open_zarr(store_path)
        try:
            if self.append_dim not in store.coords:
                return None
            values = store[self.append_dim].values
            return values[-1] if len(values) else None
        finally:
            store.close()

    def _flush(self) -> None:
        """Rewrite the ledger ATOMICALLY through the shared writer.

        The temp-file/rename body this method used to carry inline -- and
        that `base/pageledger.py:PageLedger._flush` copied verbatim -- now
        lives once in
        `quantlab.utils.atomic.write_json_atomically` (03.4-03). The `indent=2`
        stays HERE rather than moving into the helper so this ledger's on-disk
        bytes are unchanged by the extraction.
        """
        write_json_atomically(self.path, self._payload, indent=2)
