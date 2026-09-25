"""Time-window planning and the resume ledger for chunked ingestion.

Chunked ingestion densifies one time window at a time and appends each to
the Zarr store, so peak memory scales with the window rather than the whole
date range. ``TimeChunkPlanner`` turns an observed timestamp axis into such
windows, and ``ChunkLedger`` is the JSON sidecar recording which windows a
previous run already appended, so an interrupted conversion resumes at the
first unwritten window. Neither depends on a dataset, a backend or a config;
the module imports only pandas, xarray and the atomic JSON writer, so it
never participates in an import cycle.
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

    Every window edge returned by ``plan_from_timestamps()`` is a timestamp
    that actually appears in the axis, never a calendar period end, so a
    window handed to a densifier cannot name a day on which nothing traded.

    The accepted granularity tokens are the members of ``GRANULARITIES``. A
    new granularity needs both a new token there and a matching branch in
    ``_period_key``; the latter raises if the two lists drift apart.

    Example:
        >>> planner = TimeChunkPlanner("month")
        >>> planner.plan_from_timestamps(
        ...     pd.to_datetime(["2024-01-02", "2024-01-31", "2024-02-01"])
        ... )
        [(Timestamp('2024-01-02 00:00:00'), Timestamp('2024-01-31 00:00:00')),
         (Timestamp('2024-02-01 00:00:00'), Timestamp('2024-02-01 00:00:00'))]
    """

    #: Accepted granularity tokens, coarse to fine. Also the choices the
    #: ingest CLI's ``--chunk`` option offers.
    GRANULARITIES: tuple[str, ...] = ("year", "quarter", "month", "day", "hour")

    def __init__(self, granularity: str = "year") -> None:
        """Store the granularity after checking it is a known token.

        Raises:
            ValueError: If ``granularity`` is not in ``GRANULARITIES``.
        """
        if granularity not in self.GRANULARITIES:
            raise ValueError(
                f"TimeChunkPlanner: unknown granularity {granularity!r}; "
                f"accepted values are {list(self.GRANULARITIES)}."
            )
        self.granularity = granularity

    def __repr__(self) -> str:
        """Return the planner's granularity in constructor form."""
        return f"TimeChunkPlanner(granularity={self.granularity!r})"

    def _period_key(self, timestamp) -> tuple[int, int]:
        """Map one timestamp to the label of the period containing it.

        This is the single definition of "one window": ``_group_by_period()``
        groups by this key and nothing else. The second component is only
        ever compared for equality, never ordered, so it need not be a
        natural calendar number (the ``year`` rung returns ``0``; the
        ``hour`` rung combines day-of-year and hour).

        Raises:
            ValueError: If the granularity is in ``GRANULARITIES`` but has no
                branch here. Unreachable while the two agree, since
                ``__init__`` refuses unknown tokens; it exists to catch drift.
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
        # Only reachable if ``GRANULARITIES`` gained a token without a branch
        # above; a silent fallback here would produce windows of the wrong
        # size, and therefore the wrong memory bound.
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
        """Collapse a sorted, de-duplicated index into ``(first, last)`` pairs.

        One pair is emitted per contiguous run of equal period keys.
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
        """Return the windows covering ``timestamps``, with observed edges.

        Args:
            timestamps: Any iterable convertible to a ``DatetimeIndex``;
                duplicates are dropped and the axis is sorted first.

        Returns:
            Time-ordered, non-overlapping ``(start, end)`` pairs that together
            cover the de-duplicated axis exactly once. Both edges of each pair
            are timestamps present in the input.

        Raises:
            ValueError: If the axis is empty; planning zero windows would
                silently write an empty store.

        Example:
            >>> planner = TimeChunkPlanner("quarter")
            >>> planner.plan_from_timestamps(
            ...     pd.date_range("2024-01-01", "2024-06-30", freq="B")
            ... )
            [(Timestamp('2024-01-01 00:00:00'), Timestamp('2024-03-29 00:00:00')),
             (Timestamp('2024-04-01 00:00:00'), Timestamp('2024-06-28 00:00:00'))]
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
    """JSON sidecar recording which chunk windows have been appended.

    The file lives beside the Zarr store, never inside it, because rewriting
    the store replaces its whole directory. It records the pinned symbol axis
    as a sha256 fingerprint rather than the list itself, since the only
    question a resume asks is whether the roster is unchanged. A missing file
    is an empty ledger, the normal state of a first run.

    Example:
        >>> ledger = ChunkLedger(ChunkLedger.default_path("panel.zarr"))
        >>> ledger.assert_consistent(symbols, "panel.zarr")
        >>> if not ledger.is_written(start, end):
        ...     backend.append("panel.zarr")
        ...     ledger.record(start, end, rows=n, symbols=symbols)

        The method examples below continue from a ledger that has recorded
        one window, ``2024-01-02`` to ``2024-01-31``, of 3 rows on the axis
        ``symbols = ["AAPL", "MSFT"]``.
    """

    #: Appended to the store path to derive the default sidecar location.
    SUFFIX = ".chunks.json"

    def __init__(self, path: str, append_dim: str = "timestamp") -> None:
        """Load the ledger at ``path``, or start an empty one if absent.

        Args:
            path: The sidecar file.
            append_dim: The dimension the store is appended along.
        """
        self.path = str(path)
        self.append_dim = append_dim
        self._payload = self._load()

    def __repr__(self) -> str:
        """Return the ledger's path and its number of recorded windows."""
        return f"ChunkLedger(path={self.path!r}, windows={len(self.windows)})"

    @classmethod
    def default_path(cls, zarr_file_path: str) -> str:
        """Return ``<store>.chunks.json``, a sibling of the store directory.

        Example:
            >>> ChunkLedger.default_path("/data/panel.zarr")
            '/data/panel.zarr.chunks.json'
        """
        return f"{zarr_file_path}{cls.SUFFIX}"

    @staticmethod
    def fingerprint(symbols: Sequence[str]) -> str:
        """Return the sha256 of the newline-joined symbols, in the given order.

        Order-sensitive on purpose: the pinned axis is an ordered coordinate,
        and the same set in a different order would align columns differently.

        Example:
            >>> ChunkLedger.fingerprint(["AAPL", "MSFT"])[:16]
            '4a1c2f2b7fca8c6a'
            >>> ChunkLedger.fingerprint(["MSFT", "AAPL"])[:16]
            '66c9ca2d14cccb5a'
        """
        joined = "\n".join(str(symbol) for symbol in symbols)
        return hashlib.sha256(joined.encode("utf-8")).hexdigest()

    @staticmethod
    def _key(value) -> str:
        """Return the canonical ISO-string form of a window edge.

        Every comparison goes through this, so a ``pd.Timestamp``, a
        ``numpy.datetime64`` and an ISO string match the same recorded window.
        """
        return pd.Timestamp(value).isoformat()

    def _empty(self) -> dict:
        """Return the payload of a ledger with nothing recorded."""
        return {
            "append_dim": self.append_dim,
            "symbol_count": None,
            "symbol_fingerprint": None,
            "windows": [],
        }

    def _load(self) -> dict:
        """Read the sidecar if it exists, filling in any missing keys."""
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
        """A copy of the recorded windows, each ``{"start", "end", "rows"}``.

        Example:
            >>> ledger.windows[0]["end"], ledger.windows[0]["rows"]
            ('2024-01-31T00:00:00', 3)
        """
        return list(self._payload["windows"])

    @property
    def symbol_count(self) -> Optional[int]:
        """The symbol count the ledger was last written against, or None.

        Example:
            >>> ledger.symbol_count
            2
        """
        return self._payload["symbol_count"]

    @property
    def symbol_fingerprint(self) -> Optional[str]:
        """The fingerprint of the axis the ledger was last written against.

        Example:
            >>> ledger.symbol_fingerprint == ChunkLedger.fingerprint(symbols)
            True
        """
        return self._payload["symbol_fingerprint"]

    @property
    def last_end(self) -> Optional[str]:
        """The ``end`` of the last recorded window, or None for an empty ledger.

        Example:
            >>> ledger.last_end
            '2024-01-31T00:00:00'
        """
        if not self._payload["windows"]:
            return None
        return self._payload["windows"][-1]["end"]

    def is_written(self, start, end) -> bool:
        """Return whether the window ``(start, end)`` is already recorded.

        Example:
            >>> ledger.is_written("2024-01-02", "2024-01-31")
            True
            >>> ledger.is_written(pd.Timestamp("2024-02-01"), "2024-02-29")
            False
        """
        key = (self._key(start), self._key(end))
        return any(
            (window["start"], window["end"]) == key
            for window in self._payload["windows"]
        )

    def record(self, start, end, rows: int, symbols: Sequence[str]) -> None:
        """Append one window and rewrite the sidecar atomically.

        The file is written to a temporary sibling and renamed over the
        destination, so a crash mid-write leaves either the previous ledger
        or the new one, never a half-written file the next run cannot parse.

        Args:
            start: First timestamp of the window.
            end: Last timestamp of the window.
            rows: Number of rows appended for it.
            symbols: The pinned symbol axis the window was written on.

        Example:
            >>> ledger.record("2024-02-01", "2024-02-29", rows=20, symbols=symbols)
            >>> ledger.last_end
            '2024-02-29T00:00:00'
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
        """Re-fingerprint the ledger against a new pinned symbol axis.

        Call this only immediately after the store itself has been widened
        onto the same axis. Called without that widen, the fingerprint would
        match while the store's columns are still on the old axis, which is
        exactly the misalignment ``assert_consistent`` exists to catch.

        The recorded windows are left untouched: a widen changes the axis,
        not which windows have been written, and clearing them would make a
        complete store append every window a second time.

        Example:
            >>> ledger.rebase([*symbols, "NVDA"])  # the store was widened first
            >>> ledger.symbol_count, len(ledger.windows)
            (3, 1)
        """
        self._payload["append_dim"] = self.append_dim
        self._payload["symbol_count"] = len(symbols)
        self._payload["symbol_fingerprint"] = self.fingerprint(symbols)
        self._flush()

    def assert_consistent(self, symbols: Sequence[str], store_path: str) -> None:
        """Cross-check the ledger against the store before the first append.

        The two are independent records of the same history, and an append
        cannot be undone, so a resume trusts neither alone. Only the append
        dimension's coordinate is read from the store.

        Args:
            symbols: The pinned symbol axis of the run about to start.
            store_path: The Zarr store the ledger describes.

        Raises:
            ValueError: If the recorded fingerprint differs from ``symbols``
                (the roster changed between runs); if a store exists with an
                empty ledger (no record of what it holds); if the ledger has
                windows but no store exists; or if the store's last
                coordinate value differs from the last recorded window's end
                (a crash landed between the store write and the ledger
                update). A missing store with an empty ledger is the normal
                first run and passes.

        Example:
            >>> ledger.assert_consistent(symbols, "panel.zarr")
            >>> ledger.assert_consistent([*symbols, "NVDA"], "panel.zarr")
            Traceback (most recent call last):
                ...
            ValueError: ChunkLedger: refusing to resume panel.zarr -- the pinned ...
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
        """Return the store's last append-dim coordinate value, or None.

        Only the coordinate is read; the data variables are never loaded.
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
        """Rewrite the sidecar atomically through the shared JSON writer.

        ``indent=2`` is passed here so the on-disk file stays readable.
        """
        write_json_atomically(self.path, self._payload, indent=2)
