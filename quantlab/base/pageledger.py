"""Per-batch page ledger for resumable, paginated vendor downloads.

Some vendors answer a multi-symbol request as a chain of pages linked by an
opaque ``next_page_token``, sorted by symbol and then by bar timestamp. A
full-market backfill issues thousands of such batches, each of which may run
to dozens of pages, so an interrupted run must be able to continue from the
middle of a batch: restarting at page 0 re-spends every page already paid
for, and guessing a later page silently drops the symbols in between.
``PageLedger`` is the small JSON sidecar that records, for one batch, which
pages have landed, which token the next request should carry and which
parquet shards hold each page's rows.

The acquisition engine in ``quantlab/base/acquisition.py`` is the only
caller. This module imports nothing from the project except the atomic JSON
writer in ``quantlab.utils.atomic``. See ``docs/pageledger.md``.
"""

import hashlib
import json
from pathlib import Path
from typing import Iterable, Optional, Sequence

from quantlab.utils.atomic import write_json_atomically


class PageLedger:
    """JSON sidecar recording which pages of one batch have been fetched.

    Ledgers live under ``{watermark_path}/_pages/``, next to the raw data tree
    rather than inside it, because a polars directory scan of the raw tree
    would choke on a ``.json`` file. There is one file per batch and therefore
    one writer per file: batches are fetched from several worker threads at
    once, and the in-memory append that precedes each flush is not atomic, so
    a shared manifest would need a lock. If these files are ever merged into
    one, a ``threading.Lock`` around append-plus-flush becomes mandatory. A
    missing file is simply an empty ledger, which is the normal state of a
    first run.

    Args:
        path: Location of the sidecar; see ``default_path``.
        symbols: The batch's current roster. When given, a stored ledger
            whose ``symbol_fingerprint`` differs from this roster's reads
            back empty instead of being resumed onto.

    Example:
        >>> roster = ["AAPL", "MSFT"]
        >>> key = PageLedger.batch_key("alpaca", "1m", "2024-01-02",
        ...                            "2024-01-05", roster)
        >>> ledger = PageLedger(PageLedger.default_path(root, key), roster)
        >>> ledger.describe(key, "alpaca", "1m", "2024-01-02", "2024-01-05",
        ...                 roster)
        >>> ledger.resume_point()
        (0, None)
        >>> ledger.record_page(0, "tok1", rows=500, seen=["AAPL"],
        ...                    shard_paths=["raw/part-00000.pqt"])
        >>> ledger.resume_point()
        (1, 'tok1')
    """

    #: Appended to the batch key to derive the sidecar filename.
    SUFFIX = ".pages.json"

    #: The subdirectory under ``watermark_path`` that holds every page ledger.
    DIRNAME = "_pages"

    def __init__(self, path: str, symbols: Optional[Sequence[str]] = None) -> None:
        """Open the ledger at ``path``, reading it from disk if it exists.

        When ``symbols`` is supplied, a stored ledger written for a different
        roster reads back empty; see ``_load``.
        """
        self.path = str(path)
        self.symbols = None if symbols is None else tuple(str(s) for s in symbols)
        self._payload = self._load()

    def __repr__(self) -> str:
        """Return the path, the page count and the completion state."""
        return (
            f"PageLedger(path={self.path!r}, pages={len(self.pages)}, "
            f"complete={self.is_complete()})"
        )

    # -- identity -----------------------------------------------------------

    @staticmethod
    def batch_key(
        vendor: str,
        frequency: str,
        start_date: str,
        end_date: str,
        symbols: Iterable[str],
    ) -> str:
        """Return a stable 16-hex-character key identifying one batch.

        The key is the SHA-256 of ``vendor|frequency|start|end|symbols`` with
        the symbols sorted and comma-joined, truncated to 16 characters: long
        enough that collisions across a full-market backfill are not a
        concern, short enough to keep shard filenames readable. Symbols are
        sorted because a batch is a set: requesting ``["B", "A"]`` and
        ``["A", "B"]`` issues the same vendor request and returns the same
        rows, so the two must share a ledger.

        Example:
            >>> PageLedger.batch_key("alpaca", "1m", "2024-01-02", "2024-01-05",
            ...                      ["AAPL", "MSFT"])
            '91c9dc202fdc2cc1'
            >>> PageLedger.batch_key("alpaca", "1m", "2024-01-02", "2024-01-05",
            ...                      ["MSFT", "AAPL"])
            '91c9dc202fdc2cc1'
        """
        payload = "|".join(
            [
                str(vendor),
                str(frequency),
                str(start_date),
                str(end_date),
                ",".join(sorted(str(symbol) for symbol in symbols)),
            ]
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def fingerprint(symbols: Sequence[str]) -> str:
        """Return the SHA-256 of the sorted, newline-joined roster.

        Order-insensitive for the same reason ``batch_key`` is.

        Example:
            >>> PageLedger.fingerprint(["AAPL", "MSFT"])[:16]
            '4a1c2f2b7fca8c6a'
            >>> PageLedger.fingerprint(["MSFT", "AAPL"])[:16]
            '4a1c2f2b7fca8c6a'
        """
        joined = "\n".join(sorted(str(symbol) for symbol in symbols))
        return hashlib.sha256(joined.encode("utf-8")).hexdigest()

    @classmethod
    def default_path(cls, watermark_path: str, batch_key: str) -> str:
        """Return ``{watermark_path}/_pages/{batch_key}.pages.json``.

        Example:
            >>> PageLedger.default_path("/data/_watermarks/alpaca",
            ...                         "5f2a9c1e0b7d4e63")
            '/data/_watermarks/alpaca/_pages/5f2a9c1e0b7d4e63.pages.json'
        """
        return str(Path(watermark_path) / cls.DIRNAME / f"{batch_key}{cls.SUFFIX}")

    # -- storage ------------------------------------------------------------

    def _empty(self) -> dict:
        """Return a fresh payload with every key at its empty default."""
        return {
            "batch_key": None,
            "vendor": None,
            "frequency": None,
            "start_date": None,
            "end_date": None,
            "symbol_count": None,
            "symbol_fingerprint": None,
            "complete": False,
            "pages": [],
            "symbols_with_data": [],
        }

    def _load(self) -> dict:
        """Read the sidecar from disk, or return an empty payload.

        A missing, unparseable or non-object file reads back empty: a corrupt
        sidecar costs this one batch a re-fetch and nothing more. Every key
        is filled in with ``setdefault`` rather than by trusting the file's
        shape, so a file written by an older build gains the new keys at
        their defaults and a file written by a newer build keeps its extra
        keys through a read/write cycle.

        When the ledger was opened with a roster, a stored
        ``symbol_fingerprint`` that differs from the roster's reads back
        empty: resuming onto pages fetched for a different symbol set would
        skip pages never fetched for the new symbols. A ledger that has pages
        but no fingerprint is treated the same way, since nobody can say
        which roster those pages belong to. A ledger with neither fingerprint
        nor pages is kept as is, so any extra keys a newer writer stored are
        not thrown away.
        """
        path = Path(self.path)
        if not path.exists():
            return self._empty()
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (json.JSONDecodeError, OSError):
            # A corrupt sidecar costs this batch's resume and nothing more.
            return self._empty()
        if not isinstance(payload, dict):
            return self._empty()

        empty = self._empty()
        for key, value in empty.items():
            payload.setdefault(key, value)

        if self.symbols is not None:
            fingerprint = payload["symbol_fingerprint"]
            if fingerprint is None:
                # Pages with no fingerprint belong to a roster nobody can
                # identify, so they must not be resumed onto. An identity-less
                # ledger with no pages is harmless and is about to be stamped
                # by `describe()`, so it is kept rather than emptied.
                if payload["pages"]:
                    return self._empty()
            elif fingerprint != self.fingerprint(self.symbols):
                return self._empty()

        return payload

    # -- reads --------------------------------------------------------------

    @property
    def pages(self) -> list[dict]:
        """Return a copy of the recorded page records, in fetch order.

        Example:
            >>> ledger.pages[0]["index"], ledger.pages[0]["next_token"]
            (0, 'tok1')
        """
        return list(self._payload["pages"])

    @property
    def symbol_fingerprint(self) -> Optional[str]:
        """Return the stored roster fingerprint, or None before ``describe``.

        Example:
            >>> ledger.symbol_fingerprint == PageLedger.fingerprint(roster)
            True
        """
        return self._payload["symbol_fingerprint"]

    @property
    def symbol_count(self) -> Optional[int]:
        """Return the stored roster size, or None before ``describe``.

        Example:
            >>> ledger.symbol_count
            2
        """
        return self._payload["symbol_count"]

    def resume_point(self) -> tuple[int, Optional[str]]:
        """Return ``(next_page_index, page_token)`` for the next request.

        The token is the ``next_token`` carried by the last recorded page, so
        a resumed run's first request is the one that was interrupted rather
        than one that already succeeded. ``(0, None)`` means nothing has been
        recorded yet.

        Example:
            >>> ledger.resume_point()
            (1, 'tok1')
        """
        pages = self._payload["pages"]
        if not pages:
            return 0, None
        last = pages[-1]
        return int(last["index"]) + 1, last.get("next_token")

    def symbols_seen(self) -> set[str]:
        """Return every symbol that carried a row on any recorded page.

        The set accumulates over the whole batch and is only meaningful once
        ``is_complete()`` is true. Vendors sort pages by symbol first, so page
        0 of a 100-symbol batch may legitimately hold a single symbol;
        judging "queried but no data" per page would wrongly stamp the other
        99 as empty and skip them on every later run.

        Example:
            >>> ledger.symbols_seen()
            {'AAPL'}
        """
        return {str(symbol) for symbol in self._payload["symbols_with_data"]}

    def is_complete(self) -> bool:
        """Return whether the page chain has been recorded as terminated.

        Example:
            >>> ledger.is_complete()
            False
        """
        return bool(self._payload["complete"])

    # -- writes -------------------------------------------------------------

    def describe(
        self,
        batch_key: str,
        vendor: str,
        frequency: str,
        start_date: str,
        end_date: str,
        symbols: Sequence[str],
    ) -> None:
        """Record the batch identity and flush it, without recording a page.

        Called before the first request so the identity reaches disk even for
        a batch that fails on page 0. Without it, a ledger with pages but no
        fingerprint could be resumed onto by a different roster; ``_load``
        refuses such a ledger, and this method keeps one from being written.

        Example:
            >>> ledger.describe(key, "alpaca", "1m", "2024-01-02", "2024-01-05",
            ...                 roster)
            >>> ledger.symbol_count, Path(ledger.path).exists()
            (2, True)
        """
        self._payload["batch_key"] = str(batch_key)
        self._payload["vendor"] = str(vendor)
        self._payload["frequency"] = str(frequency)
        self._payload["start_date"] = str(start_date)
        self._payload["end_date"] = str(end_date)
        self._payload["symbol_count"] = len(symbols)
        self._payload["symbol_fingerprint"] = self.fingerprint(symbols)
        self.symbols = tuple(str(symbol) for symbol in symbols)
        self._flush()

    def record_page(
        self,
        index: int,
        next_token: Optional[str],
        rows: int,
        seen: Iterable[str],
        shard_paths: Sequence[str],
        last_symbol: Optional[str] = None,
        last_timestamp: Optional[str] = None,
    ) -> None:
        """Append one fetched page and rewrite the sidecar atomically.

        ``next_token`` is stored verbatim and never re-derived from a
        ``symbol|timeframe|timestamp`` tuple of our own, because the vendor's
        encoding is undocumented and may change. ``last_symbol`` and
        ``last_timestamp`` are stored alongside it as raw material for a
        token-free fallback: if a stored token is rejected, the batch can be
        re-issued with ``start`` narrowed to the last timestamp and the
        roster trimmed to the last symbol onwards. No accessor implements
        that fallback today; read ``pages[-1]`` directly if you need it.

        Args:
            index: Zero-based page number.
            next_token: The vendor's token for the following page, or None
                on the last page.
            rows: Number of rows the page carried.
            seen: Symbols that had at least one row on this page.
            shard_paths: Parquet files the page's rows were written to.
            last_symbol: Symbol of the page's final row, if known.
            last_timestamp: Timestamp of the page's final row, if known.

        Example:
            >>> ledger.record_page(1, None, rows=120, seen=["AAPL", "MSFT"],
            ...                    shard_paths=["raw/part-00001.pqt"])
            >>> ledger.resume_point()
            (2, None)
            >>> sorted(ledger.symbols_seen())
            ['AAPL', 'MSFT']
        """
        accumulated = set(self._payload["symbols_with_data"])
        accumulated.update(str(symbol) for symbol in seen)
        self._payload["symbols_with_data"] = sorted(accumulated)
        self._payload["pages"].append(
            {
                "index": int(index),
                "next_token": next_token,
                "rows": int(rows),
                "shards": [str(path) for path in shard_paths],
                "last_symbol": last_symbol,
                "last_timestamp": last_timestamp,
            }
        )
        self._flush()

    def reset(self) -> None:
        """Discard every recorded page in memory, keeping the batch identity.

        For callers that have already decided to re-fetch the batch, for
        example under ``resume=False``. The ledger answers "where within this
        batch do I resume", never "should this batch be fetched at all"; that
        second question belongs to the per-symbol watermark layer, so a
        completed ledger never vetoes a re-fetch. Re-fetching is safe because
        shard filenames are deterministic: page N of the same batch
        overwrites the same path, so a redo costs requests and never
        duplicates a row. Nothing is written to disk until the next
        ``record_page`` or ``mark_complete``.

        Example:
            >>> ledger.reset()
            >>> ledger.resume_point()
            (0, None)
            >>> ledger.symbol_count
            2
        """
        identity = {
            key: self._payload[key]
            for key in (
                "batch_key",
                "vendor",
                "frequency",
                "start_date",
                "end_date",
                "symbol_count",
                "symbol_fingerprint",
            )
        }
        self._payload = self._empty()
        self._payload.update(identity)

    def mark_complete(self) -> None:
        """Record that the page chain terminated and flush.

        Only after this is ``symbols_seen()`` a statement about the whole
        batch rather than about how far it happened to get.

        Example:
            >>> ledger.mark_complete()
            >>> ledger.is_complete()
            True
        """
        self._payload["complete"] = True
        self._flush()

    # -- consistency --------------------------------------------------------

    def assert_consistent(self, raw_root: str) -> None:
        """Raise if the ledger records a page whose shard is missing on disk.

        The ledger and the shard tree are two records of the same download,
        written at different instants. A shard is always written before its
        ledger record, so a crash between the two costs only a re-fetch that
        overwrites the same deterministic path. The opposite gap, a recorded
        page with no shard on disk, means a shard was deleted or the raw root
        moved, and resuming would leave a hole in the batch that no later
        read could detect.

        Args:
            raw_root: Directory that relative shard paths are resolved
                against.

        Raises:
            ValueError: If a recorded page names no shard, or names a shard
                that does not exist. The message says how to recover.

        Example:
            >>> ledger.assert_consistent(raw_root)  # every shard present
            >>> Path(raw_root, "raw", "part-00000.pqt").unlink()
            >>> ledger.assert_consistent(raw_root)
            Traceback (most recent call last):
                ...
            ValueError: PageLedger: refusing to resume ...
        """
        root = Path(raw_root)
        for page in self._payload["pages"]:
            shards = page.get("shards") or []
            if not shards:
                raise ValueError(
                    f"PageLedger: refusing to resume {self.path} -- error 1 of "
                    f"2: page {page['index']} is recorded but names no shard "
                    f"file, so there is no record of where its rows landed. "
                    f"Resuming would skip that page's data with nothing "
                    f"failing at the time. CURE: delete {self.path} to re-fetch "
                    f"this batch from page 0; the deterministic shard names "
                    f"mean the pages that DID land are overwritten, not "
                    f"duplicated."
                )
            for shard in shards:
                path = Path(shard)
                if not path.is_absolute():
                    path = root / shard
                if not path.exists():
                    raise ValueError(
                        f"PageLedger: refusing to resume {self.path} -- error "
                        f"2 of 2: the ledger records page {page['index']} but "
                        f"its shard {path} does not exist on disk. The two "
                        f"disagree, which means a shard was deleted or the "
                        f"raw root was moved after the ledger was written; "
                        f"resuming would leave a hole in the batch that no "
                        f"later read could detect. CURE: delete {self.path} to "
                        f"re-fetch this batch from page 0, or restore the "
                        f"missing shard under {root}."
                    )

    # -- atomic flush -------------------------------------------------------

    def _flush(self) -> None:
        """Rewrite the sidecar atomically.

        The payload is written to a temporary file in the same directory and
        renamed over the destination, so a crash mid-write leaves either the
        previous ledger or the new one, never a half-written file the next
        run could not parse. ``indent=2`` keeps the on-disk format stable.
        """
        write_json_atomically(self.path, self._payload, indent=2)
