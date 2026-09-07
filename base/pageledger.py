"""The per-batch page ledger for resumable paginated vendor fetches
(03.2-CONTEXT.md D-03/D-05, SC-3).

A `us_all` backfill issues thousands of multi-symbol batch requests, and a
vendor like Alpaca returns each batch as an opaque-token-chained sequence of
pages sorted by symbol first, then by bar timestamp. Restarting an interrupted
run at the START of a batch re-burns every page already paid for; restarting at
the wrong page silently drops the symbols in between. `PageLedger` records the
furthest position reached per batch so a resumed run continues MID-batch, and
records enough alongside the verbatim token to re-derive a resume point WITHOUT
one -- because the vendor publishes no statement about token lifetime either
way (D-03).

This module is a LEAF, exactly like its sibling `base/chunking.py`: stdlib
only, and ZERO project-internal imports. That is what keeps it unit-testable
without constructing an `Acquisition` and what makes it structurally incapable
of introducing an import cycle. (`base.acquisition` imports it, never the other
way round.)
"""

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Iterable, Optional, Sequence


class PageLedger:
    """A JSON sidecar recording which pages of ONE batch have been fetched.

    Written under `{watermark_path}/_pages/`, a SIBLING of the raw tier and
    never inside it: a polars directory scan walks every file beneath the root
    it is given, so a `.json` sidecar in the raw tree would break
    `pl.scan_parquet` outright (D-19 contract 2).

    **One file per batch, and therefore one WRITER per file.** Unlike
    `ChunkLedger`, which is written from a single sequential loop, this ledger
    is written from N worker threads at once. `_flush()` is atomic per file,
    but the in-memory `payload["pages"].append(...)` that precedes it is not --
    two threads interleaving on ONE shared manifest would silently lose a
    record, and the loss would surface only as a re-fetched page much later.
    Per-batch files remove the shared mutable state entirely, so no lock is
    needed and a corrupt sidecar costs one batch's resume rather than the run's.

    **The conditional that travels with the code:** if these files are ever
    collapsed into a single manifest, a `threading.Lock` around
    append-plus-flush stops being optional and becomes mandatory. Do not make
    that change without adding the lock in the same commit.

    A missing file is an EMPTY ledger, not an error -- that is the normal state
    of a first run.
    """

    #: Appended to the batch key to derive the sidecar filename.
    SUFFIX = ".pages.json"

    #: The subdirectory under `watermark_path` that holds every page ledger.
    DIRNAME = "_pages"

    def __init__(self, path: str, symbols: Optional[Sequence[str]] = None) -> None:
        """`symbols` is the batch's CURRENT roster.

        When supplied, a stored ledger whose recorded `symbol_fingerprint`
        differs from this roster's is discarded and reads back EMPTY rather
        than being resumed onto -- see `_load`.
        """
        self.path = str(path)
        self.symbols = None if symbols is None else tuple(str(s) for s in symbols)
        self._payload = self._load()

    def __repr__(self) -> str:
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
        """Stable across runs, so a resume finds the same batch's ledger.

        sha256 of `vendor|frequency|start|end|<sorted,comma-joined symbols>`,
        truncated to 16 hex characters -- long enough that a collision across
        the ~155 batches of a full-market backfill is not a concern, short
        enough to keep the shard filename readable.

        Symbols are SORTED before hashing because a batch is a SET. This is a
        deliberate difference from `ChunkLedger.fingerprint`, whose pinned
        symbol AXIS is ORDERED and hashes in order: there, two orderings of the
        same symbols produce two differently-aligned Zarr stores, so the order
        is part of the identity. Here, requesting `["B","A"]` and `["A","B"]`
        issues the same vendor request and returns the same rows, so treating
        them as two different batches would re-fetch data already on disk.
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
        """sha256 over the newline-joined roster, SORTED.

        `ChunkLedger.fingerprint()`'s idiom, with the one change `batch_key`
        documents: the roster here is a set, so it is sorted before hashing and
        the fingerprint is order-insensitive.
        """
        joined = "\n".join(sorted(str(symbol) for symbol in symbols))
        return hashlib.sha256(joined.encode("utf-8")).hexdigest()

    @classmethod
    def default_path(cls, watermark_path: str, batch_key: str) -> str:
        """`{watermark_path}/_pages/{batch_key}.pages.json`."""
        return str(Path(watermark_path) / cls.DIRNAME / f"{batch_key}{cls.SUFFIX}")

    # -- storage ------------------------------------------------------------

    def _empty(self) -> dict:
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
        """Read the sidecar, or an empty payload.

        Every key is filled by a per-key `setdefault` rather than by trusting
        the file's shape, so the schema is ADDITIVE IN BOTH DIRECTIONS: a
        sidecar written by an older build reads back with the new keys at their
        empty defaults, and a sidecar written by a newer build keeps its extra
        keys through a read/write cycle here. Neither reader ever crashes on
        the other's file.

        **A roster mismatch reads back EMPTY.** If this ledger was written for
        `{A,B,C}` and the current batch is `{A,B,D}`, resuming onto it would
        skip pages that were never fetched for `D` and would attribute pages to
        a roster that no longer exists. Returning an empty payload restarts the
        batch at page 0 instead, which is correct and merely costs a re-fetch.

        **A MISSING fingerprint on a ledger that has pages is treated as a
        mismatch**, not as "no opinion". Skipping the check there is the same
        failure with an extra step: the pages were fetched for a roster nobody
        can now identify, so resuming onto them is resuming onto an unknown.
        Reachable via any hand-edited, externally produced or partially
        restored ledger; `describe()` now flushes, which closes the path that
        produced it from this code (WR-08).
        """
        path = Path(self.path)
        if not path.exists():
            return self._empty()
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (json.JSONDecodeError, OSError):
            # A corrupt sidecar costs THIS batch's resume and nothing more --
            # one file per batch is what bounds the blast radius. Same tolerant
            # policy `Acquisition._read_sidecar` applies to watermarks.
            return self._empty()
        if not isinstance(payload, dict):
            return self._empty()

        empty = self._empty()
        for key, value in empty.items():
            payload.setdefault(key, value)

        if self.symbols is not None:
            fingerprint = payload["symbol_fingerprint"]
            if fingerprint is None:
                # A ledger with PAGES but no fingerprint is exactly the state
                # `describe()`'s docstring says must not be resumed onto -- "a
                # ledger with pages but no fingerprint could be resumed onto by
                # a different roster" -- and the loader used to tolerate it,
                # skipping the check entirely and resuming past pages that were
                # fetched for a symbol set nobody can now identify.
                #
                # An identity-less ledger with NO pages is harmless (there is
                # nothing to resume onto, and `describe()` is about to stamp
                # it), so it is left alone rather than discarded: emptying it
                # would throw away any forward-compatible extra keys a newer
                # writer put there.
                if payload["pages"]:
                    return self._empty()
            elif fingerprint != self.fingerprint(self.symbols):
                return self._empty()

        return payload

    # -- reads --------------------------------------------------------------

    @property
    def pages(self) -> list[dict]:
        return list(self._payload["pages"])

    @property
    def symbol_fingerprint(self) -> Optional[str]:
        return self._payload["symbol_fingerprint"]

    @property
    def symbol_count(self) -> Optional[int]:
        return self._payload["symbol_count"]

    def resume_point(self) -> tuple[int, Optional[str]]:
        """`(next_page_index, page_token)` -- where the next request starts.

        The token returned is the `next_token` the LAST recorded page carried,
        so a resumed run's first request is the one that was interrupted, not
        the one that already succeeded. `(0, None)` only when nothing is
        recorded, which is the first-run state.
        """
        pages = self._payload["pages"]
        if not pages:
            return 0, None
        last = pages[-1]
        return int(last["index"]) + 1, last.get("next_token")

    def symbols_seen(self) -> set[str]:
        """Every symbol that has carried at least one row on ANY recorded page.

        Accumulated across the WHOLE batch and never computed per page. The
        vendor sorts symbol-major, so page 0 of a 100-symbol batch legitimately
        holds one symbol; a per-page `requested - seen` would stamp the other
        99 as "queried, no data", advance their watermarks and skip them
        forever -- a silent 99% loss that looks like a successful run
        (03.2-RESEARCH.md Pitfall 4). The batch-wide set is only MEANINGFUL
        once `is_complete()`, and callers must honour that.
        """
        return {str(symbol) for symbol in self._payload["symbols_with_data"]}

    def is_complete(self) -> bool:
        return bool(self._payload["complete"])

    def last_position(self) -> tuple[Optional[str], Optional[str]]:
        """`(last_symbol, last_timestamp)` from the last recorded page.

        The TOKEN-FREE fallback D-03 asks for. The vendor publishes no
        statement about token lifetime, and its own published example token
        decodes to a plain positional `SYMBOL|TIMEFRAME|TIMESTAMP` tuple -- so
        if a stored token is ever rejected, a resume can degrade to re-issuing
        the batch with `start` narrowed to this timestamp and the roster
        trimmed to the symbols at or after this symbol. Wasteful, but correct.
        """
        pages = self._payload["pages"]
        if not pages:
            return None, None
        last = pages[-1]
        return last.get("last_symbol"), last.get("last_timestamp")

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
        """Record WHICH batch this ledger belongs to, without recording a page.

        Called before the first request so the fingerprint exists even for a
        batch that fails on page 0 -- a ledger with pages but no fingerprint
        could be resumed onto by a different roster.

        **FLUSHES.** Without the flush the identity lived in memory only and
        reached disk on the first `record_page`, so the very state the sentence
        above forbids was reachable through the normal path: a batch that died
        between `describe()` and its first successful page left a file with an
        identity-less shape for the next run to inherit. `_load` now also
        refuses a pages-carrying ledger with no fingerprint, so the two halves
        cover each other -- one keeps the state from being written, the other
        keeps it from being trusted (WR-08).
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

        `next_token` is recorded VERBATIM and is never re-derived by encoding a
        `symbol|timeframe|timestamp` tuple of our own. The vendor's encoding is
        undocumented and can change without notice; a re-derived token that
        stops matching would resume at a position the vendor never agreed to.
        `last_symbol` / `last_timestamp` are recorded ALONGSIDE it as the
        token-free fallback -- see `last_position`.
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
        """Discard every recorded page, keeping this ledger's batch identity.

        Called when a caller has ALREADY decided to re-fetch this batch -- the
        `resume=False` knob, say. The page ledger answers "where within this
        batch do I resume", never "should this batch be fetched at all"; that
        second question belongs to the per-symbol watermark layer (D-05's two
        layers, separate responsibilities). A completed ledger that vetoed a
        re-fetch would let the within-batch mechanism silently override a
        symbol-level policy the user set explicitly.

        Re-fetching is safe precisely because shard filenames are
        deterministic: page N of the same batch writes the same path and
        OVERWRITES it, so a redo costs requests and never produces a duplicate
        row.
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
        """Record that the batch's page chain terminated (`next_token is None`).

        Only after this is `symbols_seen()` a statement about the batch rather
        than about how far it happened to get.
        """
        self._payload["complete"] = True
        self._flush()

    # -- consistency --------------------------------------------------------

    def assert_consistent(self, raw_root: str) -> None:
        """Refuse to resume when the ledger and the disk disagree.

        The ledger and the shard tree are two independent records of the same
        truth, written at different instants. A resume trusts NEITHER alone: a
        ledger recording page N whose shard is not on disk means the run would
        skip a hole and produce a batch that is silently short, with nothing
        failing at the time and nothing detectable afterwards.

        The write ordering makes the opposite gap harmless: the shard is
        written strictly BEFORE the ledger record, and the shard filename is
        deterministic, so a crash in that window costs a re-fetch and an
        OVERWRITE -- never a duplicate row, never a lost page. Only
        ledger-ahead-of-disk is a genuine disagreement, and that is what this
        checks.
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
        """Rewrite the sidecar ATOMICALLY.

        Written to a temp file in the SAME directory and then `os.replace`d, so
        a crash mid-write leaves either the previous valid ledger or the new
        one -- never a half-written file that cannot be parsed, which would
        make the next run unable to resume at all. Copied from
        `base/chunking.py:ChunkLedger._flush`, whose reasoning applies here
        unchanged.
        """
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
