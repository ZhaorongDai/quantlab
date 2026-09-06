import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Self, Sequence

import polars as pl
from loguru import logger

from base.config import AcquisitionConfig
from base.pageledger import PageLedger
from enums.constant import Date
from enums.data import RAW_HIVE_KEYS, Vendor

#: The one well-formedness rule a symbol must satisfy before it becomes a
#: filesystem path segment or a query-string value.
#:
#: Imported from `acquisition/universe.py` rather than re-declared: quick task
#: 260906-eme established that this pattern admits DIGITS (`[A-Z0-9]`) for a
#: documented reason -- digit-bearing tickers are legitimate US-equity symbols.
#: A second, stricter regex written here would reject real symbols while
#: looking like a security improvement.
#:
#: The import is deferred to `_validate_symbols` (a local import inside the
#: method) because `acquisition/universe.py` imports nothing from this module
#: today, but a module-level import here would make `base.acquisition` depend
#: on the `acquisition` package and invert the layering.
_TICKER_PATTERN = re.compile(r"^[A-Z0-9]{1,7}(?:[.-][A-Z0-9]{1,2})?$")


@dataclass
class BatchOutcome:
    """What one `_fetch_batch` call did.

    `symbols_with_data` is a strict subset of `symbols` and is only a statement
    about ABSENCE when `complete` is true. A vendor that sorts symbol-major
    legitimately returns one symbol on page 0 of a 100-symbol batch, so an
    incomplete batch has no opinion at all about which symbols have no data
    (03.2-RESEARCH.md Pitfall 4).
    """

    symbols: tuple[str, ...]
    symbols_with_data: set[str] = field(default_factory=set)
    pages: int = 0
    complete: bool = False


class Acquisition(ABC):
    """Abstract base for network-fetching, config-driven data acquisition.

    Mirrors `base/data.py:Dataset`'s config lifecycle idiom (config
    property/setter, `import_path`, `class_name`), but is decoupled from the
    Dataset/Zarr layer entirely -- `Acquisition` subclasses only fetch raw
    vendor data over the network and write it to local raw files under
    `config.raw_data_dir_path`; they never touch xarray/Zarr storage. The
    matching `Dataset` subclass is responsible for converting those raw local
    files into the canonical xarray representation via `_raw_data_to_xr()`.

    `download()` performs a one-shot full backfill over
    `[config.start_date, config.end_date]`. `refresh()` performs an
    incremental fetch per-symbol, starting from that symbol's last recorded
    watermark date (falling back to `config.start_date` if no watermark
    exists yet), so repeated calls do not re-fetch the entire history.

    **Watermarks record the covered RANGE, not just its end** (260906-26o
    D-03). Recording only the end date made a widened `start_date` silently
    skip every already-fetched symbol, shipping a dataset whose per-symbol
    history depth was inconsistent with no warning at all. The schema is
    additive -- `{"start_date": ..., "last_date": ...}`, with `last_date`
    keeping its original name -- so old and new readers each tolerate the
    other's files. An unknown covered start is represented by the key being
    ABSENT and is never guessed; see `_read_coverage` and `stamp_watermarks`.
    """

    def __init__(self, config: AcquisitionConfig):
        self.config = config

    def __repr__(self):
        return f"{self.__class__.__name__}(config={self.config})"

    @property
    def config(self) -> AcquisitionConfig:
        return self._config

    @config.setter
    def config(self, config: AcquisitionConfig):
        self._config = config
        self._config.name = self.import_path

        if self._config.start_date is None:
            self._config.start_date = Date.START_DATE
        if self._config.end_date is None:
            self._config.end_date = Date.END_DATE

    @property
    def import_path(self) -> str:
        return f"{self.__class__.__module__}.{self.__class__.__qualname__}"

    @property
    def class_name(self) -> str:
        return self.__class__.__name__

    def _watermark_path(self, symbol: str) -> Path:
        return Path(self.config.watermark_path) / f"{symbol}.json"

    def _read_sidecar(self, symbol: str) -> dict | None:
        """Load a watermark sidecar's raw JSON, or None if it is absent or
        unparseable.

        The single tolerant read both `_read_watermark` and `_read_coverage`
        share, so there is exactly ONE failure policy for a corrupt sidecar
        rather than two that could drift apart.
        """
        path = self._watermark_path(symbol)
        if not path.exists():
            return None
        try:
            with open(path) as f:
                payload = json.load(f)
        except (json.JSONDecodeError, OSError):
            # A missing or corrupt watermark sidecar must never crash the
            # refresh workflow -- fall back to None (config.start_date),
            # matching Dataset._reset_symbols' FileNotFoundError-fallback
            # pattern. Worst case is a wider-than-necessary re-fetch.
            return None
        return payload if isinstance(payload, dict) else None

    def _read_watermark(self, symbol: str) -> str | None:
        """The LAST covered date for `symbol`, or None.

        Signature and meaning are deliberately unchanged by the range-aware
        schema: both loops below and
        `ConcurrentTiingoAcquisition._attempt` use this to compute an
        incremental start, and none of them wants the covered start.
        """
        payload = self._read_sidecar(symbol)
        return None if payload is None else payload.get("last_date")

    def _read_coverage(self, symbol: str) -> dict | None:
        """The covered RANGE for `symbol` as `{"start_date", "last_date"}`,
        or None when no readable sidecar exists.

        Either component may be None. In particular a LEGACY sidecar --
        `{"last_date": ...}`, the only format written before 260906-26o --
        reads back with `start_date=None`, and nothing anywhere fills that in
        from `config.start_date` or any other fallback.

        That absence is the whole point (D-04). Only the user knows what
        window those files were actually fetched over; an invented start that
        happens to be wrong reproduces exactly the silent per-symbol history
        gap this schema exists to eliminate, and reproduces it invisibly.
        Stamping is therefore an explicit, user-supplied step --
        `stamp_watermarks()` below.
        """
        payload = self._read_sidecar(symbol)
        if payload is None:
            return None
        return {
            "start_date": payload.get("start_date"),
            "last_date": payload.get("last_date"),
        }

    def _write_watermark(
        self, symbol: str, last_date: str, start_date: str | None = None
    ) -> None:
        """Record coverage for `symbol`.

        The schema is purely ADDITIVE: `last_date` keeps its name and meaning,
        so new code reads pre-26o files and pre-26o code reads new files, and
        no reader anywhere crashes on either.

        `start_date=None` omits the key ENTIRELY rather than writing a null --
        an unknown covered start is represented by absence, so it cannot be
        mistaken at read time for a recorded value.
        """
        path = self._watermark_path(symbol)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, str] = {"last_date": last_date}
        if start_date is not None:
            payload["start_date"] = start_date
        with open(path, "w") as f:
            json.dump(payload, f)

    def stamp_watermarks(self, start_date: str) -> int:
        """Fill the covered start into every sidecar that lacks one, and
        return how many files changed.

        The explicit, user-supplied migration D-04 requires. It takes the
        start from its CALLER and derives it from nothing -- not from
        `config.start_date`, not from a default. A sidecar that already
        records a start is left untouched, because overwriting a recorded
        range with a guessed one is the same silent-wrong-data failure in a
        different costume (T-26o-04).

        Issues zero network requests.
        """
        directory = Path(self.config.watermark_path)
        if not directory.exists():
            return 0

        changed = 0
        for path in sorted(directory.glob("*.json")):
            symbol = path.stem
            coverage = self._read_coverage(symbol)
            # Skips the failure manifest and anything unparseable: the
            # manifest has no `last_date`, and a corrupt file reads as None.
            if coverage is None or coverage["last_date"] is None:
                continue
            if coverage["start_date"] is not None:
                continue
            self._write_watermark(
                symbol, coverage["last_date"], start_date=start_date
            )
            changed += 1

        logger.info(
            f"Stamped covered start {start_date} onto {changed} watermark(s) "
            f"under {directory}; sidecars already recording a start were left "
            f"untouched."
        )
        return changed

    #: How many symbols one `_fetch_page` request carries.
    #:
    #: 1 is the DEGENERATE case, not a special case: a vendor whose API takes
    #: one symbol per call sets 1, and the batched path is then behaviourally
    #: identical to the per-symbol path it replaces -- same request count, same
    #: failure granularity, same resume granularity. That is what makes
    #: `_fetch_page` a real contract rather than a multi-symbol vendor's
    #: interface that a single-symbol vendor has to pretend to implement.
    #:
    #: Overridable per run via `config.kwargs["batch_size"]`.
    DEFAULT_BATCH_SIZE: int = 1

    #: Which vendor this subclass fetches from. Set by every subclass. It is
    #: written into every shard as a literal `vendor` column and hashed into
    #: every `PageLedger.batch_key`, so it is part of the on-disk contract.
    VENDOR: Vendor

    #: The PINNED projection AND order of every raw shard this subclass writes.
    #:
    #: `_write_shard` projects through `frame.select(RAW_COLUMNS)`, so the raw
    #: tier is schema-stable BY CONSTRUCTION. That matters because a directory
    #: scan derives ONE schema from the first file it opens and enforces it
    #: across all of them: a single shard written with an extra column, or with
    #: the same columns in a different order, makes the whole vendor root
    #: unreadable (03.2-RESEARCH.md Pitfall 6). Pinning at write time is the
    #: cure; relaxing the scan's strictness would reopen the silent cross-vendor
    #: merge while looking like a bug fix.
    RAW_COLUMNS: tuple[str, ...]

    def _knob(self, name: str, default=None):
        """Read a per-run tuning parameter from `config.kwargs`.

        The escape hatch `AcquisitionConfig` documents -- a knob read here
        never becomes a constructor argument no config file could reach, which
        is what keeps the whole pipeline config-driven per CLAUDE.md.
        """
        return (self.config.kwargs or {}).get(name, default)

    # -- symbol validation --------------------------------------------------

    def _validate_symbols(self, symbols: Sequence[str]) -> list[str]:
        """Reject any symbol that is not a well-formed ticker, and return the
        validated list.

        Called BEFORE path construction and BEFORE any query-string
        interpolation, because a symbol crosses two trust boundaries at once:

        - it becomes a filesystem path component under the raw root, where a
          value containing `/` or `..` would escape that root entirely
          (T-03.2-03);
        - it becomes one element of a comma-joined `symbols=` query parameter,
          where an embedded comma would silently change WHICH symbols were
          requested -- the response would look fine and the data would be for
          something else (T-03.2-04).

        One control covers both, which is why it lives here on the base rather
        than in each vendor's `_fetch_page`.

        The pattern is `acquisition/universe.py`'s `_WELL_FORMED_TICKER`, not a
        stricter one written for this method: quick task 260906-eme established
        that it admits digits deliberately, because digit-bearing tickers are
        real. A second regex here would reject legitimate symbols while looking
        like a hardening step.
        """
        validated = []
        for symbol in symbols:
            text = str(symbol)
            if not _TICKER_PATTERN.match(text):
                raise ValueError(
                    f"{self.class_name}: refusing to fetch {text!r} -- it does "
                    f"not match the well-formed ticker pattern "
                    f"{_TICKER_PATTERN.pattern}. A symbol becomes both a "
                    f"filesystem path segment under {self.config.raw_data_dir_path} "
                    f"and a comma-joined query-string value, so a separator, a "
                    f"parent reference or an embedded comma would escape the "
                    f"raw root or silently change which symbols were requested. "
                    f"Fix the roster rather than relaxing this pattern."
                )
            validated.append(text)
        return validated

    # -- raw shard layout ---------------------------------------------------

    @property
    def _hive_keys(self) -> tuple[str, ...]:
        """The hive partition key(s) for this config's frequency.

        Read from `enums.data.RAW_HIVE_KEYS`, the SAME mapping
        `dataset/stock.py` reads, so the writer and the reader cannot drift.
        """
        return RAW_HIVE_KEYS[self.config.frequency]

    def _hive_partition_values(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Add the hive key column(s) this frequency partitions on.

        For `1d` the single key is `month`, formatted `YYYY-MM` as a STRING --
        which is what the reader's `hive_schema` pins. ISO ordering makes a
        plain string comparison of `month` against the window edges correct, so
        the reader needs no date parsing and no time zone can creep in
        (03.2-RESEARCH.md Pitfall 8: an unpinned numeric-looking hive value is
        inferred as an integer and a string comparison against it silently
        matches nothing).

        `1m` and `tick` derivation -- including the US/Eastern SESSION-date
        conversion for their `date=` key (D-19 contract 7) -- lands in 03.2-06
        with the writers that consume them.
        """
        keys = self._hive_keys
        if keys == ("month",):
            return frame.with_columns(
                pl.col("timestamp").dt.strftime("%Y-%m").alias("month")
            )
        raise NotImplementedError(
            f"{self.class_name}: hive key derivation for frequency "
            f"{self.config.frequency!r} (keys {keys}) is not implemented yet. "
            f"The `1m` and `tick` writers land in 03.2-06; only `1d` "
            f"(`month=`) is wired today."
        )

    def _shard_path(
        self, partition_values: Sequence[str], batch_key: str, page_index: int
    ) -> Path:
        """`{raw_root}/{k1}={v1}/.../part-{batch_key}-{page_index:05d}.pqt`.

        DETERMINISTIC by construction -- no timestamp, no uuid, no counter.
        That determinism is what makes the crash window between the shard write
        and the ledger record cheap: a re-fetched page OVERWRITES its shard
        rather than adding a second one, so the cost of the window is a
        re-fetch and never a duplicated row (D-19 contract 4).
        """
        directory = Path(self.config.raw_data_dir_path)
        for key, value in zip(self._hive_keys, partition_values):
            directory = directory / f"{key}={value}"
        return directory / f"part-{batch_key}-{page_index:05d}.pqt"

    def _write_shard(
        self, frame: pl.DataFrame, batch_key: str, page_index: int
    ) -> list[str]:
        """Write one page's rows as one parquet file per hive partition value.

        Projects through `frame.select(self.RAW_COLUMNS)` FIRST, so every shard
        under a vendor root carries exactly the same columns in exactly the
        same order (see `RAW_COLUMNS`). Writes ONLY parquet beneath the raw
        root -- nothing else may ever land there, because a directory scan
        walks every file it finds and a stray `.json` breaks it outright.

        Returns the written paths, which the ledger records so a resume can
        cross-check itself against the disk.
        """
        if frame.is_empty():
            return []

        projected = frame.select(self.RAW_COLUMNS)
        partitioned = self._hive_partition_values(projected)
        keys = list(self._hive_keys)

        written: list[str] = []
        for values, group in partitioned.group_by(keys, maintain_order=True):
            values = tuple(str(value) for value in values)
            path = self._shard_path(values, batch_key, page_index)
            path.parent.mkdir(parents=True, exist_ok=True)
            group.drop(keys).write_parquet(path)
            written.append(str(path))
        return written

    # -- batching and pagination -------------------------------------------

    def _batches(self, symbols: Sequence[str]) -> Iterator[list[str]]:
        """Consecutive chunks of `batch_size` symbols."""
        size = max(1, int(self._knob("batch_size", self.DEFAULT_BATCH_SIZE)))
        symbols = list(symbols)
        for index in range(0, len(symbols), size):
            yield symbols[index : index + size]

    def _ledger_for(
        self, symbols: Sequence[str], start_date: str, end_date: str
    ) -> tuple[PageLedger, str]:
        """The `(ledger, batch_key)` pair identifying one batch."""
        batch_key = PageLedger.batch_key(
            self.VENDOR,
            self.config.frequency,
            start_date,
            end_date,
            symbols,
        )
        ledger = PageLedger(
            PageLedger.default_path(self.config.watermark_path, batch_key),
            symbols=symbols,
        )
        ledger.describe(
            batch_key,
            self.VENDOR,
            self.config.frequency,
            start_date,
            end_date,
            symbols,
        )
        return ledger, batch_key

    def _fetch_batch(
        self,
        symbols: Sequence[str],
        start_date: str,
        end_date: str,
        ledger: PageLedger | None = None,
        batch_key: str | None = None,
    ) -> BatchOutcome:
        """Fetch one batch to completion, page by page, resuming where a
        previous run stopped.

        **The ONLY place pagination is implemented.** Every vendor implements
        `_fetch_page` -- one request, one page -- and inherits this loop. A
        vendor with no pagination returns `None` on every call, which
        terminates the loop after one iteration and is a COMPLETE
        implementation of the contract rather than a stub.

        **Shard write strictly BEFORE the ledger record.** A crash between the
        two costs a re-fetch and a deterministic OVERWRITE -- never a
        duplicated row, and never a lost page. The reverse order would record
        pages whose data is not on disk, leaving a hole no later read could
        detect.

        The exception from `_fetch_page` PROPAGATES, but only after every page
        that DID complete has been flushed to the ledger. Swallowing it would
        report success for a short batch; flushing after the fact rather than
        before is what turns the next run into a resume instead of a restart.
        """
        symbols = self._validate_symbols(symbols)
        if ledger is None or batch_key is None:
            ledger, batch_key = self._ledger_for(symbols, start_date, end_date)

        # The ledger and the shard tree are independent records of the same
        # truth. Refuse to resume onto a disagreement rather than skipping a
        # hole (T-03.2-14).
        ledger.assert_consistent(self.config.raw_data_dir_path)

        if ledger.is_complete():
            # A COMPLETED ledger is not a veto. Reaching `_fetch_batch` at all
            # means the caller's symbol-level policy already decided this batch
            # should be fetched -- `download(resume=False)` most obviously.
            # The page ledger answers "where within this batch do I resume",
            # never "should this batch run"; those are D-05's two separate
            # layers, and letting the within-batch record override the
            # symbol-level one would silently ignore a knob the user set.
            #
            # Starting over is cheap and safe because shard filenames are
            # deterministic: page N writes the same path and OVERWRITES it.
            ledger.reset()

        outcome = BatchOutcome(
            symbols=tuple(symbols),
            symbols_with_data=set(ledger.symbols_seen()),
            pages=len(ledger.pages),
            complete=ledger.is_complete(),
        )

        page_index, page_token = ledger.resume_point()
        while True:
            frame, next_token = self._fetch_page(
                symbols, start_date, end_date, page_token
            )
            shards = self._write_shard(frame, batch_key, page_index)

            seen: set[str] = set()
            last_symbol = last_timestamp = None
            if not frame.is_empty():
                seen = {
                    str(value)
                    for value in frame.get_column("symbol").unique().to_list()
                }
                tail = frame.sort(by=["symbol", "timestamp"]).tail(1)
                last_symbol = str(tail.get_column("symbol").item())
                last_timestamp = str(tail.get_column("timestamp").item())

            ledger.record_page(
                page_index,
                next_token,
                frame.height,
                seen,
                shards,
                last_symbol=last_symbol,
                last_timestamp=last_timestamp,
            )
            outcome.symbols_with_data |= seen
            outcome.pages += 1

            if not next_token:
                ledger.mark_complete()
                outcome.complete = True
                return outcome

            page_index += 1
            page_token = next_token

    # -- entry points -------------------------------------------------------

    def download(self, symbols: list[str] | None = None) -> Self:
        """Full backfill over `[config.start_date, config.end_date]`.

        SEQUENTIAL by design at this layer: a handful of symbols needs no
        fan-out, and the concurrency plus quota-abort lift is 03.2-03's job.

        Watermark semantics are UNCHANGED from the pre-batch loop: a full
        backfill overwrites the batch's shards wholesale for the requested
        range, so recording `config.start_date` as the covered start is a TRUE
        statement about what is on disk -- including when the window was
        narrowed (260906-26o D-03).
        """
        requested = list(symbols or self.config.symbols)
        for batch in self._batches(requested):
            self._fetch_batch(
                batch,
                start_date=self.config.start_date,
                end_date=self.config.end_date,
            )
            for symbol in batch:
                self._write_watermark(
                    symbol,
                    self.config.end_date,
                    start_date=self.config.start_date,
                )
        return self

    def refresh(self, symbols: list[str] | None = None) -> Self:
        """Incremental fetch, each symbol starting at its OWN watermark.

        Batching is per-symbol here regardless of `DEFAULT_BATCH_SIZE`, because
        every symbol has a different start date and one request carries one
        `start`. Grouping symbols with unequal starts into one request would
        silently re-fetch history for some and under-fetch for others.

        Refresh fetches from each symbol's last covered date FORWARD, so the
        covered start is whatever it already was -- and if it was unknown it
        STAYS unknown. Refresh never invents coverage it did not fetch (D-06 /
        260906-26o D-04).
        """
        for symbol in symbols or list(self.config.symbols):
            coverage = self._read_coverage(symbol) or {}
            start = coverage.get("last_date") or self.config.start_date
            self._fetch_batch(
                [symbol], start_date=start, end_date=self.config.end_date
            )
            self._write_watermark(
                symbol,
                self.config.end_date,
                start_date=coverage.get("start_date"),
            )
        return self

    @abstractmethod
    def _fetch_page(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str,
        page_token: str | None = None,
    ) -> tuple[pl.DataFrame, str | None]:
        """Issue ONE vendor request and return `(rows, next_page_token)`.

        The single abstract fetch primitive. `rows` is a `pl.DataFrame`
        carrying the vendor's raw columns plus `timestamp`, `symbol` and
        `vendor`, projected and ordered to `self.RAW_COLUMNS`. A `None` token
        means this was the LAST page for `symbols`.

        **A vendor with no pagination returns `None` on every call, and that is
        a COMPLETE implementation of this contract, not a stub.** Tiingo's EOD
        endpoint returns a whole date range in one response; returning
        `(frame, None)` is the honest expression of that, and `_fetch_batch`'s
        loop terminates after one iteration.

        Implementations fetch raw vendor data and hand it back for
        `_write_shard` to persist under `self.config.raw_data_dir_path`, in the
        schema the matching `Dataset` subclass's `_raw_data_to_xr()` expects.
        Must not touch xarray/Zarr storage directly.
        """
        ...
