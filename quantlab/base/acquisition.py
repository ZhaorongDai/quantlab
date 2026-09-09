import os
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Self, Sequence

import polars as pl
from joblib import Parallel, delayed
from loguru import logger

from quantlab.base.config import AcquisitionConfig
from quantlab.base.coverage import (
    DEFAULT_LEGACY_WATERMARK_POLICY as _DEFAULT_LEGACY_WATERMARK_POLICY,
)
from quantlab.base.coverage import (
    FAILURE_MANIFEST_NAME as _FAILURE_MANIFEST_NAME,
)
from quantlab.base.coverage import (
    LEGACY_WATERMARK_POLICIES as _LEGACY_WATERMARK_POLICIES,
)
from quantlab.base.coverage import CoverageLedger
from quantlab.base.pageledger import PageLedger
from quantlab.base.progress import (
    QUOTA_EXHAUSTED_DESCRIPTION,
    CancelToken,
    NullProgressReporter,
    ProgressEvent,
    ProgressReporter,
    TqdmProgressReporter,
)
from quantlab.enums.constant import Date
from quantlab.enums.data import RAW_HIVE_KEYS, TRADEABLE_TICKER_PATTERN, Vendor
from quantlab.utils.atomic import write_json_atomically

#: The one well-formedness rule a symbol must satisfy before it becomes a
#: filesystem path segment or a query-string value.
#:
#: BOUND, not re-declared. This name is the SAME compiled object that the
#: roster builder (`acquisition/universe.py:TiingoRosterFetcher.fetch`) filters
#: on, which is what makes "everything the builder persists is fetchable" true
#: by construction rather than by coincidence. Identity is asserted directly in
#: `tests/test_ticker_pattern_reconciliation.py`.
#:
#: It is shared VIA `quantlab/enums/data.py` because neither binder may import
#: the other: an acquisition-base-to-universe import inverts the layering, and
#: a universe-to-acquisition-base import breaks
#: `tests/test_volume_guard.py`, which resolves universe.py's imports with
#: `ast` and asserts none of them is an acquisition module (the volume guard's
#: "refuse before any client exists" property is structural, not procedural).
#:
#: Until quick task 260907-10t this was a standalone `re.compile` of the same
#: literal, with a comment claiming it was imported from
#: `quantlab/acquisition/universe.py` and a deferred local import that did not
#: exist.
#: Two free-to-diverge copies -- and they HAD diverged, which is the whole bug
#: 260907-10t fixed.
#:
#: The GUARD that consults it moved to `quantlab/base/coverage.py`
#: (`CoverageLedger.validate_symbols`) in 03.4-04, so that a credential-free
#: reader validates a caller-supplied symbol through the same code path the
#: real run does. This binding stays here, unchanged and pointing at the same
#: object, because it is this module's provenance anchor and because
#: `tests/test_ticker_pattern_reconciliation.py` asserts its IDENTITY against
#: `enums.data.TRADEABLE_TICKER_PATTERN` -- `coverage.py` binds the very same
#: object, so there is still exactly one compiled pattern in the process.
#:
#: DELIBERATELY WIDER than
#: `quantlab/acquisition/universe.py:_WELL_FORMED_TICKER`, which
#: is a different guard on a different input: that one validates
#: Wikipedia-scraped change-log CELLS, where an interior delimiter means two
#: cells were merged by a parser regression. A three-segment value is that
#: exact regression signal there, and both constituent categories have ZERO
#: pattern failures across 1,151 measured symbols -- so widening it would
#: delete a real guard to buy nothing. Do not "align" the two.
_TICKER_PATTERN = TRADEABLE_TICKER_PATTERN


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


@dataclass(frozen=True)
class AcquisitionResult:
    """What ONE programmatic run did -- the in-process caller's copy of the
    outcome (03.4 D-18).

    Two outputs on purpose. This object means the caller never has to read disk
    to know what happened, while `_failures.json` still lands on disk because a
    crashed process returns nothing. They are built from the SAME accumulated
    `failures` dict at the SAME point in `_run`, so they cannot disagree.

    **Defined HERE, in `base/`, rather than in
    `quantlab/acquisition/registry.py`**, and the direction is what matters:
    the base layer must not import the acquisition package, so a result type
    living beside the registry would have to be imported backwards (or
    duplicated). `registry.py` imports it from here instead -- one definition,
    no cycle.

    **`failures` values are ALREADY SCRUBBED.** They are the same strings
    `_attempt_batch` produced through `_scrub`, never raw vendor exception
    text -- which for Tiingo echoes back a request URL carrying the API token
    as a query parameter. A new egress path for exception text that skipped
    that choke point is exactly how the next leak happens.

    `coverage` is `coverage_report()`'s shape, computed after the run, so a
    caller can render "what is on disk now" without a second traversal.
    """

    vendor: str
    requested: tuple[str, ...]
    succeeded: tuple[str, ...]
    failures: dict[str, str]
    cancelled: bool
    quota_aborted: bool
    coverage: dict


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
    additive -- `{"start_date": ..., "last_date": ..., "no_data": ...}`, with
    `last_date` keeping its original name -- so old and new readers each
    tolerate the other's files. An unknown covered start is represented by the
    key being ABSENT and is never guessed; see `_read_coverage` and
    `stamp_watermarks`.

    **`no_data` is the third key and the fourth read-time state** (03.2 D-04,
    SC-4). It means "the vendor was ASKED about this symbol over the recorded
    window and returned nothing", which is a different fact from "the fetch
    failed" and from "this symbol was never fetched". Three storage facts
    yield four states::

        never fetched         no sidecar, no manifest entry
        fetch failed          no sidecar, PLUS an entry in `_failures.json`
        fetched, data landed  sidecar, `no_data` ABSENT
        queried, no data      sidecar, `no_data` present and true

    Like `start_date`, the key is OMITTED when false rather than written as
    `false`, so absence is the default and every sidecar written before this
    phase reads back correctly as not-no-data -- true because the old code
    only ever wrote a watermark after a successful fetch. The guarantee runs
    in BOTH directions: an old reader ignores the key (`_read_watermark` is
    untouched and still returns the same `last_date`), and a new reader
    tolerates its absence.
    """

    def __init__(self, config: AcquisitionConfig):
        self.config = config
        #: The outcome of the most recent `download()`/`refresh()`, or `None`
        #: before either has run (03.4 D-18).
        #:
        #: INSTANCE state, exactly like `_abort_event` and `_no_data_marks`,
        #: and deliberately NOT on the config: `AcquisitionConfig.to_dict()` is
        #: `asdict(self)` and lands on disk beside model checkpoints, so a run
        #: outcome parked there would be persisted as if it were reproducible
        #: configuration.
        self.last_result: AcquisitionResult | None = None
        #: Where progress events go, and how a caller stops the run (03.4
        #: D-16/D-17). INSTANCE state, exactly like `last_result` and
        #: `_abort_event` -- see `attach()` for why neither may live on the
        #: config.
        self._reporter: ProgressReporter | None = None
        self._cancel_token: CancelToken | None = None
        self._resolved_reporter: ProgressReporter | None = None

    def attach(
        self,
        *,
        reporter: ProgressReporter | None = None,
        cancel: CancelToken | None = None,
    ) -> Self:
        """Attach a progress reporter and/or a cancel token for the next run.

        Returns `self`, so it chains the way the rest of this repo does:
        `TiingoAcquisition(config).attach(reporter=r, cancel=t).download()`.

        **Neither value may be assigned onto `AcquisitionConfig`, and that is a
        rule rather than a preference.** `AcquisitionConfig.to_dict()` is
        `asdict(self)` and the result lands on disk in persisted configs and in
        the JSON metadata saved beside model checkpoints
        (`base/model.py:_save_model`). A `threading.Event` is not serialisable
        at all, and a reporter is not reproducible CONFIGURATION -- it is a
        live object belonging to whoever started the run. Parking either there
        would persist a run's plumbing as if it were an experiment's
        parameters. This is the one sanctioned exception to the house rule that
        every knob rides `config.kwargs` via `_knob`.

        Passing `None` for either argument CLEARS it, so a caller can hand the
        same acquisition object to a second, unobserved run without inheriting
        the first run's reporter.
        """
        self._reporter = reporter
        self._cancel_token = cancel
        # The resolved default is derived from `_reporter` and the `progress`
        # knob, so it has to be discarded whenever either could have changed.
        self._resolved_reporter = None
        return self

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

    @property
    def _coverage(self) -> CoverageLedger:
        """The composed `CoverageLedger` every coverage question routes
        through (03.4 D-09).

        Built PER ACCESS rather than cached in the config setter, and the
        reason is `_data_type`: `AlpacaAcquisition` overrides it as an
        INSTANCE property reading `_knob("data_type")` with NO default, so
        evaluating it eagerly at config-assignment time would raise for a tick
        config whose knob is supplied afterwards -- where today
        `_watermark_root` only evaluates it lazily, at the moment a path is
        actually needed. Per-access construction is a five-attribute object and
        preserves the current evaluation timing exactly.

        `LEGACY_WATERMARK_POLICIES` / `DEFAULT_LEGACY_WATERMARK_POLICY` are
        passed as VALUES so a subclass that narrows them still governs its own
        ledger.
        """
        return CoverageLedger(
            self.config,
            data_type=self._data_type,
            legacy_policies=self.LEGACY_WATERMARK_POLICIES,
            default_legacy_policy=self.DEFAULT_LEGACY_WATERMARK_POLICY,
            owner_label=self.class_name,
        )

    @property
    def _watermark_root(self) -> Path:
        """Delegates to `CoverageLedger.watermark_root` -- see there for the
        tick `data_type` namespacing and why it exists.
        """
        return self._coverage.watermark_root

    def _watermark_path(self, symbol: str) -> Path:
        """Delegates to `CoverageLedger.watermark_path`."""
        return self._coverage.watermark_path(symbol)

    def _read_sidecar(self, symbol: str) -> dict | None:
        """Delegates to `CoverageLedger.read_sidecar` -- the one tolerant
        sidecar read, and the one failure policy for a corrupt one.
        """
        return self._coverage.read_sidecar(symbol)

    def _read_watermark(self, symbol: str) -> str | None:
        """Delegates to `CoverageLedger.read_watermark` -- the LAST covered
        date for `symbol`, or None.
        """
        return self._coverage.read_watermark(symbol)

    def _read_coverage(self, symbol: str) -> dict | None:
        """Delegates to `CoverageLedger.read_coverage` -- the covered RANGE as
        `{"start_date", "last_date", "no_data"}`, or None.
        """
        return self._coverage.read_coverage(symbol)

    def _write_watermark(
        self,
        symbol: str,
        last_date: str,
        start_date: str | None = None,
        no_data: bool = False,
    ) -> None:
        """Record coverage for `symbol`.

        The schema is purely ADDITIVE: `last_date` keeps its name and meaning,
        so new code reads pre-26o files and pre-26o code reads new files, and
        no reader anywhere crashes on either.

        `start_date=None` omits the key ENTIRELY rather than writing a null --
        an unknown covered start is represented by absence, so it cannot be
        mistaken at read time for a recorded value.

        `no_data=False` omits its key for the same reason, and the reasoning
        matters more here because the default is what OLD files inherit.
        Omitting makes absence the default, so every sidecar written before
        this phase reads back correctly as not-no-data -- true because the old
        code only ever wrote a watermark after a successful fetch. Writing
        `false` would instead put a positive claim in new files that older
        files cannot make, leaving those older files ambiguous between "had
        data" and "never said" (D-04, RESEARCH Pattern 4).

        Callers must have earned the `True`: see `_attempt_batch`, where the
        marker is computed once per COMPLETED batch and never per page.

        The write is ATOMIC (D-20). That matters specifically here because
        D-17's cancellation lands at a BATCH BOUNDARY, which is exactly when
        this file is being written, and the interrupted file is the one the
        RESUMED run reads -- a plain `open(path, "w")` truncates a good
        watermark before the first byte of the new one is written.
        """
        path = self._watermark_path(symbol)
        payload: dict[str, str | bool] = {"last_date": last_date}
        if start_date is not None:
            payload["start_date"] = start_date
        if no_data:
            payload["no_data"] = True
        #: BLAST RADIUS, bounded (D-20 analysis): `_read_sidecar` already
        #: tolerates a corrupt sidecar by returning None, which degrades to
        #: "uncovered" and a wider-than-necessary re-fetch -- never to
        #: corrupted data. So the atomic write removes noise and a false
        #: re-fetch, not a data-integrity hole. Do NOT tighten that tolerance
        #: into a raise on the grounds that writes are now atomic: files
        #: written before this phase are still on disk, and turning "degrade
        #: and re-fetch" into "crash the run" would punish operators for the
        #: defect this change just fixed.
        #: Compact, no `indent` -- unchanged from before the extraction, so
        #: every sidecar already on disk is byte-comparable with a new one.
        write_json_atomically(path, payload)

    def stamp_watermarks(self, start_date: str) -> int:
        """Fill the covered start into every sidecar that lacks one, and
        return how many files changed.

        The explicit, user-supplied migration D-04 requires. It takes the
        start from its CALLER and derives it from nothing -- not from
        `config.start_date`, not from a default. A sidecar that already
        records a start is left untouched, because overwriting a recorded
        range with a guessed one is the same silent-wrong-data failure in a
        different costume (T-26o-04).

        It fills the START and nothing else. An existing `no_data` marker is
        CARRIED THROUGH verbatim: stamping is a statement about which window a
        file covers, and it has no evidence at all about whether the vendor
        had rows in it. Dropping the marker here would silently downgrade a
        confirmed absence to "fetched, data landed".

        Issues zero network requests.
        """
        directory = self._watermark_root
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
                symbol,
                coverage["last_date"],
                start_date=start_date,
                no_data=coverage["no_data"],
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

    #: The IANA time zone whose calendar day the INTRADAY `date=` hive key is
    #: derived from -- the trading session's own day boundary (D-19 contract 7).
    #:
    #: `None` means "not declared", and `_session_date` RAISES on it rather
    #: than falling back to a plain date truncation. Only intraday frequencies
    #: read it, so a daily-only vendor never has to set it.
    #:
    #: Timestamp VALUES are unaffected: every raw shard keeps naive UTC, the
    #: same convention every other timestamp in this codebase follows. ONLY the
    #: derived partition key converts.
    SESSION_TIME_ZONE: str | None = None

    # -- orchestration constants (hoisted from ConcurrentTiingoAcquisition
    #    by 03.2-03 / D-02: ONE implementation drives every vendor) ---------

    #: Concurrent in-flight batch fetches. Overridable per run via
    #: `config.kwargs["max_workers"]`.
    DEFAULT_MAX_WORKERS = 8

    #: Written under `config.watermark_path` alongside the per-symbol
    #: watermark sidecars, because "what did and did not land" is exactly the
    #: same question those sidecars answer (T-0iy-07).
    #:
    #: BOUND from `quantlab/base/coverage.py`, where it is DECLARED, so the
    #: writer here and the credential-free reader in
    #: `quantlab/acquisition/inspector.py` name the same file by construction
    #: rather than by two literals that agree today. Every existing
    #: `self.FAILURE_MANIFEST_NAME` reference and every log message that
    #: interpolates it keeps working unchanged.
    FAILURE_MANIFEST_NAME = _FAILURE_MANIFEST_NAME

    #: What a credential value is replaced with in any captured message.
    #: Subclasses override it with vendor-specific wording; the base value is
    #: what a vendor that names no credentials would use.
    REDACTION = "<CREDENTIAL REDACTED>"

    #: The environment variables whose VALUES `_scrub` redacts, per vendor.
    #:
    #: The per-vendor hook of the one shared scrubbing choke point: adding a
    #: vendor cannot forget to redact, because forgetting means declaring an
    #: empty tuple rather than silently inheriting a Tiingo-shaped rule that
    #: does not apply (T-03.2-01).
    #:
    #: A subclass MUST populate this from module-level constants, never by
    #: reaching through its transport/client class. A test double replaces the
    #: transport wholesale, so a security control routed through it could be
    #: silently disabled by substituting a stub that happens not to define the
    #: names (03.2-02 deviation #2).
    CREDENTIAL_ENV_VARS: tuple[str, ...] = ()

    #: How many failed units are named in the summary log line. The full set
    #: always lands in the manifest; the log is a pointer, not a dump.
    _FAILURE_LOG_SAMPLE = 5

    #: What `config.kwargs["legacy_watermarks"]` may be set to (260906-26o
    #: D-04). `"warn"` skips a sidecar with no recorded covered start but
    #: reports it on every run; `"refetch"` treats unknown coverage as
    #: uncovered. See `_coverage_status` for why `"warn"` is the default.
    #:
    #: BOUND from `quantlab/base/coverage.py` for the same single-definition
    #: reason as `FAILURE_MANIFEST_NAME`: `CoverageLedger.for_config` -- the
    #: constructor the credential-free inspector uses -- needs the same policy
    #: set the real run uses and cannot import this class to get it. They stay
    #: CLASS attributes here because `ingest_us_equity.py` reads
    #: `TiingoAcquisition.LEGACY_WATERMARK_POLICIES` for its argparse choices,
    #: and because a subclass may still narrow them (`_coverage` passes
    #: whatever this class declares INTO the ledger it composes).
    LEGACY_WATERMARK_POLICIES = _LEGACY_WATERMARK_POLICIES
    DEFAULT_LEGACY_WATERMARK_POLICY = _DEFAULT_LEGACY_WATERMARK_POLICY

    #: The one command that resolves an un-stamped legacy watermark. Named
    #: verbatim in the warning, because a reported gap with no named cure is
    #: only marginally better than a silent one.
    STAMP_COMMAND_HINT = (
        "uv run python ingest_us_equity.py --stamp-legacy-watermarks <START_DATE>"
    )

    #: Wait-and-resume is OFF unless asked for, so no run silently holds a
    #: vendor's allocation window open (D-06).
    DEFAULT_WAIT_FOR_QUOTA = False

    #: Delay between resume attempts. Tiingo's reset semantics -- fixed
    #: top-of-hour bucket vs. rolling window -- are NOT established, so this
    #: is a configurable INTERVAL, not a computed resume instant. One hour
    #: measured from the moment of detection covers a rolling one-hour window
    #: exactly and a fixed top-of-hour bucket strictly. Assumption, not a
    #: vendor fact.
    DEFAULT_QUOTA_WAIT_SECONDS = 3600

    #: Bounded, because an unbounded loop against a lockout is a worse version
    #: of the problem this mechanism is fixing. 3 comes from the observed
    #: arithmetic: ~4,600 requests per window against 14,674 symbols is
    #: roughly three windows.
    DEFAULT_QUOTA_MAX_WAITS = 3

    #: HTTP statuses this vendor treats as a TRANSIENT rate limit -- back off
    #: inside the worker, retry the same batch, never touch the global abort.
    #:
    #: Empty by default: the base makes no claim about any vendor's status
    #: codes, and a vendor with no per-minute ceiling says so by leaving this
    #: empty rather than by inheriting someone else's numbers. The asymmetry
    #: this exists for: Alpaca's 429 is a 200-per-MINUTE ceiling a healthy
    #: full-market run is expected to hit and that clears in under a minute,
    #: while Tiingo's 429 is HOURLY allocation exhaustion whose correct
    #: response is to stop the world (03.2-RESEARCH.md Pitfall 1).
    RATE_LIMIT_STATUS_CODES: frozenset[int] = frozenset()

    #: How long a `rate_limited` batch waits before retrying.
    #:
    #: Alpaca's ceiling is expressed per MINUTE, so a wait comfortably inside
    #: one minute is enough for the window to roll over, and a full minute
    #: would idle every worker far longer than the limit actually lasts. 5s is
    #: a working value rather than a vendor fact: Alpaca publishes
    #: `X-RateLimit-Reset` but does not guarantee it on every response, so this
    #: is the floor the code falls back to and never a computed reset instant.
    #: Overridable per run via `config.kwargs["rate_limit_backoff_seconds"]`.
    DEFAULT_RATE_LIMIT_BACKOFF_SECONDS = 5.0

    #: How many consecutive backoffs one batch gets before it degrades to
    #: `failed` (T-03.2-17).
    #:
    #: Bounded because an unbounded retry loop against a rate limit is a worse
    #: version of the problem the backoff solves -- a run that never finishes
    #: and never reports. At the 5s default this is ~30s of patience, which
    #: covers a per-minute window rolling over at least twice; anything
    #: surviving that is not a transient and belongs in the manifest where the
    #: next run will retry it. Overridable via
    #: `config.kwargs["rate_limit_max_retries"]`.
    DEFAULT_RATE_LIMIT_MAX_RETRIES = 6

    def _knob(self, name: str, default=None):
        """Read a per-run tuning parameter from `config.kwargs`.

        The escape hatch `AcquisitionConfig` documents -- a knob read here
        never becomes a constructor argument no config file could reach, which
        is what keeps the whole pipeline config-driven per CLAUDE.md.
        """
        return (self.config.kwargs or {}).get(name, default)

    # -- global abort, backoff seam and credential scrubbing ----------------

    @property
    def _abort(self) -> threading.Event:
        """The global stop flag, shared across every worker thread.

        `threading.Event` is thread-safe by construction, so no surrounding
        lock is needed, and its `wait(timeout)` is exactly the primitive the
        resume delay wants. Created lazily so `_attempt_batch` is safe to call
        directly (tests do) without a `_run` having set one up.
        """
        event = getattr(self, "_abort_event", None)
        if event is None:
            event = self._abort_event = threading.Event()
        return event

    def _reset_abort(self) -> threading.Event:
        """A FRESH event per resume pass, so a previous pass's trip cannot
        poison the next one.
        """
        self._abort_event = threading.Event()
        return self._abort_event

    def _sleep(self, seconds: float) -> None:
        """Overridable seam: tests substitute it and assert call counts
        instead of waiting an hour.
        """
        time.sleep(seconds)

    def _is_cancelled(self) -> bool:
        """Has the CALLER asked this run to stop? (D-17.)

        Deliberately narrow: this asks about the cancel token ALONE and says
        nothing about the vendor quota abort. `_should_stop()` is the OR of the
        two and is what the batch boundary checks; keeping this one separate is
        what lets the loop report "the operator stopped this" distinctly from
        "the vendor stopped this" (RESEARCH Pitfall 1).
        """
        token = getattr(self, "_cancel_token", None)
        return token is not None and token.is_cancelled()

    def _should_stop(self) -> bool:
        """Should this batch not be attempted?

        The OR of the two independent stop conditions -- the vendor's quota
        abort and the caller's cancel token -- and the single expression the
        batch boundary checks. They are ORed for the decision and kept SEPARATE
        for the reporting: `_is_cancelled()` and `_abort.is_set()` still answer
        "which one", which is what stops a cancel from being logged, waited on,
        or resumed as if the vendor had run out of allocation (D-17, RESEARCH
        Pitfall 1).
        """
        return self._abort.is_set() or self._is_cancelled()

    # -- progress reporting (03.4 D-16) -------------------------------------

    @property
    def _active_reporter(self) -> ProgressReporter:
        """The reporter this run's events actually go to.

        Resolution order, and the middle branch is what keeps the incumbent
        stderr rendering the DEFAULT rather than making silence the default:

        1. whatever `attach()` was given, else
        2. a `TqdmProgressReporter` when `_knob("progress", True)` is truthy,
           else
        3. a `NullProgressReporter`.

        Note what branch 3 means: `progress=False` now resolves to a reporter
        that constructs no bar at all, where the replaced code constructed a
        `tqdm` with `disable=True`. Both render nothing, so the operator-visible
        behaviour is identical; not building the object is simply the honest
        expression of it.

        CACHED, because the default is stateful -- `TqdmProgressReporter` owns
        the bar across the events of a pass, so re-resolving per event would
        open a new bar per batch. `attach()` invalidates the cache.
        """
        resolved = getattr(self, "_resolved_reporter", None)
        if resolved is not None:
            return resolved
        attached = getattr(self, "_reporter", None)
        if attached is not None:
            resolved = attached
        elif self._knob("progress", True):
            resolved = TqdmProgressReporter()
        else:
            resolved = NullProgressReporter()
        self._resolved_reporter = resolved
        return resolved

    def _emit(self, event: ProgressEvent) -> None:
        """Deliver one event, never raising.

        03.4-RESEARCH Pitfall 9: the console's callback runs INSIDE
        `_run_once`'s result loop. A callback that throws would propagate out of
        that loop and tear down the joblib fan-out -- ending a multi-hour
        backfill over a UI bug. That is precisely the failure mode
        `_attempt_batch`'s "Never raises" contract exists to prevent,
        reintroduced one level up, so the same contract is applied here.

        The exception is LOGGED at warning rather than swallowed silently: a
        reporter bug that produced no signal at all would be invisible, and an
        invisible broken console is worse than a noisy one (RESEARCH A6). It is
        scrubbed on the way out for the same reason every other captured string
        is -- the message belongs to caller code this module does not control.
        """
        try:
            self._active_reporter.emit(event)
        except Exception as exc:  # noqa: BLE001 -- isolation is the point
            logger.warning(
                self._scrub(
                    f"Progress reporter {type(self._active_reporter).__name__} "
                    f"raised on a {event.kind!r} event and was ignored; the "
                    f"run is unaffected. {type(exc).__name__}: {exc}"
                )
            )

    def _close_reporter(self) -> None:
        """Release the active reporter's resources, never raising.

        Same contract as `_emit` and for the same reason: `tqdm.close()` writes
        to stderr, and a closed/redirected stream must not be what ends a run.
        """
        try:
            self._active_reporter.close()
        except Exception as exc:  # noqa: BLE001 -- isolation is the point
            logger.warning(
                self._scrub(
                    f"Progress reporter close() raised and was ignored. "
                    f"{type(exc).__name__}: {exc}"
                )
            )

    def _reset_no_data_marks(self) -> None:
        """Zero this pass's `no_data` tally, and give it a fresh lock.

        Per PASS, like `_reset_abort`, so a resumed pass reports the markers
        IT wrote rather than inheriting the previous pass's number.
        """
        self._no_data_marks = 0
        self._no_data_lock = threading.Lock()

    def _record_no_data_marks(self, count: int) -> None:
        """Accumulate `count` markers written by one batch.

        Reporting only -- nothing branches on this. The lock exists because
        `_attempt_batch` runs on `max_workers` threads; when it is absent the
        method was reached without a `_run_once` (tests call `_attempt_batch`
        directly), and there is no fan-out to race with.
        """
        if not count:
            return
        lock = getattr(self, "_no_data_lock", None)
        if lock is None:
            self._no_data_marks = getattr(self, "_no_data_marks", 0) + count
            return
        with lock:
            self._no_data_marks = getattr(self, "_no_data_marks", 0) + count

    def _scrub(self, message: str) -> str:
        """Remove every declared credential VALUE from a message before it is
        logged or written.

        This repo has already leaked one real Tiingo key. A vendor exception
        string is a path a credential travels that nobody audits -- an HTTP
        error commonly echoes back the full request URL, and Tiingo's carries
        the token as a query parameter. Scrubbing at the single choke point
        every captured message passes through is what makes the manifest safe
        to commit, paste into an issue, or ship to a log aggregator
        (T-0iy-01, T-03.2-01).

        The per-vendor part is DATA (`CREDENTIAL_ENV_VARS`), not code, so a
        new vendor inherits the control rather than reimplementing it -- and
        cannot reimplement it subtly differently.
        """
        for name in self.CREDENTIAL_ENV_VARS:
            value = os.environ.get(name)
            if value:
                message = message.replace(value, self.REDACTION)
        return message

    # -- error classification: the ONE per-vendor policy seam ---------------

    @staticmethod
    def _vendor_response(exc: BaseException):
        """The vendor `requests.Response` reachable from `exc`, or None.

        Measured, not assumed. Two `requests`-based vendor clients wrap their
        errors differently and the obvious one-liner is wrong for one of them:
        `tiingo/restclient.py:_request` catches the
        `requests.exceptions.HTTPError` and re-raises `RestClientError(e)`, so
        `RestClientError` has NO `.response` of its own and
        `getattr(exc, "response", None)` returns None every time -- the status
        lives at `exc.args[0].response.status_code`. Hence the walk over the
        exception AND its args, which also covers the direct `HTTPError` a
        plain `raise_for_status()` produces.

        Shared rather than duplicated per vendor: two independent walks of the
        same exception shape would eventually disagree, and the failure mode of
        the disagreement is a misclassified status, which is the single most
        expensive bug in this module.
        """
        for candidate in (exc, *getattr(exc, "args", ())):
            response = getattr(candidate, "response", None)
            if response is not None and getattr(response, "status_code", None):
                return response
        return None

    def _status_of(self, exc: BaseException) -> int | None:
        """The HTTP status reachable from `exc`, or None when there is none.

        DEFENSIVE by contract: an exception carrying no status is not an
        error here, it is the absence of information, and it must read as
        `None` rather than as a default. A status invented for an exception
        that has none would be classified with full confidence and be wrong.
        """
        response = self._vendor_response(exc)
        if response is None:
            return None
        status = getattr(response, "status_code", None)
        return int(status) if status else None

    def _rate_limit_headers(self, exc: BaseException) -> dict[str, str]:
        """Whatever rate-limit metadata this vendor sent, for LOGGING only.

        `{}` on the base, which is the honest answer for a vendor that
        publishes no such headers -- and it is a complete implementation, not a
        stub: `_attempt_batch` logs whatever it gets, so an empty mapping
        renders as "the vendor said nothing", which is a true statement.

        An ABSENT header must contribute no entry rather than a default one. A
        caller has to be able to distinguish "the vendor said nothing" from
        "the vendor said zero"; those mean opposite things, and a default would
        silently merge them. Nothing branches on this -- the backoff stays a
        configured constant, never a computed reset instant.
        """
        return {}

    def _classify_error(self, exc: BaseException) -> str:
        """How THIS vendor's failures map onto the shared orchestration.

        Returns one of:

        - `"failed"` -- PER-UNIT. Lands in the failure manifest, gets no
          watermark, is retried by the next run. Every other batch continues.
        - `"quota"` -- GLOBAL and SLOW to clear. Trips the shared abort Event,
          stopping dispatch for the whole run, and is deliberately EXCLUDED
          from the manifest: recording a global condition as one ticker's
          fault would defame a perfectly good symbol.
        - `"rate_limited"` -- TRANSIENT and FAST to clear. Backs off inside
          the worker and retries the same batch, without touching the global
          abort and without a manifest entry.

        The base default is the CONSERVATIVE one -- everything is per-unit --
        because a vendor whose failures were wrongly read as global would
        abort a 15,000-symbol run over one bad ticker. A vendor that really
        has a global or a transient condition says so by overriding, and the
        two vendors here read the SAME status code oppositely: Tiingo's 429 is
        hourly allocation exhaustion (`quota`), Alpaca's is a per-minute
        ceiling (`rate_limited`). That is why this method, and not the
        orchestration around it, is what varies (D-02, RESEARCH Pattern 2).
        """
        if self._status_of(exc) in self.RATE_LIMIT_STATUS_CODES:
            return "rate_limited"
        return "failed"

    # -- symbol validation --------------------------------------------------

    def _validate_symbols(self, symbols: Sequence[str]) -> list[str]:
        """Delegates to `CoverageLedger.validate_symbols` -- the rule, the
        pattern and the full rationale live there.

        The guard moved with the coverage extraction (03.4-04) so that the
        credential-free `SourceInspector` validates a caller-supplied symbol
        through the SAME code, before it builds any path. `owner_label` is
        `self.class_name`, so the rendered error text is byte-identical to its
        pre-extraction form.
        """
        return self._coverage.validate_symbols(symbols)

    # -- raw shard layout ---------------------------------------------------

    @property
    def _hive_keys(self) -> tuple[str, ...]:
        """The hive partition key(s) for this config's frequency.

        Read from `enums.data.RAW_HIVE_KEYS`, the SAME mapping
        `dataset/stock.py` reads, so the writer and the reader cannot drift.
        """
        return RAW_HIVE_KEYS[self.config.frequency]

    def _session_date(self, expr: pl.Expr) -> pl.Expr:
        """The OVERRIDABLE seam: a naive timestamp expression -> the SESSION
        date its intraday `date=` hive key is derived from (D-19 contract 7).

        This is a hook rather than a vendor `isinstance` branch on purpose. The
        base class must keep no knowledge of any concrete vendor, and the
        question this answers -- "in which calendar day's trading session does
        this instant fall?" -- is a property of the vendor's timestamp
        convention, not of the frequency.

        The default implementation converts from UTC into `SESSION_TIME_ZONE`
        and truncates. A vendor whose timestamps are ALREADY session-local
        overrides this method with a plain `expr.dt.date()` truncation, which
        is then a deliberate statement rather than a silent default.

        Declaring nothing RAISES. That is the whole point: a vendor whose
        timestamps are naive UTC and which never thought about the day boundary
        would, under a truncating default, file the last ~4 hours of every US
        session (20:00-24:00 UTC) under the FOLLOWING day. A one-trading-day
        query is then wrong at both edges, and it is wrong in the shape that
        reads as sparse data rather than as a bug -- the close missing, the
        previous session's tail present. Failing loudly at the first intraday
        write is enormously cheaper than discovering that after a backfill.
        """
        if self.SESSION_TIME_ZONE is None:
            raise NotImplementedError(
                f"{self.class_name}: SESSION_TIME_ZONE is unset, so there is no "
                f"way to derive the intraday `date=` hive key for frequency "
                f"{self.config.frequency!r}. The key is a SESSION date "
                f"(D-19 contract 7), not a UTC date: a UTC-derived key files "
                f"the last ~4 hours of every US session under the following "
                f"day and makes a one-trading-day query silently wrong at both "
                f"edges. Set SESSION_TIME_ZONE to this vendor's session time "
                f"zone, or -- if its timestamps are already session-local -- "
                f"override _session_date() with a plain date truncation."
            )
        return (
            expr.dt.replace_time_zone("UTC")
            .dt.convert_time_zone(self.SESSION_TIME_ZONE)
            .dt.date()
        )

    @property
    def _data_type(self) -> str | None:
        """Which of the vendor's data types this run fetches, or `None`.

        `None` on the base, because a vendor that offers exactly one shape of
        data per frequency has no such concept and must not be made to invent
        one. `_hive_key_expr` raises legibly if a frequency whose keys include
        `data_type` reaches it with nothing declared.

        A multi-data-type vendor overrides this and VALIDATES against its own
        accepted set on the way out. Which values are legitimate is the
        vendor's knowledge, not the base's -- and the same resolved value must
        drive the endpoint and the written projection, so that they cannot
        disagree about what a shard holds.
        """
        return None

    def _hive_key_expr(self, key: str) -> pl.Expr:
        """The expression that derives ONE hive key's value from a raw frame.

        Keyed by the key NAME rather than by frequency, so `enums.data`'s
        `RAW_HIVE_KEYS` stays the only place the per-frequency key TUPLES are
        declared and this method only has to know how each individual key is
        computed. Adding a frequency that reuses existing keys then needs no
        change here at all.
        """
        if key == "month":
            # `YYYY-MM` as a STRING, which is what the reader's `hive_schema`
            # pins. ISO ordering makes a plain string comparison against the
            # window edges correct, so the reader needs no date parsing and no
            # time zone can creep in (03.2-RESEARCH.md Pitfall 8: an unpinned
            # numeric-looking hive value is inferred as an integer and a string
            # comparison against it silently matches nothing).
            return pl.col("timestamp").dt.strftime("%Y-%m")
        if key == "date":
            # The intraday key, through the session-date seam above.
            return self._session_date(pl.col("timestamp"))
        if key == "symbol":
            # Already a real column on every raw frame; the hive key restates
            # it as a path segment. `_validate_symbols` has run before any
            # frame reaches here, so the value cannot escape the raw root
            # (T-03.2-03).
            return pl.col("symbol")
        if key == "data_type":
            # A per-RUN constant, not a per-row derivation: one fetch asks one
            # endpoint for one data type. Writing it as the LEADING key is what
            # keeps two different column sets from meeting inside one directory
            # scan, which would make the whole tick tier unreadable rather than
            # merely mixed (03.2-RESEARCH.md Pitfall 6).
            data_type = self._data_type
            if data_type is None:
                raise ValueError(
                    f"{self.class_name}: frequency "
                    f"{self.config.frequency!r} partitions on a `data_type=` "
                    f"hive key but this class resolves no data type. Override "
                    f"`_data_type` to return the one this run fetches -- it "
                    f"must be the SAME value that selected the endpoint, or a "
                    f"shard's directory name and its columns would describe "
                    f"different things."
                )
            return pl.lit(data_type)
        raise NotImplementedError(
            f"{self.class_name}: no derivation for hive key {key!r} "
            f"(frequency {self.config.frequency!r}, keys {self._hive_keys}). "
            f"Every key in enums.data.RAW_HIVE_KEYS must have one here, or the "
            f"writer and the reader would disagree about the tree's shape."
        )

    def _hive_partition_values(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Add the hive key column(s) this frequency partitions on, in the
        order `enums.data.RAW_HIVE_KEYS` declares them.

        The ORDER is load-bearing: it is the directory nesting order, and
        `_shard_path` zips it against the `group_by` tuple, so a reordering
        here silently relabels every path segment.
        """
        return frame.with_columns(
            self._hive_key_expr(key).alias(key) for key in self._hive_keys
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

        **The overwrite holds within ONE window and not across two.**
        `batch_key` hashes `start_date`/`end_date`, so the same rows re-fetched
        over a different window (every `refresh()`, whose inclusive `last_date`
        re-requests the final session) land under a DIFFERENT name in the same
        partition directory. `1d`/`1m` absorb that in `dedup_raw_frame`; tick
        never dedups, so `_write_shard` removes the superseded shards instead --
        see `_clear_superseded_shards`.
        """
        directory = Path(self.config.raw_data_dir_path)
        for key, value in zip(self._hive_keys, partition_values):
            directory = directory / f"{key}={value}"
        return directory / f"part-{batch_key}-{page_index:05d}.pqt"

    #: Whether a partition directory is scoped to ONE symbol, which is what
    #: makes `_clear_superseded_shards` sound. Derived from the hive keys
    #: rather than from the frequency, so a future layout that gains or loses
    #: the `symbol=` key gets the right answer without a second edit.
    @property
    def _partition_is_per_symbol(self) -> bool:
        return "symbol" in self._hive_keys

    def _clear_superseded_shards(self, directory: Path, batch_key: str) -> None:
        """Delete shards in `directory` written by a DIFFERENT batch key.

        Shard filenames are deterministic, but their determinism is scoped to
        ONE `(vendor, frequency, start, end, symbols)` tuple: `batch_key` hashes
        the window, so the SAME rows re-fetched over a DIFFERENT window land
        under a second filename in the same partition directory rather than
        overwriting the first. `refresh()` does exactly that on every run --
        `last_date` is inclusive, so the final session is re-requested with a
        new start and therefore a new key.

        For `1d`/`1m` that is absorbed downstream: `dataset/stock.py` runs
        `dedup_raw_frame(keep="last")`. **Tick deliberately does not dedup**
        (D-16) -- genuine quotes and trades legitimately share
        `(timestamp, symbol)` -- and that correct decision is exactly what makes
        the duplicate undetectable afterwards: a doubled trade tape is
        indistinguishable from a busy one, and every volume, VWAP and
        microstructure statistic computed off it is silently wrong forever.

        **Only sound when the partition is per-symbol**, hence the gate at the
        call site. A `1d` (`month=`) or `1m` (`date=`) directory is shared by
        every concurrently-running batch, so deleting another key's file there
        would destroy a sibling batch's data. A tick directory is
        `data_type=/date=/symbol=`, and one symbol belongs to exactly one batch
        per run, so the only files this can remove are earlier runs' shards for
        a session the current run is re-fetching in full -- which is precisely
        what supersedes them.

        Deliberately does NOT change the filename contract (D-19 contract 4):
        no timestamp, no uuid, no counter is added. The name stays derivable;
        what changes is that a superseded name is removed rather than left
        beside its successor.
        """
        for stale in directory.glob("part-*.pqt"):
            # `part-{batch_key}-{page:05d}.pqt` -- the key is the segment
            # between the first and last dashes of the stem.
            stem = stale.stem
            if not stem.startswith("part-") or "-" not in stem[5:]:
                continue
            if stem[5:].rsplit("-", 1)[0] == batch_key:
                continue
            stale.unlink(missing_ok=True)

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
            if self._partition_is_per_symbol:
                # The window is part of `batch_key`, so a re-fetch over a
                # DIFFERENT window writes a SECOND file rather than overwriting
                # the first. `1d`/`1m` absorb that in dedup; tick deliberately
                # never dedups (D-16), so the duplicate would be permanent and
                # invisible. See `_clear_superseded_shards` for why this is only
                # sound where the directory belongs to one symbol.
                self._clear_superseded_shards(path.parent, batch_key)
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

    def _refresh_batches(self, pending: Sequence[str]) -> Iterator[list[str]]:
        """Batches for a `refresh()`: symbols GROUPED by identical recorded
        `last_date`, then chunked by `batch_size` within each group.

        This is REQUEST PACKING and nothing else. D-06's window rule is
        untouched: a refresh still fetches `[symbol watermark, config.end_date]`
        and still ignores a widened `config.start_date`. Grouping only decides
        which symbols may legally travel in the SAME request, given that one
        request carries exactly one `start`.

        Grouping rather than taking `min(watermark)` over a mixed batch: most
        symbols in a routine refresh share one watermark, so the grouping is
        near-free, and the alternative -- issue the earliest start for the
        whole batch and let deduplication absorb the overlap -- is correct but
        re-fetches history nobody asked for and makes the volume guard's
        estimate systematically wrong.

        Symbols with no watermark at all bucket under `config.start_date`,
        which is exactly the start `_attempt_batch` would derive for them.
        Insertion order is preserved within and across buckets so a run is
        reproducible.
        """
        buckets: dict[str, list[str]] = {}
        for symbol in pending:
            coverage = self._read_coverage(symbol) or {}
            start = coverage.get("last_date") or self.config.start_date
            buckets.setdefault(start, []).append(symbol)

        for group in buckets.values():
            yield from self._batches(group)

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
            PageLedger.default_path(str(self._watermark_root), batch_key),
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

        **The page loop is bounded by more than a falsy token.** A vendor that
        hands back the token it was given would otherwise spin forever, and
        because `page_index` increments each iteration the shard names keep
        changing rather than overwriting -- so the failure fills the disk
        instead of stalling. See the repeated-token check below.
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

            if next_token == page_token:
                # The loop's ONLY exit is a falsy token, so a vendor that
                # echoes the token it was handed -- a mis-implemented
                # `page_token`, a proxy replaying a response, a partial outage
                # -- would run forever. `page_index` increments each iteration,
                # so the shard filenames keep CHANGING: nothing overwrites,
                # the ledger's `pages` list grows without bound, and the raw
                # root fills the disk while every individual request looks
                # successful. Refusing is the only bounded outcome, and it
                # leaves a resumable ledger behind.
                raise ValueError(
                    f"{self.class_name}: the vendor returned the SAME page "
                    f"token it was given ({next_token!r}) on page "
                    f"{page_index} of batch {batch_key}. Continuing would "
                    f"loop forever, writing a new shard every iteration until "
                    f"the raw root fills the disk. Refusing instead. The "
                    f"pages that DID land are recorded, so a re-run resumes "
                    f"rather than restarting."
                )

            page_index += 1
            page_token = next_token

    # -- entry points -------------------------------------------------------

    def download(self, symbols: list[str] | None = None) -> Self:
        """Full backfill over `[config.start_date, config.end_date]`."""
        return self._run(symbols, from_watermark=False)

    def refresh(self, symbols: list[str] | None = None) -> Self:
        """Incremental fetch, each symbol starting at its OWN watermark.

        Refresh fetches from each symbol's last covered date FORWARD, so the
        covered start is whatever it already was -- and if it was unknown it
        STAYS unknown. Refresh never invents coverage it did not fetch, and it
        does NOT honour a widened `config.start_date`; widening the covered
        range is `download()`'s job (D-06 / 260906-26o D-04).
        """
        return self._run(symbols, from_watermark=True)

    # -- concurrent, resumable, failure-isolated orchestration --------------
    #
    # Hoisted verbatim from `ConcurrentTiingoAcquisition` by 03.2-03 (D-02).
    # A full-US-market backfill is ~15.4k symbols and several hours; at that
    # scale three properties stop being niceties and none of them is
    # vendor-specific, which is why they live here once rather than once per
    # vendor:
    #
    # - **Concurrency.** The work is network-bound and each vendor client is
    #   shared, so THREADS are right and processes are not --
    #   `joblib.Parallel(backend="threading")`, matching what
    #   `base/model.py:train_cv` already uses for its CV folds.
    # - **Resumability.** A job killed at ticker 20,000 must resume near
    #   ticker 20,000. Symbols already covering the requested window are
    #   skipped ENTIRELY rather than re-requested for a one-day sliver.
    # - **Failure isolation, WITH a global exception.** One delisted ticker's
    #   404 must not abort the other 15,000, so exceptions are captured per
    #   BATCH and the watermark is written only on success. But a vendor-wide
    #   condition is not one ticker's fault: treating it as an ordinary
    #   per-symbol failure is what burned ~10,000 symbols as fast-failing
    #   requests in the observed 2026-09-06 incident. It therefore trips a
    #   global abort instead (D-05); see `_is_quota_error` and `_run_once`.
    #
    # What does NOT live here is which exception means which of those things.
    # Tiingo's 429 is hourly ALLOCATION exhaustion (global, ~an hour to
    # clear); Alpaca's 429 is a per-MINUTE rate limit a healthy run is
    # expected to hit and recover from in seconds. Same status code, opposite
    # correct response -- so classification stays per-vendor and no
    # `QUOTA_STATUS_CODES` exists at this level (03.2-RESEARCH.md Pitfall 1).

    def _run(self, symbols: list[str] | None, from_watermark: bool) -> Self:
        """The single concurrent runner both entry points delegate to.

        `download()` and `refresh()` differ ONLY in how each batch's start
        date is resolved, so they share this body rather than each carrying
        its own fan-out/resume/failure-capture copy that could drift.

        A bounded RESUME LOOP wraps the fan-out (D-06). Each pass recomputes
        `pending` from the watermarks ON DISK, which makes the resume logic
        and the skip logic literally the same code -- there is no parallel
        bookkeeping that could drift from what actually landed.

        **What is assumed and what is not.** A vendor's allocation reset
        semantics -- a fixed top-of-hour bucket versus a rolling window -- are
        NOT established, and nothing here claims to know them. That is why
        this is a configurable INTERVAL with a bounded attempt count rather
        than a computed resume-at instant. See `DEFAULT_QUOTA_WAIT_SECONDS`
        and `DEFAULT_QUOTA_MAX_WAITS` for what each default is grounded in.
        All three knobs are read from `config.kwargs` via `_knob`, never as
        constructor arguments, and waiting is OFF by default.
        """
        # Validated HERE, not only inside `_fetch_batch`. `_validate_symbols`'
        # own docstring says it runs "BEFORE path construction", and until this
        # line that was false: `_partition_by_coverage` below turns EVERY
        # symbol in the roster into a filesystem path
        # (`_watermark_path(symbol)` -> `self._watermark_root / f"{symbol}.json"`)
        # before any batch exists, so a roster entry of `../../../../etc/hosts`
        # -- from a hand-written `--symbols`, a corrupted `universe.parquet`, or
        # a future roster source -- was `exists()`-checked and `json.load`ed
        # outside the watermark root. Reads only, so the blast radius was
        # bounded; the control simply did not run where it claimed to.
        requested = self._validate_symbols(list(symbols or self.config.symbols))
        wait_for_quota = bool(
            self._knob("wait_for_quota", self.DEFAULT_WAIT_FOR_QUOTA)
        )
        wait_seconds = float(
            self._knob("quota_wait_seconds", self.DEFAULT_QUOTA_WAIT_SECONDS)
        )
        max_waits = int(
            self._knob("quota_max_waits", self.DEFAULT_QUOTA_MAX_WAITS)
        )

        failures: dict[str, str] = {}
        # Accumulated across resume PASSES, for the same reason `failures` is:
        # each pass re-derives `pending` from the watermarks on disk, so the
        # last pass alone knows nothing about what an earlier pass completed.
        # A result built from one pass would report a near-empty success list
        # for a run that in fact downloaded most of the roster.
        all_succeeded: set[str] = set()
        # Pre-set so the result is well-defined on the path where `pending` is
        # empty on the first pass and `_run_once` never runs at all.
        aborted = False
        # Pre-set for the same reason `aborted` is: on the path where `pending`
        # is empty on the first pass, `_run_once` never runs and the result
        # still has to be well-defined.
        cancelled = False
        waits = 0
        while True:
            pending = requested
            if self._knob("resume", True):
                pending, counts = self._partition_by_coverage(
                    requested, from_watermark
                )
                self._report_coverage(requested, pending, counts)
            if not pending:
                break

            aborted, pass_failures, succeeded, cancelled = self._run_once(
                pending, from_watermark
            )
            # MERGED, not replaced. `_run`'s resume loop can execute several
            # passes and each one re-derives `pending` from the watermarks on
            # disk, so a pass that aborts before it reaches an earlier pass's
            # failures returns `{}` -- and a plain reassignment then wrote an
            # EMPTY manifest for a run that had, say, 40 real 404s. The
            # manifest's own docstring calls an empty one "a meaningful
            # statement that the last run was clean", which would have been a
            # false statement about a run that was not.
            #
            # Symbols that SUCCEEDED this pass are dropped in the same update:
            # a failure the next pass cleared must not linger in the manifest
            # either. Symbols merely SKIPPED by an abort are left alone -- they
            # were not retried, so this pass has no news about them.
            for symbol in succeeded:
                failures.pop(symbol, None)
            all_succeeded.update(succeeded)
            failures.update(pass_failures)
            if cancelled:
                # BEFORE the quota branch, and it does not fall through into
                # it: a cancel must never wait, never log the
                # allocation-exhausted message, and never resume the run the
                # operator just stopped (D-17, RESEARCH Pitfall 1).
                logger.warning(
                    f"Cancelled at a batch boundary. "
                    f"{len(all_succeeded)} symbol(s) completed and their "
                    f"watermarks are on disk; "
                    f"{len(requested) - len(all_succeeded)} were never "
                    f"attempted and have no sidecar, so a re-run resumes "
                    f"exactly there and re-downloads nothing. This is an "
                    f"operator stop, not a vendor condition."
                )
                break
            if not aborted:
                break
            if not wait_for_quota:
                logger.warning(
                    "Not waiting for the allocation to reset "
                    "(wait_for_quota is off). Re-run when the window has "
                    "reset, or set wait_for_quota=True to sit through it."
                )
                break
            if waits >= max_waits:
                logger.warning(
                    f"Gave up after {waits} wait(s) (quota_max_waits="
                    f"{max_waits}); the allocation had still not reset. "
                    f"Every watermark is preserved -- re-run later to resume."
                )
                break

            waits += 1
            logger.warning(
                f"Waiting {wait_seconds:.0f}s for the request allocation to "
                f"reset, then resuming (attempt {waits}/{max_waits}). The "
                f"vendor's reset semantics are not published, so this is a "
                f"configured interval, not a computed reset time."
            )
            self._sleep(wait_seconds)

        # UNCONDITIONAL, and outside the `while True:` above, so it runs on
        # every way out of the resume loop: a cancel, an empty `pending` on the
        # first pass, a normal completed run, an abort with `wait_for_quota`
        # off, and an abort that exhausted `quota_max_waits`. The POSITION is
        # what makes the property hold -- any `break` added to that loop later
        # inherits it, including ones nobody has written yet. Do not re-express
        # this as a call on each exit that seems to need one: enumerating the
        # exits by hand is exactly what was got wrong, and the merge sat in the
        # cancel branch alone while the DEFAULT quota abort
        # (`wait_for_quota` is off unless asked for) overwrote a previous run's
        # 404s with `{}` -- which `_write_failure_manifest`'s own docstring
        # calls a meaningful statement that the last run was clean
        # (03.4 D-18, `03.4-VERIFICATION.md` gap 1, REVIEW CR-01).
        self._merge_unattempted_failures(
            failures, attempted=all_succeeded | set(failures)
        )
        self._write_failure_manifest(failures)
        # Built from the SAME accumulated `failures` dict the manifest just
        # received, at the SAME point, so `set(result.failures)` and the
        # manifest's key set cannot drift (03.4 D-18). Say plainly what that
        # equality IS: both sides are two expressions over one variable
        # evaluated once, so it is a RECEIPT that the result and the manifest
        # were assembled together -- not a check that either is correct. It
        # read True during phase verification directly on top of a manifest
        # that had just been emptied. The property that protects the operator
        # is the durability of the manifest's CONTENTS, pinned by
        # `test_the_manifest_survives_a_quota_abort_on_the_default_path`. The
        # merge above now runs on every exit path, so the receipt is issued
        # over a manifest that has already been made whole.
        self.last_result = AcquisitionResult(
            vendor=self.VENDOR,
            requested=tuple(requested),
            succeeded=tuple(sorted(all_succeeded)),
            failures=dict(failures),
            cancelled=cancelled,
            quota_aborted=aborted,
            coverage=self.coverage_report(requested),
        )
        # `Self`, not the result. `download()`/`refresh()` keep their chaining
        # contract, which the rest of this repo's idiom
        # (`Dataset.from_raw_data().save()`) depends on; the result is read off
        # `last_result` by `quantlab/acquisition/registry.py:run`.
        return self

    def _run_once(
        self, pending: list[str], from_watermark: bool
    ) -> tuple[bool, dict[str, str], set[str], bool]:
        """One concurrent pass over `pending`, returning
        `(quota_aborted, per_symbol_failures, symbols_that_succeeded,
        cancelled)`.

        **This arity WIDENED in 03.4-05 where `_attempt_batch`'s could not.**
        The asymmetry is deliberate and is L-6: `_attempt_batch`'s 3-tuple is
        destructured by two tests in `tests/test_acquisition_batching.py`,
        while nothing outside this file unpacks `_run_once`'s. So the "why did
        it stop" signal that `_attempt_batch` may not carry lives here instead,
        as a fourth element -- and `quota_aborted` and `cancelled` are separate
        booleans rather than one tri-state, because `_run` must handle them
        differently: a quota abort may WAIT and resume, a cancel must not
        (RESEARCH Pitfall 1).

        The third element exists so `_run` can ACCUMULATE failures across
        passes without a stale entry surviving a later success. It reports only
        what this pass actually completed -- a symbol skipped by the abort is
        in neither set, because this pass learned nothing about it.

        The unit of work is a BATCH, not a symbol. For a vendor whose
        `DEFAULT_BATCH_SIZE` is 1 the batch count equals the symbol count and
        this is behaviourally identical to the per-symbol fan-out it replaces
        -- same request count, same failure granularity, same resume
        granularity.
        """
        abort = self._reset_abort()
        self._reset_no_data_marks()
        max_workers = int(self._knob("max_workers", self.DEFAULT_MAX_WORKERS))

        # Materialised so the progress bar has a real total. Refresh groups by
        # recorded watermark first, because one request carries exactly one
        # `start` (see `_refresh_batches`).
        batches = list(
            self._refresh_batches(pending)
            if from_watermark
            else self._batches(pending)
        )

        def inputs():
            for batch in batches:
                # OPTIMISATION ONLY -- not where the guarantee lives. It stops
                # joblib queueing new batches, but with pre-dispatch batching
                # it cannot be relied on alone. The first-statement check in
                # `_attempt_batch` is what actually stops the vendor requests.
                #
                # The cancel token is mirrored here for the same reason the
                # abort is, and with the same caveat: joblib's default
                # `pre_dispatch` of `2 * n_jobs` means a run with fewer batches
                # than that has ALL of them queued before the first result is
                # drained, so this `break` never even executes.
                # `tests/test_acquisition_progress.py:test_cancel_leaves_a_resumable_store`
                # is deliberately sized that way, so it cannot pass on this
                # check alone (RESEARCH Pitfall 2).
                if self._should_stop():
                    break
                yield batch

        # `return_as="generator_unordered"` is load-bearing, not a style
        # choice. The default eager `Parallel(...)` call returns only once
        # every batch is done, so a `tqdm` around it would render nothing
        # for hours and then a full bar; wrapping the DISPATCH generator
        # instead fills the bar instantly, because `Parallel` consumes that
        # generator up front to queue the work. Streaming the RESULTS is the
        # only form where one tick means one batch actually landed on disk.
        stream = Parallel(
            n_jobs=max_workers, backend="threading", return_as="generator_unordered"
        )(delayed(self._attempt_batch)(batch, from_watermark) for batch in inputs())

        # The bar is no longer constructed here: `TqdmProgressReporter` builds
        # it from this event, with the same `total`, the same `desc`, the same
        # `unit="batch"` and the same `disable` semantics (D-16). What changed
        # is WHO renders, not WHAT is rendered -- a console attaches a callback
        # reporter instead and receives these same events as objects.
        total = len(batches)
        self._emit(
            ProgressEvent(
                kind="run_started",
                vendor=self.VENDOR,
                total=total,
                message=(
                    f"{self.VENDOR} "
                    f"{self.config.start_date}..{self.config.end_date}"
                ),
            )
        )
        results = []
        switched = False
        cancel_announced = False
        completed_batches = 0
        try:
            # DRAINED to completion, never broken out of. Abandoning a joblib
            # result generator mid-iteration leaves worker teardown to garbage
            # collection; draining is deterministic, and it is nearly free
            # because every remaining task is now a microsecond no-op. The bar
            # therefore terminates by FINISHING rather than by being killed --
            # and its description is switched so a racing bar cannot read as
            # "all this work succeeded".
            for result in stream:
                results.append(result)
                completed_batches += 1
                batch_symbols, batch_status, _batch_message = result
                self._emit(
                    ProgressEvent(
                        kind="batch_completed",
                        vendor=self.VENDOR,
                        completed=completed_batches,
                        total=total,
                        symbols=tuple(batch_symbols),
                        detail={"status": batch_status},
                    )
                )
                if abort.is_set() and not switched:
                    switched = True
                    self._emit(
                        ProgressEvent(
                            kind="quota_exhausted",
                            vendor=self.VENDOR,
                            completed=completed_batches,
                            total=total,
                            message=QUOTA_EXHAUSTED_DESCRIPTION,
                        )
                    )
                # Announced from the same position as the quota switch, and
                # separately from it, so a console can tell "the operator
                # stopped this" from "the vendor stopped this" (D-17).
                if self._is_cancelled() and not cancel_announced:
                    cancel_announced = True
                    self._emit(
                        ProgressEvent(
                            kind="cancelled",
                            vendor=self.VENDOR,
                            completed=completed_batches,
                            total=total,
                            message=(
                                "Cancelled -- draining the queue, not fetching"
                            ),
                        )
                    )
        finally:
            # In a `finally` so an exception escaping the drain (which the
            # per-batch isolation makes unlikely, not impossible) still closes
            # the bar instead of leaving a half-drawn one on the terminal.
            self._emit(
                ProgressEvent(
                    kind="run_finished",
                    vendor=self.VENDOR,
                    completed=completed_batches,
                    total=total,
                )
            )
            self._close_reporter()

        failures = {}
        succeeded: set[str] = set()
        for batch_symbols, status, message in results:
            if status == "failed":
                for symbol in batch_symbols:
                    failures[symbol] = message
            elif status == "ok":
                succeeded.update(batch_symbols)
        quota_messages = [
            message for _, status, message in results if status == "quota"
        ]

        # Reported from THIS pass rather than only from disk, so a run that
        # just discovered 400 empty symbols says so while it is fresh instead
        # of leaving the number discoverable only by reading sidecars. The
        # on-disk total is reported separately by `_report_coverage` at the
        # top of every pass.
        if getattr(self, "_no_data_marks", 0):
            logger.info(
                f"{self._no_data_marks} symbol(s) were queried successfully "
                f"this pass and the vendor returned NO rows for them over "
                f"{self.config.start_date}..{self.config.end_date}. Their "
                f"watermarks advanced with a 'no data' marker, so the next "
                f"run skips them instead of re-asking -- this is a recorded "
                f"absence, not a failure, and it is deliberately not in "
                f"{self.FAILURE_MANIFEST_NAME}."
            )

        # Read ONCE, after the drain, and reported separately from the quota
        # abort. A cancel does not set `abort`, so the branch below -- and its
        # allocation-exhausted log line -- is unreachable on a pure cancel: the
        # early return here is what keeps a cancelled run from being described
        # to the operator as the vendor running out of allowance.
        cancelled = self._is_cancelled()

        if not abort.is_set():
            return False, failures, succeeded, cancelled

        completed = sum(
            len(batch_symbols)
            for batch_symbols, status, _ in results
            if status == "ok"
        )
        remaining = len(pending) - completed
        detail = quota_messages[0] if quota_messages else "request allocation"
        logger.warning(
            f"Vendor request allocation exhausted -- STOPPED dispatching "
            f"rather than burning the remainder as fast-failing requests. "
            f"{completed} symbol(s) completed this pass, {remaining} remain. "
            f"Every watermark is preserved, so a re-run resumes exactly here "
            f"and re-downloads nothing. This is a global condition, so it is "
            f"NOT recorded in the per-symbol failure manifest. Vendor said: "
            f"{detail}"
        )
        return True, failures, succeeded, cancelled

    def coverage_report(self, symbols: list[str] | None = None) -> dict:
        """Classify the roster against the requested window and return the
        counts, issuing ZERO vendor requests.

        The read-only form of `_run`'s pending computation, so a dry run can
        answer "would widening the window actually re-fetch anything?" before
        committing to a multi-hour job -- and can show that the un-stamped
        legacy count really did fall to zero after stamping. It shares
        `_partition_by_coverage` with the real run rather than reimplementing
        the rule, so the two can never disagree.
        """
        # Same reason as `_run`: `_partition_by_coverage` builds a watermark
        # path per symbol, so validation has to precede it here too.
        requested = self._validate_symbols(
            list(symbols if symbols is not None else self.config.symbols)
        )
        pending, counts = self._partition_by_coverage(
            requested, from_watermark=False
        )
        return {
            "requested": len(requested),
            "pending": len(pending),
            "skipped": len(requested) - len(pending),
            **counts,
        }

    def _legacy_policy(self) -> str:
        """Delegates to `CoverageLedger.legacy_policy`."""
        return self._coverage.legacy_policy()

    def _coverage_status(self, symbol: str, from_watermark: bool = False) -> str:
        """Delegates to `CoverageLedger.coverage_status` -- see there for the
        four-state rule and the D-04 argument behind the `"legacy"` branch.
        """
        return self._coverage.coverage_status(symbol, from_watermark)

    def _classify_coverage(
        self, coverage: dict | None, from_watermark: bool = False
    ) -> str:
        """Delegates to `CoverageLedger.classify_coverage` -- the D-09 choke
        point, and the only place a `last_date == end_date` or `start_date <=`
        comparison belongs.
        """
        return self._coverage.classify_coverage(coverage, from_watermark)

    def _covers(self, symbol: str, from_watermark: bool = False) -> bool:
        """Delegates to `CoverageLedger.covers` -- the skip predicate `_run`
        filters on.
        """
        return self._coverage.covers(symbol, from_watermark)

    def _partition_by_coverage(
        self, requested: list[str], from_watermark: bool
    ) -> tuple[list[str], dict[str, int]]:
        """Delegates to `CoverageLedger.partition_by_coverage`.

        **This delegation is what D-09 makes binding.** `coverage_report()`,
        `_run` and `SourceInspector.coverage` all reach that ONE function
        object, so a mutation to it changes every answer -- which is how the
        sharing is proved by identity rather than by results that happen to
        agree. Reintroducing a body here is the violation.
        """
        return self._coverage.partition_by_coverage(requested, from_watermark)

    def _report_coverage(
        self, requested: list[str], pending: list[str], counts: dict[str, int]
    ) -> None:
        """Report the outcomes separately, so widening the window -- or a
        vendor having nothing to give -- has a visible, countable consequence
        instead of a silent one.
        """
        skipped = len(requested) - len(pending)
        # The console gets the counts as STRUCTURE; a shell run's stderr is
        # byte-unchanged, because every one of the four log lines below is
        # kept exactly as it was (D-19, RESEARCH Q3). The event is additive:
        # this method gained a consumer, not a different behaviour.
        self._emit(
            ProgressEvent(
                kind="coverage",
                vendor=self.VENDOR,
                completed=len(pending),
                total=len(requested),
                detail={
                    "requested": len(requested),
                    "pending": len(pending),
                    "skipped": skipped,
                    **counts,
                },
            )
        )
        if skipped:
            logger.info(
                f"Resume: skipping {skipped}/{len(requested)} symbols already "
                f"covering {self.config.start_date}..{self.config.end_date}; "
                f"{len(pending)} remaining."
            )
        if counts.get("no_data"):
            logger.info(
                f"{counts['no_data']} symbol(s) carry a recorded "
                f"'queried, no data' marker -- the vendor was ASKED and "
                f"returned nothing for the window on their sidecar. They are "
                f"distinct from the failures in "
                f"{self.FAILURE_MANIFEST_NAME}, which were never successfully "
                f"queried at all, and they are skipped rather than re-asked "
                f"every run."
            )
        if counts["widened"]:
            logger.info(
                f"Re-fetching {counts['widened']} symbol(s) whose recorded "
                f"coverage starts AFTER the requested "
                f"{self.config.start_date} -- their history is shallower than "
                f"this run asks for."
            )
        if counts["legacy"]:
            if self._legacy_policy() == "warn":
                logger.warning(
                    f"{counts['legacy']} symbol(s) carry a legacy watermark "
                    f"with NO recorded covered start. They were SKIPPED, and "
                    f"whether they actually cover {self.config.start_date} "
                    f"cannot be known from disk -- only you know what window "
                    f"they were fetched over, which is why this is not "
                    f"guessed. Stamp them once with: "
                    f"{self.STAMP_COMMAND_HINT}  (or set "
                    f"legacy_watermarks='refetch' to re-download them "
                    f"instead)."
                )
            else:
                logger.info(
                    f"legacy_watermarks='refetch': re-fetching "
                    f"{counts['legacy']} symbol(s) whose covered start is "
                    f"unknown."
                )

    def _attempt_batch(
        self, symbols: Sequence[str], from_watermark: bool
    ) -> tuple[list[str], str, str | None]:
        """Fetch one BATCH, returning `(symbols, status, message_or_None)`.

        `status` is one of:

        - `"ok"` -- fetched, and one watermark written per symbol.
        - `"failed"` -- a per-batch problem (a delisted ticker's 404, say).
          Every symbol in the batch lands in the manifest, none gets a
          watermark, and all are retried next run. Unchanged from before
          quota handling existed.
        - `"quota"` -- the account's allocation is gone. Trips the global
          abort and is EXCLUDED from the manifest: recording a global
          condition as one ticker's fault would defame a perfectly good
          symbol and make the manifest lie about what the last run did.
        - `"skipped"` -- never attempted, because the abort was already set.
          No watermark, not a failure, counted for the report only.

        Never raises: a propagated exception would tear down the whole
        `Parallel` fan-out and abort every other batch, which is precisely
        the failure mode this orchestration exists to prevent. The watermark
        is written only after a successful fetch, so a failed batch is
        retried by the next run instead of being silently marked complete.
        """
        # FIRST statement, and that is the whole point. joblib cannot cancel
        # work it has already queued, so this check -- not the input generator
        # -- is what actually stops the ~260 sym/s burn observed in the field.
        # Every remaining batch becomes a no-op returning in microseconds,
        # having issued zero vendor requests.
        #
        # The CANCEL token (03.4 D-17) rides the SAME position, for the same
        # reason and with the same return shape. `_should_stop()` is the OR of
        # the quota abort and the cancel token, so a cancel stops the burn at
        # exactly the granularity the quota abort already did -- a batch
        # boundary -- which is what makes a cancelled run resumable: every
        # batch that got past this line finished and wrote its watermarks, and
        # every batch that did not has no sidecar at all.
        #
        # The 3-element return is FIXED (L-6): two tests in
        # `tests/test_acquisition_batching.py` destructure it, so the cancel
        # reuses the existing `"skipped"` status rather than adding a fourth
        # "why it stopped" element. Which condition stopped it is answered by
        # `_is_cancelled()` at the `_run_once` level, where nothing unpacks a
        # fixed arity.
        if self._should_stop():
            return list(symbols), "skipped", None

        symbols = list(symbols)
        coverages = {
            symbol: (self._read_coverage(symbol) or {}) for symbol in symbols
        }

        # The start to REQUEST. A full backfill asks for `config.start_date`;
        # a refresh asks from the batch's recorded watermark forward.
        # `_refresh_batches` groups by identical `last_date`, so the batch has
        # a single well-defined start -- `min` is the defensive reading of
        # that invariant, because over-fetching a shared window is harmless
        # (deterministic shard names make it an overwrite) while
        # under-fetching would silently lose rows.
        start_date = self.config.start_date
        if from_watermark:
            start_date = min(
                (
                    coverages[symbol].get("last_date") or self.config.start_date
                    for symbol in symbols
                ),
                default=self.config.start_date,
            )

        backoff = float(
            self._knob(
                "rate_limit_backoff_seconds",
                self.DEFAULT_RATE_LIMIT_BACKOFF_SECONDS,
            )
        )
        max_retries = int(
            self._knob(
                "rate_limit_max_retries", self.DEFAULT_RATE_LIMIT_MAX_RETRIES
            )
        )

        retries = 0
        while True:
            try:
                outcome = self._fetch_batch(
                    symbols, start_date=start_date, end_date=self.config.end_date
                )
            except Exception as exc:  # noqa: BLE001 -- isolation is the point
                message = self._scrub(f"{type(exc).__name__}: {exc}")
                verdict = self._classify_error(exc)
                if verdict == "quota":
                    self._abort.set()
                    return symbols, "quota", message
                if verdict == "rate_limited" and retries < max_retries:
                    # Transient and vendor-wide but FAST: back off in this
                    # worker only. Setting the global abort here would stop
                    # every other batch over a condition that has usually
                    # cleared by the time the log line is written, which is
                    # the T-03.2-16 failure in miniature.
                    if retries == 0:
                        # Logged ONCE per batch, on the first backoff only: at
                        # `max_workers` threads x `max_retries` retries this
                        # would otherwise be the noisiest line in the run.
                        #
                        # Whatever the vendor sent, and nothing invented. This
                        # is the only diagnostic a 429 produces, and without it
                        # the backoff constant is unverifiable in the field --
                        # `X-RateLimit-Reset` is what tells an operator whether
                        # the ceiling they hit is per-minute at all.
                        headers = self._rate_limit_headers(exc)
                        logger.info(
                            f"Rate limited on a batch of {len(symbols)} "
                            f"symbol(s); backing off {backoff:.0f}s in this "
                            f"worker only (up to {max_retries} times). Vendor "
                            f"rate-limit headers: "
                            f"{headers or 'none sent'}."
                        )
                    retries += 1
                    self._sleep(backoff)
                    continue
                # A `rate_limited` verdict that has exhausted its retries
                # degrades to `failed` rather than spinning (T-03.2-17): it
                # goes to the manifest with no watermark, so the NEXT run
                # retries it instead of this one never finishing.
                return symbols, "failed", message
            break

        # -- the "queried, no data" marker set, computed ONCE for the whole
        #    batch and never per page (D-04, RESEARCH Pitfall 4).
        #
        # The vendor sorts symbol-major and only then by timestamp, so page 0
        # of a 100-symbol batch legitimately carries ONE symbol. Computing
        # `requested - seen` per page would stamp the other 99 "queried, no
        # data", advance their watermarks and skip them forever -- a silent
        # 99% loss that looks like a successful run. `_fetch_batch` therefore
        # accumulates `symbols_with_data` across every page (seeded from the
        # ledger, so a resumed run inherits what earlier pages already found)
        # and this is the only place the difference is taken.
        #
        # Two gates, and neither is redundant. `outcome.complete` means the
        # vendor handed back a null page token: an incomplete batch has no
        # opinion at all about absence, and absence-means-unknown is the house
        # rule (260906-26o D-04). The abort check covers the other shape --
        # this batch finished, but another thread has already stopped the
        # world -- because a global stop is not the moment to start recording
        # new claims about what a vendor does not have. A batch that RAISED
        # never reaches here at all; it returned `failed` or `quota` above.
        marked: set[str] = set()
        if outcome.complete and not self._abort.is_set():
            marked = set(symbols) - outcome.symbols_with_data

        recorded = 0
        for symbol in symbols:
            # The covered start to RECORD, per symbol. A full backfill
            # overwrites the shards wholesale, so the requested start is a
            # true statement about them; a refresh only extends forward, so it
            # carries each symbol's EXISTING start through -- and if that was
            # unknown it stays unknown (D-04).
            covered_start = (
                coverages[symbol].get("start_date")
                if from_watermark
                else self.config.start_date
            )
            # The marker to RECORD, per symbol, and the same asymmetry applies
            # to it. A full backfill queried exactly the window it is about to
            # record, so its answer is a true statement about that window. A
            # refresh queried only `[watermark, end_date]` while the sidecar
            # records `[covered_start, end_date]`, so it may CLEAR a marker
            # (rows arrived, so the symbol demonstrably has data in the
            # recorded window) but may never assert a new one -- it has no
            # evidence about the earlier part of the range it is stamping.
            no_data = symbol in marked
            if from_watermark:
                no_data = no_data and bool(coverages[symbol].get("no_data"))
            self._write_watermark(
                symbol,
                self.config.end_date,
                start_date=covered_start,
                no_data=no_data,
            )
            recorded += int(no_data)

        self._record_no_data_marks(recorded)
        return symbols, "ok", None

    def _merge_unattempted_failures(
        self, failures: dict[str, str], attempted: set[str]
    ) -> None:
        """Fold the EXISTING manifest's entries for symbols this run never
        reached back into `failures`, in place (03.4 D-18, RESEARCH Pitfall 4).

        **Why this exists at all.** `_write_failure_manifest` OVERWRITES, and
        `_run`'s `failures` starts empty on every call. That was safe while
        every run either completed or quota-aborted, because such a run reaches
        (or explicitly declines to speak for) every symbol. A CANCELLED run is
        the new shape: it can stop before it ever gets to a symbol that failed
        last time, and would then write `{}` -- while
        `_write_failure_manifest`'s own docstring calls an empty manifest "a
        meaningful statement that the last run was clean". That statement would
        be false about a store that still holds forty un-retried 404s, in the
        one file the operator console shows.

        **Merge rather than skip the write**, which is the other option
        RESEARCH left open. Skipping would preserve the old manifest but throw
        away failures the cancelled run DID discover; merging keeps both, and
        it is what makes `set(result.failures) == set(manifest)` true on EVERY
        exit path rather than only on the uncancelled ones -- because `_run`
        calls this BEFORE both the write and the result assembly, so the two
        are built from the same dict at the same point exactly as before.

        `attempted` is the set this run has news about: symbols it completed,
        plus symbols it failed. Entries for those are NOT restored -- a symbol
        that succeeded this run must leave the manifest, and one that failed
        this run already carries this run's message.
        """
        for symbol, message in self._coverage.read_failure_manifest().items():
            if symbol in attempted:
                continue
            failures.setdefault(symbol, message)

    def _write_failure_manifest(self, failures: dict[str, str]) -> None:
        """Persist `{symbol: message}` for this run, overwriting the previous
        manifest.

        Overwriting is correct rather than lossy: a failed symbol never got a
        watermark, so the next run puts it back in `pending` and it reappears
        here if it fails again. The manifest therefore always describes the
        LATEST run, and an empty one is a meaningful statement that the last
        run was clean (T-0iy-07).

        "The latest RUN", not the latest PASS. `_run`'s resume loop can execute
        several passes, and `failures` is accumulated across all of them rather
        than reassigned by each: a pass that aborts early has no news about the
        symbols an earlier pass already failed, and letting it erase them would
        make an empty manifest a false statement about a run that had failures
        (WR-03).

        The write is ATOMIC (D-20). It matters here for the same reason it
        matters for the watermark: a cancel lands at a batch boundary and this
        file is rewritten on the way out, so a plain write could leave a
        truncated manifest -- either unparseable, or well-formed but naming
        FEWER failures than the run actually had, which is the same false
        "the last run was clean" statement WR-03 exists to prevent.
        """
        # Through the LEDGER, not by re-joining the two parts here: a second
        # path expression is how the tick `data_type` namespacing gets
        # forgotten on one of the two sides, and this is the side that writes.
        path = self._coverage.failure_manifest_path
        # Atomic for the same reason as the watermark (D-20): a cancel lands at
        # a batch boundary and this file is written on the way out, so a plain
        # write could leave a truncated manifest that reads as either invalid
        # JSON or -- worse -- a SHORTER, well-formed failure list than the run
        # actually produced. `indent=2, sort_keys=True` is kept at this call
        # site: a human reads this file, so its formatting is the feature.
        write_json_atomically(path, failures, indent=2, sort_keys=True)

        if failures:
            sample = sorted(failures)[: self._FAILURE_LOG_SAMPLE]
            logger.warning(
                f"{len(failures)} symbol(s) failed and were skipped; first "
                f"{len(sample)}: {sample}. Full manifest: {path}. They have "
                f"no watermark, so the next run retries them."
            )

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
