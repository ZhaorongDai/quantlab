import json
import os
import re
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Self, Sequence

import polars as pl
from joblib import Parallel, delayed
from loguru import logger
from tqdm import tqdm

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
        schema: both `_refresh_batches` and `_attempt_batch` use this to
        compute an incremental start, and neither wants the covered start.
        """
        payload = self._read_sidecar(symbol)
        return None if payload is None else payload.get("last_date")

    def _read_coverage(self, symbol: str) -> dict | None:
        """The covered RANGE for `symbol` as
        `{"start_date", "last_date", "no_data"}`, or None when no readable
        sidecar exists.

        Either date component may be None. In particular a LEGACY sidecar --
        `{"last_date": ...}`, the only format written before 260906-26o --
        reads back with `start_date=None`, and nothing anywhere fills that in
        from `config.start_date` or any other fallback.

        That absence is the whole point (D-04). Only the user knows what
        window those files were actually fetched over; an invented start that
        happens to be wrong reproduces exactly the silent per-symbol history
        gap this schema exists to eliminate, and reproduces it invisibly.
        Stamping is therefore an explicit, user-supplied step --
        `stamp_watermarks()` below.

        `no_data` follows the SAME discipline from the other side: it defaults
        to `False` when the key is absent, which is the correct reading of
        every sidecar written before 03.2 because the old code only wrote a
        watermark after a successful fetch. It is read through `_read_sidecar`
        like everything else -- adding a second tolerant read for the marker
        would give a corrupt sidecar two failure policies that could drift.
        """
        payload = self._read_sidecar(symbol)
        if payload is None:
            return None
        return {
            "start_date": payload.get("start_date"),
            "last_date": payload.get("last_date"),
            "no_data": bool(payload.get("no_data", False)),
        }

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
        """
        path = self._watermark_path(symbol)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, str | bool] = {"last_date": last_date}
        if start_date is not None:
            payload["start_date"] = start_date
        if no_data:
            payload["no_data"] = True
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

        It fills the START and nothing else. An existing `no_data` marker is
        CARRIED THROUGH verbatim: stamping is a statement about which window a
        file covers, and it has no evidence at all about whether the vendor
        had rows in it. Dropping the marker here would silently downgrade a
        confirmed absence to "fetched, data landed".

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

    # -- orchestration constants (hoisted from ConcurrentTiingoAcquisition
    #    by 03.2-03 / D-02: ONE implementation drives every vendor) ---------

    #: Concurrent in-flight batch fetches. Overridable per run via
    #: `config.kwargs["max_workers"]`.
    DEFAULT_MAX_WORKERS = 8

    #: Written under `config.watermark_path` alongside the per-symbol
    #: watermark sidecars, because "what did and did not land" is exactly the
    #: same question those sidecars answer (T-0iy-07).
    FAILURE_MANIFEST_NAME = "_failures.json"

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
    LEGACY_WATERMARK_POLICIES = ("warn", "refetch")
    DEFAULT_LEGACY_WATERMARK_POLICY = "warn"

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
        requested = list(symbols or self.config.symbols)
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

            aborted, failures = self._run_once(pending, from_watermark)
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

        self._write_failure_manifest(failures)
        return self

    def _run_once(
        self, pending: list[str], from_watermark: bool
    ) -> tuple[bool, dict[str, str]]:
        """One concurrent pass over `pending`, returning
        `(quota_aborted, per_symbol_failures)`.

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
                if abort.is_set():
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

        bar = tqdm(
            total=len(batches),
            desc=f"{self.VENDOR} {self.config.start_date}..{self.config.end_date}",
            unit="batch",
            disable=not self._knob("progress", True),
        )
        results = []
        switched = False
        with bar:
            # DRAINED to completion, never broken out of. Abandoning a joblib
            # result generator mid-iteration leaves worker teardown to garbage
            # collection; draining is deterministic, and it is nearly free
            # because every remaining task is now a microsecond no-op. The bar
            # therefore terminates by FINISHING rather than by being killed --
            # and its description is switched so a racing bar cannot read as
            # "all this work succeeded".
            for result in stream:
                results.append(result)
                bar.update(1)
                if abort.is_set() and not switched:
                    switched = True
                    bar.set_description("QUOTA EXHAUSTED -- draining, not fetching")

        failures = {}
        for batch_symbols, status, message in results:
            if status == "failed":
                for symbol in batch_symbols:
                    failures[symbol] = message
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

        if not abort.is_set():
            return False, failures

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
        return True, failures

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
        requested = list(symbols if symbols is not None else self.config.symbols)
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
        policy = self._knob(
            "legacy_watermarks", self.DEFAULT_LEGACY_WATERMARK_POLICY
        )
        if policy not in self.LEGACY_WATERMARK_POLICIES:
            raise ValueError(
                f"legacy_watermarks={policy!r} is not one of "
                f"{list(self.LEGACY_WATERMARK_POLICIES)}."
            )
        return policy

    def _coverage_status(self, symbol: str, from_watermark: bool = False) -> str:
        """Classify `symbol` against the REQUESTED window, returning one of
        `"uncovered"`, `"covered"`, `"widened"` or `"legacy"`.

        A symbol is `"covered"` iff its recorded `last_date` equals
        `config.end_date` AND its recorded covered start is known and is
        `<=` `config.start_date`. ISO-8601 `YYYY-MM-DD` orders correctly under
        plain string comparison, so no date parsing happens here and no time
        zone can creep in.

        `"widened"` is the 260906-26o defect (D-03): the end date matches but
        the recorded coverage starts LATER than what is being asked for, so
        the symbol's history is shallower than the request and it must be
        re-fetched. Before this predicate existed it was skipped in silence,
        and the dataset shipped with inconsistent per-symbol history depth.

        `"legacy"` is a sidecar written before this schema: the end date
        matches but the covered start is UNKNOWN. Three responses exist and
        two are wrong. Assuming a start is forbidden outright (D-04) -- an
        assumed range that is wrong reproduces the silent gap invisibly.
        Treating unknown as uncovered is correct for integrity but re-fetches
        every already-downloaded symbol and burns a whole quota window (D-01).
        So the default is the third: treat it as covered for SKIP purposes and
        say so LOUDLY on every run until stamped. What made the D-03 failure
        dangerous was the silence, not the skip -- a run that skips these
        while printing their count and the exact command that fixes them is a
        REPORTED gap with a named cure, and only the user knows what window
        those files were fetched over. `legacy_watermarks="refetch"` is the
        opt-in escape hatch that makes this a choice rather than an accident.

        `from_watermark` (i.e. `refresh()`) short-circuits to the END-DATE
        rule alone, deliberately. Refresh requests `[watermark, end_date]` per
        symbol and never `config.start_date`, so judging it against a widened
        `config.start_date` would mark every symbol pending on every run while
        the re-fetch it triggers could not close the gap -- an endless, silent
        quota burn. Widening the covered range is `download()`'s job.

        The rule itself lives in `_classify_coverage`, over an ALREADY-READ
        coverage dict, so `_partition_by_coverage` can classify and count the
        `no_data` marker from a single read per sidecar. That is a split of
        read from rule, not a second read path -- `_read_sidecar` remains the
        only place a sidecar is opened.
        """
        return self._classify_coverage(
            self._read_coverage(symbol), from_watermark
        )

    def _classify_coverage(
        self, coverage: dict | None, from_watermark: bool = False
    ) -> str:
        """`_coverage_status`'s rule, applied to an already-read coverage dict.

        Deliberately blind to `coverage["no_data"]`. A marked symbol whose
        recorded window still covers the request is `covered` by the ordinary
        rule and is skipped; a marked symbol whose recorded window is narrower
        is `widened` and is re-fetched. The ABSENCE of a special case here is
        the design (D-04): the marker records what the vendor said about a
        WINDOW, and a branch that turned it into a permanent verdict about the
        symbol would make a later, deeper request unable to reach the vendor
        at all. Tests pin both directions so the branch cannot be added later
        as a plausible-looking "optimisation".
        """
        if coverage is None or coverage["last_date"] != self.config.end_date:
            return "uncovered"
        if from_watermark:
            return "covered"
        if coverage["start_date"] is None:
            return "legacy"
        if coverage["start_date"] <= self.config.start_date:
            return "covered"
        return "widened"

    def _covers(self, symbol: str, from_watermark: bool = False) -> bool:
        """Whether `symbol` may be skipped for the requested window.

        The skip predicate `_run` filters on. See `_coverage_status` for the
        rule and for the D-04 argument behind the `"legacy"` branch.
        """
        status = self._coverage_status(symbol, from_watermark)
        if status == "legacy":
            return self._legacy_policy() == "warn"
        return status == "covered"

    def _partition_by_coverage(
        self, requested: list[str], from_watermark: bool
    ) -> tuple[list[str], dict[str, int]]:
        """Split `requested` into what still needs fetching, plus the counts
        the run reports. One pass, so each sidecar is read exactly once.
        """
        legacy_is_skipped = self._legacy_policy() == "warn"
        pending: list[str] = []
        counts = {"covered": 0, "widened": 0, "legacy": 0, "no_data": 0}

        for symbol in requested:
            coverage = self._read_coverage(symbol)
            status = self._classify_coverage(coverage, from_watermark)
            # Counted ALONGSIDE the status rather than as a fourth status: a
            # marked symbol is `covered`/`widened`/`legacy` by exactly the same
            # rule as an unmarked one (the marker records what the vendor said
            # about a window, not a verdict about the symbol), and the count
            # exists so a run can REPORT how many symbols the vendor had
            # nothing for -- distinguishably from how many failed. What made
            # the 260906-26o defect dangerous was the silence, not the skip.
            if coverage is not None and coverage["no_data"]:
                counts["no_data"] += 1
            if status == "covered":
                counts["covered"] += 1
                continue
            if status == "legacy":
                counts["legacy"] += 1
                if legacy_is_skipped:
                    continue
            elif status == "widened":
                counts["widened"] += 1
            pending.append(symbol)

        return pending, counts

    def _report_coverage(
        self, requested: list[str], pending: list[str], counts: dict[str, int]
    ) -> None:
        """Report the outcomes separately, so widening the window -- or a
        vendor having nothing to give -- has a visible, countable consequence
        instead of a silent one.
        """
        skipped = len(requested) - len(pending)
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
        if self._abort.is_set():
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

    def _write_failure_manifest(self, failures: dict[str, str]) -> None:
        """Persist `{symbol: message}` for this run, overwriting the previous
        manifest.

        Overwriting is correct rather than lossy: a failed symbol never got a
        watermark, so the next run puts it back in `pending` and it reappears
        here if it fails again. The manifest therefore always describes the
        LATEST run, and an empty one is a meaningful statement that the last
        run was clean (T-0iy-07).
        """
        path = Path(self.config.watermark_path) / self.FAILURE_MANIFEST_NAME
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(failures, f, indent=2, sort_keys=True)

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
