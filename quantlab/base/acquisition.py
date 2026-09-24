"""Engine for downloading raw vendor data into a local hive of parquet shards.

``Acquisition`` is the abstract base every vendor client subclasses. A
subclass declares ``VENDOR`` and ``RAW_COLUMNS`` and implements
``_fetch_page``, one request returning one page of rows; the base class
supplies everything around it: symbol batching, pagination with a resumable
page ledger, concurrent dispatch across worker threads, per-batch failure
isolation, a global stop when a vendor's request allocation runs out,
cooperative cancellation, progress reporting, credential scrubbing of every
captured message, and the per-symbol watermark sidecars that let ``refresh``
fetch only what is missing.

This layer writes raw files only. Converting them into the canonical
``(timestamp, symbol)`` panel is the matching dataset class's job. Coverage
bookkeeping lives in ``quantlab.base.coverage``, the page ledger in
``quantlab.base.pageledger`` and progress events in
``quantlab.base.progress``. See ``docs/acquisition.md``.
"""

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

#: The well-formedness rule a symbol must satisfy before it becomes a
#: filesystem path segment or a query-string value. Bound from
#: ``quantlab.enums.data`` rather than compiled here, so this module, the
#: coverage ledger and the universe roster builder all filter on one compiled
#: object and cannot drift apart. It is deliberately wider than the pattern
#: the roster builder applies to scraped change-log cells, which guards a
#: different input; the two are not meant to match.
_TICKER_PATTERN = TRADEABLE_TICKER_PATTERN


@dataclass
class BatchOutcome:
    """Summary of what one ``_fetch_batch`` call did.

    ``symbols_with_data`` is a subset of ``symbols``, but it only says which
    symbols have no data once ``complete`` is true. A vendor that sorts
    symbol-major can legitimately return a single symbol on the first page of
    a hundred-symbol batch, so an incomplete batch says nothing about absence.

    Attributes:
        symbols: The symbols the batch requested.
        symbols_with_data: Symbols for which at least one row has landed.
        pages: Pages fetched so far, including those resumed from the ledger.
        complete: Whether the vendor has returned its final page.

    Example:
        >>> outcome = BatchOutcome(symbols=("AAPL", "MSFT"))
        >>> outcome.pages, outcome.complete
        (0, False)
        >>> outcome.symbols_with_data
        set()
    """

    symbols: tuple[str, ...]
    symbols_with_data: set[str] = field(default_factory=set)
    pages: int = 0
    complete: bool = False


@dataclass(frozen=True)
class AcquisitionResult:
    """Outcome of one ``download`` or ``refresh`` call, kept in memory.

    The result is the in-process caller's copy of what the run did, so nothing
    has to be read back from disk to know the outcome. It answers a narrower
    question than the failure manifest on disk: ``failures`` holds only the
    symbols this run attempted and could not fetch, whereas the manifest is a
    cross-run record that also carries entries earlier runs left behind. So
    ``set(failures)`` is a subset of both ``requested`` and the manifest's
    keys, and the messages agree on every shared key. Failure messages have
    already had credential values scrubbed out; do not add a second path for
    raw vendor exception text.

    It is defined here rather than beside the vendor registry because the
    base layer must not import a module that constructs vendor clients.

    Attributes:
        vendor: The vendor token the run fetched from.
        requested: Every symbol the run was asked for, after validation.
        succeeded: Symbols whose batches completed, sorted.
        failures: ``{symbol: scrubbed message}`` for this run's failures.
        cancelled: Whether the caller's cancel token stopped the run.
        quota_aborted: Whether the vendor's request allocation ran out.
        coverage: ``coverage_report()`` computed after the run.

    Example:
        >>> result = acq.download().last_result
        >>> result.succeeded, result.failures
        (('AAPL', 'MSFT'), {})
        >>> result.cancelled, result.quota_aborted
        (False, False)
        >>> result.coverage["covered"]
        2
    """

    vendor: str
    requested: tuple[str, ...]
    succeeded: tuple[str, ...]
    failures: dict[str, str]
    cancelled: bool
    quota_aborted: bool
    coverage: dict


class Acquisition(ABC):
    """Abstract base for config-driven, resumable raw-data downloads.

    A subclass declares ``VENDOR`` and ``RAW_COLUMNS`` and implements
    ``_fetch_page``; the base class turns that single request primitive into
    a concurrent, resumable, failure-isolated run over the config's symbol
    roster. It writes parquet shards under ``config.raw_data_dir_path`` in a
    hive layout keyed by frequency, and never touches xarray or Zarr; the
    matching dataset class converts the raw tier into the canonical panel.

    ``download()`` backfills ``[config.start_date, config.end_date]`` for
    every requested symbol. ``refresh()`` fetches each symbol forward from its
    own watermark, so repeated calls do not re-fetch history. Both skip
    symbols whose sidecar already covers the requested window.

    Coverage is recorded per symbol in a JSON sidecar under
    ``config.watermark_path`` holding ``last_date``, optionally ``start_date``
    and optionally ``no_data``. ``start_date`` is the covered range's start;
    when it is absent the covered start is unknown and is never guessed,
    which is what ``stamp_watermarks`` exists to fix. ``no_data`` means the
    vendor was asked about the symbol over that window and returned nothing,
    which is distinct from a failed fetch (an entry in the failure manifest
    and no sidecar) and from a symbol never fetched (neither). Both optional
    keys are omitted rather than written as false, so older sidecars read
    back correctly.

    Per-run tuning parameters are read from ``config.kwargs``: ``batch_size``,
    ``max_workers``, ``resume``, ``progress``, ``wait_for_quota``,
    ``quota_wait_seconds``, ``quota_max_waits``, ``legacy_watermarks``,
    ``rate_limit_backoff_seconds`` and ``rate_limit_max_retries``.

    Args:
        config: The acquisition config. ``start_date`` and ``end_date`` are
            filled with open-ended defaults when left ``None``.

    Example:
        >>> class DemoAcquisition(Acquisition):
        ...     VENDOR = "tiingo"
        ...     RAW_COLUMNS = ("timestamp", "symbol", "close", "vendor")
        ...
        ...     def _fetch_page(self, symbols, start_date, end_date,
        ...                     page_token=None):
        ...         frame = client.bars(symbols, start_date, end_date)
        ...         return frame, None
        >>> acq = DemoAcquisition(config)
        >>> acq.download().last_result.succeeded
        ('AAPL', 'MSFT')
        >>> acq.coverage_report()["covered"]
        2
    """

    def __init__(self, config: AcquisitionConfig):
        """Store the config; no result, reporter or cancel token yet."""
        self.config = config
        #: The outcome of the most recent ``download()`` or ``refresh()``, or
        #: ``None`` before either has run. Instance state rather than a config
        #: field, because ``AcquisitionConfig.to_dict()`` is persisted beside
        #: model checkpoints and a run outcome is not configuration.
        self.last_result: AcquisitionResult | None = None
        #: Where progress events go and how a caller stops the run; see
        #: ``attach()`` for why neither may live on the config.
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

        Both values are kept on the instance rather than on the config:
        ``AcquisitionConfig.to_dict()`` is written to disk beside model
        checkpoints, a ``threading.Event`` is not serialisable, and a live
        reporter is not reproducible configuration. Passing ``None`` for
        either argument clears it, so the same object can be handed to a
        second, unobserved run without inheriting the first run's reporter.

        Args:
            reporter: Destination for ``ProgressEvent`` objects. When
                ``None``, the ``progress`` knob decides between a tqdm bar
                and silence.
            cancel: Token the caller sets to stop the run at the next batch
                boundary.

        Returns:
            ``self``, so the call chains into ``download()`` or ``refresh()``.

        Example:
            >>> token = CancelToken()
            >>> acq.attach(reporter=NullProgressReporter(), cancel=token) is acq
            True
            >>> acq.download().last_result.cancelled
            False
            >>> acq.attach()  # clear both for the next run
        """
        self._reporter = reporter
        self._cancel_token = cancel
        # The resolved default is derived from `_reporter` and the `progress`
        # knob, so it has to be discarded whenever either could have changed.
        self._resolved_reporter = None
        return self

    def __repr__(self):
        """Return the class name and the config."""
        return f"{self.__class__.__name__}(config={self.config})"

    @property
    def config(self) -> AcquisitionConfig:
        """The acquisition config this object runs with.

        Example:
            >>> acq.config.symbols
            ('AAPL', 'MSFT')
        """
        return self._config

    @config.setter
    def config(self, config: AcquisitionConfig):
        """Assign ``config``, fill its ``name`` and default the date window.

        ``name`` becomes this class's import path so a persisted config can
        rebuild the object. A ``None`` ``start_date`` or ``end_date`` is
        replaced with the open-ended defaults ``Date.START_DATE`` and
        ``Date.END_DATE``.

        Example:
            >>> acq.config = AcquisitionConfig(..., start_date=None)
            >>> acq.config.start_date, acq.config.end_date
            ('1900-01-01', '2100-01-01')
        """
        self._config = config
        self._config.name = self.import_path

        if self._config.start_date is None:
            self._config.start_date = Date.START_DATE
        if self._config.end_date is None:
            self._config.end_date = Date.END_DATE

    @property
    def import_path(self) -> str:
        """Return the dotted import path of this object's class.

        Example:
            >>> TiingoAcquisition(config).import_path
            'quantlab.acquisition.tiingo.TiingoAcquisition'
        """
        return f"{self.__class__.__module__}.{self.__class__.__qualname__}"

    @property
    def class_name(self) -> str:
        """Return the bare class name.

        Example:
            >>> acq.class_name
            'DemoAcquisition'
        """
        return self.__class__.__name__

    @property
    def _coverage(self) -> CoverageLedger:
        """Return the ``CoverageLedger`` every coverage question goes through.

        Built per access rather than cached at config assignment, because a
        subclass may implement ``_data_type`` as an instance property reading
        a knob that a tick config supplies later; evaluating it eagerly would
        raise before the knob exists. The legacy policies are passed as
        values so a subclass that narrows them governs its own ledger.
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
        """Return the sidecar directory; see ``CoverageLedger.watermark_root``."""
        return self._coverage.watermark_root

    def _watermark_path(self, symbol: str) -> Path:
        """Return the sidecar path for ``symbol``."""
        return self._coverage.watermark_path(symbol)

    def _read_sidecar(self, symbol: str) -> dict | None:
        """Return the parsed sidecar for ``symbol``, or None if absent or corrupt."""
        return self._coverage.read_sidecar(symbol)

    def _read_watermark(self, symbol: str) -> str | None:
        """Return the last covered date for ``symbol``, or None."""
        return self._coverage.read_watermark(symbol)

    def _read_coverage(self, symbol: str) -> dict | None:
        """Return ``{"start_date", "last_date", "no_data"}`` for ``symbol``, or None."""
        return self._coverage.read_coverage(symbol)

    def _write_watermark(
        self,
        symbol: str,
        last_date: str,
        start_date: str | None = None,
        no_data: bool = False,
    ) -> None:
        """Record the covered range for ``symbol`` in its sidecar.

        ``start_date=None`` omits the key rather than writing null: an unknown
        covered start is represented by absence so it cannot be mistaken for
        a recorded value. ``no_data=False`` likewise omits its key, so a
        sidecar written before the marker existed reads back as "had data",
        which is correct because a watermark was only ever written after a
        successful fetch. Callers pass ``no_data=True`` only after a batch
        completed with no rows for the symbol.

        The write is atomic because cancellation lands at a batch boundary,
        which is exactly when this file is being rewritten, and the resumed
        run reads it.
        """
        path = self._watermark_path(symbol)
        payload: dict[str, str | bool] = {"last_date": last_date}
        if start_date is not None:
            payload["start_date"] = start_date
        if no_data:
            payload["no_data"] = True
        # `_read_sidecar` still tolerates a corrupt file by treating the symbol
        # as uncovered; keep that tolerance, since older files are on disk.
        # Compact JSON, so existing sidecars stay byte-comparable with new ones.
        write_json_atomically(path, payload)

    def stamp_watermarks(self, start_date: str) -> int:
        """Fill ``start_date`` into every sidecar that records no covered start.

        This is the explicit migration for sidecars written before the
        covered start was recorded. The start comes from the caller and is
        derived from nothing else, because only the operator knows what
        window those files were fetched over. Sidecars that already record a
        start are left untouched, and an existing ``no_data`` marker is
        carried through unchanged. No vendor requests are issued.

        Args:
            start_date: The covered start to record, as ``YYYY-MM-DD``.

        Returns:
            How many sidecars were rewritten.

        Example:
            >>> acq.stamp_watermarks("2020-01-01")
            1
            >>> acq.stamp_watermarks("2020-01-01")  # nothing left to stamp
            0
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

    #: How many symbols one ``_fetch_page`` request carries. A vendor whose
    #: API takes one symbol per call sets 1, and the batched path then has the
    #: same request count, failure granularity and resume granularity as a
    #: per-symbol loop. Overridable per run via ``config.kwargs["batch_size"]``.
    DEFAULT_BATCH_SIZE: int = 1

    #: Which vendor this subclass fetches from. Set by every subclass. It is
    #: written into every shard as a literal ``vendor`` column and hashed into
    #: every ``PageLedger.batch_key``, so it is part of the on-disk contract.
    VENDOR: Vendor

    #: The projection and order of every raw shard this subclass writes.
    #: ``_write_shard`` selects exactly these columns, so every shard under a
    #: vendor root has the same schema. A directory scan derives one schema
    #: from the first file it opens and enforces it across all of them, so a
    #: single shard with an extra column, or the same columns in a different
    #: order, would make the whole vendor root unreadable.
    RAW_COLUMNS: tuple[str, ...]

    #: The IANA time zone whose calendar day the intraday ``date=`` hive key
    #: is derived from, that is, the trading session's own day boundary.
    #: ``None`` means not declared, and ``_session_date`` raises on it rather
    #: than falling back to a UTC date. Only intraday frequencies read it, so
    #: a daily-only vendor never has to set it. Timestamp values in the shards
    #: stay naive UTC; only the derived partition key converts.
    SESSION_TIME_ZONE: str | None = None

    # -- orchestration constants -------------------------------------------

    #: Concurrent in-flight batch fetches. Overridable per run via
    #: ``config.kwargs["max_workers"]``.
    DEFAULT_MAX_WORKERS = 8

    #: Filename of the failure manifest, written under ``config.watermark_path``
    #: next to the per-symbol sidecars. Bound from ``quantlab.base.coverage``,
    #: where it is declared, so the writer here and the credential-free
    #: reader name the same file.
    FAILURE_MANIFEST_NAME = _FAILURE_MANIFEST_NAME

    #: What a credential value is replaced with in any captured message.
    #: Subclasses may override it with vendor-specific wording.
    REDACTION = "<CREDENTIAL REDACTED>"

    #: The environment variables whose values ``_scrub`` redacts. A subclass
    #: populates this from module-level constants, never by reaching through
    #: its transport class: a test double that replaces the transport must not
    #: be able to switch the redaction off. An empty tuple means the vendor
    #: names no credentials.
    CREDENTIAL_ENV_VARS: tuple[str, ...] = ()

    #: How many failed symbols are named in the summary log line. The full
    #: set always lands in the manifest; the log line is a pointer to it.
    _FAILURE_LOG_SAMPLE = 5

    #: Accepted values of ``config.kwargs["legacy_watermarks"]``. ``"warn"``
    #: skips a sidecar with no recorded covered start but reports it on every
    #: run; ``"refetch"`` treats unknown coverage as uncovered. Bound from
    #: ``quantlab.base.coverage`` so the credential-free ledger uses the same
    #: policy set; kept as class attributes because command-line shells read
    #: them for their argument choices and a subclass may narrow them.
    LEGACY_WATERMARK_POLICIES = _LEGACY_WATERMARK_POLICIES
    DEFAULT_LEGACY_WATERMARK_POLICY = _DEFAULT_LEGACY_WATERMARK_POLICY

    #: The command that resolves an un-stamped legacy watermark, quoted in
    #: the warning so the reported gap comes with its cure.
    STAMP_COMMAND_HINT = (
        "uv run python ingest_us_equity.py --stamp-legacy-watermarks <START_DATE>"
    )

    #: Whether a run that exhausted the vendor's request allocation waits for
    #: it to reset and resumes. Off by default so no run silently holds a
    #: vendor's allocation window open.
    DEFAULT_WAIT_FOR_QUOTA = False

    #: Delay between resume attempts, in seconds. Vendors do not publish
    #: whether their allocation resets on a fixed hourly bucket or a rolling
    #: window, so this is a configured interval rather than a computed reset
    #: instant; one hour covers both readings.
    DEFAULT_QUOTA_WAIT_SECONDS = 3600

    #: Maximum resume attempts after an allocation abort. Bounded, because an
    #: unbounded loop against a lockout is worse than the problem it fixes.
    DEFAULT_QUOTA_MAX_WAITS = 3

    #: HTTP statuses this vendor treats as a transient rate limit: the worker
    #: backs off and retries the same batch without touching the global abort.
    #: Empty by default; the base makes no claim about any vendor's status
    #: codes, and the same code can mean opposite things for two vendors (a
    #: per-minute ceiling that clears in seconds versus an hourly allocation
    #: whose correct response is to stop the run).
    RATE_LIMIT_STATUS_CODES: frozenset[int] = frozenset()

    #: How long, in seconds, a rate-limited batch waits before retrying. A
    #: working value for a per-minute ceiling, not a vendor fact; the code
    #: never computes a reset instant from headers. Overridable per run via
    #: ``config.kwargs["rate_limit_backoff_seconds"]``.
    DEFAULT_RATE_LIMIT_BACKOFF_SECONDS = 5.0

    #: How many consecutive backoffs one batch gets before it degrades to
    #: ``failed`` and goes to the manifest for the next run to retry. At the
    #: default backoff this is about 30 seconds of patience. Overridable via
    #: ``config.kwargs["rate_limit_max_retries"]``.
    DEFAULT_RATE_LIMIT_MAX_RETRIES = 6

    def _knob(self, name: str, default=None):
        """Read a per-run tuning parameter from ``config.kwargs``.

        Every knob goes through here rather than becoming a constructor
        argument, which keeps a run fully describable by its config file.
        """
        return (self.config.kwargs or {}).get(name, default)

    # -- global abort, backoff seam and credential scrubbing ----------------

    @property
    def _abort(self) -> threading.Event:
        """Return the global stop flag shared across every worker thread.

        ``threading.Event`` is thread-safe on its own, and its
        ``wait(timeout)`` is the primitive the resume delay wants. Created
        lazily so ``_attempt_batch`` can be called directly without a run
        having set one up.
        """
        event = getattr(self, "_abort_event", None)
        if event is None:
            event = self._abort_event = threading.Event()
        return event

    def _reset_abort(self) -> threading.Event:
        """Install a fresh abort event so an earlier pass's trip cannot leak."""
        self._abort_event = threading.Event()
        return self._abort_event

    def _sleep(self, seconds: float) -> None:
        """Sleep for ``seconds``; a seam tests replace to count waits instead."""
        time.sleep(seconds)

    def _is_cancelled(self) -> bool:
        """Return whether the caller's cancel token has been set.

        Deliberately asks about the cancel token alone, so the loop can report
        "the operator stopped this" separately from "the vendor stopped this".
        ``_should_stop()`` combines the two.
        """
        token = getattr(self, "_cancel_token", None)
        return token is not None and token.is_cancelled()

    def _should_stop(self) -> bool:
        """Return whether the next batch should be skipped.

        The OR of the two independent stop conditions, the vendor's quota
        abort and the caller's cancel token. They are combined for the
        decision and kept separate for reporting, so a cancel is never
        logged, waited on or resumed as if the vendor had run out of
        allocation.
        """
        return self._abort.is_set() or self._is_cancelled()

    # -- progress reporting -------------------------------------------------

    @property
    def _active_reporter(self) -> ProgressReporter:
        """Return the reporter this run's events go to.

        Resolution order: whatever ``attach()`` was given; else a
        ``TqdmProgressReporter`` when the ``progress`` knob is truthy (its
        default); else a ``NullProgressReporter``. The result is cached
        because the tqdm reporter owns one bar across a whole pass, so
        re-resolving per event would open a new bar per batch. ``attach()``
        invalidates the cache.
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
        """Deliver one event to the active reporter without ever raising.

        The reporter's callback runs inside the result loop of ``_run_once``,
        so an exception from it would tear down the whole fan-out and end a
        multi-hour backfill over a UI bug. The exception is logged at warning
        level, scrubbed like every other captured message, and otherwise
        ignored.
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
        """Close the active reporter without ever raising.

        Same contract as ``_emit``: ``tqdm.close()`` writes to stderr, and a
        closed or redirected stream must not be what ends a run.
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
        """Zero this pass's ``no_data`` tally and give it a fresh lock.

        Per pass, like ``_reset_abort``, so a resumed pass reports the
        markers it wrote rather than inheriting the previous pass's count.
        """
        self._no_data_marks = 0
        self._no_data_lock = threading.Lock()

    def _record_no_data_marks(self, count: int) -> None:
        """Add ``count`` markers written by one batch to the pass tally.

        Reporting only; nothing branches on the total. The lock exists because
        ``_attempt_batch`` runs on several threads. When no lock has been set
        up the method was reached outside a run, where there is no fan-out to
        race with.
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
        """Replace every declared credential value in ``message``.

        A vendor's HTTP error text commonly echoes the full request URL, and
        some vendors carry the API token as a query parameter, so every
        captured message passes through here before it is logged or written.
        That is what makes the failure manifest safe to paste into an issue.
        The per-vendor part is data (``CREDENTIAL_ENV_VARS``), so a new vendor
        inherits the control rather than reimplementing it.
        """
        for name in self.CREDENTIAL_ENV_VARS:
            value = os.environ.get(name)
            if value:
                message = message.replace(value, self.REDACTION)
        return message

    # -- error classification: the one per-vendor policy seam ---------------

    @staticmethod
    def _vendor_response(exc: BaseException):
        """Return the ``requests.Response`` reachable from ``exc``, or None.

        Walks the exception and its ``args`` rather than reading
        ``exc.response`` alone, because at least one vendor client re-raises
        the underlying ``HTTPError`` wrapped in its own exception type, where
        the response lives at ``exc.args[0].response``. The walk also covers
        the direct ``HTTPError`` a plain ``raise_for_status()`` produces.
        Shared across vendors so two walks of the same shape cannot disagree.
        """
        for candidate in (exc, *getattr(exc, "args", ())):
            response = getattr(candidate, "response", None)
            if response is not None and getattr(response, "status_code", None):
                return response
        return None

    def _status_of(self, exc: BaseException) -> int | None:
        """Return the HTTP status reachable from ``exc``, or None if none.

        An exception carrying no status must read as ``None`` rather than as
        a default, because an invented status would be classified with full
        confidence and be wrong.
        """
        response = self._vendor_response(exc)
        if response is None:
            return None
        status = getattr(response, "status_code", None)
        return int(status) if status else None

    def _rate_limit_headers(self, exc: BaseException) -> dict[str, str]:
        """Return whatever rate-limit headers the vendor sent, for logging.

        The base returns ``{}``, which is the complete answer for a vendor
        that publishes no such headers. An absent header must contribute no
        entry rather than a default one, so a caller can tell "the vendor
        said nothing" from "the vendor said zero". Nothing branches on the
        result; the backoff stays a configured constant.
        """
        return {}

    def _classify_error(self, exc: BaseException) -> str:
        """Map a vendor failure onto one of the three orchestration verdicts.

        Returns ``"failed"`` for a per-batch problem (the batch lands in the
        failure manifest, gets no watermark and is retried next run while
        every other batch continues), ``"quota"`` for a global condition that
        is slow to clear (trips the shared abort and is excluded from the
        manifest, because recording it as one symbol's fault would defame a
        good symbol), or ``"rate_limited"`` for a transient one (the worker
        backs off and retries the same batch).

        The base default is the conservative one: everything is per-batch
        unless the status is in ``RATE_LIMIT_STATUS_CODES``. A vendor whose
        failures were wrongly read as global would abort a full-market run
        over one bad ticker, and two vendors can read the same status code
        oppositely, so this method is what varies per vendor rather than the
        orchestration around it.
        """
        if self._status_of(exc) in self.RATE_LIMIT_STATUS_CODES:
            return "rate_limited"
        return "failed"

    # -- symbol validation --------------------------------------------------

    def _validate_symbols(self, symbols: Sequence[str]) -> list[str]:
        """Return ``symbols`` as a list, refusing any that fail the pattern.

        Delegates to ``CoverageLedger.validate_symbols`` so the credential-free
        inspector validates through the same code before building any path.
        The error text names ``self.class_name``.
        """
        return self._coverage.validate_symbols(symbols)

    # -- raw shard layout ---------------------------------------------------

    @property
    def _hive_keys(self) -> tuple[str, ...]:
        """Return the hive partition keys for this config's frequency.

        Read from ``RAW_HIVE_KEYS``, the same mapping the dataset reader uses,
        so the writer and the reader cannot drift.
        """
        return RAW_HIVE_KEYS[self.config.frequency]

    def _session_date(self, expr: pl.Expr) -> pl.Expr:
        """Return the session date an intraday ``date=`` hive key is derived from.

        The default converts the naive-UTC timestamp expression into
        ``SESSION_TIME_ZONE`` and truncates to a date. A vendor whose
        timestamps are already session-local overrides this with a plain
        ``expr.dt.date()``. This is a hook rather than a vendor check because
        the day boundary is a property of the vendor's timestamp convention,
        which the base class does not know.

        Declaring no time zone raises instead of truncating in UTC. A UTC key
        would file the last hours of every US session under the following
        day, and a one-trading-day query would then be wrong at both edges in
        a way that reads as sparse data rather than as a bug.

        Raises:
            NotImplementedError: If ``SESSION_TIME_ZONE`` is ``None``.
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
        """Return which of the vendor's data types this run fetches, or None.

        ``None`` on the base, because a vendor with one shape of data per
        frequency has no such concept. A multi-type vendor overrides this,
        validates the value against its own accepted set, and uses the same
        resolved value for both the endpoint and the written partition so
        the two cannot disagree. ``_hive_key_expr`` raises if a frequency
        that partitions on ``data_type`` reaches it with nothing declared.
        """
        return None

    def _hive_key_expr(self, key: str) -> pl.Expr:
        """Return the expression that derives one hive key from a raw frame.

        Keyed by key name rather than by frequency, so ``RAW_HIVE_KEYS`` stays
        the only place the per-frequency key tuples are declared and adding a
        frequency that reuses existing keys needs no change here.

        Raises:
            ValueError: If ``key`` is ``data_type`` and ``_data_type`` is None.
            NotImplementedError: If ``key`` has no derivation here.
        """
        if key == "month":
            # `YYYY-MM` as a string, which is what the reader's `hive_schema`
            # pins. ISO ordering makes a plain string comparison against the
            # window edges correct, so no date parsing or time zone is needed.
            return pl.col("timestamp").dt.strftime("%Y-%m")
        if key == "date":
            # The intraday key, through the session-date seam above.
            return self._session_date(pl.col("timestamp"))
        if key == "symbol":
            # Already a real column on every raw frame; `_validate_symbols`
            # has run before any frame reaches here, so the value cannot
            # escape the raw root as a path segment.
            return pl.col("symbol")
        if key == "data_type":
            # A per-run constant, not a per-row derivation: one fetch asks one
            # endpoint for one data type. It is the leading key so two column
            # sets never meet inside one directory scan.
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
        """Add the hive key columns this frequency partitions on.

        The columns are added in the order ``RAW_HIVE_KEYS`` declares them,
        which is the directory nesting order ``_shard_path`` zips against.
        """
        return frame.with_columns(
            self._hive_key_expr(key).alias(key) for key in self._hive_keys
        )

    def _shard_path(
        self, partition_values: Sequence[str], batch_key: str, page_index: int
    ) -> Path:
        """Return ``{raw_root}/{k1}={v1}/.../part-{batch_key}-{page:05d}.pqt``.

        The name is deterministic, with no timestamp, uuid or counter, so a
        page re-fetched after a crash overwrites its shard instead of adding
        a second one. The overwrite holds within one window only:
        ``batch_key`` hashes the start and end dates, so the same rows
        re-fetched over a different window (every ``refresh()``, whose
        inclusive ``last_date`` re-requests the final session) land under a
        different name in the same partition directory. Daily and minute
        data absorb that through deduplication on read; tick data is never
        deduplicated, so ``_write_shard`` removes the superseded shards
        instead.
        """
        directory = Path(self.config.raw_data_dir_path)
        for key, value in zip(self._hive_keys, partition_values):
            directory = directory / f"{key}={value}"
        return directory / f"part-{batch_key}-{page_index:05d}.pqt"

    @property
    def _partition_is_per_symbol(self) -> bool:
        """Return whether a partition directory belongs to exactly one symbol.

        Derived from the hive keys rather than from the frequency, so a layout
        that gains or loses the ``symbol=`` key gets the right answer without
        a second edit. This is what makes ``_clear_superseded_shards`` safe.
        """
        return "symbol" in self._hive_keys

    def _clear_superseded_shards(self, directory: Path, batch_key: str) -> None:
        """Delete shards in ``directory`` written under a different batch key.

        ``refresh()`` re-requests the final session with a new start date and
        therefore a new batch key, so the same rows land under a second
        filename beside the first. Tick data is deliberately never
        deduplicated, because genuine quotes and trades share
        ``(timestamp, symbol)``, which makes such a duplicate undetectable
        afterwards: a doubled trade tape looks like a busy one. Removing the
        earlier key's shards is the cure.

        Only sound when the partition directory belongs to one symbol, hence
        the gate at the call site. A ``month=`` or ``date=`` directory is
        shared by every concurrently running batch, so deleting another key's
        file there would destroy a sibling batch's data. The filename
        contract is unchanged; a superseded name is removed rather than left
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

        The frame is projected through ``RAW_COLUMNS`` first, so every shard
        under a vendor root carries the same columns in the same order. Only
        parquet is written beneath the raw root: a directory scan walks every
        file it finds, and a stray ``.json`` there would break it.

        Returns:
            The written paths, which the page ledger records so a resume can
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
                # A re-fetch over a different window writes a second file
                # rather than overwriting; tick data is never deduplicated on
                # read, so remove the superseded shards here. Only safe where
                # the directory belongs to one symbol.
                self._clear_superseded_shards(path.parent, batch_key)
            group.drop(keys).write_parquet(path)
            written.append(str(path))
        return written

    # -- batching and pagination -------------------------------------------

    def _batches(self, symbols: Sequence[str]) -> Iterator[list[str]]:
        """Yield consecutive chunks of ``batch_size`` symbols."""
        size = max(1, int(self._knob("batch_size", self.DEFAULT_BATCH_SIZE)))
        symbols = list(symbols)
        for index in range(0, len(symbols), size):
            yield symbols[index : index + size]

    def _refresh_batches(self, pending: Sequence[str]) -> Iterator[list[str]]:
        """Yield refresh batches grouped by recorded ``last_date``.

        One request carries exactly one start date, so a refresh groups
        symbols that share a watermark and chunks each group by
        ``batch_size``. This is request packing only: a refresh still
        fetches ``[watermark, config.end_date]`` per symbol. Grouping beats
        issuing the earliest start for a mixed batch, which would re-fetch
        history nobody asked for. Symbols with no watermark bucket under
        ``config.start_date``, the start ``_attempt_batch`` derives for them,
        and insertion order is preserved so a run is reproducible.
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
        """Return the ``(ledger, batch_key)`` pair identifying one batch."""
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
        """Fetch one batch page by page, resuming where a previous run stopped.

        This is the only place pagination is implemented. Every vendor
        implements ``_fetch_page`` and inherits this loop; a vendor with no
        pagination returns ``None`` as the next token on every call, which
        ends the loop after one page.

        Each page's shard is written before the ledger records the page. A
        crash between the two costs a re-fetch and a deterministic overwrite,
        never a duplicated row or a lost page; the reverse order would record
        pages whose data is not on disk. An exception from ``_fetch_page``
        propagates, but only after every page that did complete has been
        recorded, so the next run resumes instead of restarting.

        Args:
            symbols: The batch, validated here before any path is built.
            start_date: First date to request, inclusive.
            end_date: Last date to request, inclusive.
            ledger: The batch's page ledger, built from the other arguments
                when omitted.
            batch_key: The ledger's batch key, built when omitted.

        Raises:
            ValueError: If the vendor returns the same page token it was
                given, which would otherwise loop forever writing a new shard
                per iteration.
        """
        symbols = self._validate_symbols(symbols)
        if ledger is None or batch_key is None:
            ledger, batch_key = self._ledger_for(symbols, start_date, end_date)

        # The ledger and the shard tree are independent records of the same
        # truth. Refuse to resume onto a disagreement rather than skip a hole.
        ledger.assert_consistent(self.config.raw_data_dir_path)

        if ledger.is_complete():
            # A completed ledger is not a veto. Reaching `_fetch_batch` means
            # the symbol-level policy already decided this batch should run
            # (`download(resume=False)`, for instance); the ledger only says
            # where within the batch to resume. Starting over is safe because
            # page N writes the same deterministic path and overwrites it.
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
                # A falsy token is the loop's only exit, so a vendor that
                # echoes the token it was handed would run forever, and since
                # `page_index` increments each iteration nothing overwrites:
                # the raw root would fill the disk while every request looked
                # successful. Refusing leaves a resumable ledger behind.
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
        """Backfill ``[config.start_date, config.end_date]`` for the roster.

        Symbols whose sidecar already covers the window are skipped unless
        ``config.kwargs["resume"]`` is false. A widened ``start_date``
        re-fetches the symbols whose recorded coverage starts later.

        Args:
            symbols: The symbols to fetch; defaults to ``config.symbols``.

        Returns:
            ``self``; the outcome is on ``last_result``.

        Example:
            >>> acq.download().last_result.succeeded
            ('AAPL', 'MSFT')
            >>> acq.download(["AAPL"]).last_result.coverage["skipped"]
            1
        """
        return self._run(symbols, from_watermark=False)

    def refresh(self, symbols: list[str] | None = None) -> Self:
        """Fetch each symbol forward from its own watermark.

        A refresh fetches ``[last_date, config.end_date]`` per symbol, so the
        covered start stays whatever it already was, and if it was unknown it
        stays unknown. It never honours a widened ``config.start_date``;
        widening the covered range is ``download()``'s job.

        Args:
            symbols: The symbols to refresh; defaults to ``config.symbols``.

        Returns:
            ``self``; the outcome is on ``last_result``.

        Example:
            >>> acq.config.end_date = "2024-01-08"
            >>> acq.refresh().last_result.succeeded
            ('AAPL', 'MSFT')
        """
        return self._run(symbols, from_watermark=True)

    # -- concurrent, resumable, failure-isolated orchestration --------------
    #
    # A full-market backfill is tens of thousands of symbols and several
    # hours, so the loop below is concurrent (threads, since the work is
    # network-bound and the vendor client is shared), resumable (each pass
    # recomputes what is pending from the sidecars on disk) and
    # failure-isolated (exceptions are captured per batch, with a global
    # abort for vendor-wide conditions). Which exception means which of
    # those things is the one thing that stays per vendor: see
    # `_classify_error`.

    def _run(self, symbols: list[str] | None, from_watermark: bool) -> Self:
        """Run the concurrent loop both entry points delegate to.

        ``download()`` and ``refresh()`` differ only in how each batch's start
        date is resolved, so they share this body. A bounded resume loop
        wraps the fan-out: each pass recomputes ``pending`` from the sidecars
        on disk, so the resume logic and the skip logic are the same code.
        After a quota abort the loop waits ``quota_wait_seconds`` and retries
        up to ``quota_max_waits`` times, but only when ``wait_for_quota`` is
        set; a vendor's reset semantics are not published, so the wait is a
        configured interval rather than a computed instant.

        On every way out the failure manifest on disk is merged with entries
        for symbols this run never reached and rewritten, and ``last_result``
        is set. Returns ``self`` so the entry points chain.
        """
        # Validated here, not only inside `_fetch_batch`: `_partition_by_coverage`
        # below turns every requested symbol into a sidecar path before any
        # batch exists, so a malformed roster entry must be refused first.
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
        # Accumulated across resume passes, like `failures`: each pass
        # re-derives `pending` from disk, so the last pass alone knows nothing
        # about what an earlier pass completed.
        all_succeeded: set[str] = set()
        # Pre-set so the result is well-defined when `pending` is empty on the
        # first pass and `_run_once` never runs.
        aborted = False
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
            # Merged, not replaced: a pass that aborts before reaching an
            # earlier pass's failures returns `{}`, and reassigning would then
            # write an empty manifest for a run that had real failures.
            # Symbols that succeeded this pass leave the dict; symbols merely
            # skipped by an abort are left alone, since this pass has no news
            # about them.
            for symbol in succeeded:
                failures.pop(symbol, None)
            all_succeeded.update(succeeded)
            failures.update(pass_failures)
            if cancelled:
                # Checked before the quota branch and never falls through into
                # it: a cancel must not wait, log the allocation message, or
                # resume the run the operator just stopped.
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

        # Two dicts on purpose. The manifest on disk is the cross-run record
        # of every symbol known to be failing; `AcquisitionResult.failures` is
        # what this run discovered and stays inside `requested`. Carried-
        # forward entries are folded into the manifest copy only, otherwise a
        # run over a disjoint roster would report earlier runs' failures as
        # its own.
        manifest = dict(failures)
        # Unconditional and outside the loop, so it runs on every exit (cancel,
        # empty `pending`, normal completion, abort without waiting, abort
        # after the last wait) including any `break` added later. `attempted`
        # is derived from `failures`, not `manifest`: it is the set this run
        # has news about.
        self._merge_unattempted_failures(
            manifest, attempted=all_succeeded | set(failures)
        )
        self._write_failure_manifest(manifest)
        # `failures`, not `manifest`: the result is this run's own report and
        # its `coverage` covers `requested` only.
        self.last_result = AcquisitionResult(
            vendor=self.VENDOR,
            requested=tuple(requested),
            succeeded=tuple(sorted(all_succeeded)),
            failures=dict(failures),
            cancelled=cancelled,
            quota_aborted=aborted,
            coverage=self.coverage_report(requested),
        )
        # `Self`, not the result, so `download()`/`refresh()` chain; the
        # registry reads the result off `last_result`.
        return self

    def _run_once(
        self, pending: list[str], from_watermark: bool
    ) -> tuple[bool, dict[str, str], set[str], bool]:
        """Run one concurrent pass over ``pending``.

        The unit of work is a batch, not a symbol; with ``batch_size`` 1 the
        two coincide. Batches are dispatched to ``max_workers`` threads and
        their results streamed back so the progress bar advances as batches
        land. The result generator is always drained to completion rather
        than broken out of: once the abort is set every remaining batch is a
        microsecond no-op, and draining makes worker teardown deterministic.

        Returns:
            ``(quota_aborted, failures, succeeded, cancelled)``. ``failures``
            maps each failed symbol to its scrubbed message and ``succeeded``
            holds the symbols whose batches completed; a symbol skipped by
            the abort is in neither, because this pass learned nothing about
            it. ``quota_aborted`` and ``cancelled`` are separate because
            ``_run`` may wait and resume after the former but never after the
            latter.
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
            """Yield the batches to dispatch, stopping early once a stop is set.

            The early stop is an optimisation only: joblib pre-dispatches
            several batches ahead of the first result, so a small run has
            every batch queued before this check can fire. The first
            statement of ``_attempt_batch`` is what actually stops the vendor
            requests.

            Example:
                >>> list(inputs())
                [['AAPL', 'MSFT'], ['GOOG']]
            """
            for batch in batches:
                if self._should_stop():
                    break
                yield batch

        # `return_as="generator_unordered"` is load-bearing. The default eager
        # call returns only once every batch is done, so a bar around it would
        # render nothing for hours and then fill; streaming the results is the
        # only form where one tick means one batch landed on disk.
        stream = Parallel(
            n_jobs=max_workers, backend="threading", return_as="generator_unordered"
        )(delayed(self._attempt_batch)(batch, from_watermark) for batch in inputs())

        # The reporter builds the bar from this event (same total, description
        # and unit as a bar constructed here would have).
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
            # Drained to completion, never broken out of; see the docstring.
            # The description is switched on abort so a racing bar cannot
            # read as "all this work succeeded".
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
                # Announced separately from the quota switch, so a console can
                # tell an operator stop from a vendor stop.
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
            # In a `finally` so an exception escaping the drain still closes
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

        # Reported from this pass while it is fresh; the on-disk total is
        # reported separately by `_report_coverage` at the top of every pass.
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

        # Read once, after the drain. A cancel does not set `abort`, so the
        # early return below keeps a cancelled run from being described as
        # the vendor running out of allowance.
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
        """Classify the roster against the config window without fetching.

        This is the read-only form of the pending computation ``_run`` makes,
        sharing ``_partition_by_coverage`` with it so the two cannot
        disagree. Use it to ask whether widening the window would re-fetch
        anything before committing to a multi-hour job, or to confirm that
        ``stamp_watermarks`` cleared every legacy sidecar.

        Args:
            symbols: The symbols to classify; defaults to ``config.symbols``.

        Returns:
            Counts under ``requested``, ``pending``, ``skipped``, ``covered``,
            ``widened``, ``legacy`` and ``no_data``.

        Example:
            >>> acq.coverage_report()
            {'requested': 2, 'pending': 0, 'skipped': 2, 'covered': 2,
             'widened': 0, 'legacy': 0, 'no_data': 0}
            >>> acq.coverage_report(["AAPL"])["pending"]
            0
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
        """Return the resolved ``legacy_watermarks`` policy."""
        return self._coverage.legacy_policy()

    def _coverage_status(self, symbol: str, from_watermark: bool = False) -> str:
        """Return ``symbol``'s coverage state; see ``CoverageLedger``."""
        return self._coverage.coverage_status(symbol, from_watermark)

    def _classify_coverage(
        self, coverage: dict | None, from_watermark: bool = False
    ) -> str:
        """Classify a parsed sidecar against the config window."""
        return self._coverage.classify_coverage(coverage, from_watermark)

    def _covers(self, symbol: str, from_watermark: bool = False) -> bool:
        """Return whether ``symbol``'s sidecar already covers the window."""
        return self._coverage.covers(symbol, from_watermark)

    def _partition_by_coverage(
        self, requested: list[str], from_watermark: bool
    ) -> tuple[list[str], dict[str, int]]:
        """Split ``requested`` into the pending symbols and per-state counts.

        Delegates to ``CoverageLedger.partition_by_coverage``, the one
        function ``coverage_report()``, ``_run`` and the credential-free
        inspector all reach, so every caller gets the same answer.
        """
        return self._coverage.partition_by_coverage(requested, from_watermark)

    def _report_coverage(
        self, requested: list[str], pending: list[str], counts: dict[str, int]
    ) -> None:
        """Emit a ``coverage`` event and log each partition count separately.

        Each outcome gets its own line so widening the window, or a vendor
        having nothing to give, has a visible and countable consequence.
        """
        skipped = len(requested) - len(pending)
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
        """Fetch one batch and return ``(symbols, status, message)``.

        ``status`` is ``"ok"`` (fetched, one watermark written per symbol),
        ``"failed"`` (a per-batch problem such as a delisted ticker's 404;
        every symbol in the batch goes to the manifest with no watermark and
        is retried next run), ``"quota"`` (the vendor's allocation is gone;
        trips the global abort and is kept out of the manifest so a good
        symbol is not recorded as failing) or ``"skipped"`` (never attempted
        because a stop was already set; no watermark, not a failure).
        ``message`` is the scrubbed exception text for the failure verdicts
        and ``None`` otherwise.

        Never raises: a propagated exception would tear down the whole
        fan-out and abort every other batch. A rate-limited batch backs off
        in this worker and retries up to ``rate_limit_max_retries`` times
        before degrading to ``"failed"``.
        """
        # First statement on purpose. joblib cannot cancel work it has already
        # queued, so this check, not the input generator, is what stops the
        # vendor requests once the abort or the cancel token is set. Every
        # remaining batch returns here in microseconds having issued nothing,
        # and because it fires at a batch boundary a cancelled run is
        # resumable: every batch past this line wrote its watermarks and
        # every batch that did not has no sidecar. The cancel reuses the
        # `"skipped"` status; which condition stopped the run is answered at
        # the `_run_once` level.
        if self._should_stop():
            return list(symbols), "skipped", None

        symbols = list(symbols)
        coverages = {
            symbol: (self._read_coverage(symbol) or {}) for symbol in symbols
        }

        # The start to request. A full backfill asks for `config.start_date`;
        # a refresh asks from the batch's recorded watermark forward.
        # `_refresh_batches` groups by identical `last_date`, so `min` is the
        # defensive reading of that invariant: over-fetching a shared window
        # is a harmless overwrite, under-fetching would silently lose rows.
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
                    # Transient and fast to clear: back off in this worker
                    # only. Setting the global abort here would stop every
                    # other batch over a condition that has usually cleared
                    # by the time the log line is written.
                    if retries == 0:
                        # Logged once per batch, on the first backoff only;
                        # otherwise this would be the noisiest line in the
                        # run. The headers are the only diagnostic a 429
                        # produces, and they are what lets an operator check
                        # that the ceiling hit is really per-minute.
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
                # Out of retries: degrade to `failed` so it goes to the
                # manifest and the next run retries it, instead of this run
                # never finishing.
                return symbols, "failed", message
            break

        # The "queried, no data" set, computed once for the whole batch and
        # never per page. The vendor sorts symbol-major, so page 0 of a
        # hundred-symbol batch may carry one symbol; a per-page difference
        # would mark the other 99 as empty and skip them forever.
        # `_fetch_batch` accumulates `symbols_with_data` across every page,
        # seeded from the ledger, and this is the only place the difference
        # is taken. Two gates: an incomplete batch has no opinion about
        # absence, and a global stop is not the moment to record new claims
        # about what a vendor does not have. A batch that raised never
        # reaches here.
        marked: set[str] = set()
        if outcome.complete and not self._abort.is_set():
            marked = set(symbols) - outcome.symbols_with_data

        recorded = 0
        for symbol in symbols:
            # The covered start to record. A full backfill overwrites the
            # shards wholesale, so the requested start is true of them; a
            # refresh only extends forward, so it carries each symbol's
            # existing start through, unknown stays unknown.
            covered_start = (
                coverages[symbol].get("start_date")
                if from_watermark
                else self.config.start_date
            )
            # The same asymmetry for the marker. A full backfill queried
            # exactly the window it records. A refresh queried only
            # `[watermark, end_date]`, so it may clear a marker (rows arrived)
            # but never assert a new one, having no evidence about the
            # earlier part of the recorded range.
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
        """Fold manifest entries for symbols this run never reached into ``failures``.

        ``_write_failure_manifest`` rewrites the file whole and each run
        starts with an empty dict, so without this step any run that stopped
        before reaching a symbol that failed last time, or that asked for a
        different roster altogether, would erase the operator's only record
        that the symbol is still failing. Entries for symbols in
        ``attempted`` (succeeded or failed this run) are not restored: a
        success must leave the manifest and a failure already carries this
        run's message.

        ``failures`` is the manifest copy, not the run's own dict, so the
        run's ``AcquisitionResult`` is never widened with symbols it did not
        touch. The cost is that a stale entry for a symbol that left the
        universe lingers until it is cleared by hand, which is the smaller
        harm.
        """
        for symbol, message in self._coverage.read_failure_manifest().items():
            if symbol in attempted:
                continue
            failures.setdefault(symbol, message)

    def _write_failure_manifest(self, failures: dict[str, str]) -> None:
        """Write ``{symbol: message}`` as the store's cross-run failure record.

        The file is rewritten whole, but the caller has already merged in the
        on-disk entries this run had no news about, so what lands is a
        cross-run record rather than a per-run artifact. A failed symbol has
        no watermark, so the next run puts it back in ``pending`` and it
        reappears here if it fails again. The write is atomic because a
        cancel lands at a batch boundary and this file is rewritten on the
        way out; a truncated manifest would silently name fewer symbols than
        the record holds.
        """
        # Through the ledger rather than re-joining the path here, so the
        # tick `data_type` namespacing cannot be forgotten on the writing side.
        path = self._coverage.failure_manifest_path
        # `indent=2, sort_keys=True`: a human reads this file.
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
        """Issue one vendor request and return ``(rows, next_page_token)``.

        This is the single abstract fetch primitive. ``rows`` is a
        ``pl.DataFrame`` carrying the vendor's raw columns plus ``timestamp``,
        ``symbol`` and ``vendor``, which ``_write_shard`` projects onto
        ``RAW_COLUMNS`` and persists under ``config.raw_data_dir_path`` in the
        schema the matching dataset class expects. A ``None`` token means
        this was the last page for ``symbols``; a vendor with no pagination
        returns ``None`` on every call, which is a complete implementation.
        Implementations must not touch xarray or Zarr storage.

        Args:
            symbols: The batch to request.
            start_date: First date to request, inclusive.
            end_date: Last date to request, inclusive.
            page_token: The token the previous page returned, or ``None`` for
                the first page.
        """
        ...
