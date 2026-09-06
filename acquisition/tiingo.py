import json
import os
import threading
import time
from pathlib import Path
from typing import Self

import polars as pl
from joblib import Parallel, delayed
from loguru import logger
from tiingo import TiingoClient
from tqdm import tqdm

from base.acquisition import Acquisition
from base.config import AcquisitionConfig
from enums.data import TiingoColumns

# Extend when intraday frequencies are added -- never hardcode "daily" inline
# in _fetch_and_write.
_FREQUENCY_MAP = {"1d": "daily"}


class TiingoAcquisition(Acquisition):
    """Config-driven, incrementally-refreshable Tiingo EOD data acquisition.

    `TIINGO_API_KEY` is read directly from `os.environ` in `__init__` and
    passed only into the in-memory `TiingoClient` constructor argument --
    never assigned to `self.config` or any other dataclass-facing attribute.
    """

    def __init__(self, config: AcquisitionConfig):
        super().__init__(config)

        if not os.environ.get("TIINGO_API_KEY"):
            raise RuntimeError(
                "TIINGO_API_KEY environment variable is not set. Export it "
                "before running acquisition (see Tiingo dashboard for your "
                "key)."
            )
        self._client = TiingoClient(
            {"session": True, "api_key": os.environ["TIINGO_API_KEY"]}
        )

    def _fetch_and_write(
        self, symbol: str, start_date: str, end_date: str
    ) -> None:
        frequency = _FREQUENCY_MAP[self.config.frequency]
        response = self._client.get_ticker_price(
            symbol,
            fmt="json",
            startDate=start_date,
            endDate=end_date,
            frequency=frequency,
            columns=TiingoColumns.EOD,
        )
        data = pl.DataFrame(response)
        if data.is_empty():
            return

        # Tiingo's `date` field is an ISO-8601 string with a trailing `Z`
        # (UTC) offset (e.g. "2024-01-02T00:00:00.000Z"). Parse it as UTC
        # then drop the tz so the resulting dtype is a naive `pl.Datetime`,
        # matching the naive timestamps produced elsewhere in the codebase
        # (e.g. `StockDataset`'s naive `str.to_datetime()` filter bounds) --
        # parsing without an explicit time zone raises on tz-aware strings.
        data = data.with_columns(
            pl.col("date")
            .str.to_datetime(time_zone="UTC")
            .dt.replace_time_zone(None)
        )
        data = data.rename({"date": "timestamp"})
        data = data.with_columns(pl.lit(symbol).alias("symbol"))

        out_dir = Path(self.config.raw_data_dir_path) / symbol
        out_dir.mkdir(parents=True, exist_ok=True)
        data.write_parquet(out_dir / "data.pqt")


class ConcurrentTiingoAcquisition(TiingoAcquisition):
    """Resumable, concurrent, failure-isolated bulk Tiingo acquisition.

    Inherits `_fetch_and_write` UNTOUCHED -- the per-symbol vendor call and
    raw-parquet write are already correct -- and overrides only the
    ORCHESTRATION. A full-US-market backfill is ~15.4k symbols and several
    hours; at that scale three properties stop being niceties:

    - **Concurrency.** The work is network-bound and the
      `TiingoClient(session=True)` is shared, so THREADS are right and
      processes are not -- `joblib.Parallel(backend="threading")`, matching
      what `base/model.py:train_cv` already uses for its CV folds. The
      original claim here -- that a paid tier made parallel requests
      unconditionally acceptable and rate limiting unnecessary -- was
      FALSIFIED in the field on 2026-09-06: the account's allocation ran out
      after ~4,600 requests. Concurrency is still right; the assumption that
      it needed no quota handling was not (D-05).
    - **Resumability.** A job killed at ticker 20,000 must resume near ticker
      20,000. Symbols already at `config.end_date` are skipped ENTIRELY
      rather than re-requested for a one-day sliver.
    - **Failure isolation, WITH a global exception.** One delisted ticker
      returning a 404 must not abort the other 15,000: exceptions are
      captured per symbol, the watermark is written ONLY on success (so the
      next run retries the failure), and the run reports a count. But quota
      exhaustion is not one ticker's fault -- it is global and recoverable.
      Treating it as an ordinary per-symbol failure is what burned ~10,000
      symbols as fast-failing requests in the observed incident, which may
      itself have deepened the lockout. It therefore trips a global abort
      instead (D-05); see `_is_quota_error` and `_run_once`.

    Every knob -- `resume`, `max_workers`, `progress`, `legacy_watermarks`,
    `wait_for_quota`, `quota_wait_seconds`, `quota_max_waits` -- is read from
    `config.kwargs` -- the escape hatch `AcquisitionConfig` already documents
    -- rather than becoming constructor arguments no config file could reach,
    keeping the whole thing config-driven per CLAUDE.md.

    This deliberately does NOT touch `base/acquisition.py`. The sequential
    `download()`/`refresh()` loops there remain the correct default for a
    handful of symbols, and a shared seam is not worth introducing for one
    subclass.
    """

    #: Concurrent in-flight symbol fetches. Overridable per run via
    #: `config.kwargs["max_workers"]`.
    DEFAULT_MAX_WORKERS = 8

    #: Written under `config.watermark_path` alongside the per-symbol
    #: watermark sidecars, because "what did and did not land" is exactly the
    #: same question those sidecars answer (T-0iy-07).
    FAILURE_MANIFEST_NAME = "_failures.json"

    #: What the API key is replaced with in any captured message.
    REDACTION = "<TIINGO_API_KEY REDACTED>"

    #: How many failed symbols are named in the summary log line. The full
    #: set always lands in the manifest; the log is a pointer, not a dump.
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

    #: HTTP statuses that mean the account's request allocation is gone, i.e.
    #: a GLOBAL condition (D-05). 429 ONLY, deliberately: Tiingo also returns
    #: 403 for a plan-restricted single ticker, which is a PER-SYMBOL
    #: condition, and treating that as global would let one restricted ticker
    #: abort a 15,000-symbol run. A 403 whose body carries the allocation
    #: wording is still caught by the textual signal below, so the stricter
    #: status set costs nothing.
    QUOTA_STATUS_CODES = frozenset({429})

    #: The durable fragment of the observed body: "Error: You have run over
    #: your hourly request allocation. Contact us at support@tiingo.com to
    #: have these lifted." Matching the full sentence would break the moment
    #: the vendor says "daily" instead of "hourly" or edits its support
    #: address. This is still text matching and it is still brittle -- that
    #: brittleness is a conscious choice, confined to this one constant.
    QUOTA_MESSAGE_TOKEN = "request allocation"

    #: Wait-and-resume is OFF unless asked for, so no run silently holds an
    #: hourly window open (D-06).
    DEFAULT_WAIT_FOR_QUOTA = False

    #: Delay between resume attempts. Tiingo's reset semantics -- fixed
    #: top-of-hour bucket vs. rolling window -- are NOT established, so this
    #: is a configurable INTERVAL, not a computed resume instant. One hour
    #: measured from the moment of detection covers a rolling one-hour window
    #: exactly and a fixed top-of-hour bucket strictly. Assumption, not a
    #: vendor fact.
    DEFAULT_QUOTA_WAIT_SECONDS = 3600

    #: Bounded, because an unbounded loop against a lockout is a worse version
    #: of the problem this class is fixing. 3 comes from the observed
    #: arithmetic: ~4,600 requests per window against 14,674 symbols is
    #: roughly three windows.
    DEFAULT_QUOTA_MAX_WAITS = 3

    def _knob(self, name: str, default):
        return (self.config.kwargs or {}).get(name, default)

    @property
    def _abort(self) -> threading.Event:
        """The global stop flag, shared across every worker thread.

        `threading.Event` is thread-safe by construction, so no surrounding
        lock is needed, and its `wait(timeout)` is exactly the primitive the
        resume delay wants. Created lazily so `_attempt` is safe to call
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

    @staticmethod
    def _vendor_response(exc: BaseException):
        """The vendor `requests.Response` reachable from `exc`, or None.

        Measured, not assumed: `tiingo/restclient.py:_request` catches the
        `requests.exceptions.HTTPError` and re-raises `RestClientError(e)`, so
        `RestClientError` has NO `.response` of its own -- the obvious
        `getattr(exc, "response", None)` one-liner returns None every time.
        The status lives at `exc.args[0].response.status_code`. Hence the walk
        over the exception AND its args.
        """
        for candidate in (exc, *getattr(exc, "args", ())):
            response = getattr(candidate, "response", None)
            if response is not None and getattr(response, "status_code", None):
                return response
        return None

    def _is_quota_error(self, exc: BaseException) -> bool:
        """Whether `exc` means the account's request allocation is exhausted
        -- a GLOBAL, recoverable condition rather than one ticker's fault.

        Two independent signals, either sufficient:

        - **structured:** the reachable status is in `QUOTA_STATUS_CODES`;
        - **textual:** `QUOTA_MESSAGE_TOKEN` appears case-insensitively in the
          reachable response body or in the rendered exception.

        Both are needed. The status alone would miss a vendor that stops
        setting 429; the text alone would miss a 429 with an empty body. Any
        vendor text read here passes through `_scrub` first, because this is
        the text that then travels into log lines (T-26o-01).
        """
        response = self._vendor_response(exc)
        if response is not None and response.status_code in self.QUOTA_STATUS_CODES:
            return True

        body = ""
        if response is not None:
            try:
                body = response.text or ""
            except Exception:  # noqa: BLE001 -- a decode failure is not a quota signal
                body = ""
        haystack = self._scrub(f"{body}\n{exc}").lower()
        return self.QUOTA_MESSAGE_TOKEN in haystack

    def _scrub(self, message: str) -> str:
        """Remove the API key from a message before it is logged or written.

        This repo has already leaked one real Tiingo key. A vendor exception
        string is a path a credential travels that nobody audits -- an HTTP
        error commonly echoes back the full request URL, and Tiingo's carries
        `?token=<key>`. Scrubbing at the single choke point every captured
        message passes through is what makes the manifest safe to commit,
        paste into an issue, or ship to a log aggregator (T-0iy-01).
        """
        key = os.environ.get("TIINGO_API_KEY")
        if key:
            message = message.replace(key, self.REDACTION)
        return message

    def download(self, symbols: list[str] | None = None) -> Self:
        """Full backfill over `[config.start_date, config.end_date]`."""
        return self._run(symbols, from_watermark=False)

    def refresh(self, symbols: list[str] | None = None) -> Self:
        """Incremental fetch, each symbol starting at its own watermark."""
        return self._run(symbols, from_watermark=True)

    def _run(self, symbols: list[str] | None, from_watermark: bool) -> Self:
        """The single concurrent runner both entry points delegate to.

        `download()` and `refresh()` differ ONLY in how each symbol's start
        date is resolved, so they share this body rather than each carrying
        its own fan-out/resume/failure-capture copy that could drift.

        A bounded RESUME LOOP wraps the fan-out (D-06). Each pass recomputes
        `pending` from the watermarks ON DISK, which makes the resume logic
        and the skip logic literally the same code -- there is no parallel
        bookkeeping that could drift from what actually landed.

        **What is assumed and what is not.** Tiingo's reset semantics -- a
        fixed top-of-hour bucket versus a rolling window -- are NOT
        established, and nothing here claims to know them. That is why this is
        a configurable INTERVAL with a bounded attempt count rather than a
        computed resume-at instant. See `DEFAULT_QUOTA_WAIT_SECONDS` and
        `DEFAULT_QUOTA_MAX_WAITS` for what each default is grounded in. All
        three knobs are read from `config.kwargs` via `_knob`, never as
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
        """
        abort = self._reset_abort()
        max_workers = int(self._knob("max_workers", self.DEFAULT_MAX_WORKERS))

        def inputs():
            for symbol in pending:
                # OPTIMISATION ONLY -- not where the guarantee lives. It stops
                # joblib queueing new batches, but with pre-dispatch batching
                # it cannot be relied on alone. The first-statement check in
                # `_attempt` is what actually stops the vendor requests.
                if abort.is_set():
                    break
                yield symbol

        # `return_as="generator_unordered"` is load-bearing, not a style
        # choice. The default eager `Parallel(...)` call returns only once
        # every symbol is done, so a `tqdm` around it would render nothing
        # for hours and then a full bar; wrapping the DISPATCH generator
        # instead fills the bar instantly, because `Parallel` consumes that
        # generator up front to queue the work. Streaming the RESULTS is the
        # only form where one tick means one symbol actually landed on disk.
        stream = Parallel(
            n_jobs=max_workers, backend="threading", return_as="generator_unordered"
        )(delayed(self._attempt)(symbol, from_watermark) for symbol in inputs())

        bar = tqdm(
            total=len(pending),
            desc=f"Tiingo {self.config.start_date}..{self.config.end_date}",
            unit="sym",
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

        failures = {
            symbol: message
            for symbol, status, message in results
            if status == "failed"
        }
        quota_messages = [
            message for _, status, message in results if status == "quota"
        ]

        if not abort.is_set():
            return False, failures

        completed = sum(1 for _, status, _ in results if status == "ok")
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
        """
        coverage = self._read_coverage(symbol)
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
        counts = {"covered": 0, "widened": 0, "legacy": 0}

        for symbol in requested:
            status = self._coverage_status(symbol, from_watermark)
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
        """Report the three outcomes separately, so widening the window has a
        visible, countable consequence instead of a silent one.
        """
        skipped = len(requested) - len(pending)
        if skipped:
            logger.info(
                f"Resume: skipping {skipped}/{len(requested)} symbols already "
                f"covering {self.config.start_date}..{self.config.end_date}; "
                f"{len(pending)} remaining."
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

    def _attempt(
        self, symbol: str, from_watermark: bool
    ) -> tuple[str, str, str | None]:
        """Fetch one symbol, returning `(symbol, status, message_or_None)`.

        `status` is one of:

        - `"ok"` -- fetched and watermarked.
        - `"failed"` -- a per-symbol problem (a delisted ticker's 404, say).
          Lands in the manifest, gets no watermark, is retried next run.
          Unchanged from before quota handling existed.
        - `"quota"` -- the account's allocation is gone. Trips the global
          abort and is EXCLUDED from the manifest: recording a global
          condition as one ticker's fault would defame a perfectly good
          symbol and make the manifest lie about what the last run did.
        - `"skipped"` -- never attempted, because the abort was already set.
          No watermark, not a failure, counted for the report only.

        Never raises: a propagated exception would tear down the whole
        `Parallel` fan-out and abort every other symbol, which is precisely
        the failure mode this class exists to prevent. The watermark is
        written only after a successful write, so a failed symbol is retried
        by the next run instead of being silently marked complete.
        """
        # FIRST statement, and that is the whole point. joblib cannot cancel
        # work it has already queued, so this check -- not the input generator
        # -- is what actually stops the ~260 sym/s burn observed in the field.
        # Every remaining symbol becomes a no-op returning in microseconds,
        # having issued zero vendor requests.
        if self._abort.is_set():
            return symbol, "skipped", None

        coverage = self._read_coverage(symbol) or {}
        start_date = self.config.start_date
        # The covered start to RECORD. A full backfill overwrites the raw file
        # wholesale, so the requested start is a true statement about it; a
        # refresh only extends forward, so it carries the existing start
        # through -- and if that was unknown it stays unknown (D-04).
        covered_start = self.config.start_date
        if from_watermark:
            start_date = coverage.get("last_date") or self.config.start_date
            covered_start = coverage.get("start_date")

        try:
            self._fetch_and_write(
                symbol, start_date=start_date, end_date=self.config.end_date
            )
        except Exception as exc:  # noqa: BLE001 -- isolation is the point
            message = self._scrub(f"{type(exc).__name__}: {exc}")
            if self._is_quota_error(exc):
                self._abort.set()
                return symbol, "quota", message
            return symbol, "failed", message

        self._write_watermark(
            symbol, self.config.end_date, start_date=covered_start
        )
        return symbol, "ok", None

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
