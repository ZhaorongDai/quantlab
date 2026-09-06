import json
import os
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

    - **Concurrency.** D-03 puts this on Tiingo's paid tier, which makes
      parallel requests acceptable. The work is network-bound and the
      `TiingoClient(session=True)` is shared, so THREADS are right and
      processes are not -- `joblib.Parallel(backend="threading")`, matching
      what `base/model.py:train_cv` already uses for its CV folds.
    - **Resumability.** A job killed at ticker 20,000 must resume near ticker
      20,000. Symbols already at `config.end_date` are skipped ENTIRELY
      rather than re-requested for a one-day sliver.
    - **Failure isolation.** One delisted ticker returning a 404 must not
      abort the other 15,000. Exceptions are captured per symbol, the
      watermark is written ONLY on success (so the next run retries the
      failure), and the run reports a count.

    Every knob (`resume`, `max_workers`, `progress`) is read from
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

    def _knob(self, name: str, default):
        return (self.config.kwargs or {}).get(name, default)

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
        """
        requested = list(symbols or self.config.symbols)
        pending = requested
        if self._knob("resume", True):
            pending, counts = self._partition_by_coverage(requested, from_watermark)
            self._report_coverage(requested, pending, counts)

        max_workers = int(self._knob("max_workers", self.DEFAULT_MAX_WORKERS))
        # `return_as="generator_unordered"` is load-bearing, not a style
        # choice. The default eager `Parallel(...)` call returns only once
        # every symbol is done, so a `tqdm` around it would render nothing
        # for hours and then a full bar; wrapping the DISPATCH generator
        # instead fills the bar instantly, because `Parallel` consumes that
        # generator up front to queue the work. Streaming the RESULTS is the
        # only form where one tick means one symbol actually landed on disk.
        stream = Parallel(
            n_jobs=max_workers, backend="threading", return_as="generator_unordered"
        )(delayed(self._attempt)(symbol, from_watermark) for symbol in pending)

        results = list(
            tqdm(
                stream,
                total=len(pending),
                desc=f"Tiingo {self.config.start_date}..{self.config.end_date}",
                unit="sym",
                disable=not self._knob("progress", True),
            )
        )

        failures = {symbol: message for symbol, message in results if message}
        self._write_failure_manifest(failures)
        return self

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

    def _attempt(self, symbol: str, from_watermark: bool) -> tuple[str, str | None]:
        """Fetch one symbol, returning `(symbol, error_message_or_None)`.

        Never raises: a propagated exception would tear down the whole
        `Parallel` fan-out and abort every other symbol, which is precisely
        the failure mode this class exists to prevent. The watermark is
        written only after a successful write, so a failed symbol is retried
        by the next run instead of being silently marked complete.
        """
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
            return symbol, self._scrub(f"{type(exc).__name__}: {exc}")

        self._write_watermark(
            symbol, self.config.end_date, start_date=covered_start
        )
        return symbol, None

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
