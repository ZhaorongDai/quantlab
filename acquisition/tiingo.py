import json
import os
from pathlib import Path
from typing import Self

import polars as pl
from joblib import Parallel, delayed
from loguru import logger
from tiingo import TiingoClient

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

    Both knobs are read from `config.kwargs` -- the escape hatch
    `AcquisitionConfig` already documents -- rather than becoming constructor
    arguments no config file could reach, keeping the whole thing
    config-driven per CLAUDE.md.

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
            pending = [
                symbol
                for symbol in requested
                if self._read_watermark(symbol) != self.config.end_date
            ]
            skipped = len(requested) - len(pending)
            if skipped:
                logger.info(
                    f"Resume: skipping {skipped}/{len(requested)} symbols "
                    f"already at watermark {self.config.end_date}; "
                    f"{len(pending)} remaining."
                )

        max_workers = int(self._knob("max_workers", self.DEFAULT_MAX_WORKERS))
        results = Parallel(n_jobs=max_workers, backend="threading")(
            delayed(self._attempt)(symbol, from_watermark) for symbol in pending
        )

        failures = {symbol: message for symbol, message in results if message}
        self._write_failure_manifest(failures)
        return self

    def _attempt(self, symbol: str, from_watermark: bool) -> tuple[str, str | None]:
        """Fetch one symbol, returning `(symbol, error_message_or_None)`.

        Never raises: a propagated exception would tear down the whole
        `Parallel` fan-out and abort every other symbol, which is precisely
        the failure mode this class exists to prevent. The watermark is
        written only after a successful write, so a failed symbol is retried
        by the next run instead of being silently marked complete.
        """
        start_date = self.config.start_date
        if from_watermark:
            start_date = self._read_watermark(symbol) or self.config.start_date

        try:
            self._fetch_and_write(
                symbol, start_date=start_date, end_date=self.config.end_date
            )
        except Exception as exc:  # noqa: BLE001 -- isolation is the point
            return symbol, self._scrub(f"{type(exc).__name__}: {exc}")

        self._write_watermark(symbol, self.config.end_date)
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
