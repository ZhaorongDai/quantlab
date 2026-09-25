"""Full-history backfill of the whole US equity market from Tiingo.

The script downloads daily bars for every symbol in the ``us_all`` universe
category (NYSE, NASDAQ and AMEX common stock, delisted names included) over
a long window and converts them into one Zarr store. It is a thin
orchestration over ``quantlab.registry``: ``BackfillPlan`` holds every knob
in one frozen object and ``FullHistoryBackfill`` runs the stages, each of
which is independently callable and resumable. The point-in-time universe
table is built from public sources if it is missing.

Credentials: ``TIINGO_API_KEY`` must be set in the environment before
``acquire()``; ``preflight()`` needs no key and issues no request. The key is
read by the vendor client, never by this module.

Usage:
    export TIINGO_API_KEY=<your-tiingo-key>
    uv run python scripts/backfill_us_equity_full.py

Edit ``PLAN`` at the bottom of the file, or drive the stages yourself with
``scripts/`` on ``sys.path``::

    from backfill_us_equity_full import BackfillPlan, FullHistoryBackfill

    job = FullHistoryBackfill(BackfillPlan(granularity="quarter"))
    print(job.preflight())      # sizes the fetch, issues zero requests
    job.acquire()               # raw parquet, resumable
    job.to_zarr()               # Zarr store, resumable per window
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

from quantlab.registry import (
    DataSourceRegistry,
    SourceDescriptor,
    convert,
    is_configured,
    run,
)
from quantlab.universe import UniverseCatalog
from quantlab.base.acquisition import AcquisitionResult
from quantlab.base.config import AcquisitionConfig, DatasetConfig
from quantlab.base.data import ConversionResult
from quantlab.config import set_data_root, stock_kline_config, universe_config
from quantlab.dataset.stock import StockDataset


@dataclass(frozen=True)
class BackfillPlan:
    """Every knob of one backfill, in one immutable object.

    The plan is frozen so that what the stages print is what they ran. Vary
    it with ``dataclasses.replace``, which returns a new plan.

    Example:
        >>> import dataclasses
        >>> plan = BackfillPlan(granularity="quarter")
        >>> plan.start_date, plan.category
        ('2000-01-01', 'us_all')
        >>> dataclasses.replace(plan, max_workers=4).max_workers
        4
        >>> plan.granularity = "year"
        Traceback (most recent call last):
        ...
        dataclasses.FrozenInstanceError: cannot assign to field 'granularity'
    """

    #: The request window, both bounds inclusive. Stated explicitly because
    #: the ingest scripts default to a 2016 start, which a full-history run
    #: must override.
    start_date: str = "2000-01-01"
    end_date: str = "2026-09-10"

    #: Roster category, resolved by interval overlap rather than point-in-time
    #: membership: every symbol that traded at any moment inside the window,
    #: delisted names included. ``us_all`` is NYSE, NASDAQ and AMEX common
    #: stock, about 15,400 tickers, free of survivorship bias.
    category: str = "us_all"

    #: Raw subdirectory and Zarr store name. Shared with
    #: ``scripts/ingest_us_equity.py`` on purpose, so the two share watermarks.
    subdir: str = "us_all"
    store_name: str = "us_all.zarr"

    #: The vendor's registry token, resolved through ``DataSourceRegistry``.
    vendor: str = "tiingo"

    #: Per-run storage root override, applied before any config factory runs.
    #: ``None`` falls through to ``QUANTLAB_DATA_DIR``, then the repository's
    #: ``data/`` directory.
    data_root: str | None = None

    #: Rebuild the point-in-time universe table even when one is already on
    #: disk. A missing table is built automatically (see
    #: ``FullHistoryBackfill.catalog``), so this is only for refreshing a table
    #: you already have. The build reads public endpoints and needs no key.
    rebuild_universe: bool = False

    #: Passed to ``UniverseCatalog.build()``. ``False`` refuses to persist a
    #: table reconstructed from a fetcher's cached snapshot, because a stale
    #: table on disk is indistinguishable from a fresh one. Set ``True`` only
    #: when a frozen table is acceptable (for example when a change-log source
    #: is down and you would rather proceed than stop).
    allow_stale_universe: bool = False

    #: ``True`` walks each symbol forward from its own watermark instead of
    #: backfilling ``start_date..end_date``. Use it to top an existing store up
    #: to today; leave ``False`` for the initial backfill, which is also
    #: resumable because symbols already at the target watermark are skipped.
    refresh: bool = False

    #: Conversion window granularity. There is no memory guard in the
    #: conversion; the window size is the whole defence.
    granularity: str = "month"

    #: What to do when the raw roster has grown since the store was built.
    #: ``refuse`` halts and leaves the store untouched, which is the right
    #: default for a first build and the one that reports a new listing rather
    #: than deciding for you.
    on_new_listing: str = "refuse"

    #: Concurrent in-flight symbol fetches. ``None`` uses the vendor class's
    #: own ``DEFAULT_MAX_WORKERS`` (8 for Tiingo). Tiingo's EOD endpoint is one
    #: symbol per request, so this is the download parallelism, and also how
    #: fast a rate limit and a monthly quota are spent.
    max_workers: int | None = None

    #: Anything else the acquisition layer reads through
    #: ``Acquisition._knob(name, default)``. Merged into ``config.kwargs``
    #: last, so it can override ``max_workers`` too. The ones that matter for
    #: a multi-hour full-market backfill are ``wait_for_quota`` (default
    #: ``False``, abort when the vendor quota runs out; ``True`` waits and
    #: retries), ``quota_wait_seconds`` (3600), ``quota_max_waits`` (3),
    #: ``legacy_watermarks`` (``"warn"``) and ``progress`` (``True``, the
    #: tqdm reporter). Leave ``batch_size`` at Tiingo's default of 1: the EOD
    #: endpoint has no multi-symbol batch and the volume guard prices
    #: requests at one symbol each.
    extra_knobs: dict = field(default_factory=dict)

    #: Volume-guard escape hatch. ``False`` makes the first run refuse with
    #: the arithmetic in the exception, so you can read the numbers and then
    #: decide. ``True`` skips the refusal, never the arithmetic.
    force_volume: bool = False

    #: Individual ceilings. ``None`` uses the class constant, so one can be
    #: raised deliberately without disturbing the other two; the considered
    #: alternative to ``force_volume=True``.
    max_raw_bytes: int | None = None
    max_requests: int | None = None
    max_wall_clock_hours: float | None = None


class FullHistoryBackfill:
    """The stages of one full-history backfill, each callable on its own.

    Construction is cheap and makes no network call. Its one side effect is
    applying ``plan.data_root`` through ``set_data_root``, which must happen
    before any ``quantlab.config`` factory runs because the factories snapshot
    their paths as strings when called; that is why it lives in ``__init__``
    rather than inside a stage.

    Example:
        Needs ``TIINGO_API_KEY`` for ``acquire()``; the other calls shown do
        not touch the network once the universe table exists.

        >>> job = FullHistoryBackfill(BackfillPlan(granularity="quarter"))
        >>> job.source.display_name
        'Tiingo EOD'
        >>> job.preflight()["rows"]   # sizes the fetch, issues no request
        >>> job.acquire()             # raw parquet, resumable
        >>> job.to_zarr()             # Zarr store, resumable per window
    """

    def __init__(self, plan: BackfillPlan) -> None:
        """Store ``plan``, resolve its vendor and apply its data root."""
        set_data_root(plan.data_root)  # Before any config factory runs.
        self.plan = plan
        self.source: SourceDescriptor = DataSourceRegistry.get(plan.vendor)
        self._catalog: UniverseCatalog | None = None
        self._symbols: tuple[str, ...] | None = None

    # -- roster ---------------------------------------------------------

    @property
    def catalog(self) -> UniverseCatalog:
        """The universe table, loaded once and built from public sources if absent.

        Loaded lazily, so constructing the job on a machine that has never
        built the table is not an error; only asking for a roster is. A
        missing table is built rather than raised on because it is not a
        user artifact: ``build()`` needs no API key, the table does not depend
        on the window, and every path through this module needs it. This is
        also what happens the first time ``data_root`` points at a fresh
        volume, since the table lives under the root.

        Existence is decided by ``config.output_path``, the same path
        ``UniverseCatalog.load()`` reads, rather than by catching
        ``FileNotFoundError`` from the load: a ``try/except`` would also
        swallow a corrupt table and silently overwrite it, and a corrupt
        table should raise. ``save()`` creates its own parent directories.

        Example:
            Builds the table on first use when it is missing, which reads
            public endpoints.

            >>> catalog = job.catalog
            >>> "us_all" in catalog.known_categories()
        """
        if self._catalog is None:
            config = universe_config()
            missing = not Path(config.output_path).exists()
            if self.plan.rebuild_universe or missing:
                reason = "missing" if missing else "rebuild requested"
                print(
                    f"Universe table {reason} -- building from public sources "
                    f"(no API key needed): {config.output_path}"
                )
                self._catalog = (
                    UniverseCatalog(config)
                    .build(allow_stale=self.plan.allow_stale_universe)
                    .save()
                )
            else:
                self._catalog = UniverseCatalog.load(config)
        return self._catalog

    @property
    def symbols(self) -> tuple[str, ...]:
        """The roster, in the catalog's ascending order.

        The order is part of the contract: it is what makes the same plan
        resolve the same roster across runs, so that watermarks written by
        one run are met by the next.

        Example:
            Needs the universe table.

            >>> job.symbols[:3]
        """
        if self._symbols is None:
            self._symbols = tuple(
                self.catalog.get_symbols_in_range(
                    self.plan.category,
                    self.plan.start_date,
                    self.plan.end_date,
                )
            )
        return self._symbols

    # -- configs --------------------------------------------------------

    def acquisition_knobs(self) -> dict:
        """Return the ``config.kwargs`` bag the acquisition layer reads tuning from.

        Every per-run tuning parameter travels here rather than as a
        constructor argument, so that nothing is reachable from code but
        unreachable from configuration; ``Acquisition._knob(name, default)``
        is the single reader. ``extra_knobs`` is merged last and therefore
        wins, including over ``max_workers``.

        Example:
            >>> import dataclasses
            >>> FullHistoryBackfill(BackfillPlan()).acquisition_knobs()
            {'resume': True}
            >>> plan = BackfillPlan(max_workers=4, extra_knobs={"wait_for_quota": True})
            >>> FullHistoryBackfill(plan).acquisition_knobs()
            {'resume': True, 'max_workers': 4, 'wait_for_quota': True}
        """
        knobs = {
            # Stated rather than left to the reader's default so that "this
            # backfill is resumable" is visible in the config, not inferred.
            "resume": True,
        }
        # Omitted, not set to None, when the plan does not pin it: the reader
        # is `int(self._knob("max_workers", DEFAULT_MAX_WORKERS))` over a plain
        # `.get`, so a present-but-None key would raise. Absence is the only
        # spelling of "fall through to the vendor default".
        if self.plan.max_workers is not None:
            knobs["max_workers"] = self.plan.max_workers
        knobs.update(self.plan.extra_knobs)
        return knobs

    def acquisition_config(self) -> AcquisitionConfig:
        """Build the acquisition config through the descriptor's factory.

        Going through the descriptor pins the vendor by the registry rather
        than by a factory default.

        Example:
            Needs the universe table, since the config carries the roster.

            >>> config = job.acquisition_config()
            >>> config.vendor, len(config.symbols)
        """
        return self.source.config_factory(
            symbols=self.symbols,
            start_date=self.plan.start_date,
            end_date=self.plan.end_date,
            subdir=self.plan.subdir,
            kwargs=self.acquisition_knobs(),
        )

    def dataset_config(self) -> DatasetConfig:
        """Build the dataset config that maps the raw tier to the Zarr store.

        Example:
            Needs the universe table, since the config carries the roster.

            >>> job.dataset_config().zarr_file_path
        """
        return stock_kline_config(
            symbols=list(self.symbols),
            start_date=self.plan.start_date,
            end_date=self.plan.end_date,
            subdir=self.plan.subdir,
            store_name=self.plan.store_name,
            vendor=self.plan.vendor,
        )

    # -- stages ---------------------------------------------------------

    def preflight(self) -> dict:
        """Size the fetch and refuse it if it crosses a ceiling.

        Issues no request, constructs no client and needs no credential. It
        runs before anything that can spend time or money, because a guard
        that runs after the client exists has already spent what it was
        meant to save.

        Returns:
            The volume estimate from
            ``UniverseCatalog.assert_acquisition_volume_fits``.

        Raises:
            ValueError: If a ceiling is crossed and ``plan.force_volume`` is
                false; the message carries the arithmetic.

        Example:
            >>> estimate = job.preflight()
            >>> estimate["rows"], estimate["requests"]
        """
        return self.catalog.assert_acquisition_volume_fits(
            self.plan.category,
            self.plan.start_date,
            self.plan.end_date,
            frequency="1d",
            # Tiingo's EOD endpoint is one symbol per request; there is no
            # multi-symbol batch. Read off the descriptor so the guard and the
            # fetcher cannot disagree about it.
            batch_size=self.source.acquisition_cls.DEFAULT_BATCH_SIZE,
            max_raw_bytes=self.plan.max_raw_bytes,
            max_requests=self.plan.max_requests,
            max_wall_clock_hours=self.plan.max_wall_clock_hours,
            force=self.plan.force_volume,
        )

    def assert_credentials(self) -> None:
        """Fail before the roster resolves if the vendor is not configured.

        Asks the registry whether every variable the source names is present.
        Never reads or reports a value.

        Raises:
            RuntimeError: If a required environment variable is unset.

        Example:
            With ``TIINGO_API_KEY`` unset:

            >>> job.assert_credentials()
            Traceback (most recent call last):
            ...
            RuntimeError: Tiingo EOD is not configured: set TIINGO_API_KEY in ...
        """
        if not is_configured(self.source):
            missing = ", ".join(self.source.required_env)
            raise RuntimeError(
                f"{self.source.display_name} is not configured: set "
                f"{missing} in the environment. This module never reads the "
                f"value itself -- the vendor client does."
            )

    def acquire(self) -> AcquisitionResult:
        """Download the raw parquet tier and stop there.

        Resumable by re-invocation: symbols already at the target watermark
        are skipped, so a job killed at ticker 20,000 restarts near ticker
        20,000 rather than at the top.

        Returns:
            The run's ``AcquisitionResult``.

        Example:
            Needs ``TIINGO_API_KEY`` and network access; runs for hours on
            the full market.

            >>> result = job.acquire()
            >>> len(result.succeeded), len(result.failures)
        """
        self.assert_credentials()
        config = self.acquisition_config()
        print(
            f"Acquiring {len(config.symbols)} symbols from "
            f"{self.source.display_name} over "
            f"{self.plan.start_date}..{self.plan.end_date} "
            f"(refresh={self.plan.refresh})"
        )
        result = run(self.source, config, refresh=self.plan.refresh)
        print(
            f"{len(result.succeeded)} symbol(s) succeeded, "
            f"{len(result.failures)} failed"
            + (" -- run CANCELLED" if result.cancelled else "")
            + (" -- QUOTA aborted" if result.quota_aborted else "")
        )
        print(f"Raw data written under: {config.raw_data_dir_path}")
        return result

    def to_zarr(self) -> ConversionResult:
        """Convert the raw tier into the Zarr store, one window at a time.

        Reached through ``quantlab.registry.convert`` rather than a dataset
        method, so it shares the one conversion path the ingest scripts use.
        Resumable per window: a run interrupted at window 12 of 21 resumes at
        window 12.

        Returns:
            The ``ConversionResult`` of this run.

        Raises:
            RuntimeError: If no raw data exists under the configured root.

        Example:
            Needs a raw tier already written by ``acquire()``.

            >>> result = job.to_zarr()
            >>> result.windows_written, result.windows_skipped
        """
        ds_config = self.dataset_config()

        # Probe on a symbol-free config: this dataset exists only to be asked
        # `has_raw_data()`, and `symbols=None` says so at the call site.
        if not StockDataset(replace(ds_config, symbols=None)).has_raw_data():
            raise RuntimeError(
                f"No raw data under the configured root -- nothing to "
                f"convert. Run acquire() first. Expected raw shards beneath "
                f"the '{self.plan.subdir}' subdirectory for vendor "
                f"'{self.plan.vendor}'."
            )

        print(
            f"Converting {len(ds_config.symbols)} symbols to Zarr in "
            f"{self.plan.granularity} windows "
            f"(resumable; completed windows are skipped)"
        )
        result = convert(
            self.source,
            ds_config,
            granularity=self.plan.granularity,
            on_new_listing=self.plan.on_new_listing,
        )
        print(
            f"Zarr store: {result.zarr_path}\n"
            f"  windows: {result.windows_written} written, "
            f"{result.windows_skipped} skipped, "
            f"{result.windows_planned} planned"
            + (" (resumed)" if result.resumed else "")
            + ("\n  CANCELLED at a window boundary" if result.cancelled else "")
            + f"\n  rows written: {result.rows_written}"
            f"\n  symbols pinned on the axis: {result.pinned_symbols}"
            f"\n  peak window bytes: {result.peak_window_bytes}"
        )
        return result

    def run(self) -> tuple[AcquisitionResult, ConversionResult]:
        """Run preflight, acquire and to_zarr in order.

        Normally the stages are run separately, since each takes hours.

        Example:
            >>> acquired, converted = job.run()
        """
        self.preflight()
        acquired = self.acquire()
        converted = self.to_zarr()
        return acquired, converted


#: The plan the script runs when executed directly. Edit this.
PLAN = BackfillPlan(
    start_date="2000-01-01",
    end_date="2026-09-10",
    granularity="year",
    data_root="/Volumes/SSD/data",
    max_workers=32,
)


if __name__ == "__main__":
    job = FullHistoryBackfill(PLAN)

    # Stage 0: what the run is about to commit to. No request, no API key.
    print(
        f"Roster: {len(job.symbols)} symbols in {PLAN.category} "
        f"overlapping {PLAN.start_date}..{PLAN.end_date}"
    )
    print(job.preflight())

    # Stage 1: raw parquet. Hours. Resumable.
    job.acquire()

    # Stage 2: Zarr. Hours. Resumable per window.
    # Comment stage 1 out and re-run to convert an already-fetched raw tier.
    job.to_zarr()
