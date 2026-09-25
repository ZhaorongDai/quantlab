"""Full-history backfill of the whole US equity market from Tiingo.

The script downloads daily bars for every symbol in the ``us_all`` universe
category (NYSE, NASDAQ and AMEX common stock, delisted names included) over
a long window and converts them into one Zarr store, a chunked on-disk array
format that ``xarray`` reads. Keeping delisted names avoids survivorship
bias, the error of studying only the companies that survived to the present.

The roster comes from the point-in-time universe table, which records which
symbols were listed on which dates. The table is built from public sources
if it is missing. ``BackfillPlan`` holds every setting of a run in one
frozen object, and ``FullHistoryBackfill`` runs the stages through
``quantlab.registry``. Each stage can be called on its own and resumed after
an interruption: every symbol keeps a watermark, a small sidecar file
recording the dates already downloaded, and every finished conversion
window is recorded.

``TIINGO_API_KEY`` must be set in the environment before ``acquire()``.
``preflight()`` needs no key and sends no request. The key is read by the
vendor client, never by this module.

The script has no command-line options. Edit ``PLAN`` at the bottom of the
file, then run it::

    export TIINGO_API_KEY=your-key-here
    uv run python scripts/backfill_us_equity_full.py

Or drive the stages yourself, with ``scripts/`` on ``sys.path``::

    from backfill_us_equity_full import BackfillPlan, FullHistoryBackfill

    job = FullHistoryBackfill(BackfillPlan(granularity="quarter"))
    print(job.preflight())      # sizes the fetch, sends no request
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
    """Every setting of one backfill, in one immutable object.

    The plan is frozen so that what the stages print is what they ran. To
    vary it, use ``dataclasses.replace``, which returns a new plan. Each
    field is documented by the comment above it.

    Examples
    --------
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

    #: The request window, both bounds inclusive. The ingest scripts default
    #: to a 2016 start, which a full-history run must override.
    start_date: str = "2000-01-01"
    end_date: str = "2026-09-10"

    #: Roster category. The roster holds every symbol of the category that
    #: traded at any moment inside the window, not only the members on one
    #: day, so delisted names are included. ``us_all`` is NYSE, NASDAQ and
    #: AMEX common stock, about 15,400 tickers.
    category: str = "us_all"

    #: Raw subdirectory and Zarr store name. They match
    #: ``scripts/ingest_us_equity.py`` on purpose, so the two share watermarks.
    subdir: str = "us_all"
    store_name: str = "us_all.zarr"

    #: The vendor's registry token, resolved through ``DataSourceRegistry``.
    vendor: str = "tiingo"

    #: Storage root for this run, applied before any config factory runs.
    #: ``None`` falls back to ``QUANTLAB_DATA_DIR``, then to the repository's
    #: ``data/`` directory.
    data_root: str | None = None

    #: Rebuild the point-in-time universe table even when one is already on
    #: disk. A missing table is built automatically (see
    #: ``FullHistoryBackfill.catalog``), so this is only for refreshing a table
    #: you already have. The build reads public endpoints and needs no key.
    rebuild_universe: bool = False

    #: Passed to ``UniverseCatalog.build()``. ``False`` refuses to save a
    #: table rebuilt from a cached copy of a source that failed to download,
    #: because a stale table on disk looks exactly like a fresh one. Set
    #: ``True`` only when a stale table is acceptable, for example when a
    #: membership change-log source is down and you would rather proceed.
    allow_stale_universe: bool = False

    #: ``True`` continues each symbol from its own watermark instead of
    #: backfilling ``start_date..end_date``. Use it to top an existing store
    #: up to today. Leave it ``False`` for the initial backfill, which is also
    #: resumable because symbols already at the target watermark are skipped.
    refresh: bool = False

    #: Size of each conversion window (``"month"``, ``"quarter"``,
    #: ``"year"`` and so on). Nothing checks that a window fits in memory, so
    #: a smaller window is the only protection.
    granularity: str = "month"

    #: What to do when the raw files hold symbols the store does not have yet.
    #: ``refuse`` stops and leaves the store untouched, so a new listing is
    #: reported rather than handled silently.
    on_new_listing: str = "refuse"

    #: How many symbols are downloaded at the same time. ``None`` uses the
    #: vendor class's ``DEFAULT_MAX_WORKERS`` (8 for Tiingo). Tiingo's
    #: end-of-day endpoint serves one symbol per request, so this also sets
    #: how fast the rate limit and the monthly quota are spent.
    max_workers: int | None = None

    #: Any other download option, read by ``Acquisition._knob(name,
    #: default)``. Merged into ``config.kwargs`` last, so it can override
    #: ``max_workers`` too. Useful ones for a multi-hour backfill are
    #: ``wait_for_quota`` (default ``False``, stop when the quota runs out;
    #: ``True`` waits and retries), ``quota_wait_seconds`` (3600),
    #: ``quota_max_waits`` (3), ``legacy_watermarks`` (``"warn"``) and
    #: ``progress`` (``True``, a tqdm progress bar). Leave ``batch_size`` at
    #: Tiingo's default of 1: the endpoint has no multi-symbol request and
    #: the volume guard counts one request per symbol.
    extra_knobs: dict = field(default_factory=dict)

    #: Override for the volume guard, which estimates the download size and
    #: refuses it above its ceilings. ``False`` makes the first run refuse
    #: with the numbers in the exception, so you can read them and decide.
    #: ``True`` skips the refusal but still computes the estimate.
    force_volume: bool = False

    #: Individual volume-guard ceilings. ``None`` keeps the default, so one
    #: can be raised without touching the other two. Raising one ceiling is
    #: safer than ``force_volume=True``.
    max_raw_bytes: int | None = None
    max_requests: int | None = None
    max_wall_clock_hours: float | None = None


class FullHistoryBackfill:
    """The stages of one full-history backfill, each callable on its own.

    Construction is cheap and makes no network call. Its one side effect is
    applying ``plan.data_root`` through ``set_data_root``. That must happen
    before any ``quantlab.config`` factory runs, because the factories copy
    the data root into their paths when called, so it is done in
    ``__init__`` rather than inside a stage.

    Parameters
    ----------
    plan : BackfillPlan
        The settings of this run.

    Attributes
    ----------
    plan : BackfillPlan
        The settings of this run.
    source : SourceDescriptor
        The registry entry for ``plan.vendor``.

    Examples
    --------
    >>> job = FullHistoryBackfill(BackfillPlan(granularity="quarter"))
    >>> job.source.display_name
    'Tiingo EOD'

    The stages themselves need the universe table, and ``acquire()`` needs
    ``TIINGO_API_KEY`` and network access::

        job.preflight()   # sizes the fetch, sends no request
        job.acquire()     # raw parquet, resumable
        job.to_zarr()     # Zarr store, resumable per window
    """

    def __init__(self, plan: BackfillPlan) -> None:
        """Initialize the job; see the class docstring for parameters."""
        set_data_root(plan.data_root)  # Before any config factory runs.
        self.plan = plan
        self.source: SourceDescriptor = DataSourceRegistry.get(plan.vendor)
        self._catalog: UniverseCatalog | None = None
        self._symbols: tuple[str, ...] | None = None

    @property
    def catalog(self) -> UniverseCatalog:
        """The universe table, loaded once and built from public sources if absent.

        The table is loaded on first access, so constructing the job on a
        machine that has never built it is not an error. A missing table is
        built rather than reported, because ``build()`` needs no API key, the
        table does not depend on the window, and every stage needs it. This
        also covers the first run with ``data_root`` on a fresh volume, since
        the table lives under the root.

        Whether the table exists is decided by checking
        ``config.output_path``, the path ``UniverseCatalog.load()`` reads,
        not by catching ``FileNotFoundError`` from the load. Catching the
        error would also hide a corrupt table and silently overwrite it; a
        corrupt table should raise.

        Examples
        --------
        The first access builds the table if it is missing, which reads
        public web endpoints::

            catalog = job.catalog
            "us_all" in catalog.known_categories()
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
        """The roster, sorted in ascending order.

        The fixed order makes the same plan resolve the same roster on every
        run, so the watermarks written by one run are found by the next.

        Examples
        --------
        Needs the universe table::

            job.symbols[:3]
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

    def acquisition_knobs(self) -> dict:
        """Return the ``config.kwargs`` options the download layer reads.

        Every per-run download option travels in ``config.kwargs`` rather
        than as a constructor argument, so anything settable from code is
        also settable from a config file. ``Acquisition._knob(name,
        default)`` is the only reader. ``extra_knobs`` is merged last and so
        wins, including over ``max_workers``.

        Returns
        -------
        dict
            The options, always including ``"resume": True``.

        Examples
        --------
        >>> FullHistoryBackfill(BackfillPlan()).acquisition_knobs()
        {'resume': True}
        >>> plan = BackfillPlan(max_workers=4, extra_knobs={"wait_for_quota": True})
        >>> FullHistoryBackfill(plan).acquisition_knobs()
        {'resume': True, 'max_workers': 4, 'wait_for_quota': True}
        """
        knobs = {
            # Stated explicitly so the saved config shows the run is resumable.
            "resume": True,
        }
        # Leave the key out rather than setting None: the reader calls
        # int() on the value, so a None would raise instead of falling back
        # to the vendor default.
        if self.plan.max_workers is not None:
            knobs["max_workers"] = self.plan.max_workers
        knobs.update(self.plan.extra_knobs)
        return knobs

    def acquisition_config(self) -> AcquisitionConfig:
        """Build the download config through the registry's config factory.

        Using the registry's factory fixes the vendor to ``plan.vendor``
        rather than to whatever default a factory has.

        Returns
        -------
        AcquisitionConfig
            The config for the roster, window and raw subdirectory of the
            plan.

        Examples
        --------
        Needs the universe table, since the config carries the roster::

            config = job.acquisition_config()
            config.vendor, len(config.symbols)
        """
        return self.source.config_factory(
            symbols=self.symbols,
            start_date=self.plan.start_date,
            end_date=self.plan.end_date,
            subdir=self.plan.subdir,
            kwargs=self.acquisition_knobs(),
        )

    def dataset_config(self) -> DatasetConfig:
        """Build the dataset config that maps the raw files to the Zarr store.

        Returns
        -------
        DatasetConfig
            The config for the roster, window, raw subdirectory and store
            name of the plan.

        Examples
        --------
        Needs the universe table, since the config carries the roster::

            job.dataset_config().zarr_file_path
        """
        return stock_kline_config(
            symbols=list(self.symbols),
            start_date=self.plan.start_date,
            end_date=self.plan.end_date,
            subdir=self.plan.subdir,
            store_name=self.plan.store_name,
            vendor=self.plan.vendor,
        )

    def preflight(self) -> dict:
        """Estimate the size of the download and refuse it above a ceiling.

        Sends no request, builds no client and needs no credential. It runs
        before anything that can spend time or money.

        Returns
        -------
        dict
            The volume estimate from
            ``UniverseCatalog.assert_acquisition_volume_fits``.

        Raises
        ------
        ValueError
            If a ceiling is crossed and ``plan.force_volume`` is false. The
            message shows the numbers.

        Examples
        --------
        Needs the universe table::

            estimate = job.preflight()
            estimate["rows"], estimate["requests"]
        """
        return self.catalog.assert_acquisition_volume_fits(
            self.plan.category,
            self.plan.start_date,
            self.plan.end_date,
            frequency="1d",
            # Tiingo serves one symbol per request. Reading the batch size off
            # the descriptor keeps the guard and the downloader in agreement.
            batch_size=self.source.acquisition_cls.DEFAULT_BATCH_SIZE,
            max_raw_bytes=self.plan.max_raw_bytes,
            max_requests=self.plan.max_requests,
            max_wall_clock_hours=self.plan.max_wall_clock_hours,
            force=self.plan.force_volume,
        )

    def assert_credentials(self) -> None:
        """Fail early if the vendor's credentials are not set.

        Asks the registry whether every environment variable the source
        needs is present. It never reads or reports a value.

        Raises
        ------
        RuntimeError
            If a required environment variable is unset.

        Examples
        --------
        With ``TIINGO_API_KEY`` unset:

        >>> job = FullHistoryBackfill(BackfillPlan())
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
        """Download the raw parquet files and stop there.

        Calling it again resumes the download: symbols already at the target
        watermark are skipped, so a job killed at ticker 10,000 restarts
        near ticker 10,000 rather than at the top.

        Returns
        -------
        AcquisitionResult
            Which symbols succeeded and failed, and whether the run was
            cancelled or stopped by the quota.

        Raises
        ------
        RuntimeError
            If ``TIINGO_API_KEY`` is not set.

        Examples
        --------
        Needs ``TIINGO_API_KEY`` and network access, and runs for hours on
        the full market::

            result = job.acquire()
            len(result.succeeded), len(result.failures)
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
        """Convert the raw files into the Zarr store, one window at a time.

        It goes through ``quantlab.registry.convert``, the same conversion
        the ingest scripts use. Finished windows are recorded, so a run
        interrupted at window 12 of 21 resumes at window 12.

        Returns
        -------
        ConversionResult
            How many windows and rows were written or skipped, and where
            the store is.

        Raises
        ------
        RuntimeError
            If no raw data exists under the configured root.

        Examples
        --------
        Needs raw files already written by ``acquire()``::

            result = job.to_zarr()
            result.windows_written, result.windows_skipped
        """
        ds_config = self.dataset_config()

        # This dataset only answers ``has_raw_data()``, so it needs no symbols.
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
        """Run ``preflight``, ``acquire`` and ``to_zarr`` in order.

        The stages are usually run separately, since each takes hours.

        Returns
        -------
        tuple of (AcquisitionResult, ConversionResult)
            The results of ``acquire`` and ``to_zarr``.

        Examples
        --------
        Needs ``TIINGO_API_KEY`` and network access::

            acquired, converted = job.run()
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

    # Stage 0: show what the run is about to do. No request, no API key.
    print(
        f"Roster: {len(job.symbols)} symbols in {PLAN.category} "
        f"overlapping {PLAN.start_date}..{PLAN.end_date}"
    )
    print(job.preflight())

    # Stage 1: raw parquet. Takes hours; resumable.
    job.acquire()

    # Stage 2: Zarr. Takes hours; resumable per window. Comment stage 1 out
    # and re-run to convert raw files that are already downloaded.
    job.to_zarr()
