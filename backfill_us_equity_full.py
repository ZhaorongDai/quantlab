"""Full-history US-equity backfill: Tiingo -> raw parquet -> Zarr, as a
PARAMETER OBJECT rather than a command line.

Same job `ingest_us_equity.py` does, reached the other way round. That script
is an argparse front door: every knob is a flag, and a caller who wants to
drive it from Python has to build an `argparse.Namespace` or shell out. This
module has no CLI at all. `BackfillPlan` is the knob set, `FullHistoryBackfill`
owns the stages, and the `__main__` block at the bottom is one literal you
edit -- so the same object is equally usable from a notebook, a scheduler, or
another module.

**This adds no pipeline logic.** Every stage below is a call into the layered
components that already own it:

- `quantlab.acquisition.universe.UniverseCatalog` resolves the roster
- `quantlab.acquisition.registry.run()` fetches it through the source descriptor
- `quantlab.acquisition.registry.convert()` performs THE one chunked raw-to-Zarr
  conversion (03.5 D-06/D-07/SC-6)
- `quantlab.config` factories build both configs and own every path

Nothing here should grow a behaviour a component could own instead. If you find
yourself adding an `if` about market data to this file, it belongs one layer
down.

**No vendor class is named anywhere.** The vendor is a TOKEN on the plan; the
descriptor comes from `DataSourceRegistry`, and every vendor constant
(batch size, required env vars) is read off `SOURCE.acquisition_cls` (03.4
D-15 / SC-1).

**Storage lands where the two existing shells already put this roster**:
`subdir="us_all"` / `store_name="us_all.zarr"`, beneath whatever
`quantlab.config.get_data_root()` resolves (`data_root` on the plan >
`QUANTLAB_DATA_DIR` > repo-root `data/`). That is deliberate, not a copied
default -- sharing the subdirectory means sharing the WATERMARKS, so a run
started here resumes a run started by `ingest_us_equity.py` and vice versa.
Point `subdir` somewhere else and you get a second, independent backfill.

**Credentials.** `TIINGO_API_KEY` is read from the environment by the vendor
client this module never names. This file reads no credential variable itself;
it only asks the registry whether every variable the source NAMES is present
(`is_configured`), which returns a bool and never touches a value. No key is
printed, logged, or written to any artifact.

**The universe table builds itself.** The point-in-time roster table lives
under the storage root, so pointing `data_root` at a fresh volume leaves it
behind. Rather than fail with a `FileNotFoundError` naming a parquet path, the
`catalog` property builds a missing table from public, unauthenticated sources
before resolving the roster. No API key, no separate script, no first-run step
to remember.

**The stages are separate on purpose.** `acquire()` is hours and `to_zarr()`
is hours; both are independently resumable, and you will normally want the raw
tier on disk and verified before you spend the second half. `run()` does both
only because that is occasionally what you want.

**There is no RAM guard any more.** Phase 03.6 deleted the dense-panel
estimator and its per-chunk refusal by decision (SC-3), so an over-sized
`granularity` reaches OOM rather than a legible refusal naming the finer rung
that would fit. Pick the rung yourself -- the ladder is
`year -> quarter -> month -> day -> hour`. For ~15.4k symbols over 26 years,
`"month"` is the conservative default this plan ships; `"year"` densifies
~15.4k x ~252 float64 per window and is the one that OOMs a 16 GiB machine.
What DOES survive ahead of the download is the acquisition-volume guard, which
bounds disk bytes, request count and wall clock -- money and time, not memory.

Usage:
    export TIINGO_API_KEY=your-key-here
    uv run python backfill_us_equity_full.py

Edit `PLAN` at the bottom, or import and drive it yourself:

    from backfill_us_equity_full import BackfillPlan, FullHistoryBackfill

    job = FullHistoryBackfill(BackfillPlan(granularity="quarter"))
    print(job.preflight())      # sizes the fetch, issues zero requests
    job.acquire()               # raw parquet, resumable
    job.to_zarr()               # Zarr store, resumable per window
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

from quantlab.acquisition.registry import (
    DataSourceRegistry,
    SourceDescriptor,
    convert,
    is_configured,
    run,
)
from quantlab.acquisition.universe import UniverseCatalog
from quantlab.base.acquisition import AcquisitionResult
from quantlab.base.config import AcquisitionConfig, DatasetConfig
from quantlab.base.data import ConversionResult
from quantlab.config import set_data_root, stock_kline_config, universe_config
from quantlab.dataset.stock import StockDataset


@dataclass(frozen=True)
class BackfillPlan:
    """Every knob, in one immutable object.

    Frozen because a plan that mutates mid-run is a plan that cannot be
    reported afterwards: the stages below print what they are about to do, and
    a reader has to be able to trust that the printed plan is the executed one.
    Vary it with `dataclasses.replace(plan, granularity="quarter")`, which
    gives you a NEW plan rather than a changed one.
    """

    #: The request window. Both bounds are inclusive and both are stated --
    #: `ingest_us_equity.DEFAULT_START_DATE` is 2016-01-01, so a full-history
    #: run MUST override it or it silently fetches ten years instead of 26.
    start_date: str = "2000-01-01"
    end_date: str = "2026-09-10"

    #: Roster category, resolved by interval OVERLAP (not point-in-time
    #: membership): every symbol that traded at any moment inside the window,
    #: delisted included. 'us_all' is NYSE + NASDAQ + AMEX common stock,
    #: ~15.4k tickers, survivorship-bias-free.
    category: str = "us_all"

    #: Raw subdirectory and Zarr store name. Shared with `ingest_us_equity.py`
    #: on purpose -- see the module docstring on watermark sharing.
    subdir: str = "us_all"
    store_name: str = "us_all.zarr"

    #: A TOKEN, resolved through `DataSourceRegistry`. Never a class.
    vendor: str = "tiingo"

    #: Per-run storage root override, applied BEFORE any config factory runs.
    #: `None` falls through to QUANTLAB_DATA_DIR, then repo-root `data/`.
    data_root: str | None = None

    #: FORCE a rebuild of the point-in-time universe table even when one is
    #: already on disk. Leave False in normal use: a MISSING table is built
    #: automatically (see `FullHistoryBackfill.catalog`), so this flag is only
    #: for deliberately refreshing a table you already have. The build hits
    #: public endpoints, needs no API key, and is not window-dependent.
    rebuild_universe: bool = False

    #: Passed to `UniverseCatalog.build()`. False -- the library's own default
    #: -- refuses to PERSIST a table reconstructed from a fetcher's cached
    #: snapshot, because `save()` overwrites in place and a stale table on disk
    #: is indistinguishable from a fresh one. Set True only when you knowingly
    #: want a frozen table (e.g. a change-log source is down and you would
    #: rather proceed than stop).
    allow_stale_universe: bool = False

    #: True walks each symbol forward from its own watermark instead of
    #: back-filling `start_date..end_date`. Use it to top an existing store up
    #: to today; leave False for the initial backfill (which is ALSO resumable
    #: -- symbols already at the target watermark are skipped).
    refresh: bool = False

    #: Conversion window granularity. See the module docstring: there is no
    #: RAM guard, this is the whole defence.
    granularity: str = "month"

    #: What to do when the raw roster has grown since the store was built.
    #: 'refuse' halts and leaves the store untouched -- the right default for
    #: a first build, and the one that tells you a new listing appeared rather
    #: than quietly deciding for you.
    on_new_listing: str = "refuse"

    #: Concurrent in-flight symbol fetches. `None` means "use the vendor
    #: class's own `DEFAULT_MAX_WORKERS`" (Tiingo: 8) rather than a number
    #: restated here, so raising the vendor default raises this too.
    #: Tiingo's EOD endpoint is one symbol per request, so this IS the
    #: download parallelism -- but it is also how fast you spend a rate limit
    #: and a monthly quota. Raise it deliberately, not reflexively.
    max_workers: int | None = None

    #: Anything else the acquisition layer reads through
    #: `Acquisition._knob(name, default)`. Merged into `config.kwargs` LAST,
    #: so it can override `max_workers` too. The ones that matter for a
    #: multi-hour full-market backfill:
    #:
    #:   wait_for_quota      bool   default False -- abort when the vendor
    #:                              quota runs out. True parks and retries.
    #:   quota_wait_seconds  int    default 3600
    #:   quota_max_waits     int    default 3
    #:   legacy_watermarks   str    default 'warn'
    #:   batch_size          int    default 1 for Tiingo -- do NOT raise it;
    #:                              the EOD endpoint has no multi-symbol batch
    #:                              and the volume guard prices requests at 1.
    #:   progress            bool   default True -- the tqdm reporter.
    extra_knobs: dict = field(default_factory=dict)

    #: Volume-guard escape hatch. False makes the FIRST run refuse with the
    #: arithmetic in the exception, which is what you want: read the numbers,
    #: then decide. True skips the raise and never the arithmetic.
    force_volume: bool = False

    #: Individual ceilings. `None` means "use the class constant", so you can
    #: raise ONE deliberately without disturbing the other two -- the
    #: considered alternative to `force_volume=True`.
    max_raw_bytes: int | None = None
    max_requests: int | None = None
    max_wall_clock_hours: float | None = None


class FullHistoryBackfill:
    """The four stages, each independently callable and each resumable.

    Construction is cheap and side-effect-free apart from the data-root
    override, which MUST happen before any `quantlab.config` factory runs: the
    factories snapshot their paths as strings at construction time, so a root
    applied afterwards silently does nothing (DDIR-04). That is the reason
    `set_data_root` is here in `__init__` and not inside a stage.
    """

    def __init__(self, plan: BackfillPlan) -> None:
        set_data_root(plan.data_root)  # BEFORE any factory (DDIR-04).
        self.plan = plan
        self.source: SourceDescriptor = DataSourceRegistry.get(plan.vendor)
        self._catalog: UniverseCatalog | None = None
        self._symbols: tuple[str, ...] | None = None

    # -- roster ---------------------------------------------------------

    @property
    def catalog(self) -> UniverseCatalog:
        """The universe table, loaded once and reused, and BUILT if absent.

        Loaded lazily so that constructing the job on a machine that has never
        built the table is not itself an error -- only asking for a roster is.

        **A missing table is built rather than raised on.** The table is not a
        user artifact: it is reconstructed from public, unauthenticated sources
        (`build()` needs no API key), it is not window-dependent, and every
        path into this module needs it. Making the caller run a second script
        first bought nothing except a `FileNotFoundError` from four frames
        down, naming a parquet path rather than the thing to do about it --
        which is exactly what happens the first time you point `data_root` at
        a fresh volume, because the table lives under the root and does not
        follow you there.

        Existence is decided on `config.output_path` -- the same path
        `UniverseCatalog.load()` reads -- rather than by catching
        `FileNotFoundError` from the load. A `try/except` here would also
        swallow a genuinely unreadable or truncated table and silently
        overwrite it with a fresh build; a corrupt table should raise, because
        the operator needs to know the difference between "never built" and
        "broken".

        `save()` creates its own parent directories, so a fresh root needs no
        `mkdir` from this layer.
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
        """The roster, in the catalog's contractual ASCENDING order.

        The order matters and is not incidental: it is what makes "the same
        plan resolves the same roster" true across runs, which is what lets
        watermarks from run N be met by run N+1.
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
        """The `config.kwargs` bag the acquisition layer reads tuning from.

        Every per-run tuning parameter in this project travels here rather
        than as a constructor argument, so that nothing is reachable from code
        but unreachable from configuration (CLAUDE.md's config-driven rule);
        `Acquisition._knob(name, default)` is the single reader.

        `extra_knobs` is merged LAST and therefore wins, including over
        `max_workers`. That ordering is deliberate: the named field is the
        common case, and the escape hatch must be able to override anything
        without this class growing a field per vendor knob.
        """
        knobs = {
            # Stated rather than left to the reader's default so that "this
            # backfill is resumable" is visible in the config, not inferred.
            "resume": True,
        }
        # OMITTED, not set to None, when the plan does not pin it. The reader
        # is `int(self._knob("max_workers", DEFAULT_MAX_WORKERS))`, and
        # `_knob` is a plain `.get(name, default)` -- a present-but-None key
        # returns None and `int(None)` raises. Absent is the only spelling of
        # "fall through to the vendor default".
        if self.plan.max_workers is not None:
            knobs["max_workers"] = self.plan.max_workers
        knobs.update(self.plan.extra_knobs)
        return knobs

    def acquisition_config(self) -> AcquisitionConfig:
        """Built through the DESCRIPTOR's factory, so the vendor is pinned by
        the registry rather than inherited from a factory default."""
        return self.source.config_factory(
            symbols=self.symbols,
            start_date=self.plan.start_date,
            end_date=self.plan.end_date,
            subdir=self.plan.subdir,
            kwargs=self.acquisition_knobs(),
        )

    def dataset_config(self) -> DatasetConfig:
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
        """Size the fetch and refuse if it crosses a ceiling. ZERO requests,
        no client constructed, no credential required.

        Deliberately positioned before anything that can spend time or money:
        a guard that runs after the client exists has already spent the thing
        it was meant to save.
        """
        return self.catalog.assert_acquisition_volume_fits(
            self.plan.category,
            self.plan.start_date,
            self.plan.end_date,
            frequency="1d",
            # Tiingo's EOD endpoint is ONE symbol per request -- there is no
            # multi-symbol batch to amortise over. Read off the descriptor so
            # the guard and the fetcher cannot disagree about it.
            batch_size=self.source.acquisition_cls.DEFAULT_BATCH_SIZE,
            max_raw_bytes=self.plan.max_raw_bytes,
            max_requests=self.plan.max_requests,
            max_wall_clock_hours=self.plan.max_wall_clock_hours,
            force=self.plan.force_volume,
        )

    def assert_credentials(self) -> None:
        """Fail before the roster resolve, not four hours into it.

        Asks the registry whether every variable the source NAMES is present.
        Never reads or reports a value.
        """
        if not is_configured(self.source):
            missing = ", ".join(self.source.required_env)
            raise RuntimeError(
                f"{self.source.display_name} is not configured: set "
                f"{missing} in the environment. This module never reads the "
                f"value itself -- the vendor client does."
            )

    def acquire(self) -> AcquisitionResult:
        """Download to the raw parquet tier and STOP.

        Resumable by re-invocation: symbols already at the target watermark are
        skipped, so a job killed at ticker 20,000 restarts near ticker 20,000
        rather than at the top.
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

        Reached through the registry rather than through a Dataset method, so
        this shares the ONE conversion path with the three ingest shells
        instead of being a fourth implementation of it (03.5 D-06/SC-6).

        Resumable per window: a run interrupted at window 12 of 21 resumes at
        window 12.
        """
        ds_config = self.dataset_config()

        # Probe on a SYMBOL-FREE config: this dataset exists only to be asked
        # `has_raw_data()`, and saying `symbols=None` at the call site states
        # that rather than leaving a reader to prove it from the constructor.
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
        """Both halves, in order. Normally you want them separately."""
        self.preflight()
        acquired = self.acquire()
        converted = self.to_zarr()
        return acquired, converted


#: Edit this. It is the whole interface.
PLAN = BackfillPlan(
    start_date="2000-01-01",
    end_date="2026-09-10",
    granularity="year",
    data_root="/Volumes/SSD/data",
    max_workers=32,
)


if __name__ == "__main__":
    job = FullHistoryBackfill(PLAN)

    # Stage 0 -- what am I about to commit to? Zero requests, no API key.
    print(
        f"Roster: {len(job.symbols)} symbols in {PLAN.category} "
        f"overlapping {PLAN.start_date}..{PLAN.end_date}"
    )
    print(job.preflight())

    # Stage 1 -- raw parquet. Hours. Resumable.
    job.acquire()

    # Stage 2 -- Zarr. Hours. Resumable per window.
    # Comment stage 1 out and re-run to convert an already-fetched raw tier.
    job.to_zarr()
