"""Bulk-backfill the full US listed-equity market from Tiingo.

Glue only -- exactly the shape `ingest_tiingo.py` established. Every piece of
logic lives in the layered components this script merely wires together:
`quantlab.acquisition.universe.UniverseCatalog` resolves the roster,
`quantlab.acquisition.registry.run()` fetches it through the registered source
descriptor, `quantlab.acquisition.inspector.SourceInspector` answers the
credential-free coverage question, and `quantlab.dataset.stock.StockDataset`
converts it. Nothing here should grow a behaviour that a component could own
instead.

This script names no vendor class anywhere: it resolves its source from
`DataSourceRegistry`, reads every vendor constant off `SOURCE.acquisition_cls`,
builds its acquisition config through `SOURCE.config_factory` and downloads
through `registry.run()` (03.4 D-15 / SC-1 / SC-6).

`TIINGO_API_KEY` is read from the environment by the vendor client this script
never names, and is NEVER printed, logged, or written to any artifact by this
script or by anything it calls -- not the failure manifest, not an exception
message, not the `TiingoClient` config dict. This module reads no credential
environment variable AT ALL: the only paths that need one construct the client,
which is the single place the check belongs. Only symbol lists, date ranges and
paths are ever printed.

Storage is rooted at whatever `quantlab/config/__init__.py:get_data_root` resolves,
through three levels: the `--data-dir` flag for a per-run root, else the
`QUANTLAB_DATA_DIR` environment variable, else the repo-root `data/` directory.
Those are three ways to set ONE root -- this script hardcodes no volume and
adds no competing root of its own.

**Why the default run stops at raw parquet -- and it is no longer memory.**
Raw acquisition parquet is an acquisition-layer implementation detail
(CLAUDE.md); the pipeline-facing artifact is the Zarr store. The full window
used to be REFUSED: 15,424 symbols x ~5,215 trading days densifies to a
~7.2 GiB float64 grid, and the old whole-range `_raw_data_to_xr()` held that
grid, the ~29.6M-row pandas frame and conversion scratch at once, which OOMs a
16 GiB machine.

That is fixed. `--to-zarr` now runs `BaseDataset.from_raw_data_chunked()`,
which densifies and appends ONE time window at a time onto a symbol axis
pinned once over the whole range, so peak RAM scales with the WINDOW rather
than the range (D-01/D-02). `--chunk` selects the granularity (year by
default) and a run interrupted at window 12 of 21 resumes at window 12.

So the reason `--to-zarr` stays opt-in is TIME, not memory: the conversion is
still the long pole after a multi-hour download, and most runs want the raw
parquet first. The sizing guard remains, ahead of the download, but it is now
`assert_chunked_panel_fits()` -- it refuses a `--chunk` whose individual
windows would not fit and names the finer granularity that would, rather than
refusing the window outright.

Build/refresh the universe table first (`refresh_us_equity_universe.py`), then:

Usage:
    # 1. How big is this? Resolves the roster, sizes the panel and CLASSIFIES
    #    the existing watermarks against the requested window, issuing ZERO
    #    price requests. Needs no API key -- including for the coverage
    #    report, which is nothing but local file reads (03.4 SC-3).
    uv run python ingest_us_equity.py --dry-run

    # 2. The real backfill (~15.4k symbols, several hours).
    export TIINGO_API_KEY=your-key-here
    uv run python ingest_us_equity.py

    # 3. Resume after an interruption. Identical to (2): symbols already at
    #    the target watermark are skipped, so a job killed at ticker 20,000
    #    restarts near ticker 20,000 rather than at the top.
    uv run python ingest_us_equity.py

    # 4. Top up an existing backfill to today, each symbol starting from its
    #    own watermark instead of --start-date.
    uv run python ingest_us_equity.py --refresh

    # 5. Convert to Zarr, one year at a time (resumable, memory-bounded).
    uv run python ingest_us_equity.py --to-zarr

    # 6. Same, in monthly windows -- for a machine tighter than 16 GiB, or a
    #    roster dense enough that a single year does not fit.
    uv run python ingest_us_equity.py --to-zarr --chunk month

    # 7. One-off migration for watermarks written before coverage ranges were
    #    recorded (260906-26o D-04). Fills the covered start YOU supply into
    #    every sidecar that lacks one, issues ZERO price requests, and exits.
    #    Never overwrites a start that is already recorded, and the value is
    #    never guessed -- only you know what window those files were fetched
    #    over. Until they are stamped, every run reports them and skips them.
    export TIINGO_API_KEY=your-key-here
    uv run python ingest_us_equity.py --stamp-legacy-watermarks 2016-01-01

    # 8. Quota-aware backfill (D-05/D-06). The account's allocation is
    #    empirically ~4,600 requests per hour, so ~14.7k symbols needs roughly
    #    three windows. On exhaustion the run STOPS dispatching -- always,
    #    even without the flag -- instead of burning the remainder as
    #    fast-failing requests, which is what happened on 2026-09-06 and may
    #    itself have deepened the lockout. --wait-for-quota additionally sits
    #    through the reset and resumes, bounded by --quota-max-waits.
    uv run python ingest_us_equity.py --wait-for-quota
"""

import argparse
import datetime

from quantlab.acquisition.inspector import SourceInspector
from quantlab.acquisition.registry import DataSourceRegistry, run
from quantlab.acquisition.universe import UniverseCatalog
from quantlab.config import stock_kline_config, universe_config
from quantlab.dataset.stock import StockDataset
from quantlab.utils.cli import (
    add_chunk_args,
    add_concurrency_args,
    add_data_dir_arg,
    add_to_zarr_arg,
    add_volume_guard_args,
    add_window_args,
    apply_data_dir,
    print_volume_estimate,
    refuse_conversion_without_raw_data,
    resolve_symbols,
    volume_pricing,
)

#: The ONE place this script's vendor is named, and it is a TOKEN, not a class.
#:
#: SC-1's "no vendor named at the call site" means no vendor CLASS: a script
#: that backfills the US market from one vendor has that vendor as part of its
#: identity, and the alternative -- a `--source` flag -- is the merged CLI D-15
#: explicitly forbids. Every vendor-specific fact below is read off this
#: descriptor: `SOURCE.config_factory` builds the acquisition config with the
#: vendor pinned, `SOURCE.acquisition_cls` supplies the six argparse defaults
#: that used to name the class (L-5) plus the watermark-stamping WRITE, and
#: `run(SOURCE, ...)` performs the fetch.
SOURCE = DataSourceRegistry.get("tiingo")

#: D-05. The backfill window's default start. Applied as an interval-OVERLAP
#: bound, not as a listing-date cut -- see `get_symbols_in_range`.
DEFAULT_START_DATE = "2016-01-01"

#: Raw-data subdirectory and Zarr store name for this roster, kept separate
#: from `stock_kline_config`'s NASDAQ-only defaults so the two backfills have
#: independent watermarks and neither overwrites the other. Both resolve
#: BENEATH the root `config.get_data_root()` returns (D-04, 260907-rjq D-01).
DEFAULT_SUBDIR = "us_all"
DEFAULT_STORE_NAME = "us_all.zarr"

#: Tiingo's EOD endpoint is ONE symbol per request, so the volume guard is told
#: a batch size of 1. Anything larger would understate the request count by
#: exactly that factor -- and requests are the unit the request ceiling and the
#: quota that ran out on 2026-09-06 are both denominated in.
TIINGO_BATCH_SIZE = 1

#: How many resolved symbols to echo in the dry run. The point is to prove the
#: roster resolved, not to page 15,000 tickers through a terminal.
_SYMBOL_PREVIEW = 10

_GIB = 1024**3


def _print_chunk_report(report: dict) -> None:
    """The two figures a user needs before committing to a conversion: what
    the whole range totals (advisory only -- chunking is what makes it
    achievable) and what the LARGEST single window will actually allocate,
    which is the number the budget applies to (D-05).
    """
    largest = report["max_chunk"]
    print(f"  chunk granularity: {report['granularity']}")
    print(f"  chunk count:       {len(report['chunks'])}")
    print(
        f"  whole-range total: "
        f"{report['advisory']['dense_bytes'] / _GIB:.2f} GiB "
        f"(advisory -- chunking never materialises this at once)"
    )
    if largest is not None:
        print(
            f"  largest chunk:     {largest['dense_bytes'] / _GIB:.2f} GiB "
            f"({largest['start']}..{largest['end']}, "
            f"{largest['symbols']} pinned symbols x "
            f"{largest['trading_days']} trading days)"
        )
    print(
        f"  per-chunk budget:  "
        f"{UniverseCatalog.MAX_DENSE_PANEL_BYTES / _GIB:.2f} GiB "
        f"(a finer --chunk is the remedy above this)"
    )


def _print_estimate(catalog: UniverseCatalog, args, symbols: tuple[str, ...]) -> None:
    estimate = catalog.estimate_dense_panel(
        args.category, args.start_date, args.end_date
    )
    print(f"  symbols resolved:  {len(symbols)}")
    print(f"  preview:           {list(symbols[:_SYMBOL_PREVIEW])}")
    print(f"  window:            {args.start_date} .. {args.end_date}")
    print(f"  trading days (~):  {estimate['trading_days']}")
    print(f"  dense grid cells:  {estimate['dense_cells']:,}")
    print(f"  real observations: {estimate['observed_cells']:,}")
    print(f"  density:           {estimate['density']:.3f}")
    print(f"  dense float64:     {estimate['dense_bytes'] / _GIB:.2f} GiB")
    print(f"  observed float64:  {estimate['observed_bytes'] / _GIB:.2f} GiB")
    _print_chunk_report(
        catalog.assert_chunked_panel_fits(
            args.category, args.start_date, args.end_date, granularity=args.chunk
        )
    )


def _print_coverage(acq_config, symbols: tuple[str, ...]) -> None:
    """Print how the existing watermarks classify against the requested
    window, so a dry run answers "would widening --start-date actually
    re-fetch anything?" before a multi-hour job commits to it.

    **Unconditional, and needing no credential.** This computation is nothing
    but local file reads -- `open()` and `json.load()` over the watermark
    sidecars -- and `SourceInspector` performs them without constructing a
    client, without importing a vendor module and therefore without issuing a
    single vendor request. There is consequently no credential check to make
    here at all: the version that skipped itself when `TIINGO_API_KEY` was
    unset answered only for people who already had a key, which is the wrong
    audience for the one command an operator runs BEFORE committing to a
    multi-hour job (03.4 SC-3 / D-08).

    The judgement is NOT re-derived here. `SourceInspector.coverage` reaches
    the same `CoverageLedger.partition_by_coverage` object the real run
    reaches, so this report and the fetch that follows it cannot disagree about
    what "covered" means (D-09).

    The `coverage report:` header is load-bearing rather than cosmetic: it is
    what the SC-3 dry-run gates assert on, and before this rewrite the only
    place those two words appeared in this file was inside the skip line above.
    """
    report = SourceInspector().coverage(acq_config, symbols=list(symbols))
    print("  coverage report:")
    print(f"  already covered:   {report['covered']} (would be skipped)")
    print(
        f"  re-fetch, widened: {report['widened']} "
        f"(recorded coverage starts after --start-date)"
    )
    print(
        f"  legacy, no start:  {report['legacy']} "
        f"(stamp via --stamp-legacy-watermarks)"
    )
    print(f"  would fetch:       {report['pending']}/{report['requested']}")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Bulk-backfill the full US listed-equity market (NYSE + NASDAQ + "
            "AMEX common stock, delisted included) from Tiingo. Resumable and "
            "failure-isolated. Requires TIINGO_API_KEY except under --dry-run."
        )
    )
    parser.add_argument(
        "--category",
        type=str,
        default="us_all",
        help=(
            "Universe category to resolve from the persisted universe table. "
            "Defaults to 'us_all' (the full NYSE + NASDAQ + AMEX roster). "
            "Build/refresh the table first via refresh_us_equity_universe.py."
        ),
    )
    add_window_args(
        parser,
        default_start_date=DEFAULT_START_DATE,
        semantics="interval-overlap",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Resolve the roster and print a storage estimate, then exit "
            "WITHOUT constructing the acquisition or issuing a single price "
            "request. Answers 'how big will this be' before a multi-hour job."
        ),
    )
    # Read off the DESCRIPTOR, not off a named vendor class (L-5). This is one
    # of the six argparse defaults that used to type a vendor name at
    # parser-definition time and would have survived a registry landing
    # untouched. `add_concurrency_args` takes the value as a PARAMETER on
    # purpose, so `quantlab/utils/cli.py` stays vendor-agnostic.
    add_concurrency_args(
        parser, default_max_workers=SOURCE.acquisition_cls.DEFAULT_MAX_WORKERS
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help=(
            "Start each symbol from its own watermark instead of --start-date. "
            "Both modes are resumable; --refresh additionally narrows the "
            "per-symbol request window to what is actually missing."
        ),
    )
    parser.add_argument(
        "--stamp-legacy-watermarks",
        type=str,
        metavar="START_DATE",
        default=None,
        help=(
            "One-off migration: record START_DATE as the covered start in "
            "every watermark sidecar that has none, then exit WITHOUT issuing "
            "a single price request. Watermarks written before coverage "
            "ranges existed carry only an end date; the value is never "
            "guessed, because only you know what window they were fetched "
            "over (D-04). Already-recorded starts are left untouched. "
            "TIINGO_API_KEY must still be exported -- the acquisition object "
            "demands it at construction, before it knows nothing will be "
            "fetched."
        ),
    )
    parser.add_argument(
        "--legacy-watermarks",
        type=str,
        choices=list(SOURCE.acquisition_cls.LEGACY_WATERMARK_POLICIES),
        default=SOURCE.acquisition_cls.DEFAULT_LEGACY_WATERMARK_POLICY,
        help=(
            "What to do with a watermark that records no covered start "
            "(default "
            f"'{SOURCE.acquisition_cls.DEFAULT_LEGACY_WATERMARK_POLICY}'). "
            "'warn' skips it but reports the count and the stamping command "
            "on every run; 'refetch' treats unknown coverage as uncovered and "
            "re-downloads it. Passed through config.kwargs, so it stays "
            "config-driven."
        ),
    )
    parser.add_argument(
        "--wait-for-quota",
        action="store_true",
        help=(
            "When the vendor reports the request allocation is exhausted, "
            "wait --quota-wait-seconds and resume, up to --quota-max-waits "
            "times. OFF by default, so no run silently holds an hourly window "
            "open. Either way the run STOPS dispatching on exhaustion rather "
            "than burning the remainder as fast-failing requests, and every "
            "watermark is preserved so a later re-run resumes exactly there."
        ),
    )
    parser.add_argument(
        "--quota-wait-seconds",
        type=int,
        default=SOURCE.acquisition_cls.DEFAULT_QUOTA_WAIT_SECONDS,
        help=(
            "Delay between resume attempts (default "
            f"{SOURCE.acquisition_cls.DEFAULT_QUOTA_WAIT_SECONDS}). "
            "Tiingo's reset semantics -- fixed top-of-hour bucket vs. rolling "
            "window -- are not published, so this is a configured INTERVAL, "
            "not a computed reset time; one hour from the moment of detection "
            "covers a rolling hour exactly and a fixed bucket strictly."
        ),
    )
    parser.add_argument(
        "--quota-max-waits",
        type=int,
        default=SOURCE.acquisition_cls.DEFAULT_QUOTA_MAX_WAITS,
        help=(
            "How many times to wait and resume before giving up (default "
            f"{SOURCE.acquisition_cls.DEFAULT_QUOTA_MAX_WAITS}). Bounded "
            "on purpose: an unbounded loop against a lockout is a worse "
            "version of the problem. The default comes from the observed "
            "arithmetic -- ~4,600 requests per window against ~14.7k symbols "
            "is roughly three windows."
        ),
    )
    # Registered through the shared helper rather than declared here: the
    # other two shells now carry the same flag, and three declarations of one
    # flag is how their defaults drifted apart in the first place (G-03.4-1b).
    # `mode="chunked"` is what keeps THIS script's distinct promise -- one
    # --chunk window at a time, resumable -- in the help text.
    add_to_zarr_arg(parser, mode="chunked")
    add_chunk_args(parser)
    add_volume_guard_args(parser)
    add_data_dir_arg(parser)
    return parser


if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Before anything that can reach a `quantlab/config/` factory, and the position is
    # load-bearing: the factories snapshot their paths as strings at
    # construction time, so a root override applied afterwards silently does
    # nothing (DDIR-04).
    apply_data_dir(args)

    if args.end_date is None:
        args.end_date = datetime.date.today().isoformat()

    catalog = UniverseCatalog.load(universe_config())
    # Interval OVERLAP, deliberately NOT get_symbols_as_of(): a backfill wants
    # every symbol that traded at ANY point in the window, including the ~6.9k
    # that delisted inside it. Resolving membership on a single day here would
    # reintroduce exactly the survivorship bias this roster exists to remove.
    # `mode` is stated because `quantlab.utils.cli.resolve_symbols` refuses to have a
    # default -- the wrong choice here would be silent.
    symbols = resolve_symbols(args, catalog, mode="in_range")

    # `SOURCE.config_factory` is `functools.partial(stock_acquisition_config,
    # vendor="tiingo")` -- the SAME factory this script called directly before,
    # with the vendor pinned by the descriptor instead of relying on the
    # factory's incumbent default. `subdir` still travels through it, which is
    # what keeps this roster's raw tree and watermarks independent of
    # `ingest_tiingo.py`'s.
    acq_config = SOURCE.config_factory(
        symbols=symbols,
        start_date=args.start_date,
        end_date=args.end_date,
        subdir=DEFAULT_SUBDIR,
        kwargs={
            "max_workers": args.max_workers,
            "resume": True,
            "legacy_watermarks": args.legacy_watermarks,
            "wait_for_quota": args.wait_for_quota,
            "quota_wait_seconds": args.quota_wait_seconds,
            "quota_max_waits": args.quota_max_waits,
        },
    )
    # `symbols=None`, NOT the resolved roster, and this is load-bearing.
    # `BaseDataset`'s config setter calls `_reset_symbols()` for any non-None
    # symbol list, which calls `read()`, catches the FileNotFoundError a
    # not-yet-written store raises, and falls back to `from_raw_data()` -- a
    # FULL-RANGE densification, at StockDataset CONSTRUCTION time, before the
    # chunked loop is ever entered. That is precisely the OOM this path
    # exists to remove, and on a first run (no store yet) it fires every
    # single time. The symbol axis is resolved from the raw data by
    # `_raw_axes_in_range()` inside the chunked loop instead.
    ds_config = stock_kline_config(
        symbols=None,
        start_date=args.start_date,
        end_date=args.end_date,
        subdir=DEFAULT_SUBDIR,
        store_name=DEFAULT_STORE_NAME,
    )

    if args.stamp_legacy_watermarks is not None:
        # Deliberately BEFORE the roster/estimate work and before any other
        # mode: this is a pure local-file migration that issues zero price
        # requests, and it must be impossible to trigger a download by
        # mistyping it alongside another flag.
        # Reached through the DESCRIPTOR, and it stays a WRITE: D-08 makes
        # `SourceInspector` read-only, so this does not move onto it. It is
        # still the only in-repo route to `stamp_watermarks()`, and it still
        # demands a credential at construction -- deliberately unchanged, so
        # the open blocking-human checkpoint from quick task 260906-26o is
        # verified against exactly the behaviour it was raised against.
        stamped = SOURCE.acquisition_cls(acq_config).stamp_watermarks(
            args.stamp_legacy_watermarks
        )
        print(
            f"Stamped covered start {args.stamp_legacy_watermarks} onto "
            f"{stamped} watermark(s) under {acq_config.watermark_path}. "
            f"Sidecars already recording a start were left untouched. "
            f"No price requests were issued."
        )
        raise SystemExit(0)

    if args.dry_run:
        print(f"DRY RUN -- category={args.category}, no price requests issued")
        _print_estimate(catalog, args, symbols)
        print(f"  raw-data path:     {acq_config.raw_data_dir_path}")
        print(f"  watermark path:    {acq_config.watermark_path}")
        print(f"  zarr path:         {ds_config.zarr_file_path}")
        _print_coverage(acq_config, symbols)
        raise SystemExit(0)

    if not symbols:
        parser.error(
            f"{args.category} resolved to zero symbols over "
            f"{args.start_date}..{args.end_date}. Build/refresh the universe "
            f"table first: uv run python refresh_us_equity_universe.py"
        )

    # BEFORE `TiingoAcquisition(...)` and before a single request (D-09).
    # Deliberately AFTER the --stamp-legacy-watermarks and --dry-run exits
    # above: both issue zero price requests and terminate, and refusing a
    # local sidecar migration -- or refusing the very dry run whose job is to
    # tell you how big this is -- would be the guard firing at the one thing it
    # has no quarrel with.
    #
    # A SIBLING of assert_chunked_panel_fits below, not a replacement: that one
    # bounds RAM for a dense panel, this one bounds disk, request count and
    # wall clock, and either alone lets a real scenario through.
    pricing, category, guard_start, guard_end, window_assumed = volume_pricing(
        args, catalog, symbols=symbols
    )
    print_volume_estimate(
        pricing.assert_acquisition_volume_fits(
            category,
            guard_start,
            guard_end,
            frequency="1d",
            batch_size=TIINGO_BATCH_SIZE,
            rows_per_symbol_day=args.rows_per_symbol_day,
            force=args.force_volume,
        ),
        category=category,
        start_date=guard_start,
        end_date=guard_end,
        window_assumed=window_assumed,
        forced=args.force_volume,
    )

    if args.to_zarr:
        # Checked HERE, before a single byte is downloaded, rather than only
        # in front of the densification. Both positions satisfy "fires before
        # the densification allocates", but this one also spares the user a
        # multi-hour backfill that ends in a refusal they could have been told
        # about immediately (T-0iy-03, preserved from 260906-0iy). What
        # changed is only WHICH guard: the whole-range refusal is lifted, and
        # the per-chunk one names a finer --chunk as its remedy (T-13w-03).
        _print_chunk_report(
            catalog.assert_chunked_panel_fits(
                args.category, args.start_date, args.end_date, granularity=args.chunk
            )
        )

    print(
        f"Acquiring {len(symbols)} symbols from {SOURCE.display_name} "
        f"({args.start_date}..{args.end_date}, refresh={args.refresh}, "
        f"max_workers={args.max_workers}). Already-complete symbols are "
        f"skipped; per-symbol failures land in "
        f"{acq_config.watermark_path}/"
        f"{SOURCE.acquisition_cls.FAILURE_MANIFEST_NAME}."
    )
    result = run(SOURCE, acq_config, refresh=args.refresh)
    # Reported from the RESULT rather than left to the log lines: on a roster
    # this size the per-symbol logs scroll past. These counts describe THIS
    # run only -- `result.failures` stays inside `requested`, so a
    # `--symbols AAPL` smoke run reports at most one failure however many old
    # entries `_failures.json` still carries from earlier rosters. The
    # manifest is the wider, cross-run record and may name more; read it
    # through `SourceInspector.failures()` (D-18, REVIEW CR-01).
    print(
        f"{len(result.succeeded)} symbol(s) succeeded, "
        f"{len(result.failures)} failed"
    )
    print(f"Raw data written under: {acq_config.raw_data_dir_path}")

    if args.to_zarr:
        # This door densifies too, so it carries the same refusal as the other
        # two: `from_raw_data_chunked()` reaches the identical absent-root
        # ValueError when a run fetched nothing onto an empty raw tree.
        # Scoped by REACHABILITY rather than by script name -- pinning a guard
        # to the script whose bug report arrived is the mistake
        # `tests/test_volume_guard.py::
        # test_every_entry_point_that_densifies_guards_the_dense_panels_ram`
        # records in its own docstring.
        dataset = StockDataset(ds_config)
        refuse_conversion_without_raw_data(dataset, result)
        print(
            f"Converting/persisting {len(symbols)} symbols to Zarr in "
            f"{args.chunk} windows (resumable; completed windows are skipped)"
        )
        dataset.from_raw_data_chunked(
            granularity=args.chunk, on_new_listing=args.on_new_listing
        )
        print(f"Zarr store written at: {ds_config.zarr_file_path}")
    else:
        print(
            "Skipping Zarr conversion (default). Pass --to-zarr to convert; "
            "--chunk selects the window granularity."
        )
