"""Backfill the full US listed-equity market from Tiingo.

The roster is the ``us_all`` category of the point-in-time universe table
(NYSE, NASDAQ and AMEX common stock, delisted names included), resolved by
interval overlap so that every symbol that traded at any point in the window
is fetched. ``quantlab.registry.run`` performs the download through the
registry descriptor ``SOURCE``; per-symbol failures are isolated, every
symbol carries a watermark, and an interrupted run resumes where it stopped.
The raw tree and Zarr store live under ``us_all`` so they never collide with
``scripts/ingest_tiingo.py``'s NASDAQ-only defaults. No vendor class is named
here.

The default run stops at raw parquet, because the conversion is the long
pole after a multi-hour download. ``--to-zarr`` converts through
``quantlab.registry.convert``, one ``--chunk`` window at a time (a year by
default), resuming at the first unwritten window. Nothing checks that a
window fits in memory; pick a finer ``--chunk`` for a tight machine. The
pre-flight volume guard bounds disk bytes, request count and wall clock.

Storage is rooted at ``--data-dir``, else ``QUANTLAB_DATA_DIR``, else the
repository's ``data/`` directory. Requires ``TIINGO_API_KEY`` in the
environment except under ``--dry-run``; the key is never printed or logged.

Usage:
    # Build or refresh the universe table first.
    uv run python scripts/refresh_us_equity_universe.py

    # 1. Size the job: resolve the roster, profile the window and classify
    #    the existing watermarks, issuing no price request. Needs no key.
    uv run python scripts/ingest_us_equity.py --dry-run

    # 2. The real backfill (about 15k symbols, several hours).
    export TIINGO_API_KEY=your-key-here
    uv run python scripts/ingest_us_equity.py

    # 3. Resume after an interruption: identical to (2). Symbols already at
    #    the target watermark are skipped.
    uv run python scripts/ingest_us_equity.py

    # 4. Top up an existing backfill to today, each symbol starting from its
    #    own watermark instead of --start-date.
    uv run python scripts/ingest_us_equity.py --refresh

    # 5. Convert to Zarr one year at a time, or in monthly windows.
    uv run python scripts/ingest_us_equity.py --to-zarr
    uv run python scripts/ingest_us_equity.py --to-zarr --chunk month

    # 6. One-off migration for watermarks that record no covered start:
    #    stamp the start you know they were fetched from, issue no request
    #    and exit. Recorded starts are never overwritten.
    export TIINGO_API_KEY=your-key-here
    uv run python scripts/ingest_us_equity.py --stamp-legacy-watermarks 2016-01-01

    # 7. Quota-aware backfill. The run always stops dispatching when the
    #    vendor reports its allocation exhausted; with this flag it also
    #    waits through the reset and resumes, up to --quota-max-waits times.
    uv run python scripts/ingest_us_equity.py --wait-for-quota
"""

import argparse
import datetime

from dataclasses import replace

from quantlab.acquisition._support.inspector import SourceInspector
from quantlab.registry import DataSourceRegistry, convert, run
from quantlab.universe import UniverseCatalog
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
    print_conversion_result,
    print_volume_estimate,
    refuse_conversion_without_raw_data,
    resolve_symbols,
    volume_pricing,
)

#: The registered Tiingo source. The vendor is named here once, as a registry
#: token; the config factory, the argparse defaults, the watermark stamping
#: and the fetch itself are all read off this descriptor rather than off a
#: vendor class.
SOURCE = DataSourceRegistry.get("tiingo")

#: Default start of the backfill window. Applied as an interval-overlap bound
#: on the roster, not as a listing-date cut.
DEFAULT_START_DATE = "2016-01-01"

#: Raw-data subdirectory and Zarr store name for this roster, kept separate
#: from ``stock_kline_config``'s NASDAQ-only defaults so the two backfills
#: have independent watermarks and neither overwrites the other. Both resolve
#: beneath the storage root.
DEFAULT_SUBDIR = "us_all"
DEFAULT_STORE_NAME = "us_all.zarr"

#: Tiingo serves one symbol per request, so the volume guard is told a batch
#: size of 1; a larger value would understate the request count by that
#: factor, and requests are the unit the vendor quota is denominated in.
TIINGO_BATCH_SIZE = 1

#: How many resolved symbols the dry run echoes, enough to show the roster
#: resolved without paging thousands of tickers through a terminal.
_SYMBOL_PREVIEW = 10


def _print_estimate(catalog: UniverseCatalog, args, symbols: tuple[str, ...]) -> None:
    """Print the roster and window a dry run would fetch.

    The figures are denominated in symbols, trading days and observations,
    never in bytes: who resolved, over what window, and how much of the grid
    is real observation rather than the survivorship-free roster's empty
    cells.
    """
    profile = catalog._roster_window_profile(
        args.category, args.start_date, args.end_date
    )
    print(f"  symbols resolved:  {len(symbols)}")
    print(f"  preview:           {list(symbols[:_SYMBOL_PREVIEW])}")
    print(f"  window:            {args.start_date} .. {args.end_date}")
    print(f"  trading days (~):  {profile['trading_days']}")
    print(f"  real observations: {profile['observed_cells']:,}")
    print(f"  density:           {profile['density']:.3f}")


def _print_coverage(acq_config, symbols: tuple[str, ...]) -> None:
    """Print how the existing watermarks classify against the requested window.

    This answers, before a multi-hour job commits, whether widening
    ``--start-date`` would actually re-fetch anything. It needs no credential:
    ``SourceInspector`` reads the watermark sidecars from disk without
    constructing a client, and it reaches the same coverage partition the
    real run uses, so the report and the fetch cannot disagree about what
    "covered" means.
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
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Backfill the full US listed-equity market (NYSE + NASDAQ + AMEX "
            "common stock, delisted included) from Tiingo. Resumable and "
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
            "without constructing the acquisition or issuing a single price "
            "request. Answers 'how big will this be' before a multi-hour job."
        ),
    )
    # The default is read off the descriptor so that the shared helper stays
    # vendor-agnostic.
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
            "every watermark sidecar that has none, then exit without issuing "
            "a single price request. Watermarks written before coverage "
            "ranges existed carry only an end date; the value is never "
            "guessed, because only you know what window they were fetched "
            "over. Already-recorded starts are left untouched. "
            "TIINGO_API_KEY must still be exported, because the acquisition "
            "object demands it at construction."
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
            "times. Off by default, so no run silently holds an hourly window "
            "open. Either way the run stops dispatching on exhaustion rather "
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
            "Tiingo does not publish whether its quota resets at the top of "
            "the hour or over a rolling window, so this is a configured "
            "interval, not a computed reset time; one hour from the moment of "
            "detection covers either case."
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
            "arithmetic: roughly 4,600 requests per window against roughly "
            "14.7k symbols is about three windows."
        ),
    )
    # The conversion flags are shared with the other ingest scripts because
    # they all drive the same chunked conversion.
    add_to_zarr_arg(parser)
    add_chunk_args(parser)
    add_volume_guard_args(parser)
    add_data_dir_arg(parser)
    return parser


if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Must run before any config factory is called: the factories snapshot
    # their paths at construction time, so a later root override is ignored.
    apply_data_dir(args)

    if args.end_date is None:
        args.end_date = datetime.date.today().isoformat()

    catalog = UniverseCatalog.load(universe_config())
    # Interval overlap rather than point-in-time membership: a backfill wants
    # every symbol that traded at any point in the window, including the
    # ones that delisted inside it, or the roster reintroduces the
    # survivorship bias it exists to remove.
    symbols = resolve_symbols(args, catalog, mode="in_range")

    # ``SOURCE.config_factory`` already has the vendor bound. ``subdir`` keeps
    # this roster's raw tree and watermarks apart from ``ingest_tiingo.py``'s.
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
    # ``symbols=None`` keeps a single source of truth for the symbol axis: the
    # conversion pins its axis from the raw tree, which holds only what
    # actually downloaded, whereas the roster above comes from the universe
    # table. Naming the roster here would be a second, competing answer.
    ds_config = stock_kline_config(
        symbols=None,
        start_date=args.start_date,
        end_date=args.end_date,
        subdir=DEFAULT_SUBDIR,
        store_name=DEFAULT_STORE_NAME,
    )

    if args.stamp_legacy_watermarks is not None:
        # Handled before every other mode: this is a local-file migration
        # that issues no price request, and it must be impossible to trigger
        # a download by combining it with another flag. It is a write, so it
        # goes through the acquisition class rather than the read-only
        # inspector, and the class still demands a credential at
        # construction.
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

    # Pre-flight: no client has been constructed and no request issued yet.
    # Placed after the stamping and dry-run exits above, which issue no
    # request and have nothing for the guard to refuse.
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

    print(
        f"Acquiring {len(symbols)} symbols from {SOURCE.display_name} "
        f"({args.start_date}..{args.end_date}, refresh={args.refresh}, "
        f"max_workers={args.max_workers}). Already-complete symbols are "
        f"skipped; per-symbol failures land in "
        f"{acq_config.watermark_path}/"
        f"{SOURCE.acquisition_cls.FAILURE_MANIFEST_NAME}."
    )
    result = run(SOURCE, acq_config, refresh=args.refresh)
    # These counts describe this run only. The failure manifest is the
    # cross-run record and may name more symbols; read it through
    # ``SourceInspector.failures()``.
    print(
        f"{len(result.succeeded)} symbol(s) succeeded, "
        f"{len(result.failures)} failed"
    )
    print(f"Raw data written under: {acq_config.raw_data_dir_path}")

    if args.to_zarr:
        # The probe dataset exists only to answer ``has_raw_data()``;
        # ``symbols=None`` says so at the call site, spelled the same way as
        # in the other ingest scripts.
        refuse_conversion_without_raw_data(
            StockDataset(replace(ds_config, symbols=None)), result
        )
        print(
            f"Converting/persisting {len(symbols)} symbols to Zarr in "
            f"{args.chunk} windows (resumable; completed windows are skipped)"
        )
        # The conversion is the registry's; this script only renders the
        # result it returns, so what is printed is what was written.
        conversion = convert(
            SOURCE,
            ds_config,
            granularity=args.chunk,
            on_new_listing=args.on_new_listing,
        )
        print_conversion_result(conversion)
    else:
        print(
            "Skipping Zarr conversion (default). Pass --to-zarr to convert; "
            "--chunk selects the window granularity."
        )
