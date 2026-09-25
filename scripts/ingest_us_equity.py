"""Backfill the full US listed-equity market from Tiingo.

The roster is the ``us_all`` category of the point-in-time universe table:
NYSE, NASDAQ and AMEX common stock, delisted names included. Point-in-time
means the table knows which symbols were listed on which dates. The roster
is resolved by interval overlap, so every symbol that traded at any moment
inside the window is fetched, even one that delisted halfway through.
Keeping those names avoids survivorship bias, the error of studying only the
companies that survived to the present.

``quantlab.registry.run`` performs the download through the registry
descriptor ``SOURCE``; no vendor class is named here. A failed symbol does
not stop the others. Each symbol keeps a watermark, a small sidecar file
recording the date range already downloaded, so an interrupted run resumes
where it stopped. The raw files and the Zarr store live under ``us_all``, so
they never collide with the NASDAQ-only defaults of
``scripts/ingest_tiingo.py``.

The default run stops at raw parquet, because converting after a
multi-hour download is itself slow. ``--to-zarr`` converts through
``quantlab.registry.convert`` into a Zarr store (a chunked on-disk array
format that ``xarray`` reads), one ``--chunk`` window at a time (a year by
default), resuming at the first unwritten window. Nothing checks that a
window fits in memory, so pick a finer ``--chunk`` on a small machine. A
pre-flight volume guard estimates disk bytes, request count and run time
and refuses a download above its ceilings before any request is sent.

Storage is rooted at ``--data-dir``, else ``QUANTLAB_DATA_DIR``, else the
repository's ``data/`` directory. ``TIINGO_API_KEY`` must be set in the
environment except under ``--dry-run``; the key is never printed or logged.

Usage::

    uv run python scripts/ingest_us_equity.py --help

    # Build or refresh the universe table first.
    uv run python scripts/refresh_us_equity_universe.py

    # 1. Size the job: resolve the roster, profile the window and classify
    #    the existing watermarks, sending no price request. Needs no key.
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

    # 6. One-off fix for watermarks that record no start date: stamp the
    #    start you know they were fetched from, send no request and exit.
    #    Recorded starts are never overwritten.
    export TIINGO_API_KEY=your-key-here
    uv run python scripts/ingest_us_equity.py --stamp-legacy-watermarks 2016-01-01

    # 7. Quota-aware backfill. The run always stops sending requests when
    #    Tiingo reports the quota used up; with this flag it also waits for
    #    the quota to reset and resumes, up to --quota-max-waits times.
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

    The figures count symbols, trading days and observations, not bytes.
    Density is the share of (day, symbol) cells that hold a real bar; the
    rest are days on which a symbol in the roster was not listed.

    Parameters
    ----------
    catalog : UniverseCatalog
        The loaded universe table.
    args : argparse.Namespace
        Parsed command-line arguments; the category and window are read.
    symbols : tuple of str
        The resolved roster.
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
    """Print how the existing watermarks compare with the requested window.

    This shows, before a multi-hour job starts, whether moving
    ``--start-date`` earlier would actually re-fetch anything. It needs no
    credential: ``SourceInspector`` reads the watermark files from disk
    without building a client, and it sorts symbols with the same rule the
    real run uses, so the report and the download agree on what "covered"
    means.

    Parameters
    ----------
    acq_config : AcquisitionConfig
        The acquisition config the real run would use.
    symbols : tuple of str
        The resolved roster.
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
            "Backfill the full US listed-equity market (NYSE, NASDAQ and AMEX "
            "common stock, delisted names included) from Tiingo. Resumable; "
            "a failed symbol does not stop the others. Requires "
            "TIINGO_API_KEY except under --dry-run."
        )
    )
    parser.add_argument(
        "--category",
        type=str,
        default="us_all",
        help=(
            "Universe category to resolve from the saved universe table "
            "(default 'us_all', the full NYSE, NASDAQ and AMEX roster). "
            "Build or refresh the table first with "
            "scripts/refresh_us_equity_universe.py."
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
            "Resolve the roster, print its size and the state of the existing "
            "watermarks, then exit without sending a single price request. "
            "Needs no API key."
        ),
    )
    # The default comes from the descriptor so the shared helper stays
    # vendor-neutral.
    add_concurrency_args(
        parser, default_max_workers=SOURCE.acquisition_cls.DEFAULT_MAX_WORKERS
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help=(
            "Start each symbol from its own watermark instead of --start-date. "
            "Both modes are resumable; --refresh also narrows each symbol's "
            "request to the dates actually missing."
        ),
    )
    parser.add_argument(
        "--stamp-legacy-watermarks",
        type=str,
        metavar="START_DATE",
        default=None,
        help=(
            "One-off fix: record START_DATE as the start date in every "
            "watermark file that has none, then exit without sending a "
            "single price request. Older watermarks record only an end date, "
            "and the start is never guessed, because only you know what "
            "window they were fetched over. Recorded starts are left "
            "untouched. TIINGO_API_KEY must still be set, because the "
            "download object requires it when it is created."
        ),
    )
    parser.add_argument(
        "--legacy-watermarks",
        type=str,
        choices=list(SOURCE.acquisition_cls.LEGACY_WATERMARK_POLICIES),
        default=SOURCE.acquisition_cls.DEFAULT_LEGACY_WATERMARK_POLICY,
        help=(
            "What to do with a watermark that records no start date "
            "(default "
            f"'{SOURCE.acquisition_cls.DEFAULT_LEGACY_WATERMARK_POLICY}'). "
            "'warn' skips the symbol but reports the count and the stamping "
            "command on every run; 'refetch' treats unknown coverage as "
            "missing and downloads the symbol again. Passed through "
            "config.kwargs, so it is recorded in the config."
        ),
    )
    parser.add_argument(
        "--wait-for-quota",
        action="store_true",
        help=(
            "When Tiingo reports the request quota used up, wait "
            "--quota-wait-seconds and resume, up to --quota-max-waits times. "
            "Off by default, so no run silently sits waiting for an hour. "
            "Either way the run stops sending requests once the quota is "
            "used up instead of wasting the rest of the roster on requests "
            "that fail at once, and every watermark is kept so a later run "
            "resumes exactly there."
        ),
    )
    parser.add_argument(
        "--quota-wait-seconds",
        type=int,
        default=SOURCE.acquisition_cls.DEFAULT_QUOTA_WAIT_SECONDS,
        help=(
            "Seconds to wait before each resume attempt (default "
            f"{SOURCE.acquisition_cls.DEFAULT_QUOTA_WAIT_SECONDS}). "
            "Tiingo does not publish whether its quota resets at the top of "
            "the hour or over a rolling window, so this is a fixed interval, "
            "not a computed reset time; one hour from the moment the quota "
            "runs out covers either case."
        ),
    )
    parser.add_argument(
        "--quota-max-waits",
        type=int,
        default=SOURCE.acquisition_cls.DEFAULT_QUOTA_MAX_WAITS,
        help=(
            "How many times to wait and resume before giving up (default "
            f"{SOURCE.acquisition_cls.DEFAULT_QUOTA_MAX_WAITS}). Bounded on "
            "purpose, so a run cannot wait forever against a lockout. The "
            "default fits the observed numbers: about 4,600 requests per "
            "quota window against about 14.7k symbols is about three "
            "windows."
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

    # Must run before any config factory is called: the factories copy the
    # data root into their paths when called, so a later override is ignored.
    apply_data_dir(args)

    if args.end_date is None:
        args.end_date = datetime.date.today().isoformat()

    catalog = UniverseCatalog.load(universe_config())
    # Every symbol listed at any time in the window, not the members on one
    # day: dropping names that delisted inside the window would bring back
    # survivorship bias.
    symbols = resolve_symbols(args, catalog, mode="in_range")

    # ``subdir`` keeps this roster's raw files and watermarks apart from
    # those of ``ingest_tiingo.py``.
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
    # ``symbols=None``: the conversion takes its symbol axis from the raw
    # files, which hold only what actually downloaded. Passing the roster
    # here would give a second, competing answer.
    ds_config = stock_kline_config(
        symbols=None,
        start_date=args.start_date,
        end_date=args.end_date,
        subdir=DEFAULT_SUBDIR,
        store_name=DEFAULT_STORE_NAME,
    )

    if args.stamp_legacy_watermarks is not None:
        # Handled before every other mode, so that combining it with another
        # flag can never start a download. It writes files, so it goes
        # through the acquisition class (which requires the key) rather than
        # the read-only inspector.
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
            f"{args.start_date}..{args.end_date}. Build or refresh the universe "
            f"table first: uv run python scripts/refresh_us_equity_universe.py"
        )

    # Pre-flight: no client exists and no request has been sent yet. It
    # comes after the stamping and dry-run exits, which send no request.
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
    # These counts cover this run only. The failure manifest accumulates
    # across runs and may name more symbols; read it through
    # ``SourceInspector.failures()``.
    print(
        f"{len(result.succeeded)} symbol(s) succeeded, "
        f"{len(result.failures)} failed"
    )
    print(f"Raw data written under: {acq_config.raw_data_dir_path}")

    if args.to_zarr:
        # This dataset only answers ``has_raw_data()``, so it needs no symbols.
        refuse_conversion_without_raw_data(
            StockDataset(replace(ds_config, symbols=None)), result
        )
        print(
            f"Converting/persisting {len(symbols)} symbols to Zarr in "
            f"{args.chunk} windows (resumable; completed windows are skipped)"
        )
        # Print the registry's own result, so what is printed is what was
        # written.
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
