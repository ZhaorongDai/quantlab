"""Bulk-backfill the full US listed-equity market from Tiingo.

Glue only -- exactly the shape `ingest_tiingo.py` established. Every piece of
logic lives in the layered components this script merely wires together:
`acquisition.universe.UniverseCatalog` resolves the roster,
`acquisition.tiingo.ConcurrentTiingoAcquisition` fetches it, and
`dataset.stock.StockDataset` converts it. Nothing here should grow a
behaviour that a component could own instead.

`TIINGO_API_KEY` is read from the environment by `TiingoAcquisition.__init__`
and is NEVER printed, logged, or written to any artifact by this script or by
anything it calls -- not the failure manifest, not an exception message, not
the `TiingoClient` config dict. Only symbol lists, date ranges and paths are
ever printed.

Storage is rooted at `QUANTLAB_DATA_DIR` (see `config/__init__.py:_data_root`),
which is the ONLY path knob: this script hardcodes no volume and adds no
competing setting.

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
    # 1. How big is this? Resolves the roster and sizes the panel, issuing
    #    ZERO price requests -- no API key needed.
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
"""

import argparse
import datetime

from acquisition.tiingo import ConcurrentTiingoAcquisition
from acquisition.universe import UniverseCatalog
from base.chunking import TimeChunkPlanner
from config import stock_acquisition_config, stock_kline_config, universe_config
from dataset.stock import StockDataset

#: D-05. The backfill window's default start. Applied as an interval-OVERLAP
#: bound, not as a listing-date cut -- see `get_symbols_in_range`.
DEFAULT_START_DATE = "2006-01-01"

#: Raw-data subdirectory and Zarr store name for this roster, kept separate
#: from `stock_kline_config`'s NASDAQ-only defaults so the two backfills have
#: independent watermarks and neither overwrites the other. Both resolve
#: BENEATH `QUANTLAB_DATA_DIR` (D-04).
DEFAULT_SUBDIR = "us_all"
DEFAULT_STORE_NAME = "us_all.zarr"

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
    parser.add_argument(
        "--start-date",
        type=str,
        default=DEFAULT_START_DATE,
        help=(
            f"Window start (inclusive), default {DEFAULT_START_DATE}. Applied "
            f"as interval OVERLAP: every symbol that traded at ANY point in "
            f"the window is kept, INCLUDING those that delisted inside it. "
            f"Only symbols whose listing ended before this date are dropped."
        ),
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="Window end (inclusive), default today.",
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
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Process only the first N resolved symbols. For smoke-testing the "
            "pipeline end to end before committing to the full roster."
        ),
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
        "--max-workers",
        type=int,
        default=ConcurrentTiingoAcquisition.DEFAULT_MAX_WORKERS,
        help=(
            "Concurrent in-flight symbol fetches (default "
            f"{ConcurrentTiingoAcquisition.DEFAULT_MAX_WORKERS}). Passed "
            "through config.kwargs, so it stays config-driven."
        ),
    )
    parser.add_argument(
        "--to-zarr",
        action="store_true",
        help=(
            "After acquisition, convert the raw parquet into the Zarr store. "
            "The full window is no longer refused: the conversion densifies "
            "and appends ONE --chunk window at a time, so peak RAM scales "
            "with the window rather than the range, and an interrupted run "
            "resumes at the first unwritten window. OFF by default because "
            "it is slow, not because it is impossible."
        ),
    )
    parser.add_argument(
        "--chunk",
        type=str,
        choices=list(TimeChunkPlanner.GRANULARITIES),
        default="year",
        help=(
            "Time granularity of one --to-zarr conversion window (default "
            "year). Finer windows use less peak RAM and give a finer resume "
            "granularity, at the cost of more append round trips. Pass "
            "'month' for a dense year the per-chunk sizing guard refuses."
        ),
    )
    return parser


if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.end_date is None:
        args.end_date = datetime.date.today().isoformat()

    catalog = UniverseCatalog.load(universe_config())
    # Interval OVERLAP, deliberately NOT get_symbols_as_of(): a backfill wants
    # every symbol that traded at ANY point in the window, including the ~6.9k
    # that delisted inside it. Resolving membership on a single day here would
    # reintroduce exactly the survivorship bias this roster exists to remove.
    symbols = tuple(
        catalog.get_symbols_in_range(args.category, args.start_date, args.end_date)
    )
    if args.limit is not None:
        symbols = symbols[: args.limit]

    acq_config = stock_acquisition_config(
        symbols=symbols,
        start_date=args.start_date,
        end_date=args.end_date,
        subdir=DEFAULT_SUBDIR,
        kwargs={"max_workers": args.max_workers, "resume": True},
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

    if args.dry_run:
        print(f"DRY RUN -- category={args.category}, no price requests issued")
        _print_estimate(catalog, args, symbols)
        print(f"  raw-data path:     {acq_config.raw_data_dir_path}")
        print(f"  watermark path:    {acq_config.watermark_path}")
        print(f"  zarr path:         {ds_config.zarr_file_path}")
        raise SystemExit(0)

    if not symbols:
        parser.error(
            f"{args.category} resolved to zero symbols over "
            f"{args.start_date}..{args.end_date}. Build/refresh the universe "
            f"table first: uv run python refresh_us_equity_universe.py"
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
        f"Acquiring {len(symbols)} symbols from Tiingo "
        f"({args.start_date}..{args.end_date}, refresh={args.refresh}, "
        f"max_workers={args.max_workers}). Already-complete symbols are "
        f"skipped; per-symbol failures land in "
        f"{acq_config.watermark_path}/"
        f"{ConcurrentTiingoAcquisition.FAILURE_MANIFEST_NAME}."
    )
    acquisition = ConcurrentTiingoAcquisition(acq_config)
    if args.refresh:
        acquisition.refresh()
    else:
        acquisition.download()
    print(f"Raw data written under: {acq_config.raw_data_dir_path}")

    if args.to_zarr:
        print(
            f"Converting/persisting {len(symbols)} symbols to Zarr in "
            f"{args.chunk} windows (resumable; completed windows are skipped)"
        )
        StockDataset(ds_config).from_raw_data_chunked(granularity=args.chunk)
        print(f"Zarr store written at: {ds_config.zarr_file_path}")
    else:
        print(
            "Skipping Zarr conversion (default). Pass --to-zarr to convert; "
            "--chunk selects the window granularity."
        )
