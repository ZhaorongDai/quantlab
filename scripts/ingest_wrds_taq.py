"""Pull WRDS NYSE TAQ consolidated NBBO records for a symbol roster and,
optionally, resample them locally into a `[timestamp, symbol]` bar panel.

What it does
------------
Each trading day's raw NBBO records are read from
`taqm_{YYYY}.complete_nbbo_{YYYYMMDD}` (the complete NBBO table, not `nbbom`,
which misses single-venue NBBO states) with one PostgreSQL `COPY` per
(trading day, symbol batch). The raw tier lands under
`<data root>/downloads/us_equity/tick/wrds_taq/wrds/data_type=nbbo/` as hive
shards, every record kept in arrival order. With `--to-zarr` the raw tier is
resampled by `quantlab.registry.convert()` into a right-closed
bar panel (`--bar-interval`, default 1m) over the session window
(`--session-start`/`--session-end`, default regular hours 09:30-16:00 ET),
and a `<store>.nbbo_filter_stats.json` sidecar records what the default
filters dropped per session date and symbol.

This script names no vendor class: it resolves its source with
`DataSourceRegistry.get("wrds")`, builds the acquisition config through
`SOURCE.config_factory` and constructs the dataset config directly. It calls
no `quantlab/config` factory function.

Credentials
-----------
`WRDS_USERNAME` is read from the ENVIRONMENT. The password is never read by
this code at all: libpq reads it from `~/.pgpass` (mode 600), one line of the
form `wrds-pgdata.wharton.upenn.edu:9737:wrds:<username>:<password>`. No
argument accepts a username or a password, and nothing this script prints
contains either: a credential on the command line lands in shell history and
in every process listing, and this repo has already leaked one real key.

One Duo push per run
--------------------
Every new WRDS connection can push a Duo prompt to the account holder's phone.
The volume probe, the pull and the conversion therefore share ONE session
(`WrdsSession.shared()`), closed in a `finally` when the run ends. The pull
runs on that one connection; there is no worker-count flag to raise, and
`wait_for_quota` is not offered either: after a session failure the run stops
(a broken session is never reopened) and the next run resumes from the
recorded pages.

Order of checks
---------------
1. Arguments, including the session window (checked by `XnysSessionCalendar`
   before any connection, so a bad window costs no WRDS query).
2. Entitlement: every year of the window must be readable (`taqm_YYYY`);
   an unentitled year stops the run before any count or COPY.
3. Volume: rows are counted with `count(*)` per (trading day, symbol batch)
   BEFORE any data is pulled, and `SqlVolumeGuard` refuses a pull over the
   20 GiB raw-byte ceiling (or the 700M row ceiling). Long ranges run as
   several guarded date segments; a refusal names the longest segment from
   `--start-date` that fits. `--force-volume` skips the refusal, never the
   arithmetic.
4. The pull (`registry.run`), then the optional conversion.

Usage:
    export WRDS_USERNAME=<your-wrds-username>   # password lives in ~/.pgpass

    # Raw NBBO records for an explicit roster (dot notation: BRK.B).
    uv run python scripts/ingest_wrds_taq.py --symbols AAPL,MSFT,BRK.B \
        --start-date 2024-01-24 --end-date 2024-01-25

    # The same pull, resampled to 1-minute bars afterwards.
    uv run python scripts/ingest_wrds_taq.py --symbols AAPL,MSFT,BRK.B \
        --start-date 2024-01-24 --end-date 2024-01-25 --to-zarr --bar-interval 1m

    # Point-in-time S&P 500 constituents over the window (interval overlap).
    uv run python scripts/ingest_wrds_taq.py --universe sp500 \
        --start-date 2024-01-24 --end-date 2024-01-24 --to-zarr
"""

import argparse
import typing

from dataclasses import replace

from quantlab.registry import DataSourceRegistry, convert, run
from quantlab.universe import UniverseCatalog
from quantlab.base.config import NbboDatasetConfig, UniverseConfig
from quantlab.config import get_data_root
from quantlab.dataset.nbbo import NbboPanelDataset
from quantlab.dataset.session_calendar import XnysSessionCalendar
from quantlab.enums.data import BarInterval
from quantlab.utils.cli import (
    add_chunk_args,
    add_data_dir_arg,
    add_to_zarr_arg,
    add_volume_guard_args,
    add_window_args,
    apply_data_dir,
    print_conversion_result,
    print_sql_volume_estimate,
    refuse_conversion_without_raw_data,
    resolve_symbols,
)

#: The one place this script's vendor is named, as a registry token.
SOURCE = DataSourceRegistry.get("wrds")

#: The universes WRDS pulls are offered for: the point-in-time index
#: rosters, which use the dot notation TAQ needs (D-15). `nasdaq_all` and
#: `us_all` are exchange listings in hyphen notation and far too large for a
#: quote-level pull.
UNIVERSES: tuple[str, ...] = ("sp500", "nasdaq100")

#: Bar sizes, derived from the locked `BarInterval` literal.
BAR_INTERVALS: tuple[str, ...] = typing.get_args(BarInterval)

#: The Zarr store a `--to-zarr` run writes to, under
#: `<data root>/data/us_equity/tick/`. The session window is part of the
#: name: the conversion ledger fingerprints only the symbol axis, so two
#: windows sharing one store would silently skip the second window's dates
#: as already written.
DEFAULT_STORE_TEMPLATE = "wrds_nbbo_{bar_interval}_{session_start}-{session_end}.zarr"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Pull WRDS TAQ consolidated NBBO records and optionally resample "
            "them into a bar panel. Requires WRDS_USERNAME in the environment "
            "and the password in ~/.pgpass (mode 600); neither is accepted as "
            "an argument."
        )
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help=(
            "Comma-separated symbols in dot notation, e.g. AAPL,MSFT,BRK.B "
            "(BRK.B is root BRK, suffix B). Hyphenated forms are refused."
        ),
    )
    parser.add_argument(
        "--universe",
        type=str,
        choices=list(UNIVERSES),
        default=None,
        help=(
            "Point-in-time index constituents, resolved by interval OVERLAP "
            "over [--start-date, --end-date]: every symbol that was a member "
            "at any point in the window, in dot notation (BRK.B)."
        ),
    )
    add_window_args(parser)
    add_volume_guard_args(parser)
    add_data_dir_arg(parser)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=(
            "Symbols per COPY and per count(*) (default "
            f"{SOURCE.acquisition_cls.DEFAULT_BATCH_SIZE})."
        ),
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help=(
            "Incrementally refresh from each symbol's recorded watermark "
            "instead of a full backfill of the window."
        ),
    )
    add_to_zarr_arg(parser)
    add_chunk_args(parser, default="day")
    parser.add_argument(
        "--bar-interval",
        type=str,
        choices=list(BAR_INTERVALS),
        default="1m",
        help=(
            "Bar size of the --to-zarr panel (default 1m). Labels are "
            "right-closed bar ends. Every choice divides both a 390-minute "
            "regular session and a 210-minute half day."
        ),
    )
    window_help = (
        "ET wall clock HH:MM. The window may be anywhere inside 04:00-20:00 "
        "ET. On a half day only an edge inside regular hours (09:30-16:00) "
        "follows the early close; an extended-hours edge stays as given."
    )
    parser.add_argument(
        "--session-start",
        type=str,
        default="09:30",
        help=f"Session window start of the --to-zarr panel (default 09:30). {window_help}",
    )
    parser.add_argument(
        "--session-end",
        type=str,
        default="16:00",
        help=f"Session window end of the --to-zarr panel (default 16:00). {window_help}",
    )
    return parser


def _validate(parser: argparse.ArgumentParser, args) -> XnysSessionCalendar:
    """Every argument refusal, BEFORE any WRDS session exists. Returns the
    session calendar the window check built."""
    if bool(args.symbols) == bool(args.universe):
        parser.error("Exactly one of --symbols or --universe must be set.")
    if not args.start_date or not args.end_date:
        parser.error(
            "--start-date and --end-date are both required: the window is "
            "counted and priced before any data is pulled."
        )
    if args.rows_per_symbol_day is not None:
        parser.error(
            "--rows-per-symbol-day does not apply to WRDS: rows are counted "
            "server-side with count(*) per trading day and symbol batch "
            "before the pull."
        )
    if args.symbols:
        hyphenated = [
            token.strip() for token in args.symbols.split(",") if "-" in token
        ]
        if hyphenated:
            parser.error(
                f"--symbols {hyphenated} use a hyphen; WRDS TAQ uses dot "
                f"notation (e.g. BRK.B for root BRK, suffix B)."
            )
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("--batch-size must be a positive integer.")
    try:
        return XnysSessionCalendar(args.session_start, args.session_end)
    except ValueError as exc:
        parser.error(f"--session-start/--session-end: {exc}")


if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Before any path is derived from the data root (DDIR-04).
    apply_data_dir(args)

    calendar = _validate(parser, args)

    catalog = None
    if args.universe:
        reference = get_data_root() / "data" / "reference"
        catalog = UniverseCatalog.load(
            UniverseConfig(
                output_path=str(reference / "universe.parquet"),
                cache_dir=str(reference / "_cache"),
            )
        )
    # Interval overlap across the window, never one as-of day: a backfill
    # wants every symbol that was a member at any point in it.
    symbols = resolve_symbols(args, catalog, mode="in_range")
    if not symbols:
        parser.error("the roster resolved to no symbols.")

    batch_size = args.batch_size or SOURCE.acquisition_cls.DEFAULT_BATCH_SIZE
    acq_config = SOURCE.config_factory(
        symbols=symbols,
        start_date=args.start_date,
        end_date=args.end_date,
        kwargs={"batch_size": batch_size},
    )

    # Imported here so the module attribute is read at run time (the test
    # suite patches it with an offline double).
    from quantlab.acquisition.sql_volume import SqlVolumeGuard
    from quantlab.acquisition.wrds.taq import WrdsNbboVolumeProbe, WrdsSession

    try:
        session = WrdsSession.shared()
    except RuntimeError as exc:
        parser.exit(1, f"{exc}\n")

    try:
        # Entitlement, then count(*) per (day, batch), then the guard -- all
        # BEFORE the first COPY (D-16, D-21, D-24).
        try:
            rows_by_day = WrdsNbboVolumeProbe(
                session, batch_size=batch_size
            ).count_rows_by_day(symbols, args.start_date, args.end_date)
            estimate = SqlVolumeGuard(acq_config.kwargs).assert_acquisition_volume_fits(
                rows_by_day,
                symbols=len(symbols),
                start_date=args.start_date,
                end_date=args.end_date,
                force=args.force_volume,
            )
        except (RuntimeError, ValueError) as exc:
            parser.exit(1, f"{exc}\n")
        print_sql_volume_estimate(estimate, forced=args.force_volume)

        print(
            f"Acquiring {len(symbols)} symbol(s) from {SOURCE.display_name} "
            f"over {args.start_date}..{args.end_date} (refresh={args.refresh})"
        )
        result = run(SOURCE, acq_config, refresh=args.refresh)
        print(
            f"{len(result.succeeded)} symbol(s) succeeded, "
            f"{len(result.failures)} failed"
        )
        print(f"Raw data written under: {acq_config.raw_data_dir_path}")

        if args.to_zarr:
            store_name = DEFAULT_STORE_TEMPLATE.format(
                bar_interval=args.bar_interval,
                session_start=calendar.session_start.strftime("%H%M"),
                session_end=calendar.session_end.strftime("%H%M"),
            )
            ds_config = NbboDatasetConfig(
                zarr_file_path=str(
                    get_data_root() / "data" / "us_equity" / "tick" / store_name
                ),
                raw_data_dir_path=acq_config.raw_data_dir_path,
                catalog_path=str(get_data_root() / "data" / "catalog"),
                start_date=args.start_date,
                end_date=args.end_date,
                symbols=tuple(symbols),
                bar_interval=args.bar_interval,
                session_start=args.session_start,
                session_end=args.session_end,
            )
            probe_dataset = NbboPanelDataset(replace(ds_config, symbols=None))
            refuse_conversion_without_raw_data(probe_dataset, result)
            print(
                f"Resampling {len(symbols)} symbol(s) to {args.bar_interval} "
                f"bars over {args.session_start}-{args.session_end} ET in "
                f"{args.chunk} windows (resumable)"
            )
            conversion = convert(
                SOURCE,
                ds_config,
                data_type="nbbo",
                granularity=args.chunk,
                on_new_listing=args.on_new_listing,
            )
            print_conversion_result(conversion)
            print(f"Filter statistics sidecar: {probe_dataset.filter_stats_path}")
        else:
            print(
                "Skipping Zarr conversion (default). The raw shards above are "
                "the deliverable; pass --to-zarr to resample them into bars."
            )
    finally:
        WrdsSession.close_shared()
