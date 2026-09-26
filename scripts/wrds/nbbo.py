"""Download NYSE TAQ NBBO quotes from WRDS and resample them into a bar panel.

TAQ (Trade and Quote) is the NYSE's millisecond record of US equity trading,
served through WRDS one ``taqm_YYYY.complete_nbbo_YYYYMMDD`` table per
trading day. The NBBO (National Best Bid and Offer) is the highest bid and
lowest ask across all exchanges at each instant. This script pulls the NBBO
rows of a symbol roster over a window and resamples them into
``--interval`` bars inside the ``--session`` window (US Eastern time),
written as ``wrds_nbbo_{interval}_{HHMM-HHMM}.zarr`` into ``--zarr-dir``
with its filter-statistics sidecar. The raw rows go to
``<download-dir>/wrds/``; both directories default to the current one.

The roster is exactly one of:

- ``--symbols``, explicit tickers in TAQ's dot notation (``BRK.B``);
- ``--index sp500|nasdaq100``, the point-in-time members of that index over
  the window, resolved from the CRSP reference tables (pulled into
  ``<download-dir>/_reference/``, where ``index.py`` keeps them for the same
  ``--download-dir``) and mapped to the tickers they traded under. This
  needs the CRSP subscription beside the TAQ one.

``WRDS_USERNAME`` must be set in the environment. The password is never read
by this code; the PostgreSQL client library takes it from ``~/.pgpass``. One
run shares one WRDS session: ``--max-workers`` threads download in parallel,
each on its own connection, and every connection is closed at the end
whether the run succeeded or failed.

Usage::

    export WRDS_USERNAME=<your-wrds-username>   # password lives in ~/.pgpass
    uv run python scripts/wrds/nbbo.py --symbols AAPL,MSFT,BRK.B \\
        --start 2024-01-02 --end 2024-01-31
    uv run python scripts/wrds/nbbo.py --index nasdaq100 --start 2024-01-02 \\
        --interval 5m --session 09:30-16:00 --refresh
    uv run python scripts/wrds/nbbo.py --symbols AAPL --start 2024-01-02 \\
        --download-dir /data/taq/raw --zarr-dir /data/taq/zarr

``--end`` defaults to today and is clipped to the last trading day TAQ has
published. ``--refresh`` continues each symbol from its recorded watermark
instead of downloading the whole window again. ``--download-dir`` and
``--zarr-dir`` choose where the raw files and the Zarr store go; both
default to the current directory.
"""

import argparse
import typing
from dataclasses import replace
from datetime import date

import polars as pl

from quantlab.registry import DataSourceRegistry, convert, run
from quantlab.base.config import NbboDatasetConfig
from quantlab.dataset.nbbo import NbboPanelDataset
from quantlab.dataset._support.session_calendar import XnysSessionCalendar
from quantlab.dataset.crsp.membership import CrspMembership
from quantlab.dataset.crsp.reference import CrspReference
from quantlab.dataset.crsp.symbology import CrspSymbology
from quantlab.enums.data import BarInterval
from quantlab.utils.cli import (
    add_max_workers_arg,
    add_output_dir_args,
    place_downloads,
    print_conversion_result,
    resolve_output_dirs,
)

SOURCE = DataSourceRegistry.get("wrds")
NBBO_CAPABILITY = ("us_equity", "tick", "nbbo")
CRSP_CAPABILITY = ("us_equity", "1d", "crsp_daily")
ACQ = SOURCE.acquisition_cls_for(*NBBO_CAPABILITY)

#: ``--index`` name -> CRSP membership universe id.
INDEXES: dict[str, str] = {
    "sp500": CrspMembership.SP500,
    "nasdaq100": CrspMembership.NASDAQ100,
}

BAR_INTERVALS: tuple[str, ...] = typing.get_args(BarInterval)

STORE_TEMPLATE = "wrds_nbbo_{interval}_{session_start}-{session_end}.zarr"

#: The first year with TAQ millisecond tables on WRDS.
TAQ_FIRST_YEAR = 2003


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build this script's argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Download TAQ NBBO quotes from WRDS and resample them into a bar "
            "panel in Zarr. Requires WRDS_USERNAME in the environment and the "
            "password in ~/.pgpass."
        )
    )
    roster = parser.add_mutually_exclusive_group(required=True)
    roster.add_argument(
        "--symbols",
        help=(
            "Comma-separated tickers in TAQ dot notation, e.g. AAPL,MSFT,BRK.B. "
            "Hyphenated forms (BRK-B) are refused."
        ),
    )
    roster.add_argument(
        "--index",
        choices=sorted(INDEXES),
        help=(
            "Point-in-time index members over the window, every ticker that "
            "was a member at any point in it, resolved from the CRSP "
            "reference tables (needs the CRSP subscription)."
        ),
    )
    parser.add_argument(
        "--start", required=True, help="First day of the window, YYYY-MM-DD."
    )
    parser.add_argument(
        "--end",
        default=None,
        help=(
            "Last day of the window, YYYY-MM-DD. Defaults to today and is "
            "clipped to the last trading day TAQ has published."
        ),
    )
    parser.add_argument(
        "--interval",
        choices=list(BAR_INTERVALS),
        default="1m",
        help=(
            "Bar size of the panel (default 1m). Each bar is labelled by its "
            "end time. Every choice divides both a 390-minute regular session "
            "and a 210-minute half day."
        ),
    )
    parser.add_argument(
        "--session",
        default="09:30-16:00",
        help=(
            "Trading session window as HH:MM-HH:MM in US Eastern time, "
            "anywhere inside 04:00-20:00 (default 09:30-16:00). On a half "
            "day an edge inside regular hours moves to the early close."
        ),
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Continue each symbol from its watermark instead of re-downloading.",
    )
    add_max_workers_arg(parser, default=ACQ.DEFAULT_MAX_WORKERS)
    add_output_dir_args(parser)
    return parser


def _parse_symbols(parser: argparse.ArgumentParser, value: str) -> list[str]:
    """Return the sorted, upper-cased ``--symbols`` roster; refuse hyphenated forms."""
    symbols = sorted({token.strip().upper() for token in value.split(",") if token.strip()})
    hyphenated = [symbol for symbol in symbols if "-" in symbol]
    if hyphenated:
        parser.error(
            f"--symbols {hyphenated} use a hyphen; WRDS TAQ uses dot notation "
            f"(BRK.B, not BRK-B)."
        )
    if not symbols:
        parser.error("--symbols names no symbol.")
    return symbols


def _parse_session(parser: argparse.ArgumentParser, value: str) -> XnysSessionCalendar:
    """Build the session calendar from ``--session HH:MM-HH:MM``."""
    start, sep, end = value.partition("-")
    if not sep:
        parser.error(f"--session {value!r}: expected HH:MM-HH:MM.")
    try:
        return XnysSessionCalendar(start.strip(), end.strip())
    except ValueError as exc:
        parser.error(f"--session: {exc}")


def _last_published_day(session, end: date) -> date | None:
    """The last trading day TAQ has a table for, at or before ``end``'s year."""
    for year in range(end.year, TAQ_FIRST_YEAR - 1, -1):
        days = session.trading_days(year)
        if days:
            return days[-1]
    return None


def _index_tickers(session, index: str, start: str, end: str, download_dir) -> list[str]:
    """Resolve an index's point-in-time members to the tickers they traded under.

    The CRSP reference tables are pulled into ``<download_dir>/_reference``,
    the directory ``index.py`` uses for the same ``--download-dir``.
    """
    from quantlab.acquisition.wrds.crsp import CrspQueries
    from quantlab.acquisition.wrds.crsp_reference import CrspReferenceTables

    universe = INDEXES[index]
    schemas = [CrspQueries.STOCK_SCHEMA]
    if universe == CrspMembership.SP500:
        schemas.append(CrspQueries.INDEX_SCHEMA)
    else:
        schemas.extend((CrspQueries.COMPUSTAT_SCHEMA, CrspQueries.CCM_SCHEMA))
    CrspQueries.assert_entitled(session, schemas)

    crsp_acq = SOURCE.acquisition_cls_for(*CRSP_CAPABILITY)
    reference_dir = crsp_acq.reference_dir_for(
        place_downloads(SOURCE.config_factory_for(*CRSP_CAPABILITY)(symbols=()), download_dir)
    )
    CrspReferenceTables(session, reference_dir).pull(
        product_end=CrspQueries.product_end(session),
        include_sp500=universe == CrspMembership.SP500,
        include_nasdaq100=universe == CrspMembership.NASDAQ100,
    )
    reference = CrspReference(reference_dir)
    permnos = [int(p) for p in CrspMembership(reference).permnos_in_range(universe, start, end)]
    intervals = CrspSymbology(reference.table("stksecurityinfohist")).symbol_intervals()
    overlapping = intervals.filter(
        pl.col("permno").is_in(permnos)
        & (pl.col("start_date") <= date.fromisoformat(end))
        & (pl.col("end_date") >= date.fromisoformat(start))
    ).drop_nulls("symbol")
    return sorted(set(overlapping["symbol"].to_list()))


if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()
    download_dir, zarr_dir = resolve_output_dirs(args)
    calendar = _parse_session(parser, args.session)
    explicit = _parse_symbols(parser, args.symbols) if args.symbols else None
    requested_end = args.end or date.today().isoformat()

    # Imported here so the session class is resolved at run time.
    from quantlab.acquisition.wrds.taq import WrdsSession

    try:
        session = WrdsSession.shared()
    except RuntimeError as exc:
        parser.exit(1, f"{exc}\n")

    try:
        try:
            # 1. Clip the window to the last published trading day, then check
            #    that the account may read every year of it.
            start_day, end_day = date.fromisoformat(args.start), date.fromisoformat(requested_end)
            last = _last_published_day(session, end_day)
            if last is not None and end_day > last:
                print(f"clipped end {requested_end} -> {last.isoformat()} (last published TAQ day)")
                end_day = last
            if start_day > end_day:
                raise ValueError(
                    f"--start {args.start} is after the last published TAQ day "
                    f"{end_day.isoformat()}; nothing to download."
                )
            start, end = start_day.isoformat(), end_day.isoformat()
            session.assert_entitled(range(start_day.year, end_day.year + 1))

            # 2. The roster.
            symbols = explicit if explicit is not None else _index_tickers(session, args.index, start, end, download_dir)
        except (RuntimeError, ValueError, FileNotFoundError) as exc:
            parser.exit(1, f"{exc}\n")
        if not symbols:
            parser.exit(1, f"the roster resolved to no symbols over {start}..{end}.\n")
        print(f"Roster: {len(symbols)} symbol(s) over {start}..{end}")

        # 3. Download.
        acq_config = place_downloads(
            SOURCE.config_factory_for(*NBBO_CAPABILITY)(
                symbols=symbols,
                start_date=start,
                end_date=end,
                kwargs={"max_workers": args.max_workers},
            ),
            download_dir,
        )
        print(f"Acquiring from {SOURCE.display_name} (refresh={args.refresh})")
        result = run(SOURCE, acq_config, refresh=args.refresh)
        print(f"{len(result.succeeded)} symbol(s) succeeded, {len(result.failures)} failed")
        print(f"Raw data written under: {acq_config.raw_data_dir_path}")

        # 4. Resample into the bar panel.
        store_name = STORE_TEMPLATE.format(
            interval=args.interval,
            session_start=calendar.session_start.strftime("%H%M"),
            session_end=calendar.session_end.strftime("%H%M"),
        )
        ds_config = NbboDatasetConfig(
            zarr_file_path=str(zarr_dir / store_name),
            raw_data_dir_path=acq_config.raw_data_dir_path,
            start_date=start,
            end_date=end,
            symbols=tuple(symbols),
            bar_interval=args.interval,
            session_start=calendar.session_start.strftime("%H:%M"),
            session_end=calendar.session_end.strftime("%H:%M"),
        )
        probe = NbboPanelDataset(replace(ds_config, symbols=None))
        if not probe.has_raw_data():
            parser.exit(
                1,
                f"Refusing to convert: no raw data under "
                f"{acq_config.raw_data_dir_path} ({len(result.failures)} symbol(s) "
                f"failed this run). No store was written.\n",
            )
        print(f"Resampling to {args.interval} bars over {args.session} ET")
        print_conversion_result(convert(SOURCE, ds_config, data_type="nbbo", granularity="day"))
        print(f"Filter statistics sidecar: {probe.filter_stats_path}")
    finally:
        WrdsSession.close_shared()
