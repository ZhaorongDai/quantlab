"""Download ETF daily data from WRDS CRSP and save each ETF to its own Zarr store.

An ETF is used as a buy-and-hold benchmark (``BacktestConfig.benchmark_dataset``),
so it lives in a store of its own rather than in an equity panel: an ETF ranked
against the stocks it holds would be the index competing with itself, and the
equity scripts' default ``equity_common`` filter drops funds anyway.

For each ETF the script pulls its ``crsp_a_stock.dsf_v2`` rows by PERMNO into
the shared raw tier ``<data root>/downloads/us_equity/1d/wrds_crsp/wrds/`` (the
same tree, watermarks and CRSP release check as ``scripts/ingest_wrds_crsp.py``)
and converts them into ``<data root>/data/us_equity/1d/wrds_crsp_<name>_1d.zarr``
with ``CrspDatasetConfig.etf_benchmark``: the PERMNO alone, with
``security_filter="none"``. ``adjOpen``/``adjClose`` are total-return adjusted,
so a buy-and-hold on them includes the ETF's distributions.

ETFs are named with ``--etf``, comma-separated. A known name (``spy``, ``qqq``)
maps to its PERMNO; any other ETF is given as ``name=PERMNO``, and the name is
used in the store file name.

Credentials: ``WRDS_USERNAME`` must be set in the environment. The password is
never read by this code; libpq takes it from ``~/.pgpass`` (mode 600). The
whole run shares one WRDS connection, so it costs at most one Duo push.

Before any daily row is copied the run checks the account's entitlement to
the stock schema, clips the end date to the CRSP annual product end, pulls the
reference tables the conversion reads (ticker history), and counts the rows it
would copy against the ``SqlVolumeGuard`` ceilings (``--force-volume`` to
override). ``--refresh`` continues each ETF from its watermark. See
``docs/wrds_crsp.md``.

Usage:
    export WRDS_USERNAME=<your-wrds-username>   # password lives in ~/.pgpass

    # The S&P 500 and Nasdaq-100 benchmarks.
    uv run python scripts/ingest_wrds_crsp_etf.py --etf spy,qqq \\
        --start-date 2000-01-01 --end-date 2025-12-31

    # Any other ETF, by PERMNO (name=PERMNO); the name goes in the store name.
    uv run python scripts/ingest_wrds_crsp_etf.py --etf spy,myetf=12345 \\
        --start-date 2010-01-01 --end-date 2025-12-31
"""

import argparse
import re

from quantlab.registry import DataSourceRegistry, convert, run
from quantlab.base.config import QQQ_PERMNO, SPY_PERMNO, CrspDatasetConfig
from quantlab.config import get_data_root
from quantlab.dataset.crsp import CrspStockDataset
from quantlab.utils.cli import (
    add_chunk_args,
    add_data_dir_arg,
    add_volume_guard_args,
    add_window_args,
    apply_data_dir,
    print_conversion_result,
    print_sql_volume_estimate,
    refuse_conversion_without_raw_data,
)

#: The vendor descriptor, resolved from its registry token.
SOURCE = DataSourceRegistry.get("wrds")

#: The capability this script drives, the same one as the equity scripts, so
#: ETFs share their raw tier, watermarks and release check.
CAPABILITY = ("us_equity", "1d", "crsp_daily")

#: The CRSP daily download class, looked up rather than imported.
ACQ = SOURCE.acquisition_cls_for(*CAPABILITY)

#: ETFs that can be named without a PERMNO.
KNOWN_ETFS: dict[str, str] = {
    "spy": SPY_PERMNO,  # SPDR S&P 500 ETF Trust
    "qqq": QQQ_PERMNO,  # Invesco QQQ Trust (Nasdaq-100)
}

#: An ETF's own store. The same names ``ingest_wrds_crsp.py --benchmark``
#: writes, so either script produces the store the other would.
STORE_TEMPLATE = "wrds_crsp_{name}_1d.zarr"

#: A store name: lowercase letters, digits and underscores.
_NAME = re.compile(r"^[a-z0-9_]+$")


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the argument parser for this script."""
    parser = argparse.ArgumentParser(
        description=(
            "Pull ETF daily rows from WRDS CRSP by PERMNO and convert each ETF "
            "into its own Zarr store. Requires WRDS_USERNAME in the "
            "environment and the password in ~/.pgpass (mode 600); neither "
            "is accepted as an argument."
        )
    )
    parser.add_argument(
        "--etf",
        type=str,
        required=True,
        help=(
            f"Comma-separated ETFs. A known name ({', '.join(sorted(KNOWN_ETFS))}) "
            f"or name=PERMNO for any other ETF, e.g. spy,qqq or spy,myetf=12345. "
            f"Each is written to data/us_equity/1d/wrds_crsp_<name>_1d.zarr."
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
            f"PERMNOs per download query and per row-count query (default "
            f"{ACQ.DEFAULT_BATCH_SIZE})."
        ),
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help=(
            "Continue each ETF from its recorded watermark instead of "
            "downloading the whole window again."
        ),
    )
    parser.add_argument(
        "--refresh-reference",
        action="store_true",
        help=(
            "Download the _reference/ tables again even when the tables for "
            "this CRSP release are already complete."
        ),
    )
    add_chunk_args(parser, default="year")
    return parser


def _parse_etfs(parser: argparse.ArgumentParser, value: str) -> dict[str, str]:
    """Return ``{store name: PERMNO}`` for ``--etf``, refusing bad entries.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser, used to report an error and exit with status 2.
    value : str
        The ``--etf`` value.

    Returns
    -------
    dict[str, str]
        Store name to PERMNO, in the order given.

    Examples
    --------
    >>> _parse_etfs(parser, "spy, myetf=12345")
    {'spy': '84398', 'myetf': '12345'}
    """
    etfs: dict[str, str] = {}
    for token in (part.strip() for part in value.split(",")):
        if not token:
            continue
        name, _, permno = token.partition("=")
        name, permno = name.strip().lower(), permno.strip()
        if not permno:
            if name not in KNOWN_ETFS:
                parser.error(
                    f"--etf {name!r} is not a known ETF ({', '.join(sorted(KNOWN_ETFS))}); "
                    f"give any other ETF as name=PERMNO. CRSP is keyed by "
                    f"PERMNO, never by ticker."
                )
            permno = KNOWN_ETFS[name]
        if not _NAME.match(name):
            parser.error(
                f"--etf name {name!r} must be lowercase letters, digits or "
                f"underscores; it becomes part of the store file name."
            )
        if not permno.isdigit():
            parser.error(f"--etf {token!r}: {permno!r} is not a PERMNO.")
        if name in etfs:
            parser.error(f"--etf names {name!r} twice.")
        if permno in etfs.values():
            parser.error(f"--etf names PERMNO {permno} twice.")
        etfs[name] = permno
    if not etfs:
        parser.error("--etf names no ETF.")
    return etfs


def _validate(parser: argparse.ArgumentParser, args) -> dict[str, str]:
    """Refuse bad arguments before any WRDS connection is opened."""
    etfs = _parse_etfs(parser, args.etf)
    if not args.start_date or not args.end_date:
        parser.error(
            "--start-date and --end-date are both required: the window is "
            "sized before any data is pulled, and it is checked against the "
            "end date of the annual CRSP release."
        )
    if args.rows_per_symbol_day is not None:
        parser.error(
            "--rows-per-symbol-day does not apply to CRSP: rows are counted "
            "server-side with count(*) per calendar year and PERMNO batch "
            "before the pull."
        )
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("--batch-size must be a positive integer.")
    return etfs


if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Must run before any path is derived from the data root.
    apply_data_dir(args)

    etfs = _validate(parser, args)

    batch_size = args.batch_size or ACQ.DEFAULT_BATCH_SIZE
    kwargs = {"batch_size": batch_size, "clip_to_product_end": True}

    # Imported here rather than at the top so that tests can replace the
    # session class with an offline stand-in before the run starts.
    from quantlab.acquisition._support.sql_volume import SqlVolumeGuard
    from quantlab.acquisition.wrds.crsp import CrspQueries, CrspVolumeProbe
    from quantlab.acquisition.wrds.crsp_reference import CrspReferenceTables
    from quantlab.acquisition.wrds.taq import WrdsSession

    try:
        session = WrdsSession.shared()
    except RuntimeError as exc:
        parser.exit(1, f"{exc}\n")

    try:
        # 1. Check access to the stock schema before anything is copied.
        try:
            CrspQueries.assert_entitled(session, (CrspQueries.STOCK_SCHEMA,))
        except (RuntimeError, ValueError) as exc:
            parser.exit(1, f"{exc}\n")

        # 2. Clip the window to the end of the annual CRSP release, and say so.
        try:
            start_date, end_date, clipped = ACQ.resolve_window(
                session, args.start_date, args.end_date, clip=True
            )
        except (RuntimeError, ValueError) as exc:
            parser.exit(1, f"{exc}\n")
        if clipped is not None:
            print(
                f"clipped end {args.end_date} -> {clipped.isoformat()} "
                f"(crsp_a_stock annual product end)"
            )
        window = {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
        }

        # 3. The download config, with the ETF PERMNOs as its roster (sorted
        #    numerically so batch boundaries are stable across runs).
        roster = tuple(str(p) for p in sorted({int(p) for p in etfs.values()}))
        acq_config = SOURCE.config_factory_for(*CAPABILITY)(
            symbols=roster,
            start_date=window["start_date"],
            end_date=window["end_date"],
            kwargs=kwargs,
        )
        reference_dir = ACQ.reference_dir_for(acq_config)

        # 4. The reference tables the conversion reads (ticker history). No
        #    index tables: an ETF roster needs no membership.
        try:
            CrspReferenceTables(session, reference_dir).pull(
                product_end=end_date,
                include_sp500=False,
                include_nasdaq100=False,
                refresh=args.refresh_reference,
            )
        except (RuntimeError, ValueError) as exc:
            parser.exit(1, f"{exc}\n")
        print(f"Reference tables at: {reference_dir}")
        print(
            "ETFs: "
            + ", ".join(f"{name.upper()} (PERMNO {permno})" for name, permno in etfs.items())
        )

        # 5. Count rows and check the volume ceilings before any daily row is
        #    copied.
        try:
            counts = CrspVolumeProbe(
                session, batch_size=batch_size
            ).count_rows_by_year(list(roster), window["start_date"], window["end_date"])
            estimate = SqlVolumeGuard(acq_config.kwargs).assert_acquisition_volume_fits(
                counts,
                symbols=len(roster),
                start_date=window["start_date"],
                end_date=window["end_date"],
                force=args.force_volume,
                unit="year bucket",
            )
        except (RuntimeError, ValueError) as exc:
            parser.exit(1, f"{exc}\n")
        print_sql_volume_estimate(estimate, forced=args.force_volume)

        # 6. Download.
        print(
            f"Acquiring {len(roster)} ETF(s) from {SOURCE.display_name} over "
            f"{window['start_date']}..{window['end_date']} (refresh={args.refresh})"
        )
        result = run(SOURCE, acq_config, refresh=args.refresh)
        print(
            f"{len(result.succeeded)} ETF(s) succeeded, "
            f"{len(result.failures)} failed"
        )
        print(f"Raw data written under: {acq_config.raw_data_dir_path}")

        # 7. Convert each ETF into its own store. One ETF failing does not
        #    stop the others; the run exits 1 at the end if any failed.
        data_dir = get_data_root() / "data" / "us_equity" / "1d"
        failed: list[str] = []
        for name, permno in etfs.items():
            config = CrspDatasetConfig.etf_benchmark(
                permno=permno,
                zarr_file_path=str(data_dir / STORE_TEMPLATE.format(name=name)),
                raw_data_dir_path=acq_config.raw_data_dir_path,
                reference_dir=str(reference_dir),
                start_date=window["start_date"],
                end_date=window["end_date"],
            )
            refuse_conversion_without_raw_data(CrspStockDataset(config), result)
            print(
                f"Converting {name.upper()} (PERMNO {permno}) into "
                f"{config.zarr_file_path}"
            )
            try:
                print_conversion_result(
                    convert(
                        SOURCE,
                        config,
                        data_type="crsp_daily",
                        granularity=args.chunk,
                        on_new_listing=args.on_new_listing,
                    )
                )
            except (RuntimeError, ValueError) as exc:
                print(f"{name.upper()} (PERMNO {permno}) was not converted: {exc}")
                failed.append(name)
        if failed:
            parser.exit(
                1,
                f"{len(failed)} ETF(s) not converted: {', '.join(failed)}. A "
                f"PERMNO with no rows in the window (not an ETF CRSP carries, "
                f"or listed after the window) has nothing to convert.\n",
            )
    finally:
        WrdsSession.close_shared()
