"""Download CRSP daily stock data from WRDS for an index roster.

CRSP (the Center for Research in Security Prices) is the standard academic
database of US stock prices, served through WRDS (Wharton Research Data
Services). CRSP identifies each security by its PERMNO, a permanent integer
id that, unlike a ticker, never changes or gets reused. This script pulls
``crsp_a_stock.dsf_v2`` daily rows for a PERMNO roster: a point-in-time
index universe, an explicit ``--permnos`` list, the QQQ ETF, or any
combination. Point-in-time means the index members as they were on each
date, including the ones that later left, which avoids survivorship bias
(the error of studying only the companies that survived to the present).

The rows are written as raw parquet files under
``<data root>/downloads/us_equity/1d/wrds_crsp/wrds/``, one row per
``(permno, dlycaldt)`` exactly as CRSP serves them. Beside that directory
the script fills ``_reference/`` with the small CRSP, Compustat and CCM
tables that map a PERMNO to its ticker at each date and answer index
membership. Compustat is the S&P accounting database and CCM is the
CRSP/Compustat link table. These tables are pulled first, because the
roster is read from them.

With ``--to-zarr`` the raw files are also converted into up to three Zarr
stores (a chunked on-disk array format that ``xarray`` reads):

- the equity panel ``wrds_crsp_{sp500|nasdaq100|custom}_1d.zarr``, with JSON
  sidecar files describing the price adjustment, the security filter and
  the ticker mapping;
- a benchmark ETF store ``wrds_crsp_{spy|qqq}_1d.zarr`` when ``--qqq`` or
  ``--benchmark`` is given (``--benchmark`` picks the ETF tracking
  ``--universe``: SPY for ``crsp_sp500``, QQQ for ``comp_nasdaq100``), kept in
  its own store because an ETF ranked against its own holdings would be the
  index competing with itself;
- the membership panel ``wrds_crsp_{sp500|nasdaq100}_membership.zarr`` when
  ``--universe`` is given, which marks the days each security was a member.

The vendor is resolved through ``DataSourceRegistry.get("wrds")`` and its
``("us_equity", "1d", "crsp_daily")`` capability, so no vendor class is
named here.

``WRDS_USERNAME`` must be set in the environment. The password is never read
by this code; the PostgreSQL client library (libpq) takes it from
``~/.pgpass`` (mode 600), one line of the form
``wrds-pgdata.wharton.upenn.edu:9737:wrds:<username>:<password>``. No
argument accepts either value and nothing printed contains one. Every step
of a run shares one WRDS connection, so a run triggers at most one Duo
two-factor prompt.

Before any daily row is copied the run checks, in order: the arguments; that
the account may read every schema the run needs; the end date of the
annual CRSP release (an ``--end-date`` past it is clipped and the clip
printed, a ``--start-date`` past it is refused); the reference tables and
the roster resolved from them; and the row count, counted server-side per
calendar year and PERMNO batch and refused above the volume-guard ceilings
unless ``--force-volume`` is given. See ``docs/wrds_crsp.md``.

Usage::

    uv run python scripts/ingest_wrds_crsp.py --help
    export WRDS_USERNAME=<your-wrds-username>   # password lives in ~/.pgpass

    # CRSP's own point-in-time S&P 500, converted to a panel and a mask.
    uv run python scripts/ingest_wrds_crsp.py --universe crsp_sp500 \\
        --start-date 2000-01-01 --end-date 2025-12-31 --to-zarr

    # An explicit PERMNO roster (AAPL, META).
    uv run python scripts/ingest_wrds_crsp.py --permnos 14593,13407 \\
        --start-date 2020-01-01 --end-date 2024-12-31 --to-zarr

    # The QQQ benchmark, in its own store.
    uv run python scripts/ingest_wrds_crsp.py --qqq \\
        --start-date 1999-01-01 --end-date 2025-12-31 --to-zarr

    # An index roster plus its own benchmark ETF (SPY for crsp_sp500, QQQ
    # for comp_nasdaq100), each in its own store.
    uv run python scripts/ingest_wrds_crsp.py --universe crsp_sp500 \\
        --benchmark --start-date 2000-01-01 --end-date 2025-12-31 --to-zarr
"""

import argparse

from dataclasses import replace

from quantlab.registry import DataSourceRegistry, convert, run
from quantlab.base.config import (
    QQQ_PERMNO,
    SPY_PERMNO,
    ConstituentDatasetConfig,
    CrspDatasetConfig,
)
from quantlab.config import get_data_root
from quantlab.dataset.constituent import (
    CompustatNasdaq100ConstituentDataset,
    CrspSP500ConstituentDataset,
)
from quantlab.dataset.crsp import SECURITY_FILTER_PRESETS, CrspStockDataset
from quantlab.dataset.crsp.membership import CrspMembership
from quantlab.dataset.crsp.reference import CrspReference
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
)

#: The vendor descriptor, resolved from its registry token.
SOURCE = DataSourceRegistry.get("wrds")

#: The capability this script drives. One WRDS account serves several
#: products, so both the acquisition class and the config factory are read
#: off this key rather than off the descriptor's defaults.
CAPABILITY = ("us_equity", "1d", "crsp_daily")

#: The CRSP daily download class, looked up rather than imported. The script
#: uses its defaults and its window and path helpers; ``run()`` builds it.
ACQ = SOURCE.acquisition_cls_for(*CAPABILITY)

#: The point-in-time universes this vendor serves, taken from the membership
#: layer so a universe added there becomes selectable here without an edit.
UNIVERSES = CrspMembership.INDEXES

#: Map from universe id to the short name its stores carry. Store names are
#: short because people type them; universe ids name a vendor and an index.
UNIVERSE_SHORT_NAMES: dict[str, str] = {
    CrspMembership.SP500: "sp500",
    CrspMembership.NASDAQ100: "nasdaq100",
}

#: Map from universe id to the constituent dataset class that builds its mask.
CONSTITUENT_CLASSES: dict[str, type] = {
    CrspMembership.SP500: CrspSP500ConstituentDataset,
    CrspMembership.NASDAQ100: CompustatNasdaq100ConstituentDataset,
}

#: The equity panel's store name. An explicit ``--permnos`` roster is stored
#: as ``custom``: it is not an index, whatever its members happen to be.
STORE_TEMPLATE = "wrds_crsp_{name}_1d.zarr"

#: A benchmark ETF's own store, named after the ETF.
BENCHMARK_STORE_TEMPLATE = "wrds_crsp_{name}_1d.zarr"

#: The ETF ``--benchmark`` pulls for each universe, as ``(name, PERMNO)``.
UNIVERSE_BENCHMARKS: dict[str, tuple[str, str]] = {
    CrspMembership.SP500: ("spy", SPY_PERMNO),
    CrspMembership.NASDAQ100: ("qqq", QQQ_PERMNO),
}

#: The universe membership mask's store.
MEMBERSHIP_TEMPLATE = "wrds_crsp_{name}_membership.zarr"


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the argument parser for this script."""
    parser = argparse.ArgumentParser(
        description=(
            "Pull WRDS CRSP Stock v2 daily data and optionally convert it "
            "into Zarr panels. Requires WRDS_USERNAME in the environment and "
            "the password in ~/.pgpass (mode 600); neither is accepted as an "
            "argument."
        )
    )
    parser.add_argument(
        "--universe",
        type=str,
        choices=list(UNIVERSES),
        default=None,
        help=(
            "A point-in-time index universe. Every PERMNO that was a member "
            "at any time in [--start-date, --end-date] is pulled, including "
            "the ones that left the index, which avoids survivorship bias. "
            "'crsp_sp500' is CRSP's own dsp500list_v2 membership (from 1925); "
            "'comp_nasdaq100' is Compustat's index gvkeyx 000208 linked to "
            "PERMNOs through CCM (no data before 1995)."
        ),
    )
    parser.add_argument(
        "--permnos",
        type=str,
        default=None,
        help=(
            "Comma-separated PERMNOs, e.g. 14593,13407. CRSP raw files are "
            "keyed by PERMNO (the permanent security id), never by ticker. "
            "Tickers are looked up at conversion time, so a rename never "
            "affects a watermark. Combines with --universe and --qqq."
        ),
    )
    parser.add_argument(
        "--qqq",
        action="store_true",
        help=(
            f"Also pull the QQQ ETF (PERMNO {QQQ_PERMNO}) and, with --to-zarr, "
            f"write it to its own benchmark store. It is never a column of "
            f"the equity panel, because an ETF ranked against its own "
            f"holdings would be the index competing with itself."
        ),
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help=(
            f"Also pull the ETF that tracks --universe (SPY, PERMNO "
            f"{SPY_PERMNO}, for crsp_sp500; QQQ, PERMNO {QQQ_PERMNO}, for "
            f"comp_nasdaq100) and, with --to-zarr, write it to its own "
            f"benchmark store wrds_crsp_{{spy|qqq}}_1d.zarr. Requires "
            f"--universe."
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
            "Continue each PERMNO from its recorded watermark instead of "
            "downloading the whole window again."
        ),
    )
    parser.add_argument(
        "--refresh-reference",
        action="store_true",
        help=(
            "Download the _reference/ tables again even when the tables for "
            "this CRSP release are already complete. Without it, complete "
            "tables are reused and cost no query."
        ),
    )
    parser.add_argument(
        "--allow-unlinked-ndx",
        action="store_true",
        help=(
            "Continue when a Nasdaq-100 membership period has no CCM link to "
            "a PERMNO. Such a period has no security to pull, so by default "
            "it stops the run rather than silently shrinking the universe."
        ),
    )
    parser.add_argument(
        "--security-filter",
        type=str,
        choices=sorted(SECURITY_FILTER_PRESETS),
        default="equity_common",
        help=(
            "Which securities the equity panel holds (default equity_common: "
            "common stock including REITs, without ADRs, units, funds or "
            "ETFs). 'shrcd_10_11' keeps only US corporate common stock; "
            "'none' keeps every security type. Applied per date using "
            "dsf_v2's own type columns, and reported in the "
            ".crsp_filter_report.json sidecar."
        ),
    )
    add_to_zarr_arg(parser)
    add_chunk_args(parser, default="year")
    return parser


def _validate(parser: argparse.ArgumentParser, args) -> tuple[str, ...]:
    """Refuse bad arguments before any WRDS connection is opened.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser, used to report an error and exit with status 2.
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    tuple[str, ...]
        The explicit ``--permnos`` roster, possibly empty.
    """
    if args.benchmark and not args.universe:
        parser.error(
            "--benchmark pulls the ETF that tracks --universe, so it needs "
            "--universe; use --qqq or --permnos for an ETF on its own."
        )
    if not (args.universe or args.permnos or args.qqq):
        parser.error(
            "Nothing to pull: pass at least one of --universe, --permnos or "
            "--qqq. A CRSP pull with no roster would query the whole "
            "110-million-row daily table."
        )
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
    permnos: tuple[str, ...] = ()
    if args.permnos:
        tokens = [token.strip() for token in args.permnos.split(",") if token.strip()]
        bad = [token for token in tokens if not token.isdigit()]
        if bad:
            parser.error(
                f"--permnos {bad} are not PERMNOs. CRSP raw files are keyed by "
                f"PERMNO (a digit string, e.g. 14593 for AAPL), not by ticker; "
                f"resolve a ticker roster to PERMNOs first, or use --universe."
            )
        permnos = tuple(tokens)
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("--batch-size must be a positive integer.")
    return permnos


def _schemas_for(universe: str | None) -> tuple[str, ...]:
    """Return exactly the WRDS schemas a run for ``universe`` reads.

    Only these schemas are checked for access, so an S&P 500 pull never
    fails on Compustat, which most CRSP subscriptions cannot read and which
    that pull does not need.

    Parameters
    ----------
    universe : str or None
        The ``--universe`` value, or ``None`` for a PERMNO-only roster.

    Returns
    -------
    tuple of str
        The schema names.
    """
    from quantlab.acquisition.wrds.crsp import CrspQueries

    schemas = [CrspQueries.STOCK_SCHEMA]
    if universe == CrspMembership.SP500:
        schemas.append(CrspQueries.INDEX_SCHEMA)
    elif universe == CrspMembership.NASDAQ100:
        schemas.extend((CrspQueries.COMPUSTAT_SCHEMA, CrspQueries.CCM_SCHEMA))
    return tuple(schemas)


if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Must run before any path is derived from the data root.
    apply_data_dir(args)

    explicit_permnos = _validate(parser, args)

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
        # 1. Check access first, so a missing subscription stops the run
        #    before any data is copied or reference file written.
        try:
            CrspQueries.assert_entitled(session, _schemas_for(args.universe))
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

        # 3. Build the config with an empty roster; step 5 fills it in once
        #    the reference tables can say who is in the universe.
        acq_config = SOURCE.config_factory_for(*CAPABILITY)(
            symbols=(),
            start_date=window["start_date"],
            end_date=window["end_date"],
            kwargs=kwargs,
        )
        reference_dir = ACQ.reference_dir_for(acq_config)

        # 4. Pull only the reference tables this run needs, skipping them when
        #    this release's tables are already complete.
        try:
            manifest = CrspReferenceTables(session, reference_dir).pull(
                product_end=end_date,
                include_sp500=args.universe == CrspMembership.SP500,
                include_nasdaq100=args.universe == CrspMembership.NASDAQ100,
                refresh=args.refresh_reference,
            )
        except (RuntimeError, ValueError) as exc:
            parser.exit(1, f"{exc}\n")
        print(f"Reference tables at: {reference_dir}")
        for name, entry in sorted((manifest.get("tables") or {}).items()):
            rows = entry.get("rows") if isinstance(entry, dict) else entry
            print(f"  {name}: {rows:,} row(s)")

        # 5. Resolve the roster to PERMNOs; the download refuses tickers.
        roster: list[str] = []
        if args.universe:
            try:
                roster.extend(
                    CrspMembership(CrspReference(reference_dir)).permnos_in_range(
                        args.universe,
                        window["start_date"],
                        window["end_date"],
                        allow_unlinked=args.allow_unlinked_ndx,
                    )
                )
            except (RuntimeError, ValueError) as exc:
                parser.exit(1, f"{exc}\n")
        roster.extend(explicit_permnos)
        # Benchmark ETFs, by store name. Each gets its own store and never
        # becomes a column of the equity panel.
        benchmarks: dict[str, str] = {}
        if args.qqq:
            benchmarks["qqq"] = QQQ_PERMNO
        if args.benchmark:
            name, permno = UNIVERSE_BENCHMARKS[args.universe]
            benchmarks[name] = permno
        roster.extend(benchmarks.values())
        # Sort numerically, not as text ("14593" < "7000" as strings), so the
        # batch boundaries stay the same across runs of the same command.
        roster = [str(value) for value in sorted({int(p) for p in roster})]
        if not roster:
            parser.exit(1, "the roster resolved to no PERMNOs.\n")
        print(f"Roster ({len(roster)} PERMNO(s)): {', '.join(roster)}")
        acq_config = replace(acq_config, symbols=tuple(roster))

        # 6. Count rows per year and check the volume ceilings before any
        #    daily row is copied.
        try:
            counts = CrspVolumeProbe(
                session, batch_size=batch_size
            ).count_rows_by_year(roster, window["start_date"], window["end_date"])
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
        print(
            f"  note:              bytes/row {ACQ.DEFAULT_BYTES_PER_ROW} is an "
            f"assumed size for CRSP's 50 mostly short columns, not a measured "
            f"file size; measure it on a real run to replace it."
        )

        # 7. Download.
        print(
            f"Acquiring {len(roster)} PERMNO(s) from {SOURCE.display_name} "
            f"over {window['start_date']}..{window['end_date']} "
            f"(refresh={args.refresh})"
        )
        result = run(SOURCE, acq_config, refresh=args.refresh)
        print(
            f"{len(result.succeeded)} PERMNO(s) succeeded, "
            f"{len(result.failures)} failed"
        )
        print(f"Raw data written under: {acq_config.raw_data_dir_path}")

        if args.to_zarr:
            data_dir = get_data_root() / "data" / "us_equity" / "1d"
            short_name = UNIVERSE_SHORT_NAMES.get(args.universe, "custom")

            # Exclude the benchmark ETFs by roster, not only by the security
            # filter, because `--security-filter none` would otherwise let them
            # into the panel.
            equity_permnos = tuple(
                permno for permno in roster if permno not in benchmarks.values()
            )
            # With `--qqq` alone there is no equity to convert. Skip it, and
            # say so, rather than write a `custom` store holding only the ETF.
            if equity_permnos:
                ds_config = CrspDatasetConfig(
                    zarr_file_path=str(
                        data_dir / STORE_TEMPLATE.format(name=short_name)
                    ),
                    raw_data_dir_path=acq_config.raw_data_dir_path,
                    reference_dir=str(reference_dir),
                    start_date=window["start_date"],
                    end_date=window["end_date"],
                    permnos=equity_permnos,
                    security_filter=args.security_filter,
                    roster_universe=args.universe,
                )
                # Built once and reused for the raw-data check and the sidecar
                # path, because construction re-resolves the security filter.
                probe_dataset = CrspStockDataset(ds_config)
                refuse_conversion_without_raw_data(probe_dataset, result)
                print(
                    f"Converting {len(equity_permnos)} PERMNO(s) to the equity "
                    f"panel in {args.chunk} windows (resumable), filter "
                    f"{args.security_filter!r}"
                )
                print_conversion_result(
                    convert(
                        SOURCE,
                        ds_config,
                        data_type="crsp_daily",
                        granularity=args.chunk,
                        on_new_listing=args.on_new_listing,
                    )
                )
                print(
                    f"Security filter sidecar:   "
                    f"{probe_dataset.filter_report_path()}"
                )
            else:
                print(
                    f"Skipping the equity conversion: the roster holds no "
                    f"equity PERMNO -- all {len(roster)} of it is benchmark "
                    f"ETFs (PERMNO {', '.join(benchmarks.values())}), which "
                    f"get their own stores and are never equity columns. No "
                    f"'{STORE_TEMPLATE.format(name=short_name)}' is read or "
                    f"written. The benchmark stores and the universe "
                    f"membership panel still run below; they are independent "
                    f"outputs."
                )

            for name, permno in benchmarks.items():
                benchmark_config = CrspDatasetConfig.etf_benchmark(
                    permno=permno,
                    zarr_file_path=str(
                        data_dir / BENCHMARK_STORE_TEMPLATE.format(name=name)
                    ),
                    raw_data_dir_path=acq_config.raw_data_dir_path,
                    reference_dir=str(reference_dir),
                    start_date=window["start_date"],
                    end_date=window["end_date"],
                )
                print(
                    f"Converting the {name.upper()} benchmark (PERMNO "
                    f"{permno}) into its own store"
                )
                print_conversion_result(
                    convert(
                        SOURCE,
                        benchmark_config,
                        data_type="crsp_daily",
                        granularity=args.chunk,
                        on_new_listing=args.on_new_listing,
                    )
                )

            if args.universe:
                membership = CONSTITUENT_CLASSES[args.universe](
                    ConstituentDatasetConfig(
                        zarr_file_path=str(
                            data_dir / MEMBERSHIP_TEMPLATE.format(name=short_name)
                        ),
                        cache_dir=str(reference_dir),
                        start_date=window["start_date"],
                        end_date=window["end_date"],
                        kwargs={"allow_unlinked": args.allow_unlinked_ndx},
                    )
                )
                membership.from_raw_data().save()
                print(
                    f"Universe membership panel: "
                    f"{membership.config.zarr_file_path}"
                )
        else:
            print(
                "Skipping Zarr conversion (default). The raw files above are "
                "the run's output; pass --to-zarr to convert them."
            )
    finally:
        WrdsSession.close_shared()
