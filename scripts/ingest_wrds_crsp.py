"""Command-line ingest of CRSP daily stock data from WRDS for an index roster.

The script pulls ``crsp_a_stock.dsf_v2`` rows for a PERMNO roster (an index
universe, an explicit ``--permnos`` list, the QQQ ETF, or any combination)
and writes them as raw parquet shards under
``<data root>/downloads/us_equity/1d/wrds_crsp/wrds/``, one row per
``(permno, dlycaldt)`` exactly as CRSP serves them. Beside that raw root it
fills ``_reference/`` with the small CRSP, Compustat and CCM tables that map a
PERMNO to its period-correct ticker and answer index membership; they are
pulled before the roster is resolved because the roster is read from them.

With ``--to-zarr`` the raw tier is also converted into up to three Zarr
stores: the equity panel ``wrds_crsp_{sp500|nasdaq100|custom}_1d.zarr`` with
its JSON sidecars (adjustment anchor, security-filter report, symbology
report); the QQQ benchmark ``wrds_crsp_qqq_1d.zarr`` when ``--qqq`` is given,
kept in its own store because an ETF ranked against the constituents it holds
would be the index competing with itself; and the membership panel
``wrds_crsp_{sp500|nasdaq100}_membership.zarr`` when ``--universe`` is given.

The vendor is resolved through ``DataSourceRegistry.get("wrds")`` and its
``("us_equity", "1d", "crsp_daily")`` capability, so no vendor class is named
here, and every config is constructed directly.

Credentials: ``WRDS_USERNAME`` must be set in the environment. The password
is never read by this code; libpq takes it from ``~/.pgpass`` (mode 600), one
line of the form
``wrds-pgdata.wharton.upenn.edu:9737:wrds:<username>:<password>``. No
argument accepts either value and nothing printed contains one. Every step
of a run shares one WRDS connection, opened through ``WrdsSession.shared()``
and closed in a ``finally``, so a run costs at most one Duo push.

Before any daily row is copied the run checks, in this order: the arguments;
the account's entitlement to every schema the run reads; the CRSP annual
product end (an ``--end-date`` past it is clipped and the clip printed, a
``--start-date`` past it is refused); the reference tables and the roster
resolved from them; and the row volume, counted with ``count(*)`` per
calendar year and PERMNO batch and refused above the ``SqlVolumeGuard``
ceilings unless ``--force-volume`` is given. See ``docs/wrds_crsp.md``.

Usage:
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
"""

import argparse

from dataclasses import replace

from quantlab.registry import DataSourceRegistry, convert, run
from quantlab.base.config import (
    QQQ_PERMNO,
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

#: The CRSP daily acquisition class, resolved rather than named. Used for its
#: class-level defaults and its window and path helpers; ``run()`` constructs
#: it.
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

#: The QQQ benchmark's own store.
QQQ_STORE = "wrds_crsp_qqq_1d.zarr"

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
            "A CRSP-vendor point-in-time universe, resolved by interval "
            "overlap over [--start-date, --end-date]: every PERMNO that was a "
            "member at any point in the window, which keeps the securities "
            "that left the index and removes survivorship bias. "
            "'crsp_sp500' is CRSP's own dsp500list_v2 membership (from 1925); "
            "'comp_nasdaq100' is Compustat's gvkeyx 000208 linked through CCM "
            "(left-censored at 1995)."
        ),
    )
    parser.add_argument(
        "--permnos",
        type=str,
        default=None,
        help=(
            "Comma-separated PERMNOs, e.g. 14593,13407. CRSP's raw tier is "
            "keyed by PERMNO (the stable security id), never by ticker: a "
            "ticker is derived at conversion time, so a rename never touches "
            "a watermark. Combines with --universe and --qqq."
        ),
    )
    parser.add_argument(
        "--qqq",
        action="store_true",
        help=(
            f"Also pull the QQQ ETF (PERMNO {QQQ_PERMNO}) and, with --to-zarr, "
            f"write it to its own benchmark store. It is never a column of "
            f"the equity panel: an ETF ranked against the constituents it "
            f"holds is the index competing with itself in one cross section."
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
            f"PERMNOs per COPY and per count(*) (default "
            f"{ACQ.DEFAULT_BATCH_SIZE})."
        ),
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help=(
            "Incrementally refresh from each PERMNO's recorded watermark "
            "instead of a full backfill of the window."
        ),
    )
    parser.add_argument(
        "--refresh-reference",
        action="store_true",
        help=(
            "Re-pull the _reference/ tables even when this CRSP vintage's "
            "tier is already complete. Without it a complete tier is reused "
            "and costs zero round trips."
        ),
    )
    parser.add_argument(
        "--allow-unlinked-ndx",
        action="store_true",
        help=(
            "Proceed when a Nasdaq-100 membership spell has no CCM link. A "
            "spell with no PERMNO has no security at all, so by default it "
            "stops the run rather than silently shrinking the universe."
        ),
    )
    parser.add_argument(
        "--security-filter",
        type=str,
        choices=sorted(SECURITY_FILTER_PRESETS),
        default="equity_common",
        help=(
            "Which securities the equity panel holds (default equity_common: "
            "common stock, REITs included, ADRs/units/funds/ETFs dropped). "
            "'shrcd_10_11' is the narrower US-corporate-common reading; "
            "'none' keeps every security type. Evaluated per date against "
            "dsf_v2's own type columns, and reported in the "
            ".crsp_filter_report.json sidecar."
        ),
    )
    add_to_zarr_arg(parser)
    add_chunk_args(parser, default="year")
    return parser


def _validate(parser: argparse.ArgumentParser, args) -> tuple[str, ...]:
    """Refuse bad arguments before any WRDS session exists.

    Returns:
        The explicit ``--permnos`` roster as a tuple, possibly empty.
    """
    if not (args.universe or args.permnos or args.qqq):
        parser.error(
            "Nothing to pull: pass at least one of --universe, --permnos or "
            "--qqq. A CRSP pull with no roster would be a query over the "
            "whole 110-million-row daily table."
        )
    if not args.start_date or not args.end_date:
        parser.error(
            "--start-date and --end-date are both required: the window is "
            "counted and priced before any data is pulled, and it is checked "
            "against the CRSP annual product end."
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
                f"--permnos {bad} are not PERMNOs. CRSP's raw tier is keyed by "
                f"PERMNO (a digit string, e.g. 14593 for AAPL), not by ticker; "
                f"resolve a ticker roster to PERMNOs first, or use --universe."
            )
        permnos = tuple(tokens)
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("--batch-size must be a positive integer.")
    return permnos


def _schemas_for(universe: str | None) -> tuple[str, ...]:
    """Return exactly the WRDS schemas a run for ``universe`` reads.

    An S&P-only pull never asks whether the account can read Compustat,
    which most CRSP subscriptions cannot and which that pull does not need.
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

    # The data root must be applied before any path is derived from it.
    apply_data_dir(args)

    explicit_permnos = _validate(parser, args)

    batch_size = args.batch_size or ACQ.DEFAULT_BATCH_SIZE
    kwargs = {"batch_size": batch_size, "clip_to_product_end": True}

    # Imported at run time rather than at module scope so the session class is
    # looked up when the run starts, which lets an offline double replace it.
    from quantlab.acquisition._support.sql_volume import SqlVolumeGuard
    from quantlab.acquisition.wrds.crsp import CrspQueries, CrspVolumeProbe
    from quantlab.acquisition.wrds.crsp_reference import CrspReferenceTables
    from quantlab.acquisition.wrds.taq import WrdsSession

    try:
        session = WrdsSession.shared()
    except RuntimeError as exc:
        parser.exit(1, f"{exc}\n")

    try:
        # 1. Entitlement first: an unsubscribed schema stops the run with no
        #    COPY issued and no reference file written.
        try:
            CrspQueries.assert_entitled(session, _schemas_for(args.universe))
        except (RuntimeError, ValueError) as exc:
            parser.exit(1, f"{exc}\n")

        # 2. The annual product end, probed once; a clip is always printed.
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

        # 3. The acquisition config, built once with an empty roster; the
        #    roster is filled in at step 5, after the reference tables can
        #    answer who is in the universe.
        acq_config = SOURCE.config_factory_for(*CAPABILITY)(
            symbols=(),
            start_date=window["start_date"],
            end_date=window["end_date"],
            kwargs=kwargs,
        )
        reference_dir = ACQ.reference_dir_for(acq_config)

        # 4. The reference tier, on the same session, scoped to what this run
        #    needs and skipped entirely when this vintage's tier is complete.
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

        # 5. The roster, resolved to PERMNOs before it reaches the config: the
        #    acquisition refuses a ticker roster outright.
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
        if args.qqq:
            roster.append(QQQ_PERMNO)
        # Numerically sorted and de-duplicated: PERMNOs are integers rendered
        # as strings, so a text sort would put "14593" before "7000" and the
        # batch boundaries would move between two runs of the same command.
        roster = [str(value) for value in sorted({int(p) for p in roster})]
        if not roster:
            parser.exit(1, "the roster resolved to no PERMNOs.\n")
        print(f"Roster ({len(roster)} PERMNO(s)): {', '.join(roster)}")
        acq_config = replace(acq_config, symbols=tuple(roster))

        # 6. count(*) per year page, then the guard, both before the first
        #    COPY of any daily row.
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
            f"ASSUMPTION for CRSP's 50 mostly-short columns, not a measured "
            f"shard size; a live smoke run should replace it."
        )

        # 7. The pull.
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
            catalog_path = str(get_data_root() / "data" / "catalog")
            short_name = UNIVERSE_SHORT_NAMES.get(args.universe, "custom")

            # QQQ is excluded from the equity panel by roster, not only by the
            # security filter: `--security-filter none` is a legitimate choice
            # and must not silently pull the benchmark into the panel.
            equity_permnos = tuple(
                permno
                for permno in roster
                if not (args.qqq and permno == QQQ_PERMNO)
            )
            # An empty equity roster (`--qqq` alone) skips the equity
            # conversion rather than writing a `custom` store that would hold
            # nothing but the benchmark. The skip is printed so that it cannot
            # be mistaken for a conversion that quietly wrote nothing.
            if equity_permnos:
                ds_config = CrspDatasetConfig(
                    zarr_file_path=str(
                        data_dir / STORE_TEMPLATE.format(name=short_name)
                    ),
                    raw_data_dir_path=acq_config.raw_data_dir_path,
                    catalog_path=catalog_path,
                    reference_dir=str(reference_dir),
                    start_date=window["start_date"],
                    end_date=window["end_date"],
                    permnos=equity_permnos,
                    security_filter=args.security_filter,
                    roster_universe=args.universe,
                )
                # One dataset serves both the raw-data probe and the sidecar
                # path; constructing it re-resolves the security filter, so it
                # is built once.
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
                    f"equity PERMNO -- all {len(roster)} of it is the QQQ "
                    f"benchmark (PERMNO {QQQ_PERMNO}), which gets its own "
                    f"store and is never an equity column (D-15). No "
                    f"'{STORE_TEMPLATE.format(name=short_name)}' is read or "
                    f"written. The QQQ store and the universe membership panel "
                    f"still run below; they are independent outputs."
                )

            if args.qqq:
                qqq_config = CrspDatasetConfig.qqq_benchmark(
                    zarr_file_path=str(data_dir / QQQ_STORE),
                    raw_data_dir_path=acq_config.raw_data_dir_path,
                    catalog_path=catalog_path,
                    reference_dir=str(reference_dir),
                    start_date=window["start_date"],
                    end_date=window["end_date"],
                )
                print("Converting the QQQ benchmark into its own store (D-15)")
                print_conversion_result(
                    convert(
                        SOURCE,
                        qqq_config,
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
                "Skipping Zarr conversion (default). The raw shards above are "
                "the deliverable; pass --to-zarr to convert them."
            )
    finally:
        WrdsSession.close_shared()
