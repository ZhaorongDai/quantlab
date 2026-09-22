"""Pull WRDS CRSP Stock v2 daily data for a PERMNO roster and, optionally,
convert it into `[timestamp, symbol]` Zarr panels.

What it does
------------
Daily rows are read from `crsp_a_stock.dsf_v2` with one PostgreSQL `COPY` per
(calendar year, PERMNO batch). The raw tier lands under
`<data root>/downloads/us_equity/1d/wrds_crsp/wrds/month=YYYY-MM/` with one row
per `(permno, dlycaldt)`, exactly as CRSP serves it -- no derived price, no
adjusted series, no filter.

Beside that raw root the run also fills `_reference/`: the small CRSP /
Compustat / CCM tables (`stksecurityinfohist`, `stkdelists`,
`stkdistributions`, and per universe `dsp500list_v2` or
`idxcst_his` + `ccmxpf_lnkhist`). They are what turns a PERMNO into a
period-correct ticker, so they are pulled BEFORE the roster is resolved.

With `--to-zarr` the raw tier is converted into up to three stores:

- the equity panel `wrds_crsp_{sp500|nasdaq100|custom}_1d.zarr` (with its
  three JSON sidecars: the adjustment anchor, the security-filter report and
  the symbology report);
- the QQQ benchmark `wrds_crsp_qqq_1d.zarr` when `--qqq` was given -- its OWN
  store, because an ETF ranked against the constituents it holds is the index
  competing with itself (D-15);
- the universe's membership panel `wrds_crsp_{sp500|nasdaq100}_membership.zarr`
  when `--universe` was given.

This script names no vendor class: it resolves its source with
`DataSourceRegistry.get("wrds")` and reads both the acquisition class and the
config factory off the `("us_equity", "1d", "crsp_daily")` capability. It calls
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
The entitlement probe, the product-end probe, the reference pull, the volume
probe, the daily pull and the conversion therefore share ONE session
(`WrdsSession.shared()`), closed in a `finally` when the run ends. The pull
runs on that one connection; there is no worker-count flag to raise.

An ANNUAL product
-----------------
`crsp_a_stock` is the annual-update product, so its last day is a hard edge
rather than "data not in yet". An `--end-date` past it is CLIPPED and the clip
is printed verbatim; a `--start-date` past it is refused, because clipping
cannot rescue a window that is entirely past the edge. A new CRSP vintage is a
new raw tier: the acquisition stamps the vintage beside the raw root and
refuses to mix two releases in one.

Volume
------
Rows are counted with `count(*)` per (calendar year, PERMNO batch) BEFORE any
data is pulled, and `SqlVolumeGuard` refuses a pull over the 20 GiB raw-byte
ceiling (or the 700M row ceiling). The estimate's buckets are YEARS here, not
trading days, because a CRSP page is a calendar year -- so a refusal names a
boundary a re-run can actually be given. The 150 B/row figure is an ASSUMPTION
until a live smoke measures real shard bytes/row. `--force-volume` skips the
refusal, never the arithmetic.

Order of checks
---------------
1. Arguments (roster, PERMNO shape, both dates), before any connection.
2. Entitlement: every schema this run needs, probed first, so an unsubscribed
   product stops the run with zero COPY calls.
3. The product-end probe and the window clip.
4. The reference tables, then the point-in-time roster resolved from them.
5. Volume: `count(*)` per year page, then `SqlVolumeGuard`.
6. The pull (`registry.run`), then the optional conversions.

Usage:
    export WRDS_USERNAME=<your-wrds-username>   # password lives in ~/.pgpass

    # CRSP's own point-in-time S&P 500, converted to a panel and a mask.
    uv run python scripts/ingest_wrds_crsp.py --universe crsp_sp500 \
        --start-date 2000-01-01 --end-date 2025-12-31 --to-zarr

    # An explicit PERMNO roster (AAPL, META).
    uv run python scripts/ingest_wrds_crsp.py --permnos 14593,13407 \
        --start-date 2020-01-01 --end-date 2024-12-31 --to-zarr

    # The QQQ benchmark, in its own store.
    uv run python scripts/ingest_wrds_crsp.py --qqq \
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

#: The one place this script's vendor is named, as a registry token.
SOURCE = DataSourceRegistry.get("wrds")

#: The capability this script drives. One WRDS account serves several products
#: (03.10 D-12), so BOTH the acquisition class and the config factory are read
#: off this key rather than off the descriptor's TAQ defaults.
CAPABILITY = ("us_equity", "1d", "crsp_daily")

#: The CRSP daily acquisition class, resolved -- never named. Read for its
#: class-level knobs (`DEFAULT_BATCH_SIZE`) and its two window/path
#: classmethods; never constructed here, because `registry.run()` owns that.
ACQ = SOURCE.acquisition_cls_for(*CAPABILITY)

#: The point-in-time universes this vendor serves, derived from the membership
#: layer so adding one there makes it selectable here with no edit.
UNIVERSES = CrspMembership.INDEXES

#: `{universe: the short name its stores carry}`. The store names are short
#: because they are typed by humans; the universe ids are long because they
#: name a VENDOR and an index (`crsp_sp500` is not the Wikipedia S&P panel).
UNIVERSE_SHORT_NAMES: dict[str, str] = {
    CrspMembership.SP500: "sp500",
    CrspMembership.NASDAQ100: "nasdaq100",
}

#: `{universe: the constituent dataset class that builds its mask}`.
CONSTITUENT_CLASSES: dict[str, type] = {
    CrspMembership.SP500: CrspSP500ConstituentDataset,
    CrspMembership.NASDAQ100: CompustatNasdaq100ConstituentDataset,
}

#: The equity panel's store name. `custom` is the name an explicit `--permnos`
#: roster gets: it is not any index, and calling it `sp500` because the caller
#: happened to list S&P members would be a lie the store then carries forever.
STORE_TEMPLATE = "wrds_crsp_{name}_1d.zarr"

#: The QQQ benchmark's own store (D-15).
QQQ_STORE = "wrds_crsp_qqq_1d.zarr"

#: The universe membership mask's store.
MEMBERSHIP_TEMPLATE = "wrds_crsp_{name}_membership.zarr"


def _build_arg_parser() -> argparse.ArgumentParser:
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
            "A CRSP-VENDOR point-in-time universe, resolved by interval "
            "OVERLAP over [--start-date, --end-date]: every PERMNO that was a "
            "member at any point in the window, which is what keeps the "
            "securities that LEFT the index and removes survivorship bias. "
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
            f"write it to its OWN benchmark store. It is never a column of "
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
            "STOPS the run rather than silently shrinking the universe."
        ),
    )
    parser.add_argument(
        "--security-filter",
        type=str,
        choices=sorted(SECURITY_FILTER_PRESETS),
        default="equity_common",
        help=(
            "WHICH securities the equity panel holds (default equity_common: "
            "common stock, REITs included, ADRs/units/funds/ETFs dropped). "
            "'shrcd_10_11' is the narrower US-corporate-common reading; "
            "'none' keeps every security type. Evaluated PER DATE against "
            "dsf_v2's own type columns, and reported in the "
            ".crsp_filter_report.json sidecar."
        ),
    )
    add_to_zarr_arg(parser)
    add_chunk_args(parser, default="year")
    return parser


def _validate(parser: argparse.ArgumentParser, args) -> tuple[str, ...]:
    """Every argument refusal, BEFORE any WRDS session exists.

    Returns the explicit `--permnos` roster as a tuple (possibly empty).
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
    """Exactly the schemas this run reads, so an S&P-only pull never asks
    whether this account can read Compustat -- a question whose answer is "no"
    for most CRSP subscriptions and which nothing in that pull needs."""
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

    # Before any path is derived from the data root (DDIR-04).
    apply_data_dir(args)

    explicit_permnos = _validate(parser, args)

    batch_size = args.batch_size or ACQ.DEFAULT_BATCH_SIZE
    kwargs = {"batch_size": batch_size, "clip_to_product_end": True}

    # Imported here so the module attributes are read at RUN time: the test
    # suite patches `wrds.taq.WrdsSession` with an offline double, and a
    # module-scope binding would capture the real class at import time
    # (RESEARCH Pattern 1).
    from quantlab.acquisition._support.sql_volume import SqlVolumeGuard
    from quantlab.acquisition.wrds.crsp import CrspQueries, CrspVolumeProbe
    from quantlab.acquisition.wrds.crsp_reference import CrspReferenceTables
    from quantlab.acquisition.wrds.taq import WrdsSession

    try:
        session = WrdsSession.shared()
    except RuntimeError as exc:
        parser.exit(1, f"{exc}\n")

    try:
        # 1. Entitlement FIRST: an unsubscribed schema stops the run with zero
        #    COPY calls and zero reference files, rather than failing later in
        #    a way that leaves a half-filled tier behind (D-03, D-21).
        try:
            CrspQueries.assert_entitled(session, _schemas_for(args.universe))
        except (RuntimeError, ValueError) as exc:
            parser.exit(1, f"{exc}\n")

        # 2. The annual product edge. Probed once; the clip is PRINTED, never
        #    silent (T-03.10-37).
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

        # 3. THE one config-factory call. The roster is filled in at step 5,
        #    once the reference tables can answer who is in the universe.
        acq_config = SOURCE.config_factory_for(*CAPABILITY)(
            symbols=(),
            start_date=window["start_date"],
            end_date=window["end_date"],
            kwargs=kwargs,
        )
        reference_dir = ACQ.reference_dir_for(acq_config)

        # 4. The reference tier, on the SAME session. Scoped to what this run
        #    needs, and skipped entirely when this vintage's tier is complete.
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

        # 5. The roster, resolved to PERMNOs BEFORE it reaches the config.
        #    `_assert_permnos` refuses the whole run on a ticker (03.10-03), so
        #    a universe MUST arrive here already resolved.
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

        # 6. count(*) per year page, then the guard -- BOTH before the first
        #    COPY of any daily row (D-03, T-03.10-35).
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

            # The equity panel. QQQ is excluded by ROSTER and not only by the
            # security filter: `--security-filter none` is a legitimate choice,
            # and it must not silently pull the benchmark into the panel.
            equity_permnos = tuple(
                permno
                for permno in roster
                if not (args.qqq and permno == QQQ_PERMNO)
            )
            # GAP-D. An EMPTY equity roster is not a conversion (WR-01): `--qqq`
            # with no --permnos and no --universe leaves nothing here but the
            # benchmark, and running the equity path anyway named its store
            # `custom` (the fallback), collided with whatever earlier run had
            # written that name, and died on the anchor gate BEFORE the QQQ
            # block -- which is why `--qqq --to-zarr` has never produced a
            # benchmark store. The skip is PRINTED because an empty roster is an
            # outcome the operator has to be able to see: a silent one is
            # indistinguishable from a conversion that quietly wrote nothing.
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
                # `ds_config` directly: the former `replace(ds_config,
                # symbols=None)` was a no-op, since `symbols` is already None on
                # this config, and each construction re-runs the config setter
                # and re-resolves the security filter (IN-06).
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
                    f"Adjustment anchor sidecar: "
                    f"{CrspStockDataset.adjustment_sidecar_path(ds_config)}"
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
