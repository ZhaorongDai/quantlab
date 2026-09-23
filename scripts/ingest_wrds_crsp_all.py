"""Pull the WHOLE CRSP US equity market -- and, optionally, the index
membership panels beside it -- into the raw tier and the Zarr stores.

The sibling of `scripts/ingest_wrds_crsp.py`. That shell's roster is an INDEX
(`--universe crsp_sp500`) or an explicit `--permnos` list; this one's roster is
the MARKET: every security `crsp_a_stock.stksecurityinfohist` carries whose
type matches `--security-filter`, resolved point-in-time over the window.

**The roster costs nothing extra to resolve.** `stksecurityinfohist` is already
pulled unconditionally as part of the reference tier (symbology needs it,
D-04), and it carries every column the security-filter presets name. So
`CrspMarketRoster` is a read of parquet already on disk: no new query, no new
schema entitlement, no extra round trip. Measured on this project's own
reference tier (product end 2025-12-31):

    191,048 spell(s) -> 40,518 PERMNO
    equity_common    -> 30,484 PERMNO all-time
      2024 window          5,465 PERMNO    (the S&P 500 window is 520)
      2015-2025            8,821 PERMNO
      1999-2025           16,814 PERMNO

**Scale is the whole difference from the sibling shell, so the window is not
optional and the default granularity is a YEAR.** A 2024 pull is roughly
5,465 x 252 ~ 1.4M daily rows against the S&P 500 window's ~139k. The volume
guard is not bypassed for this: `count(*)` runs server-side per year bucket
BEFORE the first COPY, so a window that does not fit is refused with the real
number rather than discovered halfway through. Run a year at a time; that is
what `--chunk year` (the default) and the guard are both shaped for.

**Incremental refresh.** `--refresh` resumes each PERMNO from its recorded
watermark instead of re-pulling the window. On the Zarr side the conversion is
chunked and resumable, and appending forward is bit-for-bit identical to a
full rebuild over the same range.

**Read `--on-new-listing` before the second run.** On a whole-market store the
roster GROWS between refreshes as a matter of course -- that is what an IPO is
-- and the default `refuse` halts on exactly that. The three choices are not
interchangeable:

  refuse   (default, shared with every other shell) halts and leaves the store
           untouched. Safe, and unusable for an unattended whole-market
           refresh.
  widen    keeps the store and widens its symbol axis in place. For a
           genuinely NEW listing this is not merely fast, it is CORRECT: the
           new column's historical block is NaN because the security did not
           exist then.
  rebuild  re-densifies every window from raw onto the new symbol union. What
           you want when a PERMNO that ALREADY had history entered the roster
           late -- a raw backfill, or a changed `--security-filter`. `widen`
           would leave that security's real history NaN forever.

  The distinction is not something the flag can decide for you, because
  "appeared in the roster" looks identical in both cases from the store's side.

**Known limitation this shell inherits (D-10).** Moving `--start-date` EARLIER
between runs, or backfilling the raw tier with earlier history, silently
rewrites adjusted values: the adjustment anchor is each PERMNO's first usable
row, so giving it an earlier first row moves it. Nothing in the code checks
this (`ChunkLedger.assert_consistent` does not look at `start_date`). On a
whole-market roster there are ~8x more securities for it to happen to than on
the S&P 500. Extending `--end-date` FORWARD is safe and is the supported
incremental path.

**The index membership panels.** `--with-index-membership` additionally pulls
the S&P 500 and Nasdaq-100 reference tables and writes their point-in-time
constituent panels, so one run produces the market data and the index
membership a cross-sectional strategy selects within. It is opt-in because it
widens the WRDS entitlement check to Compustat and CCM -- schemas most CRSP
subscriptions do not carry -- and a whole-market pull must not fail on a
question it never needed to ask.

Examples:

    # One year of the whole market, raw tier only (no conversion).
    uv run python scripts/ingest_wrds_crsp_all.py \\
        --start-date 2024-01-01 --end-date 2024-12-31

    # Same year, converted, with the in-listing mask.
    uv run python scripts/ingest_wrds_crsp_all.py \\
        --start-date 2024-01-01 --end-date 2024-12-31 --to-zarr

    # Extend forward incrementally, accepting new listings.
    uv run python scripts/ingest_wrds_crsp_all.py \\
        --start-date 2024-01-01 --end-date 2025-12-31 \\
        --refresh --to-zarr --on-new-listing widen

    # The market plus both index membership panels.
    uv run python scripts/ingest_wrds_crsp_all.py \\
        --start-date 2024-01-01 --end-date 2024-12-31 \\
        --to-zarr --with-index-membership
"""

import argparse

from dataclasses import replace

from quantlab.registry import DataSourceRegistry, convert, run
from quantlab.base.config import (
    ConstituentDatasetConfig,
    CrspDatasetConfig,
)
from quantlab.config import get_data_root
from quantlab.dataset.constituent import (
    CompustatNasdaq100ConstituentDataset,
    CrspMarketConstituentDataset,
    CrspSP500ConstituentDataset,
)
from quantlab.dataset.crsp import SECURITY_FILTER_PRESETS, CrspStockDataset
from quantlab.dataset.crsp.market import MARKET, CrspMarketRoster
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

#: The capability this script drives -- the same one the index shell drives.
#: One WRDS account serves several products (03.10 D-12), so BOTH the
#: acquisition class and the config factory are read off this key.
CAPABILITY = ("us_equity", "1d", "crsp_daily")

#: The CRSP daily acquisition class, resolved -- never named.
ACQ = SOURCE.acquisition_cls_for(*CAPABILITY)

#: The equity panel's store name. `all` rather than `us_all`: the latter is
#: already taken on disk by a Tiingo-era store, and two stores under one name
#: is the collision the sibling shell's `custom` fallback exists to avoid.
STORE = "wrds_crsp_all_1d.zarr"

#: The whole-market in-listing mask's store.
MEMBERSHIP_STORE = "wrds_crsp_all_membership.zarr"

#: `{universe: (short name, constituent dataset class)}` for the OPTIONAL
#: index membership panels. Derived from the membership layer so adding an
#: index there makes it appear here with no edit.
INDEX_PANELS: dict[str, tuple[str, type]] = {
    CrspMembership.SP500: ("sp500", CrspSP500ConstituentDataset),
    CrspMembership.NASDAQ100: ("nasdaq100", CompustatNasdaq100ConstituentDataset),
}

#: The index membership panel's store name, shared with the sibling shell so
#: both write the SAME store for the same universe rather than two spellings
#: of one panel.
MEMBERSHIP_TEMPLATE = "wrds_crsp_{name}_membership.zarr"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Pull the WHOLE CRSP US equity market for a window and optionally "
            "convert it into Zarr. Requires WRDS_USERNAME in the environment "
            "and the password in ~/.pgpass (mode 600); neither is accepted as "
            "an argument."
        )
    )
    add_window_args(parser)
    add_volume_guard_args(parser)
    add_data_dir_arg(parser)
    parser.add_argument(
        "--security-filter",
        type=str,
        choices=sorted(SECURITY_FILTER_PRESETS),
        default="equity_common",
        help=(
            "WHICH securities the market roster holds (default "
            "equity_common: common stock, REITs included, ADRs/units/funds/"
            "ETFs dropped). Unlike the index shell this filter decides the "
            "ROSTER: with no index to bound it, 'none' means all 40,518 "
            "PERMNOs CRSP carries. The PER-DAY verdict is carried by the "
            "in-listing mask this run writes beside the panel, NOT by removing "
            "rows from the panel -- naming a roster in config.permnos exempts "
            "those PERMNOs from the conversion-time type filter (GAP-C), so "
            "the panel holds every row of every security in the roster and the "
            "mask says which (day, symbol) cells the universe actually "
            "contains."
        ),
    )
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
            "instead of a full backfill of the window. The supported "
            "incremental direction is FORWARD; see the module docstring on "
            "D-10 for what moving --start-date earlier silently rewrites."
        ),
    )
    parser.add_argument(
        "--refresh-reference",
        action="store_true",
        help=(
            "Re-pull the _reference/ tables even when this CRSP vintage's "
            "tier is already complete. Without it a complete tier is reused "
            "and costs zero round trips -- which is also what makes the "
            "market roster free to resolve."
        ),
    )
    parser.add_argument(
        "--with-index-membership",
        action="store_true",
        help=(
            "Also pull the S&P 500 and Nasdaq-100 reference tables and write "
            "their point-in-time constituent panels. Opt-in because it widens "
            "the entitlement check to the Compustat and CCM schemas, which "
            "most CRSP subscriptions do not carry -- a whole-market pull must "
            "not fail on a question it never needed to ask."
        ),
    )
    parser.add_argument(
        "--allow-unlinked-ndx",
        action="store_true",
        help=(
            "Only with --with-index-membership: proceed when a Nasdaq-100 "
            "membership spell has no CCM link. A spell with no PERMNO has no "
            "security at all, so by default it STOPS the run rather than "
            "silently shrinking that panel."
        ),
    )
    add_to_zarr_arg(parser)
    add_chunk_args(parser, default="year")
    return parser


def _validate(parser: argparse.ArgumentParser, args) -> None:
    """Every argument refusal, BEFORE any WRDS session exists."""
    if not args.start_date or not args.end_date:
        parser.error(
            "--start-date and --end-date are both required: the window is "
            "counted and priced before any data is pulled, and it is checked "
            "against the CRSP annual product end. On a whole-market roster an "
            "unbounded window is a query over the whole 110-million-row daily "
            "table."
        )
    if args.rows_per_symbol_day is not None:
        parser.error(
            "--rows-per-symbol-day does not apply to CRSP: rows are counted "
            "server-side with count(*) per calendar year and PERMNO batch "
            "before the pull."
        )
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("--batch-size must be a positive integer.")
    if args.allow_unlinked_ndx and not args.with_index_membership:
        parser.error(
            "--allow-unlinked-ndx only means anything with "
            "--with-index-membership: it relaxes the Nasdaq-100 link refusal, "
            "and without that flag no Nasdaq-100 panel is built at all. "
            "Accepting it silently would let a run believe it had relaxed a "
            "gate that never ran."
        )


def _schemas_for(with_index_membership: bool) -> tuple[str, ...]:
    """Exactly the schemas this run reads.

    A whole-market pull needs the stock schema and nothing else, so it never
    asks whether this account can read Compustat -- a question whose answer is
    "no" for most CRSP subscriptions and which nothing in that pull needs.
    """
    from quantlab.acquisition.wrds.crsp import CrspQueries

    schemas = [CrspQueries.STOCK_SCHEMA]
    if with_index_membership:
        schemas.extend(
            (
                CrspQueries.INDEX_SCHEMA,
                CrspQueries.COMPUSTAT_SCHEMA,
                CrspQueries.CCM_SCHEMA,
            )
        )
    return tuple(schemas)


if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Before any path is derived from the data root (DDIR-04).
    apply_data_dir(args)

    _validate(parser, args)

    batch_size = args.batch_size or ACQ.DEFAULT_BATCH_SIZE
    kwargs = {"batch_size": batch_size, "clip_to_product_end": True}

    # Imported here so the module attributes are read at RUN time: the test
    # suite patches the session with an offline double, and a module-scope
    # binding would capture the real class at import time (RESEARCH Pattern 1).
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
        #    COPY calls and zero reference files (D-03, D-21).
        try:
            CrspQueries.assert_entitled(
                session, _schemas_for(args.with_index_membership)
            )
        except (RuntimeError, ValueError) as exc:
            parser.exit(1, f"{exc}\n")

        # 2. The annual product edge. Probed once; the clip is PRINTED.
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

        # 3. THE one config-factory call. The roster is filled in at step 5.
        acq_config = SOURCE.config_factory_for(*CAPABILITY)(
            symbols=(),
            start_date=window["start_date"],
            end_date=window["end_date"],
            kwargs=kwargs,
        )
        reference_dir = ACQ.reference_dir_for(acq_config)

        # 4. The reference tier, on the SAME session. `stksecurityinfohist` is
        #    in the unconditional set, so the market roster is answerable after
        #    this step whether or not the index tables were asked for.
        try:
            manifest = CrspReferenceTables(session, reference_dir).pull(
                product_end=end_date,
                include_sp500=args.with_index_membership,
                include_nasdaq100=args.with_index_membership,
                refresh=args.refresh_reference,
            )
        except (RuntimeError, ValueError) as exc:
            parser.exit(1, f"{exc}\n")
        print(f"Reference tables at: {reference_dir}")
        for name, entry in sorted((manifest.get("tables") or {}).items()):
            rows = entry.get("rows") if isinstance(entry, dict) else entry
            print(f"  {name}: {rows:,} row(s)")

        # 5. The roster: every security of the requested type whose listing
        #    OVERLAPS the window. Overlap and not containment -- a security
        #    delisted inside the window must stay, and on a whole-market
        #    roster those are the majority of the history.
        roster_source = CrspMarketRoster(CrspReference(reference_dir))
        try:
            roster = roster_source.permnos_in_range(
                window["start_date"],
                window["end_date"],
                security_filter=args.security_filter,
            )
        except (RuntimeError, ValueError, FileNotFoundError) as exc:
            parser.exit(1, f"{exc}\n")
        if not roster:
            parser.exit(
                1,
                f"the {MARKET} roster resolved to no PERMNOs over "
                f"{window['start_date']}..{window['end_date']} under filter "
                f"{args.security_filter!r}.\n",
            )
        report = roster_source.report
        # PRINTED rather than summarised: on a roster this size the operator
        # cannot eyeball the difference between "the market" and "the market
        # minus a type I did not mean to drop".
        print(
            f"Roster ({len(roster):,} PERMNO(s) over the window; "
            f"{report['permnos_after_type_filter']:,} of "
            f"{report['permnos_before_type_filter']:,} all-time PERMNOs pass "
            f"filter {args.security_filter!r}, "
            f"{report['spells_dropped_by_type']:,} of "
            f"{report['spells_read']:,} spells dropped by type)"
        )
        acq_config = replace(acq_config, symbols=tuple(roster))

        # 6. count(*) per year page, then the guard -- BOTH before the first
        #    COPY of any daily row (D-03). This is the step that answers
        #    "does a whole-market window fit" with a measured number.
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
            f"Acquiring {len(roster):,} PERMNO(s) from {SOURCE.display_name} "
            f"over {window['start_date']}..{window['end_date']} "
            f"(refresh={args.refresh})"
        )
        result = run(SOURCE, acq_config, refresh=args.refresh)
        print(
            f"{len(result.succeeded):,} PERMNO(s) succeeded, "
            f"{len(result.failures):,} failed"
        )
        print(f"Raw data written under: {acq_config.raw_data_dir_path}")

        if args.to_zarr:
            data_dir = get_data_root() / "data" / "us_equity" / "1d"
            catalog_path = str(get_data_root() / "data" / "catalog")

            ds_config = CrspDatasetConfig(
                zarr_file_path=str(data_dir / STORE),
                raw_data_dir_path=acq_config.raw_data_dir_path,
                catalog_path=catalog_path,
                reference_dir=str(reference_dir),
                start_date=window["start_date"],
                end_date=window["end_date"],
                permnos=tuple(roster),
                security_filter=args.security_filter,
                # NOT `roster_universe=MARKET`. That field means "an index
                # provider decided membership, so a member is exempt from the
                # type filter inside its spell" (GAP-C), and it is resolved
                # through `CrspMembership.permno_intervals`, which serves
                # indexes only. Neither half fits a whole-market roster: this
                # roster IS the type filter's own output, so exempting it from
                # the type filter would be circular, and `crsp_all` is not an
                # index `CrspMembership` can answer for.
            )
            probe_dataset = CrspStockDataset(ds_config)
            refuse_conversion_without_raw_data(probe_dataset, result)
            print(
                f"Converting {len(roster):,} PERMNO(s) to the market panel in "
                f"{args.chunk} windows (resumable), filter "
                f"{args.security_filter!r}, on-new-listing "
                f"{args.on_new_listing!r}"
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
            print(f"Security filter sidecar:   {probe_dataset.filter_report_path()}")

            # The in-listing mask. Built under the SAME security filter as the
            # panel, so the two agree about what this universe is; the filter
            # rides in `kwargs` and therefore lands in the run's config.json.
            mask = CrspMarketConstituentDataset(
                ConstituentDatasetConfig(
                    zarr_file_path=str(data_dir / MEMBERSHIP_STORE),
                    cache_dir=str(reference_dir),
                    start_date=window["start_date"],
                    end_date=window["end_date"],
                    kwargs={"security_filter": args.security_filter},
                )
            )
            mask.from_raw_data().save()
            print(f"Market in-listing mask:    {mask.config.zarr_file_path}")

            if args.with_index_membership:
                for universe, (short_name, cls) in INDEX_PANELS.items():
                    panel = cls(
                        ConstituentDatasetConfig(
                            zarr_file_path=str(
                                data_dir
                                / MEMBERSHIP_TEMPLATE.format(name=short_name)
                            ),
                            cache_dir=str(reference_dir),
                            start_date=window["start_date"],
                            end_date=window["end_date"],
                            kwargs={"allow_unlinked": args.allow_unlinked_ndx},
                        )
                    )
                    panel.from_raw_data().save()
                    print(
                        f"{universe} membership panel: "
                        f"{panel.config.zarr_file_path}"
                    )
        else:
            print(
                "Skipping Zarr conversion (default). The raw shards above are "
                "the deliverable; pass --to-zarr to convert them."
            )
    finally:
        WrdsSession.close_shared()
