"""Download the whole CRSP daily market from WRDS, with its listing panel.

The *market* is every security CRSP covers, with no index restriction. The
roster is read from CRSP's security-information history in the reference
tier: every PERMNO whose listing overlaps the window and passes the
``--security-filter`` preset (default ``equity_common``: common stock
including REITs, without ADRs, units, funds or ETFs). The script pulls those
PERMNOs' ``crsp_a_stock.dsf_v2`` daily rows and writes two Zarr stores
under the data root:

- ``data/us_equity/1d/wrds_crsp_market_1d.zarr``, the daily bars on the
  PERMNO axis, with the security-filter and ticker sidecars;
- ``data/us_equity/1d/wrds_crsp_market_membership.zarr``, the listing panel
  that marks the days each security was listed and of the requested type.

The raw rows go to ``downloads/us_equity/1d/wrds_crsp/wrds/`` and the CRSP
reference tables to the sibling ``_reference/``; both tiers are shared with
``index.py`` and ``etf.py``, so reference tables already on disk for the
same CRSP release are reused and raw rows already downloaded are not
downloaded again. Index membership panels come from ``index.py``.

A store from before the rename, ``wrds_crsp_all_*.zarr``, is not read: rename
it by hand or reconvert from the unchanged raw tier by running this script.

``WRDS_USERNAME`` must be set in the environment. The password is never read
by this code; the PostgreSQL client library takes it from ``~/.pgpass``. One
run shares one WRDS connection, closed at the end whether the run succeeded
or failed.

Usage::

    export WRDS_USERNAME=<your-wrds-username>   # password lives in ~/.pgpass
    uv run python scripts/wrds/market.py --start 2000-01-01
    uv run python scripts/wrds/market.py --start 2000-01-01 \\
        --security-filter shrcd_10_11 --refresh

``--end`` defaults to today and is clipped to the last day of the annual CRSP
release. ``--refresh`` continues each PERMNO from its recorded watermark
instead of downloading the whole window again.
"""

import argparse
from dataclasses import replace
from datetime import date

from quantlab.registry import DataSourceRegistry, convert, run
from quantlab.base.config import ConstituentDatasetConfig, CrspDatasetConfig
from quantlab.config import get_data_root
from quantlab.dataset.constituent import CrspMarketConstituentDataset
from quantlab.dataset.crsp import SECURITY_FILTER_PRESETS, CrspStockDataset
from quantlab.dataset.crsp.market import CrspMarketRoster
from quantlab.dataset.crsp.reference import CrspReference
from quantlab.utils.cli import (
    add_data_dir_arg,
    apply_data_dir,
    print_conversion_result,
)

SOURCE = DataSourceRegistry.get("wrds")
CAPABILITY = ("us_equity", "1d", "crsp_daily")
ACQ = SOURCE.acquisition_cls_for(*CAPABILITY)

STORE = "wrds_crsp_market_1d.zarr"
MEMBERSHIP_STORE = "wrds_crsp_market_membership.zarr"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download the whole CRSP daily market from WRDS into Zarr, with "
            "its listing panel. Requires WRDS_USERNAME in the environment and "
            "the password in ~/.pgpass."
        )
    )
    parser.add_argument(
        "--start", required=True, help="First day of the window, YYYY-MM-DD."
    )
    parser.add_argument(
        "--end",
        default=None,
        help=(
            "Last day of the window, YYYY-MM-DD. Defaults to today and is "
            "clipped to the last day of the CRSP release."
        ),
    )
    parser.add_argument(
        "--security-filter",
        choices=sorted(SECURITY_FILTER_PRESETS),
        default="equity_common",
        help=(
            "Which securities the market holds (default equity_common: common "
            "stock including REITs, without ADRs, units, funds or ETFs). "
            "'shrcd_10_11' keeps only US corporate common stock; 'none' keeps "
            "every security type. Applied to the roster and, per date, to the "
            "panel."
        ),
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Continue each PERMNO from its watermark instead of re-downloading.",
    )
    add_data_dir_arg(parser)
    return parser


if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()
    apply_data_dir(args)  # before any path is derived from the data root
    requested_end = args.end or date.today().isoformat()

    # Imported here so the session class is resolved at run time.
    from quantlab.acquisition.wrds.crsp import CrspQueries
    from quantlab.acquisition.wrds.crsp_reference import CrspReferenceTables
    from quantlab.acquisition.wrds.taq import WrdsSession

    try:
        session = WrdsSession.shared()
    except RuntimeError as exc:
        parser.exit(1, f"{exc}\n")

    try:
        try:
            # 1. Entitlement first. A market pull needs the stock schema only.
            CrspQueries.assert_entitled(session, (CrspQueries.STOCK_SCHEMA,))

            # 2. Clip the window to the end of the annual CRSP release.
            product_end = CrspQueries.product_end(session)
            start_date, end_date, clipped = ACQ.window_for_product_end(
                product_end, args.start, requested_end, clip=True
            )
            if clipped is not None:
                print(f"clipped end {requested_end} -> {clipped.isoformat()} (CRSP release end)")
            start, end = start_date.isoformat(), end_date.isoformat()

            # 3. Reference tables, reused when this release's are on disk.
            acq_config = SOURCE.config_factory_for(*CAPABILITY)(
                symbols=(), start_date=start, end_date=end,
                kwargs={"clip_to_product_end": True},
            )
            reference_dir = ACQ.reference_dir_for(acq_config)
            manifest = CrspReferenceTables(session, reference_dir).pull(
                product_end=product_end, include_sp500=False
            )
            print(f"Reference tables at: {reference_dir}")
            for name, entry in sorted((manifest.get("tables") or {}).items()):
                print(f"  {name}: {entry['rows']:,} row(s)")

            # 4. The roster: every listing overlapping the window that passes
            #    the security filter.
            roster_source = CrspMarketRoster(CrspReference(reference_dir))
            roster = roster_source.permnos_in_range(
                start, end, security_filter=args.security_filter
            )
        except (RuntimeError, ValueError, FileNotFoundError) as exc:
            parser.exit(1, f"{exc}\n")
        if not roster:
            parser.exit(
                1,
                f"the market roster resolved to no PERMNOs over {start}..{end} "
                f"under filter {args.security_filter!r}.\n",
            )
        report = roster_source.report
        print(
            f"Roster: {len(roster):,} PERMNO(s) over {start}..{end} "
            f"({report['permnos_after_type_filter']:,} of "
            f"{report['permnos_before_type_filter']:,} all-time PERMNOs pass "
            f"filter {args.security_filter!r})"
        )
        acq_config = replace(acq_config, symbols=tuple(roster))

        # 5. Download.
        print(f"Acquiring from {SOURCE.display_name} (refresh={args.refresh})")
        result = run(SOURCE, acq_config, refresh=args.refresh)
        print(f"{len(result.succeeded):,} PERMNO(s) succeeded, {len(result.failures):,} failed")
        print(f"Raw data written under: {acq_config.raw_data_dir_path}")

        # 6. Convert: the bars, then the listing panel.
        data_dir = get_data_root() / "data" / "us_equity" / "1d"
        ds_config = CrspDatasetConfig(
            zarr_file_path=str(data_dir / STORE),
            raw_data_dir_path=acq_config.raw_data_dir_path,
            reference_dir=str(reference_dir),
            start_date=start,
            end_date=end,
            permnos=tuple(roster),
            security_filter=args.security_filter,
        )
        if not CrspStockDataset(ds_config).has_raw_data():
            parser.exit(
                1,
                f"Refusing to convert: no raw data under "
                f"{acq_config.raw_data_dir_path} ({len(result.failures)} PERMNO(s) "
                f"failed this run). No store was written.\n",
            )
        print_conversion_result(convert(SOURCE, ds_config, data_type="crsp_daily"))
        print(f"Security filter sidecar:   {CrspStockDataset(ds_config).filter_report_path()}")

        mask = CrspMarketConstituentDataset(
            ConstituentDatasetConfig(
                zarr_file_path=str(data_dir / MEMBERSHIP_STORE),
                cache_dir=str(reference_dir),
                start_date=start,
                end_date=end,
                kwargs={"security_filter": args.security_filter},
            )
        )
        mask.from_raw_data().save()
        print(f"Market listing panel:      {mask.config.zarr_file_path}")
    finally:
        WrdsSession.close_shared()
