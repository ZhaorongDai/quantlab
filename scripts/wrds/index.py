"""Download an equity index's CRSP daily bars and its membership panel from WRDS.

CRSP (the Center for Research in Security Prices) is the standard academic
database of US stock prices, served through WRDS (Wharton Research Data
Services). CRSP identifies each security by its PERMNO, a permanent integer
id that never changes or gets reused. This script resolves the point-in-time
members of one index (every PERMNO that belonged to it at any time in the
window, including the ones that later left, which avoids survivorship bias),
pulls their ``crsp_a_stock.dsf_v2`` daily rows, and writes two Zarr stores
under the data root:

- ``data/us_equity/1d/wrds_crsp_{index}_1d.zarr``, the members' daily bars
  on the PERMNO axis, with the security-filter and ticker sidecars;
- ``data/us_equity/1d/wrds_crsp_{index}_membership.zarr``, the membership
  panel that marks the days each security was a member.

The raw rows go to ``downloads/us_equity/1d/wrds_crsp/wrds/`` and the CRSP,
Compustat and CCM reference tables to the sibling ``_reference/``. Reference
tables already on disk for the same CRSP release are reused, so running this
script and ``market.py`` back to back downloads them once.

``WRDS_USERNAME`` must be set in the environment. The password is never read
by this code; the PostgreSQL client library takes it from ``~/.pgpass``. One
run shares one WRDS connection, closed at the end whether the run succeeded
or failed, so a run triggers at most one Duo two-factor prompt.

Usage::

    export WRDS_USERNAME=<your-wrds-username>   # password lives in ~/.pgpass
    uv run python scripts/wrds/index.py --index sp500 --start 2015-01-01
    uv run python scripts/wrds/index.py --index nasdaq100 --start 2015-01-01 \\
        --end 2024-12-31 --refresh

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
from quantlab.dataset.constituent import (
    CompustatNasdaq100ConstituentDataset,
    CrspSP500ConstituentDataset,
)
from quantlab.dataset.crsp import CrspStockDataset
from quantlab.dataset.crsp.membership import CrspMembership
from quantlab.dataset.crsp.reference import CrspReference
from quantlab.utils.cli import (
    add_data_dir_arg,
    apply_data_dir,
    print_conversion_result,
)

SOURCE = DataSourceRegistry.get("wrds")
CAPABILITY = ("us_equity", "1d", "crsp_daily")
ACQ = SOURCE.acquisition_cls_for(*CAPABILITY)

#: ``--index`` name -> (CRSP membership universe id, constituent dataset class).
INDEXES: dict[str, tuple[str, type]] = {
    "sp500": (CrspMembership.SP500, CrspSP500ConstituentDataset),
    "nasdaq100": (CrspMembership.NASDAQ100, CompustatNasdaq100ConstituentDataset),
}


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build this script's argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Download an index's CRSP daily bars and membership panel from "
            "WRDS into Zarr. Requires WRDS_USERNAME in the environment and the "
            "password in ~/.pgpass."
        )
    )
    parser.add_argument(
        "--index",
        required=True,
        choices=sorted(INDEXES),
        help=(
            "The index. 'sp500' is CRSP's own dsp500list_v2 membership (from "
            "1925); 'nasdaq100' is Compustat's index membership linked to "
            "PERMNOs through CCM (no data before 1995)."
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
            "clipped to the last day of the CRSP release."
        ),
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Continue each PERMNO from its watermark instead of re-downloading.",
    )
    add_data_dir_arg(parser)
    return parser


def _schemas_for(universe: str) -> tuple[str, ...]:
    """The WRDS schemas an index run reads; only these are checked for access."""
    from quantlab.acquisition.wrds.crsp import CrspQueries

    schemas = [CrspQueries.STOCK_SCHEMA]
    if universe == CrspMembership.SP500:
        schemas.append(CrspQueries.INDEX_SCHEMA)
    else:
        schemas.extend((CrspQueries.COMPUSTAT_SCHEMA, CrspQueries.CCM_SCHEMA))
    return tuple(schemas)


if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()
    apply_data_dir(args)  # before any path is derived from the data root
    universe, constituent_cls = INDEXES[args.index]
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
            # 1. Entitlement first, so a missing subscription stops the run
            #    before anything is downloaded or written.
            CrspQueries.assert_entitled(session, _schemas_for(universe))

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
                product_end=product_end,
                include_sp500=universe == CrspMembership.SP500,
                include_nasdaq100=universe == CrspMembership.NASDAQ100,
            )
            print(f"Reference tables at: {reference_dir}")
            for name, entry in sorted((manifest.get("tables") or {}).items()):
                print(f"  {name}: {entry['rows']:,} row(s)")

            # 4. The point-in-time roster, as PERMNOs.
            roster = CrspMembership(CrspReference(reference_dir)).permnos_in_range(
                universe, start, end
            )
        except (RuntimeError, ValueError, FileNotFoundError) as exc:
            parser.exit(1, f"{exc}\n")
        if not roster:
            parser.exit(1, f"the {args.index} roster resolved to no PERMNOs over {start}..{end}.\n")
        print(f"Roster: {len(roster)} PERMNO(s) over {start}..{end}")
        acq_config = replace(acq_config, symbols=tuple(roster))

        # 5. Download.
        print(f"Acquiring from {SOURCE.display_name} (refresh={args.refresh})")
        result = run(SOURCE, acq_config, refresh=args.refresh)
        print(f"{len(result.succeeded)} PERMNO(s) succeeded, {len(result.failures)} failed")
        print(f"Raw data written under: {acq_config.raw_data_dir_path}")

        # 6. Convert: the bars, then the membership panel.
        data_dir = get_data_root() / "data" / "us_equity" / "1d"
        ds_config = CrspDatasetConfig(
            zarr_file_path=str(data_dir / f"wrds_crsp_{args.index}_1d.zarr"),
            raw_data_dir_path=acq_config.raw_data_dir_path,
            reference_dir=str(reference_dir),
            start_date=start,
            end_date=end,
            permnos=tuple(roster),
            roster_universe=universe,
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

        membership = constituent_cls(
            ConstituentDatasetConfig(
                zarr_file_path=str(data_dir / f"wrds_crsp_{args.index}_membership.zarr"),
                cache_dir=str(reference_dir),
                start_date=start,
                end_date=end,
            )
        )
        membership.from_raw_data().save()
        print(f"Membership panel:          {membership.config.zarr_file_path}")
    finally:
        WrdsSession.close_shared()
