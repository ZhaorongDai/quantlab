"""Download every Sharadar table of the raw tier and build the Zarr stores.

One run pulls each table the raw tier knows (``sep``, ``sfp``, ``actions``,
``sp500``, ``sf1``, ``daily``, ``tickers``, ``indicators``; METRICS is never
downloaded) as a bulk zip into ``<download-dir>/sharadar/<table>/``, then
builds or extends eight stores in ``--zarr-dir``:

- ``sharadar_sep_1d.zarr``, stock prices on the permaticker axis, raw and
  adjusted (the default universe: domestic common stock);
- ``sharadar_sfp_1d.zarr``, fund prices (ETFs and the like), every category;
- ``sharadar_sp500_1d.zarr``, stock prices of every permaticker ever an
  S&P 500 member, with all its bars and whatever its category;
- ``sharadar_spy_1d.zarr``, SPY alone, the S&P 500 benchmark;
- ``sharadar_sf1_arq.zarr`` and ``sharadar_sf1_art.zarr``, point-in-time
  fundamentals (as reported, quarterly and trailing twelve months) on SEP's
  trading days;
- ``sharadar_daily_1d.zarr``, the DAILY valuations (market cap and EV in
  USD, PE, PB, PS, EV/EBIT, EV/EBITDA);
- ``sharadar_sp500_membership.zarr``, point-in-time S&P 500 membership.

Each store is built with ``update()``, so it keeps the chunk ledger the daily
update (``update.py``) reads; a store that already exists is extended, not
rebuilt, and a vendor correction to a stored date is reported in
``<store>.corrections.json`` rather than written. Delete a store to rebuild
it from the new bulk pull.

``SHARADAR_API_KEY`` must be set in the environment. The data is licensed for
personal use: both directories must be outside this repository, and the
script refuses one inside it.

Usage::

    export SHARADAR_API_KEY=<your-sharadar-key>
    uv run python scripts/sharadar/download.py \\
        --download-dir /data/quantlab/downloads --zarr-dir /data/quantlab/zarrs
    uv run python scripts/sharadar/download.py --start 2010-01-01 --years 10

``--download-dir`` and ``--zarr-dir`` default to the current directory.
"""

import argparse
from pathlib import Path

from quantlab.acquisition.sharadar.client import (
    SharadarClient,
    SharadarEntitlementError,
)
from quantlab.dataset.config import (
    SPY_PERMATICKER,
    ConstituentDatasetConfig,
    SharadarDailyConfig,
    SharadarDatasetConfig,
    SharadarFundamentalsConfig,
)
from quantlab.dataset.sharadar.daily import SharadarDailyDataset
from quantlab.dataset.sharadar.fundamentals import SharadarFundamentalsDataset
from quantlab.dataset.sharadar.membership import SharadarSP500ConstituentDataset
from quantlab.dataset.sharadar.stock import SharadarStockDataset
from quantlab.dataset.sharadar.tables import TABLES, VENDOR_DIR
from quantlab.utils.cli import (
    add_output_dir_args,
    inside_repository,
    print_conversion_result,
    resolve_output_dirs,
)

#: The repository this script lives in; no output may go inside it.
REPO_ROOT = Path(__file__).resolve().parents[2]

#: Pulled first: the price conversions map tickers through TICKERS.
TABLE_ORDER = ("tickers", "indicators", "sep", "sfp", "actions", "sp500", "sf1", "daily")

#: Each price store and the config fields it is built with beyond its paths:
#: the whole SEP and SFP tables, every permaticker ever an S&P 500 member
#: with all its bars (the backtest's price dataset), and SPY alone (the
#: benchmark).
PRICE_STORES = {
    "sharadar_sep_1d.zarr": {"table": "sep"},
    "sharadar_sfp_1d.zarr": {"table": "sfp"},
    "sharadar_sp500_1d.zarr": {"table": "sep", "roster_universe": "sp500"},
    "sharadar_spy_1d.zarr": {"table": "sfp", "permatickers": (SPY_PERMATICKER,)},
}
MEMBERSHIP_STORE = "sharadar_sp500_membership.zarr"
#: Each point-in-time fundamentals store and its as-reported SF1 dimension.
FUNDAMENTALS_STORES = {"sharadar_sf1_arq.zarr": "ARQ", "sharadar_sf1_art.zarr": "ART"}
#: The DAILY valuation store.
DAILY_STORE = "sharadar_daily_1d.zarr"


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build this script's argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Download every Sharadar table (bulk zips) and build the SEP, SFP, "
            "S&P 500 roster, SPY, S&P 500 membership, SF1 and DAILY Zarr stores. Requires SHARADAR_API_KEY in "
            "the environment. Both directories must be outside this "
            "repository (the data is licensed); the script refuses otherwise."
        )
    )
    parser.add_argument(
        "--start",
        default=None,
        help=(
            "First day the stores hold, YYYY-MM-DD. Default: the whole history "
            "of the pull. The raw tables are always pulled whole."
        ),
    )
    parser.add_argument(
        "--years",
        choices=("5", "10", "full"),
        default="full",
        help="History tier of the bulk files; must not exceed the plan. Default: full.",
    )
    parser.add_argument(
        "--download-workers",
        type=int,
        default=8,
        help="Threads downloading byte ranges of one bulk zip. Default: 8.",
    )
    add_output_dir_args(
        parser,
        download_help=(
            "Directory for the raw downloads: each table goes to "
            "<download-dir>/sharadar/<table>/, with its watermark beside it. "
            "Default: the current directory, which must be outside this "
            "repository."
        ),
    )
    return parser


if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()
    download_dir, zarr_dir = resolve_output_dirs(args)
    if offending := inside_repository([download_dir, zarr_dir], REPO_ROOT):
        parser.exit(
            1,
            f"refusing to write Sharadar data inside the repository ({REPO_ROOT}): "
            f"{[str(p) for p in offending]}. Pass --download-dir and --zarr-dir "
            f"outside it.\n",
        )
    assert set(TABLE_ORDER) == set(TABLES), "every raw-tier table is pulled"

    client = SharadarClient(download_workers=args.download_workers)
    try:
        for code in TABLE_ORDER:
            path = client.bulk_table(code, download_dir, years=args.years)
            print(f"{code}: {path}")
    except (SharadarEntitlementError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")

    vendor_root = download_dir / VENDOR_DIR
    zarr_dir.mkdir(parents=True, exist_ok=True)
    for store, fields in PRICE_STORES.items():
        config = SharadarDatasetConfig(
            zarr_file_path=str(zarr_dir / store),
            raw_data_dir_path=str(vendor_root),
            **fields,
            start_date=args.start,
        )
        print_conversion_result(SharadarStockDataset(config).update().last_chunk_result)

    membership = SharadarSP500ConstituentDataset(
        ConstituentDatasetConfig(
            zarr_file_path=str(zarr_dir / MEMBERSHIP_STORE),
            cache_dir=str(vendor_root),
            start_date=args.start,
        )
    )
    print_conversion_result(membership.update().last_chunk_result)

    for store, dimension in FUNDAMENTALS_STORES.items():
        config = SharadarFundamentalsConfig(
            zarr_file_path=str(zarr_dir / store),
            raw_data_dir_path=str(vendor_root),
            dimension=dimension,
            start_date=args.start,
        )
        print_conversion_result(SharadarFundamentalsDataset(config).update().last_chunk_result)

    daily = SharadarDailyConfig(
        zarr_file_path=str(zarr_dir / DAILY_STORE),
        raw_data_dir_path=str(vendor_root),
        start_date=args.start,
    )
    print_conversion_result(SharadarDailyDataset(daily).update().last_chunk_result)
