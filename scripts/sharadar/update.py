"""Refresh the Sharadar raw tier and append the new bars to the stores.

Run every morning after ``download.py`` has built the stores. One run:

1. pulls TICKERS whole (new listings and ticker changes), INDICATORS whole
   (units and the 8-K event codes), and SP500 and SF3A whole (they are
   small; SF3A's newest quarter fills in place);
2. pulls a trailing date window of SEP, SFP, ACTIONS, EVENTS and SF2 over REST: from the
   ``--trading-days``-th most recent raw date before each table's watermark
   through today (US/Eastern), so a late vendor correction to a recent day is
   seen; and every SF1 row the vendor changed since SF1's watermark (by
   ``lastupdated``), new filings included, or the whole of SF1 in bulk when
   that query fails (Sharadar stops a query after 15 seconds), and the
   same for DAILY;
3. runs ``update()`` on every store ``download.py`` built in ``--zarr-dir``
   (``sharadar_sep_1d.zarr``, ``sharadar_sfp_1d.zarr``,
   ``sharadar_sp500_1d.zarr``, ``sharadar_spy_1d.zarr``,
   ``sharadar_sp500_membership.zarr``, ``sharadar_sf1_arq.zarr``,
   ``sharadar_sf1_art.zarr``, ``sharadar_daily_1d.zarr``,
   ``sharadar_events_1d.zarr``, ``sharadar_insiders_1d.zarr``,
   ``sharadar_holdings_1d.zarr``, ``sharadar_industry_1d.zarr``,
   ``sharadar_sf1_fiscal_years.zarr`` and ``sharadar_share_class_1d.zarr``),
   each from the
   first day it already holds: new bars are appended and earlier rows are
   never rewritten. A vendor correction to a stored date of a price store is
   listed in ``<store>.corrections.json`` instead; one to SF1 or DAILY is not
   reported. Each raw file's tickers are mapped as of its own pull (its
   run's TICKERS snapshot, then TICKERS, the ACTIONS ticker changes and the
   store's ticker sidecar), so a ticker change between pulls never stops the
   update; a row still unmapped is left out and listed in
   ``<store>.unmapped.json``.

SF3 (every 13F holding, 400 MB) and SF3B (holdings by investor) feed no
store and are not refreshed here; ``download.py`` pulls them whole.

An interrupted run resumes from each table's watermark and each store's last
bar; running it twice in a day appends nothing the second time.

``SHARADAR_API_KEY`` must be set in the environment. Both directories must be
outside this repository; the script refuses one inside it.

Usage::

    export SHARADAR_API_KEY=<your-sharadar-key>
    uv run python scripts/sharadar/update.py \\
        --download-dir /data/quantlab/downloads --zarr-dir /data/quantlab/zarrs

``--download-dir`` and ``--zarr-dir`` default to the current directory and
must be the ones ``download.py`` used.
"""

import argparse
from pathlib import Path

import pandas as pd
import xarray as xr

from quantlab.acquisition.sharadar.client import (
    SharadarClient,
    SharadarEntitlementError,
    SharadarHttpError,
)
from quantlab.dataset.config import (
    SPY_PERMATICKER,
    ConstituentDatasetConfig,
    SharadarDailyConfig,
    SharadarDatasetConfig,
    SharadarEventsConfig,
    SharadarFiscalYearsConfig,
    SharadarFundamentalsConfig,
    SharadarHoldingsConfig,
    SharadarIndustryConfig,
    SharadarInsidersConfig,
    SharadarShareClassConfig,
)
from quantlab.dataset.sharadar.daily import SharadarDailyDataset
from quantlab.dataset.sharadar.events import SharadarEventsDataset
from quantlab.dataset.sharadar.fiscal_years import SharadarFiscalYearsDataset
from quantlab.dataset.sharadar.holdings import SharadarHoldingsDataset
from quantlab.dataset.sharadar.industry import SharadarIndustryDataset
from quantlab.dataset.sharadar.insiders import SharadarInsidersDataset
from quantlab.dataset.sharadar.share_class import SharadarShareClassDataset
from quantlab.dataset.sharadar.fundamentals import SharadarFundamentalsDataset
from quantlab.dataset.sharadar.membership import SharadarSP500ConstituentDataset
from quantlab.dataset.sharadar.stock import SharadarStockDataset
from quantlab.dataset.sharadar.tables import VENDOR_DIR
from quantlab.utils.cli import (
    add_output_dir_args,
    inside_repository,
    print_conversion_result,
    resolve_output_dirs,
)

#: The repository this script lives in; no output may go inside it.
REPO_ROOT = Path(__file__).resolve().parents[2]

#: Pulled whole every run: reference and membership tables are small.
WHOLE_TABLES = ("tickers", "indicators", "sp500", "sf3a")
#: Pulled as a trailing date window.
WINDOW_TABLES = ("sep", "sfp", "actions", "events", "sf2")
#: Pulled as the rows changed since the watermark (``lastupdated``).
UPDATED_TABLES = ("sf1", "daily")

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
#: The stores of the filing, ownership, industry, fiscal-year and share-class panels, with their config and dataset classes.
PANEL_STORES = {
    "sharadar_events_1d.zarr": (SharadarEventsConfig, SharadarEventsDataset),
    "sharadar_insiders_1d.zarr": (SharadarInsidersConfig, SharadarInsidersDataset),
    "sharadar_holdings_1d.zarr": (SharadarHoldingsConfig, SharadarHoldingsDataset),
    "sharadar_industry_1d.zarr": (SharadarIndustryConfig, SharadarIndustryDataset),
    "sharadar_sf1_fiscal_years.zarr": (SharadarFiscalYearsConfig, SharadarFiscalYearsDataset),
    "sharadar_share_class_1d.zarr": (SharadarShareClassConfig, SharadarShareClassDataset),
}


def _store_start(path: Path) -> str:
    """Return the first day a store holds, so an update keeps its range."""
    first = xr.open_zarr(path)["timestamp"].values[0]
    return pd.Timestamp(first).date().isoformat()


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build this script's argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Pull the recent Sharadar dates and append them to the stores "
            "download.py built. Requires SHARADAR_API_KEY in the environment. "
            "Both directories must be outside this repository (the data is "
            "licensed); the script refuses otherwise."
        )
    )
    parser.add_argument(
        "--trading-days",
        type=int,
        default=10,
        help=(
            "Raw dates re-pulled before each table's watermark, so a late "
            "vendor correction is seen. Default: 10."
        ),
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
    vendor_root = download_dir / VENDOR_DIR
    missing = [
        store for store in (*PRICE_STORES, MEMBERSHIP_STORE, *FUNDAMENTALS_STORES, DAILY_STORE, *PANEL_STORES)
        if not (zarr_dir / store).exists()
    ]
    if missing:
        parser.exit(1, f"no store {missing} in {zarr_dir}; run download.py first.\n")

    client = SharadarClient()
    try:
        for code in WHOLE_TABLES:
            print(f"{code}: {client.bulk_table(code, download_dir)}")
        for code in WINDOW_TABLES:
            print(f"{code}: {client.window_table(code, download_dir, trading_days=args.trading_days)}")
        for code in UPDATED_TABLES:
            try:
                print(f"{code}: {client.updated_table(code, download_dir)}")
            except SharadarHttpError as exc:
                # Sharadar stops a query after 15 seconds; after a mass
                # re-stamp of lastupdated the changed rows are most of the
                # table, and the bulk zip is the cheaper copy of them.
                print(f"{code}: updated pull failed ({exc}); pulling it in bulk instead")
                print(f"{code}: {client.bulk_table(code, download_dir)}")
    except (SharadarEntitlementError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")

    for store, fields in PRICE_STORES.items():
        config = SharadarDatasetConfig(
            zarr_file_path=str(zarr_dir / store),
            raw_data_dir_path=str(vendor_root),
            **fields,
            start_date=_store_start(zarr_dir / store),
        )
        dataset = SharadarStockDataset(config).update()
        print_conversion_result(dataset.last_chunk_result)
        if dataset.corrections_path().exists():
            print(f"  vendor corrections: {dataset.corrections_path()}")

    membership = SharadarSP500ConstituentDataset(
        ConstituentDatasetConfig(
            zarr_file_path=str(zarr_dir / MEMBERSHIP_STORE),
            cache_dir=str(vendor_root),
            start_date=_store_start(zarr_dir / MEMBERSHIP_STORE),
        )
    )
    print_conversion_result(membership.update().last_chunk_result)

    for store, dimension in FUNDAMENTALS_STORES.items():
        config = SharadarFundamentalsConfig(
            zarr_file_path=str(zarr_dir / store),
            raw_data_dir_path=str(vendor_root),
            dimension=dimension,
            start_date=_store_start(zarr_dir / store),
        )
        print_conversion_result(SharadarFundamentalsDataset(config).update().last_chunk_result)

    daily = SharadarDailyConfig(
        zarr_file_path=str(zarr_dir / DAILY_STORE),
        raw_data_dir_path=str(vendor_root),
        start_date=_store_start(zarr_dir / DAILY_STORE),
    )
    print_conversion_result(SharadarDailyDataset(daily).update().last_chunk_result)

    for store, (config_cls, dataset_cls) in PANEL_STORES.items():
        config = config_cls(
            zarr_file_path=str(zarr_dir / store),
            raw_data_dir_path=str(vendor_root),
            start_date=_store_start(zarr_dir / store),
        )
        print_conversion_result(dataset_cls(config).update().last_chunk_result)
