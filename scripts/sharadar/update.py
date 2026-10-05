"""Refresh the Sharadar raw tier and append the new bars to the stores.

Run every morning after ``download.py`` has built the stores. One run:

1. pulls TICKERS whole (new listings and ticker changes) and SP500 whole (it
   is small);
2. pulls a trailing date window of SEP, SFP and ACTIONS over REST: from the
   ``--trading-days``-th most recent raw date before each table's watermark
   through today (US/Eastern), so a late vendor correction to a recent day is
   seen;
3. runs ``update()`` on ``sharadar_sep_1d.zarr``, ``sharadar_sfp_1d.zarr``
   and ``sharadar_sp500_membership.zarr`` in ``--zarr-dir``, each from the
   first day it already holds: new bars are
   appended, earlier rows are never rewritten, and a vendor correction to a
   stored date is listed in ``<store>.corrections.json`` instead.

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
)
from quantlab.dataset.config import ConstituentDatasetConfig, SharadarDatasetConfig
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
WHOLE_TABLES = ("tickers", "sp500")
#: Pulled as a trailing date window.
WINDOW_TABLES = ("sep", "sfp", "actions")

PRICE_STORES = {"sep": "sharadar_sep_1d.zarr", "sfp": "sharadar_sfp_1d.zarr"}
MEMBERSHIP_STORE = "sharadar_sp500_membership.zarr"


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
        store for store in (*PRICE_STORES.values(), MEMBERSHIP_STORE)
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
    except (SharadarEntitlementError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")

    for code, store in PRICE_STORES.items():
        config = SharadarDatasetConfig(
            zarr_file_path=str(zarr_dir / store),
            raw_data_dir_path=str(vendor_root),
            table=code,
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
