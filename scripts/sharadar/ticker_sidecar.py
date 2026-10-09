"""Write the ticker sidecar of existing Sharadar price stores, without touching the stores.

A conversion or an update of a Sharadar price store writes
``<store>.sharadar_tickers.json`` beside it, from the TICKERS and ACTIONS
tables of the download directory: the ticker and company each permaticker
had over time, which a backtest shows in its Holdings tab and its settlement
and rejected-order records. A store built before sidecars existed has none,
and its permatickers read as their ids. This script writes the sidecar of
each price store ``download.py`` builds that exists under the data root
``--data-dir`` (``sharadar_sep_1d.zarr``, ``sharadar_sfp_1d.zarr``,
``sharadar_sp500_1d.zarr``, ``sharadar_spy_1d.zarr``, each at
``<data-dir>/market/sharadar/<stem>/<stem>.zarr``), naming the store's
permatickers from ``<download-dir>/sharadar``. The stores are only read.
Nothing is downloaded; run ``update.py`` first for the latest tables.

Usage::

    uv run python scripts/sharadar/ticker_sidecar.py \\
        --download-dir /data/quantlab/downloads --data-dir /data/quantlab

``--download-dir`` and ``--data-dir`` default to the current directory and
must be the ones ``download.py`` used.
"""

import argparse
from pathlib import Path

from quantlab.dataset.config import SharadarDatasetConfig
from quantlab.dataset.sharadar.stock import SharadarStockDataset
from quantlab.dataset.sharadar.tables import VENDOR_DIR
from quantlab.utils.cli import (
    add_output_dir_args,
    inside_repository,
    resolve_output_dirs,
)

#: The repository this script lives in; no output may go inside it.
REPO_ROOT = Path(__file__).resolve().parents[2]

#: The folder under the data root that holds Sharadar's market stores.
MARKET_FOLDER = Path("market") / "sharadar"

#: Each price store ``download.py`` builds and its price table.
PRICE_STORES = {
    "sharadar_sep_1d": "sep",
    "sharadar_sfp_1d": "sfp",
    "sharadar_sp500_1d": "sep",
    "sharadar_spy_1d": "sfp",
}


def store_path(folder: Path, stem: str) -> Path:
    """``<folder>/<stem>/<stem>.zarr``: a store in its own folder, beside its README.md."""
    return folder / stem / f"{stem}.zarr"


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build this script's argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Write <store>.sharadar_tickers.json beside each existing Sharadar "
            "price store from the downloaded TICKERS and ACTIONS; the stores "
            "are not changed."
        )
    )
    add_output_dir_args(
        parser,
        download_help=(
            "Directory holding the raw downloads (<download-dir>/sharadar/). "
            "Default: the current directory."
        ),
        data_help=(
            "The data root download.py wrote into; the price stores are "
            "<data-dir>/market/sharadar/<stem>/<stem>.zarr. Default: the "
            "current directory."
        ),
    )
    return parser


if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()
    download_dir, data_dir = resolve_output_dirs(args)
    if offending := inside_repository([data_dir], REPO_ROOT):
        parser.exit(
            1,
            f"refusing to write Sharadar data inside the repository ({REPO_ROOT}): "
            f"{[str(p) for p in offending]}. Pass --data-dir outside it.\n",
        )
    market_dir = data_dir / MARKET_FOLDER
    stores = {
        name: code for name, code in PRICE_STORES.items()
        if store_path(market_dir, name).exists()
    }
    if not stores:
        parser.exit(1, f"no Sharadar price store {list(PRICE_STORES)} under {market_dir}.\n")
    for name, code in stores.items():
        dataset = SharadarStockDataset(
            SharadarDatasetConfig(
                zarr_file_path=str(store_path(market_dir, name)),
                raw_data_dir_path=str(download_dir / VENDOR_DIR),
                table=code,
            )
        )
        print(f"{name}: {dataset.write_ticker_sidecar()}")
