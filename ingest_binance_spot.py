"""Rebuild the Binance spot-kline Zarr store from locally-dropped CSVs.

This script does NOT download anything (D-03: no new Binance downloader in
this phase) -- it only reads monthly Binance kline CSV files that already
exist on disk under `raw_data_dir_path` (default:
`{QUANTLAB_DATA_DIR}/downloads/crypto_spot/1d/spot/monthly/klines/`, see
`config/__init__.py:spot_kline_config()`), converts them through the same
`Dataset`/`DataBackend` abstraction US equities use (`SpotKlineDataset`), and
writes the resulting `xarray.Dataset` to the configured Zarr path.

If your Binance CSVs live somewhere else on this machine (e.g. a pre-existing
download directory that doesn't follow the `data/{market}/{frequency}/...`
convention), pass `--raw-data-dir` to point directly at that location --
no need to move, copy, or symlink files into the project's convention path.

Usage:
    uv run python ingest_binance_spot.py
    uv run python ingest_binance_spot.py --symbols BTCUSDT,ETHUSDT
    uv run python ingest_binance_spot.py --raw-data-dir ~/Downloads/spot/monthly/klines
"""

import argparse

from base.config import DatasetConfig
from config import spot_kline_config
from dataset.spot import SpotKlineDataset


def _build_dataset_config(args: argparse.Namespace) -> DatasetConfig:
    symbols = args.symbols.split(",") if args.symbols else None
    config = spot_kline_config(
        symbols=symbols,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    if args.raw_data_dir is not None:
        config.raw_data_dir_path = args.raw_data_dir
    return config


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild the Binance spot-kline Zarr store from locally-dropped "
            "monthly CSVs. Does not download anything."
        )
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help=(
            "Comma-separated symbols (e.g. BTCUSDT,ETHUSDT). Omit to process "
            "every symbol found under raw_data_dir_path."
        ),
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date (inclusive), e.g. 2024-01-01.",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date (inclusive), e.g. 2024-12-31.",
    )
    parser.add_argument(
        "--raw-data-dir",
        type=str,
        default=None,
        help=(
            "Override the default data/{market}/{frequency}/... raw CSV "
            "directory; point this at wherever your Binance CSVs already "
            "live, e.g. a pre-existing local download directory, with no "
            "need to move/copy/symlink files into the project's convention "
            "path."
        ),
    )
    return parser


if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()

    config = _build_dataset_config(args)
    SpotKlineDataset(config).from_raw_data().save()
