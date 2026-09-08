"""Rebuild the Binance spot-kline Zarr store from locally-dropped CSVs.

This script does NOT download anything (D-03: no new Binance downloader in
this phase) -- it only reads monthly Binance kline CSV files that already
exist on disk under `raw_data_dir_path` (default:
`{root}/downloads/crypto_spot/1d/spot/monthly/klines/`, see
`config/__init__.py:spot_kline_config()`), converts them through the same
`Dataset`/`DataBackend` abstraction US equities use (`SpotKlineDataset`), and
writes the resulting `xarray.Dataset` to the configured Zarr path. `{root}` is
whatever `config.get_data_root()` resolves: `--data-dir` first, then
`QUANTLAB_DATA_DIR`, then the repo-root `data/` directory.

**Two directory knobs, at two different levels, and they compose.**
`--data-dir` relocates the whole storage ROOT -- the Zarr store, the nautilus
catalog AND the default raw CSV directory move with it. `--raw-data-dir`
redirects ONLY the raw CSV directory, at one pre-existing location that need
not follow the `data/{market}/{frequency}/...` convention at all; that is its
whole purpose, so there is no need to move, copy, or symlink files into the
project's convention path. Neither supersedes the other: giving both reads
CSVs from `--raw-data-dir` and writes the Zarr under `--data-dir`, which is
the useful combination.

Usage:
    uv run python ingest_binance_spot.py
    uv run python ingest_binance_spot.py --symbols BTCUSDT,ETHUSDT
    uv run python ingest_binance_spot.py --raw-data-dir ~/Downloads/spot/monthly/klines
    uv run python ingest_binance_spot.py --data-dir /Volumes/BigDisk \
        --raw-data-dir ~/Downloads/spot/monthly/klines
"""

import argparse

from base.config import DatasetConfig
from config import spot_kline_config
from dataset.spot import SpotKlineDataset
from utils.cli import add_data_dir_arg, apply_data_dir


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
            "path. Narrower than --data-dir and composes with it: --data-dir "
            "moves the whole storage root, this redirects the raw CSV "
            "directory alone, so passing both reads CSVs from here and "
            "writes the Zarr under the relocated root."
        ),
    )
    add_data_dir_arg(parser)
    return parser


if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Before `_build_dataset_config`, which calls `spot_kline_config()`, and
    # the position is load-bearing: the factory snapshots its paths as strings
    # at construction time, so a root override applied afterwards silently does
    # nothing (DDIR-04). The narrow `--raw-data-dir` mutation inside
    # `_build_dataset_config` still runs AFTER construction, which is what
    # makes the two knobs compose rather than conflict (D-05).
    apply_data_dir(args)

    config = _build_dataset_config(args)
    SpotKlineDataset(config).from_raw_data().save()
