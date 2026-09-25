"""Rebuild the Binance spot-kline Zarr store from monthly CSVs already on disk.

Nothing is downloaded. The script reads the monthly Binance kline CSV files
under ``raw_data_dir_path``, converts them through ``SpotKlineDataset`` and
writes the resulting ``xarray.Dataset`` to the configured Zarr path. By
default the CSVs are expected under
``{root}/downloads/crypto_spot/1d/spot/monthly/klines/``, where ``{root}``
is ``--data-dir``, else ``QUANTLAB_DATA_DIR``, else the repository's
``data/`` directory.

The two directory flags work at different levels and compose. ``--data-dir``
relocates the whole storage root, so the Zarr store, the nautilus catalog and
the default raw CSV directory all move with it. ``--raw-data-dir`` redirects
only the raw CSV directory, to wherever the CSVs already live, with no need
to copy or symlink them into the project layout. Given both, CSVs are read
from ``--raw-data-dir`` and the Zarr is written under ``--data-dir``. No
credentials are needed.

Usage:
    uv run python scripts/ingest_binance_spot.py
    uv run python scripts/ingest_binance_spot.py --symbols BTCUSDT,ETHUSDT
    uv run python scripts/ingest_binance_spot.py \
        --raw-data-dir ~/Downloads/spot/monthly/klines
    uv run python scripts/ingest_binance_spot.py --data-dir /Volumes/BigDisk \
        --raw-data-dir ~/Downloads/spot/monthly/klines
"""

import argparse

from quantlab.base.config import DatasetConfig
from quantlab.config import spot_kline_config
from quantlab.dataset.spot import SpotKlineDataset
from quantlab.utils.cli import add_data_dir_arg, apply_data_dir


def _build_dataset_config(args: argparse.Namespace) -> DatasetConfig:
    """Build the ``DatasetConfig`` for the parsed arguments.

    The config comes from ``spot_kline_config``; ``--raw-data-dir`` is applied
    on top of it afterwards, which is what lets it compose with a root already
    relocated by ``--data-dir``.
    """
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
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild the Binance spot-kline Zarr store from monthly CSVs "
            "already on disk. Does not download anything."
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
            "live, with no need to move, copy or symlink them into the "
            "project's layout. Narrower than --data-dir and composes with "
            "it: --data-dir moves the whole storage root, this redirects the "
            "raw CSV directory alone, so passing both reads CSVs from here "
            "and writes the Zarr under the relocated root."
        ),
    )
    add_data_dir_arg(parser)
    return parser


if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Must run before ``spot_kline_config()`` is called: the factory snapshots
    # its paths at construction time, so a later root override is ignored.
    # The ``--raw-data-dir`` override is applied after construction, which is
    # what makes the two flags compose.
    apply_data_dir(args)

    config = _build_dataset_config(args)
    SpotKlineDataset(config).from_raw_data().save()
