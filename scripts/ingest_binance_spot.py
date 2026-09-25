"""Rebuild the Binance spot-kline Zarr store from monthly CSVs already on disk.

A kline is Binance's name for an OHLCV bar (open, high, low, close, volume).
Nothing is downloaded: the script reads the monthly kline CSV files that
Binance publishes for bulk download, converts them through
``SpotKlineDataset`` into an ``xarray.Dataset`` indexed by ``timestamp`` and
``symbol``, and writes it to the configured Zarr store (a chunked on-disk
array format that ``xarray`` reads). By default the CSVs are expected under
``{root}/downloads/crypto_spot/1d/spot/monthly/klines/``, where ``{root}``
is ``--data-dir``, else ``QUANTLAB_DATA_DIR``, else the repository's
``data/`` directory. No credentials are needed.

The two directory flags work at different levels and can be combined.
``--data-dir`` moves the whole storage root, so the Zarr store, the Nautilus
catalog and the default raw CSV directory all move with it.
``--raw-data-dir`` redirects only the raw CSV directory, to wherever the
CSVs already live, so they need not be copied or linked into the project
layout. With both, CSVs are read from ``--raw-data-dir`` and the Zarr store
is written under ``--data-dir``.

Usage::

    uv run python scripts/ingest_binance_spot.py --help
    uv run python scripts/ingest_binance_spot.py
    uv run python scripts/ingest_binance_spot.py --symbols BTCUSDT,ETHUSDT
    uv run python scripts/ingest_binance_spot.py \\
        --raw-data-dir ~/Downloads/spot/monthly/klines
    uv run python scripts/ingest_binance_spot.py --data-dir /Volumes/BigDisk \\
        --raw-data-dir ~/Downloads/spot/monthly/klines
"""

import argparse

from quantlab.base.config import DatasetConfig
from quantlab.config import get_data_root
from quantlab.dataset.spot import SpotKlineDataset
from quantlab.utils.cli import add_data_dir_arg, apply_data_dir


def _build_dataset_config(args: argparse.Namespace) -> DatasetConfig:
    """Build the ``DatasetConfig`` for the parsed arguments.

    Raw CSVs are read from
    ``downloads/crypto_spot/1d/spot/monthly/klines`` and the panel is stored
    at ``data/crypto_spot/1d/klines.zarr``, both under the storage root.
    ``--raw-data-dir`` replaces the CSV directory afterwards, which is what
    lets it combine with a root already moved by ``--data-dir``.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    DatasetConfig
        The config for ``SpotKlineDataset``.
    """
    symbols = args.symbols.split(",") if args.symbols else None
    root = get_data_root()
    config = DatasetConfig(
        raw_data_dir_path=str(
            root / "downloads" / "crypto_spot" / "1d" / "spot" / "monthly" / "klines"
        ),
        zarr_file_path=str(root / "data" / "crypto_spot" / "1d" / "klines.zarr"),
        catalog_path=str(root / "data" / "catalog"),
        market="crypto_spot",
        frequency="1d",
        start_date=args.start_date,
        end_date=args.end_date,
        symbols=symbols,
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
            "Directory holding the Binance kline CSVs, replacing the default "
            "raw CSV directory under the storage root. Point it at wherever "
            "the CSVs already live; they need not be moved, copied or linked. "
            "Unlike --data-dir it moves only the raw CSV directory, so with "
            "both flags CSVs are read from here and the Zarr store is written "
            "under the --data-dir root."
        ),
    )
    add_data_dir_arg(parser)
    return parser


if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Must run before the config is built: ``_build_dataset_config`` copies
    # the data root into its paths, so a later override is ignored. ``--raw-data-dir`` is applied after the config is built.
    apply_data_dir(args)

    config = _build_dataset_config(args)
    SpotKlineDataset(config).from_raw_data().save()
