"""Pull US-equities daily data from Tiingo and persist it as xr.Dataset/Zarr.

Full Tiingo-to-Zarr pipeline: fetches raw EOD data via TiingoAcquisition
(writing raw parquet files under the configured raw_data_dir_path), then
converts/cleans/persists it through StockDataset into a Zarr store (D-02
market/frequency convention, see config/__init__.py:stock_kline_config()).

Requires the TIINGO_API_KEY environment variable to be set -- get your key
from the Tiingo dashboard (https://api.tiingo.com/). This script never
prints or logs the key value itself, or the TiingoClient config dict; only
symbol lists and date ranges are ever logged/printed.

Usage:
    export TIINGO_API_KEY=your-key-here
    uv run python ingest_tiingo.py --symbols AAPL,MSFT
    uv run python ingest_tiingo.py --symbols AAPL --start-date 2024-01-01 --end-date 2024-12-31
    uv run python ingest_tiingo.py --symbols AAPL,MSFT --refresh

Or resolve a symbol list from the point-in-time US-equity universe table
(02-08-PLAN.md; build/refresh it first via `refresh_us_equity_universe.py`)
instead of passing --symbols explicitly:
    uv run python ingest_tiingo.py --universe sp500 --as-of-date 2015-06-01
    uv run python ingest_tiingo.py --universe nasdaq100 --as-of-date 2015-06-01
    uv run python ingest_tiingo.py --universe nasdaq_all --as-of-date 2020-01-01
    uv run python ingest_tiingo.py --universe us_all --as-of-date 2020-01-01
"""

import argparse

from acquisition.tiingo import TiingoAcquisition
from acquisition.universe import UniverseCatalog
from base.config import AcquisitionConfig, DatasetConfig
from config import stock_acquisition_config, stock_kline_config, universe_config
from dataset.stock import StockDataset
from utils.cli import (
    add_universe_args,
    add_window_args,
    resolve_symbols,
    validate_roster_args,
)


def _build_configs(
    args: argparse.Namespace,
) -> tuple[AcquisitionConfig, DatasetConfig]:
    # The catalog is loaded ONLY when a universe category has to be resolved:
    # an explicit --symbols list needs no reference table, and loading one
    # would make this script fail on a machine that has never built it.
    catalog = UniverseCatalog.load(universe_config()) if args.universe else None
    # `mode="as_of"` is stated, never defaulted: this script resolves
    # point-in-time membership on ONE day. `ingest_us_equity.py` deliberately
    # asks the same helper for `"in_range"` instead.
    symbols = resolve_symbols(args, catalog, mode="as_of")

    acq_config = stock_acquisition_config(
        symbols=symbols,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    ds_config = stock_kline_config(
        symbols=list(symbols),
        start_date=args.start_date,
        end_date=args.end_date,
    )
    return acq_config, ds_config


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Pull US-equities daily data from Tiingo and persist it as "
            "xr.Dataset/Zarr. Requires TIINGO_API_KEY."
        )
    )
    add_universe_args(parser)
    add_window_args(parser)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help=(
            "Incrementally refresh from each symbol's last recorded "
            "watermark instead of a full download() backfill."
        ),
    )
    return parser


if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()

    validate_roster_args(parser, args)

    acq_config, ds_config = _build_configs(args)

    print(f"Acquiring symbols={acq_config.symbols} via Tiingo (refresh={args.refresh})")
    acquisition = TiingoAcquisition(acq_config)
    if args.refresh:
        acquisition.refresh()
    else:
        acquisition.download()

    print(f"Converting/persisting symbols={ds_config.symbols} to Zarr")
    StockDataset(ds_config).from_raw_data().save()
