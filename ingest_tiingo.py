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
    uv run python ingest_tiingo.py --universe nasdaq_all --as-of-date 2020-01-01
"""

import argparse

from acquisition.tiingo import TiingoAcquisition
from acquisition.universe import UniverseCatalog
from base.config import AcquisitionConfig, DatasetConfig
from config import stock_acquisition_config, stock_kline_config, universe_config
from dataset.stock import StockDataset

# Maps the CLI-facing --universe choice to enums.data.UniverseCategory.
_UNIVERSE_CATEGORY_MAP = {"sp500": "sp500_constituent", "nasdaq_all": "nasdaq_all"}


def _build_configs(
    args: argparse.Namespace,
) -> tuple[AcquisitionConfig, DatasetConfig]:
    if args.universe:
        category = _UNIVERSE_CATEGORY_MAP[args.universe]
        catalog = UniverseCatalog.load(universe_config())
        symbols = tuple(catalog.get_symbols_as_of(category, args.as_of_date))
    else:
        symbols = tuple(args.symbols.split(","))

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
    parser.add_argument(
        "--symbols",
        type=str,
        required=False,
        default=None,
        help="Comma-separated symbols (e.g. AAPL,MSFT). Mutually exclusive with --universe.",
    )
    parser.add_argument(
        "--universe",
        type=str,
        choices=["sp500", "nasdaq_all"],
        default=None,
        help=(
            "Resolve a symbol list from the persisted universe table "
            "(02-08-PLAN.md) instead of --symbols. 'sp500' resolves "
            "point-in-time S&P 500 constituent membership; 'nasdaq_all' "
            "resolves the full NASDAQ-listed Common Stock roster (current + "
            "delisted). Requires --as-of-date."
        ),
    )
    parser.add_argument(
        "--as-of-date",
        type=str,
        default=None,
        help="Required with --universe; point-in-time date (YYYY-MM-DD) to resolve membership as of.",
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

    if bool(args.symbols) == bool(args.universe):
        parser.error("Exactly one of --symbols or --universe must be set.")
    if args.universe and not args.as_of_date:
        parser.error("--as-of-date is required when --universe is set.")

    acq_config, ds_config = _build_configs(args)

    print(f"Acquiring symbols={acq_config.symbols} via Tiingo (refresh={args.refresh})")
    acquisition = TiingoAcquisition(acq_config)
    if args.refresh:
        acquisition.refresh()
    else:
        acquisition.download()

    print(f"Converting/persisting symbols={ds_config.symbols} to Zarr")
    StockDataset(ds_config).from_raw_data().save()
