"""Pull US-equities daily data from Tiingo and persist it as xr.Dataset/Zarr.

Full Tiingo-to-Zarr pipeline: fetches raw EOD data via TiingoAcquisition
(writing raw parquet files under the configured raw_data_dir_path), then
converts/cleans/persists it through StockDataset into a Zarr store (D-02
market/frequency convention, see quantlab/config/__init__.py:stock_kline_config()).

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

from quantlab.acquisition.tiingo import TiingoAcquisition
from quantlab.acquisition.universe import UniverseCatalog
from quantlab.base.config import AcquisitionConfig, DatasetConfig
from quantlab.config import stock_acquisition_config, stock_kline_config, universe_config
from quantlab.dataset.stock import StockDataset
from quantlab.utils.cli import (
    add_data_dir_arg,
    add_universe_args,
    add_volume_guard_args,
    add_window_args,
    apply_data_dir,
    print_volume_estimate,
    resolve_symbols,
    validate_roster_args,
    volume_pricing,
)

#: Tiingo's EOD endpoint is ONE symbol per request -- there is no multi-symbol
#: batch to amortise over -- so the volume guard is told a batch size of 1.
#: Telling it anything larger would understate the request count by exactly
#: that factor, which is the number the request ceiling is denominated in.
TIINGO_BATCH_SIZE = 1


def _build_configs(
    args: argparse.Namespace,
    catalog=None,
) -> tuple[AcquisitionConfig, DatasetConfig]:
    # The catalog is loaded ONLY when a universe category has to be resolved:
    # an explicit --symbols list needs no reference table, and loading one
    # would make this script fail on a machine that has never built it.
    # `catalog` is accepted so `__main__` can load it ONCE and hand the same
    # instance to the volume guard rather than re-reading the parquet table.
    if catalog is None and args.universe:
        catalog = UniverseCatalog.load(universe_config())
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
    add_volume_guard_args(parser)
    add_data_dir_arg(parser)
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

    # Before anything that can reach a `quantlab/config/` factory, and the position is
    # load-bearing: the factories snapshot their paths as strings at
    # construction time, so a root override applied afterwards silently does
    # nothing (DDIR-04).
    apply_data_dir(args)

    validate_roster_args(parser, args)

    catalog = UniverseCatalog.load(universe_config()) if args.universe else None
    acq_config, ds_config = _build_configs(args, catalog)

    # BEFORE the client is constructed and before a single request (D-09).
    # `_build_configs` above builds paths and resolves a roster; it opens no
    # connection, so this is still the pre-flight position.
    pricing, category, guard_start, guard_end, window_assumed = volume_pricing(
        args, catalog, symbols=acq_config.symbols
    )
    print_volume_estimate(
        pricing.assert_acquisition_volume_fits(
            category,
            guard_start,
            guard_end,
            frequency="1d",
            batch_size=TIINGO_BATCH_SIZE,
            rows_per_symbol_day=args.rows_per_symbol_day,
            force=args.force_volume,
        ),
        category=category,
        start_date=guard_start,
        end_date=guard_end,
        window_assumed=window_assumed,
        forced=args.force_volume,
    )

    # The RAM sibling of the guard above, and the reason it is a SIBLING: that
    # one bounds raw disk bytes, request count and wall clock; this one bounds
    # the dense `[timestamp, symbol]` grid that the unconditional
    # `from_raw_data()` at the bottom of this script materialises through
    # `.to_pandas().set_index([...]).to_xarray()`. `ingest_us_equity.py`
    # already carries the chunked form of this for its `--to-zarr` path; this
    # door densifies the WHOLE window with no chunking at all, so the
    # whole-window form is the one that applies here (CR-03).
    pricing.assert_dense_panel_fits(category, guard_start, guard_end)

    print(f"Acquiring symbols={acq_config.symbols} via Tiingo (refresh={args.refresh})")
    acquisition = TiingoAcquisition(acq_config)
    if args.refresh:
        acquisition.refresh()
    else:
        acquisition.download()

    print(f"Converting/persisting symbols={ds_config.symbols} to Zarr")
    StockDataset(ds_config).from_raw_data().save()
