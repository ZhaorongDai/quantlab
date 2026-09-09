"""Pull US-equities daily data from Tiingo into raw parquet, and optionally
convert it to xr.Dataset/Zarr.

Fetches raw EOD data through the data-source registry, writing raw parquet
files under the configured raw_data_dir_path, and STOPS THERE unless
`--to-zarr` is passed. With the flag it goes on to convert/clean/persist the
raw shards through StockDataset into a Zarr store (D-02 market/frequency
convention, see quantlab/config/__init__.py:stock_kline_config()).

`--to-zarr` is OFF by default, and that default CHANGED (G-03.4-1b): this
script used to convert unconditionally, which meant a run that fetched nothing
walked into the conversion anyway and ended on StockDataset's absent-root
ValueError traceback. All three ingest shells now agree -- raw is the default
deliverable, conversion is asked for -- and the default path prints that it
skipped the conversion rather than saying nothing.

This script names no vendor class anywhere: it resolves its source from
`DataSourceRegistry`, reads every vendor constant off `SOURCE.acquisition_cls`,
builds its acquisition config through `SOURCE.config_factory` and downloads
through `registry.run()` (03.4 D-15 / SC-1 / SC-6).

Requires the TIINGO_API_KEY environment variable to be set -- get your key
from the Tiingo dashboard (https://api.tiingo.com/). This script never
prints or logs the key value itself, or the TiingoClient config dict; only
symbol lists and date ranges are ever logged/printed.

Usage:
    export TIINGO_API_KEY=your-key-here

    # Raw parquet only -- the default.
    uv run python ingest_tiingo.py --symbols AAPL,MSFT
    uv run python ingest_tiingo.py --symbols AAPL --start-date 2024-01-01 --end-date 2024-12-31
    uv run python ingest_tiingo.py --symbols AAPL,MSFT --refresh

    # Raw parquet AND the Zarr store.
    uv run python ingest_tiingo.py --symbols AAPL,MSFT --to-zarr

Or resolve a symbol list from the point-in-time US-equity universe table
(02-08-PLAN.md; build/refresh it first via `refresh_us_equity_universe.py`)
instead of passing --symbols explicitly. `--limit N` takes the first N of that
roster in ASCENDING symbol order, so the same pair of flags resolves the same
N symbols on every run and a second run resumes where the first stopped:
    uv run python ingest_tiingo.py --universe sp500 --as-of-date 2015-06-01
    uv run python ingest_tiingo.py --universe nasdaq100 --as-of-date 2015-06-01
    uv run python ingest_tiingo.py --universe nasdaq_all --as-of-date 2020-01-01
    uv run python ingest_tiingo.py --universe us_all --as-of-date 2020-01-01 --to-zarr
"""

import argparse

from quantlab.acquisition.registry import DataSourceRegistry, run
from quantlab.acquisition.universe import UniverseCatalog
from quantlab.base.config import AcquisitionConfig, DatasetConfig
from quantlab.config import stock_kline_config, universe_config
from quantlab.dataset.stock import StockDataset
from quantlab.utils.cli import (
    add_data_dir_arg,
    add_to_zarr_arg,
    add_universe_args,
    add_volume_guard_args,
    add_window_args,
    apply_data_dir,
    print_volume_estimate,
    refuse_conversion_without_raw_data,
    resolve_symbols,
    validate_roster_args,
    volume_pricing,
)

#: The ONE place this script's vendor is named, and it is a TOKEN, not a class.
#:
#: SC-1's "no vendor named at the call site" means no vendor CLASS: a script
#: called `ingest_tiingo.py` has its vendor as its whole identity, and the
#: alternative -- a `--source` flag -- is the merged CLI D-15 explicitly
#: forbids. Every vendor-specific fact below is now read off the descriptor.
SOURCE = DataSourceRegistry.get("tiingo")


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

    # `SOURCE.config_factory` is `functools.partial(stock_acquisition_config,
    # vendor="tiingo")` -- the SAME factory this script called directly before,
    # with the vendor pinned by the DESCRIPTOR instead of inherited from the
    # factory's incumbent default. The direct call produced an identical config
    # today only because "tiingo" happens to be that default: the vendor was
    # never actually routed, so the descriptor's `config_factory` was dead
    # weight in the one script that was supposed to demonstrate it (03.4-06,
    # D-15/SC-6). All three shells now build their acquisition config the same
    # way, which is what `tests/test_ingest_shells.py::
    # test_each_shell_resolves_its_source_through_the_registry` can assert
    # uniformly rather than exempting one.
    acq_config = SOURCE.config_factory(
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
            "Pull US-equities daily data from Tiingo into raw parquet, and "
            "with --to-zarr also persist it as xr.Dataset/Zarr. Requires "
            "TIINGO_API_KEY."
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
    # This script used to convert UNCONDITIONALLY, which is why the flag is a
    # deliberate default change and not a new capability: all three ingest
    # shells now stop at raw unless asked (G-03.4-1b). `mode="whole-window"`
    # because there is no chunking here -- `from_raw_data()` densifies the
    # entire range at once, which is what the dense-panel guard sizes.
    add_to_zarr_arg(parser, mode="whole-window")
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
            # Tiingo's EOD endpoint is ONE symbol per request -- there is
            # no multi-symbol batch to amortise over -- so the volume guard is
            # told a batch size of 1. Telling it anything larger would
            # understate the request count by exactly that factor, which is
            # the number the request ceiling is denominated in. Read off the
            # descriptor's acquisition class rather than restated here, so the
            # guard and the fetcher cannot disagree about the batch size.
            batch_size=SOURCE.acquisition_cls.DEFAULT_BATCH_SIZE,
            rows_per_symbol_day=args.rows_per_symbol_day,
            force=args.force_volume,
        ),
        category=category,
        start_date=guard_start,
        end_date=guard_end,
        window_assumed=window_assumed,
        forced=args.force_volume,
    )

    if args.to_zarr:
        # The RAM sibling of the guard above, and the reason it is a SIBLING:
        # that one bounds raw disk bytes, request count and wall clock; this
        # one bounds the dense `[timestamp, symbol]` grid that the
        # `from_raw_data()` at the bottom of this script materialises through
        # `.to_pandas().set_index([...]).to_xarray()`. `ingest_us_equity.py`
        # already carries the chunked form of this for its `--to-zarr` path;
        # this door densifies the WHOLE window with no chunking at all, so the
        # whole-window form is the one that applies here (CR-03).
        #
        # Conditioned on `--to-zarr` because it measures the RAM of a
        # densification that no longer always happens: refusing a raw-only
        # fetch on the size of a panel this run will never build would be a
        # fresh defect introduced by the fix, not a guard doing its job. Same
        # shape as `ingest_us_equity.py`, which already wraps
        # `assert_chunked_panel_fits` in `if args.to_zarr:`.
        #
        # The POSITION is unchanged -- still before `run(...)`, so the refusal
        # arrives before the fetch rather than after it, and the two AST
        # ordering assertions in `tests/test_volume_guard.py` still read a
        # guard line number below every densify line number.
        pricing.assert_dense_panel_fits(category, guard_start, guard_end)

    print(
        f"Acquiring symbols={acq_config.symbols} via "
        f"{SOURCE.display_name} (refresh={args.refresh})"
    )
    result = run(SOURCE, acq_config, refresh=args.refresh)
    print(
        f"{len(result.succeeded)} symbol(s) succeeded, "
        f"{len(result.failures)} failed"
    )

    print(f"Raw data written under: {acq_config.raw_data_dir_path}")

    # STAYS in the shell: `run()` is acquisition-only (D-14 amendment), and
    # this door densifies the WHOLE window -- exactly the mode
    # `assert_dense_panel_fits` above was sized for.
    if args.to_zarr:
        dataset = StockDataset(ds_config)
        # Before the densification, never after: without this, a run whose
        # every symbol failed reached `from_raw_data()` and ended on
        # `quantlab/dataset/stock.py`'s absent-root ValueError traceback --
        # which reads like a conversion bug rather than "the vendor returned
        # nothing" (G-03.4-1a).
        refuse_conversion_without_raw_data(dataset, result)
        print(f"Converting/persisting symbols={ds_config.symbols} to Zarr")
        dataset.from_raw_data().save()
        print(f"Zarr store written at: {ds_config.zarr_file_path}")
    else:
        # Said out loud rather than left as an absence: a conversion that
        # silently did not happen is the same silence this flag exists to end.
        print(
            "Skipping Zarr conversion (default). The raw shards above are the "
            "deliverable; pass --to-zarr to convert them."
        )
