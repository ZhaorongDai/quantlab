"""Download daily US-equity bars from Tiingo into raw parquet.

The script resolves a symbol roster, checks the request against the
pre-flight volume guard (an estimate of rows, bytes, requests and run time
that refuses a download above its ceilings before any request is sent) and
then downloads through ``quantlab.registry.run``. The roster is either an
explicit ``--symbols`` list or a ``--universe`` category resolved
point-in-time, meaning the index members as they stood on ``--as-of-date``
rather than today. Raw parquet under the configured ``raw_data_dir_path`` is
the default output. Each symbol keeps a watermark, a small sidecar file
recording the last date already downloaded, so ``--refresh`` can continue
from it.

``--to-zarr`` also converts the raw files into a Zarr store (a chunked
on-disk array format that ``xarray`` reads) through
``quantlab.registry.convert``, one ``--chunk`` window at a time, resuming at
the first unwritten window. Nothing checks that a window fits in memory, so
pick a finer ``--chunk`` for a large roster. No vendor class is named here;
every vendor fact is read off the registry descriptor ``SOURCE``.

``TIINGO_API_KEY`` must be set in the environment. The key is never printed
or logged; only symbol lists, date ranges and paths are.

Usage::

    uv run python scripts/ingest_tiingo.py --help
    export TIINGO_API_KEY=your-key-here

    # Raw parquet only (the default).
    uv run python scripts/ingest_tiingo.py --symbols AAPL,MSFT
    uv run python scripts/ingest_tiingo.py --symbols AAPL \\
        --start-date 2024-01-01 --end-date 2024-12-31
    uv run python scripts/ingest_tiingo.py --symbols AAPL,MSFT --refresh

    # Raw parquet and the Zarr store.
    uv run python scripts/ingest_tiingo.py --symbols AAPL,MSFT --to-zarr

    # A roster from the point-in-time universe table (build it first with
    # scripts/refresh_us_equity_universe.py). The roster is sorted, so a
    # second run with the same flags meets the watermarks the first wrote.
    uv run python scripts/ingest_tiingo.py --universe sp500 \\
        --as-of-date 2015-06-01
    uv run python scripts/ingest_tiingo.py --universe nasdaq100 \\
        --as-of-date 2015-06-01
    uv run python scripts/ingest_tiingo.py --universe us_all \\
        --as-of-date 2020-01-01 --to-zarr
"""

import argparse

from dataclasses import replace

from quantlab.registry import DataSourceRegistry, convert, run
from quantlab.universe import UniverseCatalog
from quantlab.base.config import AcquisitionConfig, DatasetConfig
from quantlab.config import stock_kline_config, universe_config
from quantlab.dataset.stock import StockDataset
from quantlab.utils.cli import (
    add_data_dir_arg,
    add_to_zarr_arg,
    add_universe_args,
    add_volume_guard_args,
    add_window_args,
    add_chunk_args,
    apply_data_dir,
    print_conversion_result,
    print_volume_estimate,
    refuse_conversion_without_raw_data,
    resolve_symbols,
    validate_roster_args,
    volume_pricing,
)

#: The registered Tiingo source. The vendor is named here once, as a registry
#: token; the config factory, the batch size and the fetch itself are all read
#: off this descriptor rather than off a vendor class.
SOURCE = DataSourceRegistry.get("tiingo")


def _build_configs(
    args: argparse.Namespace,
    catalog=None,
) -> tuple[AcquisitionConfig, DatasetConfig]:
    """Build the acquisition and dataset configs for the parsed arguments.

    The universe catalog is loaded only when a ``--universe`` category has to
    be resolved, so an explicit ``--symbols`` list works on a machine that has
    never built the reference table. ``__main__`` passes the catalog it has
    already loaded so the table is read once.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    catalog : UniverseCatalog, optional
        An already-loaded universe table, or ``None`` (the default) to load
        one on demand when a category is requested.

    Returns
    -------
    tuple[AcquisitionConfig, DatasetConfig]
        An ``(acquisition_config, dataset_config)`` pair for the same roster
        and date window.
    """
    if catalog is None and args.universe:
        catalog = UniverseCatalog.load(universe_config())
    # Membership is taken as of one day. ``ingest_us_equity.py`` asks the
    # same helper for every symbol listed at any time in the window instead.
    symbols = resolve_symbols(args, catalog, mode="as_of")

    # ``SOURCE.config_factory`` already has the vendor bound, so it is not
    # restated here.
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
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Download daily US-equity bars from Tiingo into raw parquet, and "
            "with --to-zarr also persist them as an xarray Dataset in Zarr. "
            "Requires TIINGO_API_KEY."
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
            "Continue each symbol from its last recorded watermark instead "
            "of downloading the whole window again."
        ),
    )
    # The conversion flags are shared with the other ingest scripts because
    # they all drive the same chunked conversion.
    add_to_zarr_arg(parser)
    add_chunk_args(parser)
    return parser


if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Must run before any config factory is called: the factories copy the
    # data root into their paths when called, so a later override is ignored.
    apply_data_dir(args)

    validate_roster_args(parser, args)

    catalog = UniverseCatalog.load(universe_config()) if args.universe else None
    acq_config, ds_config = _build_configs(args, catalog)

    # Pre-flight: no client exists and no request has been sent yet.
    pricing, category, guard_start, guard_end, window_assumed = volume_pricing(
        args, catalog, symbols=acq_config.symbols
    )
    print_volume_estimate(
        pricing.assert_acquisition_volume_fits(
            category,
            guard_start,
            guard_end,
            frequency="1d",
            # Tiingo serves one symbol per request, so the guard is told the
            # source's batch size of 1; a larger value would understate the
            # request count by that factor.
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

    if args.to_zarr:
        # This dataset only answers ``has_raw_data()``, so it needs no symbols.
        refuse_conversion_without_raw_data(
            StockDataset(replace(ds_config, symbols=None)), result
        )
        print(
            f"Converting/persisting symbols={ds_config.symbols} to Zarr in "
            f"{args.chunk} windows (resumable; completed windows are skipped)"
        )
        # Print the registry's own result, so what is printed is what was
        # written.
        conversion = convert(
            SOURCE,
            ds_config,
            granularity=args.chunk,
            on_new_listing=args.on_new_listing,
        )
        print_conversion_result(conversion)
    else:
        print(
            "Skipping Zarr conversion (default). The raw files above are the "
            "run's output; pass --to-zarr to convert them."
        )
