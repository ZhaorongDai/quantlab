"""Download US-equity bars, quotes or trades from Alpaca Market Data.

The script resolves a symbol roster, checks the request against the
pre-flight volume guard (an estimate of rows, bytes, requests and run time
that refuses a download above its ceilings before any request is sent) and
then downloads through ``quantlab.registry.run``. The roster is either an
explicit ``--symbols`` list or a ``--universe`` category resolved
point-in-time, meaning the index members as they stood on ``--as-of-date``
rather than today. Every run writes raw parquet files and stops there by
default. Each symbol keeps a watermark, a small sidecar file recording the
last date already downloaded, so ``--refresh`` can continue from it.

``--frequency 1d`` and ``--frequency 1m`` fetch daily and minute bars.
``--frequency tick`` fetches ``--data-type quotes`` or ``trades`` at full
resolution with no resampling. ``--to-zarr`` also converts bars into a Zarr
store (a chunked on-disk array format that ``xarray`` reads) through
``quantlab.registry.convert``, one ``--chunk`` window at a time. It is
refused for tick data, because irregular tick events do not fit the dense
``(timestamp, symbol)`` grid of a panel. Nothing checks that a window fits
in memory, and a minute-bar window is far larger than its trading-day count
suggests, so choose ``--chunk`` accordingly.

Alpaca is a second source beside Tiingo. Its raw files land under a
vendor-specific path built by the config factories, so the two vendors'
files never mix, and ``--to-zarr`` writes its own ``stock_alpaca.zarr``
store. No vendor class is named here; every vendor fact is read off the
registry descriptor ``SOURCE``.

``APCA_API_KEY_ID`` and ``APCA_API_SECRET_KEY`` must be set in the
environment. Neither is accepted as an argument or ever printed, because a
credential on the command line lands in shell history and one on a config
object lands in every saved config.

Vendor options such as ``feed``, ``adjustment``, ``asof`` and ``page_limit``
have no flag; they travel in ``config.kwargs``. ``feed`` is left unset by
default so that Alpaca picks the best feed the account allows. To set one,
build the config directly::

    from quantlab.config import stock_acquisition_config
    cfg = stock_acquisition_config(
        symbols=("AAPL",), start_date="2024-01-02", end_date="2024-01-02",
        vendor="alpaca", kwargs={"batch_size": 200, "feed": "sip"},
    )

Usage::

    uv run python scripts/ingest_alpaca.py --help
    export APCA_API_KEY_ID=your-key-id APCA_API_SECRET_KEY=your-secret

    # Daily bars for an explicit symbol list.
    uv run python scripts/ingest_alpaca.py --symbols AAPL,MSFT \\
        --start-date 2024-01-01 --end-date 2024-12-31

    # The same fetch, converted to the Zarr store afterwards.
    uv run python scripts/ingest_alpaca.py --symbols AAPL,MSFT --to-zarr \\
        --start-date 2024-01-01 --end-date 2024-12-31

    # Daily bars for a point-in-time roster.
    uv run python scripts/ingest_alpaca.py --universe sp500 \\
        --as-of-date 2024-01-02 --start-date 2024-01-01 --end-date 2024-12-31

    # Minute bars.
    uv run python scripts/ingest_alpaca.py --symbols AAPL --frequency 1m \\
        --start-date 2024-01-02 --end-date 2024-01-31

    # Quotes at full resolution. --data-type is required here and rejected
    # elsewhere. --rows-per-symbol-day sizes the volume guard and has no
    # default, because tick volume cannot be derived from a calendar.
    uv run python scripts/ingest_alpaca.py --symbols AAPL --frequency tick \\
        --data-type quotes --rows-per-symbol-day 1000000 \\
        --start-date 2024-01-02 --end-date 2024-01-02

    # Top up an existing download from each symbol's own watermark.
    uv run python scripts/ingest_alpaca.py --symbols AAPL,MSFT --refresh
"""

import argparse
import typing

from dataclasses import replace

from quantlab.registry import DataSourceRegistry, convert, run
from quantlab.universe import UniverseCatalog
from quantlab.base.config import AcquisitionConfig, DatasetConfig
from quantlab.config import stock_kline_config, universe_config
from quantlab.dataset.stock import StockDataset
from quantlab.enums.data import Frequency
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

#: The registered Alpaca source. The vendor is named here once, as a registry
#: token; the config factory, the argparse defaults and the fetch itself are
#: all read off this descriptor rather than off a vendor class.
SOURCE = DataSourceRegistry.get("alpaca")

#: The frequencies this script offers, derived from the ``Frequency`` literal
#: so a frequency added there becomes selectable here without a second edit.
FREQUENCIES: tuple[str, ...] = typing.get_args(Frequency)

#: The Zarr store ``--to-zarr`` writes for ``1d`` and ``1m`` bars. It differs
#: from ``stock_kline_config``'s default store name because the store path
#: carries no vendor segment, and sharing the default would let an Alpaca
#: conversion overwrite the Tiingo store in place.
DEFAULT_STORE_NAME = "stock_alpaca.zarr"


def _build_configs(
    args: argparse.Namespace,
    catalog=None,
) -> tuple[AcquisitionConfig, DatasetConfig]:
    """Build the acquisition and dataset configs for the parsed arguments.

    Both configs come from the ``quantlab.config`` factories, which is where
    the vendor segment of the raw path and the placement of the watermark
    directory are derived. The universe catalog is loaded only when a
    ``--universe`` category has to be resolved; ``__main__`` passes the one
    it has already loaded so the table is read once.

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
        An ``(acquisition_config, dataset_config)`` pair. The dataset config
        is built for every frequency but only used for ``1d`` and ``1m``.
    """
    if catalog is None and args.universe:
        catalog = UniverseCatalog.load(universe_config())
    # Membership is taken as of one day. A full-window backfill would want
    # every symbol that was a member at any time in the window instead.
    symbols = resolve_symbols(args, catalog, mode="as_of")

    kwargs: dict = {}
    if args.data_type is not None:
        kwargs["data_type"] = args.data_type
    if args.batch_size is not None:
        kwargs["batch_size"] = args.batch_size

    # ``SOURCE.config_factory`` already has the vendor bound, so it is not
    # restated here.
    acq_config = SOURCE.config_factory(
        symbols=symbols,
        start_date=args.start_date,
        end_date=args.end_date,
        frequency=args.frequency,
        kwargs=kwargs,
    )
    ds_config = stock_kline_config(
        symbols=list(symbols),
        start_date=args.start_date,
        end_date=args.end_date,
        frequency=args.frequency,
        vendor="alpaca",
        store_name=DEFAULT_STORE_NAME,
    )
    return acq_config, ds_config


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Download US-equity bars, quotes or trades from Alpaca Market "
            "Data. Requires APCA_API_KEY_ID and APCA_API_SECRET_KEY in the "
            "environment; neither is accepted as an argument."
        )
    )
    add_universe_args(parser)
    add_window_args(parser)
    add_volume_guard_args(parser)
    add_data_dir_arg(parser)
    parser.add_argument(
        "--frequency",
        type=str,
        choices=list(FREQUENCIES),
        default="1d",
        help=(
            "What to fetch: '1d' and '1m' are bars; 'tick' is raw quotes or "
            "trades, selected by --data-type. Tick rows are landed at full "
            "resolution with no resampling, and there is no raw-to-Zarr "
            "conversion for them: a tick run stops after the raw files land."
        ),
    )
    parser.add_argument(
        "--data-type",
        type=str,
        choices=list(SOURCE.acquisition_cls.TICK_DATA_TYPES),
        default=None,
        help=(
            "Which tick data to fetch. Required with --frequency tick and "
            "rejected otherwise. Quotes and trades are stored under the same "
            "vendor directory and told apart only by their data_type= "
            "subdirectory, so there is no default: a wrong guess would file "
            "one as the other with the wrong columns."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=(
            "Symbols per request (default "
            f"{SOURCE.acquisition_cls.DEFAULT_BATCH_SIZE}). The default is a "
            "conservative working value; Alpaca does not document the real "
            "limit. The value is passed through config.kwargs and also to the "
            "pre-flight volume guard, which counts requests by it."
        ),
    )
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


def _validate_data_type(parser: argparse.ArgumentParser, args) -> None:
    """Reject inconsistent ``--frequency``, ``--data-type`` and ``--to-zarr``.

    ``--data-type`` is required for tick and rejected otherwise, and
    ``--to-zarr`` is rejected for tick. Each case exits through
    ``parser.error`` rather than being ignored, because a silently dropped
    flag would let the user believe a fetch or conversion happened.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser, used to report the error and exit with status 2.
    args : argparse.Namespace
        Parsed command-line arguments.
    """
    if args.frequency == "tick" and args.data_type is None:
        parser.error(
            "--data-type is required when --frequency is tick "
            f"(one of {list(SOURCE.acquisition_cls.TICK_DATA_TYPES)}); there is no "
            "default, because quotes and trades share a vendor directory and "
            "are told apart only by their data_type= subdirectory."
        )
    if args.frequency != "tick" and args.data_type is not None:
        parser.error(
            f"--data-type is only valid with --frequency tick, got "
            f"--frequency {args.frequency}."
        )
    if args.frequency == "tick" and args.to_zarr:
        parser.error(
            "--to-zarr is not available with --frequency tick: converting "
            "quotes/trades to xarray needs an irregular event axis that the "
            "dense [timestamp, symbol] panel cannot express, and that "
            "conversion is not implemented yet. Drop --to-zarr; a tick run's "
            "raw files are its output."
        )


if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Must run before any config factory is called: the factories copy the
    # data root into their paths when called, so a later override is ignored.
    apply_data_dir(args)

    validate_roster_args(parser, args)
    _validate_data_type(parser, args)

    catalog = UniverseCatalog.load(universe_config()) if args.universe else None
    acq_config, ds_config = _build_configs(args, catalog)

    # Pre-flight: no client exists and no request has been sent yet. The
    # guard gets the batch size the run will use, since it counts requests
    # by it.
    pricing, category, guard_start, guard_end, window_assumed = volume_pricing(
        args, catalog, symbols=acq_config.symbols
    )
    print_volume_estimate(
        pricing.assert_acquisition_volume_fits(
            category,
            guard_start,
            guard_end,
            frequency=args.frequency,
            batch_size=args.batch_size or SOURCE.acquisition_cls.DEFAULT_BATCH_SIZE,
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
        f"Acquiring {len(acq_config.symbols)} symbol(s) from "
        f"{SOURCE.display_name} "
        f"(frequency={args.frequency}, data_type={args.data_type}, "
        f"refresh={args.refresh})"
    )
    result = run(SOURCE, acq_config, refresh=args.refresh)
    # These counts cover this run only. The ``_failures.json`` manifest
    # accumulates across runs and may name symbols this run never requested.
    print(
        f"{len(result.succeeded)} symbol(s) succeeded, "
        f"{len(result.failures)} failed"
    )
    print(f"Raw data written under: {acq_config.raw_data_dir_path}")

    if args.frequency == "tick":
        # Say so explicitly, so nobody waits for a store that is never written.
        print(
            "Stopping at raw files for tick data: converting quotes/trades to "
            "xarray needs an irregular event axis that the dense "
            "[timestamp, symbol] panel cannot express, and that conversion is "
            "not implemented yet. The raw files above are the run's output."
        )
    elif args.to_zarr:
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
