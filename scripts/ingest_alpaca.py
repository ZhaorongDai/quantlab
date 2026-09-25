"""Download US-equity bars, quotes or trades from Alpaca Market Data.

Alpaca is a second source beside Tiingo, not a replacement: raw files land
under a vendor-namespaced path built by the config factories, so the two
vendors' shards never merge. ``--frequency 1d`` and ``--frequency 1m`` fetch
bars; ``--frequency tick`` fetches ``--data-type quotes`` or ``trades`` at
full resolution with no resampling. Every run lands raw parquet and stops
there. ``--to-zarr`` converts bars into the Zarr store through
``quantlab.registry.convert``, one ``--chunk`` window at a time; it is
refused for tick data, which has no dense panel representation. Nothing
checks that a window fits in memory, and a minute-bar window is far larger
than its trading-day count suggests, so choose ``--chunk`` accordingly. No
vendor class is named here; every vendor fact is read off the registry
descriptor ``SOURCE``.

Requires ``APCA_API_KEY_ID`` and ``APCA_API_SECRET_KEY`` in the environment.
Neither is accepted as an argument or ever printed, because a credential on
the command line lands in shell history and one on a config dataclass lands
in every serialised config.

Vendor knobs such as ``feed``, ``adjustment``, ``asof`` and ``page_limit``
have no flag; they travel in ``config.kwargs``. ``feed`` is left unset by
default so the vendor picks the best feed the account allows. To set one,
build the config directly::

    from quantlab.config import stock_acquisition_config
    cfg = stock_acquisition_config(
        symbols=("AAPL",), start_date="2024-01-02", end_date="2024-01-02",
        vendor="alpaca", kwargs={"batch_size": 200, "feed": "sip"},
    )

Usage:
    export APCA_API_KEY_ID=your-key-id APCA_API_SECRET_KEY=your-secret

    # Daily bars for an explicit symbol list.
    uv run python scripts/ingest_alpaca.py --symbols AAPL,MSFT \
        --start-date 2024-01-01 --end-date 2024-12-31

    # The same fetch, converted to the Zarr store afterwards.
    uv run python scripts/ingest_alpaca.py --symbols AAPL,MSFT --to-zarr \
        --start-date 2024-01-01 --end-date 2024-12-31

    # Daily bars for a point-in-time roster.
    uv run python scripts/ingest_alpaca.py --universe sp500 \
        --as-of-date 2024-01-02 --start-date 2024-01-01 --end-date 2024-12-31

    # Minute bars.
    uv run python scripts/ingest_alpaca.py --symbols AAPL --frequency 1m \
        --start-date 2024-01-02 --end-date 2024-01-31

    # Quotes at full resolution. --data-type is required here and rejected
    # elsewhere; --rows-per-symbol-day sizes the volume guard and has no
    # default because tick volume cannot be derived from a calendar.
    uv run python scripts/ingest_alpaca.py --symbols AAPL --frequency tick \
        --data-type quotes --rows-per-symbol-day 1000000 \
        --start-date 2024-01-02 --end-date 2024-01-02

    # Top up an existing backfill from each symbol's own watermark.
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
    catalog
        An already-loaded ``UniverseCatalog``, or ``None`` to load
        one on demand when a category is requested.

    Returns
    -------
    tuple[AcquisitionConfig, DatasetConfig]
        An ``(acquisition_config, dataset_config)`` pair. The dataset config
        is built for every frequency but only used for ``1d`` and ``1m``.
    """
    if catalog is None and args.universe:
        catalog = UniverseCatalog.load(universe_config())
    # Membership is resolved point-in-time on one day; a full-window backfill
    # wants interval overlap instead (see ``resolve_symbols``).
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
            "Required with --frequency tick and rejected otherwise. Quotes "
            "and trades land under the same vendor root, told apart only by "
            "the leading data_type= hive key, so there is deliberately no "
            "default: guessing would file one as the other with the other's "
            "column projection applied."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=(
            "Symbols per request. Defaults to the source's own "
            f"DEFAULT_BATCH_SIZE ({SOURCE.acquisition_cls.DEFAULT_BATCH_SIZE}), "
            "a conservative working value rather than a verified vendor "
            "ceiling; the real limit is undocumented. Passed through "
            "config.kwargs, and also handed to the pre-flight volume guard, "
            "which prices requests partly by the batch size."
        ),
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help=(
            "Incrementally refresh from each symbol's last recorded "
            "watermark instead of a full download() backfill."
        ),
    )
    # The conversion flags are shared with the other ingest scripts because
    # they all drive the same chunked conversion.
    add_to_zarr_arg(parser)
    add_chunk_args(parser)
    return parser


def _validate_data_type(parser: argparse.ArgumentParser, args) -> None:
    """Reject inconsistent ``--frequency`` / ``--data-type`` / ``--to-zarr``.

    ``--data-type`` is required for tick and rejected otherwise, and
    ``--to-zarr`` is rejected for tick. Each case exits through
    ``parser.error`` rather than being ignored, because a silently dropped
    flag would let the user believe a fetch or conversion happened.
    """
    if args.frequency == "tick" and args.data_type is None:
        parser.error(
            "--data-type is required when --frequency is tick "
            f"(one of {list(SOURCE.acquisition_cls.TICK_DATA_TYPES)}); there is no "
            "default, because quotes and trades share a vendor root and are "
            "told apart only by the data_type= hive key."
        )
    if args.frequency != "tick" and args.data_type is not None:
        parser.error(
            f"--data-type is only valid with --frequency tick, got "
            f"--frequency {args.frequency}."
        )
    if args.frequency == "tick" and args.to_zarr:
        # There is no tick conversion: the dense (timestamp, symbol) panel
        # cannot express an irregular event axis.
        parser.error(
            "--to-zarr is not available with --frequency tick: the "
            "quotes/trades raw-to-xarray conversion needs an irregular event "
            "axis the dense [timestamp, symbol] panel cannot express, and "
            "arrives in phase 03.3 (D-18). Drop --to-zarr; a tick run's raw "
            "shards are the deliverable."
        )


if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Must run before any config factory is called: the factories snapshot
    # their paths at construction time, so a later root override is ignored.
    apply_data_dir(args)

    validate_roster_args(parser, args)
    _validate_data_type(parser, args)

    catalog = UniverseCatalog.load(universe_config()) if args.universe else None
    acq_config, ds_config = _build_configs(args, catalog)

    # Pre-flight: no client has been constructed and no request issued yet.
    # The guard is given the batch size the run will actually use, since it
    # prices the request count partly by it.
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
    # These counts describe this run only. The ``_failures.json`` manifest is
    # the cross-run record and may name symbols this run never requested.
    print(
        f"{len(result.succeeded)} symbol(s) succeeded, "
        f"{len(result.failures)} failed"
    )
    print(f"Raw data written under: {acq_config.raw_data_dir_path}")

    if args.frequency == "tick":
        # Tick data stops at raw: there is no conversion for it, and saying
        # so beats leaving the user waiting for a store that is never written.
        print(
            "Stopping at raw for tick: the quotes/trades raw-to-xarray "
            "conversion needs an irregular event axis the dense "
            "[timestamp, symbol] panel cannot express, and arrives in phase "
            "03.3 (D-18). The raw shards above are the deliverable."
        )
    elif args.to_zarr:
        # The probe dataset exists only to answer ``has_raw_data()``;
        # ``symbols=None`` says so at the call site.
        refuse_conversion_without_raw_data(
            StockDataset(replace(ds_config, symbols=None)), result
        )
        print(
            f"Converting/persisting symbols={ds_config.symbols} to Zarr in "
            f"{args.chunk} windows (resumable; completed windows are skipped)"
        )
        # The conversion is the registry's; this script only renders the
        # result it returns, so what is printed is what was written.
        conversion = convert(
            SOURCE,
            ds_config,
            granularity=args.chunk,
            on_new_listing=args.on_new_listing,
        )
        print_conversion_result(conversion)
    else:
        print(
            "Skipping Zarr conversion (default). The raw shards above are the "
            "deliverable; pass --to-zarr to convert them."
        )
