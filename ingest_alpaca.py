"""Pull US-equities data from Alpaca Market Data and land it under the D-11
vendor-namespaced raw path.

The second source, not a replacement (D-10). Alpaca and Tiingo coexist as
parallel alternatives selected BY CONFIG: this script builds every path through
`quantlab/config/__init__.py`'s factories with `vendor="alpaca"`, so the vendor segment
is DERIVED in one place rather than assembled at the call site. Constructing an
`AcquisitionConfig` or `DatasetConfig` inline here is how the path convention
drifts back into a silent two-vendor merge, and a test asserts this module
calls neither by name.

Three data types, one flag each: `--frequency 1d` and `--frequency 1m` fetch
bars; `--frequency tick` fetches `--data-type quotes` or `--data-type trades`
at FULL resolution, with no resampling or bucketing anywhere (D-16).

Credentials
-----------
`APCA_API_KEY_ID` and `APCA_API_SECRET_KEY` are read from the ENVIRONMENT by
`_AlpacaMarketDataClient.__init__`. They are market-data credentials only --
no trading API, no broker API, and no paper/live distinction, because the
market-data API does not have one (D-15).

This script accepts NEITHER value as a command-line argument and never prints
or logs either one: a CLI credential lands in shell history and in every
process listing, and a credential on a config dataclass lands in persisted
configs and in the JSON saved beside model checkpoints, because
`AcquisitionConfig.to_dict()` is `asdict(self)`. This repo has already leaked
one real vendor key exactly that way. Only symbol lists, date ranges and paths
are ever printed.

The feed question is OPEN
------------------------
`feed` (SIP vs IEX) is a `config.kwargs` parameter with NO in-code default in
either direction, and this script registers no `--feed` flag that could supply
one by habit. Alpaca's own documentation self-conflicts about whether the free
(Basic) tier reaches historical SIP data at all; see
`quantlab/acquisition/alpaca.py:AlpacaAcquisition` for both readings. Until a real
one-request probe settles it (03.2-07 human-check A), an unset feed is OMITTED
from the request and the vendor picks the best feed the account allows.

Vendor knobs with no flag
-------------------------
`feed`, `adjustment`, `asof` and `page_limit` are read from `config.kwargs`
rather than from argparse, so the CLI surface stays small and a knob is added
without a code change here. To set one, call `_build_configs`'s factories
directly, e.g.

    from quantlab.config import stock_acquisition_config
    cfg = stock_acquisition_config(
        symbols=("AAPL",), start_date="2024-01-02", end_date="2024-01-02",
        vendor="alpaca", kwargs={"batch_size": 200, "feed": "sip"},
    )

`--batch-size` is the one exception, because the pre-flight volume guard has to
be told the batch size it is pricing.

Two pre-flight guards, and neither replaces the other
---------------------------------------------------
`assert_acquisition_volume_fits` bounds raw disk bytes, request count and wall
clock. `assert_dense_panel_fits` bounds the RAM of the dense
`[timestamp, symbol]` panel the `1d`/`1m` conversion at the bottom of this
script builds. Both run BEFORE the client is constructed and before a single
request. A `1m` window that comfortably passes the first can be three orders of
magnitude over the second, so the second is sized with
`bars_per_day=390` rather than at its daily default (CR-03). Tick skips it: the
conversion it guards does not run for tick (D-18).

No migration (D-13)
-------------------
Existing Tiingo raw data and watermarks are NOT migrated to the vendor-
namespaced path and are NOT read through a fallback. Re-fetching them is a
SEPARATE, OPTIONAL `ingest_tiingo.py` / `ingest_us_equity.py` invocation that
respects the quota-abort path and resumes from each symbol's watermark --
nothing in this phase's verification depends on it completing. Plan it as a
resumable run rather than fire-and-forget: the Tiingo account's allocation was
observed to exhaust after ~4,600 requests, so a full roster needs several
windows.

Usage:
    export APCA_API_KEY_ID=your-key-id APCA_API_SECRET_KEY=your-secret

    # Daily bars for an explicit symbol list.
    uv run python ingest_alpaca.py --symbols AAPL,MSFT \
        --start-date 2024-01-01 --end-date 2024-12-31

    # Daily bars for a point-in-time roster.
    uv run python ingest_alpaca.py --universe sp500 --as-of-date 2024-01-02 \
        --start-date 2024-01-01 --end-date 2024-12-31

    # Minute bars.
    uv run python ingest_alpaca.py --symbols AAPL --frequency 1m \
        --start-date 2024-01-02 --end-date 2024-01-31

    # Quotes at full resolution. --data-type is REQUIRED here and rejected
    # everywhere else; --rows-per-symbol-day is what the volume guard prices
    # tick with, and it has no default because tick volume is not derivable
    # from a calendar.
    uv run python ingest_alpaca.py --symbols AAPL --frequency tick \
        --data-type quotes --rows-per-symbol-day 1000000 \
        --start-date 2024-01-02 --end-date 2024-01-02

    # Top up an existing backfill from each symbol's own watermark.
    uv run python ingest_alpaca.py --symbols AAPL,MSFT --refresh
"""

import argparse
import typing

from quantlab.acquisition.alpaca import AlpacaAcquisition
from quantlab.acquisition.universe import UniverseCatalog
from quantlab.base.config import AcquisitionConfig, DatasetConfig
from quantlab.config import stock_acquisition_config, stock_kline_config, universe_config
from quantlab.dataset.stock import StockDataset
from quantlab.enums.data import Frequency
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

#: The frequencies this script offers, DERIVED from the locked `Frequency`
#: literal rather than restated, so a frequency added to `quantlab/enums/data.py`
#: becomes selectable here without a second edit -- the same reason the
#: `--universe` choices are derived from `UNIVERSE_CATEGORY_MAP`.
FREQUENCIES: tuple[str, ...] = typing.get_args(Frequency)

#: The Zarr store this script's `1d`/`1m` conversions write to.
#:
#: DELIBERATELY not `stock_kline_config`'s `"stock.zarr"` default. D-11 puts the
#: vendor segment on `raw_data_dir_path` and `watermark_path`, which is what
#: keeps two vendors' RAW files apart -- but `zarr_file_path` has no vendor
#: segment, so sharing the default store name would let an Alpaca conversion
#: overwrite the Tiingo store in place. That is the same silent cross-vendor
#: merge D-11 exists to prevent, one layer up, and `ingest_us_equity.py` already
#: solved the identical problem for its second ROSTER the identical way.
DEFAULT_STORE_NAME = "stock_alpaca.zarr"


def _build_configs(
    args: argparse.Namespace,
    catalog=None,
) -> tuple[AcquisitionConfig, DatasetConfig]:
    """`(AcquisitionConfig, DatasetConfig)` from the `quantlab/config/` factories.

    The factories are where the D-11 vendor segment is derived and where
    `watermark_path` is placed as a SIBLING of the raw root rather than inside
    it; bypassing them with an inline dataclass would put both facts at this
    call site, where the next script would get them subtly wrong.
    """
    # The catalog is loaded ONLY to resolve a universe category; an explicit
    # --symbols list needs no reference table. `catalog` is accepted so
    # `__main__` can load it once and hand the same instance to the volume
    # guard rather than re-reading the parquet table.
    if catalog is None and args.universe:
        catalog = UniverseCatalog.load(universe_config())
    # `mode="as_of"` stated, never defaulted: this script resolves
    # point-in-time membership on ONE day. A full-window backfill wants
    # interval overlap instead -- see `quantlab.utils.cli.resolve_symbols`.
    symbols = resolve_symbols(args, catalog, mode="as_of")

    kwargs: dict = {}
    if args.data_type is not None:
        kwargs["data_type"] = args.data_type
    if args.batch_size is not None:
        kwargs["batch_size"] = args.batch_size

    acq_config = stock_acquisition_config(
        symbols=symbols,
        start_date=args.start_date,
        end_date=args.end_date,
        frequency=args.frequency,
        vendor="alpaca",
        kwargs=kwargs,
    )
    # Built for every frequency, but PERSISTED only for `1d`/`1m`: under
    # `tick` it merely names where 03.3 will land the conversion (D-18).
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
    parser = argparse.ArgumentParser(
        description=(
            "Pull US-equities bars, quotes or trades from Alpaca Market Data. "
            "Requires APCA_API_KEY_ID and APCA_API_SECRET_KEY in the "
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
            "trades, selected by --data-type. Tick rows are landed at FULL "
            "resolution with no resampling (D-16), and the raw-to-Zarr "
            "conversion for them is deferred to phase 03.3 (D-18) -- a tick "
            "run stops after the raw files land."
        ),
    )
    parser.add_argument(
        "--data-type",
        type=str,
        choices=list(AlpacaAcquisition.TICK_DATA_TYPES),
        default=None,
        help=(
            "REQUIRED with --frequency tick and rejected otherwise. Quotes and "
            "trades land under the same vendor root, distinguished only by the "
            "leading `data_type=` hive key, so there is deliberately no "
            "default: guessing would file one as the other with the other's "
            "column projection applied."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=(
            "Symbols per request. Defaults to "
            f"AlpacaAcquisition.DEFAULT_BATCH_SIZE ({AlpacaAcquisition.DEFAULT_BATCH_SIZE}), "
            "which is a conservative working value rather than a verified "
            "vendor ceiling -- the real limit is undocumented. Passed through "
            "config.kwargs, and also handed to the pre-flight volume guard, "
            "which prices requests partly by the batch floor."
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
    return parser


def _validate_data_type(parser: argparse.ArgumentParser, args) -> None:
    """`--data-type` is required for tick and rejected otherwise.

    Rejected rather than ignored: a `--data-type trades` silently dropped
    because the run was `--frequency 1d` would let a user believe they had
    fetched trades.
    """
    if args.frequency == "tick" and args.data_type is None:
        parser.error(
            "--data-type is required when --frequency is tick "
            f"(one of {list(AlpacaAcquisition.TICK_DATA_TYPES)}); there is no "
            "default, because quotes and trades share a vendor root and are "
            "told apart only by the data_type= hive key."
        )
    if args.frequency != "tick" and args.data_type is not None:
        parser.error(
            f"--data-type is only valid with --frequency tick, got "
            f"--frequency {args.frequency}."
        )


if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Before anything that can reach a `quantlab/config/` factory, and the position is
    # load-bearing: the factories snapshot their paths as strings at
    # construction time, so a root override applied afterwards silently does
    # nothing (DDIR-04).
    apply_data_dir(args)

    validate_roster_args(parser, args)
    _validate_data_type(parser, args)

    catalog = UniverseCatalog.load(universe_config()) if args.universe else None
    acq_config, ds_config = _build_configs(args, catalog)

    # BEFORE the client is constructed and before a single request (D-09).
    # The batch size handed over is the EFFECTIVE one -- what the base class
    # will actually use -- because the request ceiling is priced partly by the
    # per-batch floor, and pricing a different batch size than the run uses
    # would make the estimate describe a fetch nobody is about to issue.
    pricing, category, guard_start, guard_end, window_assumed = volume_pricing(
        args, catalog, symbols=acq_config.symbols
    )
    print_volume_estimate(
        pricing.assert_acquisition_volume_fits(
            category,
            guard_start,
            guard_end,
            frequency=args.frequency,
            batch_size=args.batch_size or AlpacaAcquisition.DEFAULT_BATCH_SIZE,
            rows_per_symbol_day=args.rows_per_symbol_day,
            force=args.force_volume,
        ),
        category=category,
        start_date=guard_start,
        end_date=guard_end,
        window_assumed=window_assumed,
        forced=args.force_volume,
    )

    if args.frequency != "tick":
        # A SIBLING of the volume guard above, not a replacement -- its own
        # docstring says so twice. That one bounds raw DISK bytes, request
        # count and wall clock; this one bounds the RAM of the dense
        # `[timestamp, symbol]` panel that `StockDataset.from_raw_data()` at
        # the bottom of this script materialises via
        # `.to_pandas().set_index([...]).to_xarray()`.
        #
        # Without it the guard's own ADMITTED scenario kills the process AFTER
        # a successful fetch: S&P-500 minute for one year passes the volume
        # guard at ~4,900 requests and ~3 GB on disk, and then densifies to
        # ~500 symbols x ~98,000 minute stamps x 7 variables x 8 bytes -- about
        # 4 TB -- against a 4 GiB budget. `ingest_us_equity.py` already carries
        # the chunked form of this guard for daily; the second front door
        # inherited none of it (CR-03).
        #
        # `bars_per_day` is REQUIRED here rather than defaulted: this guard
        # sizes the timestamp axis, and at `1m` a session is 390 rows. Left at
        # 1 it would admit the very fetch it exists to refuse.
        #
        # `num_variables` is Alpaca's own bar width (RAW_COLUMNS minus
        # timestamp/symbol/vendor), not the 12-column Tiingo EOD default -- a
        # guard that overstates refuses fetches that would have been fine,
        # which is how a guard gets deleted.
        pricing.assert_dense_panel_fits(
            category,
            guard_start,
            guard_end,
            num_variables=len(
                AlpacaAcquisition.RAW_COLUMNS_BY_DATA_TYPE["bars"]
            ) - 3,
            bars_per_day=pricing.BARS_PER_DAY_BY_FREQUENCY[args.frequency],
        )

    print(
        f"Acquiring {len(acq_config.symbols)} symbol(s) from Alpaca "
        f"(frequency={args.frequency}, data_type={args.data_type}, "
        f"refresh={args.refresh})"
    )
    acquisition = AlpacaAcquisition(acq_config)
    if args.refresh:
        acquisition.refresh()
    else:
        acquisition.download()
    print(f"Raw data written under: {acq_config.raw_data_dir_path}")

    if args.frequency == "tick":
        # STOPS HERE, deliberately. The dense `[timestamp, symbol]` panel this
        # project's Dataset layer is built on cannot express an irregular event
        # axis, so the tick raw-to-xarray conversion is a genuine second data
        # model and is deferred to phase 03.3 (D-18). Said out loud rather than
        # left as an absence, so a user is told where the conversion lives
        # instead of waiting for a Zarr store that this phase never writes.
        print(
            "Stopping at raw for tick: the quotes/trades raw-to-xarray "
            "conversion needs an irregular event axis the dense "
            "[timestamp, symbol] panel cannot express, and arrives in phase "
            "03.3 (D-18). The raw shards above are the deliverable."
        )
    else:
        print(f"Converting/persisting symbols={ds_config.symbols} to Zarr")
        StockDataset(ds_config).from_raw_data().save()
        print(f"Zarr store written at: {ds_config.zarr_file_path}")
