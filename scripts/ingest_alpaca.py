"""Pull US-equities data from Alpaca Market Data and land it under the D-11
vendor-namespaced raw path.

The second source, not a replacement (D-10). Alpaca and Tiingo coexist as
parallel alternatives selected BY CONFIG: this script builds every path through
`quantlab/config/__init__.py`'s factories with the vendor pinned by the
registered source descriptor, so the vendor segment is DERIVED in one place
rather than assembled at the call site. Constructing an `AcquisitionConfig` or
`DatasetConfig` inline here is how the path convention drifts back into a
silent two-vendor merge, and a test asserts this module calls neither by name.

This script names no vendor class anywhere: it resolves its source from
`DataSourceRegistry`, reads every vendor constant off `SOURCE.acquisition_cls`,
builds its acquisition config through `SOURCE.config_factory` and downloads
through `registry.run()` (03.4 D-15 / SC-1 / SC-6).

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

One pre-flight guard, and it does not bound memory
-------------------------------------------------
`assert_acquisition_volume_fits` bounds raw disk bytes, request count and wall
clock, and runs on EVERY run, BEFORE the client is constructed and before a
single request. It is now the ONLY pre-flight guard: phase 03.6 deleted the
dense-panel RAM guard that used to sit beside it, by decision (SC-3). An
over-sized conversion window therefore reaches OOM rather than a legible
refusal naming a finer `--chunk`, and a `1m` window that comfortably passes
the volume guard can still be three orders of magnitude too large to densify
-- the timestamp axis, not the trading-day count, is what a densifier
allocates against. Choose `--chunk` accordingly.

The conversion itself is CHUNKED and it is the REGISTRY'S (03.5
D-06/D-07/SC-6): this script hands `quantlab.acquisition.registry.convert()` a
source descriptor and a dataset config and renders the `ConversionResult` it
gets back, naming no Dataset subclass method. One window is densified and
appended at a time onto a symbol axis pinned once over the whole range, so
peak RAM scales with the window rather than the range; `--chunk` selects the
granularity, `--on-new-listing` says what to do about a symbol that first
appears mid-range, and an interrupted run resumes at its first unwritten
window.

The dense guard is CONDITIONAL for one reason: it measures the RAM of a
densification. Refusing a raw-only fetch because a panel this run will never
build would not fit is a defect, not a guard. Tick skips it because its
conversion does not exist at all (D-18); a run without `--to-zarr` skips it
because its conversion was not asked for.

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

    Every command below lands RAW parquet and stops there. Add --to-zarr to
    any of the bar commands to convert afterwards; it is not a default and it
    is refused outright under --frequency tick.

    # Daily bars for an explicit symbol list.
    uv run python ingest_alpaca.py --symbols AAPL,MSFT \
        --start-date 2024-01-01 --end-date 2024-12-31

    # The same fetch, converted to the Zarr store afterwards.
    uv run python ingest_alpaca.py --symbols AAPL,MSFT --to-zarr \
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
    # from a calendar. Adding --to-zarr here exits 2: there is no tick
    # conversion to opt into (D-18), and refusing beats ignoring.
    uv run python ingest_alpaca.py --symbols AAPL --frequency tick \
        --data-type quotes --rows-per-symbol-day 1000000 \
        --start-date 2024-01-02 --end-date 2024-01-02

    # Top up an existing backfill from each symbol's own watermark.
    uv run python ingest_alpaca.py --symbols AAPL,MSFT --refresh
"""

import argparse
import typing

from dataclasses import replace

from quantlab.acquisition.registry import DataSourceRegistry, convert, run
from quantlab.acquisition.universe import UniverseCatalog
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

#: The ONE place this script's vendor is named, and it is a TOKEN, not a class.
#:
#: SC-1's "no vendor named at the call site" means no vendor CLASS: a script
#: called `ingest_alpaca.py` has its vendor as its whole identity, and the
#: alternative -- a `--source` flag -- is the merged CLI D-15 explicitly
#: forbids. Every vendor-specific fact below is read off this descriptor:
#: `SOURCE.config_factory` builds the acquisition config with the vendor
#: pinned, `SOURCE.acquisition_cls` supplies the argparse defaults that used to
#: name the class (L-5), and `run(SOURCE, ...)` performs the fetch.
SOURCE = DataSourceRegistry.get("alpaca")

#: The frequencies this script offers, DERIVED from the locked `Frequency`
#: literal rather than restated, so a frequency added to `quantlab/enums/data.py`
#: becomes selectable here without a second edit -- the same reason the
#: `--universe` choices are derived from `UNIVERSE_CATEGORY_MAP`.
FREQUENCIES: tuple[str, ...] = typing.get_args(Frequency)

#: The Zarr store this script's `1d`/`1m` conversions write to when `--to-zarr`
#: asks for one.
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

    # `SOURCE.config_factory` is `functools.partial(stock_acquisition_config,
    # vendor="alpaca")` -- the SAME factory this script called directly before,
    # with the vendor pinned by the descriptor instead of restated here. That
    # is what makes `vendor=` a fact of the registered source rather than a
    # keyword this call site has to remember to get right.
    acq_config = SOURCE.config_factory(
        symbols=symbols,
        start_date=args.start_date,
        end_date=args.end_date,
        frequency=args.frequency,
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
        choices=list(SOURCE.acquisition_cls.TICK_DATA_TYPES),
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
            "Symbols per request. Defaults to the source's own "
            f"DEFAULT_BATCH_SIZE ({SOURCE.acquisition_cls.DEFAULT_BATCH_SIZE}), "
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
    # This script used to convert UNCONDITIONALLY for 1d/1m, so the flag is a
    # deliberate default change rather than a new capability: all three ingest
    # shells now stop at raw unless asked (G-03.4-1b).
    add_to_zarr_arg(parser)
    # Both flags belong here for the same reason `--to-zarr` does: there is
    # ONE conversion path (D-07), so the knobs that path takes are the same
    # knobs at every door. This is the `--chunk` / `--on-new-listing`
    # divergence disappearing as a CONSEQUENCE of sharing one conversion,
    # not as new surface grown on this script (SC-6).
    add_chunk_args(parser)
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
        # Refused rather than ignored, for the same reason `--data-type` is:
        # a flag silently dropped lets a user believe a conversion happened.
        # There is no tick conversion to opt into -- the dense
        # [timestamp, symbol] panel cannot express an irregular event axis, so
        # the raw-to-xarray step for quotes/trades is a second data model,
        # deferred to phase 03.3 (D-18).
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
    # Reported from the RESULT rather than left to the log lines: a run that
    # failed every symbol still logs plenty. These counts describe THIS run
    # only; `_failures.json` is the wider cross-run record and may name
    # symbols this run never requested, so the two numbers are related by
    # containment, not equality (D-18, REVIEW CR-01).
    print(
        f"{len(result.succeeded)} symbol(s) succeeded, "
        f"{len(result.failures)} failed"
    )
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
    elif args.to_zarr:
        # Probed on a SYMBOL-FREE config. The reason for that is NOT the one
        # this comment used to give: it argued that a non-None symbol list
        # makes `BaseDataset`'s config setter call `_reset_symbols()`, which
        # reads the store, catches a not-yet-written store's
        # `FileNotFoundError` and recovers by densifying the FULL RANGE
        # through `from_raw_data()` at CONSTRUCTION time -- so that
        # `StockDataset(ds_config)` on an empty raw tree raised before the
        # guard placed after it could run. `df7bfe9` deleted `_reset_symbols`
        # outright; the setter now assigns `name`, normalises the two dates
        # and stops, touching no symbol axis and reading no store. NO
        # construction densifies any more, for any config.
        #
        # What survives is the property the probe needs: this dataset exists
        # only to be asked `has_raw_data()`, and `replace(..., symbols=None)`
        # says so AT THE CALL SITE rather than leaving a reader to prove it
        # from the constructor. It is spelled identically in all three shells
        # -- the same `symbols=None` shape `ingest_us_equity.py` states at its
        # own `stock_kline_config` call (G-03.4-1a).
        refuse_conversion_without_raw_data(
            StockDataset(replace(ds_config, symbols=None)), result
        )
        print(
            f"Converting/persisting symbols={ds_config.symbols} to Zarr in "
            f"{args.chunk} windows (resumable; completed windows are skipped)"
        )
        # THE conversion, and it is the registry's -- not a Dataset method
        # called from here (03.5 SC-6). The tick branch ABOVE is unaffected:
        # it is reached first and stops at raw, and the parser-level refusal
        # at `_validate_data_type` fires earlier still. `convert()`'s own
        # absent-`dataset_cls` raise is a THIRD layer, for callers that never
        # touch argparse -- none of the three replaces another.
        conversion = convert(
            SOURCE,
            ds_config,
            granularity=args.chunk,
            on_new_listing=args.on_new_listing,
        )
        # Rendered from the RETURNED object, so what is printed is what was
        # actually written rather than what the config asked for.
        print_conversion_result(conversion)
    else:
        # Said out loud rather than left as an absence, exactly like the tick
        # branch above: a conversion that silently did not happen is the same
        # silence this flag exists to end.
        print(
            "Skipping Zarr conversion (default). The raw shards above are the "
            "deliverable; pass --to-zarr to convert them."
        )
