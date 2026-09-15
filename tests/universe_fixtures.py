"""Shared synthetic building blocks for the universe-filter tests (260915-p91).

Plain helpers and classes, not pytest fixtures, imported as
`tests.universe_fixtures`. It is an importable MODULE rather than test-local
code because a config rebuild (`quantlab.utils.module.load_factor_from_config`)
imports the factor class back by its dotted path, and `tests/` has no
`__init__.py`, so a test module is not reliably importable as `tests.<name>`.
`tests/backtest_fixtures.py` exists for the same reason and is reused here.

Everything is synthetic, CPU-only and offline. Configs are constructed
directly, never through the factories in `quantlab/config/__init__.py` (D-32).
"""

from pathlib import Path
from typing import NoReturn

import numpy as np
import pandas as pd
import xarray as xr
from KunQuant.Op import Builder, Input, Output, Rank
from KunQuant.ops import WindowedAvg
from KunQuant.Stage import Function

from quantlab.base.config import DatasetConfig, FactorConfig, MLConfig
from quantlab.base.factor import FactorKunQuant
from quantlab.factor.universe_filter import UniverseFilteredFactor
from quantlab.label.fret import Return

from tests.backtest_fixtures import make_stock_dataset, write_price_store

#: A never-in-universe ticker: five letters ending in `W` is the NASDAQ
#: fifth-character warrant convention, so the STATIC ticker rule removes it in
#: every window regardless of its price or volume.
WARRANT = "JUNKW"

#: RAW close below `min_price` on every bar, while its ADJUSTED close is the
#: highest of the whole panel -- so an unfiltered cross-section ranks it first
#: and an unfiltered backtest buys it. This is the `ZWZZT 0.007 -> 10.05` shape
#: from the objective, in miniature.
PENNY = "PENY"

#: RAW dollar volume (close * volume) far below `min_dollar_volume` on every
#: bar, with a perfectly ordinary price -- only the liquidity rule excludes it.
ILLIQUID = "ILQD"

#: In the universe until `drop_bar`, then RAW close falls below `min_price`
#: while its ADJUSTED close stays the highest among the COMMON symbols. It is
#: the LS-4 drop-out: selectable before the drop, ineligible after it.
DROPOUT = "DRPX"

#: Twelve plain commons plus the four special symbols above. SIXTEEN, because
#: KunQuant requires the symbol count to align with its SIMD block width -- on
#: this aarch64 machine 16 works and 13 does not (see
#: `quantlab/my_ops/preprocess.py:CrossSectionalZScore` 坑 3).
PLAIN_COMMONS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "META",
    "TSLA",
    "AVGO",
    "COST",
    "ADBE",
    "CSCO",
    "QCOM",
    "TXN",
]
SYMBOLS16 = PLAIN_COMMONS + [WARRANT, PENNY, ILLIQUID, DROPOUT]

#: The universe thresholds every fixture and test below is written against.
MIN_PRICE = 5.0
MIN_DOLLAR_VOLUME = 1_000_000.0

#: Adjusted-close levels for the three symbols whose ADJUSTED series must be
#: high enough that an UNFILTERED ranking prefers them. The ordering matters:
#: the penny name outranks everything, the drop-out outranks every common, and
#: the warrant is wild -- so any test that still selects them is reading the
#: adjusted column where it should be reading the raw one.
PENNY_ADJ_CLOSE = 900.0
DROPOUT_ADJ_CLOSE = 700.0
WARRANT_ADJ_CLOSE = 1000.0

FIRST_BAR = "2024-01-01"

#: `RankCloseFactor`'s outputs, in declaration order. Named once so a test can
#: assert against it without reaching into the class.
RANK_CLOSE_FACTOR_NAMES = ("rank_close", "ma_close", "ma_rank")


def bars(n_bars: int = 80) -> pd.DatetimeIndex:
    """The business-day timestamp axis `write_universe_store` writes."""
    return pd.bdate_range(FIRST_BAR, periods=n_bars)


def day(index: int, n_bars: int = 80) -> str:
    """The ISO date of bar `index`, for building config date strings."""
    return bars(n_bars)[index].strftime("%Y-%m-%d")


def write_universe_store(
    root: Path,
    *,
    n_bars: int = 80,
    drop_bar: int = 40,
    seed: int = 0,
) -> DatasetConfig:
    """Write a Tiingo-shaped store with junk/penny/illiquid/drop-out symbols.

    Builds on `tests.backtest_fixtures.write_price_store`, which already emits
    both the adjusted group and a RAW group at 1.7x the adjusted one, then
    overwrites the RAW `close`/`volume` (and the adjusted close) of the four
    special symbols. The adjusted and raw groups are deliberately INCONSISTENT
    for those symbols: the mask reads raw, the factor reads adjusted, so a
    raw-vs-adjusted mix-up in either direction flips results rather than
    silently agreeing.

    The store is rewritten with `mode="w"` BEFORE any dataset is constructed,
    because dataset construction reads it.
    """
    root = Path(root)
    dataset_config = write_price_store(
        root, symbols=SYMBOLS16, n_bars=n_bars, start=FIRST_BAR, seed=seed
    )

    zarr_path = Path(dataset_config.zarr_file_path)
    panel = xr.open_dataset(zarr_path).load()
    panel.close()

    symbols = [str(symbol) for symbol in panel["symbol"].values]

    def column(name: str) -> np.ndarray:
        return panel[name].values

    def put(name: str, symbol: str, values) -> None:
        panel[name].values[:, symbols.index(symbol)] = values

    ordinary_volume = 1_000_000.0

    # Every symbol starts from a comfortably in-universe RAW series, so any
    # exclusion below is attributable to exactly one rule.
    for symbol in symbols:
        put("close", symbol, 50.0 * 1.7)
        put("volume", symbol, ordinary_volume)

    # The penny name: RAW close 1.0 < min_price on every bar, ADJUSTED close
    # the highest in the panel.
    put("close", PENNY, 1.0)
    put("adjClose", PENNY, PENNY_ADJ_CLOSE)

    # The illiquid name: ordinary price, RAW dollar volume 85.0 * 100 = 8,500,
    # far under min_dollar_volume.
    put("volume", ILLIQUID, 100.0)

    # The warrant: perfectly tradeable on price and volume, wild adjusted
    # close. Only the STATIC ticker rule removes it.
    put("adjClose", WARRANT, WARRANT_ADJ_CLOSE)

    # The drop-out: in the universe until `drop_bar`, RAW close 1.0 from there
    # on, ADJUSTED close the highest among the commons throughout.
    dropout_close = np.full(n_bars, 50.0 * 1.7)
    dropout_close[drop_bar:] = 1.0
    put("close", DROPOUT, dropout_close)
    put("adjClose", DROPOUT, DROPOUT_ADJ_CLOSE)

    panel.to_zarr(zarr_path, mode="w")
    return dataset_config


def pandas_universe_mask(
    dataset_config: DatasetConfig,
    *,
    window: int = 5,
    min_price: float = MIN_PRICE,
    min_dollar_volume: float = MIN_DOLLAR_VOLUME,
    exclude_non_common: bool = True,
) -> pd.DataFrame:
    """The universe mask recomputed INDEPENDENTLY with pandas.

    Deliberately not a call into `UniverseFilteredFactor`: it is the reference
    the implementation is checked against, so it restates LS-1 from the store
    rather than sharing code with the thing under test. `True` means in.
    """
    panel = xr.open_dataset(Path(dataset_config.zarr_file_path)).load()
    panel.close()
    close = panel["close"].to_pandas()
    volume = panel["volume"].to_pandas()

    dollar = close * volume
    average = dollar.rolling(window, min_periods=window).mean()
    in_universe = (close >= min_price) & (average >= min_dollar_volume)
    if exclude_non_common:
        for symbol in in_universe.columns:
            if not UniverseFilteredFactor.is_common_ticker(str(symbol)):
                in_universe[symbol] = False
    return in_universe


class RankCloseFactor(FactorKunQuant):
    """A tiny KunQuant factor with one cross-sectional and two time-series outputs.

    - `rank_close` is a bare `Rank`, the cross-sectional op the LS-2 graph
      rewrite must reach;
    - `ma_close` is purely time-series, so it must be IDENTICAL wrapped and
      unwrapped on in-universe cells;
    - `ma_rank` is a time-series op OVER a cross-sectional one, which is where
      the LS-5 re-entry NaN cost shows up.

    It reads `adjClose`, never the raw close the mask reads.
    """

    def _get_factor_func(self) -> Function:
        builder = Builder()
        with builder:
            close = Input("adjClose")
            Output(Rank(close), "rank_close")
            Output(WindowedAvg(close, 3), "ma_close")
            Output(WindowedAvg(Rank(close), 3), "ma_rank")
        return Function(builder.ops)

    def _get_factor_names(self) -> tuple[str, ...]:
        return RANK_CLOSE_FACTOR_NAMES

    def _get_features(self, data: xr.Dataset) -> xr.Dataset:
        return data

    def _get_labels(self, data: xr.Dataset) -> NoReturn:
        raise RuntimeError(f"{type(self).__name__} is a feature, not a label")


def make_rank_close_factor(
    dataset_config: DatasetConfig,
    *,
    window: int = 5,
    njobs: int = 4,
    **config_kwargs,
) -> RankCloseFactor:
    """A bare (UNWRAPPED) `RankCloseFactor` over its own dataset instance."""
    return RankCloseFactor(
        FactorConfig(
            window=window,
            dataset=make_stock_dataset(dataset_config),
            mode="batch",
            data_columns=("adjClose",),
            njobs=njobs,
            **config_kwargs,
        )
    )


def make_return_label(
    dataset_config: DatasetConfig,
    *,
    n_forward_periods: int = 1,
    njobs: int = 4,
    **config_kwargs,
) -> Return:
    """A bare (UNWRAPPED) `Return` label over its own dataset instance."""
    return Return(
        FactorConfig(
            window=0,
            dataset=make_stock_dataset(dataset_config),
            mode="batch",
            data_columns=("adjClose",),
            njobs=njobs,
            kwargs={"n_forward_periods": n_forward_periods},
            **config_kwargs,
        )
    )


def wrap(factor, *, window: int = 5, **kwargs) -> UniverseFilteredFactor:
    """`UniverseFilteredFactor` at this module's thresholds."""
    return UniverseFilteredFactor(
        factor,
        min_price=MIN_PRICE,
        min_dollar_volume=MIN_DOLLAR_VOLUME,
        window=window,
        **kwargs,
    )


def make_wrapped_model(
    root: Path,
    dataset_config: DatasetConfig,
    *,
    model_cls,
    window: int = 5,
    n_forward_periods: int = 1,
    save_dir: str = "models",
    hyperparameters: dict | None = None,
    **dates,
):
    """A model over a WRAPPED factor and a WRAPPED label.

    Both are wrapped, per LS-3: masking the features alone would leave the
    label carrying out-of-universe rows.
    """
    config_kwargs = dict(
        factors=[wrap(make_rank_close_factor(dataset_config), window=window)],
        labels=[
            wrap(
                make_return_label(
                    dataset_config, n_forward_periods=n_forward_periods
                ),
                window=window,
            )
        ],
        model_save_dir=str(Path(root) / save_dir),
        factor_data_strategy="cal",
        label_data_strategy="cal",
        **dates,
    )
    if hyperparameters is not None:
        config_kwargs["hyperparameters"] = hyperparameters
    return model_cls(MLConfig(**config_kwargs))
