"""XGBoost pipeline on WRDS CRSP daily data, self-contained.

CRSP daily bars of a point-in-time index (S&P 500 or Nasdaq-100) ->
Alpha101 + Alpha158 factors -> open-to-open forward-return label ->
``XGBoostRegressor`` (``xgb.train``, native early stopping) -> TopN cross-sectional
backtest against a buy-and-hold ETF benchmark (SPY or QQQ), logged to
Weights & Biases.

Edit ``DATA_ROOT`` and ``Settings`` below, then run
``uv run python examples/wrds_us_equity/xgb.py`` or step through the
``# %%`` cells. Prerequisite: the CRSP stores written by
``scripts/wrds/index.py --index <universe>`` and ``scripts/wrds/etf.py
--etf spy,qqq`` under the data root (see README.md).
"""

# %% Settings
import os
import sys

# macOS only: xgboost and torch ship different OpenMP runtimes that clash in
# one process unless OpenMP runs single-threaded. Set before either imports.
if sys.platform == "darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "1")

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from loguru import logger

from quantlab.backtest.us_equity import USEquityCrossectionSelectStockVectorBt
from quantlab.base.config import (
    QQQ_PERMNO,
    SPY_PERMNO,
    ConstituentDatasetConfig,
    CrossSectionBacktestConfig,
    CrspDatasetConfig,
    DatasetConfig,
    FactorConfig,
    MLConfig,
)
from quantlab.config import get_data_root, set_data_root
from quantlab.dataset.constituent import (
    CompustatNasdaq100ConstituentDataset,
    CrspSP500ConstituentDataset,
)
from quantlab.dataset.crsp import CrspStockDataset
from quantlab.dataset.stock import StockDataset
from quantlab.factor.alpha101 import Alpha101Stock
from quantlab.factor.alpha158 import Alpha158Stock
from quantlab.label.fret import Return
from quantlab.ml_model.xgb import XGBoostRegressor

#: Storage root. ``None`` keeps quantlab's default (``QUANTLAB_DATA_DIR`` or
#: ``data/`` beside the repository), where the WRDS scripts wrote the stores.
DATA_ROOT: str | None = None

#: Per universe: the membership panel class, the benchmark ETF (name, PERMNO)
#: and the default ``top_n``.
UNIVERSES = {
    "sp500": (CrspSP500ConstituentDataset, ("spy", SPY_PERMNO), 50),
    "nasdaq100": (CompustatNasdaq100ConstituentDataset, ("qqq", QQQ_PERMNO), 10),
}
#: Columns the alpha libraries read; ``ret`` is kept for the label's dataset.
ALPHA_COLUMNS = ("adjOpen", "adjHigh", "adjLow", "adjClose", "adjVolume")


@dataclass
class Settings:
    """Everything this pipeline needs; nothing is read from argv."""

    #: ``"sp500"`` or ``"nasdaq100"``.
    universe: str = "sp500"
    #: Data window; the factor warm-up is read before ``start_date``.
    start_date: str = "2012-01-01"
    end_date: str = "2024-12-31"
    #: Training and out-of-sample test windows (inclusive).
    train_start: str = "2012-01-01"
    train_end: str = "2019-12-31"
    test_start: str = "2020-01-01"
    test_end: str = "2024-12-31"
    #: Factor lookback in calendar days, and alpha subsets (``None`` = all).
    factor_window: int = 400
    alpha101_names: tuple[str, ...] | None = None
    alpha158_names: tuple[str, ...] | None = None
    #: KunQuant executor threads.
    njobs: int = 16
    #: Label horizon in bars: open-to-open return from t+1 to t+1+horizon.
    horizon: int = 5
    #: ``xgb.train`` parameters; early-stopped on the validation RMSE.
    hyperparameters: dict = field(default_factory=lambda: {"num_boost_round": 1000, "eta": 0.05, "max_depth": 6, "nthread": 8})
    #: Early stopping on the trailing ``val_size`` of the training window;
    #: patience is in boosting rounds.
    early_stopping: bool = True
    early_stopping_patience: int = 50
    val_size: float = 0.2
    #: Backtest: rebalance every ``rebalance_periods`` bars into the top
    #: ``top_n`` scores (``None``: the universe default). ``benchmark=False``
    #: skips the ETF comparison.
    rebalance_periods: int = 5
    top_n: int | None = None
    direction: str = "long_only"
    fees: float = 0.0005
    slippage: float = 0.0005
    init_cash: float = 1_000_000.0
    benchmark: bool = True
    #: Weights & Biases: ``"online"`` (needs ``wandb login``), ``"offline"``
    #: (``wandb sync`` later) or ``"disabled"``.
    wandb_mode: str = "online"


SETTINGS = Settings()
MODEL_NAME = "xgb"


@dataclass
class Paths:
    """Every location the pipeline reads or writes, under the data root."""

    crsp_store: Path       # written by scripts/wrds/index.py
    membership_store: Path
    benchmark_store: Path  # written by scripts/wrds/etf.py
    raw_dir: Path
    reference_dir: Path
    work: Path             # this pipeline's output directory

    @classmethod
    def build(cls, s: Settings) -> "Paths":
        if DATA_ROOT is not None:
            set_data_root(DATA_ROOT)
        root = get_data_root()
        stores = root / "data" / "us_equity" / "1d"
        downloads = root / "downloads" / "us_equity" / "1d" / "wrds_crsp"
        etf = UNIVERSES[s.universe][1][0]
        return cls(
            crsp_store=stores / f"wrds_crsp_{s.universe}_1d.zarr",
            membership_store=stores / f"wrds_crsp_{s.universe}_membership.zarr",
            benchmark_store=stores / f"wrds_crsp_{etf}_1d.zarr",
            raw_dir=downloads / "wrds",
            reference_dir=downloads / "_reference",
            work=root / "data" / "pipeline" / f"wrds_{s.universe}",
        )


def stock_dataset(store: Path, p: Paths) -> StockDataset:
    """A dataset over one of the derived stores."""
    return StockDataset(DatasetConfig(
        zarr_file_path=str(store), raw_data_dir_path=str(p.raw_dir),
        market="us_equity", frequency="1d",
    ))


# %% 1. Prices and members stores
def prepare_stores(s: Settings, p: Paths) -> None:
    """Write ``prices`` (full history of every member ever) and ``members``
    (the same panel, NaN where the PERMNO was not a member that day).

    Factors read ``prices`` so rolling windows see no membership gaps; the
    label and the backtest read ``members`` so the model trains on, and
    trades, members only. The symbol axis is padded with all-NaN PERMNOs to
    a multiple of 16, the SIMD block width KunQuant batch runs need.
    """
    for store in (p.crsp_store, p.membership_store):
        if not store.exists():
            raise FileNotFoundError(
                f"{store} not found; run scripts/wrds/index.py --index "
                f"{s.universe} first (see README.md)."
            )
    crsp = CrspStockDataset(CrspDatasetConfig(
        zarr_file_path=str(p.crsp_store), raw_data_dir_path=str(p.raw_dir),
        reference_dir=str(p.reference_dir),
    ))
    membership = UNIVERSES[s.universe][0](ConstituentDatasetConfig(
        zarr_file_path=str(p.membership_store), cache_dir=str(p.reference_dir),
    ))
    prices = crsp.read().get_xarray_dataset()[[*ALPHA_COLUMNS, "close", "volume", "ret"]]
    prices = prices.sel(timestamp=slice(None, s.end_date))
    n_pad = -prices.sizes["symbol"] % 16
    pad = np.arange(-1, -n_pad - 1, -1, dtype=prices["symbol"].dtype)
    prices = prices.reindex(symbol=np.concatenate([prices["symbol"].values, pad]))
    member = (
        membership.read().get_xarray_dataset()["is_member"]
        .reindex(timestamp=prices.timestamp, symbol=prices.symbol)
        .fillna(False)
        .astype(bool)
    )
    p.work.mkdir(parents=True, exist_ok=True)
    prices.to_zarr(p.work / "prices.zarr", mode="w")
    prices.where(member).to_zarr(p.work / "members.zarr", mode="w")
    logger.info(f"prices {dict(prices.sizes)}, member cells {int(member.sum())}")


# %% 2. Factors and 3. label
def factor_objects(s: Settings, p: Paths) -> tuple[list, list]:
    """``([alpha101, alpha158], [label])``, each over a fresh dataset."""
    common = dict(mode="batch", start_date=s.start_date, end_date=s.end_date, njobs=s.njobs)
    alpha101 = Alpha101Stock(FactorConfig(
        window=s.factor_window, dataset=stock_dataset(p.work / "prices.zarr", p),
        data_columns=ALPHA_COLUMNS, factor_names=s.alpha101_names,
        file_path=str(p.work / "factor" / "alpha101.zarr"), **common,
    ))
    alpha158 = Alpha158Stock(FactorConfig(
        window=s.factor_window, dataset=stock_dataset(p.work / "prices.zarr", p),
        data_columns=ALPHA_COLUMNS, factor_names=s.alpha158_names,
        file_path=str(p.work / "factor" / "alpha158.zarr"), **common,
    ))
    label = Return(FactorConfig(
        window=2 * s.horizon + 5, dataset=stock_dataset(p.work / "members.zarr", p),
        data_columns=("adjOpen",), kwargs={"n_forward_periods": s.horizon},
        file_path=str(p.work / "label" / f"ret_{s.horizon}.zarr"), **common,
    ))
    return [alpha101, alpha158], [label]


def compute_factors(s: Settings, p: Paths) -> None:
    factors, labels = factor_objects(s, p)
    for factor in factors + labels:
        factor.cal().save(mode="w")
        logger.info(f"{type(factor).__name__} -> {factor.config.file_path}")


# %% 4. Model
def build_model(s: Settings, p: Paths) -> XGBoostRegressor:
    """A fresh head reading the stored factors and label."""
    factors, labels = factor_objects(s, p)
    return XGBoostRegressor(MLConfig(
        factors=factors, labels=labels,
        model_save_dir=str(p.work / "models" / MODEL_NAME),
        factor_data_strategy="read", label_data_strategy="read",
        start_date=s.start_date, end_date=s.end_date,
        train_start=s.train_start, train_end=s.train_end,
        test_start=s.test_start, test_end=s.test_end,
        early_stopping=s.early_stopping,
        early_stopping_patience=s.early_stopping_patience,
        val_size=s.val_size, hyperparameters=dict(s.hyperparameters),
    ))


def train(s: Settings, p: Paths) -> Path:
    """Train once on the training window; returns the checkpoint path."""
    checkpoint = Path(build_model(s, p).collect().train())
    logger.info(f"checkpoint: {checkpoint}")
    return checkpoint


# %% 5. Backtest
def backtest(s: Settings, p: Paths, checkpoint: Path):
    """TopN backtest of the test window against the buy-and-hold ETF."""
    etf_name, permno = UNIVERSES[s.universe][1]
    benchmark = None
    if s.benchmark:
        if not p.benchmark_store.exists():
            raise FileNotFoundError(
                f"{p.benchmark_store} not found; run scripts/wrds/etf.py --etf "
                f"{etf_name} first, or set Settings.benchmark=False."
            )
        benchmark = CrspStockDataset(CrspDatasetConfig.etf_benchmark(
            permno=permno, zarr_file_path=str(p.benchmark_store),
            raw_data_dir_path=str(p.raw_dir), reference_dir=str(p.reference_dir),
        ))
    result = USEquityCrossectionSelectStockVectorBt(CrossSectionBacktestConfig(
        price_dataset=stock_dataset(p.work / "members.zarr", p),
        model=build_model(s, p), model_mode="load", checkpoint=str(checkpoint),
        start_date=s.test_start, end_date=s.test_end,
        output_dir=str(p.work / "backtests" / MODEL_NAME),
        rebalance_periods=s.rebalance_periods, direction=s.direction,
        top_n=s.top_n if s.top_n is not None else UNIVERSES[s.universe][2],
        fees=s.fees, slippage=s.slippage, init_cash=s.init_cash,
        use_wandb=s.wandb_mode != "disabled", benchmark_dataset=benchmark,
    )).run()
    whole = result.metrics["whole"]
    logger.info(
        f"total return {whole.get('Total Return [%]')}%, Sharpe "
        f"{whole.get('Sharpe Ratio')}, max drawdown {whole.get('Max Drawdown [%]')}%; "
        f"run: {result.run_dir}"
    )
    return result


# %% Run everything
def main(s: Settings = SETTINGS):
    os.environ["WANDB_MODE"] = s.wandb_mode  # read at every wandb.init()
    p = Paths.build(s)
    prepare_stores(s, p)
    compute_factors(s, p)
    return backtest(s, p, train(s, p))


if __name__ == "__main__":
    main()
