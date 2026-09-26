"""RealMLP (pytabkit) pipeline on the whole CRSP market, from WRDS daily bars.

CRSP daily bars -> Alpha101 + Alpha158 factors -> open-to-open forward-return
label -> ``RealMLPRegressor`` (pytabkit's tuned-default MLP; scales its inputs
itself) -> TopN cross-sectional backtest against buy-and-hold SPY, logged to
Weights & Biases.

Every setting is a constant or a quantlab config object at the top of the
file; edit them and run ``uv run python examples/wrds_us_equity/market_realmlp.py``
or step through the ``# %%`` cells. Prerequisite: the stores written by
``scripts/wrds/market.py`` and ``scripts/wrds/etf.py --etf spy`` under the
data root (see README.md).

The market is every listed common stock, thousands of PERMNOs, so the
factor and model steps need far more memory than an index: roughly 3 GB per
1,000 PERMNOs over 13 years with all 251 features. Narrow ``START``/``END``
or pin ``factor_names`` for a first run.
"""

# %% Settings
import os
import sys

# macOS only: xgboost and torch ship different OpenMP runtimes that clash in
# one process unless OpenMP runs single-threaded. Set before either imports.
if sys.platform == "darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "1")

from pathlib import Path

import numpy as np
from loguru import logger

from quantlab.backtest.us_equity import USEquityCrossectionSelectStockVectorBt
from quantlab.base.config import (
    SPY_PERMNO,
    ConstituentDatasetConfig,
    CrossSectionBacktestConfig,
    CrspDatasetConfig,
    DatasetConfig,
    FactorConfig,
    MLConfig,
)
from quantlab.config import get_data_root
from quantlab.dataset.constituent import CrspMarketConstituentDataset
from quantlab.dataset.crsp import CrspStockDataset
from quantlab.dataset.stock import StockDataset
from quantlab.factor.alpha101 import Alpha101Stock
from quantlab.factor.alpha158 import Alpha158Stock
from quantlab.label.fret import Return
from quantlab.ml_model.realmlp import RealMLPRegressor

#: Storage root: ``QUANTLAB_DATA_DIR`` or ``data/`` beside the repository,
#: where the WRDS scripts wrote the stores. Replace with ``Path("/my/root")``.
DATA_ROOT = get_data_root()
STORES = DATA_ROOT / "data" / "us_equity" / "1d"
RAW = DATA_ROOT / "downloads" / "us_equity" / "1d" / "wrds_crsp" / "wrds"
REFERENCE = DATA_ROOT / "downloads" / "us_equity" / "1d" / "wrds_crsp" / "_reference"
#: Everything this pipeline writes goes under here.
WORK = DATA_ROOT / "data" / "pipeline" / "wrds_market"

#: Data window (the factor warm-up is read before START), training window
#: and out-of-sample test window, all inclusive.
START, END = "2012-01-01", "2024-12-31"
TRAIN_START, TRAIN_END = "2012-01-01", "2019-12-31"
TEST_START, TEST_END = "2020-01-01", "2024-12-31"
#: Label horizon in bars: open-to-open return from t+1 to t+1+HORIZON.
HORIZON = 5
#: Columns the alpha libraries read; ``ret`` is kept for the label's dataset.
ALPHA_COLUMNS = ("adjOpen", "adjHigh", "adjLow", "adjClose", "adjVolume")
#: Weights & Biases: "online" (needs ``wandb login``), "offline" or "disabled".
WANDB_MODE = "online"


def stock_dataset(store: Path) -> StockDataset:
    """A dataset over one of the derived stores."""
    return StockDataset(DatasetConfig(
        zarr_file_path=str(store), raw_data_dir_path=str(RAW),
        market="us_equity", frequency="1d",
    ))


def factors_and_label() -> tuple[list, list]:
    """``([alpha101, alpha158], [label])``; each call builds fresh objects.

    Factors read ``prices.zarr`` so rolling windows see no membership gaps;
    the label reads ``members.zarr`` so returns exist on member rows only.
    """
    alpha101 = Alpha101Stock(FactorConfig(
        window=400, dataset=stock_dataset(WORK / "prices.zarr"), mode="batch",
        data_columns=ALPHA_COLUMNS, file_path=str(WORK / "factor" / "alpha101.zarr"),
        start_date=START, end_date=END, njobs=16,
    ))
    alpha158 = Alpha158Stock(FactorConfig(
        window=400, dataset=stock_dataset(WORK / "prices.zarr"), mode="batch",
        data_columns=ALPHA_COLUMNS, file_path=str(WORK / "factor" / "alpha158.zarr"),
        start_date=START, end_date=END, njobs=16,
    ))
    label = Return(FactorConfig(
        window=2 * HORIZON + 5, dataset=stock_dataset(WORK / "members.zarr"), mode="batch",
        data_columns=("adjOpen",), kwargs={"n_forward_periods": HORIZON},
        file_path=str(WORK / "label" / f"ret_{HORIZON}.zarr"),
        start_date=START, end_date=END, njobs=16,
    ))
    return [alpha101, alpha158], [label]


# %% 1. Prices and members stores
def prepare_stores() -> None:
    """Write ``prices`` (full history of every security ever listed) and ``members``
    (the same panel, NaN where the PERMNO was not listed, or not a common
    stock, that day).

    The symbol axis is padded with all-NaN PERMNOs to a multiple of 16, the
    SIMD block width KunQuant batch runs need.
    """
    crsp = CrspStockDataset(CrspDatasetConfig(
        zarr_file_path=str(STORES / "wrds_crsp_market_1d.zarr"),
        raw_data_dir_path=str(RAW), reference_dir=str(REFERENCE),
    ))
    membership = CrspMarketConstituentDataset(ConstituentDatasetConfig(
        zarr_file_path=str(STORES / "wrds_crsp_market_membership.zarr"),
        cache_dir=str(REFERENCE),
    ))
    for store in (crsp.config.zarr_file_path, membership.config.zarr_file_path):
        if not Path(store).exists():
            raise FileNotFoundError(
                f"{store} not found; run scripts/wrds/market.py first "
                f"(see README.md)."
            )
    prices = crsp.read().get_xarray_dataset()[[*ALPHA_COLUMNS, "close", "volume", "ret"]]
    prices = prices.sel(timestamp=slice(None, END))
    n_pad = -prices.sizes["symbol"] % 16
    pad = np.arange(-1, -n_pad - 1, -1, dtype=prices["symbol"].dtype)
    prices = prices.reindex(symbol=np.concatenate([prices["symbol"].values, pad]))
    member = (
        membership.read().get_xarray_dataset()["is_member"]
        .reindex(timestamp=prices.timestamp, symbol=prices.symbol)
        .fillna(False)
        .astype(bool)
    )
    WORK.mkdir(parents=True, exist_ok=True)
    prices.to_zarr(WORK / "prices.zarr", mode="w")
    prices.where(member).to_zarr(WORK / "members.zarr", mode="w")
    logger.info(f"prices {dict(prices.sizes)}, member cells {int(member.sum())}")


# %% 2. Factors and 3. label
def compute_factors() -> None:
    factors, labels = factors_and_label()
    for factor in factors + labels:
        factor.cal().save(mode="w")
        logger.info(f"{type(factor).__name__} -> {factor.config.file_path}")


# %% 4. Model
def build_model() -> RealMLPRegressor:
    """A fresh head reading the stored factors and label."""
    factors, labels = factors_and_label()
    return RealMLPRegressor(MLConfig(
        factors=factors, labels=labels,
        model_save_dir=str(WORK / "models" / "realmlp"),
        factor_data_strategy="read", label_data_strategy="read",
        start_date=START, end_date=END,
        train_start=TRAIN_START, train_end=TRAIN_END,
        test_start=TEST_START, test_end=TEST_END,
        # Early stopping on the trailing val_size of the training window;
        # patience counts epochs.
        early_stopping=True, early_stopping_patience=50, val_size=0.2,
        # pytabkit RealMLP_TD_Regressor constructor arguments.
        hyperparameters={"n_epochs": 256, "device": "cpu", "n_threads": 8},
    ))


def train() -> Path:
    """Train once on the training window; returns the checkpoint path."""
    checkpoint = Path(build_model().collect().train())
    logger.info(f"checkpoint: {checkpoint}")
    return checkpoint


# %% 5. Backtest
def backtest(checkpoint: Path):
    """TopN backtest of the test window against buy-and-hold SPY."""
    benchmark = CrspStockDataset(CrspDatasetConfig.etf_benchmark(
        permno=SPY_PERMNO, zarr_file_path=str(STORES / "wrds_crsp_spy_1d.zarr"),
        raw_data_dir_path=str(RAW), reference_dir=str(REFERENCE),
    ))
    if not Path(benchmark.config.zarr_file_path).exists():
        raise FileNotFoundError(
            f"{benchmark.config.zarr_file_path} not found; run scripts/wrds/etf.py "
            f"--etf spy first, or pass benchmark_dataset=None below."
        )
    backtester = USEquityCrossectionSelectStockVectorBt(CrossSectionBacktestConfig(
        price_dataset=stock_dataset(WORK / "members.zarr"),
        model=build_model(), model_mode="load", checkpoint=str(checkpoint),
        start_date=TEST_START, end_date=TEST_END,
        output_dir=str(WORK / "backtests" / "realmlp"),
        # Rebalance every 5 bars into the top 100 scores; "long_short"
        # would also short the bottom 100.
        rebalance_periods=5, top_n=100, direction="long_only",
        fees=0.0005, slippage=0.0005, init_cash=1_000_000.0,
        use_wandb=WANDB_MODE != "disabled", benchmark_dataset=benchmark,
    ))
    result = backtester.run()
    whole = result.metrics["whole"]
    logger.info(
        f"total return {whole.get('Total Return [%]')}%, Sharpe "
        f"{whole.get('Sharpe Ratio')}, max drawdown {whole.get('Max Drawdown [%]')}%; "
        f"run: {result.run_dir}"
    )
    return result


# %% Run everything
def main():
    os.environ["WANDB_MODE"] = WANDB_MODE  # read at every wandb.init()
    prepare_stores()
    compute_factors()
    return backtest(train())


if __name__ == "__main__":
    main()
