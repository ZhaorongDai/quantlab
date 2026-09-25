"""End-to-end US-equity pipeline on WRDS CRSP daily data.

Data -> Alpha101 + Alpha158 factors -> forward-return label -> model
(``xgb`` / ``xgb_td`` / ``realmlp``) -> cross-sectional TopN backtest.

Every setting lives in the ``Settings`` block below: edit it and run the file
(``uv run python examples/wrds_us_equity/pipeline.py``) or step through the
``# %%`` cells in VS Code / Jupyter. There is no command-line interface.

Prerequisite: a converted CRSP S&P 500 store and its membership panel, as
written by ``scripts/ingest_wrds_crsp.py --universe crsp_sp500 --to-zarr``
(see ``docs/wrds_crsp.md`` and this directory's README).

Why two derived stores are written in step 1:

- ``prices``: the CRSP panel on every PERMNO that was ever a member in the
  window, all history kept. Factors read it, so rolling windows never see a
  gap caused by index membership.
- ``members``: the same panel with every cell NaN where the PERMNO was not an
  S&P 500 member on that day. The label and the backtest read it, so the
  model trains only on member rows and the backtest can only buy members (a
  holding that leaves the index is sold on the next bar). Without it the
  roster would include stocks before they joined the index, which is
  look-ahead bias.

Both stores pad the symbol axis with all-NaN columns up to a multiple of 16,
because KunQuant batch runs need a symbol count that is a multiple of the
host's SIMD block width (8 with AVX2, 16 with AVX-512).
"""

# %% Settings
import os
import sys

# Set before torch or xgboost is imported. W&B: every training run calls
# wandb.init(); set WANDB_MODE=online (after `wandb login`) to track runs.
os.environ.setdefault("WANDB_MODE", "disabled")
# macOS only: xgboost and torch ship different OpenMP runtimes that clash in
# one process unless OpenMP runs single-threaded.
if sys.platform == "darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import xarray as xr
from loguru import logger

from quantlab.backtest.us_equity import USEquityCrossectionSelectStockVectorBt
from quantlab.base.config import (
    ConstituentDatasetConfig,
    CrossSectionBacktestConfig,
    CrspDatasetConfig,
    DatasetConfig,
    FactorConfig,
    MLConfig,
)
from quantlab.config import get_data_root, set_data_root
from quantlab.dataset._support.masking import UniverseMask
from quantlab.dataset.constituent import CrspSP500ConstituentDataset
from quantlab.dataset.crsp import TICKER_SIDECAR_SUFFIX, CrspStockDataset
from quantlab.dataset.stock import StockDataset
from quantlab.factor.alpha101 import Alpha101Stock
from quantlab.factor.alpha158 import Alpha158Stock
from quantlab.label.fret import Return
from quantlab.ml_model.realmlp import RealMLPRegressor
from quantlab.ml_model.xgb import XGBoostRegressor
from quantlab.ml_model.xgb_td import XGBTDRegressor

#: Model heads selectable through ``Settings.model``.
MODELS = {
    "xgb": XGBoostRegressor,
    "xgb_td": XGBTDRegressor,
    "realmlp": RealMLPRegressor,
}

#: Default hyperparameters per head; ``Settings.hyperparameters`` overrides.
DEFAULT_HYPERPARAMETERS = {
    # xgb.train parameters; trained on the pooled CCC loss, early-stopped on RMSE.
    "xgb": {"num_boost_round": 1000, "eta": 0.05, "max_depth": 6, "nthread": 8},
    # pytabkit XGB_TD_Regressor constructor arguments.
    "xgb_td": {"n_estimators": 1000, "n_threads": 8},
    # pytabkit RealMLP_TD_Regressor constructor arguments.
    "realmlp": {"n_epochs": 256, "device": "cpu", "n_threads": 8},
}


@dataclass
class Settings:
    """Everything the pipeline needs; edit here, nothing is read from argv."""

    #: Storage root; ``None`` keeps quantlab's default (``QUANTLAB_DATA_DIR``
    #: or ``data/`` beside the repository), which is where the ingest script
    #: wrote the CRSP stores.
    data_root: str | None = None

    #: Model head: ``"xgb"``, ``"xgb_td"`` or ``"realmlp"``.
    model: str = "xgb"
    #: Overrides merged over ``DEFAULT_HYPERPARAMETERS[model]``.
    hyperparameters: dict = field(default_factory=dict)
    early_stopping: bool = True
    early_stopping_patience: int = 50
    #: Trailing share of the training window held out for early stopping.
    val_size: float = 0.2

    #: Data window. The factor warm-up is read before ``start_date``.
    start_date: str = "2012-01-01"
    end_date: str = "2024-12-31"
    #: Training and out-of-sample test windows (inclusive), inside the above.
    train_start: str = "2012-01-01"
    train_end: str = "2019-12-31"
    test_start: str = "2020-01-01"
    test_end: str = "2024-12-31"

    #: ``True`` trains walk-forward folds (``train_cv``) and backtests the
    #: stitched out-of-sample folds (``run_cv``); ``False`` trains once on the
    #: training window and backtests the test window.
    use_cv: bool = False
    #: Walk-forward fold geometry in bars (``use_cv=True`` only). Each fold's
    #: test segment is ``cv_train_periods // 5`` bars.
    cv_train_periods: int = 1250
    cv_gap_periods: int = 5

    #: Factor lookback in calendar days, read before every computed window.
    factor_window: int = 400
    #: Alpha subsets; ``None`` computes the whole library.
    alpha101_names: tuple[str, ...] | None = None
    alpha158_names: tuple[str, ...] | None = None
    #: KunQuant executor threads.
    njobs: int = 16

    #: Label horizon: open-to-open return from t+1 to t+1+horizon.
    horizon: int = 5

    #: Backtest: rebalance every ``rebalance_periods`` bars into the top
    #: ``top_n`` scores (``long_short`` adds the bottom ``top_n`` short).
    rebalance_periods: int = 5
    top_n: int = 50
    direction: str = "long_only"
    fees: float = 0.0005
    slippage: float = 0.0005
    init_cash: float = 1_000_000.0


SETTINGS = Settings()

#: KunQuant needs the symbol count to be a multiple of the SIMD block width;
#: 16 covers AVX-512 and every narrower host.
SYMBOL_BLOCK = 16

#: Adjusted columns the alpha libraries and the label read, plus the raw ones
#: kept for inspection.
PANEL_COLUMNS = (
    "adjOpen", "adjHigh", "adjLow", "adjClose", "adjVolume", "close", "volume", "ret",
)
ALPHA_COLUMNS = ("adjOpen", "adjHigh", "adjLow", "adjClose", "adjVolume")


def paths(s: Settings) -> dict[str, Path]:
    """Every file location the pipeline reads or writes, from the data root."""
    if s.data_root is not None:
        set_data_root(s.data_root)
    root = get_data_root()
    stores = root / "data" / "us_equity" / "1d"
    crsp_downloads = root / "downloads" / "us_equity" / "1d" / "wrds_crsp"
    work = root / "data" / "pipeline" / "wrds_sp500"
    return {
        # Written by scripts/ingest_wrds_crsp.py --universe crsp_sp500 --to-zarr.
        "crsp_store": stores / "wrds_crsp_sp500_1d.zarr",
        "membership_store": stores / "wrds_crsp_sp500_membership.zarr",
        "raw_dir": crsp_downloads / "wrds",
        "reference_dir": crsp_downloads / "_reference",
        # Written by this pipeline.
        "prices": work / "prices.zarr",
        "members": work / "members.zarr",
        "alpha101": work / "factor" / "alpha101.zarr",
        "alpha158": work / "factor" / "alpha158.zarr",
        "label": work / "label" / f"ret_{s.horizon}.zarr",
        "models": work / "models" / s.model,
        "backtests": work / "backtests" / s.model,
    }


P = paths(SETTINGS)


def stock_dataset(store: Path) -> StockDataset:
    """A dataset over one of the derived stores (not a raw-data converter)."""
    return StockDataset(DatasetConfig(
        zarr_file_path=str(store),
        raw_data_dir_path=str(P["raw_dir"]),
        market="us_equity",
        frequency="1d",
    ))


# %% 1. Read the CRSP data and write the prices / members stores
def read_crsp(s: Settings) -> tuple[xr.Dataset, xr.Dataset]:
    """Read the CRSP S&P 500 price panel and its point-in-time membership."""
    for key in ("crsp_store", "membership_store"):
        if not P[key].exists():
            raise FileNotFoundError(
                f"{P[key]} not found. Download and convert the CRSP S&P 500 "
                f"roster first: uv run python scripts/ingest_wrds_crsp.py "
                f"--universe crsp_sp500 --start-date <start> --end-date <end> "
                f"--to-zarr (see examples/wrds_us_equity/README.md)."
            )
    crsp = CrspStockDataset(CrspDatasetConfig(
        zarr_file_path=str(P["crsp_store"]),
        raw_data_dir_path=str(P["raw_dir"]),
        reference_dir=str(P["reference_dir"]),
    ))
    membership = CrspSP500ConstituentDataset(ConstituentDatasetConfig(
        zarr_file_path=str(P["membership_store"]),
        cache_dir=str(P["reference_dir"]),
    ))
    # Logs every index member the price panel lacks (a survivorship gap).
    UniverseMask.from_datasets(crsp.read(), membership.read()).report()

    prices = crsp.get_xarray_dataset()[list(PANEL_COLUMNS)]
    prices = prices.sel(timestamp=slice(None, s.end_date))
    is_member = membership.get_xarray_dataset()["is_member"]
    return prices, is_member


def pad_symbols(panel: xr.Dataset) -> xr.Dataset:
    """Append all-NaN PERMNOs -1, -2, ... up to a multiple of ``SYMBOL_BLOCK``."""
    n_pad = -panel.sizes["symbol"] % SYMBOL_BLOCK
    if n_pad == 0:
        return panel
    pad = np.arange(-1, -n_pad - 1, -1, dtype=panel["symbol"].dtype)
    return panel.reindex(symbol=np.concatenate([panel["symbol"].values, pad]))


def write_store(panel: xr.Dataset, store: Path) -> None:
    """Write a panel and copy the CRSP ticker sidecar beside it for reports."""
    store.parent.mkdir(parents=True, exist_ok=True)
    panel.to_zarr(store, mode="w")
    sidecar = Path(str(P["crsp_store"]) + TICKER_SIDECAR_SUFFIX)
    if sidecar.exists():
        shutil.copyfile(sidecar, str(store) + TICKER_SIDECAR_SUFFIX)


def prepare_stores(s: Settings) -> None:
    prices, is_member = read_crsp(s)
    prices = pad_symbols(prices)
    # Membership is on calendar days, prices on trading days; days or PERMNOs
    # the membership panel does not cover count as non-members.
    member = (
        is_member.reindex(timestamp=prices.timestamp, symbol=prices.symbol)
        .fillna(False)
        .astype(bool)
    )
    members = prices.where(member)
    write_store(prices, P["prices"])
    write_store(members, P["members"])
    logger.info(
        f"prices: {dict(prices.sizes)}; member cells {int(member.sum())} "
        f"of {member.size}; stores under {P['prices'].parent}"
    )


# %% 2. Factors (Alpha101 + Alpha158) and 3. the label
def factor_objects(s: Settings) -> tuple[list, list]:
    """The factor and label objects; each call builds fresh datasets."""
    common = dict(
        mode="batch",
        start_date=s.start_date,
        end_date=s.end_date,
        njobs=s.njobs,
    )
    alpha101 = Alpha101Stock(FactorConfig(
        window=s.factor_window,
        dataset=stock_dataset(P["prices"]),
        data_columns=ALPHA_COLUMNS,
        factor_names=s.alpha101_names,
        file_path=str(P["alpha101"]),
        **common,
    ))
    alpha158 = Alpha158Stock(FactorConfig(
        window=s.factor_window,
        dataset=stock_dataset(P["prices"]),
        data_columns=ALPHA_COLUMNS,
        factor_names=s.alpha158_names,
        file_path=str(P["alpha158"]),
        **common,
    ))
    label = Return(FactorConfig(
        window=2 * s.horizon + 5,
        dataset=stock_dataset(P["members"]),
        data_columns=("adjOpen",),
        kwargs={"n_forward_periods": s.horizon},
        file_path=str(P["label"]),
        **common,
    ))
    return [alpha101, alpha158], [label]


def compute_factors(s: Settings) -> None:
    factors, labels = factor_objects(s)
    for factor in factors + labels:
        factor.cal().save(mode="w")
        logger.info(
            f"{type(factor).__name__}: {len(factor.get_factor_names())} "
            f"column(s) -> {factor.config.file_path}"
        )


# %% 4. Model
def build_model(s: Settings):
    """A fresh model head reading the stored factors and label."""
    if s.model not in MODELS:
        raise ValueError(f"model must be one of {sorted(MODELS)}, got {s.model!r}")
    factors, labels = factor_objects(s)
    return MODELS[s.model](MLConfig(
        factors=factors,
        labels=labels,
        model_save_dir=str(P["models"]),
        factor_data_strategy="read",
        label_data_strategy="read",
        start_date=s.start_date,
        end_date=s.end_date,
        train_start=s.train_start,
        train_end=s.train_end,
        test_start=s.test_start,
        test_end=s.test_end,
        early_stopping=s.early_stopping,
        early_stopping_patience=s.early_stopping_patience,
        val_size=s.val_size,
        hyperparameters={**DEFAULT_HYPERPARAMETERS[s.model], **s.hyperparameters},
    ))


def train(s: Settings) -> Path:
    """Train once, or walk-forward; returns the checkpoint or the CV project dir."""
    model = build_model(s).collect()
    if not s.use_cv:
        checkpoint = Path(model.train())
        logger.info(f"checkpoint: {checkpoint}")
        return checkpoint
    results = model.train_cv(
        train_periods=s.cv_train_periods, gap_periods=s.cv_gap_periods
    )
    manifests = sorted(
        P["models"].rglob("cv_folds.json"), key=lambda p: p.stat().st_mtime
    )
    project_dir = manifests[-1].parent
    ic = [r.get("test_rank_ic") for r in results]
    logger.info(f"{len(results)} folds, test RankIC {ic}; project: {project_dir}")
    return project_dir


# %% 5. Backtest
def backtest(s: Settings, trained: Path):
    """Backtest the test window, or the stitched CV folds."""
    config = CrossSectionBacktestConfig(
        price_dataset=stock_dataset(P["members"]),
        model=build_model(s),
        model_mode="load",
        checkpoint=None if s.use_cv else str(trained),
        cv_project_dir=str(trained) if s.use_cv else None,
        start_date=s.test_start,
        end_date=s.test_end,
        output_dir=str(P["backtests"]),
        rebalance_periods=s.rebalance_periods,
        direction=s.direction,
        top_n=s.top_n,
        fees=s.fees,
        slippage=s.slippage,
        init_cash=s.init_cash,
    )
    backtester = USEquityCrossectionSelectStockVectorBt(config)
    result = backtester.run_cv() if s.use_cv else backtester.run()
    metrics = result.metrics["stitched"] if s.use_cv else result.metrics
    summary = {
        key: metrics["whole"].get(key)
        for key in ("Total Return [%]", "Sharpe Ratio", "Max Drawdown [%]")
    }
    logger.info(f"backtest {json.dumps(summary, default=str)}; run: {result.run_dir}")
    return result


# %% Run everything
def main(s: Settings = SETTINGS):
    """Run the five steps; the stages can also be run cell by cell."""
    global P
    P = paths(s)
    prepare_stores(s)
    compute_factors(s)
    trained = train(s)
    return backtest(s, trained)


if __name__ == "__main__":
    main()
