"""Shared settings, paths and steps of the WRDS US-equity pipelines.

Every pipeline in this directory (``xgb.py``, ``xgb_td.py``, ``realmlp.py``,
``factor_analysis.py``) runs on the same data: the point-in-time S&P 500 or
Nasdaq-100 from CRSP daily bars, Alpha101 and Alpha158 factors on adjusted
prices, and an open-to-open forward-return label on member rows only. This
module holds what they share. Edit ``DATA_ROOT`` below to point at your
storage; every other knob is a field of a dataclass.

Prerequisite: a converted CRSP store and its membership panel for the chosen
index, written by ``scripts/wrds/index.py --index sp500`` or ``--index
nasdaq100``, and for the model pipelines the index's ETF (SPY or QQQ) in its
own store, written by ``scripts/wrds/etf.py --etf spy,qqq`` (see this
directory's README and ``docs/wrds_crsp.md``).

Two derived stores are written by ``prepare_stores``:

- ``prices``: the CRSP panel on every PERMNO that was ever a member in the
  window, all history kept, so factor rolling windows never see a gap caused
  by index membership.
- ``members``: the same panel with every cell NaN where the PERMNO was not
  a member that day. The label and the backtest read it, so the model trains
  on member rows only and the backtest can only buy members.

Both pad the symbol axis with all-NaN columns to a multiple of 16, the SIMD
block width KunQuant batch runs need.
"""

import os
import sys

# Set before torch or xgboost is imported. macOS only: the two ship different
# OpenMP runtimes that clash in one process unless OpenMP runs single-threaded.
if sys.platform == "darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr
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
from quantlab.dataset._support.masking import UniverseMask
from quantlab.dataset.constituent import (
    CompustatNasdaq100ConstituentDataset,
    CrspSP500ConstituentDataset,
)
from quantlab.dataset.crsp import TICKER_SIDECAR_SUFFIX, CrspStockDataset
from quantlab.dataset.stock import StockDataset
from quantlab.factor.alpha101 import Alpha101Stock
from quantlab.factor.alpha158 import Alpha158Stock
from quantlab.label.fret import Return

#: Storage root. ``None`` keeps quantlab's default: ``QUANTLAB_DATA_DIR`` or
#: ``data/`` beside the repository, where ``scripts/wrds/*.py`` wrote the
#: stores when run as the README shows. Set it to any directory to move
#: everything, inputs and outputs, there.
DATA_ROOT: str | None = None

#: Index universes: the membership panel class and the default ``top_n``.
UNIVERSES = {
    "sp500": (CrspSP500ConstituentDataset, 50),
    "nasdaq100": (CompustatNasdaq100ConstituentDataset, 10),
}
#: Buy-and-hold benchmarks: the ETF's CRSP PERMNO and its store file name.
BENCHMARKS = {
    "spy": (SPY_PERMNO, "wrds_crsp_spy_1d.zarr"),
    "qqq": (QQQ_PERMNO, "wrds_crsp_qqq_1d.zarr"),
}
#: The benchmark ``benchmark="auto"`` picks for each universe.
DEFAULT_BENCHMARK = {"sp500": "spy", "nasdaq100": "qqq"}

#: KunQuant needs a symbol count that is a multiple of the SIMD block width.
SYMBOL_BLOCK = 16
#: Adjusted columns the alpha libraries and the label read, plus raw ones
#: kept for inspection.
PANEL_COLUMNS = (
    "adjOpen", "adjHigh", "adjLow", "adjClose", "adjVolume", "close", "volume", "ret",
)
ALPHA_COLUMNS = ("adjOpen", "adjHigh", "adjLow", "adjClose", "adjVolume")


# %% Settings shared by every pipeline
@dataclass
class DataSettings:
    """The universe, the window and the factor and label definitions."""

    #: ``"sp500"`` or ``"nasdaq100"``: picks the input stores, the membership
    #: panel and the output directory.
    universe: str = "sp500"
    #: Data window. The factor warm-up is read before ``start_date``.
    start_date: str = "2012-01-01"
    end_date: str = "2024-12-31"
    #: Factor lookback in calendar days, read before every computed window.
    factor_window: int = 400
    #: Alpha subsets; ``None`` computes the whole library (82 / 169 columns).
    alpha101_names: tuple[str, ...] | None = None
    alpha158_names: tuple[str, ...] | None = None
    #: KunQuant executor threads.
    njobs: int = 16
    #: Label horizon in bars: open-to-open return from t+1 to t+1+horizon.
    horizon: int = 5


@dataclass
class TrainSettings:
    """Windows, early stopping and walk-forward geometry, for every head."""

    #: Training and out-of-sample test windows (inclusive), inside the data
    #: window.
    train_start: str = "2012-01-01"
    train_end: str = "2019-12-31"
    test_start: str = "2020-01-01"
    test_end: str = "2024-12-31"
    #: Early stopping on the trailing ``val_size`` of the training window;
    #: patience is in boosting rounds (xgb, xgb_td) or epochs (realmlp).
    early_stopping: bool = True
    early_stopping_patience: int = 50
    val_size: float = 0.2
    #: ``True`` trains walk-forward folds (``train_cv``) and backtests the
    #: stitched out-of-sample folds (``run_cv``); ``False`` trains once on the
    #: training window and backtests the test window.
    use_cv: bool = False
    #: Walk-forward fold geometry in bars (``use_cv=True`` only); each fold's
    #: test segment is ``cv_train_periods // 5`` bars.
    cv_train_periods: int = 1250
    cv_gap_periods: int = 5


@dataclass
class BacktestSettings:
    """The TopN cross-sectional backtest over the test window."""

    #: Buy-and-hold benchmark: ``"auto"`` (SPY for sp500, QQQ for nasdaq100),
    #: ``"spy"``, ``"qqq"`` or ``None`` for no comparison.
    benchmark: str | None = "auto"
    #: Rebalance every ``rebalance_periods`` bars into the top ``top_n``
    #: scores; ``"long_short"`` also shorts the bottom ``top_n``. ``top_n``
    #: ``None`` takes the universe default (50 for sp500, 10 for nasdaq100).
    rebalance_periods: int = 5
    top_n: int | None = None
    direction: str = "long_only"
    fees: float = 0.0005
    slippage: float = 0.0005
    init_cash: float = 1_000_000.0


@dataclass
class Paths:
    """Every location a pipeline reads or writes, all under the data root."""

    crsp_store: Path       # written by scripts/wrds/index.py
    membership_store: Path
    raw_dir: Path
    reference_dir: Path
    prices: Path           # written by prepare_stores
    members: Path
    alpha101: Path         # written by compute_factors
    alpha158: Path
    label: Path
    benchmark_dir: Path    # holds the ETF stores of scripts/wrds/etf.py
    work: Path             # this universe's pipeline directory

    @classmethod
    def build(cls, data: DataSettings) -> "Paths":
        if DATA_ROOT is not None:
            set_data_root(DATA_ROOT)
        root = get_data_root()
        stores = root / "data" / "us_equity" / "1d"
        downloads = root / "downloads" / "us_equity" / "1d" / "wrds_crsp"
        u = data.universe
        work = root / "data" / "pipeline" / f"wrds_{u}"
        return cls(
            crsp_store=stores / f"wrds_crsp_{u}_1d.zarr",
            membership_store=stores / f"wrds_crsp_{u}_membership.zarr",
            raw_dir=downloads / "wrds",
            reference_dir=downloads / "_reference",
            prices=work / "prices.zarr",
            members=work / "members.zarr",
            alpha101=work / "factor" / "alpha101.zarr",
            alpha158=work / "factor" / "alpha158.zarr",
            label=work / "label" / f"ret_{data.horizon}.zarr",
            benchmark_dir=stores,
            work=work,
        )


def check_settings(data: DataSettings, backtest: BacktestSettings | None = None) -> None:
    """Refuse a universe or benchmark name that is not offered."""
    if data.universe not in UNIVERSES:
        raise ValueError(f"universe must be one of {sorted(UNIVERSES)}, got {data.universe!r}")
    if backtest is not None and backtest.benchmark not in (None, "auto", *BENCHMARKS):
        raise ValueError(
            f"benchmark must be None, 'auto' or one of {sorted(BENCHMARKS)}, "
            f"got {backtest.benchmark!r}"
        )


def stock_dataset(store: Path, paths: Paths) -> StockDataset:
    """A dataset over one of the derived stores (not a raw-data converter)."""
    return StockDataset(DatasetConfig(
        zarr_file_path=str(store),
        raw_data_dir_path=str(paths.raw_dir),
        market="us_equity",
        frequency="1d",
    ))


# %% Step 1. Read CRSP and write the prices / members stores
def read_crsp(data: DataSettings, paths: Paths) -> tuple[xr.Dataset, xr.Dataset]:
    """The CRSP price panel and the index's point-in-time membership."""
    membership_cls, _ = UNIVERSES[data.universe]
    for store in (paths.crsp_store, paths.membership_store):
        if not store.exists():
            raise FileNotFoundError(
                f"{store} not found. Download and convert the CRSP roster "
                f"first: uv run python scripts/wrds/index.py --index "
                f"{data.universe} --start <start> --end <end> (see "
                f"examples/wrds_us_equity/README.md)."
            )
    crsp = CrspStockDataset(CrspDatasetConfig(
        zarr_file_path=str(paths.crsp_store),
        raw_data_dir_path=str(paths.raw_dir),
        reference_dir=str(paths.reference_dir),
    ))
    membership = membership_cls(ConstituentDatasetConfig(
        zarr_file_path=str(paths.membership_store),
        cache_dir=str(paths.reference_dir),
    ))
    # Logs every index member the price panel lacks (a survivorship gap).
    UniverseMask.from_datasets(crsp.read(), membership.read()).report()
    prices = crsp.get_xarray_dataset()[list(PANEL_COLUMNS)]
    prices = prices.sel(timestamp=slice(None, data.end_date))
    return prices, membership.get_xarray_dataset()["is_member"]


def pad_symbols(panel: xr.Dataset) -> xr.Dataset:
    """Append all-NaN PERMNOs -1, -2, ... up to a multiple of ``SYMBOL_BLOCK``."""
    n_pad = -panel.sizes["symbol"] % SYMBOL_BLOCK
    if n_pad == 0:
        return panel
    pad = np.arange(-1, -n_pad - 1, -1, dtype=panel["symbol"].dtype)
    return panel.reindex(symbol=np.concatenate([panel["symbol"].values, pad]))


def write_store(panel: xr.Dataset, store: Path, paths: Paths) -> None:
    """Write a panel and copy the CRSP ticker sidecar beside it for reports."""
    store.parent.mkdir(parents=True, exist_ok=True)
    panel.to_zarr(store, mode="w")
    sidecar = Path(str(paths.crsp_store) + TICKER_SIDECAR_SUFFIX)
    if sidecar.exists():
        shutil.copyfile(sidecar, str(store) + TICKER_SIDECAR_SUFFIX)


def prepare_stores(data: DataSettings, paths: Paths) -> None:
    """Write ``prices`` (full history) and ``members`` (non-members NaN)."""
    prices, is_member = read_crsp(data, paths)
    prices = pad_symbols(prices)
    # Membership is on calendar days, prices on trading days; days or PERMNOs
    # the membership panel does not cover count as non-members.
    member = (
        is_member.reindex(timestamp=prices.timestamp, symbol=prices.symbol)
        .fillna(False)
        .astype(bool)
    )
    write_store(prices, paths.prices, paths)
    write_store(prices.where(member), paths.members, paths)
    logger.info(
        f"prices: {dict(prices.sizes)}; member cells {int(member.sum())} "
        f"of {member.size}; stores under {paths.work}"
    )


# %% Steps 2 and 3. Factors (Alpha101 + Alpha158) and the label
def factor_objects(data: DataSettings, paths: Paths) -> tuple[list, list]:
    """``([alpha101, alpha158], [label])``; each call builds fresh datasets."""
    common = dict(
        mode="batch", start_date=data.start_date, end_date=data.end_date, njobs=data.njobs,
    )
    alpha101 = Alpha101Stock(FactorConfig(
        window=data.factor_window,
        dataset=stock_dataset(paths.prices, paths),
        data_columns=ALPHA_COLUMNS,
        factor_names=data.alpha101_names,
        file_path=str(paths.alpha101),
        **common,
    ))
    alpha158 = Alpha158Stock(FactorConfig(
        window=data.factor_window,
        dataset=stock_dataset(paths.prices, paths),
        data_columns=ALPHA_COLUMNS,
        factor_names=data.alpha158_names,
        file_path=str(paths.alpha158),
        **common,
    ))
    label = Return(FactorConfig(
        window=2 * data.horizon + 5,
        dataset=stock_dataset(paths.members, paths),
        data_columns=("adjOpen",),
        kwargs={"n_forward_periods": data.horizon},
        file_path=str(paths.label),
        **common,
    ))
    return [alpha101, alpha158], [label]


def compute_factors(data: DataSettings, paths: Paths) -> None:
    """Compute and save both alpha libraries and the label."""
    factors, labels = factor_objects(data, paths)
    for factor in factors + labels:
        factor.cal().save(mode="w")
        logger.info(
            f"{type(factor).__name__}: {len(factor.get_factor_names())} "
            f"column(s) -> {factor.config.file_path}"
        )


# %% Step 4. Model
def build_model(
    head, hyperparameters: dict, data: DataSettings, train: TrainSettings, paths: Paths,
    model_name: str,
):
    """A fresh model head reading the stored factors and label."""
    factors, labels = factor_objects(data, paths)
    return head(MLConfig(
        factors=factors,
        labels=labels,
        model_save_dir=str(paths.work / "models" / model_name),
        factor_data_strategy="read",
        label_data_strategy="read",
        start_date=data.start_date,
        end_date=data.end_date,
        train_start=train.train_start,
        train_end=train.train_end,
        test_start=train.test_start,
        test_end=train.test_end,
        early_stopping=train.early_stopping,
        early_stopping_patience=train.early_stopping_patience,
        val_size=train.val_size,
        hyperparameters=dict(hyperparameters),
    ))


def train_model(model, train: TrainSettings) -> Path:
    """Train once, or walk-forward; returns the checkpoint or the CV project dir."""
    model.collect()
    if not train.use_cv:
        checkpoint = Path(model.train())
        logger.info(f"checkpoint: {checkpoint}")
        return checkpoint
    results = model.train_cv(
        train_periods=train.cv_train_periods, gap_periods=train.cv_gap_periods
    )
    manifests = sorted(
        Path(model.config.model_save_dir).rglob("cv_folds.json"),
        key=lambda p: p.stat().st_mtime,
    )
    project_dir = manifests[-1].parent
    ic = [r.get("test_rank_ic") for r in results]
    logger.info(f"{len(results)} folds, test RankIC {ic}; project: {project_dir}")
    return project_dir


# %% Step 5. Backtest
def benchmark_name(data: DataSettings, backtest: BacktestSettings) -> str | None:
    """The benchmark key ``backtest.benchmark`` resolves to, or ``None``."""
    if backtest.benchmark == "auto":
        return DEFAULT_BENCHMARK[data.universe]
    return backtest.benchmark


def benchmark_dataset(
    data: DataSettings, backtest: BacktestSettings, paths: Paths
) -> CrspStockDataset | None:
    """The benchmark ETF's single-symbol CRSP store, or ``None``."""
    name = benchmark_name(data, backtest)
    if name is None:
        return None
    permno, file_name = BENCHMARKS[name]
    store = paths.benchmark_dir / file_name
    if not store.exists():
        raise FileNotFoundError(
            f"No {name.upper()} benchmark store at {store}. Download it by its "
            f"PERMNO {permno}: uv run python scripts/wrds/etf.py --etf {name} "
            f"--start <start> --end <end>, or set BacktestSettings.benchmark=None."
        )
    return CrspStockDataset(CrspDatasetConfig.etf_benchmark(
        permno=permno,
        zarr_file_path=str(store),
        raw_data_dir_path=str(paths.raw_dir),
        reference_dir=str(paths.reference_dir),
    ))


def run_backtest(
    model, trained: Path, data: DataSettings, train: TrainSettings,
    backtest: BacktestSettings, paths: Paths, model_name: str, use_wandb: bool,
):
    """Backtest the test window, or the stitched CV folds, and log a summary."""
    config = CrossSectionBacktestConfig(
        price_dataset=stock_dataset(paths.members, paths),
        model=model,
        model_mode="load",
        checkpoint=None if train.use_cv else str(trained),
        cv_project_dir=str(trained) if train.use_cv else None,
        start_date=train.test_start,
        end_date=train.test_end,
        output_dir=str(paths.work / "backtests" / model_name),
        rebalance_periods=backtest.rebalance_periods,
        direction=backtest.direction,
        top_n=backtest.top_n if backtest.top_n is not None else UNIVERSES[data.universe][1],
        fees=backtest.fees,
        slippage=backtest.slippage,
        init_cash=backtest.init_cash,
        use_wandb=use_wandb,
        benchmark_dataset=benchmark_dataset(data, backtest, paths),
    )
    backtester = USEquityCrossectionSelectStockVectorBt(config)
    result = backtester.run_cv() if train.use_cv else backtester.run()
    metrics = result.metrics["stitched"] if train.use_cv else result.metrics
    summary = {
        key: metrics["whole"].get(key)
        for key in ("Total Return [%]", "Sharpe Ratio", "Max Drawdown [%]")
    }
    relative = (metrics.get("relative") or {}).get("whole") or {}
    summary.update({
        key: relative[key]
        for key in (
            "benchmark_total_return", "excess_return", "excess_max_drawdown",
            "information_ratio", "beta",
        )
        if key in relative
    })
    logger.info(f"backtest {json.dumps(summary, default=str)}; run: {result.run_dir}")
    return result


def run_model_pipeline(
    head, model_name: str, hyperparameters: dict, data: DataSettings,
    train: TrainSettings, backtest: BacktestSettings, wandb_mode: str,
):
    """The five steps in order: stores, factors and label, model, backtest."""
    check_settings(data, backtest)
    # wandb reads WANDB_MODE at every wandb.init(), so this covers every run.
    os.environ["WANDB_MODE"] = wandb_mode
    paths = Paths.build(data)
    prepare_stores(data, paths)
    compute_factors(data, paths)
    model = build_model(head, hyperparameters, data, train, paths, model_name)
    trained = train_model(model, train)
    fresh = build_model(head, hyperparameters, data, train, paths, model_name)
    return run_backtest(
        fresh, trained, data, train, backtest, paths, model_name,
        use_wandb=wandb_mode != "disabled",
    )
