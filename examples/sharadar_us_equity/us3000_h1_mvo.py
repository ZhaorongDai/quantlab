"""us3000 one-day mean-variance (R223L5C5) as a live recipe: prepare each day's stores, build the run.

The strategy of quantlab-experiment ``2026-10-07-h1-daily-mvo``, scheme
R223L5C5, rebuilt from library components only:

- universe: ``EstuConstituentDataset`` over BarraStyle's estimation universe
  (the 3,000 largest domestic common stocks by the previous bar's cap), on
  the #223 bad-print-masked Barra store;
- features (N1): Alpha101 and Alpha158 neutralised on FF48 industry and log
  cap (``NeutralizedFactor``), and the 12 Barra styles of the masked store
  (the experiment trained on the unmasked ones; ``feature-barra`` cuts the
  masked styles for the roster, ``train-cv`` retrains the CV on them);
- label: ``MemberReturn``, the open-to-open return from t+1 to t+2 kept where
  the symbol is a member at t;
- model: the experiment's ``XGBoostRegressor`` and walk-forward CV (ten
  expanding folds, one year of tests each, through 2026-10-02), retrained
  on these features, loaded from one fold's checkpoint (the last by
  default), masked by membership (``MembershipMaskedPredictor``);
- portfolio: ``MeanVarianceOptimizer`` over the USE4 factor risk model of
  the #223 stores, lambda 5, kappa 0.002, 5% cap, ``min_trade`` 1e-3,
  CLARABEL, ``style_beta`` and ``style_size`` in [-0.1, 0.1], the 200
  best-predicted candidates, rebalanced every bar.

The prices, alphas and label are the experiment's
(``pipeline/neutral_cv_wls/us3000`` and ``pipeline/h1_daily_mvo/us3000``).

Steps::

    # Once: the masked styles, the CV, the run quantlab-ibkr trades (a plain
    # run() in load mode, so scripts/live/predict_day.py accepts it).
    python us3000_h1_mvo.py feature-barra
    python us3000_h1_mvo.py train-cv
    python us3000_h1_mvo.py live-run [--fold N] [--size-bound X] [--beta-bound X]
    # Every morning, after scripts/sharadar/update.py and before predict_day.py:
    python us3000_h1_mvo.py prepare-day
    python scripts/live/predict_day.py <run> --store <live>/live_predictions.zarr \\
        --mirror <us3000>/prices.zarr --may-lag zarrs/fred_dtb3_1d.zarr

``prepare-day`` brings to the last SEP bar the stores ``predict_day.py``
does not extend: the roster price store (the run's price dataset), the full
masked Barra store (whose ``estu`` is the universe), the model's style
store (a slice of it), the membership and the price-return VT benchmark. Run
it on the server with ``QUANTLAB_DATA_DIR=/data/quantlab``.
"""

# %% Settings
import os
import sys

if sys.platform == "darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger

from quantlab.backtest.config import CrossSectionBacktestConfig
from quantlab.backtest.predefined.us_equity import (
    USEquityCrossectionSelectStockVectorBt,
)
from quantlab.config import get_data_root
from quantlab.dataset.bad_prints import BadPrintMaskedDataset
from quantlab.dataset.config import (
    ConstituentDatasetConfig,
    DatasetConfig,
    FrameDatasetConfig,
    FredRateConfig,
    SharadarDailyConfig,
    SharadarDatasetConfig,
    SharadarFiscalYearsConfig,
    SharadarFundamentalsConfig,
    SharadarIndustryConfig,
    SharadarShareClassConfig,
)
from quantlab.dataset.estu import EstuConstituentDataset
from quantlab.dataset.fred import FredRateDataset
from quantlab.dataset.memory import FrameDataset
from quantlab.dataset.sharadar.daily import SharadarDailyDataset
from quantlab.dataset.sharadar.fiscal_years import SharadarFiscalYearsDataset
from quantlab.dataset.sharadar.fundamentals import SharadarFundamentalsDataset
from quantlab.dataset.sharadar.industry import SharadarIndustryDataset
from quantlab.dataset.sharadar.share_class import SharadarShareClassDataset
from quantlab.dataset.sharadar.stock import SharadarStockDataset
from quantlab.dataset.stock import StockDataset
from quantlab.factor.config import FactorConfig, NeutralizedConfig
from quantlab.factor.predefined.alpha101 import Alpha101Stock
from quantlab.factor.predefined.alpha158 import Alpha158Stock
from quantlab.factor.predefined.barra import BarraStyle, BarraStyleParameters
from quantlab.factor.predefined.neutralized import NeutralizedFactor
from quantlab.label.predefined.member_return import MemberReturn
from quantlab.model.config import ModelConfig
from quantlab.model.predefined.membership_mask import MembershipMaskedPredictor
from quantlab.model.predefined.xgb import XGBoostRegressor
from quantlab.portfolio.config import FactorRiskStoreEstimatorConfig, MeanVarianceConfig
from quantlab.portfolio.predefined.factor_risk import FactorRiskStoreEstimator
from quantlab.portfolio.predefined.mean_variance import MeanVarianceOptimizer
from quantlab.risk.config import Use4RiskConfig
from quantlab.risk.predefined.use4 import Use4RiskModel
from quantlab.tracking.wandb import WandbTracker

DATA_ROOT = get_data_root()
STORES = DATA_ROOT / "zarrs"
VENDOR = DATA_ROOT / "downloads" / "sharadar"
FRED_RAW = DATA_ROOT / "downloads" / "fred"
#: The us3000 prices, members, alphas, neutral alphas and Barra subset.
US3000 = DATA_ROOT / "pipeline" / "neutral_cv_wls" / "us3000"
#: The one-day label, the walk-forward CV and the live runs.
H1 = DATA_ROOT / "pipeline" / "h1_daily_mvo" / "us3000"
#: The Barra exposures (with ``estu``) and USE4 stores, bad prints masked since #223.
EXPOSURES = DATA_ROOT / "pipeline" / "sharadar_barra" / "barra_style.zarr"
RISK = DATA_ROOT / "pipeline" / "sharadar_risk"
#: The 12 masked styles of the roster: the model's Barra features.
FEATURE_BARRA = US3000 / "factor" / "barra_style_223.zarr"
#: The CV over the masked features and its record.
MODELS = H1 / "models_223"
CV_RECORD = H1 / "cv_223.json"
#: Walk-forward CV: expanding, ten one-year test folds ending on END.
TRAIN_PERIODS, TEST_PERIODS = 1189, 252
#: The membership store this recipe maintains (the experiment's own one is frozen).
MEMBERSHIP = US3000 / "membership_estu.zarr"
#: The price-return VT benchmark (scripts/sharadar/price_return_benchmark.py).
BENCHMARK = STORES / "sharadar_vt_pr_1d.zarr"

#: Prices from here (warm-up of the alphas); features and label from START.
PRICE_START, START, END = "2010-01-01", "2012-01-01", "2026-10-02"
HORIZON = 1
ALPHA_COLUMNS = ("adjOpen", "adjHigh", "adjLow", "adjClose", "adjVolume")
PARAMETERS = BarraStyleParameters(risk_free_symbol="DTB3")
STYLES = tuple(name for name in BarraStyle._OUTPUTS if name.startswith("style_"))
#: R223L5C5.
OPTIMISER = {
    "risk_aversion": 5.0, "turnover_penalty": 0.002, "weight_cap": 0.05, "candidate_top_k": 200,
    "min_trade": 1e-3, "solver": "CLARABEL",
    "exposure_bounds": {"style_beta": (-0.1, 0.1), "style_size": (-0.1, 0.1)},
}
#: The live run's backtest window: a month the stores already hold.
RUN_START, RUN_END = "2026-09-01", "2026-10-02"
TRACKER = WandbTracker(mode="offline")


def stock_dataset(store: Path) -> StockDataset:
    return StockDataset(DatasetConfig(
        zarr_file_path=str(store), raw_data_dir_path=str(VENDOR), market="us_equity", frequency="1d",
    ))


def sharadar_inputs() -> list:
    """The Sharadar panels and the risk-free rate BarraStyle reads."""
    def store(name: str) -> str:
        return str(STORES / name)

    return [
        SharadarStockDataset(SharadarDatasetConfig(
            zarr_file_path=store("sharadar_sep_1d.zarr"), raw_data_dir_path=str(VENDOR))),
        SharadarDailyDataset(SharadarDailyConfig(
            zarr_file_path=store("sharadar_daily_1d.zarr"), raw_data_dir_path=str(VENDOR))),
        SharadarFundamentalsDataset(SharadarFundamentalsConfig(
            zarr_file_path=store("sharadar_sf1_art.zarr"), raw_data_dir_path=str(VENDOR), dimension="ART")),
        SharadarFiscalYearsDataset(SharadarFiscalYearsConfig(
            zarr_file_path=store("sharadar_sf1_fiscal_years.zarr"), raw_data_dir_path=str(VENDOR))),
        SharadarIndustryDataset(SharadarIndustryConfig(
            zarr_file_path=store("sharadar_industry_1d.zarr"), raw_data_dir_path=str(VENDOR))),
        SharadarShareClassDataset(SharadarShareClassConfig(
            zarr_file_path=store("sharadar_share_class_1d.zarr"), raw_data_dir_path=str(VENDOR))),
        FredRateDataset(FredRateConfig(
            zarr_file_path=store("fred_dtb3_1d.zarr"), raw_data_dir_path=str(FRED_RAW))),
    ]


def masked_barra() -> BarraStyle:
    """The full BarraStyle store of #223 (bad prints masked): the risk model's exposures and ``estu``."""
    return BarraStyle(FactorConfig(
        warmup_bars=PARAMETERS.warmup_bars, dataset=BadPrintMaskedDataset(sharadar_inputs()),
        mode="batch", data_columns=PARAMETERS.panel_columns, file_path=str(EXPOSURES),
        kwargs={"risk_free_symbol": PARAMETERS.risk_free_symbol}, njobs=64,
    ))


def feature_barra() -> BarraStyle:
    """The model's 12 styles: the masked BarraStyle, written for the roster to ``FEATURE_BARRA``."""
    return BarraStyle(FactorConfig(
        warmup_bars=PARAMETERS.warmup_bars, dataset=BadPrintMaskedDataset(sharadar_inputs()),
        mode="batch", data_columns=PARAMETERS.panel_columns,
        file_path=str(FEATURE_BARRA), factor_names=STYLES,
        kwargs={"risk_free_symbol": PARAMETERS.risk_free_symbol}, njobs=64,
    ))


def membership() -> EstuConstituentDataset:
    return EstuConstituentDataset(ConstituentDatasetConfig(
        zarr_file_path=str(MEMBERSHIP), cache_dir=str(VENDOR), start_date=PRICE_START,
        kwargs={"barra_store": str(EXPOSURES)},
    ))


def price_dataset() -> SharadarStockDataset:
    """The roster price store: every security in the universe from PRICE_START to END."""
    roster = tuple(int(s) for s in xr.open_zarr(US3000 / "prices.zarr")["symbol"].values)
    return SharadarStockDataset(SharadarDatasetConfig(
        zarr_file_path=str(US3000 / "backtest_prices_1d.zarr"), raw_data_dir_path=str(VENDOR),
        permatickers=roster,
    ))


def features() -> list:
    """N1: the neutralised Alpha101/158 and the 12 styles."""
    exposures = [
        SharadarDailyDataset(SharadarDailyConfig(
            zarr_file_path=str(STORES / "sharadar_daily_1d.zarr"), raw_data_dir_path=str(VENDOR))),
        SharadarIndustryDataset(SharadarIndustryConfig(
            zarr_file_path=str(STORES / "sharadar_industry_1d.zarr"), raw_data_dir_path=str(VENDOR))),
    ]
    alphas = [
        cls(FactorConfig(
            warmup_bars=400, dataset=stock_dataset(US3000 / "prices.zarr"), mode="batch",
            data_columns=ALPHA_COLUMNS, file_path=str(US3000 / "factor" / f"{name}.zarr"), njobs=64,
        ))
        for cls, name in ((Alpha101Stock, "alpha101"), (Alpha158Stock, "alpha158"))
    ]
    neutral = [
        NeutralizedFactor(NeutralizedConfig(
            factor=alpha, dataset=exposures, regressors=("industry", "size"),
            file_path=str(US3000 / "factor" / f"{Path(alpha.config.file_path).stem}_neutral.zarr"), njobs=64,
        ))
        for alpha in alphas
    ]
    return [*neutral, feature_barra()]


def label() -> MemberReturn:
    return MemberReturn(FactorConfig(
        warmup_bars=2 * HORIZON + 5, dataset=stock_dataset(US3000 / "prices.zarr"), mode="batch",
        data_columns=("adjOpen",),
        kwargs={"n_forward_periods": HORIZON, "members_store": str(US3000 / "members.zarr")},
        file_path=str(H1 / "label" / f"ret_{HORIZON}.zarr"), njobs=64,
    ))


def model() -> XGBoostRegressor:
    """The experiment's XGBoost (its hyperparameters; the checkpoint carries the fit)."""
    return XGBoostRegressor(ModelConfig(
        tracker=TRACKER, factors=features(), labels=[label()],
        model_save_dir=str(MODELS), factor_data_strategy="read", label_data_strategy="read",
        start_date=START, end_date=END, train_start=START, train_end="2019-12-31",
        test_start="2020-01-01", test_end=END, val_size=0.2,
        hyperparameters={"training_target": "cs_rank", "early_stopping": True, "early_stopping_patience": 50,
                         "objective": "reg:squarederror", "num_boost_round": 1000,
                         "eta": 0.05, "max_depth": 3, "min_child_weight": 200, "nthread": 32},
    ))


def risk_model() -> Use4RiskModel:
    return Use4RiskModel(Use4RiskConfig(
        exposures=masked_barra(),
        dataset=BadPrintMaskedDataset([sharadar_inputs()[0], sharadar_inputs()[1], sharadar_inputs()[-1]]),
        exposure_data_strategy="read", risk_free_symbol=PARAMETERS.risk_free_symbol,
        regression_path=str(RISK / "regression.zarr"), estimate_path=str(RISK / "estimate.zarr"),
    ))


def cv_record() -> dict:
    return json.loads(CV_RECORD.read_text())


# %% Once: the masked styles and the CV
def cut_feature_barra() -> dict:
    """Write the roster's 12 styles from the masked store over START..END, with their range."""
    roster = xr.open_zarr(US3000 / "prices.zarr")["symbol"].values
    full = xr.open_zarr(EXPOSURES)
    keep = np.intersect1d(roster, full["symbol"].values)
    subset = full[list(STYLES)].sel(timestamp=slice(START, END), symbol=keep).load()
    for name in subset.data_vars:
        subset[name].encoding = {}
    FEATURE_BARRA.parent.mkdir(parents=True, exist_ok=True)
    subset.to_zarr(FEATURE_BARRA, mode="w")
    Path(f"{FEATURE_BARRA}.range.json").write_text(json.dumps({"start": START, "end": END}))
    done = {"store": str(FEATURE_BARRA), "sizes": dict(subset.sizes), "missing_from_barra": int(len(roster) - len(keep))}
    logger.info(json.dumps(done))
    return done


def train_cv() -> dict:
    """The experiment's walk-forward CV over these features; its record goes to ``CV_RECORD``."""
    cv = model().collect().train_cv(train_periods=TRAIN_PERIODS, expanding=True, test_periods=TEST_PERIODS)
    record = {"path": str(cv.path), "cv_mean": cv.cv_mean, "folds": [
        {"index": f.index, "test": [str(d)[:10] for d in f.test_window], "metrics": f.metrics} for f in cv.folds
    ]}
    CV_RECORD.write_text(json.dumps(record, indent=2, default=str))
    logger.info(f"CV {cv.path}: {json.dumps(cv.cv_mean, default=str)[:600]}")
    return record


def fold_checkpoint(fold: int | None) -> Path:
    """The checkpoint of one fold of the experiment's CV (the last by default)."""
    record = cv_record()
    index = max(f["index"] for f in record["folds"]) if fold is None else fold
    path = Path(record["path"]) / f"fold_{index}" / f"XGBoostRegressor_cv_fold_{index}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"no checkpoint for fold {index}: {path}")
    return path


def append_feature_barra() -> int:
    """Append the masked store's new bars of the 12 styles to ``FEATURE_BARRA``, on its own symbols.

    The feature store stays a slice of the masked store, never computed on
    its own; ``predict_day.py`` then finds it current. Returns the bars appended.
    """
    stored = xr.open_zarr(FEATURE_BARRA)
    last = pd.Timestamp(stored["timestamp"].values[-1])
    full = xr.open_zarr(EXPOSURES)
    new = full[list(STYLES)].sel(timestamp=full["timestamp"] > np.datetime64(last))
    if new.sizes["timestamp"] == 0:
        return 0
    new = new.reindex(symbol=stored["symbol"].values).load()
    for name in new.data_vars:
        new[name].encoding = {}
    new.to_zarr(FEATURE_BARRA, append_dim="timestamp")
    end = str(pd.Timestamp(new["timestamp"].values[-1]).date())
    Path(f"{FEATURE_BARRA}.range.json").write_text(json.dumps({"start": START, "end": end}))
    return int(new.sizes["timestamp"])


def refresh_benchmark() -> None:
    """Rebuild the price-return VT benchmark from the (updated) SFP store.

    quantlab-ibkr reads a live day's benchmark from this store, so it must
    hold every bar the trader marks.
    """
    import subprocess

    script = Path(__file__).resolve().parents[2] / "scripts" / "sharadar" / "price_return_benchmark.py"
    subprocess.run(
        [sys.executable, str(script), "--tickers", "vt", "--sfp-store", str(STORES / "sharadar_sfp_1d.zarr"),
         "--raw-dir", str(VENDOR), "--zarr-dir", str(STORES), "--refresh"],
        check=True,
    )


# %% Every morning: the stores predict_day.py does not extend
def prepare_day() -> dict:
    prices = price_dataset()
    prices.update()
    last = pd.Timestamp(xr.open_zarr(prices.config.zarr_file_path)["timestamp"].values[-1])
    barra = masked_barra()
    _, end = barra.store_range()
    if pd.Timestamp(end) < last:
        # A date, so the recorded range holds the whole day (a Timestamp ends it at midnight).
        barra.extend(last.date().isoformat())
    styles = append_feature_barra()
    members = membership()
    members.update()
    refresh_benchmark()
    done = {
        "feature_barra_appended": styles,
        "t": str(last.date()),
        "prices": str(prices.config.zarr_file_path),
        "barra_store": barra.store_range(),
        "membership_last": str(pd.Timestamp(xr.open_zarr(MEMBERSHIP)["timestamp"].values[-1]).date()),
    }
    logger.info(json.dumps(done))
    return done


# %% Once: the live run
def live_run(
    fold: int | None,
    candidate_top_k: int | None = OPTIMISER["candidate_top_k"],
    size_bound: float | None = 0.1,
    beta_bound: float | None = 0.1,
) -> Path:
    """A plain run() of R223L5C5 in load mode with one fold's checkpoint, over RUN_START..RUN_END.

    ``candidate_top_k=None`` optimises over every member: from an empty book
    the 200 best-predicted names are mostly small caps, and a fully invested
    book capped at 5% a name cannot hold ``style_size`` above -0.1 with them
    (the backtest's first bars from cash were infeasible too; later its held
    mega caps stayed candidates). ``size_bound`` widens the ``style_size``
    bound to ``[-size_bound, size_bound]`` instead (0.75 was feasible from
    an empty book on every bar of the year to 2026-10-02, 0.1 on 62% of
    them), and ``None`` drops it (the experiment's R223L5C5NS);
    ``beta_bound`` does the same for ``style_beta``.
    """
    bounds = dict(OPTIMISER["exposure_bounds"])
    if size_bound is None:
        bounds.pop("style_size")
    else:
        bounds["style_size"] = (-size_bound, size_bound)
    if beta_bound is None:
        bounds.pop("style_beta")
    else:
        bounds["style_beta"] = (-beta_bound, beta_bound)
    if not MEMBERSHIP.exists():
        membership().update()
    optimiser = MeanVarianceOptimizer(MeanVarianceConfig(
        expected_return_label=f"ret_{HORIZON}",
        covariance=FactorRiskStoreEstimator(FactorRiskStoreEstimatorConfig(risk_model=risk_model())),
        ic=max(float(cv_record()["cv_mean"]["cv_mean_val_ic"]), 0.01), direction="long_only",
        **{**OPTIMISER, "candidate_top_k": candidate_top_k, "exposure_bounds": bounds},
    ))
    result = USEquityCrossectionSelectStockVectorBt(CrossSectionBacktestConfig(
        price_dataset=price_dataset(),
        model=MembershipMaskedPredictor(model(), membership()),
        model_mode="load", checkpoint=str(fold_checkpoint(fold)),
        start_date=RUN_START, end_date=RUN_END, output_dir=str(H1 / "live_runs"),
        rebalance_periods=1, constructor=optimiser, fees=0.0005, slippage=0.0005,
        init_cash=1_000_000.0, tracker=TRACKER,
        benchmark_dataset=FrameDataset(FrameDatasetConfig(zarr_file_path=str(BENCHMARK))),
    )).run()
    logger.info(f"live run {result.run_dir}")
    return Path(result.run_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="step", required=True)
    sub.add_parser("prepare-day")
    sub.add_parser("feature-barra")
    sub.add_parser("train-cv")
    run = sub.add_parser("live-run")
    run.add_argument("--fold", type=int, default=None)
    run.add_argument("--all-candidates", action="store_true", help="candidate_top_k=None (every member)")
    run.add_argument("--size-bound", type=float, default=0.1,
                     help="style_size in [-x, x]; 0 drops the bound (R223L5C5NS)")
    run.add_argument("--beta-bound", type=float, default=0.1, help="style_beta in [-x, x]; 0 drops the bound")
    args = parser.parse_args()
    if args.step == "prepare-day":
        prepare_day()
    elif args.step == "feature-barra":
        cut_feature_barra()
    elif args.step == "train-cv":
        train_cv()
    else:
        top_k = None if args.all_candidates else OPTIMISER["candidate_top_k"]
        print(live_run(args.fold, top_k, size_bound=args.size_bound or None, beta_bound=args.beta_bound or None))


if __name__ == "__main__":
    main()
