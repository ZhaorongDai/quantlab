"""A USE4-style factor risk model on the Sharadar Barra exposures, and its bias statistics.

``BarraStyle`` exposures (the store ``barra_style.py`` builds) + Sharadar SEP
prices + DAILY market cap + FRED's 3-month T-bill rate ->
``Use4RiskModel``'s regression store (factor returns and specific returns of
country, 48 industries, 12 styles) -> its estimate store (factor covariance
and specific risk, the raw exponentially weighted model) -> bias statistics
of factor portfolios, specific returns and random active portfolios, printed,
saved as JSON and plotted.

Every setting is a constant or a quantlab config object at the top of the
file; edit them and run ``uv run python examples/sharadar_us_equity/risk_model.py``
or step through the ``# %%`` cells. Prerequisite: the stores of
``scripts/sharadar/download.py`` and the exposures of ``barra_style.py``
(see README.md). The data is licensed for personal use, so the data root must
be outside the repository.
"""

# %% Settings
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from quantlab.config import get_data_root
from quantlab.dataset.config import FredRateConfig, SharadarDailyConfig, SharadarDatasetConfig
from quantlab.dataset.fred import FredRateDataset
from quantlab.dataset.merged import MergedDataset
from quantlab.dataset.sharadar.daily import SharadarDailyDataset
from quantlab.dataset.sharadar.stock import SharadarStockDataset
from quantlab.factor.config import FactorConfig
from quantlab.factor.predefined.barra import BarraStyle, BarraStyleParameters
from quantlab.risk.bias import risk_model_bias_statistics
from quantlab.risk.config import Use4RiskConfig
from quantlab.risk.predefined.use4 import Use4RiskModel
from quantlab.utils.cli import inside_repository

#: Storage root: ``QUANTLAB_DATA_DIR`` or ``data/`` beside the repository.
DATA_ROOT = get_data_root()
STORES = DATA_ROOT / "zarrs"
VENDOR = DATA_ROOT / "downloads" / "sharadar"
FRED_RAW = DATA_ROOT / "downloads" / "fred"
#: Where ``barra_style.py`` wrote the exposures.
EXPOSURES = DATA_ROOT / "pipeline" / "sharadar_barra" / "barra_style.zarr"
#: Everything this script writes goes under here.
WORK = DATA_ROOT / "pipeline" / "sharadar_risk"

#: The regression's range: from the second bar of the exposures store (the
#: first regression reads the exposures of the bar before) to its end.
REGRESSION_START, END = "2001-01-03", "2026-10-02"
#: The estimates' range: the correlation window (1512 bars) of regression
#: rows fits before it, so no estimate uses a shortened window.
ESTIMATE_START = "2007-01-11"
#: Rolling window of the bias statistics (12 months of daily bars, as USE4).
WINDOW = 252
#: The risk-free rate is DTB3's, broadcast across the symbols, as for the exposures.
PARAMETERS = BarraStyleParameters(risk_free_symbol="DTB3")


def price_inputs() -> MergedDataset:
    """Adjusted close, market cap and the risk-free rate, on the permaticker axis."""
    return MergedDataset([
        SharadarStockDataset(SharadarDatasetConfig(
            zarr_file_path=str(STORES / "sharadar_sep_1d.zarr"), raw_data_dir_path=str(VENDOR),
        )),
        SharadarDailyDataset(SharadarDailyConfig(
            zarr_file_path=str(STORES / "sharadar_daily_1d.zarr"), raw_data_dir_path=str(VENDOR),
        )),
        FredRateDataset(FredRateConfig(
            zarr_file_path=str(STORES / "fred_dtb3_1d.zarr"), raw_data_dir_path=str(FRED_RAW),
        )),
    ])


def barra_exposures() -> BarraStyle:
    """The exposures, read from their store; computing them is ``barra_style.py``'s job."""
    return BarraStyle(FactorConfig(
        warmup_bars=PARAMETERS.warmup_bars,
        dataset=price_inputs(),
        mode="batch",
        data_columns=PARAMETERS.panel_columns,
        file_path=str(EXPOSURES),
        kwargs={"risk_free_symbol": PARAMETERS.risk_free_symbol},
    ))


def risk_model() -> Use4RiskModel:
    """USE4S's raw model: country, FF48 industries, 12 styles; EWMA half-lives 84/504/84."""
    return Use4RiskModel(Use4RiskConfig(
        exposures=barra_exposures(),
        dataset=price_inputs(),
        exposure_data_strategy="read",
        risk_free_symbol=PARAMETERS.risk_free_symbol,
        regression_path=str(WORK / "regression.zarr"),
        estimate_path=str(WORK / "estimate.zarr"),
    ))


# %% 1. The stores
def build_stores() -> dict:
    """Build the regression store, then the estimate store, and return the timing."""
    model = risk_model()
    timing = {}
    began = time.perf_counter()
    model.regression.build(REGRESSION_START, END)
    timing["regression_minutes"] = round((time.perf_counter() - began) / 60, 1)
    began = time.perf_counter()
    model.estimate.build(ESTIMATE_START, END)
    timing["estimate_minutes"] = round((time.perf_counter() - began) / 60, 1)
    logger.info(f"stores built: {timing}")
    return timing


# %% 2. Bias statistics
def bias() -> tuple[dict, dict]:
    """Return the bias statistics of every group and a summary of each."""
    began = time.perf_counter()
    stats = risk_model_bias_statistics(risk_model(), ESTIMATE_START, END, window=WINDOW)
    summary = {"bias_minutes": round((time.perf_counter() - began) / 60, 1)}
    for group, result in stats.items():
        whole, band = result["bias"], result["band"]
        # A symbol with less than a year of outcomes is left out of the summary.
        tested = whole.notnull() & (result["count"] >= WINDOW)
        inside = (abs(whole - 1.0) <= band) & tested
        summary[group] = {
            "portfolios": int(tested.sum()),
            "mean_bias": round(float(whole.where(tested).mean()), 3),
            "median_bias": round(float(whole.where(tested).median()), 3),
            "inside_band": round(float(inside.sum() / tested.sum()), 3),
            "mean_rolling_bias": round(float(result["rolling_mean"].mean()), 3),
            "mean_rolling_mrad": round(float(result["rolling_mrad"].mean()), 3),
        }
    factor = stats["factor"]
    summary["factor_bias"] = {
        str(name): round(float(value), 3)
        for name, value in zip(factor["factor"].values, factor["bias"].values)
    }
    return stats, summary


# %% 3. Figures
def plot(stats: dict) -> None:
    """Rolling 12-month mean, 5th/95th percentile and MRAD per group; bias per factor."""
    # One panel per group (factor, specific, random), then the factors' bias.
    figure, axes = plt.subplots(4, 1, figsize=(12, 16))
    for axis, (group, result) in zip(axes, stats.items()):
        bars = result["timestamp"].values
        for name in ("rolling_mean", "rolling_p5", "rolling_p95", "rolling_mrad"):
            axis.plot(bars, result[name].values, label=name.removeprefix("rolling_"), lw=0.8)
        for level in (0.0, 1.0 - np.sqrt(2 / WINDOW), 1.0, 1.0 + np.sqrt(2 / WINDOW)):
            axis.axhline(level, color="grey", ls="--", lw=0.5)
        axis.set_title(f"{group}: rolling {WINDOW}-bar bias statistics")
        axis.set_ylim(0, 2.5)
        axis.legend(loc="upper right", ncol=4)
    factor = stats["factor"]
    names = [str(n) for n in factor["factor"].values]
    axes[3].bar(range(len(names)), factor["bias"].values)
    axes[3].fill_between(
        [-0.5, len(names) - 0.5], 1 - factor["band"].values.min(), 1 + factor["band"].values.min(),
        color="grey", alpha=0.3,
    )
    axes[3].set_xticks(range(len(names)), names, rotation=90, fontsize=6)
    axes[3].set_title("factor bias statistics over the whole range")
    figure.tight_layout()
    figure.savefig(WORK / "bias.png", dpi=120)


# %% Run
if __name__ == "__main__":
    if inside_repository([DATA_ROOT], Path(__file__).resolve().parents[2]):
        raise SystemExit(f"{DATA_ROOT} is inside the repository; the data is licensed.")
    WORK.mkdir(parents=True, exist_ok=True)
    timing = build_stores()
    stats, summary = bias()
    summary = {**timing, **summary}
    (WORK / "bias_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "factor_bias"}, indent=2))
    plot(stats)
