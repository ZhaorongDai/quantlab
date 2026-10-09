"""Residual momentum on the whole CRSP market, from WRDS daily bars and Fama-French factors.

CRSP daily market store + Fama-French daily factors -> ``ResidualMomentumFF3``
(rolling three-factor regression, formation-period residual sum over its
volatility) -> open-to-open forward-return label -> ``Factor.analyze()``: an
alphalens-style report of the score and of its cross-sectional rank, written
with the tables and configs to one directory.

Every setting is a constant or a quantlab config object at the top of the
file; edit them and run
``uv run python examples/wrds_us_equity/market_residual_momentum.py`` or
step through the ``# %%`` cells. Prerequisites: the store written by
``scripts/wrds/market.py`` and the CSV written by ``scripts/fama_french.py``
under the data root (see README.md).

The factor reads only the panel's ``ret`` (CRSP's daily return) and takes
the four Fama-French series from the CSV, so no derived store is written.
The regression and formation windows are counted in daily bars: three years,
twelve months minus the most recent one.
"""

# %% Settings
import os
import sys

# macOS only: xgboost and torch ship different OpenMP runtimes that clash in
# one process unless OpenMP runs single-threaded. Set before either imports.
if sys.platform == "darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "1")

from pathlib import Path

from loguru import logger

from quantlab.factor.config import FactorConfig
from quantlab.dataset.config import CrspDatasetConfig
from quantlab.config import get_data_root
from quantlab.dataset.crsp import CrspStockDataset
from quantlab.factor.predefined.residual_momentum import ResidualMomentumFF3
from quantlab.label.predefined.fret import Return

#: Short name of this pipeline family, used in the stores' README.md.
UNIVERSE = "market"


def store_path(folder: Path, stem: str) -> Path:
    """``<folder>/<stem>/<stem>.zarr``: a store in its own folder, beside its README.md."""
    return folder / stem / f"{stem}.zarr"


def store_folder(store: Path, about: str) -> None:
    """Create a derived store's folder and, when it has none, a short README.md."""
    store.parent.mkdir(parents=True, exist_ok=True)
    readme = store.parent / "README.md"
    if not readme.exists():
        readme.write_text(
            f"# {store.stem}\n\n{about}\n\n"
            f"Written and read by the examples/wrds_us_equity/{UNIVERSE}_*.py pipelines.\n"
        )


#: Storage root: ``QUANTLAB_DATA_DIR`` or ``data/`` beside the repository,
#: where the WRDS scripts wrote the stores. Replace with ``Path("/my/root")``.
DATA_ROOT = get_data_root()
#: The WRDS scripts' stores, one folder each: price panels and ETF bars
#: under MARKET_DIR, membership panels under UNIVERSE_DIR.
MARKET_DIR = DATA_ROOT / "market" / "wrds"
UNIVERSE_DIR = DATA_ROOT / "universe" / "wrds"
RAW = DATA_ROOT / "downloads" / "us_equity" / "1d" / "wrds_crsp" / "wrds"
REFERENCE = DATA_ROOT / "downloads" / "us_equity" / "1d" / "wrds_crsp" / "_reference"
#: The market store of scripts/wrds/market.py, read by every step.
MARKET_STORE = store_path(MARKET_DIR, "wrds_crsp_market_1d")
#: The daily Fama-French CSV of scripts/fama_french.py.
FAMA_FRENCH_CSV = DATA_ROOT / "downloads" / "fama_french" / "ff3_daily.csv"
#: Shared stores this pipeline derives, which other experiments can reuse.
FACTORS_DIR = DATA_ROOT / "factors" / "wrds_market"
LABELS_DIR = DATA_ROOT / "labels" / "wrds_market"
#: This pipeline's own models, backtests and reports.
WORK = DATA_ROOT / "runs" / "wrds_market"

#: Data window (the factor warm-up is read before START).
START, END = "2012-01-01", "2024-12-31"
#: Bars of warm-up read before START; they cover the 756-bar regression.
WARMUP_BARS = 800
#: Regression, formation and skip windows in daily bars (3 years, 12 months, 1 month).
REGRESSION_WINDOW, FORMATION_LOOKBACK, SKIP_RECENT = 756, 252, 21
#: Label horizon in bars: open-to-open return from t+1 to t+1+HORIZON.
HORIZON = 5


def market_dataset() -> CrspStockDataset:
    """A fresh dataset over the market store; each caller gets its own."""
    return CrspStockDataset(CrspDatasetConfig(
        zarr_file_path=str(MARKET_STORE), raw_data_dir_path=str(RAW),
        reference_dir=str(REFERENCE),
    ))


def factor_and_label() -> tuple[ResidualMomentumFF3, Return]:
    """``(factor, label)``; each call builds fresh objects."""
    factor = ResidualMomentumFF3(FactorConfig(
        warmup_bars=WARMUP_BARS, dataset=market_dataset(), mode="batch",
        data_columns=("ret",), factor_names=("resmom_raw", "resmom_rank"),
        file_path=str(store_path(FACTORS_DIR, "residual_momentum")),
        njobs=16,
        kwargs={
            "fama_french_csv": str(FAMA_FRENCH_CSV),
            "regression_window": REGRESSION_WINDOW,
            "formation_lookback": FORMATION_LOOKBACK,
            "skip_recent": SKIP_RECENT,
            "emit_diagnostics": False,
        },
    ))
    label = Return(FactorConfig(
        warmup_bars=2 * HORIZON + 5, dataset=market_dataset(), mode="batch",
        data_columns=("adjOpen",), kwargs={"n_forward_periods": HORIZON},
        file_path=str(store_path(LABELS_DIR, f"ret_{HORIZON}")),
        njobs=16,
    ))
    return factor, label


# %% 1. Factor and 2. label
def compute_factor_and_label() -> None:
    for path, script in ((MARKET_STORE, "scripts/wrds/market.py"),
                         (FAMA_FRENCH_CSV, "scripts/fama_french.py")):
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; run {script} first (see README.md).")
    factor, label = factor_and_label()
    store_folder(Path(factor.config.file_path), f"{type(factor).__name__} factor panel.")
    factor.build(START, END)
    logger.info(f"{type(factor).__name__} -> {factor.config.file_path}")
    # A label is stored as the factor it shifts forward; the store is shared
    # with market_factor_analysis.py.
    store_folder(Path(label.config.factor.config.file_path), (
        "Forward label, stored as the factor panel it shifts forward."
    ))
    label.build(START, END)
    logger.info(f"{type(label).__name__} -> {label.config.factor.config.file_path}")


# %% 3. Analyze
def analyze():
    """``Factor.analyze()`` of the score and its rank against the forward return."""
    factor, label = factor_and_label()
    out = WORK / "analysis" / "residual_momentum"
    result = factor.analyze(
        START, END, frets=[label], factor_names=None, quantiles=5,
        output_dir=str(out), data_strategy="read",
    )
    table = result.summary_table()
    logger.info(
        f"residual momentum: {len(table)} column(s) -> {out}\n"
        f"{table[['factor', 'fret', 'ic_mean', 'ic_t_stat', 'mean_spread', 'cumulative_long_short']]}"
    )
    return result


# %% Run everything
def main():
    compute_factor_and_label()
    return analyze()


if __name__ == "__main__":
    main()
