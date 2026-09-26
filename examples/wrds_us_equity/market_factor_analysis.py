"""Alpha101 and Alpha158 factor analysis on the whole CRSP market, from WRDS daily bars.

CRSP daily market store -> Alpha101 + Alpha158 factors -> open-to-open forward-return
label -> ``Factor.analyze()``: an alphalens-style report per factor column
(IC, quantile returns, turnover), one figure per column, written with the
tables and configs to one directory per library.

Every setting is a constant or a quantlab config object at the top of the
file; edit them and run ``uv run python examples/wrds_us_equity/market_factor_analysis.py``
or step through the ``# %%`` cells. Prerequisite: the stores written by
``scripts/wrds/market.py`` under the data root (see README.md).

The market store already holds only common stock, filtered per day at
conversion, so it is read directly through ``CrspStockDataset``: no derived
stores are written. It has thousands of PERMNOs, so the factor step needs
far more memory than an index. Narrow ``START``/``END`` or pin
``factor_names`` for a first run.
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

from quantlab.base.config import (
    CrspDatasetConfig,
    FactorConfig,
)
from quantlab.config import get_data_root
from quantlab.dataset.crsp import CrspStockDataset
from quantlab.factor.alpha101 import Alpha101Stock
from quantlab.factor.alpha158 import Alpha158Stock
from quantlab.label.fret import Return

#: Storage root: ``QUANTLAB_DATA_DIR`` or ``data/`` beside the repository,
#: where the WRDS scripts wrote the stores. Replace with ``Path("/my/root")``.
DATA_ROOT = get_data_root()
STORES = DATA_ROOT / "data" / "us_equity" / "1d"
RAW = DATA_ROOT / "downloads" / "us_equity" / "1d" / "wrds_crsp" / "wrds"
REFERENCE = DATA_ROOT / "downloads" / "us_equity" / "1d" / "wrds_crsp" / "_reference"
#: The market store of scripts/wrds/market.py, read by every step.
MARKET_STORE = STORES / "wrds_crsp_market_1d.zarr"
#: Everything this pipeline writes goes under here.
WORK = DATA_ROOT / "data" / "pipeline" / "wrds_market"

#: Data window (the factor warm-up is read before START).
START, END = "2012-01-01", "2024-12-31"
#: Label horizon in bars: open-to-open return from t+1 to t+1+HORIZON.
HORIZON = 5
#: Columns the alpha libraries read.
ALPHA_COLUMNS = ("adjOpen", "adjHigh", "adjLow", "adjClose", "adjVolume")


def market_dataset() -> CrspStockDataset:
    """A fresh dataset over the market store; each caller gets its own."""
    return CrspStockDataset(CrspDatasetConfig(
        zarr_file_path=str(MARKET_STORE), raw_data_dir_path=str(RAW),
        reference_dir=str(REFERENCE),
    ))


def factors_and_label() -> tuple[list, list]:
    """``([alpha101, alpha158], [label])``; each call builds fresh objects."""
    alpha101 = Alpha101Stock(FactorConfig(
        window=400, dataset=market_dataset(), mode="batch",
        data_columns=ALPHA_COLUMNS, file_path=str(WORK / "factor" / "alpha101.zarr"),
        start_date=START, end_date=END, njobs=16,
    ))
    alpha158 = Alpha158Stock(FactorConfig(
        window=400, dataset=market_dataset(), mode="batch",
        data_columns=ALPHA_COLUMNS, file_path=str(WORK / "factor" / "alpha158.zarr"),
        start_date=START, end_date=END, njobs=16,
    ))
    label = Return(FactorConfig(
        window=2 * HORIZON + 5, dataset=market_dataset(), mode="batch",
        data_columns=("adjOpen",), kwargs={"n_forward_periods": HORIZON},
        file_path=str(WORK / "label" / f"ret_{HORIZON}.zarr"),
        start_date=START, end_date=END, njobs=16,
    ))
    return [alpha101, alpha158], [label]


# %% 1. Factors and 2. label
def compute_factors() -> None:
    if not MARKET_STORE.exists():
        raise FileNotFoundError(
            f"{MARKET_STORE} not found; run scripts/wrds/market.py first (see README.md)."
        )
    factors, labels = factors_and_label()
    for factor in factors + labels:
        factor.cal().save(mode="w")
        logger.info(f"{type(factor).__name__} -> {factor.config.file_path}")


# %% 3. Analyze
def analyze() -> dict:
    """``Factor.analyze()`` per library; returns ``{{"alpha101": ..., "alpha158": ...}}``.

    Set ``factor_names`` to analyze a subset; the whole libraries give
    82 + 169 figures, drawn in parallel.
    """
    factors, labels = factors_and_label()
    frets = [label.read() for label in labels]
    results = {{}}
    for factor in factors:
        library = type(factor).__name__.removesuffix("Stock").lower()
        out = WORK / "analysis" / library
        results[library] = factor.read().analyze(
            frets=frets, factor_names=None, quantiles=5, output_dir=str(out)
        )
        table = results[library].summary_table()
        best = table.reindex(table["ic_mean"].abs().sort_values(ascending=False).index).head(10)
        logger.info(
            f"{{library}}: {{len(table)}} column(s) -> {{out}}\n"
            f"{{best[['factor', 'fret', 'ic_mean', 'ic_t_stat', 'mean_spread']]}}"
        )
    return results


# %% Run everything
def main() -> dict:
    compute_factors()
    return analyze()


if __name__ == "__main__":
    main()
