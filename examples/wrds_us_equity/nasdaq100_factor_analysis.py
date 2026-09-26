"""Alpha101 and Alpha158 factor analysis on the point-in-time Nasdaq-100, from WRDS CRSP daily bars.

CRSP daily bars -> Alpha101 + Alpha158 factors -> open-to-open forward-return
label -> ``Factor.analyze()``: an alphalens-style report per factor column
(IC, quantile returns, turnover), one figure per column, written with the
tables and configs to one directory per library.

Every setting is a constant or a quantlab config object at the top of the
file; edit them and run ``uv run python examples/wrds_us_equity/nasdaq100_factor_analysis.py``
or step through the ``# %%`` cells. Prerequisite: the stores written by
``scripts/wrds/index.py --index nasdaq100`` under the data root
(see README.md).
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
    ConstituentDatasetConfig,
    CrspDatasetConfig,
    DatasetConfig,
    FactorConfig,
)
from quantlab.config import get_data_root
from quantlab.dataset.constituent import CompustatNasdaq100ConstituentDataset
from quantlab.dataset.crsp import CrspStockDataset
from quantlab.dataset.stock import StockDataset
from quantlab.factor.alpha101 import Alpha101Stock
from quantlab.factor.alpha158 import Alpha158Stock
from quantlab.label.fret import Return

#: Storage root: ``QUANTLAB_DATA_DIR`` or ``data/`` beside the repository,
#: where the WRDS scripts wrote the stores. Replace with ``Path("/my/root")``.
DATA_ROOT = get_data_root()
STORES = DATA_ROOT / "data" / "us_equity" / "1d"
RAW = DATA_ROOT / "downloads" / "us_equity" / "1d" / "wrds_crsp" / "wrds"
REFERENCE = DATA_ROOT / "downloads" / "us_equity" / "1d" / "wrds_crsp" / "_reference"
#: Everything this pipeline writes goes under here.
WORK = DATA_ROOT / "data" / "pipeline" / "wrds_nasdaq100"

#: Data window (the factor warm-up is read before START).
START, END = "2012-01-01", "2024-12-31"
#: Label horizon in bars: open-to-open return from t+1 to t+1+HORIZON.
HORIZON = 5
#: Columns the alpha libraries read; ``ret`` is kept for the label's dataset.
ALPHA_COLUMNS = ("adjOpen", "adjHigh", "adjLow", "adjClose", "adjVolume")


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
    """Write ``prices`` (full history of every member ever) and ``members``
    (the same panel, NaN where the PERMNO was not a member that day).
    """
    crsp = CrspStockDataset(CrspDatasetConfig(
        zarr_file_path=str(STORES / "wrds_crsp_nasdaq100_1d.zarr"),
        raw_data_dir_path=str(RAW), reference_dir=str(REFERENCE),
    ))
    membership = CompustatNasdaq100ConstituentDataset(ConstituentDatasetConfig(
        zarr_file_path=str(STORES / "wrds_crsp_nasdaq100_membership.zarr"),
        cache_dir=str(REFERENCE),
    ))
    for store in (crsp.config.zarr_file_path, membership.config.zarr_file_path):
        if not Path(store).exists():
            raise FileNotFoundError(
                f"{store} not found; run scripts/wrds/index.py --index nasdaq100 "
                f"first (see README.md)."
            )
    prices = crsp.read().get_xarray_dataset()[[*ALPHA_COLUMNS, "close", "volume", "ret"]]
    prices = prices.sel(timestamp=slice(None, END))
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


# %% 4. Analyze
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
    prepare_stores()
    compute_factors()
    return analyze()


if __name__ == "__main__":
    main()
