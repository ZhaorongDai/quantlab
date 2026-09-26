"""Alpha101 and Alpha158 factor analysis on WRDS CRSP daily data, self-contained.

CRSP daily bars of a point-in-time index (S&P 500 or Nasdaq-100) ->
Alpha101 + Alpha158 factors -> open-to-open forward-return label ->
``Factor.analyze()``: an alphalens-style report per factor column (IC
series and distribution, monthly IC, quantile returns, long-short curve,
turnover), one figure per column, written with the tables and the configs
to one directory per library.

Edit ``DATA_ROOT`` and ``Settings`` below, then run
``uv run python examples/wrds_us_equity/factor_analysis.py`` or step
through the ``# %%`` cells. Prerequisite: the CRSP stores written by
``scripts/wrds/index.py --index <universe>`` under the data root (see
README.md).
"""

# %% Settings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger

from quantlab.base.config import (
    ConstituentDatasetConfig,
    CrspDatasetConfig,
    DatasetConfig,
    FactorConfig,
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

#: Storage root. ``None`` keeps quantlab's default (``QUANTLAB_DATA_DIR`` or
#: ``data/`` beside the repository), where the WRDS scripts wrote the stores.
DATA_ROOT: str | None = None

#: Per universe: the membership panel class.
UNIVERSES = {
    "sp500": CrspSP500ConstituentDataset,
    "nasdaq100": CompustatNasdaq100ConstituentDataset,
}
#: Columns the alpha libraries read.
ALPHA_COLUMNS = ("adjOpen", "adjHigh", "adjLow", "adjClose", "adjVolume")


@dataclass
class Settings:
    """Everything this pipeline needs; nothing is read from argv."""

    #: ``"sp500"`` or ``"nasdaq100"``.
    universe: str = "sp500"
    #: Data window; the factor warm-up is read before ``start_date``.
    start_date: str = "2012-01-01"
    end_date: str = "2024-12-31"
    #: Factor lookback in calendar days, and alpha subsets (``None`` = all,
    #: which gives 82 + 169 figures).
    factor_window: int = 400
    alpha101_names: tuple[str, ...] | None = None
    alpha158_names: tuple[str, ...] | None = None
    #: KunQuant executor threads.
    njobs: int = 16
    #: Label horizon in bars: open-to-open return from t+1 to t+1+horizon.
    horizon: int = 5
    #: Equal-count factor buckets per day.
    quantiles: int = 5
    #: ``False`` reads the stores an earlier run wrote instead of rebuilding
    #: the derived stores and recomputing the factors and the label.
    recompute: bool = True


SETTINGS = Settings()


@dataclass
class Paths:
    """Every location the pipeline reads or writes, under the data root."""

    crsp_store: Path       # written by scripts/wrds/index.py
    membership_store: Path
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
        return cls(
            crsp_store=stores / f"wrds_crsp_{s.universe}_1d.zarr",
            membership_store=stores / f"wrds_crsp_{s.universe}_membership.zarr",
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
    label reads ``members`` so returns exist on member rows only. The symbol
    axis is padded with all-NaN PERMNOs to a multiple of 16, the SIMD block
    width KunQuant batch runs need.
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
    membership = UNIVERSES[s.universe](ConstituentDatasetConfig(
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


# %% 4. Analyze
def analyze(s: Settings, p: Paths) -> dict:
    """``analyze()`` per library; returns ``{"alpha101": ..., "alpha158": ...}``."""
    factors, labels = factor_objects(s, p)
    for label in labels:
        label.read()
    results = {}
    for factor in factors:
        library = type(factor).__name__.removesuffix("Stock").lower()
        out = p.work / "analysis" / library
        results[library] = factor.read().analyze(
            frets=labels, quantiles=s.quantiles, output_dir=str(out)
        )
        table = results[library].summary_table()
        best = table.reindex(table["ic_mean"].abs().sort_values(ascending=False).index).head(10)
        logger.info(
            f"{library}: {len(table)} column(s) -> {out}\n"
            f"{best[['factor', 'fret', 'ic_mean', 'ic_t_stat', 'mean_spread']]}"
        )
    return results


# %% Run everything
def main(s: Settings = SETTINGS) -> dict:
    p = Paths.build(s)
    if s.recompute:
        prepare_stores(s, p)
        compute_factors(s, p)
    return analyze(s, p)


if __name__ == "__main__":
    main()
