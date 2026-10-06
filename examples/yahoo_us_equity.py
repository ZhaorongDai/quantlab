"""The whole pipeline on free Yahoo Finance data, held in memory.

Daily bars of the 30 Dow Jones Industrial Average stocks and the SPY ETF are
downloaded with ``yfinance`` and handed to quantlab as pandas frames through
``FrameDataset``: no Zarr store, no download tree and no credentials. From
there the pipeline is the same one the WRDS and Sharadar examples run:

1. download the bars, keep a snapshot, and wrap them in ``FrameDataset``,
2. compute the Alpha158 factor set and a 5-bar forward-return label,
3. train an XGBoost return model on 2016-2021 and test it on 2022,
4. backtest a long-only top-10 portfolio on 2023-2024 against buy-and-hold SPY,
5. read the metrics and rebuild the run from its directory.

The universe is today's Dow 30, so the sample suffers from survivorship bias:
every stock in it survived to the present. This is a demonstration of the
pipeline, not a strategy; use the point-in-time CRSP or Sharadar universes for
research.

``yfinance`` is not a quantlab dependency. Run the script from the repository
root with::

    uv run --with yfinance python examples/yahoo_us_equity.py

Everything is written under ``output/yahoo_us_equity/``: the downloaded bars
(``bars.parquet``), the model checkpoint and the backtest run directory with
``report.html``. Yahoo's adjusted prices differ in the last digits from one
request to the next, so the first run snapshots the bars and later runs read
the snapshot; delete it to download again.
"""

import os
import sys

# The environment must be set before torch or xgboost is imported.
# macOS only: xgboost and torch ship different OpenMP runtimes that clash in
# one process unless OpenMP runs single-threaded.
if sys.platform == "darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "1")

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from loguru import logger

from quantlab.backtest.config import CrossSectionBacktestConfig
from quantlab.backtest.predefined.us_equity import USEquityCrossectionSelectStockVectorBt
from quantlab.dataset.memory import FrameDataset
from quantlab.factor.config import FactorConfig
from quantlab.factor.predefined.alpha158 import Alpha158Stock
from quantlab.label.predefined.fret import Return
from quantlab.model.config import ModelConfig
from quantlab.model.predefined.xgb import XGBoostRegressor
from quantlab.portfolio.config import TopNConfig
from quantlab.portfolio.predefined.top_n import TopNConstructor
from quantlab.runs.backtest_run import BacktestRun
from quantlab.runs.trained_run import TrainedRun

warnings.filterwarnings("ignore", message="Consolidated metadata")
logger.remove()
logger.add(sys.stderr, level="WARNING")

# The Dow Jones Industrial Average as of 2026; see the survivorship note above.
DOW_30 = [
    "AAPL", "AMGN", "AMZN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS",
    "GS", "HD", "HON", "IBM", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK",
    "MSFT", "NKE", "NVDA", "PG", "SHW", "TRV", "UNH", "V", "VZ", "WMT",
]
BENCHMARK = "SPY"
START, END = "2015-06-01", "2025-01-01"
HORIZON = 5
OUTPUT = Path("output/yahoo_us_equity")

# yfinance's split- and dividend-adjusted fields, named as the stock factors
# and the US-equity backtester read them.
COLUMNS = {
    "Date": "timestamp", "Ticker": "symbol", "Open": "adjOpen", "High": "adjHigh",
    "Low": "adjLow", "Close": "adjClose", "Volume": "adjVolume",
}
ADJUSTED = ("adjOpen", "adjHigh", "adjLow", "adjClose", "adjVolume")


def download(tickers: list[str]) -> pd.DataFrame:
    """Adjusted daily bars as a long frame, one row per date and ticker.

    The first call downloads and writes the snapshot; later calls read it.
    """
    snapshot = OUTPUT / "bars.parquet"
    if snapshot.exists():
        return pd.read_parquet(snapshot)
    wide = yf.download(tickers, start=START, end=END, auto_adjust=True, progress=False)
    long = wide.stack(level="Ticker", future_stack=True).reset_index()
    long = long.dropna(subset=["Close"])
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    long.to_parquet(snapshot)
    return long


def make_model(prices: FrameDataset, root: Path) -> XGBoostRegressor:
    """The factors, the label and the model, as plain config objects."""
    factor = Alpha158Stock(
        FactorConfig(
            warmup_bars=60,  # Alpha158 looks back up to 60 bars
            dataset=prices,
            mode="batch",
            data_columns=ADJUSTED,
            file_path=str(root / "factors" / "alpha158.zarr"),
            njobs=4,
        )
    )
    label = Return(
        FactorConfig(
            warmup_bars=0,
            dataset=prices,
            mode="batch",
            data_columns=("adjOpen",),
            kwargs={"n_forward_periods": HORIZON},
            file_path=str(root / "labels" / f"ret_{HORIZON}.zarr"),
            njobs=4,
        )
    )
    return XGBoostRegressor(
        ModelConfig(
            factors=[factor],
            labels=[label],
            model_save_dir=str(root / "models"),
            factor_data_strategy="cal",
            label_data_strategy="cal",
            hyperparameters={
                "num_boost_round": 200, "max_depth": 3, "eta": 0.03,
                "subsample": 0.8, "colsample_bytree": 0.5,
            },
            val_size=0.2,
            start_date="2016-01-01",
            end_date="2022-12-31",
            train_start="2016-01-01",
            train_end="2021-12-31",
            test_start="2022-01-01",
            test_end="2022-12-31",
        )
    )


def main() -> None:
    """Download, train, backtest and rebuild."""
    # 1. Data: pandas frames in, FrameDataset out ------------------------------
    bars = download(DOW_30 + [BENCHMARK])
    stocks = bars[bars["Ticker"] != BENCHMARK]
    spy = bars[bars["Ticker"] == BENCHMARK]
    print(f"{len(stocks):,} rows for {stocks['Ticker'].nunique()} stocks, {len(spy):,} for {BENCHMARK}")
    prices = FrameDataset(stocks, columns=COLUMNS)
    print("Price panel:", dict(prices.panel(START, END).sizes))

    # 2-3. Factors, label and model --------------------------------------------
    model = make_model(prices, OUTPUT)
    print(f"{len(model.get_factor_names())} factors; label {model.get_label_names()}")
    model.collect()  # compute factors and label into one panel
    checkpoint = model.train()
    test = TrainedRun.open(checkpoint).metrics
    print(f"2022 test: IC {test['test_ic']:.4f}, RankIC {test['test_rank_ic']:.4f}")

    # 4. Backtest out of sample against buy-and-hold SPY -----------------------
    backtester = USEquityCrossectionSelectStockVectorBt(
        CrossSectionBacktestConfig(
            price_dataset=prices,
            benchmark_dataset=FrameDataset(spy, columns=COLUMNS),
            model=make_model(prices, OUTPUT),
            model_mode="load",
            checkpoint=str(checkpoint),
            start_date="2023-01-01",
            end_date="2024-12-31",
            output_dir=str(OUTPUT / "backtests"),
            rebalance_periods=HORIZON,
            constructor=TopNConstructor(TopNConfig(direction="long_only", top_n=10)),
        )
    )
    result = backtester.run()

    # 5. Results ---------------------------------------------------------------
    strategy, spy_stats = result.metrics["whole"], result.metrics["benchmark"]["whole"]
    print(f"{'2023-2024':<18}{'top-10':>10}{'SPY':>10}")
    for key in ("Total Return [%]", "Sharpe Ratio", "Max Drawdown [%]"):
        print(f"{key:<18}{strategy[key]:>10.2f}{spy_stats[key]:>10.2f}")
    print("Run directory, with the HTML report:", result.run_dir)

    # The run directory holds every config; rebuilding it reproduces the curve.
    again = BacktestRun.open(result.run_dir).rebuild_backtester().run()
    same = np.allclose(again.simulation.value.values, result.simulation.value.values)
    print("Rebuilt from its run directory, same equity curve:", same)


if __name__ == "__main__":
    main()
