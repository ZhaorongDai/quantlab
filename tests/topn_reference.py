"""The scenario behind `tests/topn_reference.npz`, the top-n regression anchor.

The file holds the weights and equity of long-only and long-short top-n
backtests. The weights were captured with `CrossSectionTopNSelector` before
top-n selection moved into the portfolio layer (#76); the equity was
recaptured for delisting settlement (#86). This module builds the same store,
model and dates, so a test re-runs the scenario through `TopNConstructor` and
compares bit for bit. Symbols list late and delist early, so the eligibility
rule (a finite score and a next-bar fill price) shapes the books.
"""

from pathlib import Path

import pandas as pd
import xarray as xr

from tests.backtest_fixtures import FirstFeatureHead, SeededHead, make_model, write_price_store

REFERENCE = Path(__file__).with_suffix(".npz")
SYMBOLS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH"]
N_BARS = 90
#: (direction, top_n, head) per captured case.
CASES = {
    "long_only": ("long_only", 3, FirstFeatureHead),
    "long_short": ("long_short", 2, SeededHead),
}
BACKTEST = dict(rebalance_periods=3, fees=0.001, slippage=0.0005, init_cash=1_000_000.0)


def _day(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def scenario(root: Path, case: str):
    """Return ``(dataset_config, model, start_date, end_date)`` of one case."""
    dataset_config = write_price_store(
        root,
        symbols=SYMBOLS,
        n_bars=N_BARS,
        seed=7,
        delist_at={"CCC": 55},
        list_at={"GGG": 45},
    )
    bars = xr.open_zarr(dataset_config.zarr_file_path).timestamp.values
    _, _, head = CASES[case]
    model = make_model(
        root / case,
        dataset_config,
        head=head,
        n=2,
        start_date=_day(bars[0]),
        end_date=_day(bars[39]),
        train_start=_day(bars[0]),
        train_end=_day(bars[34]),
        test_start=_day(bars[35]),
        test_end=_day(bars[39]),
    )
    return dataset_config, model, _day(bars[40]), _day(bars[85])
