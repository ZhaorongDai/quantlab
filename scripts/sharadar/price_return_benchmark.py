"""Build a price-return benchmark store for Sharadar funds, from the SFP store.

A backtest benchmark is bought and held on its ``adjOpen``/``adjClose``, which
are split- and dividend-adjusted: a total return. A price-return index (the
Bloomberg World Large, Mid & Small Cap Price Return Index, WLS, for example)
does not reinvest dividends, so a fund standing in for it must be priced
without them. This script reads each fund from the SFP store and writes a
one-symbol store whose ``adjOpen``/``adjHigh``/``adjLow``/``adjClose`` are the
raw prices adjusted for splits only (``price * cumprod(splitFactor)``, equal to
the raw price on the first bar) and whose ``divCash`` is 0. Every other variable
is the SFP store's. The store is read back with
``FrameDataset(FrameDatasetConfig(zarr_file_path=...))`` and passed as
``benchmark_dataset``.

It then prints, for every fund built, the annualised total and price return
and their gap (the dividend yield given up), and the correlation of the funds'
daily and weekly price returns with one another.

Usage::

    python scripts/sharadar/price_return_benchmark.py \\
        --tickers vt,spgm,acwi,urth \\
        --sfp-store /data/quantlab/zarrs/sharadar_sfp_1d.zarr \\
        --raw-dir /data/quantlab/downloads/sharadar \\
        --zarr-dir /data/quantlab/zarrs

writes ``<zarr-dir>/sharadar_<ticker>_pr_1d.zarr`` per ticker.
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import xarray as xr

from quantlab.dataset.config import SharadarDatasetConfig
from quantlab.dataset.memory import FrameDataset
from quantlab.dataset.sharadar.stock import SharadarStockDataset


def permaticker_of(raw_dir: Path, ticker: str) -> int:
    """Return the SFP permaticker of ``ticker`` from the TICKERS table."""
    rows = pl.read_parquet(raw_dir / "tickers" / "tickers.parquet").filter(
        (pl.col("table") == "SFP") & (pl.col("ticker") == ticker.upper())
    )
    if rows.height != 1:
        raise SystemExit(f"{ticker}: {rows.height} SFP rows in TICKERS, expected 1.")
    return int(rows["permaticker"][0])


def price_return_panel(panel: xr.Dataset) -> xr.Dataset:
    """Return ``panel`` with its adjusted prices replaced by split-only ones."""
    splits = panel["splitFactor"].fillna(1.0).cumprod("timestamp")
    first = panel["close"].notnull().argmax("timestamp")
    splits = splits / splits.isel(timestamp=first)
    out = panel.copy()
    for raw, adjusted in (("open", "adjOpen"), ("high", "adjHigh"), ("low", "adjLow"), ("close", "adjClose")):
        out[adjusted] = panel[raw] * splits
    out["divCash"] = xr.zeros_like(panel["divCash"]).where(panel["close"].notnull())
    return out


def annualised(series: pd.Series) -> float:
    """Return the annualised growth of a price series over its first to last bar."""
    series = series.dropna()
    years = (series.index[-1] - series.index[0]).days / 365.25
    return (series.iloc[-1] / series.iloc[0]) ** (1.0 / years) - 1.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tickers", default="vt", help="Comma-separated SFP tickers.")
    parser.add_argument("--sfp-store", required=True, help="The SFP Zarr store.")
    parser.add_argument("--raw-dir", required=True, help="<download-dir>/sharadar.")
    parser.add_argument("--zarr-dir", required=True, help="Where the benchmark stores go.")
    parser.add_argument("--start", default="1997-01-01")
    parser.add_argument("--end", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    parser.add_argument("--refresh", action="store_true", help="Replace existing stores.")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    sfp = SharadarStockDataset(SharadarDatasetConfig(
        zarr_file_path=args.sfp_store, raw_data_dir_path=str(raw_dir), table="sfp",
    ))
    total, price = {}, {}
    for ticker in [t.strip().lower() for t in args.tickers.split(",") if t.strip()]:
        permaticker = permaticker_of(raw_dir, ticker)
        panel = sfp.panel(args.start, args.end, symbols=[permaticker]).load()
        panel = panel.isel(timestamp=panel["close"].notnull().any("symbol").values)
        built = price_return_panel(panel)
        path = Path(args.zarr_dir) / f"sharadar_{ticker}_pr_1d.zarr"
        if path.exists():
            if not args.refresh:
                raise SystemExit(f"{path} exists; pass --refresh to replace it.")
            shutil.rmtree(path)
        FrameDataset(built).to_zarr(path)
        total[ticker] = panel["adjClose"].isel(symbol=0).to_series()
        price[ticker] = built["adjClose"].isel(symbol=0).to_series()
        print(
            f"{ticker.upper():5s} permaticker {permaticker}  {path}  "
            f"{total[ticker].index[0]:%Y-%m-%d}..{total[ticker].index[-1]:%Y-%m-%d}  "
            f"total {annualised(total[ticker]):+.2%}/yr  price {annualised(price[ticker]):+.2%}/yr  "
            f"gap {annualised(total[ticker]) - annualised(price[ticker]):.2%}/yr"
        )
    prices = pd.DataFrame(price).dropna()
    if prices.shape[1] > 1:
        print(f"\nPrice-return correlation, common window {prices.index[0]:%Y-%m-%d}..{prices.index[-1]:%Y-%m-%d}")
        print("daily:\n", np.round(prices.pct_change().corr(), 4))
        print("weekly:\n", np.round(prices.resample("W-FRI").last().pct_change().corr(), 4))


if __name__ == "__main__":
    main()
