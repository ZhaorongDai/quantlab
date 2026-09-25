"""Download daily NASDAQ price history from Tiingo into per-ticker parquet.

A standalone, one-off downloader that does not use the ``quantlab`` data
registry. The ``# %%`` markers split it into cells that VS Code or Jupyter
can run one at a time. It reads the ticker list from the ``ticker`` column
of ``nasdaq_stocks.parquet`` in the repository root, or of the file named by
``QUANTLAB_NASDAQ_STOCKS_PARQUET``. It then downloads each ticker's daily
prices from 1990-08-01 to 2026-09-01 with 32 parallel workers and writes
``downloads/nasdaq_data/{ticker}/data.pqt`` relative to the current working
directory. The date range is fixed in the file. For routine downloads use
``scripts/ingest_tiingo.py`` instead, which resumes, isolates failures and
converts to Zarr. The script has no command-line options.

``TIINGO_API_KEY`` must be set in the environment. The script refuses to
start without it and never prints the key.

Usage::

    export TIINGO_API_KEY=your-key-here
    uv run python scripts/download_stock_data_from_tiingo.py

    QUANTLAB_NASDAQ_STOCKS_PARQUET=/path/to/tickers.parquet \\
        uv run python scripts/download_stock_data_from_tiingo.py
"""

# %% Cell 1
import os
from ast import Break
from pathlib import Path

import polars as pl
from joblib import Parallel, delayed
from tiingo import TiingoClient
from tqdm import tqdm

# Fail fast, before the ticker list is read, when the key is missing.
if not os.environ.get("TIINGO_API_KEY"):
    raise RuntimeError(
        "TIINGO_API_KEY environment variable is not set. Export it before "
        "running this script (see Tiingo dashboard for your key)."
    )

config = {}

# Reuse one HTTP session across API calls.
config["session"] = True

config["api_key"] = os.environ["TIINGO_API_KEY"]

client = TiingoClient(config)

# %% Download stock data from Tiingo
need_stocks = pl.scan_parquet(
    str(
        Path(
            os.environ.get(
                "QUANTLAB_NASDAQ_STOCKS_PARQUET",
                Path(__file__).resolve().parent.parent / "nasdaq_stocks.parquet",
            )
        )
    )
).collect()


need_stocks.select(pl.col("assetType")).unique()
need_stocks = need_stocks["ticker"].to_list()

# already_downloaded = [
#     stock
#     for stock in os.listdir("downloads/nasdaq_data")
#     if os.path.isdir(f"downloads/nasdaq_data/{stock}")
# ]
# need_stocks = [
#     stock for stock in need_stocks if stock not in already_downloaded
# ]

start_date = "1990-08-01"
end_date = "2026-09-01"


def download_stock(stock):
    """Download one ticker's daily prices and write them as parquet.

    The frame gets a ``timestamp`` column (the vendor's ``date``) and a
    ``symbol`` column, and lands at ``downloads/nasdaq_data/{stock}/data.pqt``
    under the current working directory. Nothing is written when the vendor
    returns no rows.

    Parameters
    ----------
    stock : str
        The ticker to download.

    Examples
    --------
    Needs ``TIINGO_API_KEY`` and network access::

        download_stock("AAPL")
    """
    data = pl.DataFrame(
        client.get_ticker_price(
            stock,
            fmt="json",
            startDate=start_date,
            endDate=end_date,
            frequency="daily",
        )
    )
    if data.is_empty():
        return

    data = data.with_columns(pl.col("date").cast(pl.Datetime))
    data = data.rename({"date": "timestamp"})
    data = data.with_columns(pl.lit(stock).alias("symbol"))

    if not os.path.exists(f"downloads/nasdaq_data/{stock}"):
        os.makedirs(f"downloads/nasdaq_data/{stock}")
    data.write_parquet(f"downloads/nasdaq_data/{stock}/data.pqt")


Parallel(n_jobs=32)(
    delayed(download_stock)(stock) for stock in tqdm(need_stocks)
)
