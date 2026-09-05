# %% Cell 1
import os
from ast import Break
from pathlib import Path

import polars as pl
from joblib import Parallel, delayed
from tiingo import TiingoClient
from tqdm import tqdm

# TIINGO_API_KEY is read from the environment automatically below.
# Export it in your shell (e.g. your .bash_profile) before running this script.
if not os.environ.get("TIINGO_API_KEY"):
    raise RuntimeError(
        "TIINGO_API_KEY environment variable is not set. Export it before "
        "running this script (see Tiingo dashboard for your key)."
    )

config = {}

# To reuse the same HTTP Session across API calls (and have better performance), include a session key.
config["session"] = True

# API key comes from the environment (never hardcode it here).
config["api_key"] = os.environ["TIINGO_API_KEY"]

# Initialize
client = TiingoClient(config)

# tickers

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
