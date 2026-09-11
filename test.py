# %%
import polars as pl
df = pl.read_parquet('/Users/daizhaorong/projects/quantlab/data/downloads/us_equity/tick/nasdaq_data/alpaca/data_type=trades/date=2026-09-01/symbol=AAPL/part-b8c83db7772f1134-00000.pqt')
df
