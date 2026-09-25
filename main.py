"""Ad hoc example: read three symbols from a US-equity Zarr store.

The script opens a daily US-equity Zarr store (a chunked on-disk array
format that ``xarray`` reads and writes) through ``StockDataset``, restricted
to NVDA, AMZN and AMD, and prints NVDA's rows as a pandas frame. It is a
quick way to check that a store is readable. It is not part of the library
and has no command-line options.

The store path is hardcoded and machine-specific. Edit ``zarr_file_path``
to point at your own store before running. No credentials are needed.

Usage::

    uv run python main.py
"""

from quantlab.dataset.stock import StockDataset
from quantlab.config import DatasetConfig
import polars as pl


ds = StockDataset(
    DatasetConfig(
        zarr_file_path='/Users/daizhaorong/projects/quantlab/data/data/us_equity/1d/us_all.zarr',
        symbols=('NVDA', 'AMZN', 'AMD'),
        raw_data_dir_path='',
        market='us_equity',
        frequency='1d',
    )
)

ds.read()

print(ds.get_lazyframe().collect().filter(pl.col('symbol') == 'NVDA').to_pandas())
