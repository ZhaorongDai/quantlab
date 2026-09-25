"""Ad hoc example: open a US-equity Zarr store and print its symbols.

The script opens a daily US-equity Zarr store (a chunked on-disk array
format that ``xarray`` reads and writes) through ``StockDataset`` and prints
the symbols it holds. It is not part of the library and has no
command-line options.

The store path is hardcoded and machine-specific. Edit ``zarr_file_path``
before running. No credentials are needed.

Usage::

    uv run python cal.py
"""

from quantlab.base.config import DatasetConfig
from quantlab.dataset.stock import StockDataset

ds_cfg = DatasetConfig(
    zarr_file_path='/Users/daizhaorong/projects/quantlab/data/data/us_equity/1d/us_all.zarr',
    # start_date='2016-01-01',
    # end_date='2024-01-01',
    raw_data_dir_path='',
    market='us_equity',
    frequency='1d',
)
ds = StockDataset(ds_cfg)
print(ds.read().symbols)
