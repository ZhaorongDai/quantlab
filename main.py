from quantlab.dataset.stock import StockDataset
from quantlab.config import DatasetConfig
import polars as pl


ds = StockDataset(
    DatasetConfig(
        zarr_file_path='/Users/daizhaorong/projects/quantlab/data/data/us_equity/1d/us_all.zarr',
        symbols=('NVDA', 'AMZN', 'AMD'),
        catalog_path='',
        raw_data_dir_path='',
        frequency='1d',
    )
)

ds.read()

print(ds.get_lazyframe().collect().filter(pl.col('symbol') == 'NVDA').to_pandas())
