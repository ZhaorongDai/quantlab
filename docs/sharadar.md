# Sharadar daily stocks

Sharadar is quantlab's primary US-equity vendor ([ADR 0023](adr/0023-sharadar-is-the-primary-vendor-stores-hold-raw-prices-on-permaticker.md)). quantlab pulls Sharadar's tables from its own API (`api.sharadar.com/v1.0`), keeps them as raw parquet, and converts the stock price table (SEP) into a `(timestamp, symbol)` Zarr panel whose symbol axis is the **permaticker**.

## Prerequisites

Downloading needs a paid sharadar.com key, read only from the environment:

```bash
export SHARADAR_API_KEY=<your-sharadar-key>
```

A missing key, or an HTTP 401/403 from Sharadar, raises `SharadarEntitlementError` naming the table. Converting and reading a store need no key and no network.

The data is licensed for personal use: keep raw files and stores on your own machines, never in the repository or a tracker artifact.

## Pulling the raw tier

`SharadarClient.bulk_table(code, download_dir)` pulls one whole table as Sharadar's bulk zip and writes it as `<download_dir>/sharadar/<code>/<code>.parquet`, with the vendor's column names and order checked against the declared schema. The tables available so far are `sep` (stock prices), `tickers` (the ticker-to-permaticker mapping) and `indicators` (the data dictionary); TICKERS and INDICATORS stay parquet sidecar tables and never become Zarr stores.

```python
from quantlab.acquisition.sharadar.client import SharadarClient

client = SharadarClient()
for code in ("sep", "tickers", "indicators"):
    client.bulk_table(code, "/data/quantlab/downloads")
```

The zip is downloaded in parallel byte ranges (`SharadarClient(download_workers=8, part_bytes=64 << 20)` by default), or as one stream when the storage ignores `Range`. Rate-limit (429) and server-error responses, connections that fail to open and byte ranges whose stream breaks are retried after a back-off. A full-history SEP zip is about 1 GB.

## Building and reading the panel

```python
from quantlab.dataset.config import SharadarDatasetConfig
from quantlab.dataset.sharadar.stock import SharadarStockDataset

config = SharadarDatasetConfig(
    zarr_file_path="/data/quantlab/zarrs/sharadar_sep_1d.zarr",
    raw_data_dir_path="/data/quantlab/downloads/sharadar",
)
SharadarStockDataset(config).from_raw_data().save()
panel = SharadarStockDataset(config).panel("2024-01-02", "2024-01-05")
```

Observed on a one-row synthetic raw tier:

```python
>>> sorted(panel.data_vars), panel.symbol.dtype
(['anomaly_flag', 'close', 'high', 'low', 'open', 'volume'], dtype('int64'))
```

What the panel means:

- **Symbol axis.** Each raw row is mapped to its permaticker through the TICKERS rows of its own table (labelled `SEP` in the bulk file, `stocks` over the REST API). A renamed company keeps one column; a delisted company whose ticker was reused keeps its own column. The conversion refuses a ticker TICKERS does not know, a ticker mapped to two permatickers, and two rows of one permaticker on one date.
- **Raw prices.** `close` is SEP's `closeunadj`. `open`, `high` and `low` are SEP's split-adjusted values times `closeunadj / close`, and `volume` is SEP's split-adjusted volume divided by that ratio. Sharadar's adjusted columns are not stored, because the vendor rewrites them over the whole history on every ex-date.
- **Selection.** `permatickers=(...)` restricts the conversion; the ticker-based `symbols` field is refused.
