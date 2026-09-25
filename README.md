# quantlab

English | [简体中文](README.zh-CN.md)

quantlab is a config-driven backend for quantitative equity research, from raw market data to
factors, return predictions, target portfolio weights and a backtest.

- **Documentation:** [docs/README.md](docs/README.md)
- **Source code:** https://github.com/ZhaorongDai/quantlab2
- **Contributing:** see [Call for contributions](#call-for-contributions)
- **Bug reports:** https://github.com/ZhaorongDai/quantlab2/issues

It provides:

- one data format end to end: an `xarray.Dataset` indexed by `(timestamp, symbol)`, stored as
  Zarr, so factors and models never pass through a DataFrame
- resumable downloaders for Tiingo, Alpaca and WRDS (CRSP daily stocks, TAQ quotes), with a
  volume guard that refuses an over-sized request before it starts
- survivorship-bias-free universes: point-in-time index membership and whole-market rosters
- two factor backends, KunQuant (batch and streaming) and Polars (batch)
- deep-learning and tree models behind one interface, with walk-forward cross-validation
- a vectorized backtester built on vectorbt, with in-sample and out-of-sample metrics and
  run directories that can be rebuilt and re-run

quantlab is a research backend. It has no web front end and no order-routing engine.

```text
 data source  ->  dataset  ->  factors / labels  ->  model  ->  backtest
 (Tiingo,         (raw files    (KunQuant or        (torch or   (target weights,
  Alpaca,          to a panel)   Polars)             xgboost)    vectorbt, report)
  WRDS)
```

## Installation

quantlab needs Python 3.13 or newer and uses [uv](https://docs.astral.sh/uv/) to manage its
environment.

```bash
git clone https://github.com/ZhaorongDai/quantlab2.git
cd quantlab2
uv sync
```

Deep-learning models use a CUDA GPU when one is available and fall back to the CPU
otherwise. On macOS, a single process that imports both PyTorch and XGBoost needs
`OMP_NUM_THREADS=1`, because the two libraries ship conflicting OpenMP runtimes.

## Quick start

The example below runs offline. It lists the data sources quantlab knows about, then builds a
tiny price panel and reads it back through a dataset object.

```python
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from quantlab.base.config import DatasetConfig
from quantlab.dataset.stock import StockDataset
from quantlab.registry import DataSourceRegistry, credential_status

for source in DataSourceRegistry.all():
    print(source.vendor, credential_status(source))

root = Path(tempfile.mkdtemp())
timestamps = pd.date_range("2024-01-01", periods=5, freq="B")
symbols = ["AAPL", "MSFT", "NVDA"]
close = 100 + np.random.default_rng(0).normal(size=(5, 3)).cumsum(axis=0)
xr.Dataset(
    {"adjClose": (("timestamp", "symbol"), close)},
    coords={"timestamp": timestamps, "symbol": symbols},
).to_zarr(root / "prices.zarr", mode="w")

dataset = StockDataset(
    DatasetConfig(
        zarr_file_path=str(root / "prices.zarr"),
        raw_data_dir_path=str(root / "raw"),
        catalog_path=str(root / "catalog"),
        market="us_equity",
        frequency="1d",
        start_date="2024-01-01",
        end_date="2024-01-31",
    )
)
print(dataset.read().get_xarray_dataset())
```

Output:

```text
alpaca {'APCA_API_KEY_ID': False, 'APCA_API_SECRET_KEY': False}
tiingo {'TIINGO_API_KEY': False}
wrds {'WRDS_USERNAME': False}
<xarray.Dataset> Size: 208B
Dimensions:    (timestamp: 5, symbol: 3)
Coordinates:
  * timestamp  (timestamp) datetime64[ns] 40B 2024-01-01 ... 2024-01-05
  * symbol     (symbol) <U4 48B 'AAPL' 'MSFT' 'NVDA'
Data variables:
    adjClose   (timestamp, symbol) float64 120B ...
```

The `False` values mean the credentials for that source are not set in your environment.
The [user guides](docs/README.md) continue from here with downloading data, building factors,
training a model and running a backtest.

## Data sources and credentials

quantlab reads every credential from an environment variable. Nothing is accepted on the
command line, and no credential is written to a config file or a log.

| Variable | Used for |
|----------|----------|
| `TIINGO_API_KEY` | Tiingo US-equity end-of-day data |
| `APCA_API_KEY_ID`, `APCA_API_SECRET_KEY` | Alpaca market data (bars, quotes, trades) |
| `WRDS_USERNAME` | WRDS (CRSP, TAQ). The password comes from your `~/.pgpass` file. |
| `WANDB_API_KEY` | Weights & Biases logging during model training (optional) |
| `QUANTLAB_DATA_DIR` | Root directory for downloaded and converted data (optional) |

The data root is resolved in this order: the `--data-dir` flag of a download script, then
`QUANTLAB_DATA_DIR`, then a `data/` directory at the repository root. Converted Zarr stores go
to `data/<market>/<frequency>/` under that root and raw downloads to
`downloads/<market>/<frequency>/`.

The scripts in `scripts/` download and convert data. Each one prints its options with
`--help`.

```bash
uv run python scripts/ingest_tiingo.py --help
```

## Testing

The test suite needs no network access and no credentials.

```bash
uv run pytest
```

`tests/test_crsp_rebuild_measurements.py` rebuilds a real CRSP store, so it fails with an
explanatory message unless `QUANTLAB_DATA_ROOT` points at one. Pass
`--ignore=tests/test_crsp_rebuild_measurements.py` to leave it out. The KunQuant factor tests
compile C++ and take a few minutes.

## Status

quantlab is under active development and its interfaces can still change. Event-driven
backtesting on NautilusTrader, a service layer and a web front end are not implemented.

## Call for contributions

Issues and pull requests are welcome. Run `uv run pytest` before you open a pull request, and
write docstrings in the numpydoc format used throughout the code base: a one-line summary,
`Parameters`, `Returns` and `Raises` where they help, and a short `Examples` section.
