# Extending quantlab

quantlab is built so that each stage of the pipeline can be replaced without
touching the others: a new vendor, market, storage medium, factor, model or
backtest rule is a new class that fills in a small, fixed set of hooks. This
page shows the minimal working version of each kind of extension, with the
hooks it must implement and the registration or configuration it needs. Read
it when the built-in classes do not cover what you need. The user guide
explains what each layer does; this page assumes you have read the relevant
page there.

Every example on this page was run offline against the current code, on
synthetic data in a temporary directory, and the output shown is real. The
examples are self-contained apart from a few lines of setup (writing a small
Zarr store, building a `DatasetConfig`), which are the same as in
[`examples/backtest.py`](../../examples/backtest.py).

## Conventions every extension follows

A handful of rules hold across all layers, and following them is what makes
an extension work with the rest of the code.

The abstract base class of each layer lives in `quantlab/base/` and names the
hooks a subclass fills in. The concrete class lives with the code that uses
it: datasets in `quantlab/dataset/`, vendor clients in `quantlab/acquisition/`,
factors in `quantlab/factor/`, labels in `quantlab/label/`, model heads in
`quantlab/dl_model/` or `quantlab/ml_model/`, backtesters in
`quantlab/backtest/`. Support code that is not itself a dataset or an
acquisition goes in that layer's private `_support/` package.

Every configurable object records its class as a dotted import path in
`config.name`, and saved configs are rebuilt from that path by
`quantlab.utils.module`. A class defined in a notebook or a script cell can be
used, but its saved configs cannot be rebuilt in another process. Put classes
you want to reuse in an importable module. A class is rebuilt with the config
class it declares in its `config_cls` attribute; the base classes already set
it, so you only need to override it when you introduce a new config class.

Data moves between layers as a *panel*, an `xarray.Dataset` indexed by
`timestamp` and `symbol`. Keep market-specific names (vendor column names,
market literals) out of `quantlab/base/`; `tests/test_extensibility_contract.py`
fails if one appears there. Credentials are read from environment variables,
never from configs or source files.

Package `__init__.py` files are empty on purpose (a few guarantees depend on
it, see [Internals](internals.md#the-volume-guard)). Import implementation
modules by their full dotted path, and do not add re-exports.

## A data source

A data source is a vendor quantlab can download from. Adding one takes an
`Acquisition` subclass, which knows how to make one request, and a
`SourceDescriptor` registered with `quantlab.registry`, which tells the rest of
quantlab what the vendor serves.

`Acquisition` (`quantlab.base.acquisition`) owns everything else: batching,
worker threads, pagination, resuming an interrupted run, per-symbol
watermarks, the failure manifest, credential scrubbing and the Parquet shard
layout. A subclass sets `VENDOR` and `RAW_COLUMNS` and implements one method,
`_fetch_page`, which returns the rows of one request and the token of the next
page, or `None` on the last page. The stand-in below generates a random walk
instead of calling an API:

```python
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from quantlab.base.acquisition import Acquisition
from quantlab.base.config import AcquisitionConfig
from quantlab.dataset.stock import StockDataset
from quantlab.registry import Capability, SourceDescriptor, register_source

FIELDS = ("open", "high", "low", "close", "volume")


class DemoAcquisition(Acquisition):
    """Serves a random walk per symbol instead of calling a real API."""

    VENDOR = "demo"
    RAW_COLUMNS = ("timestamp", "symbol", "vendor", *FIELDS)
    DEFAULT_BATCH_SIZE = 2  # symbols per request

    def _fetch_page(self, symbols, start_date, end_date, page_token=None):
        days = pd.bdate_range(start_date, end_date)
        frames = []
        for symbol in symbols:
            rng = np.random.default_rng(sum(map(ord, symbol)))
            close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(days))))
            frames.append(pl.DataFrame({
                "timestamp": days.to_numpy(), "symbol": symbol,
                "vendor": self.VENDOR, "open": close, "high": close * 1.01,
                "low": close * 0.99, "close": close,
                "volume": rng.uniform(1e5, 1e6, len(days)),
            }))
        return pl.concat(frames), None  # None: this was the last page


def demo_acquisition_config(symbols, root, start_date=None, end_date=None, **_):
    return AcquisitionConfig(
        market="us_equity", frequency="1d", vendor="demo",
        raw_data_dir_path=str(Path(root) / "raw" / "demo"),
        watermark_path=str(Path(root) / "_watermarks" / "demo"),
        symbols=tuple(symbols), start_date=start_date, end_date=end_date,
    )


DEMO_SOURCE = register_source(
    SourceDescriptor(
        vendor="demo",
        display_name="Demo random walk",
        acquisition_cls=DemoAcquisition,
        config_factory=demo_acquisition_config,
        capabilities=(
            Capability(market="us_equity", frequency="1d", dataset_cls=StockDataset),
        ),
        required_env=(),  # e.g. ("DEMO_API_KEY",) for a real vendor
    )
)
```

Once registered, the source is driven through the registry like any built-in
vendor. `run()` downloads, `convert()` turns the raw tier into a Zarr panel
with the capability's `dataset_cls`:

```python
from quantlab.base.config import DatasetConfig
from quantlab.registry import DataSourceRegistry, convert, run

source = DataSourceRegistry.get("demo")
config = source.config_factory_for("us_equity", "1d")(
    ("AAA", "BBB", "CCC"), root, start_date="2024-01-02", end_date="2024-03-29"
)
result = run(source, config)
print("succeeded:", result.succeeded)
again = run(source, config)  # everything is covered, nothing is fetched
print("second run coverage:", again.coverage)

dataset_config = DatasetConfig(
    raw_data_dir_path=config.raw_data_dir_path,
    zarr_file_path=str(root / "demo.zarr"),
    catalog_path=str(root / "catalog"),
    market="us_equity", frequency="1d", vendor="demo",
)
conversion = convert(source, dataset_config, granularity="month")
print("windows written:", conversion.windows_written)
panel = StockDataset(dataset_config).read().get_xarray_dataset()
print(dict(panel.sizes), list(panel.data_vars))
```

```text
succeeded: ('AAA', 'BBB', 'CCC')
second run coverage: {'requested': 3, 'pending': 0, 'skipped': 3, 'covered': 3, 'widened': 0, 'legacy': 0, 'no_data': 0}
windows written: 3
{'timestamp': 64, 'symbol': 3} ['anomaly_flag', 'close', 'high', 'low', 'open', 'volume']
```

The raw tier lands in the layout every dataset class expects: hive
directories keyed by month for daily data, and deterministic shard names made
of a batch key and a page number, with the bookkeeping files beside the raw
root rather than inside it:

```text
_watermarks/demo/AAA.json
_watermarks/demo/_failures.json
_watermarks/demo/_pages/5b7fc3dffa5ab4a0.pages.json
raw/demo/month=2024-01/part-5b7fc3dffa5ab4a0-00000.pqt
raw/demo/month=2024-02/part-5b7fc3dffa5ab4a0-00000.pqt
...
```

For a real vendor, a few more steps apply:

- Put the class and its `register_source(...)` call in one module under
  `quantlab/acquisition/`, and import that module at the bottom of
  `quantlab/registry.py` next to the other vendors, so that
  `import quantlab.registry` lists it.
- Add the vendor name to the `Vendor` literal in `quantlab/enums/data.py`.
- Read the API key from the environment in `__init__` and raise if it is
  missing. List the variable names in `CREDENTIAL_ENV_VARS`, so the base class
  redacts them from every logged or persisted message, and repeat them in the
  descriptor's `required_env`.
- Override `_classify_error` so that an exhausted quota is reported as
  `"quota"` (the run stops dispatching) and a per-minute rate limit as
  `"rate_limited"` (the worker backs off), rather than letting every symbol
  fail one by one. `quantlab/acquisition/tiingo.py` and
  `quantlab/acquisition/alpaca.py` are the two reference implementations.
- Declare `RAW_SCHEMA` with explicit dtypes if the vendor's responses can
  infer different column types for different symbols, and return an empty
  frame with that schema when a request has no rows.
- Update the registry tests that pin the list of vendors
  (`tests/test_source_registry.py`).

## A dataset

A dataset converts a raw tier into a dense panel and stores it. For market
bars, subclass `MarketDataset` (`quantlab.base.data`) and implement four
hooks: `_raw_data_to_xr` returns the panel for the configured date range,
`_raw_data_to_xr_window` returns one time window of it on a given symbol axis
(used by chunked conversion), and `_to_kunquant` and `_to_nautilus` export to
the KunQuant factor engine and the Nautilus Trader catalog. An export you do
not support may raise. Everything else, including dates, storage, cleaning and
resumable chunked conversion, is inherited.

This dataset reads daily bars from a single long-format CSV file:

```python
import dataclasses

import numpy as np
import pandas as pd
import xarray as xr

from quantlab.base.config import DatasetConfig
from quantlab.base.data import MarketDataset


class CsvBarDataset(MarketDataset):
    """Daily bars from one long-format CSV file: date, ticker, open, ..., volume."""

    def _read_csv(self) -> pd.DataFrame:
        frame = pd.read_csv(self.config.raw_data_dir_path, parse_dates=["date"])
        frame = frame.rename(columns={"date": "timestamp", "ticker": "symbol"})
        # One row per (timestamp, symbol), as _raw_data_to_xr requires.
        return frame.drop_duplicates(["timestamp", "symbol"], keep="last")

    def _raw_data_to_xr(self) -> xr.Dataset:
        return self._raw_data_to_xr_window(self.config.start_date, self.config.end_date)

    def _raw_data_to_xr_window(self, start_date, end_date, symbols=None):
        frame = self._read_csv()
        frame = frame[frame["timestamp"].between(start_date, end_date)]
        # to_xarray() densifies: a missing (timestamp, symbol) pair becomes NaN.
        panel = frame.set_index(["timestamp", "symbol"]).to_xarray()
        if symbols is not None:
            panel = panel.reindex(symbol=list(symbols))
        return panel

    def _to_kunquant(self, data, data_columns):
        data = data.sortby(["timestamp", "symbol"])
        arrays = {
            column: np.ascontiguousarray(data[column].values.astype(np.float32))
            for column in data_columns
        }
        return arrays, data["symbol"].values, data["timestamp"].values

    def _to_nautilus(self, data, venue, n_jobs):
        raise NotImplementedError("CsvBarDataset has no Nautilus export")
```

With a CSV of three symbols over 60 business days written to `root`, the whole
lifecycle works unchanged, including resumable month-by-month conversion:

```python
config = DatasetConfig(
    raw_data_dir_path=str(root / "bars.csv"),
    zarr_file_path=str(root / "bars.zarr"),
    catalog_path=str(root / "catalog"),
    market="us_equity", frequency="1d",
    start_date="2024-01-02", end_date="2024-03-29",
)
CsvBarDataset(config).from_raw_data().save()
panel = CsvBarDataset(config).read().get_xarray_dataset()
print(dict(panel.sizes), list(panel.data_vars))

chunked = dataclasses.replace(config, zarr_file_path=str(root / "chunked.zarr"))
ds = CsvBarDataset(chunked).from_raw_data_chunked(granularity="month")
print("windows written:", ds.last_chunk_result.windows_written)
ds = CsvBarDataset(chunked).from_raw_data_chunked(granularity="month")
print("windows written on re-run:", ds.last_chunk_result.windows_written,
      "skipped:", ds.last_chunk_result.windows_skipped)
```

```text
{'timestamp': 60, 'symbol': 3} ['anomaly_flag', 'close', 'high', 'low', 'open', 'volume']
windows written: 3
windows written on re-run: 0 skipped: 3
```

The default cleaning step (`_clean`) requires lowercase `open`, `high`,
`low`, `close` and `volume` columns and adds `anomaly_flag`. Override `_clean`
for panels with other columns, and derive from `BaseDataset` instead of
`MarketDataset` for data that is not price bars (an index-membership panel,
for example). The default `_raw_data_to_xr_window` of `BaseDataset` converts
the whole range and slices it; `MarketDataset` makes the method abstract so
that a large market dataset filters each window at read time instead, as
`StockDataset._raw_data_to_xr_window` does with a Parquet scan. To make the
registry's `convert()` reach a new dataset, name it as the `dataset_cls` of
the vendor's capability.

## A storage backend

A backend is where a dataset or factor keeps its panel. `DataBackend`
(`quantlab.base.backend`) has eight abstract methods: `read`, `write`,
`to_internal`, `filter_by_date`, `filter_by_symbol`, `get_xarray_dataset`,
`get_lazyframe` and `head`. A class that leaves one out cannot be
instantiated. Two details of the contract are easy to get wrong: the
`filter_by_*` methods narrow the held data in place, while `head(path, n)`
must open the store at `path`, read at most `n` rows without loading the
whole store, and leave the held data alone.

When the medium still holds an `xarray.Dataset`, the simplest route is to
subclass `XrBackend` (`quantlab.backend`) and replace only the input and
output. This backend keeps a panel in a single NetCDF file:

```python
from pathlib import Path
from typing import Self

import polars as pl
import xarray as xr

from quantlab.backend import XrBackend


class NetcdfBackend(XrBackend):
    """Keep the panel in one NetCDF file instead of a Zarr directory."""

    def read(self, path: str, overwrite: bool = False, **kwargs) -> Self:
        if not overwrite and hasattr(self, "data"):
            return self
        if not Path(path).exists():
            raise FileNotFoundError(f"File {path} does not exist.")
        with xr.open_dataset(path, engine="scipy", **kwargs) as opened:
            self.data = opened.load()
        return self

    def write(self, path: str, **kwargs) -> Self:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.data.to_netcdf(path, engine="scipy", **kwargs)
        return self

    def head(self, path: str, n: int) -> pl.LazyFrame:
        if not Path(path).exists():
            raise FileNotFoundError(f"File {path} does not exist.")
        with xr.open_dataset(path, engine="scipy") as opened:
            bounded = opened.isel({dim: slice(0, n) for dim in opened.dims})
            frame = bounded.to_dataframe().reset_index()
        return pl.from_pandas(frame).lazy().head(n)

    # The Zarr-only growth paths used by chunked conversion and update().
    def widen_and_append(self, *args, **kwargs):
        raise NotImplementedError("NetcdfBackend cannot append; use save()")

    widen_symbol_axis = append = widen_and_append
```

Datasets create an `XrBackend` in their constructor, so plug the new backend
in by replacing it after construction:

```python
from quantlab.dataset.stock import StockDataset


class NetcdfStockDataset(StockDataset):
    def __init__(self, config):
        super().__init__(config)
        self.data_backend = NetcdfBackend()
```

With a 3-by-2 panel written through the backend, filtering, bounded reads and
the dataset round trip give:

```text
[[2.0, 3.0], [4.0, 5.0]]          # filter_by_date(...).get_xarray_dataset(...)["close"]
(2, 3)                            # head(path, 2).collect().shape
{'timestamp': 3, 'symbol': 2}     # NetcdfStockDataset(cfg).read().get_xarray_dataset().sizes
```

Keep `read`'s `overwrite` keyword: the backtester calls
`read(overwrite=True)` whenever it changes a dataset's dates, and without it
the backend would return an earlier, narrower read. For a medium that does not
hold an xarray object, implement all eight methods; `PlBackend` in
`quantlab/backend.py` is the reference for a table-shaped medium, and its
`get_xarray_dataset(indexes)` shows how to turn the named columns into the
dataset's dimensions. Chunked conversion, `Dataset.update()` and
`Factor.update()` also call `widen_and_append` (and, for a new listing,
`widen_symbol_axis`), which exist only on `XrBackend`; a backend without them
supports `from_raw_data().save()` and `read()` only.
The executable version of the contract is in `tests/test_backend_head.py`,
`tests/test_backend_indexes.py` and `tests/test_backend_overwrite.py`.

## A Polars factor

A factor turns a dataset's panel into feature columns. The Polars backend,
`FactorPolars` (`quantlab.base.factor`), is batch-only and needs one method,
`_get_factor_lazyframe`, which receives the dataset as a long-format
`polars.LazyFrame` (one row per timestamp and symbol) and returns a lazy frame
holding exactly `timestamp`, `symbol` and the factor columns. Factor names are
read from that frame's schema, so there is nothing to declare. Override
`_get_features` to use the class as a feature, or `_get_labels` to use it as a
label. In the snippets of this section and the next, `prices` is a
`DatasetConfig` over a Zarr store holding `adjClose` and `adjVolume` for eight
symbols over 60 business days, and each factor gets its own `StockDataset`
over a copy of it (`dataclasses.replace(prices)`), because a factor moves its
dataset's dates.

```python
import dataclasses

import polars as pl
import xarray as xr

from quantlab.base.config import PolarsFactorConfig
from quantlab.base.factor import FactorPolars
from quantlab.dataset.stock import StockDataset


class RelativeVolume(FactorPolars):
    """Volume over its trailing n-bar mean, minus 1."""

    def _get_factor_lazyframe(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        n = self.config.kwargs["n"]
        volume = pl.col("adjVolume")
        return (
            lf.sort(["symbol", "timestamp"])
            .with_columns(
                (volume / volume.rolling_mean(n).over("symbol") - 1.0)
                .alias(f"rel_volume_{n}")
            )
            .select(["timestamp", "symbol", f"rel_volume_{n}"])
        )

    def _get_features(self, data: xr.Dataset) -> xr.Dataset:
        return data


factor = RelativeVolume(
    PolarsFactorConfig(
        window=20,
        dataset=StockDataset(dataclasses.replace(prices)),
        kwargs={"n": 10},
        start_date="2024-02-01",
        file_path=str(root / "factors" / "rel_volume.zarr"),
    )
)
print(factor.get_factor_names())
features = factor.cal().save(mode="w").get_features()
print(dict(features.sizes), float(features["rel_volume_10"].isnull().mean()))
```

```text
('rel_volume_10',)
{'timestamp': 38, 'symbol': 8} 0.0
```

`window` is the factor's warm-up. On its own, a factor moves its dataset's
start date back by `window` calendar days before computing; inside a
backtest, the backtester starts it `window` price bars early. Calendar days
are fewer than bars, so pick a `window` comfortably larger than the longest
look-back in bars: with `window=10` here, the first bar of February would
still be NaN. Column names are the store's own (`adjVolume` in a Tiingo-shaped
store), and parameters belong in `config.kwargs`, so one class serves many
configs. `quantlab/factor/momentum.py` is the reference implementation.

## A KunQuant factor

KunQuant compiles a factor formula into native code and can also run it one
bar at a time on live data. A `FactorKunQuant` subclass implements
`_get_factor_names` and `_get_factor_func`, which builds the formula as a
graph of KunQuant operators. `Input` names must match `config.data_columns`
and each `Output` name must be one of the factor names:

```python
import KunQuant.ops as op
from KunQuant.Op import Builder, Input, Output
from KunQuant.Stage import Function

from quantlab.base.config import FactorConfig
from quantlab.base.factor import FactorKunQuant


class MaDeviation(FactorKunQuant):
    """Close over its n-bar moving average, minus 1."""

    def _get_factor_names(self):
        return (f"ma_dev_{self.config.kwargs['n']}",)

    def _get_factor_func(self):
        n = self.config.kwargs["n"]
        builder = Builder()
        with builder:
            close = Input("adjClose")
            deviation = op.Div(close, op.WindowedAvg(close, n))
            Output(op.SubConst(deviation, 1.0), self._get_factor_names()[0])
        return Function(builder.ops)

    def _get_features(self, data):
        return data


factor = MaDeviation(
    FactorConfig(
        window=10,
        dataset=StockDataset(dataclasses.replace(prices)),
        mode="batch",
        data_columns=("adjClose",),
        kwargs={"n": 5},
        start_date="2024-02-01",
        file_path=str(root / "factors" / "ma_dev.zarr"),
        njobs=2,
    )
)
panel = factor.cal().get_features()
print(dict(panel.sizes), round(float(panel["ma_dev_5"].isel(timestamp=0, symbol=0)), 6))
```

```text
{'timestamp': 38, 'symbol': 8} -0.001579
```

The value matches the same formula computed with pandas on the first symbol.
Compilation needs a working C++ compiler and takes a few seconds per call. In
batch mode the number of symbols must be a multiple of the SIMD block width
KunQuant uses (8 on most machines), which is why this panel has eight symbols.
KunQuant graphs can only look backwards in time; a forward-looking label is
computed as a trailing value and shifted in `_get_labels`, as
`quantlab/label/fret.py` does. Existing operator compositions to reuse are in
`quantlab/factor/alpha101.py`, `quantlab/factor/alpha158.py` and
`quantlab/my_ops/preprocess.py`.

## A model head

A model head is the part of a return model that is specific to one learning
algorithm. `BaseModel` (`quantlab.base.model`) owns everything shared:
collecting features and labels, the train/validation/test split, walk-forward
cross-validation, checkpoints with a `config.json` beside them, and
`predict_panel`. Two variants add the framework-specific loop.

`MLModel` is for numpy-based libraries such as tree models. Its four hooks
are `_init_model`, `_preprocess`, `_fit_model` (fit once, with the library's
own early stopping if it has one, and leave the fitted model in `self.model`)
and `_forward`. Arrays are `[time, symbol, feature]` in and
`[time, symbol, label]` out. In the snippets below, `factors` is a list of
two past-return factors (1 and 5 bars) and `labels` a 5-bar forward-return
label, built like the ones in `examples/backtest.py` over 120 business days of
eight symbols. A ridge regression:

```python
import numpy as np

from quantlab.base.config import MLConfig
from quantlab.base.model import MLModel


class RidgeHead(MLModel):
    """Ridge regression on every symbol-bar with finite features and labels."""

    def _init_model(self, num_features, num_labels, hyperparameters):
        return None  # the coefficients are created in _fit_model

    def _preprocess(self, data):
        return np.array(data, dtype=np.float64, copy=True)

    def _fit_model(self, train_x, train_y, val_x, val_y):
        x = train_x.reshape(-1, train_x.shape[-1])
        y = train_y.reshape(-1, train_y.shape[-1])
        keep = np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
        x, y = x[keep], y[keep]
        alpha = self._resolved_hyperparameters()["alpha"]
        gram = x.T @ x + alpha * np.eye(x.shape[1])
        self.model = {"coef": np.linalg.solve(gram, x.T @ y)}

    def _forward(self, x):
        return np.nan_to_num(x) @ self.model["coef"]

    def _resolved_hyperparameters(self):
        return {"alpha": float(self.config.hyperparameters.get("alpha", 1.0))}


ridge = RidgeHead(MLConfig(
    factors=factors, labels=labels, model_save_dir=str(root / "models"),
    factor_data_strategy="cal", label_data_strategy="cal",
    hyperparameters={"alpha": 0.1}, val_size=0.2,
    train_start="2024-01-02", train_end="2024-04-30",
    test_start="2024-05-01", test_end="2024-06-14",
))
checkpoint = ridge.collect().train()
print(checkpoint.name)
reloaded = RidgeHead(ridge.config).load(checkpoint)
print(np.allclose(reloaded.model["coef"], ridge.model["coef"]))
```

```text
RidgeHead_total.joblib
True
```

`_resolved_hyperparameters` is optional. When it returns a mapping, the
hyperparameters actually in effect are recorded in the checkpoint's
`config.json`, so a run stays reproducible if a default changes. It is called
before `_init_model` (to log the run config), so either compute it from
`self.config`, as here, or return `None` until `_init_model` has run, as
`XGBoostRegressor` does. The checkpoint
is a joblib pickle of `self.model`; only load files you trust.

`DLModel` is for PyTorch networks trained in an epoch loop with early
stopping and rollback to the best epoch. Its hooks are `_init_model` (return
an `nn.Module` for the panel shape), `_init_optim`, `_preprocess` (applied to
tensors at training and prediction time alike), and `_train_one_batch`,
`_val_one_batch` and `_test_one_batch`. Batches are
`[batch, symbol, feature]` tensors already on the model's device:

```python
import torch
from torch import nn

from quantlab.base.config import DLConfig
from quantlab.base.model import DLModel


class LinearHead(DLModel):
    """One linear layer applied to every symbol's features."""

    def _init_model(self, num_symbols, num_features, num_labels, hyperparameters):
        return nn.Linear(num_features, num_labels)

    def _init_optim(self, model):
        return torch.optim.Adam(model.parameters(), lr=self.config.lr)

    def _preprocess(self, data):
        return torch.nan_to_num(data, nan=0.0)

    def _train_one_batch(self, epoch, x, y):
        self.optim.zero_grad()
        loss = self._masked_mse(x, y)
        loss.backward()
        self.optim.step()
        return loss.detach()

    def _val_one_batch(self, epoch, x, y):
        return self._masked_mse(x, y)

    def _test_one_batch(self, epoch, x, y):
        return self._masked_mse(x, y)

    def _masked_mse(self, x, y):
        mask = torch.isfinite(y)  # labels are NaN at the end of the panel
        return nn.functional.mse_loss(self.model(x)[mask], y[mask])


linear = LinearHead(DLConfig(
    factors=factors, labels=labels, model_save_dir=str(root / "models"),
    factor_data_strategy="cal", label_data_strategy="cal",
    epochs=3, batch_size=16, num_workers=0, lr=1e-2,
    train_start="2024-01-02", train_end="2024-04-30",
    test_start="2024-05-01", test_end="2024-06-14",
))
print(linear.collect().train().name)
```

```text
LinearHead_total.pth
```

Every training run opens a Weights & Biases run; set `WANDB_MODE=disabled` in
the environment to keep it offline. On macOS, set `OMP_NUM_THREADS=1` before
importing anything when one process uses both torch and xgboost. The
reference heads are `quantlab/ml_model/xgb.py` and the modules in
`quantlab/dl_model/`.

## A backtest market or selection rule

A backtester is `VectorBtBacktester` (`quantlab.backtest.engine_vectorbt`)
plus three members: `config_cls`, the config class it accepts; `MARKET`, a
`MarketSpec` naming the fill and valuation price columns and the year length
used for annualizing; and `_generate_signals`, which turns the model's
prediction panel and the prices into target weights. The weights must follow
the contract described in
[Backtesting](../user-guide/backtesting.md#target-weights-the-signal-format):
all-NaN rows on hold bars, all-finite rows with gross exposure at most 1 on
rebalance bars. The backtester checks this before simulating.

This backtester trades a round-the-clock market on unadjusted `open` and
`close` columns and holds every symbol the model scores above zero, equally
weighted:

```python
import numpy as np
import xarray as xr

from quantlab.backtest.engine_vectorbt import VectorBtBacktester
from quantlab.backtest.selection import rebalance_mask
from quantlab.base.backtest import MarketSpec
from quantlab.base.config import BacktestConfig

#: Unadjusted open/close columns, a 24/7 market: 365 trading days of 1440 minutes.
ROUND_THE_CLOCK = MarketSpec(
    fill_price_column="open",
    valuation_price_column="close",
    trading_days_per_year=365,
    session_minutes_per_day=1440,
)


class PositiveScoreEqualWeight(VectorBtBacktester):
    """Hold every symbol with a positive score, equally weighted."""

    config_cls = BacktestConfig
    MARKET = ROUND_THE_CLOCK

    def _generate_signals(self, predictions, prices):
        label = self.config.model.get_label_names()[0]
        scores = predictions[label].transpose("timestamp", "symbol").values
        next_fill = (
            prices[self.MARKET.fill_price_column]
            .shift(timestamp=-1)
            .transpose("timestamp", "symbol")
            .values
        )
        rebalance = rebalance_mask(scores.shape[0], self.config.rebalance_periods)

        weights = np.full(scores.shape, np.nan)  # NaN row: hold
        for t in np.flatnonzero(rebalance):
            chosen = np.isfinite(scores[t]) & np.isfinite(next_fill[t]) & (scores[t] > 0)
            row = np.zeros(scores.shape[1])  # rebalance row: all finite
            if chosen.any():
                row[chosen] = 1.0 / chosen.sum()  # gross exposure <= 1
            weights[t] = row
        return xr.Dataset(
            {"weight": (("timestamp", "symbol"), weights)},
            coords={"timestamp": prices.timestamp, "symbol": prices.symbol},
        )
```

Run with the price data and model from `examples/backtest.py`
(`BacktestConfig(price_dataset=..., model=..., model_mode="train",
rebalance_periods=5, ...)`), the first five rebalance rows hold this many
symbols, and the market's year length is used for annualizing:

```text
[10, 12, 8, 10, 9]                   # symbols held per rebalance row
365 days 00:00:00 8760.0             # year_freq("1D"), bars per year at "1h"
```

Requiring a finite next-bar fill price keeps a symbol that is about to lose
its prices from being bought. Everything after `_generate_signals` is
inherited: execution at the next bar's open, forced liquidation of delisted
holdings, metrics and the run directory.

Two smaller variations need even less code. To reuse top-N selection on a
different market, subclass `USEquityCrossectionSelectStockVectorBt` and set
only `MARKET`. To reuse the selection rule with another engine, call
`CrossSectionTopNSelector(direction, top_n).select(scores, next_fill_price,
rebalance)` from `quantlab.backtest.selection`; it depends on no simulation
engine. A new selection parameter belongs on a new config dataclass derived
from `BacktestConfig`, named in `config_cls`, so that `config.json` records it
and `load_backtester_from_config` can rebuild the run. Construction-time checks
go in `_validate_config`, which runs at the end of the config setter.

A different simulation engine is a sibling of `VectorBtBacktester`: subclass
`BaseBacktester` and implement `_simulate`, `_simulate_benchmark`,
`_engine_stats` and `_period_returns_stats`, returning the engine-neutral
`SimulationResult` described in the docstring of `quantlab.base.backtest`.
