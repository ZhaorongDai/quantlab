# Concepts

This page describes how quantlab is put together: the stages of the research pipeline and
what each one consumes and produces, the panel format every stage exchanges, the config
dataclasses that drive every object, how a saved `config.json` is turned back into live
objects, what a backtest run directory contains, and where each piece lives in the package.
Read it after the [quickstart](../getting-started/quickstart.md), or whenever you want to know
where a new piece of code belongs. The task-oriented pages of the user guide assume the
vocabulary introduced here.

## The pipeline

quantlab turns market data into a simulated trading record through a chain of stages. Each
stage is a separate object with a narrow input and output, so any one of them can be replaced
without touching the others:

```text
vendor API --> acquisition --> raw tier --> dataset --> factors, labels --> model
                (download)    (parquet)    (Zarr panel)   (panels)        (predictions)
                                                                              |
                            run directory <-- backtest <-- target weights <---+
```

| Stage | Input | Output | Base class |
|-------|-------|--------|------------|
| Acquisition | a vendor API and an `AcquisitionConfig` | raw parquet files on disk | `quantlab.base.acquisition.Acquisition` |
| Dataset | the raw files | a price panel in a Zarr store | `quantlab.base.data.BaseDataset` |
| Factor and label | a price panel | a factor panel | `quantlab.base.factor.Factor` |
| Model | factor and label panels | a prediction panel | `quantlab.base.model.BaseModel` |
| Selection | a prediction panel | a target-weight panel | a strategy-specific class |
| Backtest | target weights and prices | a simulation, metrics and a run directory | `quantlab.base.backtest.BaseBacktester` |

Acquisition downloads raw data from a vendor (Tiingo, Alpaca or WRDS) and writes it unchanged
to a directory tree of parquet files, the raw tier. Keeping the vendor's rows as they arrived
means a change of mind about cleaning or filtering is a re-conversion, not a re-download.
Downloads are resumable: completed pieces are recorded on disk and skipped next time. See
[data sources](data-sources.md).

A dataset converts the raw tier into a dense panel and stores it as Zarr. The dataset object
is also how every later stage reads prices: it knows its store and answers a request for a
date range with `panel(start, end)`. See [datasets](datasets.md). A related kind of dataset holds index membership as a
boolean panel, used to restrict research to the stocks that were actually in an index on each
date; see [universes](universes.md).

A factor computes a feature from a dataset, for example a momentum or volatility measure. A
label is a factor shifted forward in time by `Forward`: it is the quantity to be predicted,
typically the return over the next few bars, and its value at bar t reads a fixed number of
bars after t (its lookahead). Both produce a panel. Factors are written either
as KunQuant operator graphs, compiled to native code and able to run bar by bar on live data,
or as Polars expressions, which are batch only. See [factors](factors.md).

A model learns to predict the labels from the factors. At every split boundary it drops the
last lookahead bars of the earlier segment, so no label it fits on reads a bar of the later
one. Its prediction panel has one variable per label on the same `(timestamp, symbol)` grid
as its inputs. Model heads come in two
families: torch networks trained one cross-section of symbols per step (`TorchModel`) and tree or tabular models fitted
in one call with the library's own early stopping (`LibraryModel`). The torch family ships
`GATsRegressor` (Qlib's GATs) and `MASTERRegressor` (MASTER, gated by market-wide features
such as those of the `MarketFeatures` factor); the library family ships `XGBoostRegressor`,
`XGBTDRegressor` and `RealMLPRegressor`. See [models](models.md).

Selection turns predictions into target weights: for every symbol, the fraction of the
portfolio it should hold. The cross-sectional selector ranks the symbols on each rebalance bar
and holds the best `top_n` (and optionally shorts the worst `top_n`). A dedicated portfolio
optimisation stage is not implemented yet; today the selector plays that role inside the
backtester.

The backtester runs the model over a date window, asks the selector for weights, simulates
the resulting trades with vectorbt, computes performance metrics and writes everything to a
run directory. See [backtesting](backtesting.md).

## Panels

A panel is an `xarray.Dataset` whose data variables all have the dimensions
`("timestamp", "symbol")`, in that order: one row per bar, one column per instrument, one
variable per field. A price panel holds variables such as `adjOpen` and `adjClose`; a factor
panel holds one variable per factor; a weight panel holds a single variable `weight`. Every
stage hands the next one a panel, never a pandas DataFrame, and every store on disk is a Zarr
directory holding exactly such a panel.

Here are two variables of the quickstart's price panel:

```text
<xarray.Dataset> Size: 106kB
Dimensions:    (timestamp: 400, symbol: 16)
Coordinates:
  * timestamp  (timestamp) datetime64[ns] 3kB 2022-01-03 ... 2023-07-14
  * symbol     (symbol) <U3 192B 'S00' 'S01' 'S02' 'S03' ... 'S13' 'S14' 'S15'
Data variables:
    adjOpen    (timestamp, symbol) float64 51kB 49.74 50.21 ... 47.74 71.36
    adjClose   (timestamp, symbol) float64 51kB 50.13 49.87 ... 49.86 71.54
```

Panels are dense. A symbol that was not trading on a date, because it had not listed yet or
had already been delisted, has NaN in that cell rather than a missing row. This is what makes
the data survivorship-bias free: survivorship bias is the error of studying only the companies
that still exist today, which overstates returns because the failures have silently dropped
out. A panel that keeps delisted symbols as columns, with values up to their last trading day,
lets models and backtests see the failures too. The backtester settles a holding whose
prices stop into cash at its last valuation, and records the settlement.

The `symbol` coordinate is usually a ticker string. Panels built from CRSP use the PERMNO
instead, CRSP's permanent integer identifier for a security, because tickers are reused and
change over time while a PERMNO never does.

Storage objects in `quantlab.backend` read and write panels: `XrBackend` for Zarr stores and
`PlBackend` for parquet via Polars lazy frames. When a stage asks for data it calls
`get_xarray_dataset()` on the object that holds it, which returns the panel on
`(timestamp, symbol)`.

## Config dataclasses

Every object in quantlab is built from one config dataclass, defined in
`quantlab.base.config`, and keeps it as its `config` attribute:

| Config | Builds |
|--------|--------|
| `AcquisitionConfig` | a vendor download |
| `DatasetConfig` (and `CrspDatasetConfig`, `NbboDatasetConfig`) | a market dataset |
| `ConstituentDatasetConfig` | an index-membership dataset |
| `UniverseConfig` | the point-in-time universe catalog |
| `FactorConfig`, `PolarsFactorConfig` | a KunQuant or Polars factor or label |
| `ModelConfig` | a model head, torch or tree/tabular |
| `BacktestConfig`, `CrossSectionBacktestConfig` | a backtester |

Configs nest the way the objects do. A factor config holds the dataset object it reads from;
a model config holds lists of factor and label objects; a backtest config holds the price
dataset and the model. Each field is documented where it is defined, so the docstring of a
config class is the reference for its parameters.

The point of routing everything through configs is reproducibility. A research result is only
useful if you can say exactly how it was produced: which data, which dates, which factor
parameters, which hyperparameters, which fees. Because an object's behaviour is fully
determined by its config, writing the config down is enough to rebuild the object, and
quantlab writes it down automatically every time it trains a model or runs a backtest.
Parameters that change results (a lookback window, a filter threshold, a fee) are config
fields rather than constants in the code for the same reason.

Two behaviours of configs are worth knowing. First, an object takes ownership of its config
and completes it on assignment: it records its own class in the `name` field, fills in dates
that were left out and, for factors, resolves factor names. Second, reading is a query with
the date range as an argument: a dataset answers `panel(start, end)`, a factor answers
`compute(start, end)` or `read(start, end)`, and neither changes a config. A factor config
holds no dates; its `warmup_bars` says how many bars before `start` it reads. A model passes
its dates to its factors and labels per request and a backtester does the same with its
window, so the dates you set on the outermost object win, and one dataset or factor object
can serve several consumers.

## Rebuilding objects from config.json

`get_config()` on any dataset, factor, model or backtester returns its config as a plain,
JSON-friendly dict, with nested objects replaced by their own config dicts. The `name` entry
holds the dotted import path of the class, for example
`quantlab.dataset.stock.StockDataset`. The loaders in `quantlab.utils.module` reverse this:
they import the named class, rebuild any nested objects first, and construct the object with
the config class the class declares. In the example below, `root` is a directory holding a
small `stock.zarr` store with 30 daily bars for three symbols, from 1 January 2024.

```python
from quantlab.base.config import DatasetConfig
from quantlab.dataset.stock import StockDataset
from quantlab.utils.module import load_dataset_from_config

dataset = StockDataset(DatasetConfig(
    zarr_file_path=str(root / "stock.zarr"),
    raw_data_dir_path=str(root / "raw" / "tiingo"),
    market="us_equity", frequency="1d", vendor="tiingo",
    start_date="2024-01-08",
))
config = dataset.get_config()
print(config["name"], config["start_date"])
rebuilt = load_dataset_from_config(config)
print(type(rebuilt).__name__, rebuilt.panel("2024-01-08", "2024-02-09").sizes)
```

```text
quantlab.dataset.stock.StockDataset 2024-01-08
StockDataset Frozen({'timestamp': 25, 'symbol': 3})
```

There is one loader per layer: `load_dataset_from_config`, `load_factor_from_config`,
`load_model_from_config` and `load_backtester_from_config`. A model is rebuilt by the
`from_config` class method of the class its config names, so the backtester loader rebuilds any
predictor the same way. The backtester loader is the one
you will use most, on the `config.json` of a run directory, as the last step of the quickstart
shows. It insists that every config field is present in the file instead of filling gaps from
current defaults, because a default that changed since the run would silently produce a
different backtest.

A rebuilt object refers to the same files as the original: the same Zarr stores, the same
checkpoint. Rebuilding therefore reproduces a run as long as those files are unchanged, and
the fingerprints described below tell you when they are not. Because the class is found by
its import path, a config written before a class was moved or renamed cannot be loaded until
the path in the file is updated.

## Model checkpoint directories

`train()` creates a new trial directory under the model's `model_save_dir`, named after the
class and the time, for example `XGBoostRegressor_trial_20260925_174617_906613/`. Inside,
`XGBoostRegressor_total/` holds the checkpoint (`.joblib` for tree models, `.pth` for torch
models) and a `config.json` with the model's full config plus two records: the
hyperparameters the library actually trained with, and the factor names, label names and
symbols the model was trained on. `train_cv()` writes one sub-directory per fold and a
`cv_folds.json` manifest listing each fold's dates and checkpoint, which the backtester reads
to replay the folds. Training and evaluation metrics go to Weights & Biases.

## Backtest run directories

Each call to `run()` writes a new directory under the backtest config's `output_dir`, named
after the backtester class and the time. Its contents are:

| File | Contents |
|------|----------|
| `config.json` | the full nested config of the backtester, the price dataset and the model, plus the data fingerprints; input to `load_backtester_from_config` |
| `weights.zarr` | the target-weight panel, one `weight` per `(timestamp, symbol)` |
| `equity.zarr` | the simulated portfolio value and per-bar returns |
| `metrics.json` | performance statistics for the whole window, the in-sample part and the out-of-sample part, with their date ranges and explanatory notes |
| `settlements.json` | holdings turned into cash because their symbol delisted |
| `fingerprint.json` | a SHA-256 digest, date range and shape of each input the run read |
| `report.html` | an interactive plotly report of the equity curve and summary statistics |

The in-sample part of the window is the stretch of bars that overlaps the data the model was
trained on (including the bars its labels looked ahead to); results there are optimistic by
construction. The out-of-sample part is everything else. Both are reported separately, so a
backtest that accidentally overlaps its training data is visible rather than silently
flattering.

The fingerprints make re-runs checkable. When a run is rebuilt from its `config.json`, the
new run computes the same digests and logs a warning for every input whose data differs from
what the original run read, for example because a store was updated with newer prices.

A cross-validated backtest (`run_cv()`) writes the same files for the stitched curve across all
folds and adds a `folds/` directory with each fold's own weights and equity. See
[backtesting](backtesting.md).

## Package layout

The code follows one rule in three places. `quantlab/base/` holds the root base class of each
layer and nothing else. Each layer's top level holds its extension framework: the classes you
subclass to write your own factor, label, model or backtester. Each layer's `predefined/`
package holds the implementations quantlab ships. To add a new model you subclass
`quantlab.model.torch_model.TorchModel` or `quantlab.model.library_model.LibraryModel`; to
add a KunQuant factor you subclass `quantlab.factor.kunquant.FactorKunQuant`; to add a data
source you subclass `quantlab.base.acquisition.Acquisition` and register it. The
[extending guide](../developer-guide/extending.md) walks through each case.

```text
quantlab/
    base/            the root base class of each layer
        config.py        every config dataclass
        acquisition.py   Acquisition: resumable vendor downloads
        data.py          BaseDataset, MarketDataset: raw tier to Zarr panel
        constituent.py   IndexConstituentDataset: index-membership panels
        factor.py        Factor
        model.py         BaseModel: the training lifecycle shared by every model
        backtest.py      BaseBacktester, the Predictor protocol and the result types
        backend.py       DataBackend, ModelBackend: storage interfaces
        rebuild.py       BaseStoreRebuilder
    acquisition/     one module per vendor: tiingo.py, alpaca.py, wrds/
    dataset/         one entry per dataset: stock.py, spot.py, constituent.py, crsp/, nbbo/
    factor/          kunquant.py (FactorKunQuant), polars.py (FactorPolars)
        predefined/      Alpha101, Alpha158, literature alphas, momentum, residual momentum,
                         market features
    label/           forward.py (Forward: a factor shifted into a label)
        predefined/      fret.py: the forward-return and volatility labels
    model/           torch_model.py (TorchModel), library_model.py (LibraryModel, Rows,
                     MlBackend), ensemble.py (BaseEnsemble), training_target.py, torch_data.py,
                     torch_training.py (target transforms, losses, stopping rules)
        predefined/      xgb.py, xgb_td.py, realmlp.py, gats.py, master.py, seed_ensemble.py,
                         model_ensemble.py
    backtest/        engine_vectorbt.py (VectorBtBacktester), selection.py (rebalance schedule)
        predefined/      us_equity.py: the US-equity backtester
    portfolio/       portfolio construction: the rule from one bar's scores to weights
        predefined/      top_n.py: TopNConstructor
    my_ops/          custom KunQuant operators
    utils/           config loaders (module.py), metrics, progress, chunking, download ledgers,
                     report, CLI helpers
    config/          data-root resolution and config factories for the bundled datasets
    enums/           shared literal types (markets, frequencies, vendors)
    backend.py       XrBackend (Zarr) and PlBackend (parquet)
    registry.py      DataSourceRegistry: which vendor serves which data
    universe.py      point-in-time symbol universe
scripts/             command-line entry points, one folder per vendor
    wrds/            index.py, market.py, etf.py, nbbo.py: download and convert WRDS data
tests/               the pytest suite, fully offline
examples/            runnable example scripts
```

Directories whose name starts with an underscore, such as `quantlab/dataset/_support/`, hold
private helpers for the layer above them and are not meant to be imported from outside it.
Import every class by its full module path, for example
`from quantlab.factor.predefined.alpha158 import Alpha158Stock`; the package `__init__.py` files do not
re-export anything.
