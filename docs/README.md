# quantlab documentation

English | [简体中文](zh-CN/README.md)

This is the user guide for quantlab. Each page introduces one part of the pipeline and works
through runnable examples. Start with the guide for the stage you need, or read them in order:
the guides follow the path that data takes, from a vendor to a backtest report.

Every code example that prints output shows output that was produced by running it. Examples
that need a vendor account say what they need and do not print results.

## Getting data

| Guide | What it covers |
|-------|----------------|
| [Acquisition](acquisition.md) | How a download runs: batches, failure isolation, incremental refresh, the raw file layout and the volume guard |
| [Data source registry](registry.md) | The catalogue of sources, the `run()` and `convert()` entry points, progress events and the read-only inspector |
| [Resumable downloads](pageledger.md) | How a multi-page download resumes after an interruption |
| [WRDS CRSP daily stocks](wrds_crsp.md) | Research-grade US daily data by PERMNO, total-return adjustment and delisting returns |
| [WRDS TAQ quotes](wrds_taq.md) | National best bid and offer quotes and their resampling to bars |

## Universes

| Guide | What it covers |
|-------|----------------|
| [Index constituents](constituent.md) | Point-in-time membership panels for the S&P 500 and Nasdaq-100 |
| [Universe filtering](universe.md) | Price and liquidity filters implemented as a factor wrapper |

## Building panels

| Guide | What it covers |
|-------|----------------|
| [Datasets](dataset.md) | From raw files to the `(timestamp, symbol)` panel, and adding a market |
| [Chunked conversion](chunking.md) | Converting a large date range one window at a time |
| [Storage backends](backend.md) | Zarr and Parquet storage, appending, and writing a backend |

## Research

| Guide | What it covers |
|-------|----------------|
| [Factors](factor.md) | The KunQuant and Polars factor backends, labels and normalization |
| [Models](model.md) | The model hierarchy, training, cross-validation and checkpoints |
| [Backtesting](backtest.md) | Target weights, simulation, metrics and run directories |

## Package layout

```text
quantlab/
  base/         abstract contracts: datasets, factors, models, backtesters, acquisition
  acquisition/  Tiingo, Alpaca and WRDS downloaders
  dataset/      concrete datasets: spot klines, US stocks, CRSP, NBBO, index constituents
  factor/       factor sets: Alpha101, Alpha158, momentum, universe filter
  label/        forward-return labels
  dl_model/     PyTorch model heads: MLP, GRU and LSTM
  ml_model/     XGBoost head and checkpoint storage for non-torch models
  backtest/     vectorbt engine, top-N selection, US-equity backtester
  backend.py    Zarr and Parquet storage backends
  registry.py   catalogue of data sources and the run() and convert() entry points
  universe.py   point-in-time symbol universes and the download volume guard
  config/       config factories and packaged instrument metadata
  utils/        command-line helpers, metrics, serialization, report writer
scripts/        command-line entry points for downloading and converting data
tests/          the test suite
```

Every public class and function also has a docstring in the numpydoc format, with an
`Examples` section. Use `help()` on a class to read it, for example
`help(quantlab.base.model.DLModel)`.
