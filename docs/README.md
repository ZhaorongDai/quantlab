# quantlab documentation

quantlab is a configuration-driven backend for quantitative equity research: it downloads
market data, turns it into panels, computes factors, trains return models and backtests the
resulting portfolios. This documentation is organised in three parts. If you are new, read
*Getting started* first, then the *Concepts* page, then whichever user-guide page matches what
you want to do.

## Getting started

[Installation](getting-started/installation.md) covers requirements, installing with `uv`,
running the tests, GPU and macOS notes, and the environment variables that hold vendor
credentials.

[Quickstart](getting-started/quickstart.md) is a ten-minute tour of the whole pipeline on
synthetic data, from a price panel to a backtest report.

## User guide

[Concepts](user-guide/concepts.md) explains the pipeline stages and what each consumes and
produces, the panel format, configuration objects, and the directories that models and
backtests write.

[Data sources](user-guide/data-sources.md) explains how to download from Tiingo, Alpaca and
WRDS, where the files go, how to resume an interrupted download, and how to check what is
already on disk.

[WRDS: CRSP and TAQ](user-guide/wrds.md) covers the research-grade US stock data available
through a WRDS account: CRSP daily stock files and TAQ best-quote data.

[Datasets and storage](user-guide/datasets.md) explains how raw files become a panel, how to
filter and store it, and how to convert histories too large to fit in memory.

[Universes](user-guide/universes.md) explains survivorship bias, point-in-time index
membership, and how to restrict factors and backtests to the stocks that were actually
tradable on each day.

[Factors and labels](user-guide/factors.md) shows how to compute the built-in factor sets and
how to write your own factors with KunQuant or Polars, plus the forward-return labels models
learn to predict.

[Models](user-guide/models.md) covers the available model heads, training, prediction,
evaluation, walk-forward cross-validation and checkpoints.

[Backtesting](user-guide/backtesting.md) explains how predictions become target weights, when
trades are filled, how delisted holdings are handled, and how to read and reproduce a
backtest run.

## Developer guide

[Extending quantlab](developer-guide/extending.md) walks through adding a data source, a
dataset, a storage backend, a factor, a model head and a backtest rule, each with a minimal
working example.

[Internals](developer-guide/internals.md) describes the machinery that makes long jobs safe to
interrupt: resumable downloads and conversions, rebuilds, the volume check, atomic writes and
data fingerprints.

## Examples and API reference

The [examples](../examples/README.md) directory has runnable scripts that go with these pages.
Every public class and function has a numpydoc docstring; read it with `help()` in Python or
in your editor.
