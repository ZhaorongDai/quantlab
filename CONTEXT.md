# quantlab

A config-driven quantitative research pipeline: market data from a vendor, through factors, return
models and portfolio optimisation, to target holdings and a backtest. This glossary fixes the words
the code, docs and scripts use for the concepts specific to this project.

## Language

### Data acquisition

**Vendor**:
An external company quantlab downloads market data from, such as WRDS.
_Avoid_: provider, data provider

**Source**:
One registered capability of a vendor: a market, a frequency and a data type that the registry
can download and convert.
_Avoid_: provider, feed

**ETF**:
An exchange-traded fund downloaded from CRSP by PERMNO as its own single-symbol store.
_Avoid_: benchmark (a benchmark is the role an ETF plays inside a backtest, not the data)


**Index**:
The set of stocks that are members of a named equity index (S&P 500, Nasdaq-100) on a given date,
and the market data restricted to those members.
_Avoid_: constituents-only, universe (a universe is a filtered selection built on top of index or
market data, not the raw membership)

**Market**:
Every stock a vendor covers for an exchange group, with no membership restriction.
_Avoid_: all, full, full market, whole market, all-stocks
