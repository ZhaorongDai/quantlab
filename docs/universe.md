# Price and liquidity universe filter (universe)

English | [简体中文](zh-CN/universe.md)

`UniverseFilteredFactor` restricts a KunQuant factor or label to the symbols that are tradable at each bar: raw close above a price floor and trailing average dollar volume above a liquidity floor. It is a factor wrapper. The wrapper is itself a `FactorKunQuant`, so it can be placed in `MLConfig.factors` and `MLConfig.labels` and used with a backtester without any change to the model or backtest layers.

The filter is separate from index membership (see the `constituent` guide). Membership says which symbols belong to an index on a date; this filter says which symbols are expensive and liquid enough to trade. The two can be used together.

## Prerequisites

The factor backend is KunQuant, which compiles the factor graph and needs a working C++ compiler. Batch runs need the number of symbols to be a multiple of the SIMD block width of the host; 16 symbols work on the machine used for this page.

## The basics

### The rule

A symbol is in the universe at bar `t` when two conditions hold. The raw `close` at `t` is at least `min_price`, and the mean of raw `close * volume` over the `window` bars ending at `t` is at least `min_dollar_volume`. Both use the raw columns and never the adjusted ones, because adjusted history is changed by later splits and dividends and cannot say whether a stock was cheap at the time. Nothing after `t` affects the decision at `t`. A window that is not yet full, or that contains a NaN, counts as out of the universe.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `min_price` | `5.0` | Minimum raw close |
| `min_dollar_volume` | `1_000_000.0` | Minimum trailing mean of raw `close * volume` |
| `window` | `20` | Number of bars in the trailing mean |

The examples use a synthetic 30-bar panel with 16 symbols: thirteen ordinary names `S00` to `S12`, `PENY` (raw close of $1 but the highest adjusted close, so an unfiltered ranking prefers it), `ILQD` (ordinary price, tiny volume) and `DRPX` (in the universe until bar 20, then raw close falls to $1). The first session builds the store and a small KunQuant factor with one cross-sectional output (`Rank`) and one time-series output (`WindowedAvg`).

```python
>>> import os, tempfile
>>> import numpy as np, pandas as pd, xarray as xr
>>> from KunQuant.Op import Builder, Input, Output, Rank
>>> from KunQuant.ops import WindowedAvg
>>> from KunQuant.Stage import Function
>>> from quantlab.base.config import DatasetConfig, FactorConfig
>>> from quantlab.base.factor import FactorKunQuant
>>> from quantlab.dataset.stock import StockDataset
>>> from quantlab.factor.universe_filter import UniverseFilteredFactor
>>> def write_store(root, symbols):
...     n_bars, n = 30, len(symbols)
...     rng = np.random.default_rng(0)
...     adjusted = 50 * np.exp(np.cumsum(rng.normal(0, 0.02, (n_bars, n)), axis=0))
...     raw = adjusted * 1.7
...     opened = adjusted * 1.001
...     volume = np.full((n_bars, n), 1e6)
...     at = symbols.index
...     raw[:, at("PENY")] = 1.0
...     adjusted[:, at("PENY")] = 900.0
...     volume[:, at("ILQD")] = 100.0
...     raw[20:, at("DRPX")] = 1.0
...     adjusted[:, at("DRPX")] = 700.0
...     dims = ("timestamp", "symbol")
...     panel = xr.Dataset(
...         {"adjOpen": (dims, opened), "adjClose": (dims, adjusted),
...          "close": (dims, raw), "volume": (dims, volume)},
...         coords={"timestamp": pd.bdate_range("2024-01-01", periods=n_bars), "symbol": symbols},
...     )
...     store = os.path.join(root, "stock.zarr")
...     panel.to_zarr(store, mode="w")
...     return DatasetConfig(
...         raw_data_dir_path=os.path.join(root, "raw"), zarr_file_path=store,
...         catalog_path=os.path.join(root, "catalog"), market="us_equity", frequency="1d",
...     )
>>> class RankClose(FactorKunQuant):
...     def _get_factor_names(self):
...         return ("rank_close", "ma_close")
...     def _get_factor_func(self):
...         builder = Builder()
...         with builder:
...             close = Input("adjClose")
...             Output(Rank(close), "rank_close")
...             Output(WindowedAvg(close, 3), "ma_close")
...         return Function(builder.ops)
...     def _get_features(self, data):
...         return data
...     def _get_labels(self, data):
...         raise RuntimeError("RankClose is a feature")
>>> symbols = [f"S{i:02d}" for i in range(13)] + ["PENY", "ILQD", "DRPX"]
>>> dataset_config = write_store(tempfile.mkdtemp(), symbols)
>>> def make_factor():
...     config = FactorConfig(window=3, dataset=StockDataset(dataset_config), mode="batch",
...                           data_columns=("adjClose",), njobs=4)
...     return RankClose(config)
```

### Wrapping a factor

The wrapper takes the inner factor and the three parameters. Its `config` is the inner factor's own config object, not a copy, so dates written by a model or backtester land on the inner factor.

```python
>>> wrapped = UniverseFilteredFactor(make_factor(), min_price=5.0, min_dollar_volume=1_000_000.0, window=3)
>>> wrapped.config is wrapped.factor.config
True
```

`compute_universe_mask` returns the mask for a panel: 1.0 where a symbol is in the universe and NaN elsewhere. The first `window - 1` rows are NaN for everyone because the trailing mean needs a full window. PENY fails on price and ILQD on dollar volume at every bar. DRPX is in until bar 19 (2024-01-26) and out from bar 20.

```python
>>> panel = StockDataset(dataset_config).read().get_xarray_dataset()
>>> mask = wrapped.compute_universe_mask(panel)
>>> mask.sel(symbol=["S00", "PENY", "ILQD", "DRPX"]).isel(timestamp=[0, 1, 2, 19, 20]).to_pandas()
symbol      S00  PENY  ILQD  DRPX
timestamp                        
2024-01-01  NaN   NaN   NaN   NaN
2024-01-02  NaN   NaN   NaN   NaN
2024-01-03  1.0   NaN   NaN   1.0
2024-01-26  1.0   NaN   NaN   1.0
2024-01-29  1.0   NaN   NaN   NaN
```

### Symbols are masked, not dropped

`cal()` runs the compiled graph and `get_features()` returns the result with the mask applied. Being out of the universe blanks cells; it never removes a column. The symbol axis of the output equals the input's, and a symbol that is out for the whole window stays as an all-NaN column. The symbol axis therefore does not depend on the date window, so a model trained on one window can be given a panel from another.

```python
>>> features = wrapped.cal().get_features()
>>> features.sizes
Frozen({'timestamp': 30, 'symbol': 16})
>>> bool(features["rank_close"].sel(symbol="PENY").isnull().all())
True
```

### Cross-sectional operators see only the universe

Blanking the output of an out-of-universe symbol would not be enough for an operator such as `Rank`, because the symbol would still take part in every rank. The wrapper therefore rewrites the factor graph: the mask is added as an extra input, and every input of every cross-sectional operator is divided by it. Dividing by 1.0 leaves a value unchanged and dividing by NaN gives NaN, so out-of-universe symbols are absent from every rank and cross-sectional z-score. Time-series operators are not rewritten and still see full history.

The last-bar ranks below show the effect. Without the filter PENY ranks first (1.0) among 16 symbols; with it PENY is absent, and the remaining symbols are ranked among 13.

```python
>>> unfiltered = make_factor().cal().get_features()
>>> unfiltered["rank_close"].isel(timestamp=-1).sel(symbol=["S00", "S01", "PENY"]).to_pandas()
symbol
S00     0.4375
S01     0.5000
PENY    1.0000
Name: rank_close, dtype: float32
>>> features["rank_close"].isel(timestamp=-1).sel(symbol=["S00", "S01", "PENY", "ILQD", "DRPX"]).to_pandas()
symbol
S00     0.461538
S01     0.538462
PENY         NaN
ILQD         NaN
DRPX         NaN
Name: rank_close, dtype: float32
```

The purely time-series output `ma_close` is identical with and without the filter wherever the filtered output is defined.

```python
>>> in_universe = features["ma_close"].notnull()
>>> bool((features["ma_close"] == unfiltered["ma_close"]).where(in_universe, True).all())
True
```

## Common tasks

### Wrap the factors and the labels of a model

Wrap both. If only the factors are wrapped, label rows for out-of-universe symbols remain; if only the labels are wrapped, the cross-sectional operators in the factors are still affected by those symbols. The price dataset given to the backtester is not wrapped: the prices of held positions must stay available. Model and backtester calls are shown here as a shape only.

```python
from quantlab.factor.alpha101 import Alpha101Stock
from quantlab.factor.universe_filter import UniverseFilteredFactor
from quantlab.label.fret import Return

factors = [UniverseFilteredFactor(Alpha101Stock(factor_config))]
labels = [UniverseFilteredFactor(Return(label_config))]
model = XGBoostRegressor(MLConfig(factors=factors, labels=labels, ...))
```

The wrapper moves the inner dataset's start date earlier by `2 * window + 10` calendar days so that the trailing mean is full at the first requested bar. It never moves a start date later.

### Labels are masked at their own timestamp

A forward-return label at bar `t` describes a position opened after `t`. The wrapper applies the mask after the label's forward shift, so the mask at `t` decides the label at `t`. A symbol that is in the universe at its last in-universe bar keeps its label for that bar, including the return earned as it drops out. Masking before the shift would have used the universe at a later bar to decide an earlier one.

`Return` with `n_forward_periods=1` is the open-to-open return from the next open to the open after it. DRPX leaves the universe at bar 20 (2024-01-29); its label at bar 19 is unchanged and the labels from bar 20 are blank.

```python
>>> from quantlab.label.fret import Return
>>> def make_label():
...     config = FactorConfig(window=0, dataset=StockDataset(dataset_config), mode="batch",
...                           data_columns=("adjOpen",), njobs=4, kwargs={"n_forward_periods": 1})
...     return Return(config)
>>> label = UniverseFilteredFactor(make_label(), min_price=5.0, min_dollar_volume=1_000_000.0, window=3)
>>> labels = label.cal().get_labels()
>>> labels["ret_1"].sel(symbol="DRPX").isel(timestamp=slice(17, 23)).to_pandas()
timestamp
2024-01-24    0.013593
2024-01-25   -0.006783
2024-01-26    0.041368
2024-01-29         NaN
2024-01-30         NaN
2024-01-31         NaN
Name: ret_1, dtype: float32
>>> make_label().cal().get_labels()["ret_1"].sel(symbol="DRPX").isel(timestamp=slice(17, 23)).to_pandas()
timestamp
2024-01-24    0.013593
2024-01-25   -0.006783
2024-01-26    0.041368
2024-01-29   -0.012877
2024-01-30    0.002454
2024-01-31    0.008919
Name: ret_1, dtype: float32
```

### A holding that leaves the universe

A symbol that drops out has all-NaN features, so a model's prediction for it is NaN, and a NaN score is not selectable. The filter does not act on positions directly. At the first rebalance bar after the drop the symbol is no longer selectable, its target weight becomes zero, and the position is closed at the next bar's open. Eligibility is only re-evaluated on rebalance bars, so a holding can be kept for up to `rebalance_periods - 1` bars after it leaves. A smaller `rebalance_periods` shortens the delay.

The session below uses the ranks computed above as stand-in scores and the top-2 selector of the backtest layer, rebalancing on rows 0 and 3 of a six-bar window. DRPX is selected at the first rebalance (2024-01-25, before it drops) and has weight 0.0 at the second (2024-01-30). Rows without a rebalance are all NaN, which means "keep the current position".

```python
>>> from quantlab.backtest.selection import CrossSectionTopNSelector, rebalance_mask
>>> scores = features["rank_close"].isel(timestamp=slice(18, 24))
>>> selector = CrossSectionTopNSelector(direction="long_only", top_n=2)
>>> weights = selector.select(scores, xr.full_like(scores, 100.0), rebalance_mask(6, 3))["weight"]
>>> held = weights.isel(timestamp=[0, 3]).to_pandas()
>>> held.loc[:, (held != 0).any()]
symbol      DRPX  S07  S08
timestamp                 
2024-01-25   0.5  0.5  0.0
2024-01-30   0.0  0.5  0.5
```

### Save and read back

Factor stores written through the wrapper hold the outputs of the rewritten graph, before the output mask. `read()` recomputes the mask from the dataset's raw close and volume and applies it to what `get_features()` returns. A store must therefore be written by the wrapper: a store written by the unwrapped inner factor already contains cross-sectional values that include out-of-universe symbols, and reading it through the wrapper cannot remove them.

```python
>>> root = tempfile.mkdtemp()
>>> def make_stored():
...     config = FactorConfig(window=3, dataset=StockDataset(dataset_config), mode="batch",
...                           data_columns=("adjClose",), njobs=4,
...                           file_path=os.path.join(root, "rank.zarr"),
...                           start_date="2024-01-01", end_date="2024-02-09")
...     return UniverseFilteredFactor(RankClose(config), window=3)
>>> _ = make_stored().cal().save(mode="w")
>>> back = make_stored().read().get_features()
>>> back.sizes
Frozen({'timestamp': 30, 'symbol': 16})
>>> bool(back["rank_close"].sel(symbol="PENY").isnull().all())
True
```

### Serialize and rebuild

`get_config()` returns the wrapper's three parameters and the inner factor's config under `"factor"`. `from_config()` rebuilds the wrapper and the inner factor from that dict. It does not fill a missing parameter from the current defaults, so a stored run cannot be rebuilt with a different universe than it used.

```python
>>> cfg = wrapped.get_config()
>>> sorted(cfg)
['factor', 'min_dollar_volume', 'min_price', 'name', 'window']
>>> cfg["window"], cfg["min_price"], cfg["min_dollar_volume"]
(3, 5.0, 1000000.0)
>>> UniverseFilteredFactor.from_config({"factor": cfg["factor"], "window": 3})
Traceback (most recent call last):
  ...
ValueError: UniverseFilteredFactor.from_config: refusing to rebuild -- missing key(s) ['min_dollar_volume', 'min_price'], unknown key(s) []. Missing parameters are NOT filled from the current defaults, ...
```

## Extending

The thresholds are parameters; the rule itself is the method `compute_universe_mask(panel)`, which returns a `(timestamp, symbol)` array that is 1.0 in the universe and NaN elsewhere. A subclass can add a condition by refining that result. The batch path (`cal()` and `read()`) calls the method. The streaming path builds its mask row inside `cal_stream()` and does not, so a rule that must also apply to streaming has to be repeated there.

The example adds a cap on the raw close. Ordinary symbols start near 85 and drift, so only 5 of the 13 stay under 80 on the last bar.

```python
>>> class CappedUniverse(UniverseFilteredFactor):
...     def compute_universe_mask(self, panel):
...         mask = super().compute_universe_mask(panel)
...         return mask.where(panel["close"] <= 80.0).rename(mask.name)
>>> capped = CappedUniverse(make_factor(), min_price=5.0, min_dollar_volume=1_000_000.0, window=3)
>>> capped.cal().get_features()["rank_close"].isel(timestamp=-1).notnull().sum().item()
5
>>> features["rank_close"].isel(timestamp=-1).notnull().sum().item()
13
```

## Notes

The wrapper wraps `FactorKunQuant` subclasses only. A Polars factor is refused because its cross-sectional logic is a Polars expression and cannot be rewritten, and masking only its outputs would leave out-of-universe symbols inside every rank. Wrapping a wrapper is refused, because the two masks would compose silently. The window must be at least one bar.

```python
>>> UniverseFilteredFactor(object())
Traceback (most recent call last):
  ...
TypeError: UniverseFilteredFactor wraps a FactorKunQuant, got object. A Polars factor's cross-sectional expressions are polars expressions, not a KunQuant op graph, so they cannot be rewritten -- and masking only the OUTPUTS would leave every out-of-universe symbol sitting inside each rank/zscore, which is exactly what this class exists to prevent.
>>> UniverseFilteredFactor(wrapped)
Traceback (most recent call last):
  ...
TypeError: UniverseFilteredFactor cannot wrap another UniverseFilteredFactor: the inner wrapper would mask the cross-sections a second time, and the two masks' parameters would silently compose. Wrap the innermost factor once, with the parameters you want.
>>> UniverseFilteredFactor(make_factor(), window=0)
Traceback (most recent call last):
  ...
ValueError: window must be >= 1 bar, got 0; it is the number of bars the trailing dollar-volume mean is taken over.
```

The mask needs the raw `close` and `volume` variables in the dataset. A panel that lacks either raises `ValueError` from `compute_universe_mask`, which lists the variables that are present. Calling `get_features()` or `get_labels()` before `cal()`, `read()` or `cal_stream()` raises `RuntimeError`.

```python
>>> wrapped.compute_universe_mask(panel.drop_vars("volume"))
Traceback (most recent call last):
  ...
ValueError: UniverseFilteredFactor needs the RAW column 'volume' to decide universe membership, and the dataset panel does not carry it (present: ['adjClose', 'adjOpen', 'close']). The mask reads RAW close/volume, never the adjusted columns: adjusted history is depressed by splits and dividends, so a penny stock today can look like a $50 stock in 2015.
>>> UniverseFilteredFactor(make_factor(), window=3).get_features()
Traceback (most recent call last):
  ...
RuntimeError: UniverseFilteredFactor: no universe mask has been computed yet, so the outputs cannot be masked. Call cal(), read() or cal_stream() first.
```

Two KunQuant limits apply to every caller. Batch runs always start at bar 0, because in KunQuant 0.1.11 a non-zero start gives wrong results for every cross-sectional operator. The number of symbols must be a multiple of the SIMD block width. A panel with 13 symbols on the machine used for this page fails with the KunQuant error below, and the fix is to pad or trim the symbol set.

```python
>>> symbols13 = [f"S{i:02d}" for i in range(10)] + ["PENY", "ILQD", "DRPX"]
>>> config13 = write_store(tempfile.mkdtemp(), symbols13)
>>> config = FactorConfig(window=3, dataset=StockDataset(config13), mode="batch",
...                       data_columns=("adjClose",), njobs=4)
>>> UniverseFilteredFactor(RankClose(config), window=3).cal()
Traceback (most recent call last):
  ...
RuntimeError: Bad shape at adjClose
```

In streaming mode, each bar's `data` dict passed to `cal_stream()` must include the raw `close` and `volume` arrays in addition to `config.data_columns`; the inherited push sends only `data_columns`, and a missing key raises `ValueError`. A time-series operator applied on top of a cross-sectional one is NaN for its whole window after a symbol re-enters the universe, because its input was NaN while the symbol was out.

## See also

The `constituent` guide covers index membership panels, which answer a different question and can be combined with this filter. The `factor` guide covers `FactorKunQuant` and the operators, `model` covers passing wrapped factors to a model, and `backtest` covers rebalancing and how positions are closed. Class docstring: `quantlab.factor.universe_filter.UniverseFilteredFactor`.
