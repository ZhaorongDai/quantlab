# Backtesting

A backtest replays a strategy over historical prices to see how it would have
performed. In quantlab the backtester is the last stage of the pipeline: it
takes a trained return model and a price dataset, turns the model's
predictions into target portfolio weights, simulates trading those weights
with the vectorbt library, and writes a self-contained run directory with the
weights, the equity curve, the metrics and an HTML report. This page explains
what the backtester does at each step, how signals are timed and executed, how
the metrics are split into in-sample and out-of-sample parts, how to backtest
a walk-forward cross-validation, and how to rebuild and re-run a stored
backtest. Read it after [Models](models.md). To write your own market
conventions or selection rule, see
[Extending quantlab](../developer-guide/extending.md#a-backtest-market-or-selection-rule).

The runnable companion to this page is
[`examples/backtest.py`](../../examples/backtest.py). It builds everything from
synthetic data in a temporary directory, needs no credentials or network
access and runs in under a minute on a laptop CPU:

```bash
uv run python examples/backtest.py
```

All output shown on this page comes from that script.

## What the backtester does

The concrete class shipped with quantlab is
`USEquityCrossectionSelectStockVectorBt` in `quantlab.backtest.us_equity`. It
is a *cross-sectional stock-selection* backtester: on each rebalance day it
ranks every symbol in the price dataset by the model's score and holds the
best ones. It is configured with a `CrossSectionBacktestConfig`
(`quantlab.base.config`).

A call to `run()` performs these steps in order:

1. Prepare the model. In `model_mode="train"` the model is trained on the
   dates in its own config; in `model_mode="load"` a checkpoint is restored.
2. Re-date every factor so that it covers the backtest window plus a warm-up
   period, recompute the features, and let the model predict a *panel* (an
   `xarray.Dataset` indexed by `timestamp` and `symbol`) of scores.
3. Turn the scores into target weights on the rebalance bars.
4. Simulate the weights with vectorbt.
5. Compute metrics for the whole window and separately for the parts of it
   that the model did and did not see during training.
6. Write everything to a new run directory under `output_dir`.

The class hierarchy mirrors those responsibilities. `BaseBacktester`
(`quantlab.base.backtest`) owns the steps that do not depend on a simulation
engine: dates, warm-up, the in-sample split, metrics and persistence.
`VectorBtBacktester` (`quantlab.backtest.engine_vectorbt`) implements the
simulation with vectorbt. The concrete class adds the market's price columns
(a `MarketSpec`) and the rule that turns scores into weights.

## A first backtest

The example trains a small linear model on a one-factor momentum signal
inside the backtest itself. With `model_mode="train"`, the backtester calls the
model's `collect()` and `train()` before predicting:

```python
from quantlab.backtest.us_equity import USEquityCrossectionSelectStockVectorBt
from quantlab.base.config import CrossSectionBacktestConfig

backtester = USEquityCrossectionSelectStockVectorBt(
    CrossSectionBacktestConfig(
        price_dataset=fresh_dataset(prices),   # a StockDataset
        model=make_model(root, prices, **dates),
        model_mode="train",
        start_date=day(175),
        end_date=day(N_BARS - 1),
        output_dir=str(root / "backtests"),
        rebalance_periods=5,
        direction="long_only",
        top_n=3,
        fees=0.0005,
        slippage=0.0005,
    )
)
result = backtester.run()
```

`run()` returns a `BacktestResult` whose `metrics` mapping is the same
content as the run's `metrics.json`:

```text
== long-only top 3, train mode
run directory: USEquityCrossectionSelectStockVectorBt_20260925_175054_760512
  Total Return [%]      9.170
  Sharpe Ratio          1.888
  Max Drawdown [%]      6.116
  Total Orders            126
  turnover/rebal. [%]    139.9
  training window     ('2023-01-02', '2023-09-15')
  in-sample range     ('2023-09-04', '2023-09-15')
  out-of-sample       [('2023-09-18', '2024-02-23')]
  out-of-sample Sharpe 1.713
files: ['config.json', 'equity.zarr', 'fingerprint.json', 'liquidations.json', 'metrics.json', 'report.html', 'weights.zarr']
first rebalance: {'S08': 0.3333, 'S10': 0.3333, 'S11': 0.3333}
forced liquidation: S08 signal 2023-12-18 fill 2023-12-19 at 72.50
```

To backtest a model you have already trained, pass `model_mode="load"` and
the checkpoint path. The `model` object must still be configured with the
same factors and labels, because the backtester uses them to recompute the
features; the checkpoint supplies only the fitted parameters. The example
reuses the checkpoint that the first run trained:

```python
checkpoint = result.metrics["trained_checkpoint"]
long_short = USEquityCrossectionSelectStockVectorBt(
    CrossSectionBacktestConfig(
        price_dataset=fresh_dataset(prices),
        model=make_model(root, prices, **dates),
        model_mode="load",
        checkpoint=checkpoint,
        start_date=day(200),
        end_date=day(N_BARS - 1),
        output_dir=str(root / "backtests"),
        rebalance_periods=5,
        direction="long_short",
        top_n=3,
    )
)
ls_result = long_short.run()
```

The config is validated when the backtester is constructed, before any data
is read: a wrong config class raises `TypeError`, and invalid dates, a
`rebalance_periods` below 1, negative fees or slippage, a non-positive
`init_cash`, an unknown `direction` or a `score_label` the model does not
predict raise `ValueError`. See the docstrings of
`quantlab.base.config.BacktestConfig` and `CrossSectionBacktestConfig` for
every field.

## Target weights: the signal format

Whatever rule produces them, the signals the engine simulates are *target
weights*: an `xarray.Dataset` with one variable, `weight`, on
`(timestamp, symbol)`, with exactly the axes of the price data. A weight is
the fraction of portfolio value a symbol should be worth after the order
fills; a negative weight is a short position. Target weights say where the
portfolio should be, not how many shares to trade, so the same signal works
whatever the current holdings are.

Each row of the weight panel is one of two kinds:

- A hold row is entirely NaN. It means "do nothing on this bar; keep the
  current positions".
- A rebalance row is entirely finite. Symbols that should not be held get
  exactly `0.0`. The *gross exposure* of the row, the sum of the absolute
  weights, must be at most 1, so the portfolio is never leveraged.

The backtester checks these rules before simulating and raises `ValueError`
naming the first offending bar. The strictness is deliberate: vectorbt reads
a NaN on a rebalance row as "keep this position", and a single NaN would hold
cash that the other orders on that row need, silently blocking the rebalance.
A rebalance row with nothing to buy is therefore all zeros (the portfolio goes
to cash), never all NaN.

The weights of every run are saved as `weights.zarr`, so any other tool can
read exactly what was traded.

## Execution timing

A signal formed at bar *t* fills at the open of bar *t + 1*. For daily bars:
the model sees data up to Monday's close, forms its scores, and the orders
execute at Tuesday's open. Internally the weight panel is shifted one bar
before it reaches vectorbt, and the fill price is the market's fill column
(`adjOpen`, the split- and dividend-adjusted open, for US equities).
Positions are valued at the market's valuation column (`adjClose`).

The one-bar delay is there to prevent *look-ahead bias*, which is using
information that would not have been available when the decision was made.
The close of bar *t* is only known once bar *t* has ended, so no real order
could have filled at that price with a signal computed from it. Filling at
the next bar's open is the earliest honest price.

Two consequences follow. First, the label your model learns should describe
the return that the backtest actually earns, which starts at the open of
*t + 1*. The `Return` label in `quantlab.label.fret` is defined that way
(`adjOpen[t + n + 1] / adjOpen[t + 1] - 1`), and so is the label in the
example script. Second, a target percentage is measured against the
portfolio's value at the fill price of the bar it executes on, which is
vectorbt's default.

## Rebalancing and top-N selection

The portfolio rebalances on the first bar of the window and every
`rebalance_periods` bars after it; the other bars are hold rows. The last bar
of the window never rebalances, because a signal formed there has no next bar
inside the window to fill on. `quantlab.backtest.selection.rebalance_mask`
computes this schedule.

On each rebalance bar, `CrossSectionTopNSelector` builds equal-weight
portfolios from the scores:

- With `direction="long_only"`, the `top_n` highest-scoring symbols get a
  weight of `1 / top_n` each, for a gross exposure of 100%.
- With `direction="long_short"`, the `top_n` highest-scoring symbols get
  `+0.5 / top_n` each and the `top_n` lowest-scoring get `-0.5 / top_n`
  each. The two books never share a symbol, so the gross exposure is 100% and
  the *net exposure* (the sum of the signed weights) is zero.

The example prints the exposures of the first long/short rebalance row:

```text
== long/short top 3 / bottom 3, load mode
run directory: USEquityCrossectionSelectStockVectorBt_20260925_175055_950306
  Total Return [%]      3.257
  Sharpe Ratio          1.382
  Max Drawdown [%]      2.315
  Total Orders            175
  turnover/rebal. [%]    144.1
  gross exposure 1.0 net exposure 0.0
```

The score is the model's prediction of the label named by `score_label`, or
of its first label when `score_label` is `None`. A symbol is eligible on a bar
only when its score is finite and it has a price on the next bar, since that
is where the order would fill. Ties are broken by symbol order, so the same
panel always gives the same weights. When fewer than `top_n` symbols are
eligible, the book is split among the eligible ones and a warning names the
bar.

## Warm-up

A factor needs history before it has a value: a 20-bar momentum is undefined
for the first 20 bars it sees. The *warm-up* is the extra history read before
the backtest window so that every factor has a value on the first bar of the
window. The backtester takes the largest `window` among the model's factor
configs and starts every factor that many price bars before `start_date`.
The count is in bars on the price calendar, not in calendar days, so weekends
and holidays do not shorten it. If the price data does not reach back far
enough, the warm-up starts at the first available bar and a warning says how
many bars short it is.

## Universe, listings and delistings

The universe of a backtest is every symbol in the price dataset. Predictions
are aligned onto the price data's symbols by label, and a symbol the model
has no prediction for gets a NaN score and is never selected. If the price
dataset contains only today's survivors, the backtest inherits their
*survivorship bias* (the delisted losers are missing from history); build the
price dataset from a point-in-time universe to avoid it, as described in
[Universes](universes.md).

A symbol whose prices are NaN at the start of the window and that has never
been held is treated as not yet listed. It trades normally once its prices
appear.

A holding whose prices disappear is treated as delisted. Both price columns
are forward-filled before they reach vectorbt, so the position keeps its last
known value, and at the next rebalance the holding is sold at its last known
fill price while the rest of the portfolio rebalances normally. This is a
*forced liquidation*. It is logged, recorded in `SimulationResult.liquidations`
and written to `liquidations.json`. The record from the example run:

```json
{"symbol": "S08", "axis_symbol": "S08", "signal_timestamp": "2023-12-18T00:00:00",
 "fill_timestamp": "2023-12-19T00:00:00", "price": 72.50323178958098}
```

`axis_symbol` is the label on the panel's symbol axis. `symbol` is the name a
person reads: for CRSP data, whose axis holds numeric PERMNOs (CRSP's
permanent security identifiers), it is the ticker the security traded under on
the fill day; for other data it equals `axis_symbol`. Between the last real
price and the forced sale the position is valued at its frozen last price, so
a large `rebalance_periods` makes that stretch longer.

The forward fill matters. Without it, vectorbt keeps a NaN-priced holding and
silently skips every later rebalance of the whole portfolio, not just the
delisted symbol.

## Costs and assumptions

Three config fields set the trading costs and the starting capital:

| Field | Default | Meaning |
|---|---|---|
| `fees` | `0.0005` | Proportional fee on the traded value of each fill (5 basis points). |
| `slippage` | `0.0005` | Proportional price penalty: buys fill at `price * (1 + slippage)`, sells at `price * (1 - slippage)`. |
| `init_cash` | `1_000_000.0` | Starting cash of the simulated portfolio. |

Fractional shares are allowed, so target weights are hit exactly. No borrow
fee or short-financing cost is modelled, so short-side returns are
optimistic; this caveat is also written into the notes of `metrics.json` and
`report.html`. Returns are computed from adjusted prices, so splits and
dividends are already reflected.

Annualized figures (Sharpe ratio, annualized return and volatility, annualized
turnover) use the market's own year length from `MarketSpec.year_freq`: 252
bars per year for daily US-equity bars, 252 times 390 divided by the bar
length in minutes for intraday bars, and calendar-based counts for bars
longer than a day (about 52 for weekly bars).

## In-sample and out-of-sample metrics

*In-sample* bars are bars the model saw while it was trained; performance
there says little about the future. *Out-of-sample* bars are bars it never
saw, and they are the honest test. The backtester reports the two separately
whenever the backtest window overlaps the model's training data.

The model's *effective training window* runs from its `train_start` to its
`train_end` plus the label horizon. The horizon is the largest
`n_forward_periods` among the model's labels, read from their
`config.kwargs`: the label on the `train_end` bar is computed from the next
`n` bars of prices, so those bars also influenced training. The horizon is
counted in bars. In load mode the dates recorded in the `config.json` next to
the checkpoint are used, since those are the dates the checkpoint was really
trained on; a warning is logged if `config.model` says otherwise.

In the example the model's `train_end` is 2023-09-08 and its label looks 5
bars ahead, so the effective training window ends 5 price bars later, on
2023-09-15. The backtest window deliberately starts at 2023-09-04, so the bars
from 2023-09-04 to 2023-09-15 are in-sample and a warning is logged:

```text
  training window     ('2023-01-02', '2023-09-15')
  in-sample range     ('2023-09-04', '2023-09-15')
  out-of-sample       [('2023-09-18', '2024-02-23')]
```

All three blocks come from one continuous simulation; the sub-periods are
never re-simulated, because that would reset the capital and change the path.
`metrics.json` has these top-level keys:

| Key | Content |
|---|---|
| `whole` | vectorbt's portfolio statistics for the whole window, plus the three turnover rows and `Total Orders`. |
| `in_sample`, `out_of_sample` | Return statistics and order, trade and turnover counts restricted to those bars, or `null` when there are none. |
| `training_window`, `in_sample_range`, `out_of_sample_ranges` | The date ranges that define the split. |
| `trained_checkpoint` | Train mode only: the checkpoint the run produced. |
| `notes` | The caveats also shown at the bottom of the report. |

A few metric definitions are worth knowing. Trade statistics use vectorbt's
position view: one trade is one symbol's round trip from entry back to flat,
so trimming a holding back to its target weight is not a closed trade.
`Total Orders` is the number of fills. *Turnover* is the traded value on a
fill bar divided by the portfolio value before it, reported in percent like
every other `[%]` row; a full switch of a long-only book (sell everything,
buy something else) is about 200%, which is why the example's
`Turnover per Rebalance [%]` is around 140.

## Replaying cross-validation with `run_cv()`

*Walk-forward cross-validation* trains a model on a rolling sequence of
windows: fold 0 trains on the first stretch of history and is tested on the
bars right after it, fold 1 shifts forward, and so on (see
[Models](models.md)). `BaseModel.train_cv` saves one checkpoint per fold and a
manifest, `cv_folds.json`, in the trial directory. `run_cv()` reads that
manifest and backtests the whole sequence:

```python
cv_model.collect()
folds = cv_model.train_cv(train_periods=100)
cv_project_dir = Path(folds[0]["checkpoint"]).parent.parent

cv_backtester = USEquityCrossectionSelectStockVectorBt(
    CrossSectionBacktestConfig(
        price_dataset=fresh_dataset(prices),
        model=make_model(root, prices, **dates),
        model_mode="load",
        cv_project_dir=str(cv_project_dir),
        start_date=day(100),
        end_date=day(N_BARS - 1),
        output_dir=str(root / "backtests"),
        rebalance_periods=5,
        direction="long_only",
        top_n=3,
    )
)
cv_result = cv_backtester.run_cv()
```

Each fold whose test segment lies inside the backtest window is backtested on
its own test segment with its own checkpoint. The fold weights are then
concatenated and simulated once as a single *stitched* curve, with capital
carried across fold boundaries, so the stitched curve is a trading path you
could actually have followed with a model retrained every fold. Before
loading any model, `run_cv()` checks on the price calendar that the selected
test segments abut exactly: a gap would leave bars that no model traded, and
an overlap would have two models trading the same bars, so both raise
`ValueError`. `run_cv()` requires `model_mode="load"` and `cv_project_dir`.

```text
== run_cv over 10 folds (stitched)
run directory: USEquityCrossectionSelectStockVectorBt_20260925_175118_514435
  Total Return [%]     15.392
  Sharpe Ratio          1.837
  Max Drawdown [%]      6.116
  Total Orders            209
  turnover/rebal. [%]    150.8
  fold 0: 2023-05-22..2023-06-16 return   3.26%
  fold 1: 2023-06-19..2023-07-14 return  -0.24%
  fold 2: 2023-07-17..2023-08-11 return   1.31%
files: ['config.json', 'equity.zarr', 'fingerprint.json', 'folds', 'liquidations.json', 'metrics.json', 'report.html', 'weights.zarr']
```

The stitched curve is out-of-sample except for the first label-horizon bars
of each fold, which overlap that fold's effective training window when there
is no gap between training and test segments. `metrics.json` holds a
`stitched` block (with `in_sample_ranges` and `out_of_sample_ranges` as lists)
and a `folds` list with every fold's own metrics from its independent
simulation. Each fold's weights and equity are also written under
`folds/fold_{i}/`.

## The run directory

Every `run()` and `run_cv()` creates a new directory
`{output_dir}/{ClassName}_{YYYYmmdd_HHMMSS_ffffff}/`. An existing directory is
never overwritten. The artifacts are written into a hidden staging directory
first and renamed into place only when all of them succeeded, so a crashed
run leaves no half-written directory behind.

| File | Content |
|---|---|
| `config.json` | Every config field, the nested price dataset and model configs, and the data fingerprints. Enough to rebuild the run. |
| `weights.zarr` | The target weights on `(timestamp, symbol)`. |
| `equity.zarr` | Portfolio `value` and per-bar `returns` on `timestamp`. |
| `metrics.json` | The metric blocks described above. |
| `liquidations.json` | One record per forced liquidation. |
| `fingerprint.json` | A content hash and extent of every dataset the run read. |
| `report.html` | The human-readable report. |
| `folds/` | `run_cv()` only: per-fold `weights.zarr` and `equity.zarr`. |

All JSON files are strict JSON: NaN and infinities are written as `null` and
timestamps as ISO strings.

`report.html` is a single page with a summary table of the dates and settings,
an interactive chart of the equity curve, the drawdown (the fall from the
running peak) and the monthly returns on a shared time axis, a year-by-month
heatmap of monthly returns, a metric table with columns for the whole window,
in-sample, out-of-sample and their difference, and the notes. The in-sample
range is shaded on the chart, and two triangles mark the deepest drawdown from
its lowest point to its recovery. The page loads plotly.js from a CDN, so
viewing it needs network access, while the file itself stays small (about
40 KB for the example runs).

Set `use_wandb=True` to also log the metrics and the report to a separate
Weights & Biases run in the project `{ClassName}_backtest`. It is off by
default, so nothing leaves the machine unless you ask for it.

## Rebuilding and re-running a backtest

`quantlab.utils.module.load_backtester_from_config` rebuilds a backtester,
its price dataset and its model from a run's `config.json`. Calling `run()`
or `run_cv()` on the result repeats the backtest:

```python
import json
from quantlab.utils.module import load_backtester_from_config

saved = json.loads((ls_result.run_dir / "config.json").read_text())
rebuilt = load_backtester_from_config(saved)
again = rebuilt.run()
```

```text
rebuilt from config.json, identical equity curve: True
```

Every class is recorded by its dotted import path, so the classes must be
importable when you rebuild. The example works because its classes live in
the running script; in a project, put your factors, labels and model heads in
a module. Every config field must be present in the file; missing fields are
refused rather than filled from today's defaults, because a default that has
changed since the run would silently produce a different backtest.

A rebuilt train-mode config trains the model again. To replay exactly the
model a train-mode run produced, set `model_mode` to `"load"` and
`checkpoint` to the path recorded under `trained_checkpoint` before
rebuilding.

Data can change under a stored backtest: stores get appended to, and adjusted
prices are restated after every split or dividend. The data fingerprints in
`config.json` guard against this. A fingerprint is a SHA-256 hash of the
values the run consumed, together with the first and last timestamp and the
axis sizes; one is recorded for the price data, for each factor's input, and
in train mode for the training data. The rebuilt backtester compares its own
fingerprints against the stored ones and logs a warning such as
`data fingerprint mismatch for 'price_dataset' (differing fields: digest, end)`
for each difference. It does not refuse to run, because a backtest on updated
data is often exactly what you want; the warning makes sure it is not a
surprise.

## Limitations

Benchmark comparison is a single-symbol buy and hold (`benchmark_dataset`,
see `docs/backtest.md`); there is no weighted or multi-asset benchmark. Borrow
costs for short positions are not modelled. The rebalance schedule is anchored
to the first bar of the window, so the whole book turns over on the same day
and results can depend on which day the backtest starts. Portfolio
construction is equal-weight top-N; there is no optimizer, risk model or
neutralization. On intraday data, `run_cv()` compares fold boundaries by
calendar day, so two folds that meet within one day are rejected as
overlapping. An event-driven engine is reserved but not implemented.
