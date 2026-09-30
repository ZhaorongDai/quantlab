# Portfolio construction

English | [简体中文](zh-CN/portfolio.md)

Portfolio construction is the step between a model and a backtest. On every rebalance bar it takes the model's predictions for that bar and the weights currently held, and returns the weights to hold after the bar. The backtest then trades those weights.

Every rule derives from `PortfolioConstructor` (`quantlab/base/portfolio.py`). Two rules ship in `quantlab/portfolio/predefined/`:

- `TopNConstructor`: equal weights on the `top_n` best scores, long-only or long-short.
- `MeanVarianceOptimizer`: Markowitz weights with a turnover penalty, solved with cvxpy. It prices risk with a risk model. The shipped risk model is `LedoitWolfRiskModel`, a Ledoit-Wolf shrunk covariance of trailing returns.

A backtest holds its rule in the `constructor` field of its config (see [Backtesting](backtest.md)).

## Prerequisites

Run the sessions from the repository root with `uv run python`. They build each bar's input by hand, so they need no store, model or GPU. In a backtest, the backtester builds the same input from the price dataset and the model.

## The basics

### One bar at a time

A rule has one method to implement, `construct(context)`. The `PortfolioContext` it receives describes one bar and holds nothing from a later bar:

| Field | Content |
| --- | --- |
| `timestamp` | The bar. The weights returned fill at the next bar. |
| `predictions` | Every label's prediction at the bar, one variable per label, on `symbol`. |
| `tradable` | Whether each symbol can be traded at the bar: by default, whether it has a real fill price there. |
| `current_weights` | The weights held now, valued at the bar. They are all 0.0 before the first rebalance. |
| `returns` | The last `lookback_bars` one-bar returns, ending at the bar. It is empty for a rule with `lookback_bars` of 0. |
| `staleness` | Bars since each symbol's last real price. |
| `factors` | The values at the bar of the factors the rule declares in `required_factors()`, or `None`. |

A rule returns one weight per symbol:

- The weights are finite, 0.0 where nothing is held, with a gross exposure of at most one.
- An all-NaN row means "hold the current position".
- A symbol that is held but not tradable is a *locked position*; `context.locked` marks them. It must keep its current weight. A symbol that is neither tradable nor held gets 0.0.

A rule keeps no state between bars. Because it only ever sees one bar, it cannot look ahead. It can also be called from the bar handler of an event-driven engine.

```python
>>> import numpy as np, pandas as pd, xarray as xr
>>> from quantlab.base.config import TopNConfig
>>> from quantlab.base.portfolio import PortfolioContext
>>> from quantlab.portfolio.predefined.top_n import TopNConstructor
>>> symbols = ["AAA", "BBB", "CCC", "DDD"]
>>> def on_symbols(values):
...     return xr.DataArray(values, dims="symbol", coords={"symbol": symbols})
>>> context = PortfolioContext(
...     timestamp=pd.Timestamp("2024-03-01"),
...     predictions=xr.Dataset({"ret_5": on_symbols([0.8, -0.1, 0.5, 0.2])}),
...     tradable=on_symbols([True, True, True, False]),
...     current_weights=on_symbols([0.0, 0.0, 0.0, 0.25]),
... )
>>> context.locked.values
array([False, False, False,  True])
>>> TopNConstructor(TopNConfig(direction="long_only", top_n=2)).construct(context).values
array([0.375, 0.   , 0.375, 0.25 ])

```

`DDD` is halted while held, so it keeps its 0.25. The two best tradable scores share the remaining 0.75.

Equal scores are ranked by symbol order. When a book's cut falls inside a group of equal scores, the symbol order, not the model, decides which of them are held, and the row counts the tied symbols left out as a `tie_at_cutoff` event. Below, `BBB`, `CCC` and `DDD` score the same, `BBB` comes first, and two are left out. A model that predicts only a handful of distinct values makes this event appear on most rebalances.

```python
>>> tied = PortfolioContext(
...     timestamp=pd.Timestamp("2024-03-01"),
...     predictions=xr.Dataset({"ret_5": on_symbols([0.8, 0.5, 0.5, 0.5])}),
...     tradable=on_symbols([True, True, True, True]),
...     current_weights=on_symbols([0.0, 0.0, 0.0, 0.0]),
... )
>>> weights = TopNConstructor(TopNConfig(direction="long_only", top_n=2)).construct(tied)
>>> weights.values
array([0.5, 0.5, 0. , 0. ])
>>> weights.attrs
{'events': {'tie_at_cutoff': 2}}

```

### In a backtest

The vectorised backtest calls `construct_panel`. It loops `construct` over the rebalance bars and builds each bar's context:

- The current weights are the holdings the earlier rebalances really left. They are modelled the way the simulation trades them, including rejected orders and delisting settlements.
- The return window comes from the last known price of each symbol, so a halt shows as zero returns and then the whole move on the day trading resumes.
- A rule's `lookback_bars` is added to the backtest's warm-up, so the first backtest bar already has a full window.

The backtester calls the rule's `bind(predictor)` when it is built, before any data is read or any model trained. This is where a rule checks the labels it needs, so a misconfigured rule fails at once.

A bar the rule cannot decide raises `PortfolioConstructionError`, for example when an optimisation is infeasible or the solver fails. The backtest holds the current position on that bar and logs a warning. `metrics.json` lists every such bar under `portfolio_construction` (`failed_bar_count`, `failed_bars`), along with any event a rule reported, such as `tie_at_cutoff` above or `closed_without_risk` below, with its `count` (the symbols it involved over all its bars) and one record per bar.

The run's `config.json` records the rule with all its parameters and its risk model, and `load_backtester_from_config` rebuilds it.

## Mean-variance optimisation

### The problem

On each rebalance bar, `MeanVarianceOptimizer` solves

    maximise    w'mu - risk_aversion / 2 * w'Sigma w - turnover_penalty * |w - w_current|_1

- `mu` is the expected return of each symbol.
- `Sigma` is their covariance.
- `w_current` is the context's current weights.

The constraints depend on `direction`:

| `direction` | Constraints |
| --- | --- |
| `"long_only"` | `sum(w) = 1`, `0 <= w <= weight_cap`: fully invested. |
| `"long_short"` | `sum(w) = 0`, `sum(abs(w)) <= 1`, `abs(w) <= weight_cap`: dollar-neutral. |

In a long-short book the gross exposure of one is a ceiling, not a target. When the expected returns do not pay for the risk and the trading, part of the book stays uninvested, down to no position at all.

The turnover penalty is charged on the change from the weights actually held, so it prices the trades that will really happen. A locked position keeps its weight: it counts in the risk term, and the other symbols share what is left of the budget.

The candidates are the symbols that pass all of these checks:

- They are tradable.
- They are not locked.
- The risk model covers them.
- They have a finite expected-return prediction or are held.

A held candidate without a prediction gets an expected return of 0.0, so its turnover cost decides whether it is closed. Every other symbol gets 0.0.

### A first optimisation

The optimiser needs to know about the predictor's labels, which a backtest hands it through `bind`. Here a small stand-in plays the predictor: it predicts a 5-bar return `ret_5` and a 5-bar volatility `vol_5`, both in the labels' own units (`"raw"`, see [Calibration](#calibration)).

```python
>>> from quantlab.base.config import LedoitWolfConfig, MeanVarianceConfig
>>> from quantlab.portfolio.predefined.ledoit_wolf import LedoitWolfRiskModel
>>> from quantlab.portfolio.predefined.mean_variance import MeanVarianceOptimizer
>>> class Label:
...     def __init__(self, name, span):
...         self.name, self.span = name, span
...     def get_factor_names(self):
...         return (self.name,)
...     def span_bars(self):
...         return self.span
>>> class Predictor:
...     labels = [Label("ret_5", 5), Label("vol_5", 5)]
...     label_scales = {"ret_5": "raw", "vol_5": "raw"}
>>> rng = np.random.default_rng(0)
>>> window = rng.normal(0.0, 1.0, size=(60, 4)) * [0.010, 0.015, 0.020, 0.025]
>>> context = PortfolioContext(
...     timestamp=pd.Timestamp("2024-03-26"),
...     predictions=xr.Dataset({
...         "ret_5": on_symbols([0.8, -0.1, -0.3, 0.2]),
...         "vol_5": on_symbols([0.05, 0.03, 0.04, 0.06]),
...     }),
...     tradable=on_symbols([True, True, True, True]),
...     current_weights=on_symbols([0.0, 0.0, 0.0, 0.0]),
...     returns=xr.DataArray(window, dims=("timestamp", "symbol"), coords={
...         "timestamp": pd.bdate_range("2023-12-29", periods=60), "symbol": symbols}),
... )
>>> optimizer = MeanVarianceOptimizer(MeanVarianceConfig(
...     expected_return_label="ret_5",
...     risk_model=LedoitWolfRiskModel(LedoitWolfConfig(lookback_bars=60)),
...     ic=0.05, risk_aversion=5.0, weight_cap=0.4,
... ))
>>> optimizer.bind(Predictor())
>>> optimizer.lookback_bars, optimizer.span
(60, 5)
>>> weights = optimizer.construct(context)
>>> weights.values.round(3)
array([0.4  , 0.4  , 0.009, 0.191])

```

The book is fully invested and within the cap. With so small an expected return it leans toward the low-volatility symbols `AAA` and `BBB`. `optimizer.problem_inputs(context)` returns the candidates with their `mu`, `Sigma` and current weights, which is useful to see why the optimiser chose what it did.

### Calibration

A model's prediction is usually a score, not a return. It ranks symbols, but its scale depends on the model and on how its training target was transformed. `calibration` says how the prediction becomes the expected return `mu`:

- `"grinold"` (the default) sets `mu = ic * sigma * z`. Here `z` is the prediction's cross-sectional z-score over the candidates, `sigma` each symbol's volatility over the span (from the covariance's diagonal), and `ic` the model's information coefficient, for example the mean IC of a walk-forward cross-validation run. Any model's output can feed it, and `risk_aversion` keeps its meaning from one model to the next.
- `"raw"` takes the prediction itself as `mu`. It only makes sense for a model that predicts returns in their own units.

Every predictor reports, per label, whether its prediction is in the label's own units: `label_scales` maps each label to `"raw"` or `"standardized"`. A model is `"raw"` when it was fitted on the label unchanged. An ensemble reports `"standardized"` for a label it averages over several members, because it z-scores each member first. `bind` refuses `"raw"` calibration on a `"standardized"` label:

```python
>>> class RankedPredictor(Predictor):
...     label_scales = {"ret_5": "standardized", "vol_5": "raw"}
>>> MeanVarianceOptimizer(MeanVarianceConfig(
...     expected_return_label="ret_5", calibration="raw",
...     risk_model=LedoitWolfRiskModel(LedoitWolfConfig(lookback_bars=60)),
...     risk_aversion=5.0,
... )).bind(RankedPredictor())
Traceback (most recent call last):
    ...
ValueError: calibration='raw' reads the prediction of 'ret_5' as a return, but the predictor reports its scale as 'standardized', not 'raw' (a model fitted on a transformed target, or a label an ensemble averages); use calibration='grinold'

```

### Spans

`mu`, `sigma` and `Sigma` are all expressed over the span of `expected_return_label`: 5 bars for `ret_5`. The span is read from the label, never configured. A risk model estimates the covariance of one-bar returns, and the optimiser multiplies it by the span, because variance grows linearly with time:

```python
>>> inputs = optimizer.problem_inputs(context)
>>> one_bar = LedoitWolfRiskModel(LedoitWolfConfig(lookback_bars=60)).estimate(context)
>>> bool(np.allclose(inputs.covariance, 5 * one_bar.covariance))
True

```

Rebalancing every span (`rebalance_periods` equal to the span) makes the horizon the optimiser plans over the one the portfolio is held for.

### Volatility from a model

Volatility moves fast and is fairly predictable. Correlations move slowly and are better estimated from history. With `volatility_label` set, the optimiser takes each symbol's volatility from a model's prediction:

- The covariance becomes the predicted volatilities around the risk model's historical correlations: `Sigma = D C D`.
- The Grinold `sigma` becomes the prediction itself.

The volatility label, such as `Volatility`, is a span-scale volatility. It must have the span of the expected-return label and a `"raw"` scale, and `bind` checks both.

```python
>>> with_volatility = MeanVarianceOptimizer(MeanVarianceConfig(
...     expected_return_label="ret_5", volatility_label="vol_5",
...     risk_model=LedoitWolfRiskModel(LedoitWolfConfig(lookback_bars=60)),
...     ic=0.05, risk_aversion=5.0, weight_cap=0.4,
... ))
>>> with_volatility.bind(Predictor())
>>> covariance = with_volatility.problem_inputs(context).covariance
>>> np.sqrt(np.diag(covariance)).round(4)
array([0.05, 0.03, 0.04, 0.06])
>>> with_volatility.construct(context).values.round(3)
array([0.4  , 0.399, 0.   , 0.201])

```

A symbol without a finite, positive volatility prediction has no risk estimate. It is treated like a symbol without enough history, as described under [Risk models](#risk-models).

One predictor has to supply both labels. A `ModelEnsemble` of a return model and a volatility model does this: a label that only one member predicts is passed through unchanged and keeps that member's scale (see [Models](model.md)).

### Long-short and the candidate pool

With `direction="long_short"` the book is dollar-neutral:

```python
>>> long_short = MeanVarianceOptimizer(MeanVarianceConfig(
...     expected_return_label="ret_5",
...     risk_model=LedoitWolfRiskModel(LedoitWolfConfig(lookback_bars=60)),
...     ic=0.05, risk_aversion=5.0, weight_cap=0.4, direction="long_short",
... ))
>>> long_short.bind(Predictor())
>>> weights = long_short.construct(context)
>>> weights.values.round(3)
array([ 0.4  , -0.212, -0.261,  0.073])
>>> bool(abs(weights.sum()) < 1e-12), float(abs(weights).sum().round(3))
(True, 0.945)

```

A universe of thousands of symbols is slow to optimise at every rebalance. `candidate_top_k` restricts each bar to a pool:

- the `candidate_top_k` symbols with the largest `mu` (largest `abs(mu)` long-short);
- plus every symbol currently held, so that a holding that fell out of the top can still be closed at its turnover cost.

Symbols outside the pool get 0.0. The z-score of the Grinold calibration is taken over every candidate before the pool is cut, so `mu` does not depend on `candidate_top_k`.

## Risk models

A risk model estimates the covariance of one-bar returns at one bar. Its `estimate(context, volatility=None)` returns a `CovarianceEstimate`: the symbols it covers and their covariance. A risk model is an estimator, run afresh at every bar from what the context holds. It is not trained and has no checkpoint. A forecast that feeds it, such as predicted volatility, comes from a model through the predictor.

`LedoitWolfRiskModel` reads the last `lookback_bars` one-bar returns and shrinks their sample covariance toward a scaled identity with the Ledoit-Wolf coefficient (`sklearn.covariance.ledoit_wolf`), so the estimate stays well conditioned when there are more symbols than bars. It then splits the covariance into correlations and volatilities, and replaces the volatilities with the given ones when `volatility` is passed.

A symbol is covered only when all of these hold:

- Every return in its window is finite.
- The returns are not all equal.
- Its staleness is at most `max_stale_bars` (default 5).

The other symbols are left out, and the optimiser never selects them. A held symbol that the risk model does not cover is closed, and the row reports it as a `closed_without_risk` event:

```python
>>> short_history = window.copy()
>>> short_history[:5, 2] = np.nan  # CCC listed five bars into the window
>>> import dataclasses
>>> late = dataclasses.replace(
...     context,
...     current_weights=on_symbols([0.0, 0.0, 0.3, 0.0]),
...     returns=context.returns.copy(data=short_history),
... )
>>> weights = optimizer.construct(late)
>>> weights.values.round(3)
array([0.4, 0.4, 0. , 0.2])
>>> weights.attrs["events"]
{'closed_without_risk': ['CCC']}

```

### Reserved: factor risk models

The interface leaves room for a factor risk model, which is not implemented yet:

- A risk model can return a `FactorCovarianceEstimate`, the covariance in factor form `B F B' + diag(D)`: exposures `B`, factor covariance `F` and specific variances `D`. Its `factor_form()` makes the optimiser price risk as `|F^(1/2) B' w|^2 + w' diag(D) w`, without building the dense matrix.
- A risk model can declare in `required_factors()` the `Factor` panels it reads, for example its exposures. The backtest computes them over its window, with each factor's own warm-up, and puts their values at the bar in `context.factors`.

The optimiser, the driver and the backtester need no change for such a model.

## A full example

`examples/wrds_us_equity/sp500_xgb_mvo.py` is an enhanced S&P 500 index: it runs the whole pipeline on the point-in-time index members.

1. Alpha101 and Alpha158 factors.
2. A `Return` and a `Volatility` label, both over 5 bars.
3. A `ModelEnsemble` of two `XGBoostRegressor` heads, one per label.
4. The mean-variance optimiser over the day's members, with `volatility_label` set and a pool of 200 candidates, rebalancing every 5 bars against buy-and-hold SPY. A stock removed from the index is settled at its last close, like a delisting.

It needs the S&P 500 and SPY stores of `scripts/wrds/index.py --index sp500` and `scripts/wrds/etf.py --etf spy` (see the [examples README](../examples/wrds_us_equity/README.md)).

A run on the training server (one GPU; commit `211b324`, 2026-09-30; about 8 minutes once the stores exist) gave, over the out-of-sample window 2020-2024:

| Model | Test IC | Test rank IC |
| --- | --- | --- |
| Return model (`ret_5`) | -0.0005 | -0.0004 |
| Volatility model (`vol_5`) | 0.513 | 0.481 |

| Backtest | Strategy | SPY |
| --- | --- | --- |
| Total return | 41.6% | 96.8% |
| Sharpe ratio | 0.51 | |
| Maximum drawdown | 32.3% | |
| Annualised excess return | -6.4% | |
| Tracking error | 10.1% | |
| Information ratio | -0.75 | |
| Beta | 0.68 | |

No rebalance failed and no order was rejected. On 10 rebalances, 11 holdings in all were closed because the risk model had no estimate for them (`closed_without_risk`).

Read these numbers as a working pipeline, not as a strategy. The return model has no out-of-sample skill: its test IC is zero. With no signal in `mu`, the risk term decides the weights, and the book becomes a low-volatility portfolio with a beta of 0.68, which trails the index in a rising market. The volatility model does carry information: its IC is the correlation of predicted with realised volatility, which measures how well it ranks risk, not alpha. The optimiser also constrains neither the tracking error nor the active weights against the index, so this is an index-member portfolio, not yet a controlled enhanced index.

## Writing a rule

Subclass `PortfolioConstructor`:

1. Set `config_cls` to a frozen dataclass of the rule's parameters.
2. Implement `construct`.
3. Override `lookback_bars` when the rule reads a return window, `required_factors` when it reads factor panels, and `bind` to check the predictor's labels.

`get_config` and `from_config` serialise the rule as its config's fields plus the class's import path. A field holding another component, such as a risk model, is nested, so no extra code is needed to rebuild a run.

A new risk model subclasses `RiskModel` in the same way and implements `estimate`. See [Extending quantlab](developer-guide/extending.md).

## See also

- [Backtesting](backtest.md): the `constructor` field, execution and the run directory.
- [Models](model.md): `label_scales` and ensembles that combine a return model and a volatility model.
- [Factors](factor.md): the `Return` and `Volatility` labels.
