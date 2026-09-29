"""Orchestration behind ``quantlab.api.backtest``.

The caller's prices (and benchmark) become ``FrameDataset``s, the weights or scores
become a weight panel on exactly the price axes, and ``WeightsVectorBt.run_weights``
simulates it with the market conventions the caller chose. The backtest layer is
imported inside the function, so importing ``quantlab.api`` does not load vectorbt or
plotly.
"""

import numpy as np
import pandas as pd
import xarray as xr

from quantlab.api._report import BacktestReport
from quantlab.dataset.memory import FrameDataset
from quantlab.utils.frame import library_of, to_field_panel, to_panel

#: ``market`` name to ``(trading_days_per_year, session_minutes_per_day)``.
MARKETS = {"equity": (252, 390), "crypto": (365, 1440)}

#: Offending labels named in an error message.
LABELS_SHOWN = 5


def backtest(
    prices,
    *,
    weights,
    scores,
    top_n,
    direction,
    rebalance_periods,
    fill,
    valuation,
    market,
    trading_days_per_year,
    session_minutes_per_day,
    fees,
    slippage,
    init_cash,
    benchmark,
    output_dir,
    columns,
) -> BacktestReport:
    """Backtest weights or top-N scores on ``prices``; see ``quantlab.api.backtest``."""
    from quantlab.backtest.predefined.weights import WeightsVectorBt
    from quantlab.base.config import WeightsBacktestConfig

    library = library_of(prices)
    _check_signal(weights, scores, top_n, direction)
    days, minutes = _annualization(market, trading_days_per_year, session_minutes_per_day)

    panel = to_panel(prices, columns=columns, required=(fill, valuation), purpose="prices")
    if panel.sizes["timestamp"] < 2:
        raise ValueError(
            f"prices hold {panel.sizes['timestamp']} bar(s); a backtest needs at least "
            f"two, since a weight at one bar fills at the next."
        )
    benchmark_dataset = None
    if benchmark is not None:
        benchmark_panel = to_panel(
            benchmark,
            columns=_keys_present(columns, benchmark),
            required=(fill, valuation),
            purpose="benchmark",
        )
        benchmark_dataset = FrameDataset(benchmark_panel)

    if weights is not None:
        weight = _weights_on(panel, weights, columns)
    else:
        weight = _selected(panel, scores, columns, fill, top_n, direction, rebalance_periods)

    timestamps = pd.DatetimeIndex(panel["timestamp"].values)
    config = WeightsBacktestConfig(
        price_dataset=FrameDataset(panel),
        start_date=timestamps[0].strftime("%Y-%m-%d"),
        end_date=timestamps[-1].strftime("%Y-%m-%d"),
        output_dir=None if output_dir is None else str(output_dir),
        rebalance_periods=rebalance_periods,
        fees=fees,
        slippage=slippage,
        init_cash=init_cash,
        benchmark_dataset=benchmark_dataset,
        use_wandb=False,
        fill_price_column=fill,
        valuation_price_column=valuation,
        trading_days_per_year=days,
        session_minutes_per_day=minutes,
        direction=None if scores is None else direction,
        top_n=top_n,
    )
    backtester = WeightsVectorBt(config)
    result = backtester.run_weights(weight)
    return BacktestReport(result, backtester, library)


def _check_signal(weights, scores, top_n, direction) -> None:
    """Raise unless exactly one signal is given with the arguments that belong to it."""
    if (weights is None) == (scores is None):
        raise ValueError(
            "Pass exactly one of weights= and scores=: weights are backtested as given, "
            "scores are turned into weights by a top-N selection."
        )
    if scores is not None:
        if top_n is None:
            raise ValueError("scores= needs top_n=, the number of names held per side.")
        return
    misplaced = [
        text
        for text, given in (
            (f"top_n={top_n!r}", top_n is not None),
            (f"direction={direction!r}", direction != "long_only"),
        )
        if given
    ]
    if misplaced:
        raise ValueError(
            f"{', '.join(misplaced)} applies only with scores=, to select names; "
            f"given weights are traded as they are. Drop it, or pass scores= instead."
        )


def _annualization(market, trading_days_per_year, session_minutes_per_day) -> tuple[int, int]:
    """Return ``(days, minutes)`` of ``market``, each overridable."""
    if market not in MARKETS:
        raise ValueError(
            f"Unknown market {market!r}. Valid markets: "
            f"{', '.join(repr(name) for name in MARKETS)}; override the annualization "
            f"with trading_days_per_year= and session_minutes_per_day=."
        )
    days, minutes = MARKETS[market]
    return (
        days if trading_days_per_year is None else trading_days_per_year,
        minutes if session_minutes_per_day is None else session_minutes_per_day,
    )


def _keys_present(columns, frame) -> dict | None:
    """Return the entries of ``columns`` naming a column (or index level) ``frame`` has."""
    library = library_of(frame)
    if not columns:
        return None
    if library == "pandas":
        names = set(frame.columns) | {name for name in frame.index.names if name}
    elif library == "xarray":
        names = set(frame.dims) | set(frame.data_vars)
    else:
        names = set(frame.collect_schema().names())
    return {name: target for name, target in columns.items() if name in names}


def _on_price_axes(values: xr.DataArray, panel: xr.Dataset, what: str) -> None:
    """Raise naming the first bars or symbols of ``values`` the prices do not have."""
    for dim, noun in (("timestamp", "bar"), ("symbol", "symbol")):
        extra = np.setdiff1d(values[dim].values, panel[dim].values)
        if extra.size:
            shown = ", ".join(
                repr(pd.Timestamp(v).isoformat() if dim == "timestamp" else str(v))
                for v in extra[:LABELS_SHOWN]
            )
            raise ValueError(
                f"The {what} name {extra.size} {noun}(s) the prices do not have, for "
                f"example {shown}. Every {noun} of the {what} must be one of the "
                f"prices'."
            )


def _weights_on(panel: xr.Dataset, weights, columns) -> xr.DataArray:
    """Return the caller's weights as a weight panel on exactly the price axes.

    On a bar the weights name, a price symbol they leave out gets 0; a bar they do not
    name at all is a hold (all NaN). A NaN the caller wrote stays NaN, so a row mixing
    it with finite weights is refused by the backtester, naming the bar.
    """
    values, given = to_field_panel(weights, "weight", columns=columns, purpose="weights")
    _on_price_axes(values, panel, "weights")
    axes = {"timestamp": panel["timestamp"].values, "symbol": panel["symbol"].values}
    values = values.reindex(axes)
    given = given.reindex(axes, fill_value=False)
    bar_given = given.any("symbol")
    return values.where(given, xr.where(bar_given, 0.0, np.nan))


def _selected(panel, scores, columns, fill, top_n, direction, rebalance_periods):
    """Return top-N weights selected from ``scores`` on the price axes.

    A symbol without a score on a bar is not eligible there, and neither is one without
    a fill price on the next bar, exactly as in the library's selection backtester.
    """
    from quantlab.backtest.selection import CrossSectionTopNSelector, rebalance_mask

    values, _ = to_field_panel(scores, "score", columns=columns, purpose="scores")
    _on_price_axes(values, panel, "scores")
    values = values.reindex(
        timestamp=panel["timestamp"].values, symbol=panel["symbol"].values
    )
    selector = CrossSectionTopNSelector(direction=direction, top_n=top_n)
    next_fill = panel[fill].shift(timestamp=-1)
    mask = rebalance_mask(panel.sizes["timestamp"], rebalance_periods)
    return selector.select(values, next_fill, mask)["weight"]
