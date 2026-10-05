"""Orchestration behind ``quantlab.api.backtest``.

The caller's prices (and benchmark) become ``FrameDataset``s, the weights or scores
become a weight panel on exactly the price axes, and ``WeightsVectorBt.run_weights``
simulates it with the market conventions the caller chose. The backtest layer is
imported inside the functions, so importing ``quantlab.api`` does not load vectorbt or
plotly.
"""

import numpy as np
import pandas as pd
import xarray as xr

from quantlab.api._report import BacktestReport
from quantlab.dataset.memory import FrameDataset
from quantlab.utils.frame import (
    columns_present,
    library_of,
    to_field_panel,
    to_panel_with_zone,
)

#: The ``market`` names: ``equity`` reads ``US_EQUITY_MARKET``'s annualization.
MARKET_NAMES = ("equity", "crypto")

#: Crypto trades every day around the clock. The library has no crypto backtester and
#: so no crypto ``MarketSpec`` to read these from; a spec here would also have to name
#: price columns, which the caller chooses with ``fill`` and ``valuation``.
CRYPTO_TRADING_DAYS_PER_YEAR = 365
CRYPTO_SESSION_MINUTES_PER_DAY = 1440

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

    panel, zone = to_panel_with_zone(
        prices, columns=columns, required=(fill, valuation), purpose="prices"
    )
    if panel.sizes["timestamp"] < 2:
        raise ValueError(
            f"prices hold {panel.sizes['timestamp']} bar(s); a backtest needs at least "
            f"two, since a weight at one bar fills at the next."
        )
    benchmark_dataset = None
    if benchmark is not None:
        benchmark_dataset = FrameDataset(
            _benchmark_panel(panel, zone, benchmark, columns, fill, valuation)
        )

    price_dataset = FrameDataset(panel)
    timestamps = pd.DatetimeIndex(panel["timestamp"].values)
    config = WeightsBacktestConfig(
        price_dataset=price_dataset,
        start_date=timestamps[0].strftime("%Y-%m-%d"),
        end_date=timestamps[-1].strftime("%Y-%m-%d"),
        output_dir=None if output_dir is None else str(output_dir),
        rebalance_periods=rebalance_periods,
        fees=fees,
        slippage=slippage,
        init_cash=init_cash,
        benchmark_dataset=benchmark_dataset,
        fill_price_column=fill,
        valuation_price_column=valuation,
        trading_days_per_year=days,
        session_minutes_per_day=minutes,
        direction=None if scores is None else direction,
        top_n=top_n,
    )
    if weights is not None:
        weight = _weights_on(panel, zone, weights, columns)
    else:
        weight = _selected(panel, zone, scores, columns, config)
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
    if market not in MARKET_NAMES:
        raise ValueError(
            f"Unknown market {market!r}. Valid markets: "
            f"{', '.join(repr(name) for name in MARKET_NAMES)}; override the "
            f"annualization with trading_days_per_year= and session_minutes_per_day=."
        )
    if market == "equity":
        from quantlab.backtest.predefined.us_equity import US_EQUITY_MARKET

        days = US_EQUITY_MARKET.trading_days_per_year
        minutes = US_EQUITY_MARKET.session_minutes_per_day
    else:
        days, minutes = CRYPTO_TRADING_DAYS_PER_YEAR, CRYPTO_SESSION_MINUTES_PER_DAY
    return (
        days if trading_days_per_year is None else trading_days_per_year,
        minutes if session_minutes_per_day is None else session_minutes_per_day,
    )


def _zone_hint(prices_zone: str | None, what: str, zone: str | None) -> str:
    """Return a sentence naming both time zones when they differ, else ``""``.

    Naive timestamps are taken as UTC, so naive and UTC count as the same zone.
    """

    def _said(who: str, name: str | None) -> str:
        return f"{who} are naive, taken as UTC" if name is None else f"{who} were {name}"

    if (prices_zone or "UTC") == (zone or "UTC"):
        return ""
    return (
        f" The bars may differ only by time zone: {_said('prices', prices_zone)}; "
        f"{_said(what, zone)}."
    )


def _onto_price_axes(
    panel: xr.Dataset, prices_zone, what: str, field
) -> tuple[xr.DataArray, xr.DataArray]:
    """Return ``field``'s values and given-mask reindexed onto the price axes.

    Raises
    ------
    ValueError
        Naming the first bars or symbols ``field`` has and the prices lack, with a
        time-zone hint when the two inputs came in different zones.
    """
    for dim, noun in (("timestamp", "bar"), ("symbol", "symbol")):
        extra = np.setdiff1d(field.values[dim].values, panel[dim].values)
        if extra.size:
            shown = ", ".join(
                repr(pd.Timestamp(v).isoformat() if dim == "timestamp" else str(v))
                for v in extra[:LABELS_SHOWN]
            )
            hint = _zone_hint(prices_zone, what, field.zone) if dim == "timestamp" else ""
            raise ValueError(
                f"The {what} name {extra.size} {noun}(s) the prices do not have, for "
                f"example {shown}. Every {noun} of the {what} must be one of the "
                f"prices'.{hint}"
            )
    axes = {"timestamp": panel["timestamp"].values, "symbol": panel["symbol"].values}
    return field.values.reindex(axes), field.given.reindex(axes, fill_value=False)


def _benchmark_panel(panel, prices_zone, benchmark, columns, fill, valuation):
    """Return the benchmark panel, refusing one whose bars miss prices' in another zone.

    A benchmark in the same zone that lacks some price bars is left to the backtester,
    which carries its previous price forward on them.
    """
    benchmark_panel, zone = to_panel_with_zone(
        benchmark,
        columns=columns_present(columns, benchmark),
        required=(fill, valuation),
        purpose="benchmark",
    )
    hint = _zone_hint(prices_zone, "benchmark", zone)
    missing = np.setdiff1d(panel["timestamp"].values, benchmark_panel["timestamp"].values)
    if hint and missing.size:
        shown = ", ".join(repr(pd.Timestamp(v).isoformat()) for v in missing[:LABELS_SHOWN])
        raise ValueError(
            f"The benchmark has no bar at {missing.size} price bar(s), for example "
            f"{shown}.{hint}"
        )
    return benchmark_panel


def _weights_on(panel: xr.Dataset, prices_zone, weights, columns) -> xr.DataArray:
    """Return the caller's weights as a weight panel on exactly the price axes.

    On a bar that has a weight, a price symbol without one (left out of a long frame,
    NaN in a wide one) gets 0; a bar without any weight is a hold (all NaN). A NaN
    written in a long frame or a panel stays NaN, which keeps that symbol's holding
    on that bar.
    """
    field = to_field_panel(weights, "weight", columns=columns, purpose="weights")
    values, given = _onto_price_axes(panel, prices_zone, "weights", field)
    bar_given = given.any("symbol")
    return values.where(given, xr.where(bar_given, 0.0, np.nan))


def _selected(panel, prices_zone, scores, columns, config):
    """Return top-N weights selected from ``scores`` on the price axes.

    Exactly as in the library's backtester, through ``DecisionInputs``: the top-N rule
    is bound to the scores as one standardized label, a symbol is selectable where it
    has a score and is tradable (a fill price at that bar), and a held symbol that is
    not tradable keeps its current weight, the holdings replayed from the prices with
    ``config``'s sizing basis, fees and slippage on its rebalance schedule.
    """
    from quantlab.base.config import TopNConfig
    from quantlab.runs.prediction_panel import LabelSpec
    from quantlab.portfolio.decision_inputs import DecisionInputs
    from quantlab.portfolio.predefined.top_n import TopNConstructor

    field = to_field_panel(scores, "score", columns=columns, purpose="scores")
    values, _ = _onto_price_axes(panel, prices_zone, "scores", field)
    constructor = TopNConstructor(TopNConfig(direction=config.direction, top_n=config.top_n))
    constructor.bind([LabelSpec(name="score", scale="standardized", delay=1, span=None)])
    return DecisionInputs(
        config.price_dataset,
        constructor,
        fill_column=config.fill_price_column,
        valuation_column=config.valuation_price_column,
        rebalance_periods=config.rebalance_periods,
        anchor=panel["timestamp"].values[0],
        execution=config.execution,
    ).weights(values.to_dataset(name="score"))["weight"]
