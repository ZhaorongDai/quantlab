"""Orchestration behind ``quantlab.api.analyze_factors``.

The caller's factors become a panel and the forward returns a fret panel, either the
caller's own or computed from prices by ``forward_returns``; both go to the analyzer's
panel path, ``FactorAnalyzer.analyze_panels``, the one ``Factor.analyze`` runs through. The
analyzer is imported only when used, so importing ``quantlab.api`` does not load it.
"""

from numbers import Integral

import numpy as np
import pandas as pd
import polars as pl
import xarray as xr

from quantlab.api import _labels
from quantlab.api._factor_report import FactorReport
from quantlab.utils.frame import (
    INDEX_COLUMNS,
    columns_present,
    library_of,
    to_field_panel,
    to_panel_with_zone,
)

#: The fret name of returns whose value has no name of its own (a wide frame).
UNNAMED_RETURNS = "returns"


def analyze_factors(
    factors, returns, *, prices, price, span, delay, quantiles, plot, columns
) -> FactorReport:
    """Analyze ``factors`` against forward returns; see ``quantlab.api.analyze_factors``."""
    span = _check_arguments(returns, prices, price, span, delay, quantiles, plot)
    library = library_of(factors)
    source = prices if returns is None else returns
    _check_columns_named(columns, factors, source, "returns" if prices is None else "prices")

    panel, zone = to_panel_with_zone(
        factors, columns=columns_present(columns, factors), purpose="factors"
    )
    _check_factor_columns(panel)
    if prices is not None:
        label = _labels.forward_returns(
            prices,
            price=price,
            span=span,
            delay=int(delay),
            binary=False,
            columns=columns_present(columns, prices),
            as_xarray=True,
        )
        fret_name, returns_zone = f"the forward returns of prices (price={price!r})", zone
    else:
        field = to_field_panel(returns, UNNAMED_RETURNS, columns=columns, purpose="returns")
        label = field.values.rename(_returns_name(returns, columns)).to_dataset()
        fret_name, returns_zone = "returns", field.zone
    _check_shared_cells(panel, label, quantiles, zone, returns_zone, fret_name)

    from quantlab.analysis.factor_report import FactorAnalyzer, Fret

    analysis = FactorAnalyzer(quantiles=int(quantiles), plot=plot).analyze_panels(
        panel, [Fret(label, span, fret_name)]
    )
    return FactorReport(analysis, "polars" if library == "polars" else "pandas")


def _check_arguments(returns, prices, price, span, delay, quantiles, plot) -> int:
    """Raise on a missing or doubled source or a bad argument; return the horizon."""
    if (returns is None) == (prices is None):
        raise ValueError(
            "analyze_factors: pass exactly one of returns= and prices=: returns are your "
            "own forward returns, prices are turned into forward returns by "
            "quantlab.api.forward_returns."
        )
    if returns is not None and span is None:
        raise ValueError(
            "analyze_factors: returns= needs span=, the number of bars your returns span; "
            "it sets how returns compound and the IC's Newey-West lags."
        )
    span = 1 if span is None else span
    for name, value, least in (
        ("span", span, 1), ("delay", delay, 0), ("quantiles", quantiles, 2)
    ):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"analyze_factors: {name} must be an int, got {value!r}.")
        if value < least:
            raise ValueError(f"analyze_factors: {name} must be at least {least}, got {value}.")
    if not isinstance(price, str):
        raise TypeError(f"analyze_factors: price must be a column name, got {price!r}.")
    if not isinstance(plot, bool):
        raise TypeError(f"analyze_factors: plot must be True or False, got {plot!r}.")
    if returns is not None:
        misplaced = [
            text
            for text, given in (
                (f"price={price!r}", price != "open"),
                (f"delay={delay!r}", delay != 1),
            )
            if given
        ]
        if misplaced:
            raise ValueError(
                f"analyze_factors: {', '.join(misplaced)} applies only with prices=, to "
                f"compute the forward returns; given returns are used as they are. Drop "
                f"it, or pass prices= instead."
            )
    return int(span)


def _check_columns_named(columns, factors, source, source_name: str) -> None:
    """Raise naming every ``columns`` entry neither the factors nor the source has."""
    if not columns:
        return
    known = set(columns_present(columns, factors)) | set(columns_present(columns, source))
    absent = [name for name in columns if name not in known]
    if absent:
        raise ValueError(
            f"analyze_factors: columns= maps {absent} but neither the factors nor the "
            f"{source_name} have such a column."
        )


def _check_factor_columns(panel: xr.Dataset) -> None:
    """Raise when the factors hold no factor column, or one that is not numeric."""
    if not panel.data_vars:
        raise ValueError(
            "analyze_factors: the factors have no factor column beside timestamp and "
            "symbol."
        )
    other = [
        name for name, var in panel.data_vars.items() if not np.issubdtype(var.dtype, np.number)
    ]
    if other:
        raise ValueError(
            f"analyze_factors: factor column(s) {', '.join(repr(n) for n in other)} are "
            f"not numeric; drop them or convert them to numbers first."
        )


def _returns_name(returns, columns) -> str:
    """Return the fret name of ``returns``: its one value column's or variable's name.

    A long frame's value column and a panel's variable name the fret; a wide frame,
    whose columns are symbols, and an unnamed ``DataArray`` give ``UNNAMED_RETURNS``.
    """
    if isinstance(returns, xr.DataArray):
        return str(returns.name) if returns.name is not None else UNNAMED_RETURNS
    if isinstance(returns, xr.Dataset):
        return str(next(iter(returns.data_vars)))
    if isinstance(returns, pl.LazyFrame):
        names = returns.collect_schema().names()
    elif isinstance(returns, pl.DataFrame):
        names = list(returns.columns)
    else:
        names = list(returns.columns)
        if isinstance(returns.index, pd.MultiIndex):
            names = [name for name in returns.index.names if name is not None] + names
    renames = columns_present(columns, returns)
    names = [renames.get(name, name) for name in names]
    if "symbol" not in names:
        return UNNAMED_RETURNS
    return str(next(name for name in names if name not in INDEX_COLUMNS))


def _check_shared_cells(
    panel: xr.Dataset, label: xr.Dataset, quantiles, zone, returns_zone, fret_name: str
) -> None:
    """Raise when factors and returns share no cell, or too few symbols for ``quantiles``.

    The largest cross-section is the most symbols on one bar with a finite return and at
    least one finite factor value; with fewer than ``quantiles`` of them no bar can be
    bucketed.
    """
    factors, returns = xr.align(panel, label, join="inner")
    if factors.sizes["timestamp"] == 0 or factors.sizes["symbol"] == 0:
        raise ValueError(
            f"analyze_factors: the factors and {fret_name} share no (timestamp, symbol) "
            f"cells.{_zone_hint(zone, returns_zone)}"
        )
    has_factor = np.zeros((factors.sizes["timestamp"], factors.sizes["symbol"]), dtype=bool)
    for var in factors.data_vars.values():
        has_factor |= np.isfinite(var.transpose(*INDEX_COLUMNS).values.astype(float))
    has_return = np.zeros_like(has_factor)
    for var in returns.data_vars.values():
        has_return |= np.isfinite(var.transpose(*INDEX_COLUMNS).values.astype(float))
    largest = int((has_factor & has_return).sum(axis=1).max())
    if quantiles > largest:
        raise ValueError(
            f"analyze_factors: quantiles={quantiles} is more than the largest cross-section "
            f"of the factors and {fret_name}, {largest} symbol(s) on one bar with both a "
            f"factor value and a return; pass quantiles={largest} or fewer."
        )


def _zone_hint(factors_zone: str | None, returns_zone: str | None) -> str:
    """Return a sentence naming both time zones when they differ, else ``""``.

    Naive timestamps are taken as UTC, so naive and UTC count as the same zone.
    """
    if (factors_zone or "UTC") == (returns_zone or "UTC"):
        return ""

    def said(who: str, name: str | None) -> str:
        return f"the {who} are naive, taken as UTC" if name is None else f"the {who} were {name}"

    return (
        f" The bars may differ only by time zone: {said('factors', factors_zone)}; "
        f"{said('returns', returns_zone)}."
    )
