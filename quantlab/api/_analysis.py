"""Orchestration behind ``quantlab.api.analyze_factors``.

The caller's factors become a panel and the forward returns a label panel, either the
caller's own or computed from prices by ``forward_returns``; both go to the analyzer's
panel path, ``FactorAnalyzer.analyze_panels``, the one ``Factor.analyze`` runs through. The
analyzer is imported only when used, so importing ``quantlab.api`` does not load it.
"""

from numbers import Integral

import numpy as np

from quantlab.api import _labels
from quantlab.utils.frame import columns_present, library_of, to_field_panel, to_panel

#: The fret name of a wide returns frame, whose value column has no name of its own.
WIDE_RETURNS_NAME = "returns"


def analyze_factors(factors, returns, *, prices, price, span, delay, quantiles, columns):
    """Analyze ``factors`` against forward returns; see ``quantlab.api.analyze_factors``."""
    _check_arguments(returns, prices, price, span, delay, quantiles)
    library = library_of(factors)
    source = prices if returns is None else returns
    _check_columns_named(columns, factors, source, "returns" if prices is None else "prices")

    panel = to_panel(factors, columns=columns_present(columns, factors), purpose="factors")
    _check_factor_columns(panel)
    if prices is not None:
        label = _labels.forward_returns(
            prices,
            price=price,
            span=int(span),
            delay=int(delay),
            binary=False,
            columns=columns_present(columns, prices),
            as_xarray=True,
        )
        fret_label = f"the forward returns of prices (price={price!r})"
    else:
        values = to_field_panel(
            returns, WIDE_RETURNS_NAME, columns=columns, purpose="returns", keep_name=True
        ).values
        label = values.to_dataset()
        fret_label = "returns"

    from quantlab.analysis.factor_report import FactorAnalyzer

    analysis = FactorAnalyzer(quantiles=int(quantiles)).analyze_panels(
        panel, [label], [int(span)], factor_label="factors", fret_labels=[fret_label]
    )
    analysis.library = "polars" if library == "polars" else "pandas"
    return analysis


def _check_arguments(returns, prices, price, span, delay, quantiles) -> None:
    """Raise on a missing or doubled source, or an argument of the wrong type or range."""
    if (returns is None) == (prices is None):
        raise ValueError(
            "analyze_factors: pass exactly one of returns= and prices=: returns are your "
            "own forward returns, prices are turned into forward returns by "
            "quantlab.api.forward_returns."
        )
    for name, value, least in (
        ("span", span, 1), ("delay", delay, 0), ("quantiles", quantiles, 2)
    ):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"analyze_factors: {name} must be an int, got {value!r}.")
        if value < least:
            raise ValueError(f"analyze_factors: {name} must be at least {least}, got {value}.")
    if not isinstance(price, str):
        raise TypeError(f"analyze_factors: price must be a column name, got {price!r}.")
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


def _check_factor_columns(panel) -> None:
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
