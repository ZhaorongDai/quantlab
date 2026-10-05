"""Orchestration behind ``quantlab.api.forward_returns``.

The library labels ``Return`` and ``BinaryReturn`` read the ``adjOpen`` field. The caller's
chosen price column is exposed under that field on a ``FrameDataset``, so the label reads
it unchanged. The labels fix ``delay`` at 1; any other delay wraps the label's trailing
factor, its public ``config.factor``, in a ``Forward`` with the caller's ``span`` and
``delay``. Label classes are imported only when used, so importing ``quantlab.api`` does
not load KunQuant.
"""

from numbers import Integral

import pandas as pd
import polars as pl
import xarray as xr

from quantlab.api._compute import NJOBS, compute_over, output_library
from quantlab.base.config import FactorConfig, ForwardConfig
from quantlab.label.forward import Forward
from quantlab.dataset._support.frame import INDEX_COLUMNS, to_panel

#: The field the library labels read.
LABEL_FIELD = "adjOpen"

#: Price columns suggested, in order, when the requested one is missing.
PREFERRED_PRICES = ("close", "open")


def forward_returns(frame, *, price, span, delay, binary, columns, as_xarray):
    """Compute the forward-return label; see ``quantlab.api.forward_returns``."""
    _check_arguments(price, span, delay, binary)
    span, delay = int(span), int(delay)
    library = output_library(frame, as_xarray)
    _check_price_present(frame, price, columns)
    purpose = f"forward_returns(price={price!r})"
    panel = to_panel(frame, columns=columns, required=(price,), purpose=purpose)
    panel = panel[[price]].rename({price: LABEL_FIELD})

    if binary:
        from quantlab.label.predefined.fret import BinaryReturn as label_cls
    else:
        from quantlab.label.predefined.fret import Return as label_cls

    def build(dataset):
        label = label_cls(
            FactorConfig(
                warmup_bars=0,
                dataset=dataset,
                mode="batch",
                data_columns=(LABEL_FIELD,),
                kwargs={"n_forward_periods": span},
                njobs=NJOBS,
            )
        )
        if delay != label.config.delay:
            label = Forward(ForwardConfig(factor=label.config.factor, span=span, delay=delay))
        return label

    return compute_over(panel, build, library)


def _check_price_present(frame, price: str, columns) -> None:
    """Raise when ``price`` is not a column of ``frame`` after ``columns`` renames.

    The message lists the columns present and suggests one of them as ``price=``:
    ``close`` or ``open`` when present, else the first other column. A ``columns``
    mapping naming an absent column is left to ``to_panel``, which reports it.
    """
    if isinstance(frame, xr.Dataset):
        names = list(frame.dims) + list(frame.data_vars)
    elif isinstance(frame, pl.LazyFrame):
        names = frame.collect_schema().names()
    elif isinstance(frame, pl.DataFrame):
        names = list(frame.columns)
    else:
        names = list(frame.columns)
        if isinstance(frame.index, pd.MultiIndex):
            names = [name for name in frame.index.names if name is not None] + names
    renames = dict(columns or {})
    present = [renames.get(name, name) for name in names]
    if price in present:
        return
    candidates = [name for name in present if name not in INDEX_COLUMNS]
    preferred = [name for name in PREFERRED_PRICES if name in candidates]
    suggestion = (preferred or candidates or [None])[0]
    hint = (
        f"Pass price={suggestion!r} to use one of them, or " if suggestion else "Pass "
    )
    raise ValueError(
        f"forward_returns: price column {price!r} is not in the frame. Present columns: "
        f"{present}. {hint}columns={{'yours': {price!r}}} to map one of yours onto it."
    )


def _check_arguments(price, span, delay, binary) -> None:
    """Raise on an argument of the wrong type or range, before any conversion."""
    if not isinstance(price, str):
        raise TypeError(f"forward_returns: price must be a column name, got {price!r}.")
    for name, value, least in (("span", span, 1), ("delay", delay, 0)):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"forward_returns: {name} must be an int, got {value!r}.")
        if value < least:
            raise ValueError(
                f"forward_returns: {name} must be at least {least}, got {value}."
            )
    if not isinstance(binary, bool):
        raise TypeError(f"forward_returns: binary must be True or False, got {binary!r}.")
