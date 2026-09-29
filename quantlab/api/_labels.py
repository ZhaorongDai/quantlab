"""Orchestration behind ``quantlab.api.forward_returns``.

The library labels ``Return`` and ``BinaryReturn`` read the ``adjOpen`` field. The caller's
chosen price column is exposed under that field on a ``FrameDataset``, so the label reads
it unchanged. The labels fix ``delay`` at 1; any other delay re-wraps the label's trailing
factor in a ``Forward`` with the caller's ``span`` and ``delay``. Label classes are named by
dotted path and imported only when used, so importing ``quantlab.api`` does not load
KunQuant.
"""

import dataclasses
import os
from numbers import Integral

import pandas as pd

from quantlab.base.config import FactorConfig
from quantlab.dataset.memory import FrameDataset
from quantlab.label.forward import Forward
from quantlab.utils.frame import INDEX_COLUMNS, library_of, to_frame, to_panel
from quantlab.utils.module import get_cls_from_path

#: The field the library labels read.
LABEL_FIELD = "adjOpen"

#: Label class by ``binary``.
_LABELS = {
    False: "quantlab.label.predefined.fret.Return",
    True: "quantlab.label.predefined.fret.BinaryReturn",
}


def forward_returns(frame, *, price, span, delay, binary, columns, as_xarray):
    """Compute the forward-return label; see ``quantlab.api.forward_returns``."""
    _check_arguments(price, span, delay, binary)
    span, delay = int(span), int(delay)
    library = "xarray" if as_xarray else library_of(frame)
    purpose = f"forward_returns(price={price!r})"
    panel = to_panel(frame, columns=columns, required=(price,), purpose=purpose)
    panel = panel[[price]].rename({price: LABEL_FIELD})

    label_cls = get_cls_from_path(_LABELS[binary])
    label = label_cls(
        FactorConfig(
            warmup_bars=0,
            dataset=FrameDataset(panel),
            mode="batch",
            data_columns=(LABEL_FIELD,),
            kwargs={"n_forward_periods": span},
            njobs=os.cpu_count() or 1,
        )
    )
    if delay != label.config.delay:
        label = Forward(dataclasses.replace(label.config, delay=delay))

    timestamps = pd.DatetimeIndex(panel["timestamp"].values)
    result = label.compute(timestamps[0], timestamps[-1])
    return to_frame(result.transpose(*INDEX_COLUMNS), library)


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
