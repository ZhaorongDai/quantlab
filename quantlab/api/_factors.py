"""Orchestration behind ``quantlab.api.compute_factors``.

The short-name catalog maps each name onto a factor class and onto the fields that class
reads, so a caller's canonical columns (``open``, ``high``, ``low``, ``close``,
``volume``, ``amount``) reach the class under the names it programs against. Factor
classes are named by dotted path and imported only when used, so importing
``quantlab.api`` does not load KunQuant.
"""

from dataclasses import dataclass

import numpy as np

from quantlab.api._compute import NJOBS, compute_over, output_library
from quantlab.base.config import FactorConfig
from quantlab.base.factor import Factor
from quantlab.dataset.memory import FrameDataset
from quantlab.utils.frame import to_panel
from quantlab.utils.module import get_cls_from_path

#: The canonical price and volume columns of a frame.
CANONICAL_COLUMNS = ("open", "high", "low", "close", "volume")

#: The adjusted fields the ``*Stock`` factor classes read, by canonical column.
_EQUITY_FIELDS = {
    "open": "adjOpen",
    "high": "adjHigh",
    "low": "adjLow",
    "close": "adjClose",
    "volume": "adjVolume",
}

#: The fields the ``*SpotKline`` factor classes read: the canonical names plus ``amount``.
_CRYPTO_FIELDS = {name: name for name in CANONICAL_COLUMNS + ("amount",)}


@dataclass(frozen=True)
class _Entry:
    """One short name: the factor class and the fields it reads, by canonical column."""

    class_path: str
    fields: dict


#: Short name to catalog entry.
CATALOG = {
    "alpha101": _Entry("quantlab.factor.predefined.alpha101.Alpha101Stock", _EQUITY_FIELDS),
    "alpha158": _Entry("quantlab.factor.predefined.alpha158.Alpha158Stock", _EQUITY_FIELDS),
    "alpha101_crypto": _Entry(
        "quantlab.factor.predefined.alpha101.Alpha101SpotKline", _CRYPTO_FIELDS
    ),
    "alpha158_crypto": _Entry(
        "quantlab.factor.predefined.alpha158.Alpha158SpotKline", _CRYPTO_FIELDS
    ),
}


def compute_factors(frame, factor, *, columns, as_xarray):
    """Compute ``factor`` over every bar of ``frame``; see ``quantlab.api.compute_factors``."""
    library = output_library(frame, as_xarray)
    cls, entry, label = _resolve(factor)
    if entry is None:
        panel = to_panel(frame, columns=columns, purpose=label)
    else:
        panel = to_panel(frame, columns=columns, required=tuple(entry.fields), purpose=label)
        panel = panel[list(entry.fields)].rename(entry.fields)
    numeric = tuple(
        name for name, var in panel.data_vars.items() if np.issubdtype(var.dtype, np.number)
    )
    return compute_over(panel, lambda dataset: cls(_config(cls, dataset, numeric)), library)


def _resolve(factor) -> tuple[type, "_Entry | None", str]:
    """Return ``(class, catalog entry or None, label for messages)`` for ``factor``."""
    if isinstance(factor, str):
        if factor not in CATALOG:
            raise ValueError(
                f"Unknown factor short name {factor!r}. Valid names: "
                f"{', '.join(repr(name) for name in CATALOG)}; or pass a Factor subclass."
            )
        entry = CATALOG[factor]
        return get_cls_from_path(entry.class_path), entry, repr(factor)
    if not (isinstance(factor, type) and issubclass(factor, Factor)):
        raise TypeError(
            f"factor must be a short name ({', '.join(repr(n) for n in CATALOG)}) or a "
            f"Factor subclass, got {factor!r}."
        )
    # The nearest catalog class in the method resolution order wins, so a subclass of a
    # catalog class reads that entry's fields. Paths are compared rather than classes,
    # so a Polars factor does not import the KunQuant catalog classes.
    by_path = {entry.class_path: entry for entry in CATALOG.values()}
    for base in factor.__mro__:
        entry = by_path.get(f"{base.__module__}.{base.__qualname__}")
        if entry is not None:
            return factor, entry, factor.__name__
    return factor, None, factor.__name__


def _config(cls: type, dataset: FrameDataset, variables: tuple[str, ...]):
    """Return a batch config of ``cls`` over ``dataset`` with no warm-up.

    The whole frame is computed, so there is nothing before its first bar to warm up
    from: the leading bars of a rolling window are NaN. A KunQuant factor is fed every
    numeric variable of the panel.
    """
    config_cls = getattr(cls, "config_cls", None)
    if config_cls is None:
        raise TypeError(f"{cls.__name__} declares no config_cls to build its config from.")
    if issubclass(config_cls, FactorConfig):
        return config_cls(
            warmup_bars=0,
            dataset=dataset,
            mode="batch",
            data_columns=variables,
            njobs=NJOBS,
        )
    return config_cls(warmup_bars=0, dataset=dataset)

