"""The two contracts of a panel's ``symbol`` axis: its order and its dtype.

A ``symbol`` axis is sorted numerically when every label is an integer (or a
digit string) and lexicographically otherwise; ``sort_symbol_axis`` is the one
place that rule is spelled out. Separately, labels a caller holds must be
re-spelled in the dtype the stored axis carries before they are used to index
it, which ``normalize_to_axis_dtype`` does. Indexing an int64 axis with digit
strings does not raise: ``reindex`` matches nothing and silently returns an
all-NaN panel of the right shape, which is the failure this module prevents.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

#: numpy/pandas dtype kinds that denote a textual axis: ``O`` (Python object
#: and pandas' ``str`` dtype), ``U`` (fixed-width unicode, what a list literal
#: round-trips to in zarr), ``S`` (bytes) and ``T`` (``np.dtypes.StringDType``,
#: what ``xr.open_zarr`` decodes an object-encoded coordinate to).
_TEXTUAL_KINDS = frozenset("OUST")


def _is_integral(value: object) -> bool:
    """Return True if ``value`` is an integer or a string of digits.

    ``bool`` is excluded deliberately: it is an ``int`` subclass, and an axis
    of booleans must not be treated as an integer universe.
    """
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, np.integer)):
        return True
    return isinstance(value, str) and value.isdigit()


def sort_symbol_axis(values: Iterable) -> list:
    """Sort a symbol axis, numerically when every label is integral.

    Security identifiers such as PERMNOs are integers that are sometimes
    carried as digit strings; plain ``sorted()`` on the text would put
    ``"14593"`` before ``"7000"``. When every label passes ``_is_integral`` the
    sort key is ``int``; a mixed or non-integer axis falls back to ``str``
    comparison rather than raising, because this function orders labels and
    does not validate them.

    Only the order changes. Elements keep their input type, so digit strings
    come back as digit strings; converting them is ``normalize_to_axis_dtype``'s
    job.

    Args:
        values: The labels to sort.

    Returns:
        A new sorted list, empty for empty input.

    Example:
        >>> sort_symbol_axis(["10107", "14593", "7000"])
        ['7000', '10107', '14593']
    """
    materialised = list(values)
    if not materialised:
        return []
    key = int if all(_is_integral(value) for value in materialised) else str
    return sorted(materialised, key=key)


def normalize_to_axis_dtype(labels: Iterable, stored_index: pd.Index) -> list:
    """Re-spell ``labels`` in the dtype ``stored_index`` carries.

    The stored axis decides. A textual axis (see ``_TEXTUAL_KINDS``) receives
    ``str(label)`` for each label; any other axis receives
    ``pd.Index(labels).astype(dtype)``. The two directions are not symmetric:
    ``astype(object)`` boxes an integer instead of rendering it, which is why
    textual targets use ``str`` explicitly. Textual is decided by dtype kind,
    never by width, because a fixed-width store's width depends on the labels
    it happens to hold.

    Args:
        labels: The labels a caller wants to select or reindex with.
        stored_index: The axis of the store being indexed.

    Returns:
        A list of labels in the stored dtype, empty for empty input.

    Raises:
        ValueError: If any label has no spelling in the stored dtype. The
            message names the offending labels. Labels are never coerced to a
            guess or dropped, since either would hand ``reindex`` a request
            that misses silently.

    Example:
        >>> normalize_to_axis_dtype(["10107", "7000"], pd.Index([7000, 10107]))
        [10107, 7000]
    """
    materialised = list(labels)
    if not materialised:
        return []

    dtype = stored_index.dtype
    if getattr(dtype, "kind", None) in _TEXTUAL_KINDS:
        return [str(label) for label in materialised]

    try:
        return pd.Index(materialised).astype(dtype).tolist()
    except (TypeError, ValueError) as error:
        rejected = [
            label
            for label in materialised
            if not _coercible(label, dtype)
        ]
        shown = rejected[:10] if rejected else materialised[:10]
        elided = (
            f" (and {len(rejected) - 10} more)"
            if len(rejected) > 10
            else ""
        )
        raise ValueError(
            f"normalize_to_axis_dtype: refusing to coerce {shown!r}{elided} "
            f"onto the stored axis dtype {dtype!r}. Guessing a value here is "
            f"worse than failing: a wrong label does not raise downstream, it "
            f"MISSES -- `reindex` drops every label it cannot match and fills "
            f"NaN, so the panel comes back the right shape, the right dtype "
            f"and entirely empty (measured 2026-09-20 on this very path: 0 of "
            f"12 cells survived, no exception, no log line). Align the label "
            f"type to the axis at the CALLING layer, where it is known what "
            f"those labels are, rather than having this function guess. "
            f"Underlying error: {error}"
        ) from error


def _coercible(label: object, dtype) -> bool:
    """Return True if ``label`` alone survives ``astype(dtype)``.

    Used only to name the offending labels in ``normalize_to_axis_dtype``'s
    error message, so it points at the bad label rather than the whole request.
    """
    try:
        pd.Index([label]).astype(dtype)
    except (TypeError, ValueError):
        return False
    return True
