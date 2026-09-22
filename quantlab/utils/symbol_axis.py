"""The symbol axis's two contracts, each stated ONCE (03.11-02).

A panel's `symbol` axis carries TWO separate contracts, and before this module
existed neither had a home:

1. **ORDER.** The axis is sorted NUMERICALLY when its labels are integers.
2. **DTYPE.** A caller's labels must be normalised to the dtype the STORED
   axis carries before they are used to index it -- never the other way
   round, and never by an unconditional `str()`.

Both were re-expressed at every call site instead -- eight-plus bare
`sorted()` calls (`quantlab/dataset/stock.py:506-511`,
`quantlab/dataset/crsp/__init__.py:1372-1375`, `quantlab/base/constituent.py:198`,
`quantlab/dataset/masking.py:105`, `quantlab/base/model.py:310` and `:1220`,
`quantlab/utils/fingerprint.py:58`, `quantlab/dataset/chunking.py:210`) and an
unconditional `[str(symbol) for symbol in symbols]` at
`quantlab/dataset/backend.py:451`. N independent spellings of one contract are
N things that can drift, and the drift is invisible: on today's universe every
one of them agrees with every other.

The second contract is not a style point. Measured 2026-09-20 (xarray 2026.7.0
/ zarr 3.3.0), `XrBackend.widen_symbol_axis` stringified its request against an
int64 store, so the reindex matched nothing, the superset guard compared
`str(...)` on BOTH sides and therefore saw nothing dropped, and the resulting
ALL-NaN panel was renamed over the authoritative store, whose original was then
`rmtree`d -- 0 of 12 cells survived, with no exception and no log line.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

#: numpy/pandas dtype kinds that spell a TEXTUAL axis: `O` (python object and
#: pandas 3's `str` dtype, measured `pd.Index(['A']).dtype.kind == 'O'`), `U`
#: (numpy fixed-width unicode, what a list literal round-trips to in zarr),
#: `S` (bytes) and `T` (`np.dtypes.StringDType()`, what `xr.open_zarr` decodes
#: an object-encoded coordinate to).
_TEXTUAL_KINDS = frozenset("OUST")


def _is_integral(value: object) -> bool:
    """Can `value` be read as an integer WITHOUT guessing?

    `bool` is excluded deliberately: it is an `int` subclass, so a `[True,
    False]` axis would otherwise sort "numerically" and read as a legitimate
    integer universe.
    """
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, np.integer)):
        return True
    return isinstance(value, str) and value.isdigit()


def sort_symbol_axis(values: Iterable) -> list:
    """Sort a symbol axis, numerically where that is meaningful.

    **The order is part of the contract, and it is NUMERIC.** PERMNOs are
    integers -- rendered as strings on the 03.10-era axis -- so `sorted()` on
    the text would put ``"14593"`` before ``"7000"``.
    ``quantlab/utils/cli.py:resolve_symbols`` slices this list for
    ``--limit``; an unstable or surprising order truncates to a different
    batch on every run, and the second run never meets the watermarks the
    first one wrote.

    That paragraph is not new wording: it MOVED here verbatim from
    ``quantlab/dataset/crsp/membership.py:permnos_in_range``, which was the
    only place in the repository that stated it. Moving rather than copying is
    the point of this module -- a second copy is a second thing to drift.

    **Why the trap is invisible today.** Historical PERMNOs happen to be five
    digits (~10000-93436), so numeric and lexicographic order COINCIDE on the
    current universe -- measured 2026-09-20::

        sorted(str) : ['10107', '14593', '7000', '93436']
        sorted(int) : [7000, 10107, 14593, 93436]

    The two forks the moment one four-digit PERMNO appears, and not before.
    ``tests/test_symbol_axis_contract.py`` pins both halves.

    This function decides ORDER ONLY. **The returned elements have the same
    type as the input elements** -- digit strings come back as digit strings.
    Converting them is `normalize_to_axis_dtype`'s contract, and doing both
    here would collapse the two contracts back into one place, which is the
    situation this module exists to end.

    A mixed or non-integer axis falls back to `str` comparison rather than
    raising: this function orders, it does not police. A mixed axis is a
    defect upstream, and raising here would replace a diagnosable panel with
    an exception that has no panel to look at.
    """
    materialised = list(values)
    if not materialised:
        return []
    key = int if all(_is_integral(value) for value in materialised) else str
    return sorted(materialised, key=key)


def normalize_to_axis_dtype(labels: Iterable, stored_index: pd.Index) -> list:
    """Re-spell `labels` in the dtype `stored_index` actually carries.

    The stored axis decides. A caller holding digit strings must reach an
    int64 store's labels, and a caller holding integers must reach a string
    store's -- because the alternative is not an error, it is a SILENT MISS:
    `reindex` drops every label it cannot match and writes NaN in its place.

    **The two directions are not symmetric, and `astype` alone does not do
    it.** Measured 2026-09-20 (pandas 3.0.5 / numpy 2.5.3)::

        pd.Index(['10107']).astype('int64').tolist()  -> [10107]   int
        pd.Index([10107]).astype(object).tolist()     -> [10107]   int  (!)

    The second row is the trap: `astype(object)` boxes the INT, it does not
    render it. So a textual target dtype normalises with `str` and a
    non-textual one goes through `astype`. Textual is decided by dtype KIND
    (`_TEXTUAL_KINDS`), never by a width or a dtype literal -- a fixed-width
    store's width is a property of its LABELS (`<U1` for `A`, `<U9` for
    `SATX-WS-A`), so anything pinned to one width reproduces at one label set
    and nowhere else.

    On a string axis this is byte-for-byte what
    `quantlab/dataset/backend.py:451` did before -- `str(symbol)` per label.
    The behaviour change is confined to the axes where the old spelling was
    wrong.

    A label with no spelling in the target dtype raises `ValueError` naming
    the offenders and the dtype. It does NOT coerce to something, and it does
    NOT drop the label: either would hand `reindex` a request that misses
    silently, which is the entire failure this function was written to
    remove.
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
    """Would `label` alone survive `astype(dtype)`?

    Used only to NAME the offenders in the error above. Per-label so the
    message points at the one bad ticker in a universe of 7,700 rather than at
    the whole request.
    """
    try:
        pd.Index([label]).astype(dtype)
    except (TypeError, ValueError):
        return False
    return True
