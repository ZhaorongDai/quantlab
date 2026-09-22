"""The two symbol-axis contracts, each pinned once (03.11-02).

`quantlab/utils/symbol_axis.py` exists because the symbol axis carries TWO
contracts that were previously re-expressed at every call site -- its ORDER
(numeric, not lexicographic) and its DTYPE (the stored axis decides, not the
caller). Eight-plus sites each wrote their own `sorted()` / `str()`, so the two
contracts could drift independently and silently.

Every test here names the mutation that reddens it, following this
repository's convention.
"""

import numpy as np
import pandas as pd
import pytest

from quantlab.utils.symbol_axis import normalize_to_axis_dtype, sort_symbol_axis

# ---------------------------------------------------------------------------
# sort_symbol_axis -- the ORDER contract
# ---------------------------------------------------------------------------


def test_integer_permnos_sort_numerically() -> None:
    """PERMNOs are integers. `7000` comes before `10107`, always.

    RED under: `sorted(values)` replaced by `sorted(values, key=str)`, or by
    any text-first comparison.
    """
    assert sort_symbol_axis([7000, 14593, 93436, 10107]) == [
        7000,
        10107,
        14593,
        93436,
    ]


def test_non_numeric_labels_fall_back_to_lexicographic_order() -> None:
    """Tickers are not numbers, and this function is the ONE ordering source
    for the symbol axis -- so it has to serve the ticker axis too rather than
    refuse it.

    RED under: raising on a non-integer label, or leaving the input order
    untouched.
    """
    assert sort_symbol_axis(["AAPL", "MSFT", "A"]) == ["A", "AAPL", "MSFT"]


def test_digit_strings_sort_numerically_without_changing_element_type() -> None:
    """The 03.10-era axis spelled PERMNOs as digit STRINGS. Those order
    numerically too -- but this function decides ORDER ONLY; converting the
    elements is `normalize_to_axis_dtype`'s job, and doing both here would be
    the two contracts collapsing back into one place.

    RED under: returning `[int(v) for v in ...]`, or sorting digit strings
    lexicographically.
    """
    result = sort_symbol_axis(["10107", "7000"])
    assert result == ["7000", "10107"]
    assert all(isinstance(value, str) for value in result), [
        type(value) for value in result
    ]


def test_an_empty_axis_sorts_to_an_empty_list() -> None:
    """An empty universe is a legitimate input (a window before any listing),
    not an error.

    RED under: indexing element 0 to decide the key, without an empty guard.
    """
    assert sort_symbol_axis([]) == []


def test_a_mixed_axis_falls_back_instead_of_raising() -> None:
    """A mixed axis is a defect upstream, but THIS function raising on it
    would turn a diagnosable panel into an exception with no panel to look at.
    It orders; it does not police.

    RED under: `key=int` applied unconditionally (raises `ValueError`), or an
    explicit type check that raises.
    """
    assert sort_symbol_axis([1, "AAPL"]) == [1, "AAPL"]


def test_the_numeric_and_lexicographic_orders_actually_diverge() -> None:
    """The regression that makes all of the above worth having, fixed from a
    real measurement (RESEARCH 03.11 R2-H, measured 2026-09-20):

        sorted(str) : ['10107', '14593', '7000', '93436']
        sorted(int) : [7000, 10107, 14593, 93436]

    Historical PERMNOs happen to be five digits (~10000-93436), so on today's
    universe the two orders COINCIDE -- which is exactly why a lexicographic
    slip survives every existing suite. One four-digit PERMNO forks them.

    RED under: `sort_symbol_axis` degenerating to `sorted(values, key=str)`,
    which makes the two sides of the first assertion equal.
    """
    values = [7000, 14593, 93436, 10107]

    numeric = [str(value) for value in sort_symbol_axis(values)]
    lexicographic = sorted(str(value) for value in values)
    assert numeric != lexicographic, (numeric, lexicographic)
    assert numeric == ["7000", "10107", "14593", "93436"]
    assert lexicographic == ["10107", "14593", "7000", "93436"]

    # ... and the non-vacuity half: with only five-digit PERMNOs they agree,
    # so a suite built solely on today's universe cannot see the difference.
    five_digit = [14593, 93436, 10107]
    assert [str(value) for value in sort_symbol_axis(five_digit)] == sorted(
        str(value) for value in five_digit
    )


# ---------------------------------------------------------------------------
# normalize_to_axis_dtype -- the DTYPE contract
# ---------------------------------------------------------------------------


def test_digit_strings_normalize_onto_an_int64_axis() -> None:
    """The whole point: a caller holding digit strings must reach an int64
    store's labels, not miss every one of them.

    RED under: `[str(symbol) for symbol in symbols]`, i.e.
    `quantlab/backend.py:451` as it stood before this plan.
    """
    stored = pd.Index([10107], dtype="int64")
    result = normalize_to_axis_dtype(["10107"], stored)

    assert result == [10107]
    assert all(
        isinstance(value, (int, np.integer)) and not isinstance(value, bool)
        for value in result
    ), [type(value) for value in result]


def test_integers_normalize_onto_a_string_axis() -> None:
    """The reverse direction, and it is NOT symmetric under `astype`: measured
    2026-09-20, `pd.Index([10107]).astype(object).tolist()` returns the python
    int `10107`, not `'10107'`. A textual axis normalizes by `str`.

    RED under: `pd.Index(list(labels)).astype(stored_index.dtype).tolist()`
    applied unconditionally, which returns `[10107]` here and then misses
    every label of the string store.
    """
    stored = pd.Index(["10107"], dtype=object)
    result = normalize_to_axis_dtype([10107], stored)

    assert result == ["10107"]
    assert all(isinstance(value, str) for value in result), [
        type(value) for value in result
    ]


def test_a_label_that_cannot_reach_the_axis_dtype_raises() -> None:
    """A ticker handed to an int64 axis has no int64 spelling. Coercing it to
    SOMETHING and carrying on is the failure mode this whole plan exists to
    remove: the downstream `reindex` would miss silently and NaN the panel.

    RED under: swallowing the conversion error and falling back to `str()`, or
    to dropping the offending label.
    """
    stored = pd.Index([10107], dtype="int64")

    with pytest.raises(ValueError) as excinfo:
        normalize_to_axis_dtype(["AAPL"], stored)

    message = str(excinfo.value)
    assert "AAPL" in message
    assert "int64" in message
