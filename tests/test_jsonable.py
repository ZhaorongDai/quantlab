"""Strictness of `quantlab/utils/jsonable.py:to_jsonable` (phase 03.7, code review WR-07).

`to_jsonable` serializes every JSON artifact a backtest or a CV run persists:
`metrics.json`, `config.json`, `fingerprint.json`, `liquidations.json` and
`cv_folds.json`. The fold dicts in `cv_folds.json` carry whatever a model's
`_fit` returns. Before the fix the function was strict only syntactically:

- **Arrays were silently truncated.** Anything without a special case fell
  through to `str(value)`, so `np.arange(2000.0)` persisted as numpy's
  abbreviated repr `'[0.000e+00 1.000e+00 ... 1.999e+03]'`. A `pd.Series`, a
  `pd.Index` and a set were stringified the same way, unrecoverably.
- **Colliding keys were silently dropped.** Keys were coerced with `str()`
  without a check, so `{1: 'x', '1': 'y'}` became `{'1': 'y'}`.
- **Unknown types were silently stringified.** An object had no JSON form, and
  the persisted artifact still parsed.

What is locked here, and what turns each lock red:

- arrays (any rank, including datetime64 and NaN elements), `pd.Series` and
  `pd.Index` become full nested lists, element by element through the same
  rules (a bare `tolist()` would turn datetime64[ns] into integers);
- sets and frozensets become lists in a deterministic order;
- two keys that collide after `str()` coercion raise `ValueError` naming
  the key;
- a value of an unsupported type raises `TypeError` naming the type;
- `decimal.Decimal` keeps its exact digits as a string, and the existing
  conversions (NaN/inf -> None, timestamps -> ISO, Path, Enum) are unchanged.
"""

import decimal
import enum
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantlab.utils.jsonable import to_jsonable


def _strict_roundtrip(value):
    return json.loads(json.dumps(value, allow_nan=False))


def test_large_arrays_are_persisted_in_full_not_as_an_abbreviated_repr():
    values = np.arange(2000.0)

    converted = to_jsonable({"per_bar": values})

    assert converted == {"per_bar": values.tolist()}
    assert _strict_roundtrip(converted) == converted


def test_array_elements_go_through_the_scalar_rules():
    assert to_jsonable(np.array([[1, 2], [3, 4]])) == [[1, 2], [3, 4]]
    assert to_jsonable(np.array([1.0, np.nan, np.inf])) == [1.0, None, None]
    stamps = np.array(["2024-01-02", "NaT"], dtype="datetime64[ns]")
    assert to_jsonable(stamps) == ["2024-01-02T00:00:00", None]
    assert to_jsonable(np.float64(2.5)) == 2.5
    assert to_jsonable(np.array(7)) == 7


def test_series_and_indexes_become_lists():
    assert to_jsonable(pd.Series([1.0, np.nan, 3.0])) == [1.0, None, 3.0]
    assert to_jsonable(pd.Index(["AAA", "BBB"])) == ["AAA", "BBB"]
    assert to_jsonable(pd.DatetimeIndex(["2024-01-02"])) == ["2024-01-02T00:00:00"]


def test_sets_become_deterministically_ordered_lists():
    assert to_jsonable({"b", "a", "c"}) == ["a", "b", "c"]
    assert to_jsonable(frozenset({3, 1, 2})) == [1, 2, 3]


def test_keys_colliding_after_str_coercion_raise_naming_the_key():
    with pytest.raises(ValueError, match=r"'1'"):
        to_jsonable({1: "x", "1": "y"})


def test_unsupported_types_raise_naming_the_type():
    class Opaque:
        pass

    with pytest.raises(TypeError, match="Opaque"):
        to_jsonable({"nested": [Opaque()]})


def test_existing_conversions_are_unchanged():
    class Color(enum.Enum):
        RED = "red"

    value = {
        "nan": float("nan"),
        "timestamp": pd.Timestamp("2024-01-02 15:30"),
        "nat": pd.NaT,
        "timedelta": pd.Timedelta(days=1),
        "path": Path("/tmp/x"),
        "enum": Color.RED,
        "decimal": decimal.Decimal("0.00012345678901234567890"),
        "tuple": (1, "a", None, True),
    }

    assert to_jsonable(value) == {
        "nan": None,
        "timestamp": "2024-01-02T15:30:00",
        "nat": None,
        "timedelta": "1 days 00:00:00",
        "path": "/tmp/x",
        "enum": "red",
        "decimal": "0.00012345678901234567890",
        "tuple": [1, "a", None, True],
    }
