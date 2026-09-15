"""Convert arbitrary result payloads into strictly standard JSON values.

vectorbt `stats()` output carries NaN, +/-inf, `pd.Timestamp`, `pd.Timedelta`
(including `NaT`) and numpy scalars. Plain `json.dump` either emits the
non-standard `NaN` / `Infinity` tokens (which strict parsers reject) or raises
on the timestamp types, so every JSON artifact a backtest run persists goes
through `to_jsonable` first (03.7-RESEARCH.md Pitfall 10).

**Lossless, or loud (code review WR-07).** This function used to fall back to
`str(value)` for any type it did not know, and to coerce dict keys with `str()`
without checking. A numpy array therefore persisted as numpy's abbreviated repr
(`np.arange(2000.0)` -> `'[0.000e+00 1.000e+00 ... 1.999e+03]'`), a `pd.Series`
or a set became unrecoverable text, and `{1: 'x', '1': 'y'}` silently lost
`'x'`. The output was strict JSON only syntactically. Now arrays, Series,
Indexes and sets become full lists, colliding keys raise `ValueError`, and an
unsupported type raises `TypeError` instead of being stringified.

A LEAF module: stdlib, numpy and pandas only, zero project-internal imports.
"""

import datetime
import decimal
import enum
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["to_jsonable"]


def _is_nat(value: object) -> bool:
    if value is pd.NaT:
        return True
    if isinstance(value, (np.datetime64, np.timedelta64)):
        return bool(np.isnat(value))
    return False


def _canonical(item: object) -> str:
    """A total, deterministic sort key for already-converted JSON values."""
    return json.dumps(item, sort_keys=True)


def to_jsonable(value: object) -> object:
    """Return `value` rebuilt from JSON-safe types only.

    - dict -> dict with str keys and converted values; two keys that are equal
      after `str()` coercion raise `ValueError` (one value would be lost);
    - list/tuple -> list; `np.ndarray` (any rank), `pd.Series` and `pd.Index`
      -> nested lists, every element converted by these same rules (a 0-d
      array -> its scalar); set/frozenset -> list in a deterministic order;
    - bool / numpy bool -> bool; numpy integer -> int;
    - float / numpy floating -> float, with NaN, +inf and -inf -> None;
    - Timestamp / datetime64 / datetime / date -> ISO-8601 string;
    - Timedelta / timedelta64 / timedelta -> str;
    - NaT (either kind) -> None;
    - `pathlib.Path` -> str; `decimal.Decimal` -> str (exact digits);
      `enum.Enum` -> its value, converted;
    - str, int and None pass through;
    - anything else raises `TypeError` naming the type. Persisting its
      `str()` would hide data loss behind syntactically valid JSON.
    """
    if value is None:
        return None
    # Enum before str/int: a StrEnum or IntEnum is also a str or int.
    if isinstance(value, enum.Enum):
        return to_jsonable(value.value)
    if isinstance(value, str):
        return str(value)
    # bool before int: bool is an int subclass.
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if _is_nat(value):
        return None
    if isinstance(value, np.datetime64):
        return pd.Timestamp(value).isoformat()
    # pd.Timestamp subclasses datetime.datetime, which subclasses date.
    if isinstance(value, (datetime.datetime, datetime.date)):
        return value.isoformat()
    if isinstance(value, np.timedelta64):
        return str(pd.Timedelta(value))
    # pd.Timedelta subclasses datetime.timedelta.
    if isinstance(value, datetime.timedelta):
        return str(value)
    if isinstance(value, dict):
        converted: dict = {}
        for key, item in value.items():
            name = str(key)
            if name in converted:
                raise ValueError(
                    f"to_jsonable: dict keys collide after str() coercion on "
                    f"{name!r} (from key {key!r}); one of the values would be "
                    f"silently dropped"
                )
            converted[name] = to_jsonable(item)
        return converted
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        # Iterate rather than `tolist()`: `tolist()` turns datetime64[ns] and
        # timedelta64[ns] elements into bare integers.
        if value.ndim == 0:
            return to_jsonable(value[()])
        return [to_jsonable(item) for item in value]
    if isinstance(value, (pd.Series, pd.Index)):
        return to_jsonable(value.to_numpy())
    if isinstance(value, (set, frozenset)):
        return sorted((to_jsonable(item) for item in value), key=_canonical)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, decimal.Decimal):
        return str(value)
    raise TypeError(
        f"to_jsonable: cannot convert a {type(value).__module__}."
        f"{type(value).__qualname__} to JSON; add an explicit conversion "
        f"instead of persisting its str()"
    )
