"""Convert arbitrary result payloads into strictly standard JSON values.

Backtest statistics carry NaN, infinities, ``pd.Timestamp``, ``pd.Timedelta``
(including ``NaT``) and numpy scalars. Plain ``json.dump`` either emits the
non-standard ``NaN`` and ``Infinity`` tokens, which strict parsers reject, or
raises on the timestamp types. Every JSON artifact a backtest run persists is
passed through ``to_jsonable`` first.

The conversion either keeps every value or fails with an error; it never
silently loses data. Arrays, Series, Indexes and sets become full lists, dict
keys that collide after ``str()`` raise ``ValueError``, and an unsupported
type raises ``TypeError`` instead of being stringified. This module has no
project-internal imports, so any layer can use it.
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
    """Return True if ``value`` is pandas ``NaT`` or a numpy NaT datetime/timedelta."""
    if value is pd.NaT:
        return True
    if isinstance(value, (np.datetime64, np.timedelta64)):
        return bool(np.isnat(value))
    return False


def _canonical(item: object) -> str:
    """Return a deterministic sort key for an already-converted JSON value.

    Serialising to a JSON string makes values of mixed types comparable, so
    a set of them can always be sorted.
    """
    return json.dumps(item, sort_keys=True)


def to_jsonable(value: object) -> object:
    """Return ``value`` rebuilt from JSON-safe types only.

    Conversion rules:

    - ``dict`` becomes a dict with ``str`` keys and converted values; two keys
      that are equal after ``str()`` raise ``ValueError``.
    - ``list``, ``tuple``, ``np.ndarray`` (any rank), ``pd.Series`` and
      ``pd.Index`` become lists, element by element; a 0-d array becomes its
      scalar; ``set`` and ``frozenset`` become lists in a deterministic order.
    - Booleans and integers (Python or numpy) become ``bool`` and ``int``;
      floats become ``float`` with NaN and infinities mapped to ``None``.
    - Timestamps, ``datetime64``, ``datetime`` and ``date`` become ISO-8601
      strings; timedeltas become ``str``; ``NaT`` of either kind becomes
      ``None``; ``Path`` and ``Decimal`` become ``str``; an ``Enum`` is replaced
      by its converted value; ``str``, ``int`` and ``None`` pass through.

    Parameters
    ----------
    value : object
        The object to convert.

    Returns
    -------
    object
        A structure made only of dict, list, str, int, float, bool and None.

    Raises
    ------
    ValueError
        If two dict keys collide after ``str()`` coercion.
    TypeError
        If ``value`` (or any nested value) has an unsupported type.

    Examples
    --------
    >>> to_jsonable({"sharpe": np.float64("nan"), "start": pd.Timestamp("2024")})
    {'sharpe': None, 'start': '2024-01-01T00:00:00'}
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
