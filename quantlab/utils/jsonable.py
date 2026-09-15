"""Convert arbitrary result payloads into strictly standard JSON values.

vectorbt `stats()` output carries NaN, +/-inf, `pd.Timestamp`, `pd.Timedelta`
(including `NaT`) and numpy scalars. Plain `json.dump` either emits the
non-standard `NaN` / `Infinity` tokens (which strict parsers reject) or raises
on the timestamp types, so every JSON artifact a backtest run persists goes
through `to_jsonable` first (03.7-RESEARCH.md Pitfall 10).

A LEAF module: stdlib, numpy and pandas only, zero project-internal imports.
"""

import datetime
import enum
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


def to_jsonable(value: object) -> object:
    """Return `value` rebuilt from JSON-safe types only.

    - dict -> dict with str keys and converted values; list/tuple -> list;
    - bool / numpy bool -> bool; numpy integer -> int;
    - float / numpy floating -> float, with NaN, +inf and -inf -> None;
    - Timestamp / datetime64 / datetime / date -> ISO-8601 string;
    - Timedelta / timedelta64 / timedelta -> str;
    - NaT (either kind) -> None;
    - `pathlib.Path` -> str; `enum.Enum` -> its value, converted;
    - str, int and None pass through; anything else -> `str(value)`.
    """
    if value is None:
        return None
    # Enum before str/int: a StrEnum or IntEnum is also a str or int.
    if isinstance(value, enum.Enum):
        return to_jsonable(value.value)
    if isinstance(value, str):
        return value
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
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return str(value)
