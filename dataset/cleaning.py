"""Shared raw-market-data cleaning, reused by every Dataset subclass.

Two entry points, called at two different pipeline stages:
- dedup_raw_frame(): tabular (polars), called by each subclass's
  _raw_data_to_xr() BEFORE the final .to_xarray() conversion — required
  because a non-unique (timestamp, symbol) index crashes to_xarray()
  with `ValueError: cannot convert a DataFrame with a non-unique
  MultiIndex into xarray` (see 02-RESEARCH.md Pitfall 2).
- clean_market_data(): xr.Dataset, called once centrally in
  base/data.py:Dataset.from_raw_data() AFTER conversion — anomaly-flagging
  and schema validation only. Missing-timestamp NaN-gap behavior requires
  no code here: pandas.DataFrame.set_index([...]).to_xarray() already
  produces the full (timestamp, symbol) cartesian product with NaN for
  absent combinations, given a unique index (verified empirically, see
  02-RESEARCH.md Pattern 3).

Do NOT add forward-fill/interpolate/fillna anywhere in this module — that
would fabricate data the pipeline never actually observed (D-06).

This module is a leaf: it has zero project-internal imports (only polars,
xarray, numpy, loguru, stdlib `typing`), so it can never be the source of a
future import cycle.
"""

from typing import Literal

import polars as pl

REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")


def dedup_raw_frame(
    data: pl.LazyFrame, keep: Literal["first", "last"] = "last"
) -> pl.LazyFrame:
    """Deterministic dedup on (timestamp, symbol) — D-05.

    keep="last" is the default because later-arriving files in a vendor's
    monthly-drop workflow more often represent corrected/reprocessed data
    than earlier ones (per 02-RESEARCH.md's Code Examples section) — this
    satisfies D-05's "deterministic and documented" requirement.

    Must run before `.to_xarray()`; a non-unique (timestamp, symbol)
    MultiIndex raises ValueError there (see module docstring / Pitfall 2).
    """
    return data.unique(subset=["timestamp", "symbol"], keep=keep)
