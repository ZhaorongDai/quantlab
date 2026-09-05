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

import numpy as np
import polars as pl
import xarray as xr
from loguru import logger

REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")

# Price-like variables checked for zero/negative values and extreme jumps.
# Includes Tiingo's adjusted-price variants when present.
_PRICE_LIKE_COLUMNS = (
    "open",
    "high",
    "low",
    "close",
    "adjOpen",
    "adjHigh",
    "adjLow",
    "adjClose",
)

# A single-step percentage change on `close` beyond this threshold is
# flagged as an extreme jump. Simple, documented, fixed rule — deeper
# statistical/ML anomaly detection is explicitly out of scope for v1
# (CONTEXT.md D-08 / Deferred Ideas).
_EXTREME_JUMP_THRESHOLD = 0.5


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


def flag_anomalies(data: xr.Dataset) -> xr.Dataset:
    """Flag (not delete/correct) zero/negative price or extreme jumps — D-07.

    Adds a boolean data variable `anomaly_flag` (same [timestamp, symbol]
    dims as the source data) that is True wherever any present price-like
    variable is zero/negative, or `close` makes a single-step percentage
    change beyond `_EXTREME_JUMP_THRESHOLD`. Never mutates, drops, or
    corrects the underlying values — anomalies stay visible for later
    investigation (auto-correction would hide the evidence a debugging
    session needs).
    """
    present_price_columns = [
        c for c in _PRICE_LIKE_COLUMNS if c in data.data_vars
    ]

    if present_price_columns:
        anomaly = xr.zeros_like(data[present_price_columns[0]], dtype=bool)
    else:
        anomaly = xr.DataArray(
            np.zeros(
                (data.sizes.get("timestamp", 0), data.sizes.get("symbol", 0)),
                dtype=bool,
            ),
            dims=["timestamp", "symbol"],
            coords={"timestamp": data["timestamp"], "symbol": data["symbol"]},
        )

    for col in present_price_columns:
        anomaly = anomaly | (data[col] <= 0)

    if "close" in data.data_vars:
        close = data["close"]
        shifted = close.shift(timestamp=1)
        pct_change = close.diff(dim="timestamp") / shifted
        # A shifted (prior) value that is itself zero/negative/NaN makes the
        # percentage change undefined/meaningless — only compare against a
        # strictly-positive prior price. Comparisons against NaN evaluate to
        # False elementwise, so no separate NaN-handling is needed.
        valid_prior = shifted > 0
        jump = (np.abs(pct_change) > _EXTREME_JUMP_THRESHOLD) & valid_prior
        # jump is one element shorter along `timestamp` (diff drops the
        # first row); reindex back onto the full grid, treating the first
        # timestamp (no prior value to diff against) as not-a-jump.
        jump = jump.reindex(timestamp=data["timestamp"], fill_value=False)
        anomaly = anomaly | jump

    anomaly = anomaly.astype(bool)
    anomaly.name = "anomaly_flag"

    flagged_count = int(anomaly.sum().item())
    if flagged_count > 0:
        logger.warning(
            f"flag_anomalies: flagged {flagged_count} anomalous "
            f"(timestamp, symbol) data point(s) (zero/negative price or "
            f"extreme jump) — values left unmodified, see `anomaly_flag`."
        )

    return data.assign(anomaly_flag=anomaly)


def validate_schema(
    data: xr.Dataset, required_columns: tuple[str, ...] = REQUIRED_COLUMNS
) -> xr.Dataset:
    """Basic schema/non-null validation — D-08.

    Raises `ValueError` naming the missing column(s) if any
    `required_columns` entry is absent from `data.data_vars`. For columns
    present but not in `required_columns` (non-key columns) containing
    unexpected nulls, logs a `loguru.logger.warning` and returns `data`
    unchanged — does NOT raise, consistent with D-07's flag-don't-delete
    philosophy. Deeper statistical anomaly detection beyond flagging is
    explicitly out of scope for v1 (CONTEXT.md D-08).
    """
    missing = [col for col in required_columns if col not in data.data_vars]
    if missing:
        raise ValueError(
            f"validate_schema: required column(s) missing from dataset: "
            f"{missing}"
        )

    non_key_columns = [
        col for col in data.data_vars if col not in required_columns
    ]
    for col in non_key_columns:
        null_count = int(data[col].isnull().sum().item())
        if null_count > 0:
            logger.warning(
                f"validate_schema: column '{col}' has {null_count} "
                f"unexpected null value(s) — not raising, per flag-don't-"
                f"delete philosophy (D-07)."
            )

    return data


def clean_market_data(data: xr.Dataset) -> xr.Dataset:
    """Single entry point called from base/data.py:Dataset.from_raw_data().

    NaN-gap behavior (D-06) requires no logic here — already guaranteed by
    the dedup step (dedup_raw_frame(), run by each subclass's
    _raw_data_to_xr() before this function ever sees the data) plus
    pandas.DataFrame.set_index([...]).to_xarray()'s cartesian-product
    behavior upstream. This function only layers schema validation (D-08)
    and anomaly-flagging (D-07) on top. This module never forward-fills,
    backward-fills, interpolates, or fills missing market-data values —
    NaN gaps always pass through unchanged.
    """
    data = validate_schema(data)
    data = flag_anomalies(data)
    return data
