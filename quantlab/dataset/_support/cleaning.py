"""Shared raw-market-data cleaning, reused by every Dataset subclass.

Four entry points. Two are called at two different pipeline stages of the
raw-market-data path:
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

The third, clean_membership_panel(), is the non-OHLCV counterpart of
clean_market_data(): it is the `_clean()` route for an index-membership
boolean panel, which has none of the five required OHLCV columns and for
which anomaly-flagging would be meaningless.

The fourth, clean_nbbo_panel(), is the same kind of validator for the WRDS
TAQ NBBO bar panel (phase 03.9, D-14).

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
    `required_columns` entry is absent from `data.data_vars`. That behaviour,
    and the signature, are unchanged.

    **The discrimination rule.** A dense `[timestamp, symbol]` panel is a
    cartesian product (D-06), so a symbol that did not trade in a period is
    NaN in EVERY variable at that cell. That is the panel's design, not an
    anomaly, and counting those cells produced million-row false alarms on a
    real minute panel — which is exactly what teaches an operator to ignore
    the one warning that matters. The two sources of a null are separated by
    a STRUCTURAL MASK:

    - `structural mask` = logical AND over `isnull()` of every column in the
      `required_columns` ARGUMENT. True precisely where no bar exists at all.
    - `unexpected nulls` for a column = `isnull() & ~structural_mask` — nulls
      on cells where a bar DOES exist. Only these are warned about.

    So `trade_count` null only where the bar is absent is silent, while a
    `vwap` the vendor withheld entirely still warns — and the count it reports
    is now the number of cells that actually have a bar behind them.

    **The mask's degenerate case: an EMPTY INGEST.** If the required columns
    are themselves null everywhere — an empty vendor response written to a
    store, a mis-parsed CSV, a fully-failed backfill — the mask is True on
    every cell, `~mask` is False on every cell, and the per-column loop
    reports 0 for EVERY column. This function used to go completely silent on
    exactly the panel it exists to catch (the `vwap` claim above was false in
    that configuration), and the structural report was `logger.info`, so
    nothing surfaced at warning level either. Now that case is detected
    up front, reported at `logger.error` — deliberately LOUDER than the
    per-column warnings it was swallowing, because it means the ingest
    produced nothing usable at all — and the mask is disabled for that panel
    so the per-column loop reports real whole-column counts. It still does
    not raise: raising is reserved for a SCHEMA violation (a missing column);
    an all-null panel is a data-CONTENT problem, and `clean_market_data()`
    runs inside `from_raw_data()` on every ingest with nothing catching it,
    so an abort here would kill a chunked backfill on the first window where
    nothing traded.

    **Two deliberate widenings, both closing blind spots.**

    - REQUIRED columns are now null-checked too. `open` missing on a bar whose
      `close` exists was previously silent, because the loop skipped every
      required column; it now warns through the same mask.
    - The mask is built from the `required_columns` ARGUMENT, never from the
      module-level `REQUIRED_COLUMNS`, because
      `dataset/spot.py:SpotKlineDataset._clean()` passes Title-Case
      `("Open","High","Low","Close","Volume")`. A hardcoded mask would raise
      `KeyError` on every Binance ingest.

    A column whose dims differ from the mask's falls back to the plain
    whole-column null count rather than broadcasting into a meaningless larger
    array — no column silently loses its check to a broadcasting accident.

    Still logs a `loguru.logger.warning` and returns `data` unchanged — does
    NOT raise, consistent with D-07's flag-don't-delete philosophy. Deeper
    statistical anomaly detection beyond flagging is explicitly out of scope
    for v1 (CONTEXT.md D-08).
    """
    missing = [col for col in required_columns if col not in data.data_vars]
    if missing:
        raise ValueError(
            f"validate_schema: required column(s) missing from dataset: "
            f"{missing}"
        )

    structural_mask = None
    for col in required_columns:
        is_null = data[col].isnull()
        structural_mask = (
            is_null if structural_mask is None else (structural_mask & is_null)
        )

    empty_ingest = False
    if structural_mask is not None:
        structural_cells = int(structural_mask.sum().item())
        total_cells = int(structural_mask.size)
        if total_cells > 0 and structural_cells == total_cells:
            empty_ingest = True
            logger.error(
                f"validate_schema: EVERY required column is null on EVERY one "
                f"of the {total_cells} (timestamp, symbol) cell(s) — no bar "
                f"exists anywhere in this panel. That is NOT the dense "
                f"panel's cartesian product (D-06); it is an empty ingest: an "
                f"empty vendor response, a mis-parsed file, or a fully-failed "
                f"backfill. Required columns: {list(required_columns)}. The "
                f"structural mask is disabled for this panel, so the "
                f"per-column counts below are raw whole-column null counts. "
                f"Not raising, per flag-don't-delete philosophy (D-07)."
            )
        elif structural_cells > 0 and total_cells > 0:
            logger.info(
                f"validate_schema: {structural_cells}/{total_cells} "
                f"({structural_cells / total_cells:.1%}) "
                f"(timestamp, symbol) cell(s) hold no bar at all — null in "
                f"every required column. That is the dense panel's cartesian "
                f"product (D-06), not an anomaly; nulls on those cells are "
                f"excluded from the counts below."
            )

    for col in data.data_vars:
        column = data[col]
        if empty_ingest:
            # The mask is True on every cell here, so `~mask` is False on
            # every cell and the discriminating branch below would report 0
            # for every column — the exact silence BL-01 describes. Fall back
            # to raw counts: with no bar anywhere there is no bar-exists grid
            # left to discriminate against.
            null_count = int(column.isnull().sum().item())
            scope = (
                "raw null value(s) — whole-column count, because the "
                "structural mask is disabled on an empty-ingest panel (see "
                "the ERROR above)"
            )
        elif structural_mask is not None and tuple(column.dims) == tuple(
            structural_mask.dims
        ):
            null_count = int(
                (column.isnull() & ~structural_mask).sum().item()
            )
            scope = (
                "null value(s) on (timestamp, symbol) cells where a bar DOES "
                "exist"
            )
        else:
            null_count = int(column.isnull().sum().item())
            scope = (
                f"null value(s) — its dims {tuple(column.dims)} differ from "
                f"the required-column grid, so the whole column is counted"
            )
        if null_count > 0:
            logger.warning(
                f"validate_schema: column '{col}' has {null_count} {scope} "
                f"— not raising, per flag-don't-delete philosophy (D-07)."
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


def clean_membership_panel(data: xr.Dataset) -> xr.Dataset:
    """Membership-panel counterpart to `clean_market_data()`, called from
    `base/constituent.py:IndexConstituentDataset._clean()`.

    Validates the panel's shape and returns it unchanged. It never modifies,
    fills or re-sorts anything -- a panel that violates the contract is a bug
    in the densification, and silently repairing it here would hide that.

    Raises `ValueError` naming the specific violation for each of:

    - `data_vars` is not exactly `{"is_member"}`;
    - `is_member.dtype` is not `bool`;
    - `is_member.dims` is not exactly `("timestamp", "symbol")`;
    - the `timestamp` coordinate is not strictly increasing (which includes
      duplicates) -- `XrBackend.filter_by_date` slices with
      `.sel(timestamp=slice(...))`, which silently returns wrong results on an
      unsorted index rather than raising.

    **Why the inherited default is unusable here, not merely unnecessary.**
    `clean_market_data()` calls `validate_schema()`, which hard-raises on the
    five missing OHLCV columns (`open`/`high`/`low`/`close`/`volume`) that a
    membership panel does not and cannot have. Even past that,
    `flag_anomalies()` would append a second, meaningless all-False
    `anomaly_flag` variable to a panel whose only variable is already a
    boolean -- doubling its on-disk size to record nothing.
    """
    variables = set(data.data_vars)
    if variables != {"is_member"}:
        raise ValueError(
            f"clean_membership_panel: expected exactly one data variable "
            f"'is_member', got {sorted(variables)}"
        )

    is_member = data["is_member"]
    if is_member.dtype != np.dtype(bool):
        raise ValueError(
            f"clean_membership_panel: 'is_member' must have dtype bool, got "
            f"{is_member.dtype}"
        )
    if tuple(is_member.dims) != ("timestamp", "symbol"):
        raise ValueError(
            f"clean_membership_panel: 'is_member' must have dims "
            f"('timestamp', 'symbol'), got {tuple(is_member.dims)}"
        )

    timestamps = data["timestamp"].values
    # Compared elementwise rather than via `np.diff(...) > 0` so the check
    # stays dtype-agnostic (a bare `0` against a timedelta64 is deprecated).
    if timestamps.size > 1 and not np.all(timestamps[:-1] < timestamps[1:]):
        raise ValueError(
            "clean_membership_panel: the 'timestamp' coordinate must be "
            "strictly increasing (no duplicates, no out-of-order rows) -- "
            "XrBackend.filter_by_date slices it with .sel(slice(...)), which "
            "returns wrong results silently on an unsorted index."
        )

    return data


#: The exact variable set of an NBBO bar panel (phase 03.9, D-11/D-12/D-19),
#: every one float64. `n_updates` and `n_ambiguous_ties` are counts but are
#: stored as float64 from the start: `BaseDataset._pin_append_dtypes` promotes
#: integer variables on append anyway, and a count that is NaN for a symbol
#: added later (widen) cannot be an integer.
NBBO_PANEL_VARIABLES = (
    "bid",
    "ask",
    "bid_size",
    "ask_size",
    "mid",
    "spread",
    "spread_bps",
    "imbalance",
    "n_updates",
    "tw_spread",
    "tw_bid_size",
    "tw_ask_size",
    "n_ambiguous_ties",
)


def clean_nbbo_panel(data: xr.Dataset) -> xr.Dataset:
    """NBBO-panel counterpart to `clean_market_data()`, called from
    `dataset/nbbo.py:NbboPanelDataset._clean()` (D-14).

    Validates the panel's shape and returns it unchanged. Like
    `clean_membership_panel`, it never modifies, fills or re-sorts anything --
    a panel that violates the contract is a bug in the resampler, and silently
    repairing it here would hide that.

    Raises `ValueError` naming the specific violation for each of:

    - `data_vars` is not exactly `NBBO_PANEL_VARIABLES`;
    - a variable's dtype is not float64;
    - a variable's dims are not exactly `("timestamp", "symbol")`;
    - the `timestamp` coordinate is not strictly increasing.

    **No `anomaly_flag`, no OHLCV check, and no filling of any kind.** The
    inherited `clean_market_data()` hard-raises on the five OHLCV columns an
    NBBO panel does not have. Carry-forward over empty bars is the RESAMPLER's
    state semantics (the prevailing NBBO stays in force until replaced), never
    a cleaning step -- this module's no-fill rule stands.
    """
    variables = set(data.data_vars)
    expected = set(NBBO_PANEL_VARIABLES)
    if variables != expected:
        raise ValueError(
            f"clean_nbbo_panel: expected exactly the variables "
            f"{sorted(expected)}, got {sorted(variables)} (missing "
            f"{sorted(expected - variables)}, unexpected "
            f"{sorted(variables - expected)})"
        )

    for name in NBBO_PANEL_VARIABLES:
        variable = data[name]
        if variable.dtype != np.dtype("float64"):
            raise ValueError(
                f"clean_nbbo_panel: {name!r} must have dtype float64, got "
                f"{variable.dtype}"
            )
        if tuple(variable.dims) != ("timestamp", "symbol"):
            raise ValueError(
                f"clean_nbbo_panel: {name!r} must have dims "
                f"('timestamp', 'symbol'), got {tuple(variable.dims)}"
            )

    timestamps = data["timestamp"].values
    # Elementwise, as in `clean_membership_panel`, to stay dtype-agnostic.
    if timestamps.size > 1 and not np.all(timestamps[:-1] < timestamps[1:]):
        raise ValueError(
            "clean_nbbo_panel: the 'timestamp' coordinate must be strictly "
            "increasing (no duplicates, no out-of-order rows) -- "
            "XrBackend.filter_by_date slices it with .sel(slice(...)), which "
            "returns wrong results silently on an unsorted index."
        )

    return data
