"""Cleaning and validation rules shared by every dataset class.

The raw-market-data path calls two of the functions here at two different
stages. ``dedup_raw_frame`` runs on the tabular (polars) frame before it is
converted to xarray, because a duplicated ``(timestamp, symbol)`` pair makes
that conversion fail. ``clean_market_data`` runs once on the resulting
``xarray.Dataset`` and only validates the schema and flags anomalies; the NaN
gaps of a dense panel are produced by the conversion itself and pass through
untouched. ``clean_membership_panel`` and ``clean_nbbo_panel`` are the
equivalent validators for an index-membership panel and an NBBO quote panel,
neither of which carries OHLCV columns.

Nothing in this module fills, interpolates or corrects a value: a gap or an
outlier is reported, never repaired, so the pipeline never contains data that
was not observed. The module imports only third-party libraries.
"""

from typing import Literal

import numpy as np
import polars as pl
import xarray as xr
from loguru import logger

REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")

# Price-like variables checked for zero/negative values and extreme jumps,
# including the adjusted-price variants some vendors supply.
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

# A single-step percentage change in ``close`` beyond this fraction is
# flagged as an extreme jump.
_EXTREME_JUMP_THRESHOLD = 0.5


def dedup_raw_frame(
    data: pl.LazyFrame, keep: Literal["first", "last"] = "last"
) -> pl.LazyFrame:
    """Drop duplicate ``(timestamp, symbol)`` rows deterministically.

    Must run before the frame is converted to xarray, which rejects a
    non-unique index. ``keep="last"`` is the default because in a vendor's
    periodic-drop workflow a later file more often carries corrected data
    than an earlier one.

    Args:
        data: A long-format frame with ``timestamp`` and ``symbol`` columns.
        keep: Which duplicate to retain, ``"first"`` or ``"last"``.

    Returns:
        The frame with at most one row per ``(timestamp, symbol)`` pair.

    Example:
        >>> frame = pl.DataFrame({
        ...     "timestamp": ["2024-01-02", "2024-01-02"],
        ...     "symbol": ["AAA", "AAA"],
        ...     "close": [10.0, 10.5],
        ... }).lazy()
        >>> dedup_raw_frame(frame).collect()["close"].to_list()
        [10.5]
        >>> dedup_raw_frame(frame, keep="first").collect()["close"].to_list()
        [10.0]
    """
    return data.unique(subset=["timestamp", "symbol"], keep=keep)


def flag_anomalies(data: xr.Dataset) -> xr.Dataset:
    """Add a boolean ``anomaly_flag`` variable marking suspicious prices.

    A cell is flagged when any present price-like variable is zero or
    negative, or when ``close`` moves by more than ``_EXTREME_JUMP_THRESHOLD``
    from a strictly positive prior close. The underlying values are never
    changed or dropped, so an anomaly stays visible for later investigation.
    A warning with the flagged count is logged when the count is non-zero.

    Args:
        data: A panel on ``(timestamp, symbol)``.

    Returns:
        ``data`` with an additional ``anomaly_flag`` variable on the same
        dimensions.

    Example:
        ``BBB`` prints a zero close and ``AAA`` jumps from 10.5 to 20.0:

        >>> panel["close"].values
        array([[10. ,  5. ],
               [10.5,  0. ],
               [20. ,  5.2],
               [20.5,  5.3]])
        >>> flag_anomalies(panel)["anomaly_flag"].values
        array([[False, False],
               [False,  True],
               [ True, False],
               [False, False]])
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
        # A prior close that is zero, negative or NaN makes the percentage
        # change meaningless, so only compare against a strictly positive
        # prior. Comparisons against NaN are False elementwise.
        valid_prior = shifted > 0
        jump = (np.abs(pct_change) > _EXTREME_JUMP_THRESHOLD) & valid_prior
        # ``diff`` drops the first timestamp; reindex onto the full axis and
        # treat that first row as not-a-jump.
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
    """Check that required columns exist and report unexpected nulls.

    A dense panel is the cartesian product of its axes, so a symbol that did
    not trade in a period is null in every variable at that cell. Those cells
    are structural, not anomalous, and are excluded from the null counts: a
    cell counts as "no bar" when every column in ``required_columns`` is null
    there, and only nulls on cells where a bar does exist are warned about. A
    column laid out on other dimensions than the required columns is counted
    whole instead.

    If every required column is null on every cell, the panel is treated as
    an empty ingest: that is reported at error level, the structural mask is
    disabled and each column's raw null count is reported instead. No null
    condition raises; only a missing column does, because a data-content
    problem must not abort an ingest that is running unattended.

    Args:
        data: A panel on ``(timestamp, symbol)``.
        required_columns: The variables that must be present. Callers whose
            columns use another spelling pass their own tuple, and the
            structural mask is built from that argument.

    Returns:
        ``data``, unchanged.

    Raises:
        ValueError: If any of ``required_columns`` is missing.

    Example:
        >>> validate_schema(ohlcv) is ohlcv
        True
        >>> validate_schema(ohlcv.drop_vars("volume"))
        Traceback (most recent call last):
        ValueError: validate_schema: required column(s) missing from dataset: ['volume']
        >>> validate_schema(quotes, required_columns=("bid", "ask")) is quotes
        True
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
            # With the mask True everywhere, the discriminating branch below
            # would report zero for every column; fall back to raw counts.
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
    """Validate the schema of a market panel and flag its anomalies.

    This is the single cleaning step applied to every market-data panel after
    it has been converted to xarray. Deduplication has already happened on
    the tabular frame, and the NaN gaps of the dense panel come from the
    conversion itself, so nothing here fills or alters a value.

    Args:
        data: A panel on ``(timestamp, symbol)`` carrying the OHLCV columns.

    Returns:
        ``data`` with an ``anomaly_flag`` variable added.

    Example:
        >>> cleaned = clean_market_data(ohlcv)
        >>> list(cleaned.data_vars)
        ['open', 'high', 'low', 'close', 'volume', 'anomaly_flag']
        >>> cleaned["anomaly_flag"].dtype
        dtype('bool')
    """
    data = validate_schema(data)
    data = flag_anomalies(data)
    return data


def clean_membership_panel(data: xr.Dataset) -> xr.Dataset:
    """Validate an index-membership panel and return it unchanged.

    The OHLCV cleaning path cannot be reused here: ``validate_schema`` would
    raise on the missing price columns, and ``flag_anomalies`` would add a
    meaningless all-False flag beside the panel's only variable. Nothing is
    modified, filled or re-sorted; a panel that breaks the contract is a bug
    upstream, and repairing it here would hide that.

    Args:
        data: A panel whose only variable is a boolean ``is_member`` on
            ``(timestamp, symbol)``.

    Returns:
        ``data``, unchanged.

    Raises:
        ValueError: If the variables are not exactly ``{"is_member"}``, if
            ``is_member`` is not boolean or not on ``("timestamp", "symbol")``,
            or if the ``timestamp`` coordinate is not strictly increasing.
            Label-based date slicing silently returns wrong results on an
            unsorted index, which is why the last case is refused.

    Example:
        >>> clean_membership_panel(membership) is membership
        True
        >>> clean_membership_panel(membership.astype(int))
        Traceback (most recent call last):
        ValueError: clean_membership_panel: 'is_member' must have dtype bool, got int64
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
    # Compared elementwise rather than via ``np.diff(...) > 0`` so the check
    # stays dtype-agnostic (a bare ``0`` against a timedelta64 is deprecated).
    if timestamps.size > 1 and not np.all(timestamps[:-1] < timestamps[1:]):
        raise ValueError(
            "clean_membership_panel: the 'timestamp' coordinate must be "
            "strictly increasing (no duplicates, no out-of-order rows) -- "
            "XrBackend.filter_by_date slices it with .sel(slice(...)), which "
            "returns wrong results silently on an unsorted index."
        )

    return data


#: The exact variable set of an NBBO bar panel, every one float64. The two
#: count variables, ``n_updates`` and ``n_ambiguous_ties``, are float64 as
#: well: integer variables are promoted on append anyway, and a count that is
#: NaN for a symbol added later cannot be an integer.
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
    """Validate an NBBO quote-bar panel and return it unchanged.

    Like ``clean_membership_panel``, this never modifies, fills or re-sorts.
    The OHLCV path does not apply because the panel has no price columns, and
    no filling of any kind is done here: carrying the prevailing quote forward
    over an empty bar is the resampler's responsibility, not a cleaning step.

    Args:
        data: A panel whose variables are exactly ``NBBO_PANEL_VARIABLES``.

    Returns:
        ``data``, unchanged.

    Raises:
        ValueError: If the variable set differs from ``NBBO_PANEL_VARIABLES``,
            if any variable is not float64 or not on ``("timestamp",
            "symbol")``, or if the ``timestamp`` coordinate is not strictly
            increasing.

    Example:
        >>> clean_nbbo_panel(quotes) is quotes
        True
        >>> clean_nbbo_panel(quotes.drop_vars("mid"))
        Traceback (most recent call last):
        ValueError: clean_nbbo_panel: expected exactly the variables ['ask', ...
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
    # Elementwise, as in ``clean_membership_panel``, to stay dtype-agnostic.
    if timestamps.size > 1 and not np.all(timestamps[:-1] < timestamps[1:]):
        raise ValueError(
            "clean_nbbo_panel: the 'timestamp' coordinate must be strictly "
            "increasing (no duplicates, no out-of-order rows) -- "
            "XrBackend.filter_by_date slices it with .sel(slice(...)), which "
            "returns wrong results silently on an unsorted index."
        )

    return data
