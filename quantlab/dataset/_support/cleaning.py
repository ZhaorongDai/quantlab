"""Cleaning and validation rules shared by every dataset class.

A *panel* is an ``xarray.Dataset`` indexed by ``timestamp`` and ``symbol``.
It is *dense*: it has a cell for every timestamp and symbol pair, and a
symbol that did not trade at a timestamp is NaN there. OHLCV means the open,
high, low, close and volume columns of a price bar.

Raw market data passes through this module at two stages.
``dedup_raw_frame`` runs on the long-format polars frame before it is
converted to xarray, because a duplicated ``(timestamp, symbol)`` pair makes
that conversion fail. ``clean_market_data`` runs once on the resulting panel;
it only checks the schema and flags anomalies, and leaves the NaN gaps of the
dense panel alone. ``clean_membership_panel`` and ``clean_nbbo_panel`` are the
matching validators for an index-membership panel and for an NBBO quote
panel (NBBO, the National Best Bid and Offer, is the best bid and ask across
all US exchanges). Neither of those has OHLCV columns.

Nothing here fills, interpolates or corrects a value. A gap or an outlier is
reported, never repaired, so the pipeline only ever holds observed data.
The module imports only third-party libraries.
"""

from typing import Literal

import numpy as np
import polars as pl
import xarray as xr
from loguru import logger

REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")

# Price variables checked for zero or negative values and extreme jumps,
# including the split- and dividend-adjusted versions some vendors supply.
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
    """Drop duplicate ``(timestamp, symbol)`` rows, keeping a fixed one.

    This must run before the frame is converted to xarray, which rejects a
    non-unique index. The default ``keep="last"`` is chosen because when a
    vendor delivers files periodically, a later file is more likely to hold
    corrected data than an earlier one.

    Parameters
    ----------
    data : pl.LazyFrame
        A long-format frame with ``timestamp`` and ``symbol`` columns.
    keep : {"first", "last"}, default "last"
        Which duplicate to keep.

    Returns
    -------
    pl.LazyFrame
        The frame with at most one row per ``(timestamp, symbol)`` pair.

    Examples
    --------
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

    A cell is flagged when any price variable present is zero or negative,
    or when ``close`` changes by more than ``_EXTREME_JUMP_THRESHOLD`` (50%)
    from a strictly positive previous close. The values themselves are never
    changed or dropped, so an anomaly stays visible for later investigation.
    If any cell is flagged, a warning with the count is logged.

    Parameters
    ----------
    data : xr.Dataset
        A panel on ``(timestamp, symbol)``.

    Returns
    -------
    xr.Dataset
        ``data`` with an additional ``anomaly_flag`` variable on the same
        dimensions.

    Examples
    --------
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
        # A percentage change from a zero, negative or NaN close is
        # meaningless, so only a strictly positive previous close counts.
        # Comparisons with NaN are False.
        valid_prior = shifted > 0
        jump = (np.abs(pct_change) > _EXTREME_JUMP_THRESHOLD) & valid_prior
        # ``diff`` drops the first timestamp; put it back as "no jump".
        jump = jump.reindex(timestamp=data["timestamp"], fill_value=False)
        anomaly = anomaly | jump

    anomaly = anomaly.astype(bool)
    anomaly.name = "anomaly_flag"

    flagged_count = int(anomaly.sum().item())
    if flagged_count > 0:
        logger.warning(
            f"flag_anomalies: flagged {flagged_count} anomalous "
            f"(timestamp, symbol) data point(s) (zero/negative price or "
            f"extreme jump); values left unmodified, see `anomaly_flag`."
        )

    return data.assign(anomaly_flag=anomaly)


def validate_schema(
    data: xr.Dataset, required_columns: tuple[str, ...] = REQUIRED_COLUMNS
) -> xr.Dataset:
    """Check that the required columns exist and report unexpected nulls.

    A dense panel has a cell for every timestamp and symbol, so a symbol that
    did not trade at a timestamp is null in every variable there. Such cells
    are expected and are left out of the null counts. A cell counts as "no
    bar" when every column in ``required_columns`` is null there, and only
    nulls on cells that do have a bar are warned about. A column laid out on
    other dimensions than the required columns is counted in full instead.

    If every required column is null in every cell, the panel is treated as
    an empty ingest (for example an empty vendor response). That is logged
    as an error, the "no bar" exclusion is switched off, and each column's
    full null count is reported. Nulls never raise; only a missing column
    does, because a data problem must not abort an ingest running unattended.

    Parameters
    ----------
    data : xr.Dataset
        A panel on ``(timestamp, symbol)``.
    required_columns : tuple of str, default ``REQUIRED_COLUMNS``
        The variables that must be present. Callers whose columns are
        spelled differently pass their own tuple; the "no bar" test uses
        these columns too.

    Returns
    -------
    xr.Dataset
        ``data``, unchanged.

    Raises
    ------
    ValueError
        If any of ``required_columns`` is missing.

    Examples
    --------
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
                f"validate_schema: EVERY required column is null on every one "
                f"of the {total_cells} (timestamp, symbol) cell(s); no bar "
                f"exists anywhere in this panel. That is not the normal gaps "
                f"of a dense panel; it is an empty ingest: an empty vendor "
                f"response, a mis-parsed file, or a fully failed backfill. "
                f"Required columns: {list(required_columns)}. Empty cells are "
                f"not excluded for this panel, so the per-column counts below "
                f"are whole-column null counts. Not raising: data problems "
                f"are flagged, never deleted or fatal."
            )
        elif structural_cells > 0 and total_cells > 0:
            logger.info(
                f"validate_schema: {structural_cells}/{total_cells} "
                f"({structural_cells / total_cells:.1%}) "
                f"(timestamp, symbol) cell(s) hold no bar at all (null in "
                f"every required column). Those are the normal gaps of a dense "
                f"panel, not anomalies; nulls on those cells are excluded "
                f"from the counts below."
            )

    for col in data.data_vars:
        column = data[col]
        if empty_ingest:
            # With every cell masked, the next branch would report zero for
            # every column, so fall back to whole-column counts.
            null_count = int(column.isnull().sum().item())
            scope = (
                "raw null value(s), whole-column count, because empty cells "
                "are not excluded on an empty-ingest panel (see the ERROR "
                "above)"
            )
        elif structural_mask is not None and tuple(column.dims) == tuple(
            structural_mask.dims
        ):
            null_count = int(
                (column.isnull() & ~structural_mask).sum().item()
            )
            scope = (
                "null value(s) on (timestamp, symbol) cells where a bar does "
                "exist"
            )
        else:
            null_count = int(column.isnull().sum().item())
            scope = (
                f"null value(s); its dims {tuple(column.dims)} differ from "
                f"the required-column grid, so the whole column is counted"
            )
        if null_count > 0:
            logger.warning(
                f"validate_schema: column '{col}' has {null_count} {scope} "
                f"(not raising: data problems are flagged, never deleted or fatal)."
            )

    return data


def clean_market_data(data: xr.Dataset) -> xr.Dataset:
    """Validate the schema of a market panel and flag its anomalies.

    This is the one cleaning step applied to every market-data panel after
    it has been converted to xarray. Duplicates were already removed from
    the polars frame, and the NaN gaps of the dense panel are expected, so
    nothing here fills or changes a value.

    Parameters
    ----------
    data : xr.Dataset
        A panel on ``(timestamp, symbol)`` carrying the OHLCV columns.

    Returns
    -------
    xr.Dataset
        ``data`` with an ``anomaly_flag`` variable added.

    Examples
    --------
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

    An index-membership panel says, for each day and symbol, whether the
    symbol belonged to the index. The OHLCV checks do not apply:
    ``validate_schema`` would raise on the missing price columns, and
    ``flag_anomalies`` would add a meaningless all-False flag. Nothing is
    modified, filled or re-sorted. A panel that breaks these rules points to
    a bug in the code that built it, and repairing it here would hide that.

    Parameters
    ----------
    data : xr.Dataset
        A panel whose only variable is a boolean ``is_member`` on
        ``(timestamp, symbol)``.

    Returns
    -------
    xr.Dataset
        ``data``, unchanged.

    Raises
    ------
    ValueError
        If the variables are not exactly ``{"is_member"}``, if
        ``is_member`` is not boolean or not on ``("timestamp", "symbol")``,
        or if the ``timestamp`` coordinate is not strictly increasing.
        The last case is refused because slicing by date label silently
        returns wrong results on an unsorted index.

    Examples
    --------
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
    # Compare neighbours directly instead of ``np.diff(...) > 0``, which works
    # for any dtype (comparing a timedelta64 with a bare ``0`` is deprecated).
    if timestamps.size > 1 and not np.all(timestamps[:-1] < timestamps[1:]):
        raise ValueError(
            "clean_membership_panel: the 'timestamp' coordinate must be "
            "strictly increasing (no duplicates, no out-of-order rows), "
            "because XrBackend.filter_by_date slices it with "
            ".sel(slice(...)), which silently returns wrong results on an "
            "unsorted index."
        )

    return data


#: The exact variables of an NBBO bar panel, all float64. The two counts,
#: ``n_updates`` and ``n_ambiguous_ties``, are float64 too: integers are
#: promoted to float when data is appended anyway, and a count must be able
#: to hold NaN for a symbol that joins the panel later.
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

    An NBBO quote-bar panel holds, per bar and symbol, the best bid and ask
    across US exchanges and statistics derived from them. Like
    ``clean_membership_panel``, this never modifies, fills or re-sorts. The
    OHLCV checks do not apply because the panel has no price columns.
    Carrying the last quote forward over a bar with no updates is done by
    the resampler that builds the panel, not here.

    Parameters
    ----------
    data : xr.Dataset
        A panel whose variables are exactly ``NBBO_PANEL_VARIABLES``.

    Returns
    -------
    xr.Dataset
        ``data``, unchanged.

    Raises
    ------
    ValueError
        If the variable set differs from ``NBBO_PANEL_VARIABLES``,
        if any variable is not float64 or not on ``("timestamp",
        "symbol")``, or if the ``timestamp`` coordinate is not strictly
        increasing.

    Examples
    --------
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
    # Compare neighbours directly, as in ``clean_membership_panel``.
    if timestamps.size > 1 and not np.all(timestamps[:-1] < timestamps[1:]):
        raise ValueError(
            "clean_nbbo_panel: the 'timestamp' coordinate must be strictly "
            "increasing (no duplicates, no out-of-order rows), because "
            "XrBackend.filter_by_date slices it with .sel(slice(...)), which "
            "silently returns wrong results on an unsorted index."
        )

    return data
