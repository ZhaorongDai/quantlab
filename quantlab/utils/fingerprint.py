"""Content fingerprint of the data a backtest run read.

A stored backtest is reproducible only if the data under it has not changed,
and datasets do change: stores are appended to and adjusted prices are
re-based retroactively. ``dataset_fingerprint`` records, for one dataset, the
time range, the axis sizes and a sha256 digest over the values of the
variables a run consumed. A rebuild from a persisted config computes the same
record and compares it against the stored one.

NaN has many bit patterns, so every NaN is rewritten to one canonical NaN and
every ``-0.0`` to ``0.0`` before hashing; otherwise two reads of identical data
could report different digests. This module has no project-internal imports.
"""

import hashlib

import numpy as np
import pandas as pd
import xarray as xr

__all__ = ["dataset_fingerprint"]


def _iso(value) -> str:
    """Return ``value`` as an ISO-8601 timestamp string."""
    return pd.Timestamp(value).isoformat()


def dataset_fingerprint(ds: xr.Dataset, variables: list[str]) -> dict:
    """Fingerprint ``variables`` of a ``(timestamp, symbol)`` dataset.

    The dataset is sorted by timestamp and symbol first, so the digest does not
    depend on axis order. The hash covers, in order: the int64 nanosecond
    timestamps, the NUL-joined symbol names, then for each variable in sorted
    order its name and its float64 values on ``(timestamp, symbol)`` after NaN
    and signed-zero canonicalisation.

    Parameters
    ----------
    ds : xr.Dataset
        A panel indexed by ``timestamp`` and ``symbol``.
    variables : list[str]
        Names of the data variables to include.

    Returns
    -------
    dict
        A dict with keys ``algorithm`` (``"sha256"``), ``digest``, ``variables``
        (sorted), ``start`` and ``end`` (ISO strings, ``None`` when the
        timestamp axis is empty), ``n_timestamps`` and ``n_symbols``.

    Raises
    ------
    ValueError
        If any requested variable is not in ``ds``.

    Examples
    --------
    >>> record = dataset_fingerprint(prices, ["open", "close"])
    >>> record["digest"] == stored_record["digest"]
    True
    """
    names = sorted(variables)
    missing = [name for name in names if name not in ds.data_vars]
    if missing:
        raise ValueError(
            f"cannot fingerprint variables {missing}: not in the dataset "
            f"(data variables: {sorted(ds.data_vars)})"
        )

    ds = ds.sortby(["timestamp", "symbol"])
    timestamps = ds["timestamp"].values.astype("datetime64[ns]")
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(timestamps.astype("int64")).tobytes())
    digest.update("\x00".join(map(str, ds["symbol"].values.tolist())).encode())
    for name in names:
        values = np.asarray(
            ds[name].transpose("timestamp", "symbol").values, dtype=np.float64
        )
        values = np.where(np.isnan(values), np.nan, values)  # one NaN bit pattern
        values = values + 0.0  # -0.0 becomes 0.0
        digest.update(name.encode())
        digest.update(np.ascontiguousarray(values).tobytes())

    return {
        "algorithm": "sha256",
        "digest": digest.hexdigest(),
        "variables": names,
        "start": _iso(timestamps[0]) if timestamps.size else None,
        "end": _iso(timestamps[-1]) if timestamps.size else None,
        "n_timestamps": int(ds.sizes["timestamp"]),
        "n_symbols": int(ds.sizes["symbol"]),
    }
