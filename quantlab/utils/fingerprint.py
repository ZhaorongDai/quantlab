"""Content fingerprint of the data a backtest run actually read (03.7 D-27).

A stored backtest is only reproducible if the data under it has not changed.
Datasets get appended to, and Tiingo re-bases adjusted prices retroactively
after new dividends, so a rebuild from a persisted config can silently compute
over different numbers. `dataset_fingerprint` records, for one dataset, the
time range, the axis sizes and a sha256 over the values of the variables the
run consumed, so a rebuild can compare its own record against the stored one.

**NaN-canonical.** NaN has many bit patterns: two reads of one store, or two
code paths producing "missing", can carry NaNs whose payload bits differ. Such
arrays compare as the same missing values but differ byte for byte, so hashing
raw bytes would report a changed digest on unchanged data. Every NaN is
rewritten to one canonical NaN and every -0.0 to 0.0 before hashing.

A LEAF module: hashlib, numpy, pandas and xarray only, zero project-internal
imports.
"""

import hashlib

import numpy as np
import pandas as pd
import xarray as xr

__all__ = ["dataset_fingerprint"]


def _iso(value) -> str:
    return pd.Timestamp(value).isoformat()


def dataset_fingerprint(ds: xr.Dataset, variables: list[str]) -> dict:
    """Fingerprint `variables` of a `(timestamp, symbol)` dataset.

    The dataset is sorted by timestamp and symbol first, so the digest does not
    depend on axis order. The hash covers, in order: the int64 nanosecond
    timestamps, the NUL-joined symbol names, then for each variable in sorted
    order its name and its float64 values on `(timestamp, symbol)` after NaN
    and signed-zero canonicalization.

    Returns `{"algorithm": "sha256", "digest", "variables" (sorted), "start",
    "end" (ISO strings, None for an empty timestamp axis), "n_timestamps",
    "n_symbols"}`. A variable missing from `ds` raises `ValueError` naming it.
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
        values = values + 0.0  # -0.0 -> 0.0
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
