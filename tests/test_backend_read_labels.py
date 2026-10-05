"""`XrBackend.read` hands back string labels as object strings, as a frame gives them.

Zarr reads a `str` coordinate back as numpy's variable-width `StringDType`, which does
not cast to the fixed-width `str` dtype the backtester aligns weights with. Every store
read goes through `XrBackend.read`, so the labels are normalised there, once.
"""

import numpy as np
import pandas as pd
import xarray as xr

from quantlab.backend.zarr import XrBackend


def test_string_labels_come_back_as_object_strings(tmp_path):
    panel = xr.Dataset(
        {"close": (("timestamp", "symbol"), np.ones((2, 2)))},
        coords={
            "timestamp": pd.date_range("2024-01-02", periods=2),
            "symbol": np.array(["AAA", "BBB"], dtype=object),
        },
    )
    XrBackend().to_internal(panel).write(str(tmp_path / "p.zarr"))

    read = XrBackend().read(tmp_path / "p.zarr").data

    assert read["symbol"].dtype == object
    assert read["symbol"].values.tolist() == ["AAA", "BBB"]
    assert read["symbol"].values.astype(str).tolist() == ["AAA", "BBB"]


def test_integer_labels_are_left_alone(tmp_path):
    panel = xr.Dataset(
        {"close": (("timestamp", "symbol"), np.ones((1, 2)))},
        coords={"timestamp": pd.date_range("2024-01-02", periods=1), "symbol": [10, 20]},
    )
    XrBackend().to_internal(panel).write(str(tmp_path / "p.zarr"))

    assert XrBackend().read(tmp_path / "p.zarr").data["symbol"].dtype.kind == "i"
