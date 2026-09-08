"""Regression tests for RESEARCH.md Pitfall 1: XrBackend.write() crashed with
FileExistsError the second time it was called against the same Zarr path,
because it never passed `mode` to `xr.Dataset.to_zarr()` (which defaults to
Zarr's `mode="w-"`, i.e. "fail if exists").
"""

from pathlib import Path

import pytest
import xarray as xr

from quantlab.dataset.backend import XrBackend


def _make_dataset(close_values: list[list[float]]) -> xr.Dataset:
    return xr.Dataset(
        {"close": (["timestamp", "symbol"], close_values)},
        coords={
            "timestamp": ["2024-01-01", "2024-01-02"][: len(close_values)],
            "symbol": ["A", "B"],
        },
    )


def test_write_once_succeeds_and_is_readable(tmp_path: Path) -> None:
    path = str(tmp_path / "test.zarr")
    data = _make_dataset([[1.0, 2.0]])

    XrBackend().to_internal(data).write(path)

    readback = xr.open_dataset(path)
    assert readback["close"].values.tolist() == [[1.0, 2.0]]


def test_write_twice_same_path_does_not_raise(tmp_path: Path) -> None:
    path = str(tmp_path / "test.zarr")
    data = _make_dataset([[1.0, 2.0]])

    XrBackend().to_internal(data).write(path)

    # Second write against the SAME path, fresh XrBackend instance, must NOT
    # raise FileExistsError (RESEARCH.md Pitfall 1).
    data2 = _make_dataset([[3.0, 4.0]])
    XrBackend().to_internal(data2).write(path)

    readback = xr.open_dataset(path)
    assert readback["close"].values.tolist() == [[3.0, 4.0]]


def test_explicit_mode_kwarg_is_respected(tmp_path: Path) -> None:
    """An explicit caller-supplied `mode=` must NOT be silently overridden by
    the new default. We prove this by explicitly passing `mode="w-"` (Zarr's
    "fail if exists" mode) on a second write to an existing path: if the fix
    used `kwargs["mode"] = "w"` (unconditional override) instead of
    `kwargs.setdefault("mode", "w")`, this would incorrectly succeed instead
    of raising `FileExistsError`.
    """
    path = str(tmp_path / "test.zarr")
    data = _make_dataset([[1.0, 2.0]])

    XrBackend().to_internal(data).write(path)

    data2 = _make_dataset([[3.0, 4.0]])
    with pytest.raises(FileExistsError):
        XrBackend().to_internal(data2).write(path, mode="w-")
