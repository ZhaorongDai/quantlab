"""A store holds each 1-D index coordinate in one chunk (#232).

Zarr fixes an array's chunk grid at creation. xarray writes a dimension
coordinate with no encoding as one chunk of the creating write's length, so a
store created from a one-bar window kept ``timestamp`` in one-element chunks
for ever after: 7233 chunks on the Sharadar stores, ~3 s per open. These
tests pin that every append keeps ``timestamp`` (and ``symbol``) in a single
chunk, and that ``rechunk_index_coordinates`` repairs an existing store in
place without changing a value, an attribute or a data fingerprint.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr

from quantlab.backend.zarr import XrBackend, rechunk_index_coordinates
from quantlab.runs.record import _dataset_fingerprint

_DATES = pd.date_range("2024-01-01", periods=10)


def _panel(symbols=("A", "B")) -> xr.Dataset:
    values = np.arange(len(_DATES) * len(symbols), dtype="float64")
    return xr.Dataset(
        {
            "close": (
                ("timestamp", "symbol"),
                values.reshape(len(_DATES), len(symbols)),
            ),
            "volume": (
                ("timestamp", "symbol"),
                values.reshape(len(_DATES), len(symbols)).astype("int64"),
            ),
        },
        coords={
            "timestamp": _DATES,
            "symbol": np.array(list(symbols), dtype=object),
        },
        attrs={"source": "test"},
    )


def _chunks(path: Path) -> dict:
    group = zarr.open_group(str(path), mode="r", use_consolidated=False)
    return {name: tuple(array.chunks) for name, array in group.arrays()}


def _consolidated_chunks(path: Path) -> dict:
    """Chunk shapes as the consolidated metadata records them."""
    group = zarr.open_group(str(path), mode="r", use_consolidated=True)
    return {name: tuple(array.chunks) for name, array in group.arrays()}


def _append_windows(path: Path, cuts=(1, 4, 7)) -> None:
    panel = _panel()
    edges = [0, *cuts, len(_DATES)]
    for low, high in zip(edges[:-1], edges[1:]):
        XrBackend().to_internal(
            panel.isel(timestamp=slice(low, high))
        ).widen_and_append(str(path), append_dim_size=len(_DATES))


def test_appending_windows_keeps_the_index_coordinates_in_one_chunk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A store created from a one-bar window still ends on one coordinate chunk."""
    monkeypatch.setattr(XrBackend, "APPEND_DIM_CHUNK", 4)
    path = tmp_path / "panel.zarr"
    _append_windows(path)

    chunks = _chunks(path)
    assert chunks["timestamp"] == (len(_DATES),)
    assert chunks["symbol"] == (2,)
    # The data variables keep their pinned grid.
    assert chunks["close"] == (4, 2)
    assert chunks["volume"] == (4, 2)
    # Consolidated metadata, which xarray opens by default, agrees.
    assert _consolidated_chunks(path) == chunks
    xr.testing.assert_identical(xr.open_zarr(path).load(), _panel())


def test_a_block_by_block_symbol_widen_keeps_the_timestamp_in_one_chunk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The chunked widen rewrites the store by appending blocks to a sidecar."""
    monkeypatch.setattr(XrBackend, "APPEND_DIM_CHUNK", 2)
    monkeypatch.setattr(XrBackend, "MAX_WIDEN_BYTES", 0)
    path = tmp_path / "panel.zarr"
    XrBackend().to_internal(
        _panel(("A",)).isel(timestamp=slice(0, 6))
    ).append(str(path), append_dim_size=len(_DATES))
    XrBackend().to_internal(
        _panel(("A", "B")).isel(timestamp=slice(6, 10))
    ).widen_and_append(
        str(path), append_dim_size=len(_DATES), fill_values={"volume": 0}
    )

    chunks = _chunks(path)
    assert chunks["timestamp"] == (len(_DATES),)
    assert chunks["symbol"] == (2,)
    assert _consolidated_chunks(path) == chunks


def _fragmented_store(path: Path, zarr_format: int = 3) -> None:
    """A store built the way the Sharadar stores were: one bar per write."""
    panel = _panel()
    for index in range(len(_DATES)):
        window = panel.isel(timestamp=slice(index, index + 1))
        if index == 0:
            window.to_zarr(path, mode="w", zarr_format=zarr_format)
        else:
            window.to_zarr(path, mode="a", append_dim="timestamp")
    assert _chunks(path)["timestamp"] == (1,)


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_rechunk_repairs_an_existing_store_without_changing_a_value(
    tmp_path: Path, zarr_format: int
) -> None:
    path = tmp_path / "panel.zarr"
    _fragmented_store(path, zarr_format)
    before = xr.open_zarr(path).load()
    before_attrs = {
        name: dict(array.attrs)
        for name, array in zarr.open_group(
            str(path), mode="r", use_consolidated=False
        ).arrays()
    }
    before_fingerprint = _dataset_fingerprint(before, ["close", "volume"])
    data_files = sorted(
        str(p.relative_to(path))
        for name in ("close", "volume")
        for p in (path / name).rglob("*")
    )
    data_bytes = [(path / f).read_bytes() for f in data_files if (path / f).is_file()]

    assert rechunk_index_coordinates(path) == ["timestamp"]

    chunks = _chunks(path)
    assert chunks["timestamp"] == (len(_DATES),)
    assert _consolidated_chunks(path) == chunks
    after = xr.open_zarr(path).load()
    xr.testing.assert_identical(after, before)
    assert after["timestamp"].encoding["units"] == before["timestamp"].encoding["units"]
    assert _dataset_fingerprint(after, ["close", "volume"]) == before_fingerprint
    group = zarr.open_group(str(path), mode="r", use_consolidated=False)
    assert dict(group.attrs) == {"source": "test"}
    assert {name: dict(a.attrs) for name, a in group.arrays()} == before_attrs
    # The data variables were not touched at all.
    assert data_bytes == [
        (path / f).read_bytes() for f in data_files if (path / f).is_file()
    ]
    # No sidecar left behind.
    assert sorted(p.name for p in tmp_path.iterdir()) == ["panel.zarr"]


def test_rechunk_is_a_no_op_on_a_store_already_in_one_chunk(
    tmp_path: Path,
) -> None:
    path = tmp_path / "panel.zarr"
    _panel().to_zarr(path, mode="w")
    stamp = (path / "timestamp" / "zarr.json").read_bytes()

    assert rechunk_index_coordinates(path) == []
    assert (path / "timestamp" / "zarr.json").read_bytes() == stamp


def test_rechunk_keeps_a_store_without_consolidated_metadata_unconsolidated(
    tmp_path: Path,
) -> None:
    path = tmp_path / "panel.zarr"
    panel = _panel()
    panel.isel(timestamp=slice(0, 1)).to_zarr(path, mode="w", consolidated=False)
    panel.isel(timestamp=slice(1, None)).to_zarr(
        path, mode="a", append_dim="timestamp", consolidated=False
    )

    assert rechunk_index_coordinates(path) == ["timestamp"]
    root = json.loads((path / "zarr.json").read_text())
    assert root.get("consolidated_metadata") is None
    xr.testing.assert_identical(xr.open_zarr(path, consolidated=False).load(), panel)


def test_rechunk_refuses_a_missing_store(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        rechunk_index_coordinates(tmp_path / "absent.zarr")
