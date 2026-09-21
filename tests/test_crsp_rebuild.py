"""The offline store-rebuild entry point (phase 03.11 W0).

Two layers are under test here, in the project's usual `base/` ABC +
`dataset/` concrete-class split:

- `quantlab.base.rebuild.BaseStoreRebuilder` -- the framework-agnostic
  four-step skeleton (assert inputs -> backup -> clear -> convert -> measure)
  and its `RebuildMeasurement` carrier.
- `quantlab.dataset.crsp_rebuild.CrspStoreRebuilder` -- the CRSP specifics:
  four sidecars, the offline `registry.convert()` call, seven measurements.

Everything in this file runs on SYNTHETIC fixtures under `tmp_path`, so it is
part of the ordinary regression gate and never touches a real store. The real
`data/` gate lives in `tests/test_crsp_rebuild_measurements.py`, which is
deliberately excluded from the full-suite command because it DELETES and
re-converts the on-disk panel.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.base.rebuild import BaseStoreRebuilder, RebuildMeasurement


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _config(zarr_path: Path) -> SimpleNamespace:
    """The smallest object `BaseStoreRebuilder` reads: one `zarr_file_path`.

    A `SimpleNamespace` on purpose -- the ABC must not depend on any concrete
    config class, and pinning it against a real `DatasetConfig` here would
    hide an accidental attribute dependency behind a passing test.
    """
    return SimpleNamespace(zarr_file_path=str(zarr_path))


def _make_store(zarr_path: Path, suffixes: tuple[str, ...]) -> tuple[Path, ...]:
    """Create a fake store DIRECTORY plus one sidecar file per suffix."""
    zarr_path.mkdir(parents=True, exist_ok=True)
    (zarr_path / ".zgroup").write_text('{"zarr_format": 2}')
    written = []
    for suffix in suffixes:
        sidecar = Path(str(zarr_path) + suffix)
        sidecar.write_text(json.dumps({"stale": True}))
        written.append(sidecar)
    return tuple(written)


class _FakeRebuilder(BaseStoreRebuilder):
    """A minimal concrete subclass: records its call order, measures nothing.

    `calls` is the instrument for the ordering behaviour -- `rebuild()` must
    run assert -> backup -> clear -> `_convert` -> `_measure`, and must not
    reach `_measure` when `_convert` raises.
    """

    SIDECAR_SUFFIXES = (".alpha.json", ".beta.json")

    def __init__(self, config, *, data_root, inputs=(), convert_raises=None):
        super().__init__(config, data_root=data_root)
        self.calls: list[str] = []
        self._inputs = tuple(Path(p) for p in inputs)
        self._convert_raises = convert_raises

    def _required_inputs(self) -> tuple[Path, ...]:
        return self._inputs

    def _convert(self) -> object:
        self.calls.append("convert")
        if self._convert_raises is not None:
            raise self._convert_raises
        return "converted"

    def _measure(self) -> dict[str, int]:
        self.calls.append("measure")
        return {"fake_metric": 7}

    def _measure_dims(self) -> tuple[dict[str, int], int]:
        return {"timestamp": 3, "symbol": 2}, 5


# ---------------------------------------------------------------------------
# Task 1 -- BaseStoreRebuilder
# ---------------------------------------------------------------------------


def test_data_root_must_be_an_existing_directory(tmp_path: Path):
    """No cwd fallback, no default: a bad `data_root` is refused immediately."""
    missing = tmp_path / "not-a-directory"
    with pytest.raises(ValueError) as excinfo:
        _FakeRebuilder(_config(tmp_path / "x.zarr"), data_root=missing)
    assert "_FakeRebuilder" in str(excinfo.value)


def test_data_root_error_names_the_resolved_path_and_the_git_command(
    tmp_path: Path,
):
    """The message must be actionable in a worktree, where `data/` is absent.

    The recorded accident (03.11 T-03.11-03) is an executor reading the WRONG
    tree and reporting a false green, so the refusal names both the path it
    resolved and the command that produces the right one.
    """
    missing = tmp_path / "nope"
    with pytest.raises(ValueError) as excinfo:
        _FakeRebuilder(_config(tmp_path / "x.zarr"), data_root=missing)
    message = str(excinfo.value)
    assert str(missing.resolve()) in message
    assert "--git-common-dir" in message


def test_assert_inputs_present_names_every_missing_absolute_path(
    tmp_path: Path,
):
    present = tmp_path / "present"
    present.mkdir()
    absent_a = tmp_path / "absent_a"
    absent_b = tmp_path / "absent_b"
    rebuilder = _FakeRebuilder(
        _config(tmp_path / "x.zarr"),
        data_root=tmp_path,
        inputs=(present, absent_a, absent_b),
    )
    with pytest.raises(FileNotFoundError) as excinfo:
        rebuilder.assert_inputs_present()
    message = str(excinfo.value)
    assert str(absent_a) in message
    assert str(absent_b) in message
    assert str(present) not in message
    assert message.startswith("_FakeRebuilder: refusing")


def test_assert_inputs_present_is_silent_when_every_input_exists(
    tmp_path: Path,
):
    one = tmp_path / "one"
    one.mkdir()
    rebuilder = _FakeRebuilder(
        _config(tmp_path / "x.zarr"), data_root=tmp_path, inputs=(one,)
    )
    assert rebuilder.assert_inputs_present() is None


def test_backup_returns_none_when_the_store_does_not_exist(tmp_path: Path):
    rebuilder = _FakeRebuilder(
        _config(tmp_path / "x.zarr"), data_root=tmp_path
    )
    assert rebuilder.backup(tmp_path / "backup") is None


def test_backup_copies_the_store_and_every_sidecar(tmp_path: Path):
    store = tmp_path / "x.zarr"
    _make_store(store, _FakeRebuilder.SIDECAR_SUFFIXES)
    rebuilder = _FakeRebuilder(_config(store), data_root=tmp_path)

    dest = tmp_path / "backup"
    returned = rebuilder.backup(dest)

    assert returned == str(dest)
    assert (dest / "x.zarr" / ".zgroup").exists()
    for suffix in _FakeRebuilder.SIDECAR_SUFFIXES:
        assert (dest / f"x.zarr{suffix}").exists()


def test_clear_removes_the_store_and_every_existing_sidecar(tmp_path: Path):
    store = tmp_path / "x.zarr"
    sidecars = _make_store(store, _FakeRebuilder.SIDECAR_SUFFIXES)
    rebuilder = _FakeRebuilder(_config(store), data_root=tmp_path)

    removed = rebuilder.clear()

    assert not store.exists()
    for sidecar in sidecars:
        assert not sidecar.exists()
    assert removed == tuple(
        sorted([str(store)] + [str(s) for s in sidecars])
    )


def test_clear_is_idempotent_when_nothing_exists(tmp_path: Path):
    """Clearing an absent store is a no-op, not an error.

    `rebuild()` calls `clear()` unconditionally, so a first-ever conversion
    (no store yet) must not blow up on the way to `_convert()`.
    """
    rebuilder = _FakeRebuilder(
        _config(tmp_path / "x.zarr"), data_root=tmp_path
    )
    assert rebuilder.clear() == ()


def test_rebuild_runs_assert_backup_clear_convert_measure_in_order(
    tmp_path: Path,
):
    store = tmp_path / "x.zarr"
    _make_store(store, _FakeRebuilder.SIDECAR_SUFFIXES)
    inputs_dir = tmp_path / "raw"
    inputs_dir.mkdir()
    rebuilder = _FakeRebuilder(
        _config(store), data_root=tmp_path, inputs=(inputs_dir,)
    )

    measurement = rebuilder.rebuild(backup_dir=tmp_path / "backup")

    assert rebuilder.calls == ["convert", "measure"]
    # backup ran BEFORE clear: the copy exists even though the original is gone
    assert (tmp_path / "backup" / "x.zarr" / ".zgroup").exists()
    assert measurement.backup_path == str(tmp_path / "backup")
    assert measurement.metrics == {"fake_metric": 7}
    assert measurement.dims == {"timestamp": 3, "symbol": 2}
    assert measurement.data_var_count == 5
    assert str(store) in measurement.removed


def test_rebuild_does_not_measure_when_convert_raises(tmp_path: Path):
    store = tmp_path / "x.zarr"
    _make_store(store, _FakeRebuilder.SIDECAR_SUFFIXES)
    inputs_dir = tmp_path / "raw"
    inputs_dir.mkdir()
    boom = RuntimeError("conversion blew up")
    rebuilder = _FakeRebuilder(
        _config(store),
        data_root=tmp_path,
        inputs=(inputs_dir,),
        convert_raises=boom,
    )

    with pytest.raises(RuntimeError, match="conversion blew up"):
        rebuilder.rebuild(backup_dir=tmp_path / "backup")

    assert rebuilder.calls == ["convert"]


def test_rebuild_measurement_data_root_is_an_absolute_path_string(
    tmp_path: Path,
):
    """`RebuildMeasurement` is the record the SUMMARY quotes verbatim.

    `data_root` being an ABSOLUTE string is the whole point: printing it is
    how an operator sees WHICH tree was read, which is the only defence
    against the worktree false-green (T-03.11-03).
    """
    store = tmp_path / "x.zarr"
    rebuilder = _FakeRebuilder(_config(store), data_root=tmp_path)

    measurement = rebuilder.rebuild()

    assert isinstance(measurement, RebuildMeasurement)
    assert measurement.data_root == str(tmp_path.resolve())
    assert Path(measurement.data_root).is_absolute()
    assert measurement.store_path == str(store)
    assert measurement.backup_path is None
