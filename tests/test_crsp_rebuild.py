"""The offline store-rebuild entry point (phase 03.11 W0).

Two layers are under test here, in the project's usual `base/` ABC +
`dataset/` concrete-class split:

- `quantlab.base.rebuild.BaseStoreRebuilder` -- the framework-agnostic
  four-step skeleton (assert inputs -> backup -> clear -> convert -> measure)
  and its `RebuildMeasurement` carrier.
- `quantlab.dataset.crsp.rebuild.CrspStoreRebuilder` -- the CRSP specifics:
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

from quantlab.base.config import CrspDatasetConfig
from quantlab.base.rebuild import BaseStoreRebuilder, RebuildMeasurement
from quantlab.dataset._support.cleaning import REQUIRED_COLUMNS
from quantlab.dataset.crsp.rebuild import (
    CRSP_SIDECAR_SUFFIXES,
    CrspStoreRebuilder,
)


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


# ---------------------------------------------------------------------------
# Task 2 -- CrspStoreRebuilder
# ---------------------------------------------------------------------------


def _crsp_config(root: Path, *, store_name: str = "crsp.zarr") -> CrspDatasetConfig:
    """A CRSP config built DIRECTLY, never through `quantlab/config`.

    The layout mirrors the real tree exactly, including its two traps:
    `raw_data_dir_path` TERMINATES at the `/wrds` vendor segment, and
    `reference_dir` is a SIBLING of `wrds_crsp`, not a child of `wrds`.
    """
    return CrspDatasetConfig(
        zarr_file_path=str(root / "data" / "data" / "us_equity" / "1d" / store_name),
        raw_data_dir_path=str(
            root / "data" / "downloads" / "us_equity" / "1d" / "wrds_crsp" / "wrds"
        ),
        reference_dir=str(
            root
            / "data"
            / "downloads"
            / "us_equity"
            / "1d"
            / "wrds_crsp"
            / "_reference"
        ),
        start_date="2024-01-01",
        end_date="2024-12-31",
        security_filter="equity_common",
        roster_universe="crsp_sp500",
    )


def _write_measurable_panel(store: Path, *, with_anomaly_flag: bool = True) -> None:
    """A 3x2 panel whose seven measurements are all known by construction.

    One cell -- `(t2, s1)` -- is a STRUCTURAL gap: null in every one of
    `cleaning.REQUIRED_COLUMNS`, which is the dense panel's cartesian product
    (D-06), not a defect. `adjClose` is null exactly there (so its null count
    equals the structural count) while `adjVolume` carries one EXTRA null at
    `(t1, s0)` -- deliberately different, so a `_measure()` that conflated the
    two counts could not pass.
    """
    nan = np.nan
    ohlcv = np.array(
        [
            [10.0, 11.0],
            [0.0, 12.0],  # close == 0 at (t1, s0)
            [13.0, nan],  # structural gap at (t2, s1)
        ]
    )
    variables = {name: (["timestamp", "symbol"], ohlcv.copy()) for name in REQUIRED_COLUMNS}
    variables["adjClose"] = (
        ["timestamp", "symbol"],
        np.array([[10.0, -1.0], [9.0, 12.0], [13.0, nan]]),
    )
    variables["adjVolume"] = (
        ["timestamp", "symbol"],
        np.array([[100.0, 200.0], [nan, 300.0], [400.0, nan]]),
    )
    if with_anomaly_flag:
        variables["anomaly_flag"] = (
            ["timestamp", "symbol"],
            np.array([[False, True], [True, False], [False, False]]),
        )

    panel = xr.Dataset(
        variables,
        coords={
            "timestamp": pd.date_range("2024-01-02", periods=3, freq="D"),
            "symbol": [10107, 14593],
        },
    )
    store.parent.mkdir(parents=True, exist_ok=True)
    panel.to_zarr(store, mode="w")


def test_crsp_sidecar_suffixes_are_exactly_the_five_audit_files():
    """All five, and the symbology report is deliberately among them.

    `.crsp_symbology_report.json` stopped being GENERATED in 03.11-07, which is
    exactly why it must stay on the CLEARING list: a suffix dropped from here
    leaves a file describing a mechanism that no longer exists sitting beside a
    store it never described.

    `.crsp_tickers.json` arrived in 03.11-09 and is on the list for the
    complementary reason -- it is generated, it names the PANEL's PERMNOs, and
    the store-exists guard in `_write_identity_reports` means a rebuild never
    overwrites it. Left behind, it would spell the NEW store's numbers with the
    OLD store's roster.
    """
    assert CRSP_SIDECAR_SUFFIXES == (
        ".chunks.json",
        ".crsp_adjustment.json",
        ".crsp_filter_report.json",
        ".crsp_symbology_report.json",
        ".crsp_tickers.json",
    )
    assert CrspStoreRebuilder.SIDECAR_SUFFIXES == CRSP_SIDECAR_SUFFIXES


def test_required_inputs_are_the_raw_vendor_and_reference_directories(
    tmp_path: Path,
):
    config = _crsp_config(tmp_path)
    rebuilder = CrspStoreRebuilder(config, data_root=tmp_path)

    raw, reference = rebuilder._required_inputs()

    assert raw == Path(config.raw_data_dir_path)
    assert reference == Path(config.reference_dir)
    assert raw.name == "wrds", "raw path must terminate at the vendor segment"
    assert reference.parent == raw.parent, "_reference is a SIBLING of wrds"


def test_required_inputs_resolve_relative_config_paths_under_data_root(
    tmp_path: Path,
):
    """A relative config path is joined onto `data_root`, never onto the cwd."""
    config = _crsp_config(tmp_path)
    config.raw_data_dir_path = "data/downloads/us_equity/1d/wrds_crsp/wrds"
    config.reference_dir = "data/downloads/us_equity/1d/wrds_crsp/_reference"
    rebuilder = CrspStoreRebuilder(config, data_root=tmp_path)

    raw, reference = rebuilder._required_inputs()

    assert raw == tmp_path / "data/downloads/us_equity/1d/wrds_crsp/wrds"
    assert reference == tmp_path / "data/downloads/us_equity/1d/wrds_crsp/_reference"


def test_rebuild_refuses_a_missing_raw_tier_before_deleting_anything(
    tmp_path: Path,
):
    """T-03.11-01: the destructive step is unreachable on a bad input.

    The store and all four sidecars must still be on disk after the refusal --
    a rebuild that cleared first and discovered the missing raw tier second
    would have destroyed the only panel there was.
    """
    config = _crsp_config(tmp_path)
    store = Path(config.zarr_file_path)
    sidecars = _make_store(store, CRSP_SIDECAR_SUFFIXES)
    Path(config.reference_dir).mkdir(parents=True, exist_ok=True)
    # raw_data_dir_path deliberately NOT created

    rebuilder = CrspStoreRebuilder(config, data_root=tmp_path)
    with pytest.raises(FileNotFoundError) as excinfo:
        rebuilder.rebuild(backup_dir=tmp_path / "backup")

    assert config.raw_data_dir_path in str(excinfo.value)
    assert store.exists()
    for sidecar in sidecars:
        assert sidecar.exists()


def test_measure_reports_the_seven_contract_keys(tmp_path: Path):
    config = _crsp_config(tmp_path)
    _write_measurable_panel(Path(config.zarr_file_path))
    rebuilder = CrspStoreRebuilder(config, data_root=tmp_path)

    metrics = rebuilder._measure()

    assert set(metrics) == {
        "anomaly_flag_true",
        "adj_close_le_zero",
        "close_eq_zero",
        "adj_close_nan",
        "adj_volume_nan",
        "structural_gaps",
        "symbol_count",
    }
    assert metrics["anomaly_flag_true"] == 2
    assert metrics["adj_close_le_zero"] == 1
    assert metrics["close_eq_zero"] == 1
    assert metrics["adj_close_nan"] == 1
    assert metrics["adj_volume_nan"] == 2
    assert metrics["structural_gaps"] == 1
    assert metrics["symbol_count"] == 2


def test_measure_counts_structural_gaps_from_cleaning_required_columns(
    tmp_path: Path,
):
    """The structural rule is `cleaning.REQUIRED_COLUMNS`, not a second copy.

    The panel's one all-null cell is the only structural gap, and `adjClose`
    is null exactly there -- so `adj_close_nan == structural_gaps` is the
    "no surplus NaN" statement the real gate asserts, reproduced in miniature.
    """
    config = _crsp_config(tmp_path)
    _write_measurable_panel(Path(config.zarr_file_path))
    rebuilder = CrspStoreRebuilder(config, data_root=tmp_path)

    metrics = rebuilder._measure()

    assert metrics["structural_gaps"] == metrics["adj_close_nan"]
    assert metrics["adj_volume_nan"] > metrics["structural_gaps"]


def test_measure_reports_zero_anomalies_when_the_variable_is_absent(
    tmp_path: Path,
):
    """A panel with no `anomaly_flag` measures 0, it does not raise KeyError."""
    config = _crsp_config(tmp_path)
    _write_measurable_panel(
        Path(config.zarr_file_path), with_anomaly_flag=False
    )
    rebuilder = CrspStoreRebuilder(config, data_root=tmp_path)

    assert rebuilder._measure()["anomaly_flag_true"] == 0


def test_measure_dims_reports_sizes_and_data_var_count(tmp_path: Path):
    config = _crsp_config(tmp_path)
    _write_measurable_panel(Path(config.zarr_file_path))
    rebuilder = CrspStoreRebuilder(config, data_root=tmp_path)

    dims, data_var_count = rebuilder._measure_dims()

    assert dims == {"timestamp": 3, "symbol": 2}
    # 5 required columns + adjClose + adjVolume + anomaly_flag
    assert data_var_count == len(REQUIRED_COLUMNS) + 3
