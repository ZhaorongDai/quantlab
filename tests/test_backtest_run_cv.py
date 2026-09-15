"""`BaseBacktester.run_cv()`, the model-CV backtest (phase 03.7, plan 10).

`run_cv` answers "how well does the model's cross-validation actually trade".
It reads the `cv_folds.json` manifest a `train_cv` run wrote into its project
directory, backtests every fold's out-of-sample test segment with that fold's
own checkpoint, and stitches the segments into one out-of-sample curve.

What is locked here, and what turns it red:

- **D-36, the manifest is a versioned persisted format.** A missing file, a
  missing `format_version`, a version other than 1 and an empty fold list are
  each refused with a message naming the problem. A reader that guesses at an
  unknown version would silently misread an old or future training run.
- **D-16, one checkpoint and one test segment per fold.** Each fold loads the
  checkpoint the manifest names for it, in fold order, and its weights cover
  exactly its own test bars. A fold that trades bars outside its test segment
  trades data its model was trained on.
- **D-17 per fold.** In/out-of-sample is decided with THAT fold's train dates.
  With no gap and a 2-bar label horizon, the first two test bars of every fold
  are in-sample (the label on `train_end` reads them), and each fold logs its
  own overlap warning.
- **D-35, contiguity before stitching.** The selected folds' test segments must
  follow each other bar for bar on the price calendar. A gap or an overlap is
  refused before any checkpoint is loaded or any simulation runs. Stitching
  across a gap would silently drop the uncovered bars from the "out-of-sample"
  curve; stitching an overlap would trade some bars twice with two different
  models. Either way the stitched curve would describe no real trading path.
- **Fold selection.** Only folds whose test segment lies inside the config's
  backtest window are run.
- **D-35, the stitched curve.** It is ONE continuous simulation over the
  concatenated fold weights, so capital carries across fold boundaries, while
  every fold also gets its own simulation that starts from `init_cash`.
- **D-24 / D-35, the run directory.** The top-level artifacts describe the
  stitched curve, and `folds/fold_{i}/` holds each fold's weights and equity.

Everything is synthetic, CPU-only and offline. Configs are constructed
directly, never through the factories in `quantlab/config/__init__.py` (D-32).
"""

import json
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from loguru import logger

import quantlab.backtest.engine_vectorbt as engine_module
from quantlab.backtest.us_equity import USEquityCrossectionSelectStockVectorBt
from quantlab.base.config import CrossSectionBacktestConfig
from tests.backtest_fixtures import (
    make_model,
    make_stock_dataset,
    write_price_store,
)

N_BARS = 80
TRAIN_PERIODS = 30
#: `_cv_folds`: test_periods = 30 // 5 = 6, (80 - 30) // 6 = 8 folds, whose
#: test segments cover bars 30..77 with no gap between them.
TEST_PERIODS = 6
N_FOLDS = 8
FIRST_TEST_BAR = TRAIN_PERIODS
LAST_TEST_BAR = TRAIN_PERIODS + N_FOLDS * TEST_PERIODS - 1
HORIZON = 2
REBALANCE_PERIODS = 2
TOP_N = 2
INIT_CASH = 1_000_000.0

OVERLAP_WARNING = "overlaps the model's effective training window"

_UNSET = object()


@pytest.fixture(autouse=True)
def _offline_wandb(monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("WANDB_SILENT", "true")


@pytest.fixture
def warning_messages():
    """Every loguru WARNING emitted during the test, as plain message text."""
    messages: list[str] = []
    handler_id = logger.add(messages.append, level="WARNING", format="{message}")
    yield messages
    logger.remove(handler_id)


def _day(ts) -> str:
    return pd.Timestamp(str(ts)).strftime("%Y-%m-%d")


def _model_dates(bars) -> dict:
    return dict(
        start_date=_day(bars[0]),
        end_date=_day(bars[N_BARS - 1]),
        train_start=_day(bars[0]),
        train_end=_day(bars[TRAIN_PERIODS - 1]),
        test_start=_day(bars[TRAIN_PERIODS]),
        test_end=_day(bars[N_BARS - 1]),
    )


@pytest.fixture(scope="module")
def cv_project(tmp_path_factory):
    """One real `train_cv` run, shared read-only by every test in the module.

    Tests never write into this directory: edited manifests are copies in the
    test's own tmp_path, and their checkpoint paths still point here.
    """
    root = tmp_path_factory.mktemp("cv_project")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("WANDB_MODE", "disabled")
        mp.setenv("WANDB_SILENT", "true")
        dataset_config = write_price_store(root, n_bars=N_BARS)
        bars = xr.open_zarr(dataset_config.zarr_file_path).timestamp.values
        model = make_model(
            root / "train",
            dataset_config,
            n_forward_periods=HORIZON,
            **_model_dates(bars),
        )
        model.collect()
        model.train_cv(train_periods=TRAIN_PERIODS, gap_periods=0)

    manifests = sorted((root / "train" / "models").rglob("cv_folds.json"))
    assert len(manifests) == 1, manifests
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert len(manifest["folds"]) == N_FOLDS
    return types.SimpleNamespace(
        root=root,
        dataset_config=dataset_config,
        bars=bars,
        project_dir=manifests[0].parent,
        manifest=manifest,
    )


def _test_bars(cv, fold: int) -> np.ndarray:
    first = FIRST_TEST_BAR + fold * TEST_PERIODS
    return cv.bars[first : first + TEST_PERIODS].astype("datetime64[ns]")


def _backtester(
    tmp_path: Path,
    cv,
    *,
    cv_project_dir=_UNSET,
    checkpoint=None,
    start_bar: int = FIRST_TEST_BAR,
    end_bar: int = LAST_TEST_BAR,
) -> USEquityCrossectionSelectStockVectorBt:
    project_dir = cv.project_dir if cv_project_dir is _UNSET else cv_project_dir
    return USEquityCrossectionSelectStockVectorBt(
        CrossSectionBacktestConfig(
            price_dataset=make_stock_dataset(cv.dataset_config),
            model=make_model(
                tmp_path / "backtest",
                cv.dataset_config,
                n_forward_periods=HORIZON,
                **_model_dates(cv.bars),
            ),
            model_mode="load",
            checkpoint=None if checkpoint is None else str(checkpoint),
            cv_project_dir=None if project_dir is None else str(project_dir),
            start_date=_day(cv.bars[start_bar]),
            end_date=_day(cv.bars[end_bar]),
            output_dir=str(tmp_path / "runs"),
            rebalance_periods=REBALANCE_PERIODS,
            direction="long_only",
            top_n=TOP_N,
            fees=0.0,
            slippage=0.0,
            init_cash=INIT_CASH,
        )
    )


def _edited_project(tmp_path: Path, payload: dict) -> Path:
    project = tmp_path / "edited_project"
    project.mkdir()
    (project / "cv_folds.json").write_text(json.dumps(payload), encoding="utf-8")
    return project


def _spy_load(monkeypatch, backtester) -> list[str]:
    model = backtester.config.model
    real_load = model.load
    loaded: list[str] = []

    def _spy(p):
        loaded.append(str(p))
        return real_load(p)

    monkeypatch.setattr(model, "load", _spy)
    return loaded


def _spy_from_orders(monkeypatch) -> list[pd.Index]:
    real_from_orders = engine_module.vbt.Portfolio.from_orders
    calls: list[pd.Index] = []

    def _spy(*args, **kwargs):
        calls.append(kwargs["close"].index)
        return real_from_orders(*args, **kwargs)

    monkeypatch.setattr(
        engine_module,
        "vbt",
        types.SimpleNamespace(Portfolio=types.SimpleNamespace(from_orders=_spy)),
    )
    return calls


# --------------------------------------------------------------------------
# D-36: the manifest reader refuses bad input
# --------------------------------------------------------------------------


def test_run_cv_refuses_a_missing_manifest(tmp_path, cv_project):
    empty = tmp_path / "not_a_cv_project"
    empty.mkdir()
    backtester = _backtester(tmp_path, cv_project, cv_project_dir=empty)

    with pytest.raises(FileNotFoundError) as excinfo:
        backtester.run_cv()
    assert str(empty / "cv_folds.json") in str(excinfo.value)


def test_run_cv_refuses_an_unknown_format_version(tmp_path, cv_project):
    project = _edited_project(tmp_path, {**cv_project.manifest, "format_version": 2})
    backtester = _backtester(tmp_path, cv_project, cv_project_dir=project)

    with pytest.raises(
        ValueError, match=r"format_version 2 is not supported \(supported: 1\)"
    ):
        backtester.run_cv()


def test_run_cv_refuses_a_missing_format_version(tmp_path, cv_project):
    payload = {"folds": cv_project.manifest["folds"]}
    project = _edited_project(tmp_path, payload)
    backtester = _backtester(tmp_path, cv_project, cv_project_dir=project)

    with pytest.raises(ValueError, match=r"no format_version"):
        backtester.run_cv()


def test_run_cv_refuses_an_empty_fold_list(tmp_path, cv_project):
    project = _edited_project(tmp_path, {"format_version": 1, "folds": []})
    backtester = _backtester(tmp_path, cv_project, cv_project_dir=project)

    with pytest.raises(ValueError, match=r"no folds"):
        backtester.run_cv()


def test_run_requires_a_checkpoint_and_run_cv_requires_a_project_dir(
    tmp_path, cv_project
):
    """model_mode="load" needs a checkpoint (run) or a cv_project_dir (run_cv),
    and each entry point refuses when its own input is the one missing."""
    only_project = _backtester(tmp_path / "a", cv_project)
    with pytest.raises(ValueError, match=r"run\(\).*checkpoint"):
        only_project.run()

    checkpoint = cv_project.manifest["folds"][0]["checkpoint"]
    only_checkpoint = _backtester(
        tmp_path / "b", cv_project, cv_project_dir=None, checkpoint=checkpoint
    )
    with pytest.raises(ValueError, match=r"run_cv\(\).*cv_project_dir"):
        only_checkpoint.run_cv()

    with pytest.raises(ValueError, match=r"checkpoint.*cv_project_dir"):
        _backtester(tmp_path / "c", cv_project, cv_project_dir=None)


# --------------------------------------------------------------------------
# D-16 / D-17: per-fold backtests
# --------------------------------------------------------------------------


def test_each_fold_loads_its_own_checkpoint_and_trades_only_its_test_segment(
    tmp_path, cv_project, monkeypatch
):
    backtester = _backtester(tmp_path, cv_project)
    loaded = _spy_load(monkeypatch, backtester)

    result = backtester.run_cv()

    assert loaded == [fold["checkpoint"] for fold in cv_project.manifest["folds"]]
    assert [record["fold"] for record in result.folds] == list(range(N_FOLDS))
    for record in result.folds:
        fold = record["fold"]
        np.testing.assert_array_equal(
            record["weights"].timestamp.values.astype("datetime64[ns]"),
            _test_bars(cv_project, fold),
        )
        assert record["checkpoint"] == cv_project.manifest["folds"][fold]["checkpoint"]


@pytest.mark.parametrize("recorded", ["stale-absolute", "relative"])
def test_run_cv_resolves_fold_checkpoints_inside_the_project_dir_not_the_cwd(
    tmp_path, cv_project, monkeypatch, recorded
):
    """Code review WR-03: fold checkpoints are found in `cv_project_dir`, never via the cwd.

    The project directory is copied elsewhere, as when a project is moved, and
    its manifest is rewritten two ways:

    - `stale-absolute`: every entry points at the project's OLD location, which
      no longer exists. The old `Path(checkpoint)` raised FileNotFoundError.
    - `relative`: every entry is the relative path a relative `model_save_dir`
      writes, and the process runs from another directory holding a same-named
      DECOY checkpoint at each of those relative paths. The old code resolved
      against the cwd and silently loaded the decoys: another training run's
      model.

    Both go red on the old code. The fix resolves `{experiment}/{file}`
    inside `cv_project_dir`, and the resolved paths are what gets recorded.
    """
    import shutil

    moved = tmp_path / "moved" / cv_project.project_dir.name
    shutil.copytree(cv_project.project_dir, moved)
    payload = json.loads((moved / "cv_folds.json").read_text(encoding="utf-8"))
    cwd = tmp_path / "elsewhere"
    cwd.mkdir()
    for entry in payload["folds"]:
        original = Path(entry["checkpoint"])
        tail = (
            Path("models") / cv_project.project_dir.name / original.parent.name / original.name
        )
        if recorded == "stale-absolute":
            entry["checkpoint"] = str(tmp_path / "gone" / tail)
        else:
            entry["checkpoint"] = str(tail)
            decoy = cwd / tail
            decoy.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(cv_project.manifest["folds"][0]["checkpoint"], decoy)
    (moved / "cv_folds.json").write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.chdir(cwd)

    backtester = _backtester(tmp_path, cv_project, cv_project_dir=moved)
    loaded = _spy_load(monkeypatch, backtester)
    result = backtester.run_cv()

    expected = [
        str(moved / Path(entry["checkpoint"]).parent.name / Path(entry["checkpoint"]).name)
        for entry in payload["folds"]
    ]
    assert loaded == expected
    assert [record["checkpoint"] for record in result.folds] == expected


def test_per_fold_split_uses_the_folds_own_train_dates(
    tmp_path, cv_project, warning_messages
):
    result = _backtester(tmp_path, cv_project).run_cv()

    assert len(result.folds) == N_FOLDS
    for record in result.folds:
        bars = _test_bars(cv_project, record["fold"])
        metrics = record["metrics"]
        assert tuple(metrics["in_sample_range"]) == (_day(bars[0]), _day(bars[1]))
        assert [tuple(r) for r in metrics["out_of_sample_ranges"]] == [
            (_day(bars[2]), _day(bars[-1]))
        ]
        assert tuple(metrics["training_window"])[1] == _day(bars[1])

    overlaps = [m for m in warning_messages if OVERLAP_WARNING in m]
    assert len(overlaps) == N_FOLDS, overlaps


# --------------------------------------------------------------------------
# D-35: contiguity is asserted before anything runs
# --------------------------------------------------------------------------


def test_non_contiguous_folds_are_refused_before_any_simulation(
    tmp_path, cv_project, monkeypatch
):
    folds = cv_project.manifest["folds"]
    payload = {"format_version": 1, "folds": folds[:3] + folds[4:]}
    backtester = _backtester(
        tmp_path, cv_project, cv_project_dir=_edited_project(tmp_path, payload)
    )
    calls = _spy_from_orders(monkeypatch)
    loaded = _spy_load(monkeypatch, backtester)

    with pytest.raises(ValueError, match=r"gap") as excinfo:
        backtester.run_cv()

    message = str(excinfo.value)
    assert _day(folds[2]["test_end"]) in message
    assert _day(folds[4]["test_start"]) in message
    assert calls == []
    assert loaded == []


def test_overlapping_folds_are_refused(tmp_path, cv_project, monkeypatch):
    folds = [dict(fold) for fold in cv_project.manifest["folds"]]
    moved_end = FIRST_TEST_BAR + 3 * TEST_PERIODS  # one bar past fold 2's end
    folds[2]["test_end"] = np.datetime_as_string(cv_project.bars[moved_end])
    payload = {"format_version": 1, "folds": folds}
    backtester = _backtester(
        tmp_path, cv_project, cv_project_dir=_edited_project(tmp_path, payload)
    )
    calls = _spy_from_orders(monkeypatch)

    with pytest.raises(ValueError, match=r"overlap") as excinfo:
        backtester.run_cv()

    message = str(excinfo.value)
    assert _day(folds[2]["test_end"]) in message
    assert _day(folds[3]["test_start"]) in message
    assert calls == []


def test_folds_outside_the_config_window_are_skipped(
    tmp_path, cv_project, monkeypatch
):
    first_kept = N_FOLDS - 3
    backtester = _backtester(
        tmp_path,
        cv_project,
        start_bar=FIRST_TEST_BAR + first_kept * TEST_PERIODS,
        end_bar=LAST_TEST_BAR,
    )
    loaded = _spy_load(monkeypatch, backtester)

    result = backtester.run_cv()

    kept = cv_project.manifest["folds"][first_kept:]
    assert [record["fold"] for record in result.folds] == [f["fold"] for f in kept]
    assert loaded == [fold["checkpoint"] for fold in kept]


# --------------------------------------------------------------------------
# D-35: one continuous stitched simulation
# --------------------------------------------------------------------------


def _strict_json(path: Path):
    def _reject(token):
        raise ValueError(f"non-standard JSON constant {token!r} in {path}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject)


def test_stitched_curve_is_one_continuous_simulation(
    tmp_path, cv_project, monkeypatch
):
    calls = _spy_from_orders(monkeypatch)

    result = _backtester(tmp_path, cv_project).run_cv()

    assert len(calls) == N_FOLDS + 1
    for fold in range(N_FOLDS):
        np.testing.assert_array_equal(
            calls[fold].to_numpy().astype("datetime64[ns]"),
            _test_bars(cv_project, fold),
        )
    stitched_bars = cv_project.bars[FIRST_TEST_BAR : LAST_TEST_BAR + 1].astype(
        "datetime64[ns]"
    )
    np.testing.assert_array_equal(
        calls[-1].to_numpy().astype("datetime64[ns]"), stitched_bars
    )
    np.testing.assert_array_equal(
        result.simulation.value.timestamp.values.astype("datetime64[ns]"),
        stitched_bars,
    )


def test_stitched_capital_is_not_reset_at_fold_boundaries(tmp_path, cv_project):
    """Fold 1's own simulation starts flat at init_cash; the stitched curve
    arrives at fold 1 still holding fold 0's last book, so its value moves
    across the boundary bar. A curve glued from per-fold values, whether reset
    to init_cash or rescaled to the previous end value, stays flat there."""
    result = _backtester(tmp_path, cv_project).run_cv()

    stitched = result.simulation.value
    fold0 = result.folds[0]["simulation"].value
    fold1 = result.folds[1]["simulation"].value
    boundary = _test_bars(cv_project, 1)[0]
    before = _test_bars(cv_project, 0)[-1]

    assert float(fold1.values[0]) == pytest.approx(INIT_CASH)
    assert abs(float(stitched.sel(timestamp=boundary)) - INIT_CASH) > 1.0
    assert (
        abs(
            float(stitched.sel(timestamp=boundary))
            - float(stitched.sel(timestamp=before))
        )
        > 1e-6
    )
    # Until the first boundary the stitched path IS fold 0's path.
    np.testing.assert_allclose(
        stitched.values[:TEST_PERIODS], fold0.values, rtol=1e-12
    )


def test_stitched_weights_equal_the_concatenated_fold_weights(tmp_path, cv_project):
    result = _backtester(tmp_path, cv_project).run_cv()

    expected = xr.concat([record["weights"] for record in result.folds], dim="timestamp")
    xr.testing.assert_identical(result.weights, expected)


def test_run_cv_run_directory_contents(tmp_path, cv_project):
    result = _backtester(tmp_path, cv_project).run_cv()
    run_dir = result.run_dir
    folds = cv_project.manifest["folds"]

    assert run_dir.parent == tmp_path / "runs"
    assert {p.name for p in run_dir.iterdir()} == {
        "config.json",
        "weights.zarr",
        "equity.zarr",
        "metrics.json",
        "liquidations.json",
        "fingerprint.json",
        "report.html",
        "folds",
    }

    # --- stitched artifacts --------------------------------------------------
    np.testing.assert_array_equal(
        xr.open_zarr(run_dir / "weights.zarr")["weight"].values,
        result.weights["weight"].values,
    )
    np.testing.assert_array_equal(
        xr.open_zarr(run_dir / "equity.zarr")["value"].values,
        result.simulation.value.values,
    )

    # --- per-fold artifacts --------------------------------------------------
    assert sorted(p.name for p in (run_dir / "folds").iterdir()) == sorted(
        f"fold_{record['fold']}" for record in result.folds
    )
    for record in result.folds:
        fold_dir = run_dir / "folds" / f"fold_{record['fold']}"
        assert {p.name for p in fold_dir.iterdir()} == {"weights.zarr", "equity.zarr"}
        np.testing.assert_array_equal(
            xr.open_zarr(fold_dir / "weights.zarr")["weight"].values,
            record["weights"]["weight"].values,
        )
        np.testing.assert_array_equal(
            xr.open_zarr(fold_dir / "equity.zarr")["value"].values,
            record["simulation"].value.values,
        )

    # --- metrics.json --------------------------------------------------------
    metrics = _strict_json(run_dir / "metrics.json")
    assert set(metrics) == {"stitched", "folds", "notes"}
    assert "Total Return [%]" in metrics["stitched"]["whole"]
    assert metrics["notes"]
    assert [entry["fold"] for entry in metrics["folds"]] == list(range(N_FOLDS))
    for entry in metrics["folds"]:
        bars = _test_bars(cv_project, entry["fold"])
        assert entry["test_start"] == _day(bars[0])
        assert entry["test_end"] == _day(bars[-1])
        assert entry["checkpoint"] == folds[entry["fold"]]["checkpoint"]
        assert entry["metrics"]["in_sample_range"] == [_day(bars[0]), _day(bars[1])]
        assert "Total Return [%]" in entry["metrics"]["whole"]
    assert metrics["stitched"]["in_sample_ranges"] == [
        [_day(bars[0]), _day(bars[1])]
        for bars in (_test_bars(cv_project, fold) for fold in range(N_FOLDS))
    ]

    # --- liquidations.json ---------------------------------------------------
    liquidations = _strict_json(run_dir / "liquidations.json")
    assert set(liquidations) == {"stitched", "folds"}
    assert [entry["fold"] for entry in liquidations["folds"]] == list(range(N_FOLDS))

    # --- fingerprint.json and config.json: the union window (D-27) -----------
    fingerprint = _strict_json(run_dir / "fingerprint.json")
    first_day = _day(cv_project.bars[FIRST_TEST_BAR])
    last_day = _day(cv_project.bars[LAST_TEST_BAR])
    assert _day(fingerprint["price_dataset"]["start"]) == first_day
    assert _day(fingerprint["price_dataset"]["end"]) == last_day
    factor = fingerprint["factor[0]:PastReturnFactor"]
    assert _day(factor["start"]) < first_day, "the factor range must include warm-up"
    assert _day(factor["end"]) == last_day

    config = _strict_json(run_dir / "config.json")
    assert config["cv_project_dir"] == str(cv_project.project_dir)
    assert config["data_fingerprint"] == fingerprint
