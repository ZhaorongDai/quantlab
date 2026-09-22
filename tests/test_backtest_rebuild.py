"""A backtest rebuilds from its persisted config.json and re-runs identically (phase 03.7, plan 11).

What is locked here:

- **D-25, the config rebuilds everything.** `load_backtester_from_config`
  turns the `config.json` a run wrote back into a backtester: the backtester
  class, its `CrossSectionBacktestConfig`, the price dataset, the model with its
  factors and labels (and the checkpoint reference), and every scalar
  parameter. Re-running the rebuilt backtester reproduces the same target
  weights and the same equity curve, for `run()` in load and train mode and for
  `run_cv()`.
- **D-26, the backtester round trip.** `get_config()` -> JSON -> loader ->
  `get_config()` is the identity, with the declared config classes on the way.
- **D-27, fingerprints on rebuild.** `data_fingerprint` in `config.json` is a
  record of what the original run read, not a config field. The loader moves it
  onto `expected_fingerprint`, so an unchanged store re-runs silently and a
  changed store re-runs with a warning naming the dataset, and still completes.
- **Security (RESEARCH Security Domain).** A `name` that is not a
  `BaseBacktester` subclass is refused before any dataset or model is built.

Fixture classes must live in an importable module (`tests.backtest_fixtures`):
a rebuild resolves every class by its dotted `name`, and a class defined in
`__main__` or inside a test function cannot be imported back.

Everything is synthetic, CPU-only and offline. Configs are constructed
directly, never through the factories in `quantlab/config/__init__.py` (D-32).
"""

import ast
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr
from loguru import logger

import quantlab.utils.module as module_utils
from quantlab.backtest.us_equity import USEquityCrossectionSelectStockVectorBt
from quantlab.base.config import CrossSectionBacktestConfig, MLConfig
from quantlab.utils.jsonable import to_jsonable
from tests.backtest_fixtures import (
    SYMBOLS,
    make_model,
    make_stock_dataset,
    train_checkpoint,
    write_price_store,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

N_BARS = 60
BARS = pd.bdate_range("2024-01-01", periods=N_BARS)
TRAIN_END_BAR = 24
WINDOW_START_BAR = 30
WINDOW_END_BAR = 50
REBALANCE_PERIODS = 2
TOP_N = 2
INIT_CASH = 1_000_000.0

#: run_cv geometry, as in tests/test_backtest_run_cv.py: `_cv_folds` with
#: train_periods=30 over 80 bars gives 8 folds of 6 test bars, bars 30..77.
CV_N_BARS = 80
CV_BARS = pd.bdate_range("2024-01-01", periods=CV_N_BARS)
CV_TRAIN_PERIODS = 30
CV_FIRST_TEST_BAR = 30
CV_LAST_TEST_BAR = 77

#: `BaseBacktester._compare_fingerprints` starts every warning with this.
FINGERPRINT_WARNING = "data fingerprint mismatch"

#: The distinctive substring of `quantlab.base.backtest.FINGERPRINT_PARTIAL_NOTE`,
#: the tail `_compare_fingerprints(partial=True)` appends instead of "continuing".
#: Spelled out here rather than imported on purpose: an ImportError at module
#: level would break collection of this whole file, and these locks must be able
#: to go red on code that does not define the constant yet. Any future rewording
#: of that note must keep this substring.
PARTIAL_WARNING = "comparison is PARTIAL"


@pytest.fixture(autouse=True)
def _offline_wandb(monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("WANDB_SILENT", "true")


@pytest.fixture
def warning_messages():
    """Every loguru WARNING emitted during the test, as plain message text."""
    messages: list[str] = []
    handler_id = logger.add(
        lambda message: messages.append(message.record["message"]),
        level="WARNING",
    )
    yield messages
    logger.remove(handler_id)


def _day(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def _json(cfg: dict) -> dict:
    """What `config.json` holds: `to_jsonable` then a strict JSON trip."""
    return json.loads(json.dumps(to_jsonable(cfg), allow_nan=False))


def _model_dates() -> dict:
    return dict(
        start_date=_day(BARS[0]),
        end_date=_day(BARS[29]),
        train_start=_day(BARS[0]),
        train_end=_day(BARS[TRAIN_END_BAR]),
        test_start=_day(BARS[TRAIN_END_BAR + 1]),
        test_end=_day(BARS[29]),
    )


def _backtester(
    root: Path,
    dataset_config,
    *,
    model_mode: str = "load",
    checkpoint=None,
    **overrides,
) -> USEquityCrossectionSelectStockVectorBt:
    kwargs = dict(
        price_dataset=make_stock_dataset(dataset_config),
        model=make_model(root / "backtest", dataset_config, **_model_dates()),
        model_mode=model_mode,
        checkpoint=None if checkpoint is None else str(checkpoint),
        start_date=_day(BARS[WINDOW_START_BAR]),
        end_date=_day(BARS[WINDOW_END_BAR]),
        output_dir=str(root / "runs"),
        rebalance_periods=REBALANCE_PERIODS,
        direction="long_only",
        top_n=TOP_N,
        fees=0.0,
        slippage=0.0,
        init_cash=INIT_CASH,
    )
    kwargs.update(overrides)
    return USEquityCrossectionSelectStockVectorBt(CrossSectionBacktestConfig(**kwargs))


def _trained(root: Path):
    dataset_config = write_price_store(root / "store", n_bars=N_BARS)
    checkpoint = train_checkpoint(
        make_model(root / "train", dataset_config, **_model_dates())
    )
    return dataset_config, checkpoint


def _read_run_config(run_dir: Path) -> dict:
    return json.loads((run_dir / "config.json").read_text(encoding="utf-8"))


def _assert_same_run_artifacts(first_dir: Path, second_dir: Path) -> None:
    """Identical weights.zarr and exactly equal equity values on disk."""
    assert first_dir != second_dir
    xr.testing.assert_identical(
        xr.open_zarr(first_dir / "weights.zarr").load(),
        xr.open_zarr(second_dir / "weights.zarr").load(),
    )
    np.testing.assert_array_equal(
        xr.open_zarr(first_dir / "equity.zarr")["value"].values,
        xr.open_zarr(second_dir / "equity.zarr")["value"].values,
    )


def _fingerprint_warnings(messages: list[str]) -> list[str]:
    return [m for m in messages if FINGERPRINT_WARNING in m]


def _cv_original(tmp_path: Path) -> USEquityCrossectionSelectStockVectorBt:
    """The run_cv setup: a CV-sized store, a `train_cv` project, and a backtester over it.

    Writes `CV_N_BARS` bars, trains one `train_cv` project (asserting it wrote
    exactly one `cv_folds.json`), and returns a backtester pointed at that
    project over the `CV_FIRST_TEST_BAR..CV_LAST_TEST_BAR` window. Shared by the
    rebuild lock and the failure-path lock so neither duplicates the setup.
    """
    dataset_config = write_price_store(tmp_path / "store", n_bars=CV_N_BARS)
    model_dates = dict(
        start_date=_day(CV_BARS[0]),
        end_date=_day(CV_BARS[CV_N_BARS - 1]),
        train_start=_day(CV_BARS[0]),
        train_end=_day(CV_BARS[CV_TRAIN_PERIODS - 1]),
        test_start=_day(CV_BARS[CV_TRAIN_PERIODS]),
        test_end=_day(CV_BARS[CV_N_BARS - 1]),
    )
    trainer = make_model(tmp_path / "train", dataset_config, **model_dates)
    trainer.collect()
    trainer.train_cv(train_periods=CV_TRAIN_PERIODS, gap_periods=0)
    manifests = sorted((tmp_path / "train" / "models").rglob("cv_folds.json"))
    assert len(manifests) == 1, manifests

    return USEquityCrossectionSelectStockVectorBt(
        CrossSectionBacktestConfig(
            price_dataset=make_stock_dataset(dataset_config),
            model=make_model(tmp_path / "backtest", dataset_config, **model_dates),
            model_mode="load",
            cv_project_dir=str(manifests[0].parent),
            start_date=_day(CV_BARS[CV_FIRST_TEST_BAR]),
            end_date=_day(CV_BARS[CV_LAST_TEST_BAR]),
            output_dir=str(tmp_path / "runs"),
            rebalance_periods=REBALANCE_PERIODS,
            direction="long_only",
            top_n=TOP_N,
            fees=0.0,
            slippage=0.0,
            init_cash=INIT_CASH,
        )
    )


# --------------------------------------------------------------------------
# Task 1: the loader rebuilds the whole backtester (D-25, D-26)
# --------------------------------------------------------------------------


def test_rebuilt_backtester_has_the_same_class_config_class_and_config(tmp_path):
    dataset_config = write_price_store(tmp_path / "store", n_bars=N_BARS)
    original = _backtester(
        tmp_path, dataset_config, checkpoint=tmp_path / "never_read.joblib"
    )
    saved = _json(original.get_config())

    rebuilt = module_utils.load_backtester_from_config(saved)

    assert type(rebuilt) is USEquityCrossectionSelectStockVectorBt
    assert type(rebuilt.config) is CrossSectionBacktestConfig
    assert type(rebuilt.config.model.config) is MLConfig
    assert rebuilt.expected_fingerprint is None
    assert _json(rebuilt.get_config()) == saved


def test_rebuild_from_a_run_config_moves_the_fingerprint_to_expected(tmp_path):
    dataset_config, checkpoint = _trained(tmp_path)
    result = _backtester(tmp_path, dataset_config, checkpoint=checkpoint).run()
    saved = _read_run_config(result.run_dir)
    assert "data_fingerprint" in saved

    rebuilt = module_utils.load_backtester_from_config(saved)

    assert rebuilt.expected_fingerprint == saved["data_fingerprint"]
    assert "data_fingerprint" not in rebuilt.get_config()
    assert rebuilt.config.checkpoint == str(checkpoint)


def test_backtest_config_path_fields_are_stored_absolute(tmp_path, monkeypatch):
    """Code review WR-03: `checkpoint`, `cv_project_dir` and `output_dir` persist as absolute paths.

    A persisted `config.json` holding relative paths rebuilds against
    whatever directory the rebuild runs in (D-25). The backtester is built
    from `tmp_path` with relative spellings and the config is read back. The
    old setter kept them verbatim and goes red.
    """
    dataset_config = write_price_store(tmp_path / "store", n_bars=N_BARS)
    monkeypatch.chdir(tmp_path)

    config = _json(
        _backtester(
            tmp_path,
            dataset_config,
            checkpoint="ckpt/model.joblib",
            output_dir="runs",
            cv_project_dir="models/project",
        ).get_config()
    )

    for key, relative in (
        ("checkpoint", "ckpt/model.joblib"),
        ("output_dir", "runs"),
        ("cv_project_dir", "models/project"),
    ):
        assert Path(config[key]).is_absolute(), (key, config[key])
        assert Path(config[key]).resolve() == (tmp_path / relative).resolve()


def test_loader_does_not_mutate_its_input(tmp_path):
    dataset_config = write_price_store(tmp_path / "store", n_bars=N_BARS)
    saved = _json(
        _backtester(
            tmp_path, dataset_config, checkpoint=tmp_path / "never_read.joblib"
        ).get_config()
    )
    saved["data_fingerprint"] = {"price_dataset": {"digest": "abc"}}
    before = copy.deepcopy(saved)

    module_utils.load_backtester_from_config(saved)

    assert saved == before


@pytest.mark.parametrize("field_name", ["init_cash", "fees", "score_label", "use_wandb"])
def test_rebuild_refuses_a_config_missing_a_field(tmp_path, monkeypatch, field_name):
    """Code review WR-06: a config missing a field is refused, not filled from today's defaults.

    `init_cash` is the executor-reported gap; `fees`, `score_label` and
    `use_wandb` have defaults that could change later. The old loader built
    the config with `**config` and silently took the current default, so an
    older config.json rebuilt into a different backtest. The refusal must
    name the field and come before any dataset or model is built. Red on the
    old code: no ValueError, and the stubbed loaders are called.
    """
    dataset_config = write_price_store(tmp_path / "store", n_bars=N_BARS)
    saved = _json(
        _backtester(
            tmp_path, dataset_config, checkpoint=tmp_path / "never_read.joblib"
        ).get_config()
    )
    del saved[field_name]
    built: list[dict] = []
    monkeypatch.setattr(
        module_utils, "load_dataset_from_config", lambda cfg: built.append(cfg)
    )
    monkeypatch.setattr(
        module_utils, "load_model_from_config", lambda cfg: built.append(cfg)
    )

    with pytest.raises(ValueError, match=field_name):
        module_utils.load_backtester_from_config(saved)

    assert built == []


def test_rebuild_round_trips_every_field_with_non_default_values(tmp_path):
    """Code review WR-06: every config field survives the rebuild, checked with NON-default values.

    A round trip that uses a field's default cannot detect that the field was
    dropped: the loader would fill in the same default. So every defaulted
    field gets a value different from its default, verified at the top of
    the test, except `benchmark_dataset` (D-08 refuses anything but None) and
    `name` (rebuilt from the class). Every scalar field must then come back
    equal. This is a lock rather than a red-first test: it passes before the
    fix too, and goes red if `to_dict`/`get_config` ever drops a field or the
    loader substitutes a default.
    """
    from dataclasses import MISSING, fields

    dataset_config = write_price_store(tmp_path / "store", n_bars=N_BARS)
    original = _backtester(
        tmp_path,
        dataset_config,
        model_mode="load",
        checkpoint=tmp_path / "never_read.joblib",
        cv_project_dir=str(tmp_path / "cv_project"),
        fees=0.0007,
        slippage=0.0003,
        init_cash=250_000.0,
        score_label="fwd_ret_1",
        use_wandb=True,
        rebalance_periods=3,
        direction="long_short",
        top_n=1,
    )
    for field in fields(CrossSectionBacktestConfig):
        if field.default is MISSING or field.name in ("benchmark_dataset", "name"):
            continue
        assert getattr(original.config, field.name) != field.default, field.name

    saved = _json(original.get_config())
    rebuilt = module_utils.load_backtester_from_config(saved)

    for field in fields(CrossSectionBacktestConfig):
        if field.name in ("price_dataset", "model", "benchmark_dataset"):
            continue
        assert getattr(rebuilt.config, field.name) == getattr(
            original.config, field.name
        ), field.name
    assert _json(rebuilt.get_config()) == saved


def test_non_backtester_class_is_refused_before_building_anything(
    tmp_path, monkeypatch
):
    dataset_config = write_price_store(tmp_path / "store", n_bars=N_BARS)
    saved = _json(
        _backtester(
            tmp_path, dataset_config, checkpoint=tmp_path / "never_read.joblib"
        ).get_config()
    )
    tampered = dict(saved, name="quantlab.dataset.stock.StockDataset")

    dataset_calls: list[dict] = []
    model_calls: list[dict] = []
    monkeypatch.setattr(
        module_utils, "load_dataset_from_config", lambda cfg: dataset_calls.append(cfg)
    )
    monkeypatch.setattr(
        module_utils, "load_model_from_config", lambda cfg: model_calls.append(cfg)
    )

    with pytest.raises(TypeError, match=r"quantlab\.dataset\.stock\.StockDataset"):
        module_utils.load_backtester_from_config(tampered)

    assert dataset_calls == []
    assert model_calls == []


def _imported_modules(path: Path) -> set[str]:
    """Every absolute module name `path` imports, including `from a import b` as `a.b`."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, f"relative import in {path}: resolve it first"
            names.add(node.module or "")
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return names


def test_backtester_rebuild_imports_no_config_factories():
    """D-32: neither the loader nor these locks reach `quantlab.config`."""
    # Positive control: the scan sees this file's own real imports.
    assert "quantlab.utils.module" in _imported_modules(Path(__file__))

    for path in (Path(__file__), REPO_ROOT / "quantlab/utils/module.py"):
        offending = sorted(
            name
            for name in _imported_modules(path)
            if name == "quantlab.config" or name.startswith("quantlab.config.")
        )
        assert offending == [], f"{path} imports {offending}"


# --------------------------------------------------------------------------
# Task 2: a rebuilt config re-runs identically and notices changed data (D-25, D-27)
# --------------------------------------------------------------------------


def test_load_mode_rebuild_reproduces_weights_and_equity(tmp_path, warning_messages):
    dataset_config, checkpoint = _trained(tmp_path)
    first = _backtester(tmp_path, dataset_config, checkpoint=checkpoint).run()

    rebuilt = module_utils.load_backtester_from_config(_read_run_config(first.run_dir))
    second = rebuilt.run()

    _assert_same_run_artifacts(first.run_dir, second.run_dir)
    xr.testing.assert_identical(first.weights, second.weights)
    np.testing.assert_array_equal(
        first.simulation.value.values, second.simulation.value.values
    )
    assert (
        second.metrics["whole"]["Total Return [%]"]
        == first.metrics["whole"]["Total Return [%]"]
    )
    assert rebuilt.expected_fingerprint is not None
    assert _fingerprint_warnings(warning_messages) == []


def test_train_mode_rebuild_retrains_and_reproduces_weights_and_equity(
    tmp_path, warning_messages
):
    dataset_config = write_price_store(tmp_path / "store", n_bars=N_BARS)
    first = _backtester(tmp_path, dataset_config, model_mode="train").run()
    saved = _read_run_config(first.run_dir)
    assert saved["model_mode"] == "train"
    assert saved["checkpoint"] is None

    rebuilt = module_utils.load_backtester_from_config(saved)
    # Code review WR-04: `BaseModel.train` names its project directory to the
    # microsecond and never reuses an existing one, so the rebuild retrains
    # immediately. The old `time.sleep(1.1)` wall-clock workaround is gone;
    # without the fix this line raises "... already exists" within the second.
    second = rebuilt.run()

    checkpoints = sorted(Path(saved["model"]["model_save_dir"]).rglob("*.joblib"))
    assert len(checkpoints) == 2, "the rebuilt run must train its own model"
    _assert_same_run_artifacts(first.run_dir, second.run_dir)
    assert (
        second.metrics["whole"]["Total Return [%]"]
        == first.metrics["whole"]["Total Return [%]"]
    )
    assert _fingerprint_warnings(warning_messages) == []


def test_train_mode_run_records_its_checkpoint_and_replays_it_in_load_mode(tmp_path):
    """Code review WR-04: a train-mode run records the checkpoint it trained.

    Before the fix neither config.json nor metrics.json named the model a
    train-mode backtest produced. That backtest could not be replayed against
    the exact model, only retrained, and retraining is not bit-reproducible for
    torch or GPU heads. `trained_checkpoint` must now be in both files, point at
    the one checkpoint trained, and replay identically through a load-mode
    rebuild. The loader treats it as a record, so the replay's own config
    (load mode, nothing trained) carries none. Red on the old code
    (KeyError on `trained_checkpoint`).
    """
    dataset_config = write_price_store(tmp_path / "store", n_bars=N_BARS)
    first = _backtester(tmp_path, dataset_config, model_mode="train").run()
    saved = _read_run_config(first.run_dir)
    metrics = json.loads((first.run_dir / "metrics.json").read_text(encoding="utf-8"))

    recorded = saved["trained_checkpoint"]
    assert Path(recorded).is_absolute() and Path(recorded).is_file(), recorded
    assert metrics["trained_checkpoint"] == recorded
    assert sorted(Path(saved["model"]["model_save_dir"]).rglob("*.joblib")) == [
        Path(recorded)
    ]

    replay = module_utils.load_backtester_from_config(
        dict(saved, model_mode="load", checkpoint=recorded)
    )
    second = replay.run()

    _assert_same_run_artifacts(first.run_dir, second.run_dir)
    assert "trained_checkpoint" not in _read_run_config(second.run_dir)
    assert len(sorted(Path(saved["model"]["model_save_dir"]).rglob("*.joblib"))) == 1


def test_run_cv_rebuild_reproduces_the_stitched_curve(tmp_path, warning_messages):
    original = _cv_original(tmp_path)
    first = original.run_cv()

    rebuilt = module_utils.load_backtester_from_config(_read_run_config(first.run_dir))
    second = rebuilt.run_cv()

    assert len(second.folds) == len(first.folds) > 1
    _assert_same_run_artifacts(first.run_dir, second.run_dir)
    xr.testing.assert_identical(first.weights, second.weights)
    np.testing.assert_array_equal(
        first.simulation.value.values, second.simulation.value.values
    )
    assert _fingerprint_warnings(warning_messages) == []


def test_changed_store_rebuild_warns_and_completes(tmp_path, warning_messages):
    dataset_config, checkpoint = _trained(tmp_path)
    first = _backtester(tmp_path, dataset_config, checkpoint=checkpoint).run()
    saved = _read_run_config(first.run_dir)

    # Overwrite one adjusted close inside the backtest window directly in the
    # Zarr array (group opened "r+"), the way a Tiingo re-base rewrites history.
    bar, symbol = WINDOW_START_BAR + 5, 0
    group = zarr.open_group(dataset_config.zarr_file_path, mode="r+")
    old = float(group["adjClose"][bar, symbol])
    group["adjClose"][bar, symbol] = old * 1.25
    # Positive control: the change is visible through xarray, where the run reads.
    assert float(
        xr.open_zarr(dataset_config.zarr_file_path)["adjClose"].values[bar, symbol]
    ) == pytest.approx(old * 1.25)

    rebuilt = module_utils.load_backtester_from_config(saved)
    assert _fingerprint_warnings(warning_messages) == []
    second = rebuilt.run()

    assert second.run_dir.is_dir()
    mismatches = _fingerprint_warnings(warning_messages)
    assert any("'price_dataset'" in m for m in mismatches), warning_messages


# --------------------------------------------------------------------------
# Task 3: a failed run still reports the changed data (D-03.11-UAT-A)
# --------------------------------------------------------------------------


def _boom(*_args, **_kwargs):
    """Stands in for any real post-fingerprint failure inside the backtest window.

    For example a ticker-era checkpoint predicted against a PERMNO panel, or
    "the feature panel lacks N of the symbols this model was trained on".
    """
    raise ValueError("predict_panel: representative downstream failure")


def test_a_raise_inside_the_window_still_reports_the_changed_data(
    tmp_path, warning_messages, monkeypatch
):
    """A run that dies inside `_backtest_window` still says the data changed (D-03.11-UAT-A).

    Control arm: without any raise, the two fingerprint mismatches are reported
    exactly as they are today — same count, same trailing text, no partial
    marker. That arm proves the mismatch is detectable at all and that the happy
    path gained no extra or reworded warning.

    Probe arm: `predict_panel` raises after `_redate_factors` has already
    recorded the factor fingerprint and before the price fingerprint exists. The
    factor mismatch must be reported, marked partial, and the original
    `ValueError` must be what propagates.
    """
    dataset_config, checkpoint = _trained(tmp_path)
    first = _backtester(tmp_path, dataset_config, checkpoint=checkpoint).run()
    saved = _read_run_config(first.run_dir)

    # A real data change at the same path `_trained` wrote: one symbol
    # disappears, so the factor and the price fingerprint both differ
    # (digest + n_symbols).
    write_price_store(tmp_path / "store", symbols=SYMBOLS[:-1], n_bars=N_BARS)

    # --- control: no raise, the mismatch is reported exactly as today --------
    warning_messages.clear()
    module_utils.load_backtester_from_config(saved).run()

    control = _fingerprint_warnings(warning_messages)
    assert len(control) == 2, control
    assert any("'factor[0]:PastReturnFactor'" in m for m in control), control
    assert any("'price_dataset'" in m for m in control), control
    assert all(m.endswith("(D-27); continuing") for m in control), control
    assert all(PARTIAL_WARNING not in m for m in control), control

    # --- probe: a raise inside the window, after a fingerprint exists --------
    warning_messages.clear()
    rebuilt = module_utils.load_backtester_from_config(saved)
    monkeypatch.setattr(rebuilt.config.model, "predict_panel", _boom)

    with pytest.raises(ValueError, match="representative downstream failure"):
        rebuilt.run()

    # The price fingerprint does not exist yet at raise time.
    assert sorted(rebuilt._fingerprints) == ["factor[0]:PastReturnFactor"]
    probe = _fingerprint_warnings(warning_messages)
    assert len(probe) == 1, probe
    assert "'factor[0]:PastReturnFactor'" in probe[0], probe
    assert "n_symbols: expected 6, got 5" in probe[0], probe
    assert PARTIAL_WARNING in probe[0], probe
    # `price_dataset` was not read YET, not "not read": warning about it would
    # be a false alarm invented by the fix.
    assert all("not read by this run" not in m for m in warning_messages), (
        warning_messages
    )


def test_a_failing_partial_diagnostic_never_replaces_the_real_exception(
    tmp_path, warning_messages, monkeypatch
):
    """A broken diagnostic cannot become the exception the caller sees (D-03.11-UAT-A).

    The failure-path guard swallows whatever the diagnostic raises, but it does
    not hide it: the broken diagnostic is reported as its own warning, and the
    original `ValueError` propagates unchanged.
    """
    dataset_config, checkpoint = _trained(tmp_path)
    first = _backtester(tmp_path, dataset_config, checkpoint=checkpoint).run()
    saved = _read_run_config(first.run_dir)
    write_price_store(tmp_path / "store", symbols=SYMBOLS[:-1], n_bars=N_BARS)

    warning_messages.clear()
    rebuilt = module_utils.load_backtester_from_config(saved)
    monkeypatch.setattr(rebuilt.config.model, "predict_panel", _boom)

    def broken_diagnostic(*_args, **_kwargs):
        raise RuntimeError("the diagnostic itself is broken")

    monkeypatch.setattr(rebuilt, "_compare_fingerprints", broken_diagnostic)

    with pytest.raises(ValueError, match="representative downstream failure") as excinfo:
        rebuilt.run()

    assert "the diagnostic itself is broken" not in repr(excinfo.value)
    reported = [
        m
        for m in warning_messages
        if "diagnostic" in m and m not in _fingerprint_warnings(warning_messages)
    ]
    assert reported, warning_messages
    assert any("the diagnostic itself is broken" in m for m in reported), reported


def test_run_cv_reports_a_partial_comparison_when_a_fold_raises(
    tmp_path, warning_messages, monkeypatch
):
    """`run_cv` carries the same failure-path diagnostic (D-03.11-UAT-A).

    What this test does NOT claim: that the store changed. It did not. The
    fold-0 window is narrower than the stitched window the expected fingerprint
    describes, so the differing fields are `end` / `n_timestamps` / `digest` by
    construction, not because any data moved. What is locked is that the
    diagnostic RUNS on the failure path, that every warning it emits is marked
    partial, and that the original exception propagates. That range caveat is
    exactly why the partial marker exists.
    """
    original = _cv_original(tmp_path)
    first = original.run_cv()
    # The probe depends on the fold window being narrower than the stitched one.
    assert len(first.folds) > 1

    rebuilt = module_utils.load_backtester_from_config(_read_run_config(first.run_dir))
    monkeypatch.setattr(rebuilt.config.model, "predict_panel", _boom)
    warning_messages.clear()

    with pytest.raises(ValueError, match="representative downstream failure"):
        rebuilt.run_cv()

    partial = _fingerprint_warnings(warning_messages)
    assert partial, warning_messages
    assert all(PARTIAL_WARNING in m for m in partial), partial
    assert any("'factor[0]:PastReturnFactor'" in m for m in partial), partial
    assert all("not read by this run" not in m for m in warning_messages), (
        warning_messages
    )
