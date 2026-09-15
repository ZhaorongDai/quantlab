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
import time
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
    # `BaseModel.train` names its project directory to the second
    # (`{class}_trial_%Y%m%d_%H%M%S`) under the same model_save_dir, so a retrain
    # inside the same second as the first run would collide with its directory.
    time.sleep(1.1)
    second = rebuilt.run()

    checkpoints = sorted(Path(saved["model"]["model_save_dir"]).rglob("*.joblib"))
    assert len(checkpoints) == 2, "the rebuilt run must train its own model"
    _assert_same_run_artifacts(first.run_dir, second.run_dir)
    assert (
        second.metrics["whole"]["Total Return [%]"]
        == first.metrics["whole"]["Total Return [%]"]
    )
    assert _fingerprint_warnings(warning_messages) == []


def test_run_cv_rebuild_reproduces_the_stitched_curve(tmp_path, warning_messages):
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

    original = USEquityCrossectionSelectStockVectorBt(
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
