"""Every run records its code; a rebuild compares it and only warns.

``run.json`` of a backtest run and of a trained unit holds ``code``: the
quantlab git commit and dirty flag (context only), the sha256 of every module
defining a class of the run's component tree or a base class of one, each
marked component or framework module with the component paths using it, and
the versions of the key libraries. A rebuild compares module digests and
library versions and warns per difference, component modules before
framework modules.

Everything is synthetic, CPU-only and offline.
"""

import sys
import uuid

import pandas as pd
import pytest
import xarray as xr
from loguru import logger

import quantlab.runs.record as code_record_module
from quantlab.backtest.predefined.us_equity import USEquityCrossectionSelectStockVectorBt
from quantlab.base.config import CrossSectionBacktestConfig, TopNConfig
from quantlab.portfolio.predefined.top_n import TopNConstructor
from quantlab.runs.backtest_run import BacktestRun
from quantlab.runs.record import code_of, code_record, compare
from tests.backtest_fixtures import make_model, make_stock_dataset, train_checkpoint, write_price_store

N_BARS = 60


def _day(ts) -> str:
    """Return ``ts`` as an ISO day."""
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


@pytest.fixture
def warnings_logged():
    """Collect the loguru warnings emitted during the test."""
    messages: list[str] = []
    handler_id = logger.add(messages.append, level="WARNING", format="{message}")
    yield messages
    logger.remove(handler_id)


def _code_warnings(messages: list[str]) -> list[str]:
    """The code-mismatch warnings among ``messages``."""
    return [m for m in messages if "code mismatch" in m]


def _store(tmp_path):
    """A 60-bar price store and its bars."""
    dataset_config = write_price_store(tmp_path / "store", n_bars=N_BARS)
    return dataset_config, xr.open_zarr(dataset_config.zarr_file_path).timestamp.values


def _dates(bars) -> dict:
    """Train on bars 0..24, test on 25..29."""
    return dict(
        start_date=_day(bars[0]), end_date=_day(bars[29]),
        train_start=_day(bars[0]), train_end=_day(bars[24]),
        test_start=_day(bars[25]), test_end=_day(bars[29]),
    )


def _backtester(tmp_path, dataset_config, bars, constructor=None, *, mode="train", checkpoint=None):
    """A backtest over bars 30..50 writing its run under ``tmp_path / "runs"``."""
    return USEquityCrossectionSelectStockVectorBt(
        CrossSectionBacktestConfig(
            price_dataset=make_stock_dataset(dataset_config),
            model=make_model(tmp_path / "backtest", dataset_config, **_dates(bars)),
            model_mode=mode,
            checkpoint=None if checkpoint is None else str(checkpoint),
            start_date=_day(bars[30]),
            end_date=_day(bars[50]),
            output_dir=str(tmp_path / "runs"),
            rebalance_periods=5,
            constructor=constructor or TopNConstructor(TopNConfig(direction="long_only", top_n=2)),
        )
    )


def test_a_code_record_holds_git_modules_and_libraries(tmp_path):
    dataset_config, bars = _store(tmp_path)
    model = make_model(tmp_path / "m", dataset_config, **_dates(bars))

    record = code_of(model)

    assert set(record["git"]) == {"commit", "dirty"}
    assert record["git"]["commit"] and isinstance(record["git"]["dirty"], bool)
    modules = record["modules"]
    head = modules["tests.backtest_fixtures"]
    assert head["framework"] is False
    assert {"", "factors.0", "labels.0"} <= set(head["components"])
    assert modules["quantlab.base.model"]["framework"] is True
    assert modules["quantlab.base.model"]["components"] == [""]
    # Shipped implementations are component modules, layer frameworks are not.
    assert modules["quantlab.dataset.stock"]["framework"] is False
    assert modules["quantlab.factor.polars"]["framework"] is True
    assert all(len(entry["sha256"]) == 64 for entry in modules.values())
    # Standard-library and installed modules are not recorded; their libraries are.
    assert not {"abc", "typing", "xarray.core.dataset"} & set(modules)
    assert {"numpy", "xarray", "polars"} <= set(record["libraries"])


def test_outside_a_repository_git_is_null(monkeypatch):
    def no_git(*args, **kwargs):
        raise OSError("git is not installed")

    monkeypatch.setattr(code_record_module.subprocess, "run", no_git)

    assert code_record([])["git"] is None


def test_component_modules_are_listed_before_framework_modules(warnings_logged):
    record = {
        "modules": {
            "quantlab.base.factor": {"sha256": "a", "framework": True, "components": ["factors.0"]},
            "mine.factors": {"sha256": "b", "framework": False, "components": ["factors.0"]},
        },
        "libraries": {"numpy": "2.0"},
    }
    changed = {
        "modules": {
            "quantlab.base.factor": {"sha256": "A", "framework": True, "components": ["factors.0"]},
            "mine.factors": {"sha256": "B", "framework": False, "components": ["factors.0"]},
        },
        "libraries": {"numpy": "2.0"},
    }

    compare({"code": record}, {"code": record}, owner="run")
    assert warnings_logged == []
    compare({"code": record}, {"code": changed}, owner="run")

    first, second = warnings_logged
    assert "component module 'mine.factors' (used by 'factors.0')" in first
    assert "framework module 'quantlab.base.factor'" in second


def test_a_run_and_its_trained_unit_record_their_code(tmp_path):
    dataset_config, bars = _store(tmp_path)
    run = BacktestRun.open(_backtester(tmp_path, dataset_config, bars).run().run_dir)

    assert run.code["modules"]["quantlab.base.backtest"]["framework"] is True
    assert "price_dataset" in run.code["modules"]["quantlab.dataset.stock"]["components"]
    unit = run.trained_run()
    assert unit.code["modules"]["tests.backtest_fixtures"]["components"][0] == ""
    assert unit.code["libraries"] == run.code["libraries"]


def test_an_unchanged_rebuild_is_silent(tmp_path, warnings_logged):
    dataset_config, bars = _store(tmp_path)
    first = _backtester(tmp_path, dataset_config, bars).run()

    warnings_logged.clear()
    BacktestRun.open(first.run_dir).rebuild_backtester().run()

    assert _code_warnings(warnings_logged) == []


def test_a_changed_library_version_warns_on_rebuild_and_on_retraining(
    tmp_path, warnings_logged, monkeypatch
):
    dataset_config, bars = _store(tmp_path)
    first = _backtester(tmp_path, dataset_config, bars).run()
    installed = code_record_module._libraries()
    monkeypatch.setattr(
        code_record_module, "_libraries", lambda: {**installed, "numpy": "0.0.1"}
    )

    warnings_logged.clear()
    rebuilt = BacktestRun.open(first.run_dir).rebuild_backtester()
    at_rebuild = _code_warnings(warnings_logged)
    rebuilt.run()

    assert len(at_rebuild) == 1 and "library 'numpy'" in at_rebuild[0]
    assert "got 0.0.1" in at_rebuild[0]
    training = [m for m in _code_warnings(warnings_logged) if "training: code mismatch" in m]
    assert len(training) == 1 and "library 'numpy'" in training[0]


def test_a_changed_component_module_warns_naming_it_and_its_component(
    tmp_path, warnings_logged, monkeypatch
):
    """A user's constructor module, edited after the run, is named with its path."""
    name = f"code_record_fixture_{uuid.uuid4().hex}"
    source = tmp_path / f"{name}.py"
    source.write_text(
        "from quantlab.portfolio.predefined.top_n import TopNConstructor\n\n\n"
        "class MyTopN(TopNConstructor):\n"
        '    """TopN under another name."""\n'
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    module = __import__(name)
    monkeypatch.setitem(sys.modules, name, module)
    dataset_config, bars = _store(tmp_path)
    checkpoint = train_checkpoint(make_model(tmp_path / "train", dataset_config, **_dates(bars)))
    constructor = module.MyTopN(TopNConfig(direction="long_only", top_n=2))
    first = _backtester(
        tmp_path, dataset_config, bars, constructor, mode="load", checkpoint=checkpoint
    ).run()
    recorded = BacktestRun.open(first.run_dir).code["modules"]
    assert recorded[name]["framework"] is False
    assert recorded["quantlab.portfolio.predefined.top_n"]["framework"] is False
    assert recorded["quantlab.base.portfolio"]["framework"] is True

    source.write_text(source.read_text() + "\n# tuned after the run\n")
    warnings_logged.clear()
    BacktestRun.open(first.run_dir).rebuild_backtester()

    (warning,) = _code_warnings(warnings_logged)
    assert f"component module {name!r} (used by 'constructor')" in warning
    assert "source changed" in warning
