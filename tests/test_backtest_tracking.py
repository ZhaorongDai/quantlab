"""Backtests track through the config's tracker (ticket #103, ADR 0015).

Every test puts a tracker in the backtest config and calls a public entry
point (``run``, ``run_cv``, ``quantlab.api.backtest``, a rebuild through
``BacktestRun``); it asserts what an outsider sees: the run the tracker
opened, its summary, its files and how it ended. The report's file name
appears where the attached file is the subject.
"""

import dataclasses
import json
import subprocess
import sys

import numpy as np
import pytest

from quantlab.backtest.config import BacktestConfig
from quantlab.tracking.base import NullTracker
from quantlab.runs.backtest_run import BacktestRun
from quantlab.tracking.wandb import WandbTracker
from tests.test_backtest_contracts import REPO_ROOT
from tests.test_backtest_persistence import (
    OVERLAP_END_BAR,
    OVERLAP_START_BAR,
    _backtester,
    _trained_store,
)
from tests.test_backtest_run_cv import _backtester as _cv_backtester
from tests.test_backtest_run_cv import cv_project  # noqa: F401 (module fixture)
from tests.tracking_fixtures import RecordingTracker

PROJECT = "USEquityCrossectionSelectStockVectorBt_backtest"


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    return _trained_store(tmp_path_factory.mktemp("trained"))


def _tracked(tmp_path, trained, tracker, **overrides):
    dataset_config, checkpoint = trained
    return _backtester(
        tmp_path,
        dataset_config,
        checkpoint,
        tag="tracked",
        window_start_bar=OVERLAP_START_BAR,
        window_end_bar=OVERLAP_END_BAR,
        tracker=tracker,
        **overrides,
    )


def test_the_default_config_tracks_nowhere():
    assert BacktestConfig.__dataclass_fields__["tracker"].default == NullTracker()
    assert "use_wandb" not in BacktestConfig.__dataclass_fields__


def test_a_run_is_one_finished_run_with_the_metric_blocks_and_the_report(tmp_path, trained):
    tracker = RecordingTracker()

    result = _tracked(tmp_path, trained, tracker).run()

    (run,) = tracker.runs
    assert (run.project, run.group, run.name) == (PROJECT, None, result.run_dir.name)
    assert run.finished and not run.failed
    json.dumps(run.config, allow_nan=False)
    assert run.config["tracker"] == tracker.get_config()
    # Opened before the backtest; the fingerprints it read are added after.
    assert "data_fingerprint" not in run.config
    recorded = BacktestRun.open(result.run_dir).data_fingerprint
    assert json.loads(json.dumps(run.config_updates[-1]["data_fingerprint"])) == recorded
    assert run.summary, "the summary must receive metrics"
    for key, value in run.summary.items():
        assert key.split("/")[0] in {"whole", "in_sample", "out_of_sample"}, key
        assert isinstance(value, (int, float)) and not isinstance(value, bool), key
        assert np.isfinite(value), key
    assert any(key.startswith("in_sample/") for key in run.summary)
    assert any(key.startswith("out_of_sample/") for key in run.summary)
    assert "whole/Total Turnover [%]" in run.summary
    assert run.summary["whole/Total Return [%]"] == pytest.approx(
        result.metrics["whole"]["Total Return [%]"]
    )
    assert run.files == [result.run_dir / "report.html"]


def test_a_backtest_lands_in_a_local_mlflow_store(tmp_path, trained, monkeypatch):
    """Summary keys like ``whole/Total Return [%]`` and fingerprint params are
    cleaned to keys MLflow takes, so the backtest itself never fails on them."""
    pytest.importorskip("mlflow")
    from mlflow import MlflowClient

    from quantlab.tracking.mlflow import MlflowTracker

    monkeypatch.delenv("MLFLOW_ALLOW_FILE_STORE", raising=False)
    store = f"file:{tmp_path / 'mlruns'}"

    result = _tracked(tmp_path, trained, MlflowTracker(tracking_uri=store)).run()

    client = MlflowClient(tracking_uri=store)
    (run,) = client.search_runs([client.get_experiment_by_name(PROJECT).experiment_id])
    assert (run.info.run_name, run.info.status) == (result.run_dir.name, "FINISHED")
    assert run.data.metrics["whole/Total Return ___"] == pytest.approx(
        result.metrics["whole"]["Total Return [%]"]
    )
    assert any(key.startswith("data_fingerprint/") for key in run.data.params)
    artifacts = {artifact.path for artifact in client.list_artifacts(run.info.run_id)}
    assert "report.html" in artifacts


def test_a_run_kept_in_memory_is_tracked_without_the_report(tmp_path, trained):
    tracker = RecordingTracker()

    result = _tracked(tmp_path, trained, tracker, output_dir=None).run()

    assert result.run_dir is None
    (run,) = tracker.runs
    assert run.name.startswith("USEquityCrossectionSelectStockVectorBt_")
    assert "whole/Total Return [%]" in run.summary
    assert run.files == []
    assert run.finished and not run.failed


def test_a_failed_backtest_is_finished_as_failed(tmp_path, trained):
    tracker = RecordingTracker()
    dataset_config, _ = trained
    backtester = _tracked(tmp_path, (dataset_config, tmp_path / "missing.joblib"), tracker)

    with pytest.raises(FileNotFoundError):
        backtester.run()

    (run,) = tracker.runs
    assert run.name.startswith(f"{PROJECT.removesuffix('_backtest')}_")
    assert run.finished and run.failed
    assert run.summary == {} and run.files == []


def test_run_cv_tracks_the_stitched_metrics_in_one_run(tmp_path, cv_project):  # noqa: F811
    tracker = RecordingTracker()
    plain = _cv_backtester(tmp_path, cv_project)
    backtester = type(plain)(dataclasses.replace(plain.config, tracker=tracker))

    result = backtester.run_cv()

    (run,) = tracker.runs
    assert (run.project, run.name) == (PROJECT, result.run_dir.name)
    assert run.summary["whole/Total Return [%]"] == pytest.approx(
        result.metrics["stitched"]["whole"]["Total Return [%]"]
    )
    assert {key.split("/")[0] for key in run.summary} <= {
        "whole", "in_sample", "out_of_sample", "benchmark", "relative"
    }
    assert run.files == [result.run_dir / "report.html"]
    assert run.finished and not run.failed


def test_the_tracker_round_trips_through_the_run(tmp_path, trained):
    tracker = WandbTracker(project="research", entity="team", mode="disabled")
    result = _tracked(tmp_path, trained, tracker).run()

    run = BacktestRun.open(result.run_dir)
    assert run.rebuild("tracker") == tracker
    assert run.rebuild_backtester().config.tracker == tracker


def test_a_backtest_under_the_default_config_imports_no_tracking_library(tmp_path):
    code = (
        "import sys\n"
        "import numpy as np, pandas as pd\n"
        "import quantlab.api as qa\n"
        "bars = pd.bdate_range('2024-01-01', periods=5)\n"
        "prices = pd.DataFrame({'timestamp': np.repeat(bars, 2), 'symbol': ['AAA', 'BBB'] * 5,\n"
        "    'open': np.linspace(10.0, 14.0, 10), 'close': np.linspace(10.5, 14.5, 10)})\n"
        "weights = pd.DataFrame({'timestamp': [bars[0]], 'symbol': ['AAA'], 'weight': [1.0]})\n"
        f"qa.backtest(prices, weights=weights, output_dir={str(tmp_path)!r})\n"
        "print(sorted(name for name in ('wandb', 'mlflow') if name in sys.modules))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO_ROOT
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "[]"
