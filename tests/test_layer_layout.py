"""The layer layout: `base/` holds the root base classes, each layer's top level holds its
extension framework, and `<layer>/predefined/` holds the shipped implementations.

What is locked here, and what turns it red:

- importing any `quantlab/base` module loads no torch (the training target of both model
  variants lives in `quantlab/model/training_target.py`, not on `BaseModel`);
- no `quantlab/base` module imports the factor, label, model, backtest or portfolio layers;
- no layer's framework module (a top-level file of `quantlab/factor`, `quantlab/label`,
  `quantlab/model`, `quantlab/backtest`, `quantlab/portfolio`) imports that layer's
  `predefined` package, so a user's own factor, label, model, backtester or portfolio
  construction rule needs nothing from the shipped ones;
- the portfolio layer never imports the backtest layer (the backtest layer may import
  the portfolio layer), so an event-driven engine can depend on the portfolio layer alone;
- cvxpy is imported by the mean-variance optimiser only, never by the base package or
  the portfolio framework;
- the tracking root module imports no tracking library, wandb is imported by the W&B
  tracker only and mlflow by the MLflow tracker only (ADR 0015);
- every `predefined` package's `__init__.py` is empty.
- the walk-forward module (`quantlab/utils/walk_forward.py`), which the base layer
  imports, imports no quantlab module outside `quantlab.utils`;
- the run layer (`quantlab/runs`) imports only `quantlab.utils`, itself, the storage
  backend (`quantlab.backend`) and the base root modules it names
  (`RUNS_BASE_MODULES`), never the model, factor, label,
  backtest or portfolio layers; its `__init__.py` is empty; and the trained-run
  module imports no quantlab module outside `quantlab.utils` and the run-directory
  mechanism, so the base layer may import it.

Static checks, plus one subprocess import; offline.
"""

import subprocess
import sys

import pytest

from tests.test_backtest_contracts import (
    REPO_ROOT,
    _is_or_under,
    _python_files,
    _resolved_imports,
)

BASE = REPO_ROOT / "quantlab/base"
LAYERS = ("factor", "label", "model", "backtest", "portfolio")


def test_importing_the_base_layer_loads_no_torch():
    modules = sorted(
        f"quantlab.base.{path.stem}" for path in BASE.glob("*.py") if path.stem != "__init__"
    )
    code = (
        "import sys\n"
        + "".join(f"import {name}\n" for name in modules)
        + "print('torch' in sys.modules)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO_ROOT
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "False"


def test_base_never_imports_the_layers_above_it():
    offenders = {
        path.name: sorted(
            name
            for name in _resolved_imports(path)
            if any(_is_or_under(name, f"quantlab.{layer}") for layer in LAYERS)
        )
        for path in _python_files(BASE)
    }
    assert {name: found for name, found in offenders.items() if found} == {}


@pytest.mark.parametrize("layer", LAYERS)
def test_framework_modules_never_import_their_predefined_package(layer):
    root = REPO_ROOT / "quantlab" / layer
    predefined = f"quantlab.{layer}.predefined"
    # Positive control: the predefined package exists and its modules are seen.
    assert (root / "predefined/__init__.py").is_file()
    offenders = {
        path.name: sorted(
            name for name in _resolved_imports(path) if _is_or_under(name, predefined)
        )
        for path in sorted(root.glob("*.py"))
    }
    assert {name: found for name, found in offenders.items() if found} == {}


@pytest.mark.parametrize("layer", LAYERS)
def test_predefined_init_files_are_empty(layer):
    for init in (REPO_ROOT / "quantlab" / layer / "predefined").rglob("__init__.py"):
        assert init.stat().st_size == 0, init


def test_the_portfolio_layer_never_imports_the_backtest_layer():
    root = REPO_ROOT / "quantlab/portfolio"
    files = _python_files(root)
    # Positive control: the shipped rules are seen.
    assert any(path.name == "top_n.py" for path in files)
    offenders = {
        path.name: sorted(
            name for name in _resolved_imports(path) if _is_or_under(name, "quantlab.backtest")
        )
        for path in files
    }
    assert {name: found for name, found in offenders.items() if found} == {}


def test_only_the_mean_variance_optimizer_imports_a_solver():
    importers = sorted(
        str(path.relative_to(REPO_ROOT))
        for path in _python_files(REPO_ROOT / "quantlab")
        if any(_is_or_under(name, "cvxpy") for name in _resolved_imports(path))
    )
    assert importers == ["quantlab/portfolio/predefined/mean_variance.py"]


def test_the_tracking_root_module_imports_no_tracking_library():
    code = (
        "import sys\n"
        "import quantlab.base.tracking\n"
        "print(sorted(name for name in ('wandb', 'mlflow') if name in sys.modules))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO_ROOT
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "[]"


@pytest.mark.parametrize("library", ["wandb", "mlflow"])
def test_only_its_tracker_imports_a_tracking_library(library):
    importers = sorted(
        str(path.relative_to(REPO_ROOT))
        for path in _python_files(REPO_ROOT / "quantlab")
        if any(_is_or_under(name, library) for name in _resolved_imports(path))
    )
    assert importers == [f"quantlab/tracking/{library}.py"]


def test_the_walk_forward_module_stays_in_utils():
    path = REPO_ROOT / "quantlab/utils/walk_forward.py"
    outside = sorted(
        name
        for name in _resolved_imports(path)
        if name.startswith("quantlab") and not name.startswith("quantlab.utils")
    )
    assert outside == [], outside


RUNS = REPO_ROOT / "quantlab/runs"
#: The base root modules the run layer may import: the component rule (rebuilds,
#: the component tree) and the prediction panel a backtest run reads.
RUNS_BASE_MODULES = ("quantlab.base.component", "quantlab.base.portfolio")


def test_the_run_layer_imports_only_utils_and_named_base_modules():
    files = _python_files(RUNS)
    # Positive control: the mechanism and the trained run are seen.
    assert {"directory.py", "trained_run.py", "backtest_run.py"} <= {
        path.name for path in files
    }
    allowed = ("quantlab.utils", "quantlab.runs", "quantlab.backend", *RUNS_BASE_MODULES)
    offenders = {
        path.name: sorted(
            name
            for name in _resolved_imports(path)
            if name.startswith("quantlab")
            and not any(_is_or_under(name, root) for root in allowed)
        )
        for path in files
    }
    assert {name: found for name, found in offenders.items() if found} == {}
    assert (RUNS / "__init__.py").stat().st_size == 0


def test_the_trained_run_module_imports_no_quantlab_layer():
    allowed = ("quantlab.utils", "quantlab.runs.directory")
    outside = sorted(
        name
        for name in _resolved_imports(RUNS / "trained_run.py")
        if name.startswith("quantlab") and not any(_is_or_under(name, a) for a in allowed)
    )
    assert outside == [], outside
