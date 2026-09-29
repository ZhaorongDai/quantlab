"""The layer layout: `base/` holds the root base classes, each layer's top level holds its
extension framework, and `<layer>/predefined/` holds the shipped implementations.

What is locked here, and what turns it red:

- importing any `quantlab/base` module loads no torch (the training target of both model
  variants lives in `quantlab/model/training_target.py`, not on `BaseModel`);
- no `quantlab/base` module imports the factor, label, model or backtest layers;
- no layer's framework module (a top-level file of `quantlab/factor`, `quantlab/label`,
  `quantlab/model`, `quantlab/backtest`) imports that layer's `predefined` package, so a
  user's own factor, label, model or backtester needs nothing from the shipped ones;
- every `predefined` package's `__init__.py` is empty.

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
LAYERS = ("factor", "label", "model", "backtest")


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
