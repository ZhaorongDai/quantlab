"""Regression lock for the macOS OpenMP guard in `tests/conftest.py` (260914-lno).

On macOS, xgboost's wheel links Homebrew's libomp while torch bundles its own.
A process mixing both segfaults (torch then xgboost) or deadlocks (xgboost then
torch). `tests/conftest.py` sets `OMP_NUM_THREADS=1` on macOS, before any
import that can load torch, which is the only setting measured to survive the
full mixed sequence. A ctypes preload of Homebrew libomp was rejected because
it makes torch's own GRU / cross_entropy segfault.

What is locked, and what turns it red:

- (a) darwin only: a fresh interpreter that applies the guard exactly as
  conftest writes it and then runs torch -> xgboost -> torch -> threaded
  xgboost -> torch at realistic op sizes exits 0 within the timeout. Deleting
  or weakening the guard's effect turns it red (crash or timeout).
- (b) the guard is the first executable code in conftest, ahead of every
  other import, and on macOS the variable is set in this test process.
  Moving it below an import that can load torch turns it red.
- (c) the guard does nothing off macOS and never overrides an explicit
  value. Tested by executing conftest's own guard statement against a fake
  `sys`/`os`, so no platform patching of the running interpreter is needed.

There is deliberately NO test asserting that an unguarded process crashes: a
test whose purpose is to segfault is flaky and hostile. That control was
measured instead (2026-09-14, fresh subprocesses, 120 s timeout): without the
guard the same sequence exited -11 before its first torch op completed.
"""

import ast
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFTEST = REPO_ROOT / "tests" / "conftest.py"


def _conftest_body() -> list[ast.stmt]:
    body = ast.parse(CONFTEST.read_text(encoding="utf-8")).body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]  # module docstring
    return body


def _guard_node(body: list[ast.stmt] | None = None) -> ast.If:
    """The guard statement, looked up in `body` (a fresh parse if omitted).

    Pass the same `body` you index into: nodes from two separate parses never
    compare equal."""
    body = _conftest_body() if body is None else body
    guards = [
        node
        for node in body
        if isinstance(node, ast.If) and "OMP_NUM_THREADS" in ast.unparse(node)
    ]
    assert len(guards) == 1, f"expected one OMP_NUM_THREADS guard in conftest, got {len(guards)}"
    return guards[0]


def _run_guard(platform: str, environ: dict) -> dict:
    """Execute conftest's own guard statement against a fake sys/os."""
    code = compile(ast.Module(body=[_guard_node()], type_ignores=[]), str(CONFTEST), "exec")
    fake_sys = SimpleNamespace(platform=platform)
    fake_os = SimpleNamespace(environ=environ)
    exec(code, {"sys": fake_sys, "os": fake_os})
    return environ


# --------------------------------------------------------------------------
# (b) presence and ordering
# --------------------------------------------------------------------------


def test_guard_is_the_first_executable_code_in_conftest():
    """Only `import os`, `import sys` may precede the guard; everything else
    (numpy, polars, quantlab -- any of which can load torch) comes after."""
    body = _conftest_body()
    guard = _guard_node(body)
    before = body[: body.index(guard)]
    assert [ast.unparse(node) for node in before] == ["import os", "import sys"]
    assert ast.unparse(guard) == (
        "if sys.platform == 'darwin':\n    os.environ.setdefault('OMP_NUM_THREADS', '1')"
    )


@pytest.mark.skipif(sys.platform != "darwin", reason="the guard only acts on macOS")
def test_guard_took_effect_in_this_test_process():
    assert os.environ.get("OMP_NUM_THREADS"), "conftest guard did not set OMP_NUM_THREADS"


# --------------------------------------------------------------------------
# (c) no-op off macOS, explicit value wins
# --------------------------------------------------------------------------


@pytest.mark.parametrize("platform", ["linux", "win32"])
def test_guard_does_nothing_off_macos(platform):
    assert _run_guard(platform, {}) == {}


def test_guard_sets_single_threaded_openmp_on_macos():
    assert _run_guard("darwin", {}) == {"OMP_NUM_THREADS": "1"}


def test_guard_never_overrides_an_explicit_value():
    assert _run_guard("darwin", {"OMP_NUM_THREADS": "4"}) == {"OMP_NUM_THREADS": "4"}


# --------------------------------------------------------------------------
# (a) the mixed sequence survives under the guard
# --------------------------------------------------------------------------

MIXED_SEQUENCE = """
import numpy as np
import torch
import xgboost as xgb
from joblib import Parallel, delayed

def torch_ops():
    gru = torch.nn.GRU(32, 64, batch_first=True)
    gru(torch.randn(256, 50, 32))
    torch.nn.functional.cross_entropy(torch.randn(20000, 2), torch.randint(0, 2, (20000,)))

rng = np.random.default_rng(0)
X = rng.standard_normal((50000, 30)).astype("float32")
y = X[:, 0] + 0.1 * rng.standard_normal(50000).astype("float32")
d = xgb.DMatrix(X, label=y)

torch_ops(); print("torch-1", flush=True)
xgb.train({"tree_method": "hist"}, d, 30); print("xgb-main", flush=True)
torch_ops(); print("torch-2", flush=True)
Parallel(n_jobs=2, backend="threading")(
    delayed(xgb.train)({"tree_method": "hist"}, d, 30) for _ in range(2)
); print("xgb-threads", flush=True)
torch_ops(); print("torch-3", flush=True)
print("OMP_NUM_THREADS", os.environ.get("OMP_NUM_THREADS"), "torch_threads", torch.get_num_threads())
"""


@pytest.mark.skipif(
    sys.platform != "darwin",
    reason="the torch/xgboost libomp clash this guards against is macOS-specific",
)
def test_mixed_torch_xgboost_sequence_survives_under_the_guard():
    """Fresh interpreter, environment scrubbed of OMP_NUM_THREADS, then the
    guard statement exactly as conftest writes it, then the sequence."""
    script = "import os\nimport sys\n" + ast.unparse(_guard_node()) + "\n" + MIXED_SEQUENCE
    env = {k: v for k, v in os.environ.items() if k != "OMP_NUM_THREADS"}
    env["PYTHONPATH"] = str(REPO_ROOT)
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=240,
        )
    except subprocess.TimeoutExpired as exc:
        partial = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        pytest.fail(f"mixed sequence hung (>240 s); progress:\n{partial}")
    assert result.returncode == 0, (
        f"exit {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr[-2000:]}"
    )
    assert "torch-3" in result.stdout
    assert "OMP_NUM_THREADS 1 torch_threads 1" in result.stdout
