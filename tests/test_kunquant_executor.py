"""Locks `quantlab/factor/kunquant.py:shared_executor`, the KunQuant executor seam.

KunQuant's multi-thread executor can deadlock in its destructor: a worker that
has read the `closing` flag but not yet parked misses the destructor's wake-up,
and the destructor's `join` then waits forever while holding the GIL. The whole
process freezes. The
seam is therefore "no executor is destroyed while the program runs": every
KunQuant run takes its executor from `shared_executor`, which creates one per
thread count and releases it only at exit, after its workers have settled.

The stress test runs in a subprocess with a time limit, so a regression fails
the test instead of hanging the suite. Creating and dropping an executor per
run hangs that subprocess within a few hundred to a thousand runs.
"""

import ast
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from quantlab.factor.kunquant import shared_executor

_ROOT = Path(__file__).resolve().parents[1]

# 2000 batch runs and 2000 stream contexts over three thread counts. With a
# fresh executor per run this hangs in essentially every attempt (measured:
# hangs after 331, 630 and 869 create/drop cycles at 4 threads).
_STRESS = textwrap.dedent(
    """
    import numpy as np
    from KunQuant.Driver import KunCompilerConfig
    from KunQuant.jit import cfake
    from KunQuant.Op import Builder, Input, Output, Rank
    from KunQuant.ops import WindowedAvg
    from KunQuant.runner import KunRunner as kr
    from KunQuant.Stage import Function

    from quantlab.factor.kunquant import shared_executor

    def function():
        builder = Builder()
        with builder:
            close = Input("close")
            Output(WindowedAvg(close, 5), "ma")
            Output(Rank(close), "rank")
        return Function(builder.ops)

    lib = cfake.compileit(
        [
            ("Batch", function(), KunCompilerConfig(input_layout="TS", output_layout="TS")),
            ("Stream", function(), KunCompilerConfig(input_layout="STREAM", output_layout="STREAM")),
        ],
        "test_kunquant_executor_stress",
        cfake.CppCompilerConfig(),
    )
    batch, stream = lib.getModule("Batch"), lib.getModule("Stream")
    close = np.random.default_rng(0).random((50, 16)).astype(np.float32)
    for i in range(2000):
        threads = (2, 4, 8)[i % 3]
        kr.runGraph(shared_executor(threads), batch, {"close": close}, 0, 50)
        ctx = kr.StreamContext(shared_executor(threads), stream, 16)
        handle = ctx.queryBufferHandle("close")
        for t in range(3):
            ctx.pushData(handle, close[t])
            ctx.run()
        del ctx
    print("finished")
    """
)


def test_shared_executor_is_one_per_thread_count():
    """The same thread count returns the same executor; another count does not."""
    assert shared_executor(3) is shared_executor(3)
    assert shared_executor(3) is not shared_executor(5)


def test_repeated_runs_on_shared_executors_finish():
    """Thousands of batch and stream runs, then a normal exit, finish in time.

    The subprocess exits right after its last run, so the exit-time release
    of the executors is exercised too.
    """
    try:
        done = subprocess.run(
            [sys.executable, "-c", _STRESS],
            cwd=_ROOT,
            capture_output=True,
            text=True,
            timeout=90,
        )
    except subprocess.TimeoutExpired:
        pytest.fail("KunQuant runs deadlocked: an executor was destroyed (issue #74)")
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "finished"


class _ConstructorFinder(ast.NodeVisitor):
    """Collect the enclosing function of every ``createMultiThreadExecutor`` call."""

    def __init__(self) -> None:
        self.scope = ["<module>"]
        self.found: set[str] = set()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Attribute) and node.func.attr == "createMultiThreadExecutor":
            self.found.add(self.scope[-1])
        self.generic_visit(node)


def test_only_shared_executor_constructs_executors():
    """No module under quantlab/ or tests/ builds its own multi-thread executor.

    A locally built executor is destroyed when its last reference goes, which
    is the deadlock this seam exists to avoid.
    """
    offenders = {}
    for path in sorted([*(_ROOT / "quantlab").rglob("*.py"), *(_ROOT / "tests").rglob("*.py")]):
        finder = _ConstructorFinder()
        finder.visit(ast.parse(path.read_text(), filename=str(path)))
        if finder.found:
            offenders[str(path.relative_to(_ROOT))] = sorted(finder.found)
    assert offenders == {"quantlab/factor/kunquant.py": ["shared_executor"]}
