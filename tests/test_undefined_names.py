"""No quantlab module reads a name it never defines or imports, or imports a name it never uses.

A moved import can leave a name behind that only one rarely called method reads: during
#148 ``BacktestConfig.to_dict`` kept calling ``config_to_dict`` after its import was
dropped, and no test called it, so the suite stayed green. ruff (a dev dependency) checks
every module statically for an undefined name (F821) and an unused import (F401), the
residue a move leaves on either side.

Runs ruff in a subprocess; offline.
"""

import subprocess
import sys

from tests.test_backtest_contracts import REPO_ROOT

_RULES = "F821,F401"


def _ruff(*paths) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "ruff", "check", "--select", _RULES, "--output-format", "concise",
         "--no-cache", *map(str, paths)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )


def test_no_module_reads_an_undefined_name_or_imports_an_unused_one():
    result = _ruff(REPO_ROOT / "quantlab")
    assert result.returncode == 0, result.stdout + result.stderr


def test_an_undefined_name_and_an_unused_import_are_caught(tmp_path):
    # Positive control: the #148 shape (a method reading a name never imported) and a
    # leftover import are reported; a function-level import is not.
    probe = tmp_path / "probe.py"
    probe.write_text(
        "import json\n"
        "class Config:\n"
        "    def to_dict(self):\n"
        "        from os import sep\n"
        "        return config_to_dict(self), sep\n",
        encoding="utf-8",
    )
    result = _ruff(probe)
    assert result.returncode == 1
    assert "F821 Undefined name `config_to_dict`" in result.stdout
    assert "F401 [*] `json` imported but unused" in result.stdout
    assert "sep" not in result.stdout
