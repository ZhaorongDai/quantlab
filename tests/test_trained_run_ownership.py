"""ADR 0018: a training run's files are written and read only through
the run layer: `quantlab/runs/trained_run.py` and, for the `run.json` every run
type shares, the run-directory mechanism `quantlab/runs/directory.py` (#132).

What is locked here, and what turns it red, in the library, the examples and the
Python sessions (`>>>` / `...` lines) of the docs, docstrings included:

- a string literal naming a training run's record file (`run.json`, or the
  `cv_folds.json` / `ensemble.json` it replaced) or evaluation file (`ic_series.csv`,
  `test_predictions.zarr`), alone or as the last part of a path. Prose may name them;
  code reads them through `TrainedRun`. `config.json` and `metrics.json` are not
  checked: a backtest run directory writes files of the same names;
- walking up two directory levels (`.parent.parent`, `.parents[...]`), the way a trial
  directory used to be guessed from a fold's checkpoint. A line that starts from
  `__file__` (a package root) is exempt;
- taking `.parent` of a checkpoint: a unit's directory is `TrainedRun.open(...).path`.

Static, offline.
"""

import ast
import re

from tests.test_backtest_contracts import REPO_ROOT

OWNERS = (
    REPO_ROOT / "quantlab/runs/trained_run.py",
    REPO_ROOT / "quantlab/runs/directory.py",
)

_RECORD_FILES = (
    "run.json", "cv_folds.json", "ensemble.json", "ic_series.csv", "test_predictions.zarr"
)
_WALK_UP = "walks up two directory levels"
_RULES = {
    "names a trained-run file": re.compile(
        r"""["'](?:[^"'\n]*/)?(?:%s)["']""" % "|".join(map(re.escape, _RECORD_FILES))
    ),
    _WALK_UP: re.compile(r"\.parent\.parent\b|\.parents\["),
    "takes the directory of a checkpoint": re.compile(r"checkpoint[\w\"'\]\)]*\.parent\b"),
}
_SESSION_LINE = re.compile(r"^\s*(?:>>>|\.\.\.)\s?(.*)$")


def _session_code(line: str) -> str:
    """Return the code of a ``>>>`` / ``...`` line, or "" for prose and expected output."""
    match = _SESSION_LINE.match(line)
    return match.group(1) if match else ""


def _docstring_lines(tree: ast.Module) -> set:
    """Return the line numbers the docstrings of ``tree`` span."""
    spans = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            first = node.body[0] if node.body else None
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                spans.update(range(first.lineno, first.end_lineno + 1))
    return spans


def _python_sources():
    """Yield each library and example file with its code lines; a docstring keeps only its session code."""
    for root in ("quantlab", "examples"):
        for path in sorted((REPO_ROOT / root).rglob("*.py")):
            if path in OWNERS or "__pycache__" in path.parts:
                continue
            text = path.read_text(encoding="utf-8")
            docstrings = _docstring_lines(ast.parse(text, filename=str(path)))
            yield path, [
                _session_code(line) if number in docstrings else line
                for number, line in enumerate(text.splitlines(), start=1)
            ]


def _doc_sessions():
    """Yield each docs page with the code of its Python sessions; prose and output are blanked."""
    for path in sorted((REPO_ROOT / "docs").rglob("*.md")):
        if {"adr", "research"} & set(path.relative_to(REPO_ROOT / "docs").parts):
            continue
        yield path, [_session_code(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _offences(sources):
    """Return ``path:line: rule: code`` for every line a rule matches."""
    found = []
    for path, lines in sources:
        for number, line in enumerate(lines, start=1):
            for rule, pattern in _RULES.items():
                if rule == _WALK_UP and "__file__" in line:
                    continue
                if pattern.search(line):
                    found.append(f"{path.relative_to(REPO_ROOT)}:{number}: {rule}: {line.strip()}")
    return found


def test_only_the_trained_run_module_names_or_lays_out_run_files_in_code():
    assert _offences(_python_sources()) == []


def test_doc_sessions_read_runs_through_trained_run():
    assert _offences(_doc_sessions()) == []


def test_the_rules_catch_what_they_lock():
    caught = _offences(
        [
            (REPO_ROOT / "x.py", [
                'trial = Path(folds[0]["checkpoint"]).parent.parent',
                'manifest = json.loads((trial / "cv_folds.json").read_text())',
                'record = unit / "run.json"',
                'path = f"{unit}/ensemble.json"',
                'trial = checkpoint.parents[1]',
                'metrics = json.loads((checkpoint.parent / "metrics.json").read_text())',
            ]),
        ]
    )
    assert {offence.split(":")[1] for offence in caught} == {"1", "2", "3", "4", "5", "6"}
    clean = _offences(
        [
            (REPO_ROOT / "x.py", [
                'raise ValueError(f"{path} (its run.json) differs")',
                'ROOT = Path(__file__).resolve().parent.parent',
                'run = TrainedRun.open(checkpoint)',
                'json.loads((run_dir / "metrics.json").read_text())',
            ]),
        ]
    )
    assert clean == []
