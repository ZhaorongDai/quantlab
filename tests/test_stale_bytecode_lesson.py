"""Executable lesson: an equal-length source edit that preserves mtime leaves
the OLD bytecode live (phase 03.11 gap G-03.11-4).

CPython decides whether a cached `__pycache__/*.pyc` is stale by comparing two
numbers recorded in its 16-byte header against the source file on disk: the
source's **mtime (whole seconds)** and its **size in bytes**. Both unchanged
means "fresh" -- the source text is never read, never hashed, never compared.

This phase walked into that. A reproduction experiment edited a module in
place, swapping one identifier for a **deliberately equal-length** misspelling,
and later restored the original text. Size was identical by construction, the
restore preserved mtime, so the `.pyc` compiled from the poisoned text stayed
"fresh" and kept executing. The measured consequence: the same clean source
tree produced `5 failed, 21 passed` on
`tests/test_crsp_ticker_sidecar.py`, while an identical run against a clean
cache directory produced `98 passed` across the four suites. `git status` was
empty and a content grep found nothing, because nothing WAS wrong with the
source -- the divergence lived only in the derived artifact.

What is locked, and what turns it red:

- (a) the trap itself: after an equal-length edit with the original mtime
  restored, a brand-new interpreter still imports the OLD constant. If CPython
  ever strengthened its invalidation criterion (content hashing by default,
  sub-second mtime in the header), this goes red and the lesson below is
  obsolete -- read the failure before deleting anything.
- (b) `touch` is an effective remedy: pushing the source mtime forward makes
  the very next interpreter recompile. Deleting the `os.utime(path, None)`
  call turns it red.
- (c) a clean `-X pycache_prefix=<dir>` is the other effective remedy, and is
  the one this phase's verification used to get an uncontaminated measurement.

Two safe ways to run a source-injection experiment, both proved above:

1. Do not edit the file. Inject through a subclass or a `monkeypatch.setattr`
   -- no source on disk changes, so no cache can disagree with it. This phase's
   03.11-REVIEW.md used this for its WR-02 reproduction.
2. If the file must be edited, `touch` it after restoring (or delete the
   cache: `find . -path ./.venv -prune -o -name '__pycache__' -type d
   -print0 | xargs -0 rm -rf`). Never trust size alone to prove a revert.

The two real misspellings this phase injected are deliberately NOT written out
anywhere in this file: the acceptance criterion for that cleanup is a
`git grep` for those two tokens returning zero matches across `quantlab/` and
`tests/`, and a mention in prose is indistinguishable from a leftover to grep.
The neutral constants `"AAAA"` / `"BBBB"` carry the equal-length property,
which is the only property the trap actually needs.

Every probe below runs in its OWN subprocess. This is not tidiness: within one
interpreter, `importlib.reload` re-executes an already-loaded module object and
`sys.modules` short-circuits a plain re-import, so neither consults the `.pyc`
invalidation criterion at all. An in-process version of this file would pass
while testing nothing.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

MODULE_NAME = "stale_bytecode_probe"

# The two constants MUST be the same length -- that is the whole mechanism.
OLD_VALUE = "AAAA"
NEW_VALUE = "BBBB"
assert len(OLD_VALUE) == len(NEW_VALUE)

PROBE_TIMEOUT = 60


def _source(value: str) -> str:
    return f'VALUE = "{value}"\n\n\ndef value() -> str:\n    return VALUE\n'


def _child_env() -> dict[str, str]:
    """Environment for a probe interpreter.

    Scrubbed of the three variables that would silently disable the mechanism
    under test: no bytecode written at all, no implicit cwd on `sys.path`, or
    a cache directory relocated out from under the test.
    """
    env = dict(os.environ)
    for key in ("PYTHONDONTWRITEBYTECODE", "PYTHONSAFEPATH", "PYTHONPYCACHEPREFIX"):
        env.pop(key, None)
    return env


def _probe(workdir: Path, *interpreter_args: str) -> subprocess.CompletedProcess:
    """Import the synthetic module in a FRESH interpreter and print its value.

    `cwd` is what puts the synthetic module on the import path -- `python -c`
    prepends the current directory to `sys.path`. Nothing mutates this
    process's own `sys.path`.
    """
    code = f"import {MODULE_NAME} as m; print(m.value())"
    try:
        return subprocess.run(
            [sys.executable, *interpreter_args, "-c", code],
            cwd=workdir,
            env=_child_env(),
            capture_output=True,
            text=True,
            timeout=PROBE_TIMEOUT,
        )
    except subprocess.TimeoutExpired as exc:  # pragma: no cover - defensive
        pytest.fail(f"probe interpreter hung (>{PROBE_TIMEOUT}s) in {workdir}")


def _value_of(result: subprocess.CompletedProcess, what: str) -> str:
    """The probe's answer, or a failure that says WHY there is no answer.

    A bare "expected BBBB, got AAAA" cannot distinguish the trap firing from a
    child that never started, so stdout and stderr ride along on every path.
    """
    assert result.returncode == 0, (
        f"{what}: probe exited {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert lines, f"{what}: probe printed nothing\nstderr:\n{result.stderr}"
    return lines[-1].strip()


@pytest.fixture
def stale_bytecode(tmp_path: Path) -> Path:
    """A module whose source says `BBBB` and whose live bytecode says `AAAA`.

    Builds the trap and asserts it is armed, so a test consuming this fixture
    starts from a proven-poisoned state rather than an assumed one.
    """
    module = tmp_path / f"{MODULE_NAME}.py"
    module.write_text(_source(OLD_VALUE), encoding="utf-8")

    # Age the source before the first compile. The header stores mtime in WHOLE
    # SECONDS, so a module written, poisoned and touched inside one second is
    # judged fresh throughout and the `touch` remedy silently does nothing --
    # a real property of the criterion, and a test flake if left to wall-clock
    # luck. An hour of backdating makes every later comparison unambiguous.
    aged_ns = time.time_ns() - 3600 * 1_000_000_000
    os.utime(module, ns=(aged_ns, aged_ns))

    first = _probe(tmp_path)
    assert _value_of(first, "first import") == OLD_VALUE

    cached = sorted((tmp_path / "__pycache__").glob(f"{MODULE_NAME}.*.pyc"))
    assert cached, (
        "the first import wrote no .pyc, so there is no cache to go stale; "
        f"contents: {sorted(p.name for p in tmp_path.rglob('*'))}"
    )

    before = module.stat()

    # The equal-length edit: same byte count, different text.
    module.write_text(_source(NEW_VALUE), encoding="utf-8")
    after_edit = module.stat()
    assert after_edit.st_size == before.st_size, (
        "the two constants are not the same length -- the trap needs an edit "
        "that leaves st_size untouched"
    )

    # Restoring mtime is what completes the trap: both header fields now match.
    os.utime(module, ns=(before.st_atime_ns, before.st_mtime_ns))
    restored = module.stat()
    assert restored.st_mtime_ns == before.st_mtime_ns
    assert restored.st_size == before.st_size

    return module


def test_equal_length_edit_with_restored_mtime_keeps_the_old_bytecode_live(
    stale_bytecode: Path,
) -> None:
    """(a) The lesson. Source says BBBB; a brand-new interpreter says AAAA."""
    result = _probe(stale_bytecode.parent)
    observed = _value_of(result, "import after an equal-length edit")
    assert observed == OLD_VALUE, (
        f"expected the STALE value {OLD_VALUE!r} (the trap firing) but got "
        f"{observed!r}: CPython invalidated the cache despite an unchanged "
        "mtime+size, so the invalidation criterion has changed. Re-derive the "
        "lesson in this module's docstring rather than deleting it."
    )
    assert stale_bytecode.read_text(encoding="utf-8") == _source(NEW_VALUE), (
        "the source on disk must say BBBB -- that divergence from the running "
        "bytecode IS what this test demonstrates"
    )


def test_touching_the_source_invalidates_the_stale_bytecode(
    stale_bytecode: Path,
) -> None:
    """(b) Remedy 1: push mtime forward. Next interpreter recompiles."""
    os.utime(stale_bytecode, None)  # this is `touch`
    result = _probe(stale_bytecode.parent)
    observed = _value_of(result, "import after touching the source")
    assert observed == NEW_VALUE, (
        f"expected {NEW_VALUE!r} after touching the source, got {observed!r} "
        "-- touch is no longer an effective remedy; delete the cache instead"
    )


def test_a_clean_pycache_prefix_also_bypasses_the_stale_bytecode(
    stale_bytecode: Path, tmp_path: Path
) -> None:
    """(c) Remedy 2: a cache directory with nothing in it.

    The source file is not touched here, so the poisoned `__pycache__` is still
    sitting next to it and still judged fresh -- the prefix simply sends the
    interpreter somewhere else to look. This is the forensic move this phase's
    verification used to obtain an uncontaminated measurement; it is NOT a fix,
    because the default cache path stays poisoned for everyone else.
    """
    prefix = tmp_path / "clean_pycache_prefix"
    result = _probe(stale_bytecode.parent, "-X", f"pycache_prefix={prefix}")
    observed = _value_of(result, "import under a clean pycache_prefix")
    assert observed == NEW_VALUE, (
        f"expected {NEW_VALUE!r} under a clean pycache_prefix, got {observed!r}"
    )
    assert (stale_bytecode.parent / "__pycache__").exists(), (
        "the poisoned default cache must still be on disk -- this remedy "
        "sidesteps it rather than removing it"
    )
