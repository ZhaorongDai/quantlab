"""The code record of a run: which quantlab commit, which source, which libraries.

A run is reproducible from its config, its data and its code. ``code_record``
records the code part for one component tree: the git commit of the quantlab
repository (and whether its tracked files had uncommitted changes), a sha256
digest of the source file of every module that defines a class of the tree,
and the versions of the libraries whose behaviour decides a run's numbers.

A quantlab module outside the shipped implementations is a *framework*
module: ``quantlab.base`` and each layer's extension framework, such as
``quantlab.factor.polars`` or ``quantlab.portfolio.decision_inputs``. Every
other recorded module is a *component* module: the shipped implementations
(``quantlab.<layer>.predefined``, the datasets of ``quantlab.dataset``) and a
user's own classes. Standard-library and installed third-party modules are
not recorded, nor a class without a source file (one defined in a notebook);
the libraries' versions are. quantlab's own modules are recorded however it
is installed.

``dirty`` reports uncommitted changes to tracked files only; an untracked
file is not counted. ``git`` is ``None`` when quantlab is not imported from a
git working tree (an installed copy, say).

``compare_code`` compares two records the way a rebuild needs: a differing,
missing or extra module digest and a differing library version each log a
warning, framework modules apart from component modules; the git commit is
context and is not compared. Nothing is raised.

This module has no project-internal imports.
"""

import hashlib
import importlib.metadata
import inspect
import subprocess
import sys
import sysconfig
from collections.abc import Iterable, Mapping
from pathlib import Path

from loguru import logger

__all__ = ["LIBRARIES", "code_record", "compare_code"]

#: The distributions whose versions a code record holds, when installed.
LIBRARIES = (
    "numpy",
    "pandas",
    "xarray",
    "polars",
    "xgboost",
    "torch",
    "vectorbt",
    "KunQuant",
    "cvxpy",
)

#: Directories of installed third-party and standard-library code.
_INSTALLED = tuple(
    Path(path).resolve()
    for key in ("stdlib", "platstdlib", "purelib", "platlib")
    if (path := sysconfig.get_paths().get(key))
)


def code_record(components: Iterable[tuple[str, object]]) -> dict:
    """Return the code record of a component tree.

    Parameters
    ----------
    components : iterable of (str, object) pairs
        Every component of the run with its component path; the root's path
        is ``""``. ``walk_components`` yields the rest of a tree.

    Returns
    -------
    dict
        ``git``: ``{"commit", "dirty"}`` of the repository quantlab is
        imported from, or ``None`` outside one. ``modules``: by module name,
        ``{"sha256", "framework", "components"}``, the component paths whose
        class or a base class of it the module defines. ``libraries``: by
        distribution name, the installed version, for those of ``LIBRARIES``
        that are installed.

    Examples
    --------
    >>> record = code_record([("", backtester), *walk_components(backtester)])
    >>> sorted(record)
    ['git', 'libraries', 'modules']
    >>> record["modules"]["tests.backtest_fixtures"]["framework"]
    False
    >>> record["modules"]["quantlab.base.factor"]["framework"]
    True
    """
    modules: dict[str, dict] = {}
    for path, item in components:
        for cls in type(item).__mro__:
            source = _source_file(cls)
            if source is None:
                continue
            entry = modules.setdefault(
                cls.__module__,
                {
                    "sha256": _digest(source),
                    "framework": _is_framework(cls.__module__),
                    "components": [],
                },
            )
            if path not in entry["components"]:
                entry["components"].append(path)
    return {
        "git": _git(),
        "modules": dict(sorted(modules.items())),
        "libraries": _libraries(),
    }


def compare_code(expected: Mapping, actual: Mapping, *, owner: str) -> None:
    """Compare two code records, logging one warning per difference.

    Module digests and library versions are compared; the git commit is
    context only. Component modules are reported before framework modules,
    each warning naming the module, whether it is a component or a framework
    module, and the component paths using it. Nothing is raised: a run on
    changed code still runs.

    Parameters
    ----------
    expected, actual : mapping
        ``code_record`` of the earlier run and of this one.
    owner : str
        The name that opens every warning.

    Examples
    --------
    ``run`` is a ``BacktestRun``; its rebuilt backtester's code is compared::

        compare_code(run.code, code_of(rebuilt), owner=str(run.path))
    """
    wanted = dict(expected.get("modules") or {})
    got = dict(actual.get("modules") or {})
    names = list(dict.fromkeys([*wanted, *got]))
    names.sort(key=lambda name: (_entry(name, wanted, got)["framework"], name))
    for name in names:
        entry = _entry(name, wanted, got)
        kind = "framework module" if entry["framework"] else "component module"
        used = ", ".join(repr(path) for path in entry["components"]) or "no component"
        if name not in got:
            problem = "in the expected record but not used by this run"
        elif name not in wanted:
            problem = "used by this run but absent from the expected record"
        elif wanted[name].get("sha256") != got[name].get("sha256"):
            problem = "its source changed since the expected run"
        else:
            continue
        logger.warning(
            f"{owner}: code mismatch for {kind} {name!r} (used by {used}): "
            f"{problem}; continuing"
        )
    old = dict(expected.get("libraries") or {})
    new = dict(actual.get("libraries") or {})
    for name in list(dict.fromkeys([*old, *new])):
        if old.get(name) != new.get(name):
            logger.warning(
                f"{owner}: code mismatch for library {name!r}: expected version "
                f"{old.get(name)}, got {new.get(name)}; continuing"
            )


def _entry(name: str, wanted: dict, got: dict) -> dict:
    """Return the record of module ``name`` from this run, else the expected one."""
    return got.get(name) or wanted[name]


def _is_framework(name: str) -> bool:
    """Return whether module ``name`` is a quantlab framework module (see the module docs)."""
    parts = name.split(".")
    return (
        parts[0] == "quantlab"
        and "predefined" not in parts
        and parts[:2] != ["quantlab", "dataset"]
    )


def _is_quantlab(name: str) -> bool:
    """Return whether module ``name`` is part of quantlab."""
    return name == "quantlab" or name.startswith("quantlab.")


def _source_file(cls: type) -> Path | None:
    """Return the source file of the module defining ``cls``, unless it is installed code.

    quantlab's own modules are never installed code here, however quantlab
    is installed.
    """
    module = sys.modules.get(cls.__module__)
    if module is None or getattr(module, "__file__", None) is None:
        return None
    try:
        source = Path(inspect.getsourcefile(module) or module.__file__).resolve()
    except TypeError:  # a built-in or extension module
        return None
    if source.suffix != ".py" or not source.is_file():
        return None
    if not _is_quantlab(cls.__module__) and _installed(source):
        return None
    return source


def _installed(path: Path) -> bool:
    """Return whether ``path`` lies in the standard library or an installed package."""
    return any(path.is_relative_to(root) for root in _INSTALLED)


def _digest(path: Path) -> str:
    """Return the sha256 of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git() -> dict | None:
    """Return the commit and dirty flag of the repository quantlab lives in, or None."""
    package = sys.modules.get("quantlab")
    if package is None or getattr(package, "__file__", None) is None:
        return None
    directory = Path(package.__file__).resolve().parent
    if _installed(directory):
        # An installed copy: a repository around the environment is not quantlab's.
        return None

    def git(*args: str) -> str | None:
        try:
            done = subprocess.run(
                ["git", "-C", str(directory), *args],
                capture_output=True, text=True, timeout=10, check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return done.stdout if done.returncode == 0 else None

    commit = git("rev-parse", "HEAD")
    if commit is None:
        return None
    status = git("status", "--porcelain", "--untracked-files=no")
    return {"commit": commit.strip(), "dirty": bool(status and status.strip())}


def _libraries() -> dict:
    """Return the installed version of each of ``LIBRARIES``, without importing them."""
    versions = {}
    for name in LIBRARIES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions
