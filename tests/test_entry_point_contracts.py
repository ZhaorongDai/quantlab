"""Every repo-root entry point still satisfies the config contracts it uses.

775 tests passed on the commit that reinstated `DatasetConfig.market` as a
REQUIRED field (03.5 D-02) while `main.py` and `cal.py` -- two tracked entry
points, one of them the project's nominal `main.py` -- raised

    TypeError: DatasetConfig.__init__() missing 1 required keyword-only
    argument: 'market'

on their first line of real work. The suite could not see it for a structural
reason that will recur: no test imports a repo-root script, so the scripts are
outside every guarantee the `tests/` tree provides. This module closes exactly
that gap, and nothing wider.

**Why this is a STATIC scan rather than `runpy.run_path` over each script.**
Half the repo-root scripts do their work AT IMPORT -- `main.py` opens a
multi-gigabyte Zarr store, `train_model.py` builds factor graphs, `test.py`
reads one developer's absolute parquet path. Executing them in CI would fail
for reasons that have nothing to do with the contract under test, and a guard
that is red for unrelated reasons is a guard nobody keeps. The `ast` scan below
asks the one question the TypeError answered: does every config constructed by
a script pass every field that config REQUIRES. It needs no data, no
credential and no network, and it fails on the next required-field addition
rather than on a developer's next run.

The scripts that ARE import-safe (the ones behind an `if __name__ ==
"__main__":` guard) additionally get a real import, below -- so a broken import
line in an ingest shell is caught as an import, not as a pattern match.
"""

import ast
import dataclasses
import importlib
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The scripts in scope: every `*.py` at the repository ROOT. Discovered rather
#: than listed, because a list is exactly the thing that goes stale -- the next
#: entry point someone adds is covered by existing here, not by remembering.
ENTRY_POINTS = sorted(
    [*REPO_ROOT.glob("*.py"), *(REPO_ROOT / "scripts").glob("*.py")]
)


def _config_dataclasses() -> dict:
    """`{name: cls}` for every config dataclass a script could construct.

    Read off `quantlab.base.config` at run time rather than hardcoded: a new
    config class is covered the moment it exists, and a renamed one cannot
    leave a stale name silently matching nothing.
    """
    from quantlab.base import config as config_module

    return {
        name: obj
        for name, obj in vars(config_module).items()
        if isinstance(obj, type) and dataclasses.is_dataclass(obj)
    }


def _required_fields(cls) -> set[str]:
    """Fields with NO default -- the ones whose absence is a TypeError."""
    return {
        field.name
        for field in dataclasses.fields(cls)
        if field.default is dataclasses.MISSING
        and field.default_factory is dataclasses.MISSING
    }


def _construction_sites(path: Path, known: dict) -> list[tuple[str, int, set]]:
    """`(config_name, lineno, keywords)` for every config constructed here.

    A `**kwargs` call is SKIPPED rather than failed: its field set is not
    knowable statically, and guessing would make this scan lie in the
    direction that costs a real edit.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    sites = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = (
            func.id
            if isinstance(func, ast.Name)
            else func.attr
            if isinstance(func, ast.Attribute)
            else None
        )
        if name not in known:
            continue
        if any(keyword.arg is None for keyword in node.keywords):
            continue
        sites.append(
            (name, node.lineno, {kw.arg for kw in node.keywords})
        )
    return sites


@pytest.mark.parametrize(
    "script", ENTRY_POINTS, ids=lambda path: path.name
)
def test_every_config_a_root_script_builds_passes_every_required_field(
    script: Path,
) -> None:
    """THE regression. `main.py:7` and `cal.py:22` both built a
    `DatasetConfig` without `market` after D-02 made it required.

    RED under: adding a required field to any config dataclass without walking
    the repo-root scripts -- which is precisely what happened, and what the
    775-test suite was blind to.
    """
    known = _config_dataclasses()
    for name, lineno, provided in _construction_sites(script, known):
        missing = _required_fields(known[name]) - provided
        assert not missing, (
            f"{script.name}:{lineno} constructs {name} without "
            f"{sorted(missing)}. Every field listed is required (no default), "
            f"so this call raises TypeError the moment the script runs."
        )


def test_the_scan_actually_reaches_the_two_scripts_that_broke() -> None:
    """A scan that matched nothing would pass every parametrisation above
    while guaranteeing nothing at all. This pins that `main.py` and `cal.py` --
    the two CR-03 casualties -- are genuinely inspected, and that
    `DatasetConfig` is the class it inspects them for.
    """
    known = _config_dataclasses()
    assert "DatasetConfig" in known
    assert "market" in _required_fields(known["DatasetConfig"])

    for filename in ("main.py", "cal.py"):
        sites = _construction_sites(REPO_ROOT / filename, known)
        assert [name for name, _, _ in sites].count("DatasetConfig") == 1, (
            f"{filename} no longer builds exactly one DatasetConfig; this "
            f"test's premise moved and the scan may be inspecting nothing."
        )
        for _, _, provided in sites:
            assert "market" in provided


def _is_import_safe(path: Path) -> bool:
    """True when the module's top level only DEFINES things.

    Approximated by the presence of an `if __name__ == "__main__":` guard,
    which is this repository's own marker for "running it is opt-in": the
    ingest shells have one, the exploratory scratch scripts do not.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return any(
        isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
        for node in tree.body
    )


IMPORT_SAFE_ENTRY_POINTS = [p for p in ENTRY_POINTS if _is_import_safe(p)]


@pytest.mark.parametrize(
    "script", IMPORT_SAFE_ENTRY_POINTS, ids=lambda path: path.name
)
def test_every_main_guarded_entry_point_imports(script: Path) -> None:
    """The literal smoke test, over the subset where importing is meaningful.

    A module-level `from quantlab.x import Y` that no longer resolves is not
    something the `ast` scan above can see, and it breaks the shell just as
    completely as a missing config field.
    """
    sys.path.insert(0, str(REPO_ROOT))
    try:
        importlib.import_module(script.stem)
    finally:
        sys.path.remove(str(REPO_ROOT))


def test_the_import_smoke_test_covers_the_ingest_shells() -> None:
    """`_is_import_safe` is a heuristic, so the set it produces is asserted
    rather than trusted: the four ingest shells are the entry points a user is
    most likely to run, and they must be in it."""
    covered = {path.name for path in IMPORT_SAFE_ENTRY_POINTS}
    for shell in (
        "ingest_tiingo.py",
        "ingest_alpaca.py",
        "ingest_us_equity.py",
        "ingest_binance_spot.py",
    ):
        assert shell in covered
