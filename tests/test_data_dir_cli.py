"""The `--data-dir` per-run storage root: precedence, the two shared CLI
helpers, and the ordering that makes an override actually reach disk.

The knob is one knob with three levels -- CLI override > `QUANTLAB_DATA_DIR` >
repo-root `data/` (260907-rjq D-01). These tests pin every cell of that
precedence, the reset contract that keeps a process-global override from
leaking, and the end-to-end path from argv through a real config factory.

Every test that cares about the env layer sets or deletes `QUANTLAB_DATA_DIR`
explicitly: a developer machine may have it exported, and an assertion that
silently depends on that is an assertion that passes for the wrong reason.
"""

import argparse
import ast
import importlib
from pathlib import Path

import pytest

import quantlab.config as config
from quantlab.utils.cli import add_data_dir_arg, apply_data_dir

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Full dotted name of the configuration package. Matched by EQUALITY,
#: never by first dot-separated segment: every first-party import now
#: shares one umbrella package, so a leading-segment comparison would
#: sweep in every sibling package or none at all.
CONFIG_MODULE = "quantlab.config"


# ---------------------------------------------------------------------------
# config.get_data_root() / config.set_data_root() -- D-01, D-06, D-07, D-08
# ---------------------------------------------------------------------------


def test_override_wins_when_env_is_unset(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("QUANTLAB_DATA_DIR", raising=False)
    config.set_data_root(tmp_path)

    assert config.get_data_root() == tmp_path


def test_override_wins_over_the_env_var(monkeypatch, tmp_path) -> None:
    """D-01 cell 2: the CLI beats the environment. This is the whole point of
    the flag -- a per-run root on a machine that already exports one.
    """
    monkeypatch.setenv("QUANTLAB_DATA_DIR", str(tmp_path / "from_env"))
    config.set_data_root(tmp_path / "from_cli")

    assert config.get_data_root() == tmp_path / "from_cli"


def test_env_var_wins_when_the_override_is_cleared(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("QUANTLAB_DATA_DIR", str(tmp_path / "from_env"))
    config.set_data_root(None)

    assert config.get_data_root() == Path(str(tmp_path / "from_env"))


def test_repo_default_when_neither_override_nor_env_is_set(monkeypatch) -> None:
    monkeypatch.delenv("QUANTLAB_DATA_DIR", raising=False)
    config.set_data_root(None)

    assert config.get_data_root() == REPO_ROOT / "data"


def test_set_data_root_accepts_a_string_and_a_path(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("QUANTLAB_DATA_DIR", raising=False)

    assert config.set_data_root(str(tmp_path)) == tmp_path
    assert config.get_data_root() == tmp_path
    assert config.set_data_root(tmp_path / "sub") == tmp_path / "sub"
    assert config.get_data_root() == tmp_path / "sub"


def test_set_data_root_expands_a_tilde(monkeypatch) -> None:
    """D-07: a quoted `--data-dir '~/quantlab-data'` reaches Python with a
    literal tilde; without expansion the run creates a directory named `~`.
    """
    monkeypatch.delenv("QUANTLAB_DATA_DIR", raising=False)
    stored = config.set_data_root("~/quantlab-data-xyz")

    assert "~" not in str(stored)
    assert stored == Path.home() / "quantlab-data-xyz"
    assert config.get_data_root() == stored


def test_set_data_root_does_not_resolve_symlinks(monkeypatch, tmp_path) -> None:
    """D-07: the env knob is `Path(env_value)` with no resolution, so resolving
    only the CLI path would make the two knobs behave differently on a
    symlinked root and defeat a user who passed a symlink deliberately.
    """
    monkeypatch.delenv("QUANTLAB_DATA_DIR", raising=False)
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)

    assert config.set_data_root(link) == link
    assert config.get_data_root() == link
    assert config.get_data_root() != real


def test_set_data_root_neither_creates_nor_requires_the_directory(
    monkeypatch, tmp_path
) -> None:
    """D-08: the acquisition layer creates its own directories and the env knob
    validates nothing today; validating one knob and not the other is how two
    knobs drift apart.
    """
    monkeypatch.delenv("QUANTLAB_DATA_DIR", raising=False)
    missing = tmp_path / "does" / "not" / "exist"

    assert config.set_data_root(missing) == missing
    assert not missing.exists()
    assert config.get_data_root() == missing


def test_set_data_root_none_clears_and_returns_none(monkeypatch, tmp_path) -> None:
    """D-06: a process-global knob needs a reset, or the first test that sets
    it silently redirects every later test in the session.
    """
    monkeypatch.delenv("QUANTLAB_DATA_DIR", raising=False)
    config.set_data_root(tmp_path)

    assert config.set_data_root(None) is None
    assert config.get_data_root() == REPO_ROOT / "data"


@pytest.mark.parametrize("value", ["", "   ", "\t"])
def test_set_data_root_rejects_an_empty_value(value: str) -> None:
    """D-07: an env var of `""` is already falsy and falls through to the
    default, but a user who typed `--data-dir ""` meant something, and silently
    meaning "repo default" is wrong.
    """
    with pytest.raises(ValueError):
        config.set_data_root(value)


# ---------------------------------------------------------------------------
# utils.cli.add_data_dir_arg() / apply_data_dir() -- D-03, D-04
# ---------------------------------------------------------------------------


def test_add_data_dir_arg_registers_the_flag_and_returns_the_parser() -> None:
    parser = argparse.ArgumentParser()

    assert add_data_dir_arg(parser) is parser
    assert parser.parse_args([]).data_dir is None
    assert parser.parse_args(["--data-dir", "/X"]).data_dir == "/X"


def test_apply_data_dir_is_a_no_op_when_the_flag_is_absent(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.delenv("QUANTLAB_DATA_DIR", raising=False)
    config.set_data_root(tmp_path)

    assert apply_data_dir(argparse.Namespace(data_dir=None)) is None
    assert config.get_data_root() == tmp_path

    # An args object that never saw `add_data_dir_arg` must not raise either --
    # `apply_data_dir` is safe to call from a script that has not (yet) wired
    # the flag.
    assert apply_data_dir(argparse.Namespace()) is None
    assert config.get_data_root() == tmp_path


def test_apply_data_dir_sets_the_root_and_returns_it(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("QUANTLAB_DATA_DIR", str(tmp_path / "from_env"))

    stored = apply_data_dir(argparse.Namespace(data_dir=str(tmp_path)))

    assert stored == tmp_path
    assert config.get_data_root() == tmp_path


def test_utils_cli_does_not_import_config_at_module_scope() -> None:
    """D-04: `quantlab/utils/cli.py`'s module docstring pins its module-scope
    project dependency surface at `quantlab.base.chunking` and
    `quantlab.base.data`. Importing `set_data_root` from the configuration
    package at module scope would drag `quantlab.dataset.backend`,
    `quantlab.dataset.spot`, `quantlab.dataset.stock` and `quantlab.base.config`
    into every import of the dependency-light CLI helper module.
    """
    tree = ast.parse((REPO_ROOT / "quantlab" / "utils" / "cli.py").read_text())

    module_scope_imports: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            module_scope_imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module_scope_imports.add(node.module or "")

    assert not {
        name for name in module_scope_imports if name == CONFIG_MODULE
    }, (
        "quantlab/utils/cli.py imports the configuration package at module "
        "scope; D-04 requires the "
        "import inside apply_data_dir() so the dependency-light CLI module "
        "does not pull in the whole dataset layer."
    )


# ---------------------------------------------------------------------------
# End to end: argv -> apply_data_dir -> set_data_root -> a real factory path
# ---------------------------------------------------------------------------


def test_data_dir_relocates_the_universe_table(monkeypatch, tmp_path) -> None:
    import refresh_us_equity_universe as script

    monkeypatch.delenv("QUANTLAB_DATA_DIR", raising=False)
    args = script._build_arg_parser().parse_args(["--data-dir", str(tmp_path)])
    apply_data_dir(args)

    cfg = config.universe_config()

    assert cfg.output_path.startswith(str(tmp_path))
    assert cfg.cache_dir.startswith(str(tmp_path))


def test_omitting_data_dir_leaves_the_universe_paths_untouched(
    monkeypatch,
) -> None:
    """DDIR-03: the flag is additive. With it omitted, paths must be identical
    to what they were before the flag existed -- for repo-default users here,
    and for `QUANTLAB_DATA_DIR` users in `tests/test_config_paths.py`.
    """
    import refresh_us_equity_universe as script

    monkeypatch.delenv("QUANTLAB_DATA_DIR", raising=False)
    args = script._build_arg_parser().parse_args([])
    apply_data_dir(args)

    cfg = config.universe_config()

    # The doubled `data/data/` is today's convention, not a typo: the storage
    # ROOT defaults to the repo's `data/` directory and every factory nests its
    # own `data/` (or `downloads/`) subtree beneath whatever root answered.
    # DDIR-03 is about reproducing today's paths exactly, so this pins the
    # literal rather than the shape.
    assert cfg.output_path == str(
        REPO_ROOT / "data" / "data" / "reference" / "universe.parquet"
    )
    assert cfg.cache_dir == str(REPO_ROOT / "data" / "data" / "reference" / "_cache")


# ---------------------------------------------------------------------------
# DDIR-04: the override must be applied BEFORE anything that reaches a
# `config/` factory. Enforced structurally, over a DERIVED script list.
# ---------------------------------------------------------------------------

#: The entry points expected to offer `--data-dir`. This literal is the
#: EXPECTATION; the list actually guarded below is globbed from the repo, so a
#: sixth entry point that adds the flag is covered automatically and a script
#: that drops it fails here rather than slipping through unnoticed.
EXPECTED_DATA_DIR_SCRIPTS = {
    "ingest_alpaca.py",
    "ingest_binance_spot.py",
    "ingest_tiingo.py",
    "ingest_us_equity.py",
    "refresh_us_equity_universe.py",
}


def _scripts_offering_the_flag() -> set[str]:
    """Repo-root scripts that REGISTER `--data-dir`, found by AST.

    Registration is a CALL to `add_data_dir_arg`, never merely the presence of
    the name. A substring scan was tried first and escaped a mutation: deleting
    `add_data_dir_arg(parser)` from `ingest_alpaca.py` left the import in the
    `from utils.cli import (...)` block, the grep still matched, and the whole
    flag went dead on that script while the suite stayed green.
    """
    found: set[str] = set()
    for path in sorted(REPO_ROOT.glob("*.py")):
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:  # pragma: no cover - not a script we guard
            continue
        if any(
            isinstance(node, ast.Call) and _called_name(node) == "add_data_dir_arg"
            for node in ast.walk(tree)
        ):
            found.add(path.name)
    return found


def _main_block(tree: ast.Module) -> ast.If:
    """Return the module-level `if __name__ == "__main__":` node."""
    for node in tree.body:
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if (
            isinstance(test, ast.Compare)
            and isinstance(test.left, ast.Name)
            and test.left.id == "__name__"
            and any(
                isinstance(c, ast.Constant) and c.value == "__main__"
                for c in test.comparators
            )
        ):
            return node
    raise AssertionError("no `if __name__ == '__main__':` block found")


def _called_name(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _calls_in_source_order(node: ast.AST) -> list[tuple[int, int, str]]:
    calls = [
        (child.lineno, child.col_offset, _called_name(child))
        for child in ast.walk(node)
        if isinstance(child, ast.Call) and _called_name(child) is not None
    ]
    return sorted(calls)  # type: ignore[arg-type]


def _path_consuming_names(tree: ast.Module) -> set[str]:
    """Every name in this module whose invocation can reach a
    `quantlab/config/` factory: the names the module imports from the
    configuration package, plus every module-level function that calls one of
    them, transitively.

    Derived rather than listed. The plan proposed a `_build*` name prefix as
    the proxy for "reaches a factory", but every one of these scripts opens
    `__main__` with `_build_arg_parser()`, which reaches nothing and must run
    BEFORE `parse_args()` -- a prefix rule would make the ordering
    unsatisfiable. Reachability is what the rule was reaching for, and it
    catches `_build_configs`/`_build_dataset_config` by derivation rather than
    by spelling.
    """
    denied = {
        alias.asname or alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        and (node.module or "") == CONFIG_MODULE
        for alias in node.names
    }
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }
    grew = True
    while grew:
        grew = False
        for name, func in functions.items():
            if name in denied:
                continue
            if any(called in denied for _, _, called in _calls_in_source_order(func)):
                denied.add(name)
                grew = True
    return denied


def test_every_script_offering_the_flag_is_guarded_here() -> None:
    """The guarded list is globbed from the repo, not hardcoded into the guard.

    A sixth acquisition entry point that offers `--data-dir` must be covered by
    the ordering guard below rather than exempted from it by omission -- and a
    script that quietly drops the flag must fail, not silently shrink the
    guarantee. Add the new filename to `EXPECTED_DATA_DIR_SCRIPTS`.
    """
    assert _scripts_offering_the_flag() == EXPECTED_DATA_DIR_SCRIPTS


@pytest.mark.parametrize("script_name", sorted(EXPECTED_DATA_DIR_SCRIPTS))
def test_apply_data_dir_precedes_every_factory_reaching_call(
    script_name: str,
) -> None:
    """DDIR-04, enforced structurally in every `__main__`.

    The `config/` factories snapshot their paths as plain strings at
    construction time, so an override applied AFTER one of them constructs
    silently does nothing -- the run reports one root and writes to another,
    and the printed path is the lie. This is the T-rjq-02 mitigation.
    """
    assert script_name in _scripts_offering_the_flag(), (
        f"{script_name} no longer offers --data-dir; if that is deliberate, "
        "remove it from EXPECTED_DATA_DIR_SCRIPTS in the same commit."
    )

    # And the flag reaches the parser the script's own `__main__` uses. The
    # structural check above proves the call is written; this proves it lands,
    # which is what a user typing `--data-dir` actually depends on. Without it,
    # a registration moved into a branch that never runs would still pass.
    module = importlib.import_module(script_name.removesuffix(".py"))
    assert hasattr(module._build_arg_parser().parse_args([]), "data_dir"), (
        f"{script_name}'s _build_arg_parser() does not register --data-dir; "
        "apply_data_dir(args) below it will silently no-op."
    )

    tree = ast.parse((REPO_ROOT / script_name).read_text())
    denied = _path_consuming_names(tree)
    calls = _calls_in_source_order(_main_block(tree))
    names = [called for _, _, called in calls]

    assert names.count("apply_data_dir") == 1, (
        f"{script_name}'s __main__ must call apply_data_dir exactly once; "
        f"found {names.count('apply_data_dir')}."
    )
    apply_index = names.index("apply_data_dir")

    reaching = [i for i, called in enumerate(names) if called in denied]
    assert reaching, (
        f"{script_name}'s __main__ calls nothing that reaches a config/ "
        f"factory -- the deny set {sorted(denied)} matched no call, so this "
        "guard is asserting nothing. Check the derivation, not the script."
    )
    first = min(reaching)
    assert apply_index < first, (
        f"{script_name}: apply_data_dir(args) runs AFTER {names[first]}(), "
        "which reaches a config/ factory. The factories snapshot their paths "
        "at construction time, so an override applied later silently does "
        "nothing and the run writes somewhere other than it reports "
        "(DDIR-04). Move the call up, directly after parse_args()."
    )


# ---------------------------------------------------------------------------
# D-05: --data-dir and ingest_binance_spot.py's --raw-data-dir COMPOSE
# ---------------------------------------------------------------------------


def test_binance_data_dir_alone_roots_every_path(monkeypatch, tmp_path) -> None:
    import ingest_binance_spot as script

    monkeypatch.delenv("QUANTLAB_DATA_DIR", raising=False)
    args = script._build_arg_parser().parse_args(["--data-dir", str(tmp_path)])
    apply_data_dir(args)

    cfg = script._build_dataset_config(args)

    assert cfg.raw_data_dir_path.startswith(str(tmp_path))
    assert cfg.zarr_file_path.startswith(str(tmp_path))
    assert cfg.catalog_path.startswith(str(tmp_path))


def test_binance_raw_data_dir_composes_with_data_dir(monkeypatch, tmp_path) -> None:
    """D-05: neither flag supersedes the other -- they operate at different
    levels. `--data-dir` relocates the whole root; `--raw-data-dir` points at
    one pre-existing directory that does not follow the project's convention
    at all. Reading CSVs from there and writing the Zarr under the relocated
    root is the useful combination.
    """
    import ingest_binance_spot as script

    monkeypatch.delenv("QUANTLAB_DATA_DIR", raising=False)
    raw = tmp_path / "elsewhere" / "klines"
    args = script._build_arg_parser().parse_args(
        ["--data-dir", str(tmp_path / "root"), "--raw-data-dir", str(raw)]
    )
    apply_data_dir(args)

    cfg = script._build_dataset_config(args)

    assert cfg.raw_data_dir_path == str(raw)
    assert not cfg.raw_data_dir_path.startswith(str(tmp_path / "root"))
    assert cfg.zarr_file_path.startswith(str(tmp_path / "root"))
