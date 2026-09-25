"""The `--data-dir` per-run storage root: precedence and the two shared CLI
helpers.

The knob is one knob with three levels -- CLI override > `QUANTLAB_DATA_DIR` >
repo-root `data/` (260907-rjq D-01). These tests pin every cell of that
precedence and the reset contract that keeps a process-global override from
leaking.

Every test that cares about the env layer sets or deletes `QUANTLAB_DATA_DIR`
explicitly: a developer machine may have it exported, and an assertion that
silently depends on that is an assertion that passes for the wrong reason.
"""

import argparse
import ast
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
    package at module scope would drag `quantlab.backend`,
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
