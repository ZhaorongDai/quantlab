"""The library's storage root (`set_data_root` precedence) and the shared CLI
helpers the WRDS scripts use for their output directories.

The root is one knob with three levels -- `set_data_root` > `QUANTLAB_DATA_DIR` >
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
from quantlab.utils.cli import (
    add_output_dir_args,
    place_downloads,
    resolve_output_dirs,
)

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


def test_env_var_wins_when_the_override_is_cleared(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("QUANTLAB_DATA_DIR", str(tmp_path / "from_env"))
    config.set_data_root(None)

    assert config.get_data_root() == Path(str(tmp_path / "from_env"))


def test_repo_default_when_neither_override_nor_env_is_set(monkeypatch) -> None:
    monkeypatch.delenv("QUANTLAB_DATA_DIR", raising=False)
    config.set_data_root(None)

    assert config.get_data_root() == REPO_ROOT / "data"


def test_set_data_root_accepts_a_string_and_a_path(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.delenv("QUANTLAB_DATA_DIR", raising=False)

    assert config.set_data_root(str(tmp_path)) == tmp_path
    assert config.get_data_root() == tmp_path
    assert config.set_data_root(tmp_path / "sub") == tmp_path / "sub"
    assert config.get_data_root() == tmp_path / "sub"


def test_set_data_root_expands_a_tilde(monkeypatch) -> None:
    """D-07: a quoted `'~/quantlab-data'` from a shell reaches Python with a
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


def test_set_data_root_none_clears_and_returns_none(
    monkeypatch, tmp_path
) -> None:
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
    default, but a caller who passed `""` meant something, and silently
    meaning "repo default" is wrong.
    """
    with pytest.raises(ValueError):
        config.set_data_root(value)


# ---------------------------------------------------------------------------
# utils.cli.add_output_dir_args() / resolve_output_dirs() / place_downloads()
# ---------------------------------------------------------------------------


def test_add_output_dir_args_defaults_both_directories_to_the_current_one() -> (
    None
):
    parser = argparse.ArgumentParser()

    assert add_output_dir_args(parser) is parser
    args = parser.parse_args([])
    assert (args.download_dir, args.zarr_dir) == (".", ".")
    assert resolve_output_dirs(args) == (Path.cwd(), Path.cwd())


def test_resolve_output_dirs_anchors_relative_paths_and_expands_tilde(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    args = add_output_dir_args(argparse.ArgumentParser()).parse_args(
        ["--download-dir", "raw", "--zarr-dir", "~/zarr"]
    )

    download_dir, zarr_dir = resolve_output_dirs(args)

    assert download_dir == tmp_path / "raw"
    assert zarr_dir == tmp_path / "home" / "zarr"
    assert download_dir.is_absolute() and zarr_dir.is_absolute()


def test_place_downloads_keeps_the_vendor_directory_and_moves_both_tiers(
    tmp_path,
) -> None:
    """The factory's `.../wrds_crsp/wrds` becomes `<download_dir>/wrds`, and
    the watermarks sit beside it under `_watermarks/wrds`, so everything an
    acquisition derives from the raw directory's parent (`_reference/`,
    `_vintage/`) lands in the chosen directory too."""
    from quantlab.base.config import AcquisitionConfig

    original = AcquisitionConfig(
        raw_data_dir_path="/root/downloads/us_equity/1d/wrds_crsp/wrds",
        watermark_path="/root/downloads/us_equity/1d/wrds_crsp/_watermarks/wrds",
        symbols=("14593",),
        start_date="2020-01-01",
        end_date="2020-12-31",
        kwargs={"data_type": "crsp_daily"},
        market="us_equity",
        frequency="1d",
        vendor="wrds",
    )

    moved = place_downloads(original, tmp_path / "raw")

    assert moved.raw_data_dir_path == str(tmp_path / "raw" / "wrds")
    assert moved.watermark_path == str(
        tmp_path / "raw" / "_watermarks" / "wrds"
    )
    assert (
        Path(moved.raw_data_dir_path).parent
        == Path(moved.watermark_path).parent.parent
    )
    assert (moved.symbols, moved.start_date, moved.end_date, moved.kwargs) == (
        original.symbols,
        original.start_date,
        original.end_date,
        original.kwargs,
    )
    assert (
        original.raw_data_dir_path
        == "/root/downloads/us_equity/1d/wrds_crsp/wrds"
    )


def test_utils_cli_does_not_import_config_at_module_scope() -> None:
    """D-04: `quantlab/utils/cli.py` imports nothing from the project at
    module scope. Importing the configuration package there would drag
    `quantlab.backend`, `quantlab.dataset.spot`, `quantlab.dataset.stock` and
    `quantlab.base.config` into every import of the dependency-light CLI
    helper module.
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
        "scope; D-04 keeps the dependency-light CLI module free of the "
        "dataset layer."
    )
