"""Structural and refusal gates for `scripts/ingest_wrds_crsp_all.py`.

The shell is loaded by FILE PATH, never by module name: `scripts/` is not on
`sys.path` and must never be, because a `scripts/wrds/` folder would shadow
the `wrds` package the WRDS session imports.

The properties asserted here are the shared shell contracts -- no vendor class
named in a shell, the source resolved through the registry, `apply_data_dir`
before any config factory -- stated over this one shell.

Every check is offline: the shell is behind an `if __name__ == "__main__":`
guard, so importing it runs nothing, and `--help` needs no WRDS credential.
"""

from __future__ import annotations

import ast
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SHELL = REPO_ROOT / "scripts" / "ingest_wrds_crsp_all.py"

#: Fully qualified names of the acquisition base and every concrete vendor
#: client. Reaching any of them from a shell means the shell constructs its
#: source directly instead of going through the registry.
_FORBIDDEN_ACQUISITION_MODULES = frozenset(
    {
        "quantlab.base.acquisition",
        "quantlab.acquisition.tiingo",
        "quantlab.acquisition.alpaca",
    }
)


def _tree() -> ast.Module:
    return ast.parse(SHELL.read_text(encoding="utf-8"), filename=str(SHELL))


def _run_help() -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SHELL), "--help"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )


def _run_args(*argv: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SHELL), *argv],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )


def test_the_shell_exists_and_is_git_tracked():
    """Existence alone is not enough: an untracked file present only on this
    machine satisfies `Path.exists()` while being absent for everyone else."""
    assert SHELL.is_file()

    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(SHELL.relative_to(REPO_ROOT))],
        capture_output=True,
        cwd=REPO_ROOT,
    )
    assert tracked.returncode == 0, f"{SHELL.name} is not tracked by git"


def test_the_help_screen_renders_without_a_credential():
    """The import chain resolves and argparse builds.

    A module-level `from quantlab.x import Y` that no longer resolves is not
    something an `ast` scan can see, and it breaks the shell just as
    completely as a bad argument default. `--help` is the cheapest real
    execution of the whole import graph.
    """
    result = _run_help()

    assert result.returncode == 0, result.stderr
    assert "--security-filter" in result.stdout
    assert "--with-index-membership" in result.stdout


def test_the_shell_names_no_vendor_acquisition_class():
    """The registry is the only way a shell reaches a vendor.

    A shell that imports `quantlab.acquisition.tiingo` has hardcoded its
    vendor, and the registry token above it becomes decoration. Checked on the
    module's OWN source rather than on what it happens to have imported at run
    time, because a transitive import is not the shell naming anything.
    """
    named: list[str] = []
    for node in ast.walk(_tree()):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module in _FORBIDDEN_ACQUISITION_MODULES:
                named.append(node.module)
        elif isinstance(node, ast.Import):
            named.extend(
                alias.name
                for alias in node.names
                if alias.name in _FORBIDDEN_ACQUISITION_MODULES
            )

    assert named == [], f"{SHELL.name} reaches a vendor class directly: {named}"


def test_the_source_is_resolved_through_the_registry():
    """`DataSourceRegistry.get(...)` is the one place the vendor is named."""
    source = [
        node
        for node in ast.walk(_tree())
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "DataSourceRegistry"
    ]

    assert len(source) == 1, "expected exactly one DataSourceRegistry.get call"


def test_apply_data_dir_precedes_every_config_factory_reaching_call():
    """DDIR-04: every path is derived from the data root, so the root must be
    applied first.

    Asserted by LINE ORDER inside the `__main__` block, which is the only
    thing that makes the ordering real: a later `apply_data_dir` leaves every
    already-constructed path pointing at the previous root, and nothing warns.
    """
    tree = _tree()
    main_block = [
        node
        for node in tree.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
    ]
    assert len(main_block) == 1, "the shell must have exactly one __main__ guard"

    apply_lines = [
        node.lineno
        for node in ast.walk(main_block[0])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "apply_data_dir"
    ]
    factory_lines = [
        node.lineno
        for node in ast.walk(main_block[0])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"config_factory_for", "reference_dir_for"}
    ]

    assert apply_lines, "apply_data_dir is never called"
    assert factory_lines, "no config factory call found -- did the flow change?"
    assert min(apply_lines) < min(factory_lines)


def test_the_conversion_is_opt_in_and_defaults_to_not_converting():
    """`--to-zarr` off by default, and the default path SAYS so.

    A conversion that silently did not happen is the same class of silence the
    flag exists to end.
    """
    result = _run_help()

    assert "--to-zarr" in result.stdout
    source = SHELL.read_text(encoding="utf-8")
    assert "Skipping Zarr conversion (default)" in source


def test_the_volume_guard_precedes_the_pull():
    """The guard is worthless if it runs after the first COPY.

    Asserted by line order, for the same reason as the data-dir ordering
    above: both calls exist either way, and only their order is the contract.
    """
    tree = _tree()
    guard_lines = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "assert_acquisition_volume_fits"
    ]
    run_lines = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "run"
    ]

    assert guard_lines, "the volume guard is never called"
    assert run_lines, "the acquisition is never run"
    assert max(guard_lines) < min(run_lines)


def test_the_force_volume_flag_exists_and_is_explicit():
    """There is no env var and no config key that disables the guard
    wholesale; the escape hatch is per-run and visible in the command."""
    assert "--force-volume" in _run_help().stdout


# -- the refusals, all before any WRDS session exists ------------------------


def test_a_missing_window_is_refused():
    """On a whole-market roster an unbounded window is a query over the whole
    110-million-row daily table, so the window is not optional."""
    result = _run_args("--start-date", "2024-01-01")

    assert result.returncode != 0
    assert "--start-date and --end-date are both required" in result.stderr


def test_allow_unlinked_ndx_without_the_index_flag_is_refused():
    """Accepting it silently would let a run believe it had relaxed a gate
    that never ran -- no Nasdaq-100 panel is built without the index flag."""
    result = _run_args(
        "--start-date", "2024-01-01",
        "--end-date", "2024-12-31",
        "--allow-unlinked-ndx",
    )

    assert result.returncode != 0
    assert "--allow-unlinked-ndx only means anything" in result.stderr


def test_a_non_positive_batch_size_is_refused():
    result = _run_args(
        "--start-date", "2024-01-01",
        "--end-date", "2024-12-31",
        "--batch-size", "0",
    )

    assert result.returncode != 0
    assert "--batch-size must be a positive integer" in result.stderr


def test_rows_per_symbol_day_is_refused_for_crsp():
    """CRSP counts rows server-side, so the tick-sizing knob is meaningless
    here and is refused rather than ignored."""
    result = _run_args(
        "--start-date", "2024-01-01",
        "--end-date", "2024-12-31",
        "--rows-per-symbol-day", "1000",
    )

    assert result.returncode != 0
    assert "does not apply to CRSP" in result.stderr


def test_an_unknown_security_filter_is_refused_by_argparse():
    """The choices are derived from `SECURITY_FILTER_PRESETS`, so a preset
    added there becomes selectable here with no edit -- and a typo cannot
    silently fall back to the default universe."""
    result = _run_args(
        "--start-date", "2024-01-01",
        "--end-date", "2024-12-31",
        "--security-filter", "equity_commmon",
    )

    assert result.returncode != 0
    assert "invalid choice" in result.stderr


# -- entitlement scoping ----------------------------------------------------


@pytest.mark.parametrize(
    "with_index, expected",
    [
        (False, 1),
        (True, 4),
    ],
    ids=["market-only", "with-index-membership"],
)
def test_the_entitlement_check_is_scoped_to_what_the_run_needs(
    with_index: bool, expected: int
):
    """A whole-market pull must not ask whether this account can read
    Compustat.

    Most CRSP subscriptions cannot, and a pull that never touches those
    schemas would fail on a question it never needed to ask. Imported from
    the shell's own module so the test reads the real function rather than a
    restatement of it.
    """
    spec = importlib.util.spec_from_file_location("ingest_wrds_crsp_all", SHELL)
    assert spec is not None and spec.loader is not None
    shell = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shell)

    schemas = shell._schemas_for(with_index)

    assert len(schemas) == expected
    assert "crsp_a_stock" in schemas
    if not with_index:
        assert not any("comp" in s for s in schemas)
