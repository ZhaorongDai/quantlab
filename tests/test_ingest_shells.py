"""Home for the thin-ingest-shell proofs: ROADMAP success criteria SC-6 and
SC-1, requirement D-15.

SC-6 — every in-repo ingest entry point reaches its data source THROUGH the
registry.
SC-1 — no vendor class is named at the call site.
D-15 — the three us-equity ingest scripts are reduced to thin shells and KEPT,
not deleted: `ingest_us_equity.py` is not redundant (its `mode="in_range"`
roster, its independent `us_all` watermarks, its chunked `--to-zarr` and its
`--stamp-legacy-watermarks` all differ from `ingest_tiingo.py`).

Scaffolded by plan 03.4-01 (Wave 0). The shells still name their vendor
classes today; plans 03.4-02 and 03.4-06 thin them and fill this file in.

TWO RULES THIS FILE IS SUBJECT TO, both from incidents recorded in
`.planning/STATE.md`:

1. EVERY test here must be a real assertion. A pytest file with zero tests
   exits **5** ("no tests ran"), which a per-file command reads as green, so a
   placeholder, a body that is only a no-op statement, or a skip/xfail marker
   is indistinguishable from a passing file.

2. No test here is named for a selector `03.4-VALIDATION.md` assigns to a
   later plan (`no_shell_names_a_vendor_class`, `us_equity_keeps`). Those must
   match ZERO tests until the behaviour exists.
"""

import ast
import subprocess
from pathlib import Path

#: Opens this file's structural assertion messages and appears in exactly one
#: file in the repository, so a failure here is attributable to THIS arm rather
#: than to a neighbouring purity test that would have failed anyway. Mirrors
#: `tests/test_volume_guard.py:_RESOLVER_TOKEN`, which owns the same idiom for
#: the universe module.
_SHELL_RESOLVER_TOKEN = "SHELL-FORBIDDEN-IMPORT-RESOLVED"

#: The three us-equity ingest entry points D-15 keeps. `ingest_binance_spot.py`
#: is deliberately absent: it serves the crypto-spot market, reaches no
#: us-equity vendor, and is outside this phase's scope.
SHELL_FILES = (
    "ingest_tiingo.py",
    "ingest_alpaca.py",
    "ingest_us_equity.py",
)

#: Fully qualified names of the acquisition base and every concrete vendor
#: client. Reaching any of them from a shell means the shell constructs its
#: source directly instead of going through the registry (SC-1/SC-6).
_FORBIDDEN_ACQUISITION_MODULES = frozenset(
    {
        "quantlab.base.acquisition",
        "quantlab.acquisition.tiingo",
        "quantlab.acquisition.alpaca",
    }
)


# ---------------------------------------------------------------------------
# Copied VERBATIM from `tests/test_volume_guard.py:717-745`.
#
# A copy rather than a cross-test import: `tests/` is not a package (no
# `tests/__init__.py`; `pyproject.toml` sets only `pythonpath = ["."]`), so
# importing one test module from another would couple this file's collection to
# the other file's import-time state. The copy is kept honest by
# `test_the_import_resolver_sees_absolute_and_relative_spellings` below.
# ---------------------------------------------------------------------------
def _resolved_imports(source_path, module_name: str) -> set[str]:
    """Every module `source_path` imports, as a fully qualified dotted name.

    Relative imports are resolved against `module_name`'s own package, which is
    the whole point: `from ..base.acquisition import X` and
    `from ..base import acquisition` name the same module as
    `import quantlab.base.acquisition` and must be seen as such. `from X import
    y` also contributes `X.y`, because `y` may itself be a submodule.
    """
    import ast

    package = module_name.rpartition(".")[0]
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package
                for _ in range(node.level - 1):
                    base = base.rpartition(".")[0]
            else:
                base = ""
            target = f"{base}.{node.module}" if base and node.module else (
                node.module or base
            )
            found.add(target)
            found.update(f"{target}.{alias.name}" for alias in node.names)
    return found


def test_the_import_resolver_sees_absolute_and_relative_spellings(
    tmp_path: Path,
) -> None:
    """Self-test for the resolver plan 06's purity assertion depends on.

    Three spellings name the SAME module and all three must be seen, or a shell
    could keep its vendor coupling in whichever spelling the check happens to
    miss and the purity test would pass while proving nothing:

    - `import quantlab.base.acquisition` (absolute)
    - `from quantlab.acquisition.tiingo import TiingoAcquisition`
      (`from X import y`, which also contributes `X.y` since `y` may itself be
      a submodule)
    - `from ..base.acquisition import Acquisition` (relative, resolved against
      the importing module's own package)

    The negative control is the reason this is an AST resolver rather than a
    substring scan at all: a module named only inside a DOCSTRING contributes
    nothing. A `grep` for "quantlab.acquisition.alpaca" would flag the shell
    below as impure on the strength of prose, and a purity test that cries wolf
    on a comment gets weakened until it stops catching anything.
    """
    source = tmp_path / "shell_sample.py"
    source.write_text(
        '"""A docstring naming quantlab.acquisition.alpaca in prose only.\n'
        "\n"
        "    A substring scan would resolve this mention; an AST walk must not.\n"
        '"""\n'
        "import quantlab.base.acquisition\n"
        "from quantlab.acquisition.tiingo import TiingoAcquisition\n"
        "from ..base.acquisition import Acquisition\n"
        'MENTION = "quantlab.acquisition.alpaca"\n',
        encoding="utf-8",
    )

    resolved = _resolved_imports(source, "quantlab.acquisition.shell_sample")

    assert "quantlab.base.acquisition" in resolved
    assert "quantlab.acquisition.tiingo" in resolved
    assert "quantlab.acquisition.tiingo.TiingoAcquisition" in resolved
    assert "quantlab.base.acquisition.Acquisition" in resolved

    # Negative control: prose and string literals are not imports.
    assert "quantlab.acquisition.alpaca" not in resolved, (
        f"{_SHELL_RESOLVER_TOKEN}: the resolver treated a docstring/string "
        f"mention as an import, which is the false positive an AST walk exists "
        f"to avoid"
    )

    # The forbidden set is reachable from this sample, proving the intersection
    # the real purity test performs is expressible against this resolver's
    # output shape.
    assert resolved & _FORBIDDEN_ACQUISITION_MODULES == {
        "quantlab.base.acquisition",
        "quantlab.acquisition.tiingo",
    }


def test_every_shell_file_exists_and_is_git_tracked() -> None:
    """D-15 keeps all three shells; thinning is not deleting.

    `ingest_us_equity.py` in particular is NOT redundant with `ingest_tiingo.py`
    (L-1): its `mode="in_range"` roster includes the ~6.9k symbols that
    delisted inside the window, removing survivorship bias, and it writes to an
    independent `us_all` watermark tree. A later plan that "simplified" the
    shells by deleting one would lose that, so existence is asserted here in
    Wave 0, before any thinning happens.

    Git-tracking is asserted as well as existence: an untracked file present
    only on this machine would satisfy a bare `Path.exists()` while being
    absent for everyone else.
    """
    repo_root = Path(__file__).resolve().parent.parent

    for name in SHELL_FILES:
        path = repo_root / name
        assert path.is_file(), f"{name} must exist -- D-15 keeps all three shells"

        tracked = subprocess.run(
            ["git", "ls-files", "--", name],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()
        assert tracked == [name], (
            f"{name} must be tracked by git; `git ls-files` returned {tracked!r}"
        )

    # The shells must remain parseable at all times -- a later plan thins them,
    # and a syntax error there would otherwise only surface at run time.
    for name in SHELL_FILES:
        ast.parse((repo_root / name).read_text(encoding="utf-8"))
