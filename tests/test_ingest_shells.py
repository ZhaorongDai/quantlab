"""Home for the thin-ingest-shell proofs: ROADMAP success criteria SC-6 and
SC-1, requirement D-15.

SC-6 — every in-repo ingest entry point reaches its data source THROUGH the
registry.
SC-1 — no vendor class is named at the call site.
D-15 — the three us-equity ingest scripts are reduced to thin shells and KEPT,
not deleted: `ingest_us_equity.py` is not redundant (its `mode="in_range"`
roster, its independent `us_all` watermarks, its CHUNKED conversion and its
`--stamp-legacy-watermarks` all differ from `ingest_tiingo.py`). The
`--to-zarr` FLAG is no longer among the differences -- all three shells carry
it as of G-03.4-1b -- but the chunked, resumable conversion behind it still
is.

Scaffolded by plan 03.4-01 (Wave 0); filled in by 03.4-02 (`ingest_tiingo.py`)
and 03.4-06 (`ingest_alpaca.py`, `ingest_us_equity.py`). All three shells are
thin as of 03.4-06 and none of them names a vendor class.

THE RULE THIS FILE IS SUBJECT TO, from an incident recorded in
`.planning/STATE.md`: EVERY test here must be a real assertion. A pytest file
with zero collected tests exits **5**, which a per-file command reads as green,
so a placeholder, a body that is only a no-op statement, or a skip/xfail marker
is indistinguishable from a passing file. (Wave 0's second rule -- "no test is
named for a selector a later plan owns" -- expired when 03.4-06 landed the
behaviour those selectors describe; `no_shell_names_a_vendor_class` and
`us_equity_keeps` now match real tests deliberately.)

WHAT THE PURITY SCAN CANNOT CATCH, stated so nobody mistakes it for total: a
vendor class reached through a DOTTED STRING resolved at run time --
`get_cls_from_path("quantlab.acquisition.tiingo.TiingoAcquisition")` -- appears
in the AST as a string constant and a generic call, and no import or identifier
scan can see it. That idiom exists in this repo (`Acquisition.import_path` feeds
it) and D-03 rejected it for the descriptor's `acquisition_cls` precisely
because a direct class reference is the checkable form. If a shell ever grows a
dotted-path resolution, this file will not notice; review is the control there,
not this scan.
"""

import ast
import importlib.util
import io
import json
import subprocess
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from quantlab.acquisition.registry import (
    Capability,
    DataSourceRegistry,
    SourceDescriptor,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

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


# ---------------------------------------------------------------------------
# D-15 / SC-1: no shell names a vendor CLASS
# ---------------------------------------------------------------------------


def _identifiers(tree: ast.Module) -> set[str]:
    """Every NAME this module binds or reads, in all three shapes a vendor
    class could hide in.

    `ast.Name` catches a bare reference and a construction; `ast.Attribute`
    catches `module.TiingoAcquisition`; the import aliases catch
    `from ... import AlpacaAcquisition as _A`, where neither of the first two
    would see the vendor name at all.

    An AST walk and NOT a substring scan, for the reason the resolver self-test
    above already demonstrates in the other direction: a class named inside a
    docstring or a comment is prose, not a call site, and a purity test that
    goes red on prose is a test people delete rather than satisfy. `-` this
    file's own module docstring names `TiingoAcquisition` twice.
    """
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    names |= {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    names |= {
        alias.asname or alias.name.rsplit(".", 1)[-1]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    return names


@pytest.mark.parametrize("shell", SHELL_FILES)
def test_no_shell_names_a_vendor_class(shell: str) -> None:
    """SC-1, in the CLASS sense, proved two independent ways per shell.

    Arm 1 -- identifiers: no `ast.Name`, `ast.Attribute` attribute or imported
    alias ends in `Acquisition`. That is the direct reading of "no vendor named
    at the call site": `TiingoAcquisition(cfg)`, `alpaca.AlpacaAcquisition`,
    and `from ... import AlpacaAcquisition as _A` are all caught, and a
    docstring mention is not.

    Arm 2 -- imports: the shell's resolved import set is DISJOINT from
    `{quantlab.base.acquisition, quantlab.acquisition.tiingo,
    quantlab.acquisition.alpaca}`. This is the arm that survives a rename: a
    vendor class called `TiingoFetcher` would sail past arm 1 and be caught
    here, because the coupling that actually matters is which module the shell
    reaches into, not what the class happens to be called.

    Neither arm objects to the vendor TOKEN. `DataSourceRegistry.get("tiingo")`
    is the irreducible minimum in a script whose whole identity is its vendor,
    and the stricter reading would need a `--source` flag -- i.e. exactly the
    merged CLI D-15 forbids. That ruling is recorded at each shell's own
    `SOURCE` constant so the next reader does not "finish the job".
    """
    path = REPO_ROOT / shell
    tree = ast.parse(path.read_text(encoding="utf-8"))

    named = sorted(
        name for name in _identifiers(tree) if name.endswith("Acquisition")
    )
    assert not named, (
        f"{shell} names vendor acquisition class(es) {named}. Read the fact "
        f"off SOURCE.acquisition_cls instead -- a class named here is a vendor "
        f"binding that survives the registry landing untouched (SC-1 / L-5)."
    )

    reached = _resolved_imports(path, shell.removesuffix(".py"))
    forbidden = sorted(reached & _FORBIDDEN_ACQUISITION_MODULES)
    assert not forbidden, (
        f"{_SHELL_RESOLVER_TOKEN}: {shell} imports {forbidden}. A shell reaches "
        f"its vendor ONLY through the descriptor the registry returns (SC-6)."
    )


# ---------------------------------------------------------------------------
# D-15 / SC-6: each shell RESOLVES its source through the registry
#
# The tests below load each shell into a throwaway module object with
# `DataSourceRegistry.get` monkeypatched to a recorder returning a stub
# descriptor. That is what turns "no vendor name is present" (an absence, which
# a shell could satisfy by hardcoding something else) into "every vendor fact
# came from the descriptor the registry handed back" (a presence).
# ---------------------------------------------------------------------------

#: Deliberately absurd values, so an assertion that sees one knows it came from
#: HERE and not from a real vendor class that happens to agree.
class _StubAcquisition:
    """Stands in for `SOURCE.acquisition_cls` while a shell is being loaded.

    Carries every constant the three shells read off the descriptor's
    acquisition class, and NOTHING else -- so a shell that reaches for a
    vendor-class attribute this stub does not declare fails loudly here rather
    than silently continuing to depend on the real class.
    """

    DEFAULT_BATCH_SIZE = 7717
    DEFAULT_MAX_WORKERS = 7718
    DEFAULT_QUOTA_WAIT_SECONDS = 7719
    DEFAULT_QUOTA_MAX_WAITS = 7720
    TICK_DATA_TYPES = ("stub_quotes", "stub_trades")
    RAW_COLUMNS_BY_DATA_TYPE = {"bars": ("t", "s", "v", "w", "x")}
    LEGACY_WATERMARK_POLICIES = ("stub_warn", "stub_refetch")
    DEFAULT_LEGACY_WATERMARK_POLICY = "stub_warn"
    FAILURE_MANIFEST_NAME = "_stub_failures.json"

    def __init__(self, config):  # pragma: no cover - defensive
        raise AssertionError(
            "a shell constructed the acquisition class while merely building "
            "configs or a parser; construction is where the credential is "
            "demanded and it belongs in registry.run() or the stamping branch"
        )


#: What each shell must ask the registry for. `ingest_us_equity.py` and
#: `ingest_tiingo.py` share a vendor and are still two entry points: they
#: resolve rosters through different modes and write under different `subdir`s,
#: so neither one's watermarks satisfy the other's coverage (L-1).
SHELL_VENDOR_TOKENS = {
    "ingest_tiingo.py": "tiingo",
    "ingest_alpaca.py": "alpaca",
    "ingest_us_equity.py": "tiingo",
}


class _RecordingFactory:
    """A stand-in for `descriptor.config_factory` that records and returns a
    sentinel, so "the config came from the descriptor" is an IDENTITY check.
    """

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.sentinel = object()

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return self.sentinel


def _stub_descriptor(factory: _RecordingFactory) -> SourceDescriptor:
    """A descriptor that is NEVER registered.

    Built directly rather than through `register_source`, so no test here can
    leak a fake vendor into `DataSourceRegistry.SOURCES` -- the failure mode
    `isolated_registry` exists to prevent, avoided by not needing the fixture
    at all.
    """
    return SourceDescriptor(
        vendor="tiingo",  # a real token: `Vendor` is a closed Literal
        display_name="Stub Source (test double)",
        acquisition_cls=_StubAcquisition,  # type: ignore[arg-type]
        config_factory=factory,
        capabilities=(Capability(market="us_equity", frequency="1d"),),
        required_env=(),
    )


def _load_shell_against_a_stub_registry(shell: str, monkeypatch):
    """Execute `shell`'s module body with `DataSourceRegistry.get` recording.

    Loaded through `importlib.util` into a module object that is NOT inserted
    into `sys.modules`: the real `ingest_*` modules are imported by
    `tests/test_data_dir_cli.py` and `tests/test_ingest_tiingo_universe_wiring.py`,
    and replacing them with a stub-wired copy would hand those tests a shell
    whose `SOURCE` is a fake.

    Executing the module BODY is the point. `SOURCE = DataSourceRegistry.get(...)`
    is a module-level statement, so the only way to observe which token it asks
    for -- and to substitute what comes back -- is to run it again under the
    patch. Importing the already-imported module would observe nothing.

    Returns `(module, requested_tokens, factory)`.
    """
    factory = _RecordingFactory()
    descriptor = _stub_descriptor(factory)
    requested: list[str] = []

    def _fake_get(cls, vendor: str):
        requested.append(vendor)
        return descriptor

    monkeypatch.setattr(DataSourceRegistry, "get", classmethod(_fake_get))

    spec = importlib.util.spec_from_file_location(
        f"_stubbed_{shell.removesuffix('.py')}", REPO_ROOT / shell
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.SOURCE is descriptor, (
        f"{shell}'s SOURCE is not the descriptor the registry returned; it "
        f"resolves its source somewhere other than DataSourceRegistry.get()"
    )
    return module, requested, factory


def _config_factory_call_sites(tree: ast.Module) -> list[int]:
    """Lines where `<anything>.config_factory(...)` is CALLED."""
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "config_factory"
    ]


@pytest.mark.parametrize("shell", SHELL_FILES)
def test_each_shell_resolves_its_source_through_the_registry(
    shell: str, monkeypatch
) -> None:
    """SC-6: the descriptor the registry returns is where the vendor comes from.

    Three arms, and the third is what makes the first two mean something:

    1. The shell asks `DataSourceRegistry.get` exactly once, for its own token.
    2. Its `SOURCE` IS the object the registry handed back (identity, not
       equality -- an equal-looking descriptor built locally would pass an
       equality check while proving the opposite).
    3. It builds its acquisition config through `SOURCE.config_factory` and
       imports NO acquisition-config factory of its own. Arm 3 is the one that
       closes the loophole 03.4-02 left open in `ingest_tiingo.py`: calling
       `stock_acquisition_config` directly produced a byte-identical config
       because `"tiingo"` is that factory's incumbent default, so the vendor was
       never actually routed and `config_factory` was dead weight in the very
       script meant to demonstrate the registry.
    """
    module, requested, factory = _load_shell_against_a_stub_registry(
        shell, monkeypatch
    )

    assert requested == [SHELL_VENDOR_TOKENS[shell]], (
        f"{shell} asked the registry for {requested}, expected "
        f"[{SHELL_VENDOR_TOKENS[shell]!r}] exactly once"
    )

    path = REPO_ROOT / shell
    tree = ast.parse(path.read_text(encoding="utf-8"))
    reached = _resolved_imports(path, shell.removesuffix(".py"))
    assert "quantlab.config.stock_acquisition_config" not in reached, (
        f"{shell} imports stock_acquisition_config directly. Build the "
        f"acquisition config through SOURCE.config_factory instead, so the "
        f"vendor is a fact of the registered descriptor rather than the "
        f"factory default this call site happens to inherit."
    )

    sites = _config_factory_call_sites(tree)
    assert len(sites) == 1, (
        f"{shell} must call SOURCE.config_factory(...) exactly once; found "
        f"call sites at {sites}"
    )

    # And the call actually LANDS on the descriptor's factory, for the two
    # shells whose config building is reachable without running `__main__`.
    # `ingest_us_equity.py` builds its config inside `__main__` (its roster
    # resolution needs the universe table), so for that one the structural
    # single-call-site assertion above plus the SOURCE identity assertion in
    # the loader are the proof.
    if hasattr(module, "_build_configs"):
        acq_config, _ds_config = module._build_configs(
            _shell_args(shell)
        )
        assert acq_config is factory.sentinel, (
            f"{shell}._build_configs did not return the descriptor factory's "
            f"result -- something else built the acquisition config"
        )
        assert len(factory.calls) == 1
        assert "vendor" not in factory.calls[0], (
            f"{shell} passes vendor= to the descriptor's factory. The partial "
            f"already pins it; restating it here reintroduces the call-site "
            f"vendor literal the descriptor exists to remove."
        )


def _shell_args(shell: str):
    """A minimal `argparse.Namespace` for a shell's `_build_configs`."""
    import argparse

    base = dict(
        symbols="AAPL,MSFT",
        universe=None,
        as_of_date=None,
        start_date="2024-01-01",
        end_date="2024-01-31",
    )
    if shell == "ingest_alpaca.py":
        base.update(frequency="1d", data_type=None, batch_size=None)
    return argparse.Namespace(**base)


def test_every_argparse_default_that_named_a_vendor_class_now_reads_the_descriptor(
    monkeypatch,
) -> None:
    """L-5, proved DYNAMICALLY rather than by reading the source.

    The constructor call is the obvious vendor binding; the argparse defaults
    that read vendor class constants are seven more, and they name the vendor at
    PARSER-DEFINITION time, so a registry landing would leave every one of them
    untouched while SC-6's letter was satisfied.

    The proof is that with a stub descriptor in place, the parsers come back
    carrying the STUB's absurd constants. A source scan for
    `SOURCE.acquisition_cls` would pass equally well against a file that wrote
    the expression in a comment; this cannot.
    """
    alpaca, _requested, _factory = _load_shell_against_a_stub_registry(
        "ingest_alpaca.py", monkeypatch
    )
    parser = alpaca._build_arg_parser()
    actions = parser._option_string_actions

    assert tuple(actions["--data-type"].choices) == _StubAcquisition.TICK_DATA_TYPES
    assert str(_StubAcquisition.DEFAULT_BATCH_SIZE) in actions["--batch-size"].help

    # The validation message names the same set, from the same place.
    with pytest.raises(SystemExit):
        alpaca._validate_data_type(
            parser,
            __import__("argparse").Namespace(frequency="tick", data_type=None),
        )

    us_equity, _requested, _factory = _load_shell_against_a_stub_registry(
        "ingest_us_equity.py", monkeypatch
    )
    parsed = us_equity._build_arg_parser().parse_args([])
    assert parsed.max_workers == _StubAcquisition.DEFAULT_MAX_WORKERS
    assert parsed.quota_wait_seconds == _StubAcquisition.DEFAULT_QUOTA_WAIT_SECONDS
    assert parsed.quota_max_waits == _StubAcquisition.DEFAULT_QUOTA_MAX_WAITS
    assert parsed.legacy_watermarks == (
        _StubAcquisition.DEFAULT_LEGACY_WATERMARK_POLICY
    )
    assert tuple(
        us_equity._build_arg_parser()._option_string_actions[
            "--legacy-watermarks"
        ].choices
    ) == _StubAcquisition.LEGACY_WATERMARK_POLICIES


# ---------------------------------------------------------------------------
# D-15 / L-1: `ingest_us_equity.py` is NOT redundant with `ingest_tiingo.py`
# ---------------------------------------------------------------------------

#: The eight flags that are DISTINCT to `ingest_us_equity.py` and must
#: survive thinning. `--limit` and `--max-workers` arrive through
#: `add_concurrency_args`; the rest are declared in the script.
#:
#: `--to-zarr`, `--chunk` and `--on-new-listing` are deliberately NOT here any
#: more. 03.5 D-07 collapsed the repository to ONE conversion path, all three
#: US-equity shells reach it through `registry.convert()`, and all three
#: therefore carry all three flags. A flag every shell has cannot evidence
#: what makes THIS shell distinct, and leaving it on a list named "what makes
#: it distinct" would make the list say something false. The sharing is
#: asserted instead -- positively, in
#: `test_the_three_conversion_flags_are_shared_by_every_us_equity_shell`
#: below -- rather than merely un-pinned, because "these three shells agree"
#: is itself a property SC-6 asked for and a later divergence should go red.
_US_EQUITY_FLAGS = (
    "--dry-run",
    "--stamp-legacy-watermarks",
    "--wait-for-quota",
    "--quota-wait-seconds",
    "--quota-max-waits",
    "--max-workers",
    "--legacy-watermarks",
    "--limit",
)

#: The conversion knobs every US-equity shell offers, because every US-equity
#: shell reaches the same `registry.convert()` (03.5 SC-6/D-07).
_SHARED_CONVERSION_FLAGS = ("--to-zarr", "--chunk", "--on-new-listing")

#: The three shells that share them. `ingest_binance_spot.py` is deliberately
#: absent: it is the fourth converting door, and it cannot reach `convert()`
#: because binance is not a registered source (03.5 D-09).
_US_EQUITY_SHELLS = ("ingest_tiingo", "ingest_alpaca", "ingest_us_equity")


def _keyword(call: ast.Call, name: str):
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    return None


def _calls_named(tree: ast.Module, name: str) -> list[ast.Call]:
    """Every `name(...)` / `<obj>.name(...)` CALL node in the module."""
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == name:
            found.append(node)
        elif isinstance(func, ast.Attribute) and func.attr == name:
            found.append(node)
    return found


def test_us_equity_keeps_every_capability_that_makes_it_distinct() -> None:
    """L-1's SIX load-bearing differences, pinned so a later "simplification"
    has to argue with a red test rather than with a reviewer's memory.

    D-15 asked whether `ingest_us_equity.py` is redundant with
    `ingest_tiingo.py --universe us_all`. It is not, and merging them would
    delete, in order: the survivorship-correct `mode="in_range"` roster (the
    ~6.9k symbols that delisted INSIDE the window, which an as-of roster on one
    day cannot see), the independent `us_all` watermark tree and Zarr store
    (neither script's watermarks satisfy the other's coverage), the deliberate
    `symbols=None` dataset config, the credential-free `--dry-run`, the only
    in-repo route to watermark stamping, and the four quota/concurrency knobs.

    **It was seven, and the seventh LEFT because it stopped being a
    difference.** That entry was "the chunked conversion behind `--to-zarr`
    with its per-chunk RAM guard", and the parenthetical beside it said the
    flag was shared but the chunking was not. 03.5 D-07 collapsed the
    repository to ONE conversion path: `ingest_tiingo.py` and
    `ingest_alpaca.py` now reach the same `registry.convert()` through the
    same `assert_chunked_panel_fits`, and carry `--chunk` and
    `--on-new-listing` too. Nothing was deleted from this script -- the
    capability is simply no longer DISTINCT to it, which is what SC-6 asked
    for, and a list named "what makes it distinct" cannot keep an entry every
    shell now satisfies. No seventh survivor was found to replace it; the
    sharing itself is pinned by
    `test_the_three_conversion_flags_are_shared_by_every_us_equity_shell`.

    The flag arm goes through the REAL parser rather than the source, so a
    registration moved into a branch that never runs fails here. The three
    structural arms are AST assertions on call keywords, not substring
    searches: this file's own docstrings say `mode="in_range"` in prose, and a
    text scan would happily count that.
    """
    import ingest_us_equity

    parser = ingest_us_equity._build_arg_parser()
    missing = [
        flag for flag in _US_EQUITY_FLAGS
        if flag not in parser._option_string_actions
    ]
    assert not missing, (
        f"ingest_us_equity.py dropped {missing} while being thinned. Thinning "
        f"removes vendor bindings, never operator capabilities (D-15)."
    )

    tree = ast.parse(
        (REPO_ROOT / "ingest_us_equity.py").read_text(encoding="utf-8")
    )

    roster_modes = [
        _keyword(call, "mode")
        for call in _calls_named(tree, "resolve_symbols")
    ]
    assert [
        node.value for node in roster_modes if isinstance(node, ast.Constant)
    ] == ["in_range"], (
        "ingest_us_equity.py must resolve its roster with mode='in_range'. "
        "'as_of' would resolve membership on ONE day and silently reintroduce "
        "the survivorship bias this roster exists to remove."
    )

    assert ingest_us_equity.DEFAULT_SUBDIR == "us_all"
    assert ingest_us_equity.DEFAULT_STORE_NAME == "us_all.zarr"
    subdirs = [
        _keyword(call, "subdir")
        for call in _calls_named(tree, "config_factory")
        + _calls_named(tree, "stock_kline_config")
    ]
    assert subdirs and all(
        isinstance(node, ast.Name) and node.id == "DEFAULT_SUBDIR"
        for node in subdirs
    ), (
        "both configs must carry subdir=DEFAULT_SUBDIR ('us_all'): the "
        "independent watermark tree is what makes this backfill resumable "
        "separately from ingest_tiingo.py's."
    )

    dataset_symbols = [
        _keyword(call, "symbols")
        for call in _calls_named(tree, "stock_kline_config")
    ]
    assert dataset_symbols and all(
        isinstance(node, ast.Constant) and node.value is None
        for node in dataset_symbols
    ), (
        "the DatasetConfig must be built with symbols=None. The symbol axis "
        "for the conversion is resolved from the RAW DATA, by "
        "_raw_axes_in_range() inside the chunked loop, pinned once over the "
        "whole range; a roster named in the config would be a second, "
        "competing answer to the same question, and the two genuinely differ "
        "-- mode='in_range' resolves membership from the universe table while "
        "the raw tree holds only what actually downloaded. (This used to be "
        "justified by BaseDataset's config setter calling _reset_symbols() "
        "and densifying the full range at construction time. df7bfe9 deleted "
        "that method; no construction densifies any more, for any config.)"
    )


def test_the_three_conversion_flags_are_shared_by_every_us_equity_shell() -> None:
    """SC-6: one conversion path, therefore the same conversion knobs at every
    door.

    The negative half of this property -- `--to-zarr` / `--chunk` /
    `--on-new-listing` leaving `_US_EQUITY_FLAGS` -- is not a property at all
    on its own: three flags can disappear from a list because they became
    shared, or because somebody deleted them. Those two outcomes are
    indistinguishable to a test that only checks the list got shorter, so the
    sharing is asserted HERE, positively, and a shell that later drops one
    goes red.

    `ingest_tiingo.py` and `ingest_alpaca.py` had `--to-zarr` and nothing else
    before 03.5: they converted the WHOLE window in one allocation, so there
    was no granularity to select and no new-listing policy to state. They
    carry both now because they reach the same chunked `registry.convert()`
    that `ingest_us_equity.py` does -- the flags arrived as a CONSEQUENCE of
    sharing one conversion path, not as new surface grown on two shells.

    Goes through each shell's REAL parser, for the reason the sibling test
    gives: a registration moved into a branch that never runs would satisfy a
    source scan and fail a user.
    """
    import importlib

    for shell in _US_EQUITY_SHELLS:
        parser = importlib.import_module(shell)._build_arg_parser()
        missing = [
            flag
            for flag in _SHARED_CONVERSION_FLAGS
            if flag not in parser._option_string_actions
        ]
        assert not missing, (
            f"{shell}.py does not offer {missing}. All three US-equity shells "
            f"reach one conversion through registry.convert() (03.5 D-07), so "
            f"they take the same knobs; a shell that offers fewer either "
            f"stopped delegating or grew a second conversion path."
        )


# ---------------------------------------------------------------------------
# SC-3: the dry run answers on a machine with no credential configured
# ---------------------------------------------------------------------------


def _watermark_sidecars(config, *, covered=(), widened=(), legacy=()) -> None:
    """Write watermark sidecars under `config`'s ledger root.

    Adapted from `tests/test_source_inspector.py:_sidecar_tree` (Wave-1 copy
    convention: `tests/` is not a package, so a cross-test import would couple
    this file's collection to that file's import-time state). Trimmed to the
    three states the coverage REPORT distinguishes; the `no_data` marker is
    orthogonal and is exercised where the ledger itself is tested.
    """
    from quantlab.base.coverage import CoverageLedger

    root = CoverageLedger.for_config(config).watermark_root
    root.mkdir(parents=True, exist_ok=True)
    for symbol in covered:
        (root / f"{symbol}.json").write_text(
            json.dumps(
                {"start_date": config.start_date, "last_date": config.end_date}
            )
        )
    for symbol in widened:
        (root / f"{symbol}.json").write_text(
            json.dumps({"start_date": "2024-01-15", "last_date": config.end_date})
        )
    for symbol in legacy:
        (root / f"{symbol}.json").write_text(
            json.dumps({"last_date": config.end_date})
        )


def test_the_dry_run_needs_no_credential(
    acquisition_config, no_credentials
) -> None:
    """SC-3's visible consequence inside this repository.

    `ingest_us_equity.py` used to print
    ``coverage report:   skipped (export TIINGO_API_KEY to see it)`` -- for a
    computation that is nothing but `open()` and `json.load()` -- because
    `TiingoAcquisition.__init__` demands a credential at CONSTRUCTION, before it
    could know the caller only wanted to count sidecars. 03.4-06 deleted that
    branch and routed the report through `SourceInspector`.

    Asserted on REAL COUNTS, not merely on the absence of the skip line: a
    helper that printed the header and four zeroes would satisfy an
    absence-only test perfectly while answering nothing. The tree below puts
    one symbol in each of the three states the report distinguishes, so every
    printed figure has to be non-trivially right.

    The structural arm is the other half. This module reads NO credential
    environment variable at all any more -- it does not even import `os` -- so
    there is no code path here that could reintroduce the skip. What that arm
    cannot see is a credential read hidden behind an indirection (a name
    imported from elsewhere, a `getattr` on a module object); the behavioural
    arm above is what covers that, since any such read would have to change the
    answer under `no_credentials` to be worth writing.
    """
    import ingest_us_equity

    for name in no_credentials:
        import os as _os

        assert _os.environ.get(name) is None, name

    config = acquisition_config(
        vendor="tiingo",
        symbols=("AAPL", "MSFT", "GOOG", "TSLA"),
        subdir="us_all",
        kwargs={"legacy_watermarks": "warn"},
    )
    _watermark_sidecars(
        config, covered=("AAPL",), widened=("MSFT",), legacy=("GOOG",)
    )

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        ingest_us_equity._print_coverage(config, tuple(config.symbols))
    printed = buffer.getvalue()

    assert "  coverage report:" in printed, printed
    # The DELETED line, matched on its distinctive prefix rather than on the
    # bare word "skipped" -- the `already covered:` line legitimately reads
    # "(would be skipped)", and an over-broad negative would go red on correct
    # output, which is how a gate gets weakened until it catches nothing.
    assert "skipped (export" not in printed, printed
    assert "already covered:   1" in printed, printed
    assert "re-fetch, widened: 1" in printed, printed
    assert "legacy, no start:  1" in printed, printed
    # AAPL covered, GOOG legacy-and-skipped under the 'warn' policy -> MSFT and
    # TSLA remain. A helper printing four zeroes could not produce this line.
    assert "would fetch:       2/4" in printed, printed

    # Structural arm: no credential environment read is even expressible here.
    path = REPO_ROOT / "ingest_us_equity.py"
    reached = _resolved_imports(path, "ingest_us_equity")
    assert "os" not in reached, (
        "ingest_us_equity.py imports `os` again. The only reason it ever did "
        "was the credential-presence check this plan deleted; the vendor "
        "client is the one place that check belongs (T-03.4-06-04)."
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    env_reads = sorted(
        {
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and node.attr in {"environ", "getenv"}
        }
    )
    assert not env_reads, (
        f"ingest_us_equity.py reads the environment via {env_reads}; the dry "
        f"run must not be made to require a credential in order to work."
    )


# ---------------------------------------------------------------------------
# DDIR-04: --data-dir is applied before anything that can reach a factory
# ---------------------------------------------------------------------------


def _called_name(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _main_block(tree: ast.Module) -> ast.If:
    for node in tree.body:
        if (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "__name__"
        ):
            return node
    raise AssertionError("no `if __name__ == '__main__':` block found")


def _factory_reaching_names(tree: ast.Module) -> set[str]:
    """Every name whose invocation can reach a `quantlab/config/` factory.

    DERIVED, not listed -- the reachability idiom quick task 260907-rjq
    established in `tests/test_data_dir_cli.py` and adapted here (the copy
    convention again: `tests/` is not a package). A `_build*` NAME-PREFIX rule
    was tried there and is unsatisfiable, because every one of these scripts
    opens `__main__` with `_build_arg_parser()`, which reaches nothing and must
    run before `parse_args()`.

    One addition this copy makes, and it is the reason the copy is not
    redundant: `SOURCE.config_factory` is now a factory-reaching call that the
    `quantlab.config` import scan CANNOT see, because the descriptor carries
    the factory and the shell no longer imports it. Deriving the deny set from
    imports alone would have quietly stopped covering `ingest_us_equity.py`'s
    acquisition config the moment 03.4-06 landed.
    """
    denied = {
        alias.asname or alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        and (node.module or "").startswith("quantlab.config")
        for alias in node.names
    }
    denied.add("config_factory")

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
            if any(
                _called_name(call) in denied
                for call in ast.walk(func)
                if isinstance(call, ast.Call)
            ):
                denied.add(name)
                grew = True
    return denied


@pytest.mark.parametrize("shell", SHELL_FILES)
def test_apply_data_dir_still_precedes_every_config_factory(shell: str) -> None:
    """DDIR-04 survives the thinning, asserted structurally per shell.

    The `quantlab/config/` factories snapshot their paths as plain STRINGS at
    construction time, so a `--data-dir` override applied after one of them has
    constructed silently does nothing: the run prints one root and writes to
    another, and the printed path is the lie (T-rjq-02).

    A duplicate of `tests/test_data_dir_cli.py`'s guard in shape only -- that
    one is scoped to every script offering the flag and derives its deny set
    from `quantlab.config` imports; this one is scoped to the three shells and
    additionally denies `SOURCE.config_factory`, which those imports can no
    longer see (see `_factory_reaching_names`). The emptiness of the deny set
    is asserted rather than tolerated, so this cannot go vacuous the way an
    ordering assertion over nothing would.
    """
    tree = ast.parse((REPO_ROOT / shell).read_text(encoding="utf-8"))
    denied = _factory_reaching_names(tree)

    calls = sorted(
        (node.lineno, node.col_offset, _called_name(node))
        for node in ast.walk(_main_block(tree))
        if isinstance(node, ast.Call) and _called_name(node) is not None
    )
    names = [called for _lineno, _col, called in calls]

    assert names.count("apply_data_dir") == 1, (
        f"{shell}'s __main__ must call apply_data_dir exactly once; found "
        f"{names.count('apply_data_dir')}"
    )
    reaching = [index for index, called in enumerate(names) if called in denied]
    assert reaching, (
        f"{shell}: nothing in __main__ reaches a config factory -- the deny "
        f"set {sorted(denied)} matched no call, so this guard asserts nothing. "
        f"Check the derivation, not the script."
    )
    assert names.index("apply_data_dir") < min(reaching), (
        f"{shell}: apply_data_dir(args) runs AFTER {names[min(reaching)]}(), "
        f"which reaches a config/ factory. Move it directly after parse_args()."
    )
