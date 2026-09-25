"""G-03.4-1: nothing fetched must not end in a conversion traceback, and the
Zarr conversion is an explicit `--to-zarr` opt-in in all three ingest shells.

Two halves of one UAT gap.

(a) `ingest_alpaca.py` / `ingest_tiingo.py` walked into
`StockDataset.from_raw_data()` after a run in which every symbol failed, and
ended on `quantlab/dataset/stock.py`'s uncaught absent-root `ValueError` -- a
traceback that reads like a bug in the conversion layer when the real event was
"the vendor returned nothing". `quantlab.utils.cli
.refuse_conversion_without_raw_data` now translates that into a clean non-zero
exit BEFORE the densification.

(b) The three shells disagreed about what a run without arguments does:
`ingest_us_equity.py` needed `--to-zarr`, the other two converted
unconditionally. They now agree, and the default path says so out loud.

THE RULE THIS FILE IS SUBJECT TO, inherited from `tests/test_ingest_shells.py`:
every test here is a real assertion. A pytest file with zero collected tests
exits 5, which a per-file command reads as green.

TWO KINDS OF ASSERTION, deliberately, and the split is the L-5 lesson from
`tests/test_ingest_shells.py`: flag REGISTRATION is checked through the REAL
parser, never by scanning the source, because a registration moved into a
branch that never executes is invisible to a text scan and fully visible to
`parse_args`. Only the ORDERING claim -- the guard precedes the densification
-- is structural, because there is no way to observe it without running a real
fetch.
"""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

from quantlab.base.config import DatasetConfig
from quantlab.dataset.stock import StockDataset
from quantlab.utils.cli import refuse_conversion_without_raw_data

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Where the ingest shells live. They moved out of the repository ROOT after
#: these gates were written, which left every `REPO_ROOT / shell` below
#: pointing at a path that does not exist.
SCRIPTS_DIR = REPO_ROOT / "scripts"

#: The three front doors. Kept as one list so a fourth shell joins every
#: assertion below by being added HERE, rather than by somebody remembering to
#: extend each test -- the reachability lesson this module's own
#: `test_every_entry_point_that_densifies_refuses_first` records.
INGEST_SHELLS = ("ingest_tiingo.py", "ingest_alpaca.py", "ingest_us_equity.py")


class _FakeResult:
    """The two `AcquisitionResult` fields the guard reads, and nothing else.

    A stub rather than the real frozen dataclass because the guard's contract
    is exactly "counts `succeeded`, counts `failures`" -- constructing the
    full seven-field result would imply the guard depends on `coverage`,
    `quota_aborted` and the rest, which it deliberately does not.
    """

    def __init__(self, succeeded=(), failures=None):
        self.succeeded = tuple(succeeded)
        self.failures = dict(failures or {})


def _dataset(tmp_path: Path) -> StockDataset:
    """A real `StockDataset` on a real (possibly absent) tmp_path raw root.

    `symbols=None` on purpose: a non-None symbol list makes `BaseDataset`'s
    config setter call `_reset_symbols()`, which falls back to a full-range
    `from_raw_data()` at CONSTRUCTION time -- the very densification these
    tests are about guarding, fired before the guard could run.
    """
    return StockDataset(
        DatasetConfig(
            raw_data_dir_path=str(tmp_path / "raw" / "tiingo"),
            zarr_file_path=str(tmp_path / "stock.zarr"),
            market="us_equity",
            frequency="1d",
            vendor="tiingo",
        )
    )


def _write_shard(tmp_path: Path) -> Path:
    """One file under the raw root, so `has_raw_data()` answers True.

    A real file on a real path rather than a monkeypatched predicate: the
    whole point of the second half of the guard's condition is that it probes
    the DISK, and a patched probe would assert the guard's shape while
    proving nothing about the fact it reads.
    """
    shard = tmp_path / "raw" / "tiingo" / "month=2024-01" / "part-0.pqt"
    shard.parent.mkdir(parents=True, exist_ok=True)
    shard.write_bytes(b"")
    return shard


# ---------------------------------------------------------------------------
# (a) The guard: three behaviours, and the middle one is why it probes disk
# ---------------------------------------------------------------------------


def test_zero_success_and_an_empty_raw_root_refuses_with_a_readable_message(
    tmp_path,
):
    """The gap itself: every symbol failed, nothing on disk, so there is
    nothing to convert and the run must say so and exit non-zero.

    `SystemExit` with a string message is a NON-ZERO exit (CPython prints the
    message to stderr and exits 1), which is the contract here: the previous
    behaviour was an uncaught `ValueError` traceback pointing at
    `dataset/stock.py`, i.e. also non-zero but unreadable and misattributed.

    The message is asserted for what it must CONTAIN -- the raw root, both
    counts, and where to read the per-symbol reasons -- rather than verbatim,
    so rewording it stays free while dropping a fact does not.
    """
    dataset = _dataset(tmp_path)
    result = _FakeResult(succeeded=(), failures={"AAPL": "401", "MSFT": "401"})

    with pytest.raises(SystemExit) as excinfo:
        refuse_conversion_without_raw_data(dataset, result)

    message = str(excinfo.value)
    assert str(tmp_path / "raw" / "tiingo") in message
    assert "0 symbol(s) successfully" in message
    assert "2 failed" in message
    assert "SourceInspector.failures()" in message
    assert "No Zarr" in message


def test_zero_success_but_raw_data_on_disk_converts_anyway(tmp_path):
    """The half of the condition that is easy to leave out, and the reason the
    guard probes the disk instead of counting successes.

    A run in which every symbol was SKIPPED because its watermark already
    covers the window reports zero successes too -- and it has raw data that
    must still be converted. A guard written as `if not result.succeeded:
    refuse` would refuse this legitimate run, turning a resume into a failure.
    """
    dataset = _dataset(tmp_path)
    _write_shard(tmp_path)

    # Returns None rather than raising: the conversion proceeds.
    assert (
        refuse_conversion_without_raw_data(dataset, _FakeResult(succeeded=()))
        is None
    )


def test_a_successful_run_is_never_probed_or_refused(tmp_path):
    """Anything succeeded -> nothing to decide, and no disk probe at all.

    Asserted on an ABSENT raw root: if the guard probed disk unconditionally
    it would refuse here, since `has_raw_data()` is False. Passing therefore
    proves the short circuit rather than merely proving the happy path.
    """
    dataset = _dataset(tmp_path)
    assert not dataset.has_raw_data()

    assert (
        refuse_conversion_without_raw_data(dataset, _FakeResult(succeeded=("AAPL",)))
        is None
    )


def test_the_guard_and_the_dataset_decide_on_one_predicate():
    """The ancestor shape of BOTH 03.4 UAT gaps: one contract written twice.

    `_scan_raw`'s absent-root branch and the shell-level refusal must read the
    SAME fact, or a run can pass one and fail the other -- which is exactly
    what happened. Asserted structurally, because the failure mode is a
    SECOND spelling appearing, and both spellings answering True on the day it
    is added.
    """
    cli_source = (REPO_ROOT / "quantlab" / "utils" / "cli.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(cli_source)
    guard = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "refuse_conversion_without_raw_data"
    )
    probes = [
        node.func.attr
        for node in ast.walk(guard)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]
    assert "has_raw_data" in probes, (
        "the CLI guard must ask the dataset, not the filesystem directly"
    )
    assert not {"rglob", "glob", "iterdir", "exists"} & set(probes), (
        "quantlab/utils/cli.py has grown its own raw-root probe. The predicate "
        "lives once, on StockDataset.has_raw_data(); a second spelling is how "
        "the shell and the scan came to disagree about whether there was "
        "anything to convert."
    )

    # The dataset side, asserted on CALL nodes rather than on the text: this
    # file's own prose says `rglob("*.pqt")` and so does `has_raw_data`'s
    # docstring, and a substring scan counts both. Walking the AST of the
    # enclosing function is what tells a probe apart from a sentence about a
    # probe.
    stock_tree = ast.parse(
        (REPO_ROOT / "quantlab" / "dataset" / "stock.py").read_text(encoding="utf-8")
    )
    owners = [
        function.name
        for function in ast.walk(stock_tree)
        if isinstance(function, ast.FunctionDef)
        for call in ast.walk(function)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "rglob"
    ]
    assert owners == ["has_raw_data"], (
        f"the raw-presence probe must live once, in has_raw_data(); found it "
        f"in {owners}. `_scan_raw` reading its own copy is precisely how the "
        f"scan and the shell came to disagree about whether there was "
        f"anything to convert."
    )


# ---------------------------------------------------------------------------
# (b) `--to-zarr` is registered on all three shells, and defaults to off
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shell", INGEST_SHELLS)
def test_every_shell_registers_to_zarr_and_defaults_to_not_converting(shell):
    """Through the REAL parser, never a source scan (L-5).

    A source scan matches an `add_argument("--to-zarr", ...)` that sits in a
    branch which never executes, so it would report a flag the user cannot
    actually pass. `parse_args([])` cannot be fooled that way, and it also
    pins the DEFAULT, which is the half of this change that alters existing
    behaviour: `ingest_tiingo.py` and `ingest_alpaca.py` used to convert
    unconditionally.
    """
    import importlib

    module = importlib.import_module(shell.removesuffix(".py"))
    parser = module._build_arg_parser()

    assert "--to-zarr" in parser._option_string_actions, (
        f"{shell} does not register --to-zarr; the three front doors must "
        f"agree about what a run without arguments does."
    )
    action = parser._option_string_actions["--to-zarr"]
    assert action.default is False
    assert action.const is True, "--to-zarr must be a store_true, taking no value"


def test_alpaca_refuses_tick_with_to_zarr_at_exit_2():
    """The flag combination that cannot mean anything, refused rather than
    ignored.

    There is no tick conversion to opt into -- the dense [timestamp, symbol]
    panel cannot express an irregular event axis, so quotes/trades stop at raw
    (D-18). Silently dropping `--to-zarr` here would let a user believe a Zarr
    store was written; that silence is the whole subject of this file.

    Exit 2 and the `argparse` usage block, in the same shape as the existing
    `--data-type` validation. Run in a SUBPROCESS because `parser.error`
    raises `SystemExit` from inside argparse's own error handling, and because
    the exit code is the contract a shell script would read.
    """
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS_DIR / "ingest_alpaca.py"),
            "--symbols",
            "AAPL",
            "--frequency",
            "tick",
            "--data-type",
            "quotes",
            "--rows-per-symbol-day",
            "1000",
            "--to-zarr",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )

    assert completed.returncode == 2, completed.stderr
    assert "--to-zarr is not available with --frequency tick" in completed.stderr
    assert "irregular event axis" in completed.stderr


# ---------------------------------------------------------------------------
# Reachability: the guard is in front of every densification, in every shell
# ---------------------------------------------------------------------------


def _main_body(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.If) and "__main__" in ast.unparse(node.test):
            return node.body
    raise AssertionError(f"{path} has no __main__ block")


def _call_linenos(statements, predicate) -> list[int]:
    """Line numbers of matching CALL nodes.

    Copied in shape from `tests/test_volume_guard.py:_call_linenos` (the
    `tests/` copy convention: this directory is not a package, so a cross-test
    import would couple this file's collection to that file's import-time
    state). Counts `ast.Call` nodes rather than substrings, so a mention in a
    comment or a docstring cannot satisfy the assertion.
    """
    return sorted(
        node.lineno
        for statement in statements
        for node in ast.walk(statement)
        if isinstance(node, ast.Call) and predicate(node)
    )


def test_every_entry_point_that_densifies_refuses_first():
    """Scoped by REACHABILITY, not by script name.

    `ingest_us_equity.py --to-zarr` has the identical hole -- its
    `from_raw_data_chunked()` reaches the same absent-root `ValueError` -- and
    it was not in either bug report. Pinning this to the two scripts a user
    happened to run is a mistake this repository has made before: an earlier
    reachability guard scoped ITSELF to one script, so the second door's gap
    was invisible to it.

    So: any `__main__` that reaches a densification -- `from_raw_data` /
    `from_raw_data_chunked` directly, or `registry.convert()`, which is how
    all three shells reach it since 03.5 -- must also call the guard, and must
    call it FIRST. A guard that runs after the densification has already taken
    the traceback it exists to prevent.
    """
    densifying = 0
    for shell in INGEST_SHELLS:
        path = SCRIPTS_DIR / shell
        body = _main_body(path)

        def _is_densify(node) -> bool:
            # A bare `convert(...)` counts, and that arm is why this test did
            # not quietly start covering nothing. Since 03.5 the shells reach
            # the densification through
            # `quantlab.registry.convert()` rather than by naming
            # a Dataset method, so an attribute-only detector would find zero
            # densifying doors and every ordering assertion below would be
            # vacuous. The attribute arms stay: a shell that goes back to
            # calling `from_raw_data*` directly must still be caught.
            if isinstance(node.func, ast.Name) and node.func.id == "convert":
                return True
            return isinstance(node.func, ast.Attribute) and node.func.attr in (
                "from_raw_data",
                "from_raw_data_chunked",
            )

        def _is_refusal(node) -> bool:
            return (
                isinstance(node.func, ast.Name)
                and node.func.id == "refuse_conversion_without_raw_data"
            )

        densify_sites = _call_linenos(body, _is_densify)
        if not densify_sites:
            continue
        densifying += 1
        refusals = _call_linenos(body, _is_refusal)
        assert refusals, (
            f"{shell} densifies at line(s) {densify_sites} with no "
            f"refuse_conversion_without_raw_data in its __main__ body: a run "
            f"that fetched nothing reaches StockDataset's absent-root "
            f"ValueError as an uncaught traceback."
        )
        assert min(refusals) < min(densify_sites), (
            f"{shell}: the refusal at {refusals} must precede the "
            f"densification at {densify_sites}."
        )

    assert densifying == 3, (
        f"expected all three ingest shells to densify, found {densifying}. If "
        f"a door stopped densifying, say so here rather than letting this "
        f"test silently cover less than it claims."
    )


def test_refusal_precedes_every_symbol_bearing_dataset_construction():
    """The refusal must also precede the CONSTRUCTION of a symbols-bearing
    `StockDataset`, not merely the explicit `from_raw_data()` call.

    The sibling ordering assertion above compares the refusal against the
    explicit densification call sites, and that is exactly the blind spot this
    test closes. `BaseDataset`'s config setter USED TO call `_reset_symbols()`
    for any non-None symbol list, which called `read()`, caught the
    not-yet-written store's `FileNotFoundError` and recovered by densifying
    the full range through `from_raw_data()` at CONSTRUCTION time -- with no
    `from_raw_data` token anywhere at the call site for an AST scan to find.

    So `dataset = StockDataset(ds_config); refuse_conversion_without_raw_data(
    dataset, result)` READ correctly and passed the sibling assertion, while
    raising `stock.py`'s absent-root `ValueError` one line before the guard it
    was placed in front of. That shipped, and the UAT reproduction still
    tracebacked with the guard nominally in place (G-03.4-1a, second order).

    **`df7bfe9` deleted `_reset_symbols` outright**, so construction no longer
    densifies for ANY config and the second-order trap this test was written
    for cannot currently be sprung. Since 03.5 the shells hold only the
    symbol-free probe and delegate the conversion to `registry.convert()`, so
    the loop below finds no bare construction in any of them and every
    iteration falls through. KEPT DELIBERATELY, and said out loud rather than
    left for a reader to discover: this is a regression guard, not live
    coverage. Restoring an eager construction-time read -- in the config
    setter, in a `__init__`, or in a subclass -- puts the trap back, and it is
    a shape this repository has already shipped once.

    A construction is exempt only when its argument is a `replace(...,
    symbols=None)` -- the symbol-free probe the guard is allowed to hold.
    All three shells state that shape at the call site, including
    `ingest_us_equity.py`, whose `ds_config` already carries `symbols=None`
    from its factory: the redundant `replace` there is what makes the property
    checkable where it is relied on, rather than one factory call away.
    """
    for shell in INGEST_SHELLS:
        path = SCRIPTS_DIR / shell
        body = _main_body(path)

        def _is_bare_construction(node) -> bool:
            if not (isinstance(node.func, ast.Name) and node.func.id == "StockDataset"):
                return False
            if not node.args:
                return False
            arg = node.args[0]
            # The exempt shape: StockDataset(replace(ds_config, symbols=None)).
            return not (
                isinstance(arg, ast.Call)
                and isinstance(arg.func, ast.Name)
                and arg.func.id == "replace"
                and any(
                    kw.arg == "symbols"
                    and isinstance(kw.value, ast.Constant)
                    and kw.value.value is None
                    for kw in arg.keywords
                )
            )

        constructions = _call_linenos(body, _is_bare_construction)
        if not constructions:
            continue

        refusals = _call_linenos(
            body,
            lambda node: isinstance(node.func, ast.Name)
            and node.func.id == "refuse_conversion_without_raw_data",
        )
        assert refusals, (
            f"{shell} constructs a symbols-bearing StockDataset at line(s) "
            f"{constructions} with no refuse_conversion_without_raw_data in "
            f"its __main__ body."
        )
        assert min(refusals) < min(constructions), (
            f"{shell}: the refusal at {refusals} must precede the "
            f"StockDataset construction at {constructions}. A construction "
            f"that carries a non-None symbol list densifies inside "
            f"_reset_symbols(), so a guard placed after it can never run."
        )
