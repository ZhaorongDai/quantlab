"""G-03.4-1: nothing fetched must not end in a conversion traceback.

An ingest shell that walked into `StockDataset.from_raw_data()` after a run in
which every symbol failed ended on `quantlab/dataset/stock.py`'s uncaught
absent-root `ValueError` -- a traceback that reads like a bug in the
conversion layer when the real event was "the vendor returned nothing".
`quantlab.utils.cli.refuse_conversion_without_raw_data` translates that into a
clean non-zero exit BEFORE the densification.

Every test here drives the guard as a library function against a real
`StockDataset` and a temporary raw root. The shells that call it are not
exercised from here: `scripts/` is not importable from the test suite.
"""

import ast
from pathlib import Path

import pytest

from quantlab.base.config import DatasetConfig
from quantlab.dataset.stock import StockDataset
from quantlab.utils.cli import refuse_conversion_without_raw_data

REPO_ROOT = Path(__file__).resolve().parent.parent


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
