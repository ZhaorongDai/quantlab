"""WR-04: a cleanup failure inside the rebuild rollback must not replace the
failure it is cleaning up after.

`BaseDataset._restore_rebuild_asides` runs from inside
`from_raw_data_chunked`'s `except BaseException: ... raise` handler. An
exception raised INSIDE an exception handler replaces the exception being
handled: the operator of a multi-hour rebuild is shown "Directory not empty"
and the real cause survives only in `__context__`. That is the whole of WR-04,
which `03.6-REVIEW.md` recorded as `advisory` with
`evidence_status: not independently reproduced`. This module is the missing
evidence: both arms were written RED-FIRST against HEAD `00cd680` and observed
failing before a single production line moved.

**The induction route, and the substitution it makes.** The review's reported
trigger is a `shutil.rmtree(..., ignore_errors=True)` that silently fails on a
permission/NFS/SIGKILL residue, leaving a NON-EMPTY directory at the store
path, so the following `os.replace` raises `OSError` with POSIX `ENOTEMPTY`.
A test cannot portably manufacture an unremovable directory. It CAN
manufacture the identical control flow: put a regular FILE at
`asides["store"]` and a real directory at `asides["store_aside"]`. Then
`Path(asides["store"]).exists()` is True, `shutil.rmtree(<a regular file>,
ignore_errors=True)` swallows its `NotADirectoryError` and leaves the path in
place -- the exact silent no-op the review's mechanism needs -- and the
following `os.replace(<directory>, <existing regular file>)` raises `OSError`
with `ENOTDIR`. Same exception class, same escape point, same consequence:
`ENOTDIR` stands in for the reported `ENOTEMPTY`.

Everything here is offline: `tmp_path` directories, no store, no network.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from loguru import logger

from quantlab.base.config import BaseDatasetConfig
from quantlab.base.data import BaseDataset

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


class _TrivialDataset(BaseDataset):
    """The minimal concrete `BaseDataset`, mirroring
    `tests/test_chunked_ingest.py`'s `_UnboundedDataset`.

    `_restore_rebuild_asides` reads nothing off `self` but `class_name`, so
    nothing heavier is needed and no part of the conversion loop is exercised
    by this module. Defined locally rather than imported across test files.
    """

    def __init__(self, config: BaseDatasetConfig, panel: xr.Dataset):
        self._panel = panel
        super().__init__(config)

    def _raw_data_to_xr(self) -> xr.Dataset:
        return self._panel.copy(deep=True)


def _dataset(tmp_path: Path) -> _TrivialDataset:
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
    values = np.full((len(dates), 2), 100.0)
    panel = xr.Dataset(
        {
            name: (["timestamp", "symbol"], values.copy())
            for name in ("open", "close")
        },
        coords={"timestamp": dates, "symbol": ["A", "B"]},
    )
    config = BaseDatasetConfig(zarr_file_path=str(tmp_path / "panel.zarr"))
    return _TrivialDataset(config, panel)


def _captured_logs():
    """A loguru sink at WARNING, tagged with the level name.

    loguru does not propagate to stdlib `logging`, so pytest's `caplog` sees
    nothing -- the same constraint `tests/test_chunked_ingest.py`'s
    `_captured_warnings` records. WARNING rather than ERROR because this
    module asserts on BOTH levels and must be able to tell them apart: the
    success path logs WARNING, the cleanup failure logs ERROR.
    """
    messages: list[str] = []
    sink_id = logger.add(
        messages.append, level="WARNING", format="{level.name}|{message}"
    )
    return messages, sink_id


def _asides(root: Path, *, restorable: bool) -> dict:
    """An `asides` dict with the five real keys the production code builds.

    `restorable=True` reproduces the ordinary shape: a real store directory
    where the partial rebuild sits and a real aside directory holding the
    complete pre-rebuild copy. `restorable=False` swaps the store for a
    regular FILE, which is the ENOTDIR induction the module docstring
    describes.
    """
    root.mkdir(parents=True, exist_ok=True)
    store = root / "panel.zarr"
    store_aside = root / "panel.zarr.superseded.tmp"

    store_aside.mkdir()
    (store_aside / "marker.txt").write_text("pre-rebuild")

    if restorable:
        store.mkdir()
        (store / "marker.txt").write_text("partial rebuild")
    else:
        store.write_text("a regular file, not a directory")

    return {
        "store": str(store),
        "store_aside": str(store_aside),
        "ledger": str(root / "ledger.json"),
        "ledger_aside": str(root / "ledger.json.superseded.tmp"),
        "ledger_existed": False,
    }


# ---------------------------------------------------------------------------
# WR-04
# ---------------------------------------------------------------------------


def test_a_failed_restore_does_not_replace_the_original_exception(
    tmp_path: Path,
) -> None:
    """The exception that actually failed the rebuild reaches the caller.

    This is WR-04 stated as a test. The sandwich below is
    `from_raw_data_chunked`'s handler in miniature: something raises, the
    handler restores the asides, the handler re-raises. If the restore raises
    on its way out, the `raise` never runs and the operator is handed a
    cleanup error instead of the cause.

    RED on HEAD `00cd680`: an `OSError` (`ENOTDIR`) escapes
    `_restore_rebuild_asides` and reaches the caller, so this
    `pytest.raises(RuntimeError)` fails.
    """
    dataset = _dataset(tmp_path)
    asides = _asides(tmp_path / "wreck", restorable=False)

    with pytest.raises(RuntimeError, match="original failure"):
        try:
            raise RuntimeError("original failure")
        except BaseException:
            dataset._restore_rebuild_asides(asides)
            raise


def test_a_failed_restore_logs_the_surviving_aside_and_reports_no_rollback(
    tmp_path: Path,
) -> None:
    """A cleanup failure is REPORTED, not raised, and reports no rollback.

    Three things, in the order the operator needs them: the call returns
    rather than raising; an ERROR names the aside path that still holds the
    complete pre-rebuild copy, so the manual remedy is in the log rather than
    inferred; and the return value is `False`, so
    `ConversionResult.rebuild_rolled_back` cannot claim a rollback that did
    not happen.

    RED on HEAD `00cd680`: the call raises `OSError`, so it never returns and
    none of the three hold.
    """
    dataset = _dataset(tmp_path)
    asides = _asides(tmp_path / "wreck", restorable=False)

    messages, sink_id = _captured_logs()
    try:
        restored = dataset._restore_rebuild_asides(asides)
    finally:
        logger.remove(sink_id)

    assert restored is False
    errors = [m for m in messages if m.startswith("ERROR|")]
    assert errors, f"no ERROR was logged; captured: {messages}"
    assert any(asides["store_aside"] in m for m in errors), (
        "the ERROR does not name the aside path that still holds the "
        f"complete copy; captured: {errors}"
    )
