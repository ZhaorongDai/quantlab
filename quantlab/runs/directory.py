"""Run directories: the mechanism every run type of quantlab is written and read through.

A run is a directory. It holds its files and, written last, ``run.json``,
the record that makes the directory a run: a directory without it is not a
run, so a crash can never leave one that opens. Every ``run.json`` starts
with the same header:

- ``format_version``: one version shared by every run type
  (``FORMAT_VERSION``); a run written in another version is refused with a
  message to re-run or retrain it, never read wrongly;
- ``kind``: what the run is, which decides the type ``open_run`` returns;
- ``written_at``: when the record was written, an ISO 8601 UTC timestamp.

The rest of the record belongs to the run type. Paths inside it are relative
to the run's directory (``relative_name`` / ``recorded_path``), so a run copied
elsewhere still opens. A run whose files are written at once can be
written through ``staged``: into a hidden sibling first, renamed into place
when complete.

The run types are ``quantlab.runs.trained_run.TrainedRun`` (kinds
``"model"``, ``"ensemble"``, ``"walk_forward"``) and
``quantlab.runs.backtest_run.BacktestRun`` (kinds ``"run"``, ``"run_cv"``,
``"run_weights"``, ``"fold"``). ``open_run`` reads any run
and returns its type, which is imported only when a run of that kind is
opened.

This module imports no quantlab module outside ``quantlab.utils``; the run
types in ``KINDS`` are named by import path and imported only by
``open_run``.

Examples
--------
>>> run = open_run("models/MyHead_trial_20240601_120000_000000")
>>> type(run).__name__, run.kind
('TrainedRun', 'model')
"""

import importlib
import json
import shutil
from collections.abc import Collection, Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from quantlab.utils.atomic import write_json_atomically
from quantlab.utils.jsonable import to_jsonable

#: The ``run.json`` structure every run type writes and reads.
FORMAT_VERSION = 2

RUN_FILE = "run.json"

#: Which type reads a run of each kind, as an import path, imported on use.
KINDS = {
    "model": "quantlab.runs.trained_run.TrainedRun",
    "ensemble": "quantlab.runs.trained_run.TrainedRun",
    "walk_forward": "quantlab.runs.trained_run.TrainedRun",
    "run": "quantlab.runs.backtest_run.BacktestRun",
    "run_cv": "quantlab.runs.backtest_run.BacktestRun",
    "run_weights": "quantlab.runs.backtest_run.BacktestRun",
    "fold": "quantlab.runs.backtest_run.BacktestRun",
}


def run_directory(path: Path | str) -> Path:
    """Return the run directory of ``path``: ``path`` itself, or the directory of a file in it.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.

    Examples
    --------
    >>> run_directory(unit / "run.json") == unit
    True
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")
    return path if path.is_dir() else path.parent


def record_path(directory: Path | str) -> Path:
    """Return where the ``run.json`` of the run in ``directory`` is.

    Examples
    --------
    >>> record_path("unit").as_posix()
    'unit/run.json'
    """
    return Path(directory) / RUN_FILE


def read_record(directory: Path | str, kinds: Collection[str] | None = None) -> dict:
    """Return the ``run.json`` of the run in ``directory``, after checking its header.

    Parameters
    ----------
    directory : Path or str
        The run's directory.
    kinds : collection of str, optional
        The kinds the caller reads; any kind when not given.

    Returns
    -------
    dict
        The whole record, header included.

    Raises
    ------
    ValueError
        If ``directory`` has no ``run.json``, it was written in a
        ``format_version`` other than ``FORMAT_VERSION``, or its kind is not
        one of ``kinds``.

    Examples
    --------
    >>> read_record(unit)["kind"]
    'model'
    """
    path = record_path(directory)
    if not path.is_file():
        raise ValueError(
            f"{directory} has no {RUN_FILE}, so it is not a run this quantlab "
            f"can read; re-run or retrain it"
        )
    record = json.loads(path.read_text(encoding="utf-8"))
    version = record.get("format_version") if isinstance(record, dict) else None
    if version != FORMAT_VERSION:
        raise ValueError(
            f"{path} has format_version {version}, but this quantlab reads "
            f"format_version {FORMAT_VERSION}; re-run or retrain it"
        )
    if kinds is not None and record.get("kind") not in kinds:
        raise ValueError(
            f"{path} is a run of kind {record.get('kind')!r}, not one of {sorted(kinds)}"
        )
    return record


def write_record(directory: Path | str, kind: str, record: dict) -> Path:
    """Write ``record`` as the ``run.json`` of ``directory``, atomically, under the header.

    Call it last: the record is what makes the directory a run. NaN and inf
    become null.

    Parameters
    ----------
    directory : Path or str
        The run's directory, already holding its files.
    kind : str
        The run's kind, one ``open_run`` knows.
    record : dict
        The run type's own entries; paths in it relative to ``directory``.

    Returns
    -------
    Path
        The ``run.json`` written.

    Examples
    --------
    >>> path = write_record(unit, "model", {...})
    >>> read_record(unit)["kind"]
    'model'
    """
    if kind not in KINDS:
        raise ValueError(f"unknown run kind {kind!r}; known: {sorted(KINDS)}")
    path = record_path(directory)
    header = {
        "format_version": FORMAT_VERSION,
        "kind": kind,
        "written_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json_atomically(path, to_jsonable({**header, **record}), indent=2)
    return path


def relative_name(path: Path | str, directory: Path | str) -> str:
    """Return ``path`` relative to the run ``directory``, as recorded in ``run.json``.

    Raises
    ------
    ValueError
        If ``path`` is not inside ``directory``.

    Examples
    --------
    >>> relative_name(unit / "fold_0", unit)
    'fold_0'
    """
    return Path(path).absolute().relative_to(Path(directory).absolute()).as_posix()


def recorded_path(directory: Path | str, name: str | None) -> Path | None:
    """Return the path a ``run.json`` entry names, or None when nothing is recorded.

    The inverse of ``relative_name``.

    Examples
    --------
    >>> recorded_path(unit, "fold_0") == unit / "fold_0", recorded_path(unit, None)
    (True, None)
    """
    return None if name is None else Path(directory) / name


@contextmanager
def staged(final: Path | str) -> Iterator[Path]:
    """Yield a hidden staging directory that becomes ``final`` when the block succeeds.

    The run's files are written into ``.{name}.partial`` beside ``final``,
    which is renamed into place when the block exits normally (a rename
    within one filesystem is atomic). On any exception, including
    ``KeyboardInterrupt``, the staging directory is removed and the error
    re-raised, so ``final`` is complete or absent.

    Raises
    ------
    RuntimeError
        If ``final`` already exists; it is never overwritten.

    Examples
    --------
    >>> with staged(root / "run_1") as staging:
    ...     write_files(staging)
    >>> (root / "run_1").is_dir()
    True
    """
    final = Path(final)
    if final.exists():
        raise RuntimeError(f"{final} already exists")
    staging = final.parent / f".{final.name}.partial"
    staging.mkdir(parents=True)
    try:
        yield staging
        if final.exists():
            raise RuntimeError(f"{final} already exists")
        staging.rename(final)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def open_run(path: Path | str):
    """Read the run at ``path`` as the type its kind names.

    Parameters
    ----------
    path : Path or str
        The run's directory, its ``run.json``, or a file the run type opens
        from (a model's checkpoint).

    Returns
    -------
    object
        The run, as its type's ``open(path)`` reads it.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the run has no ``run.json``, another ``format_version``, or a
        kind no run type reads.

    Examples
    --------
    >>> open_run(checkpoint) == TrainedRun.open(checkpoint)
    True
    """
    directory = run_directory(path)
    kind = read_record(directory).get("kind")
    if kind not in KINDS:
        raise ValueError(
            f"{record_path(directory)} is a run of kind {kind!r}, which "
            f"no run type of this quantlab reads; known: {sorted(KINDS)}"
        )
    module, _, name = KINDS[kind].rpartition(".")
    return getattr(importlib.import_module(module), name).open(path)
