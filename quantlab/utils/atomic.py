"""Atomic JSON, JSON-lines and parquet file writes.

Every JSON sidecar quantlab persists (resume ledgers, acquisition watermarks,
failure manifests) goes through ``write_json_atomically``, every REST page
through ``write_jsonl_atomically``, every parquet file written whole from
memory (raw vendor pulls, the Ticker map's days) through
``write_parquet_atomically``, and a file too large for memory through
``move_durably`` once it is finished. The payload is
written to a temporary file in the destination's own directory and then
renamed over the destination, so an interrupted write leaves either the
previous complete file or the new one, never a truncated file that a resumed
run would fail to parse.

No lock is taken: two concurrent writers still race, but the loser sees a
complete file rather than a fragment. This module imports only the standard
library so any layer can use it without creating an import cycle.
"""

import json
import os
import tempfile
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TextIO

__all__ = [
    "make_directories_durably",
    "move_durably",
    "sync_directory",
    "write_json_atomically",
    "write_jsonl_atomically",
    "write_parquet_atomically",
]


def write_json_atomically(path: str | Path, payload: object, **json_kwargs) -> None:
    """Serialise ``payload`` as JSON to ``path`` with an atomic replace.

    Missing parent directories are created, and the file gets the
    permissions a plain write would (``0o666`` less the umask). ``json_kwargs`` is forwarded
    unchanged to ``json.dump``, so each caller keeps its own formatting
    (compact, ``indent=2``, ``sort_keys=True`` and so on).

    On any exception, including ``KeyboardInterrupt``, the temporary file is
    removed and the exception propagates, leaving the previous file (if any)
    untouched.

    Parameters
    ----------
    path : str | Path
        Destination file.
    payload : object
        Any object ``json.dump`` accepts.
    **json_kwargs
        Formatting options passed to ``json.dump``.

    Examples
    --------
    >>> write_json_atomically("run/watermark.json", {"last": "2024-01-31"})
    >>> write_json_atomically("run/manifest.json", data, indent=2, sort_keys=True)
    """
    with _atomic_text_file(path) as handle:
        json.dump(payload, handle, **json_kwargs)


@contextmanager
def _atomic_text_file(path: str | Path) -> Iterator[TextIO]:
    """Yield a text file that replaces ``path``, synced, when the block ends.

    On any exception the temporary file is removed and ``path`` is left as it was.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # The temp file must live in the destination's directory: `os.replace` is
    # atomic only as a same-filesystem rename. Staged in the system temp dir it
    # could become a cross-device copy, which is interruptible.
    handle = tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=str(destination.parent),
        prefix=destination.name + ".",
        suffix=".tmp",
        delete=False,
    )
    try:
        with handle:
            yield handle
            handle.flush()
            os.fchmod(handle.fileno(), _PLAIN_MODE)
            # fsync before the rename, and after the mode change so the mode is
            # durable too: `flush()` only reaches the OS page cache, and a power
            # loss after the rename could otherwise publish an empty file over a
            # good one.
            os.fsync(handle.fileno())
        os.replace(handle.name, destination)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise


def _plain_mode() -> int:
    """Return the mode a plain write gives a file: ``0o666`` less the umask.

    Read once, at import: setting the process umask to read it back is not
    thread-safe, so no write does it.
    """
    umask = os.umask(0)
    os.umask(umask)
    return 0o666 & ~umask


#: The mode the writers give their file before the rename. ``NamedTemporaryFile``
#: creates it owner-only and ``os.replace`` keeps that mode, while a file in the
#: shared data tier must stay as readable as a plain write leaves it.
_PLAIN_MODE = _plain_mode()


def write_jsonl_atomically(path: str | Path, records: Iterable[dict]) -> None:
    """Write ``records`` to ``path``, one JSON object per line, durably.

    Keys are sorted and non-ASCII text is kept, so a record's line is the same
    on every pull. The file is synced before its rename, and the rename and any
    directory created for it after, so a watermark written next cannot outlive
    the file through a power loss.

    Parameters
    ----------
    path : str | Path
        Destination file; missing directories are created.
    records : iterable of dict
        The records, written in order.

    Examples
    --------
    >>> import tempfile
    >>> page = Path(tempfile.mkdtemp()) / "pages" / "000001.jsonl"
    >>> write_jsonl_atomically(page, [{"ticker": "AAPL"}])
    >>> page.read_text()
    '{"ticker": "AAPL"}\\n'
    """
    destination = Path(path)
    make_directories_durably(destination.parent)
    with _atomic_text_file(destination) as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    sync_directory(destination.parent)


def sync_directory(path: str | Path) -> None:
    """Make the entries of directory ``path`` durable, such as a rename into it.

    Parameters
    ----------
    path : str | Path
        An existing directory.

    Examples
    --------
    >>> import tempfile
    >>> sync_directory(tempfile.mkdtemp())
    """
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def move_durably(source: str | Path, destination: str | Path) -> None:
    """Rename a finished file to ``destination``, durably, in one step.

    For a file too large to hold in memory, written by its own writer. It is
    synced before the rename and the rename after, so a power loss leaves the
    previous ``destination`` or the whole new one.

    Parameters
    ----------
    source : str | Path
        The finished file, on the destination's filesystem.
    destination : str | Path
        Where it goes; missing directories are created.

    Examples
    --------
    >>> import tempfile
    >>> root = Path(tempfile.mkdtemp())
    >>> _ = (root / "partial").write_text("rows")
    >>> move_durably(root / "partial", root / "table" / "bulk.parquet")
    >>> (root / "table" / "bulk.parquet").read_text()
    'rows'
    """
    descriptor = os.open(source, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    destination = Path(destination)
    make_directories_durably(destination.parent)
    os.replace(source, destination)
    sync_directory(destination.parent)


def make_directories_durably(path: str | Path) -> None:
    """Create directory ``path`` and its missing parents, each one's entry synced.

    Syncing only the last directory is not enough: a power loss could still
    drop an entry higher up, and everything under it.

    Parameters
    ----------
    path : str | Path
        The directory.

    Examples
    --------
    >>> import tempfile
    >>> root = Path(tempfile.mkdtemp())
    >>> make_directories_durably(root / "rest" / "2024")
    >>> (root / "rest" / "2024").is_dir()
    True
    """
    missing = []
    directory = Path(path)
    while not directory.exists():
        missing.append(directory)
        directory = directory.parent
    for directory in reversed(missing):
        directory.mkdir(exist_ok=True)
        sync_directory(directory.parent)


def write_parquet_atomically(path: str | Path, frame) -> None:
    """Write ``frame`` as parquet to ``path`` with an atomic replace.

    The frame is written to a temporary file in the destination's directory
    (created if missing) and renamed over it, so an interrupted write leaves
    either the previous complete file or the new one. The file gets the
    permissions a plain write would (``0o666`` less the umask). On any exception the
    temporary file is removed and the exception propagates.

    Parameters
    ----------
    path : str | Path
        Destination file.
    frame : polars.DataFrame
        Anything with a ``write_parquet(path)`` method; this module imports
        only the standard library.

    Examples
    --------
    >>> import polars as pl
    >>> write_parquet_atomically("raw/tickers.parquet", pl.DataFrame({"ticker": ["AAPL"]}))  # doctest: +SKIP
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Same directory as the destination, so the rename stays on one filesystem.
    handle = tempfile.NamedTemporaryFile(
        dir=str(destination.parent), prefix=destination.name + ".", suffix=".tmp", delete=False
    )
    try:
        with handle:
            # Through the descriptor, before the frame is written over the file.
            os.fchmod(handle.fileno(), _PLAIN_MODE)
        frame.write_parquet(handle.name)
        move_durably(handle.name, destination)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise
