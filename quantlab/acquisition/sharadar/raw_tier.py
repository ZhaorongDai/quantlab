"""The writers of Sharadar's raw tier: bulk pulls, watermarks and the TICKERS history.

``SharadarClient`` (``quantlab.acquisition.sharadar.client``) pulls tables
into ``<download-dir>/sharadar/<table>/`` and records what it pulled with
these writers. The layout, and every reader of it (``raw_through``,
``scan_raw_table``, ``pull_time``, ``tickers_history``, ``snapshot_for``),
stay in ``quantlab.dataset.sharadar.tables``: the dataset layer reads the
raw tier, only the acquisition layer writes it.

A table's watermark is a dated watermark (``quantlab.utils.watermark``)
labelled here: ``<sharadar>/<table>/_watermark.json`` holding ``table`` and
``through``.
Unlike Massive's watermarks it may move back: a date-window pull records the
window's last day, whatever the watermark was.

A bulk pull commits in one rename (``publish_bulk``): the file's name carries
its pull time, and readers take the newest bulk and ignore pulls older than it.
What follows the rename is repeatable, so a process killed at any point leaves
readers correct data, and the next bulk pull finishes the cleanup.

Examples
--------
>>> import tempfile
>>> from datetime import date
>>> from quantlab.dataset.sharadar.tables import read_watermark
>>> root = tempfile.mkdtemp()
>>> write_table_watermark(root, "sep", date(2024, 1, 11))
>>> read_watermark(root, "sep")
datetime.date(2024, 1, 11)
"""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path

import polars as pl

from quantlab.dataset.sharadar.tables import (
    BULK_PREFIX,
    TABLES,
    UPDATED_PREFIX,
    WINDOW_PREFIX,
    bulk_file,
    latest_bulk,
    pull_time,
    raw_files,
    raw_table_dir,
    snapshot_for,
    table,
    tickers_history_file,
    tickers_snapshot_file,
    tickers_snapshots,
)
from quantlab.utils.atomic import make_directories_durably, move_durably, write_parquet_atomically
from quantlab.utils.watermark import WATERMARK_FILE, write_watermark

#: Prefix of the scratch directory of a bulk pull, beside the table's raw files.
_SCRATCH_PREFIX = ".pull-"

#: What one row of the TICKERS history is: a security under one ticker in one table.
_HISTORY_KEY = ["table", "permaticker", "ticker"]


def write_table_watermark(vendor_root: str | Path, code: str, through: date) -> None:
    """Record that a table's raw tier is complete through ``through``.

    Written after a bulk pull (the day of the pull) and after each
    date-window or updated pull, once its rows are on disk; an interrupted
    pull leaves the previous one. Read back by
    ``quantlab.dataset.sharadar.tables.read_watermark``.

    Examples
    --------
    Writes ``/data/downloads/sharadar/sep/_watermark.json``::

        write_table_watermark("/data/downloads/sharadar", "sep", date(2024, 1, 11))
    """
    write_watermark(raw_table_dir(vendor_root, code) / WATERMARK_FILE, through, {"table": table(code).code})


@contextmanager
def bulk_scratch(vendor_root: str | Path, code: str) -> Iterator[Path]:
    """Yield a scratch directory beside a table's raw files for one bulk pull.

    It is on the table's filesystem, so the finished file can be renamed into
    place, and readers never look in it. What a killed pull left is removed
    first.

    Parameters
    ----------
    vendor_root : str or Path
        ``<download-dir>/sharadar``.
    code : str
        The table's code.

    Yields
    ------
    Path
        The scratch directory, removed afterwards.

    Raises
    ------
    ValueError
        If the table holds a bulk file ``latest_bulk`` refuses, before anything
        is downloaded.

    Examples
    --------
    >>> import tempfile
    >>> with bulk_scratch(tempfile.mkdtemp(), "sep") as scratch:
    ...     scratch.is_dir()
    True
    """
    latest_bulk(vendor_root, code)
    directory = raw_table_dir(vendor_root, code)
    make_directories_durably(directory)
    _remove_scratch(directory)
    with tempfile.TemporaryDirectory(dir=directory, prefix=_SCRATCH_PREFIX) as scratch:
        yield Path(scratch)


def _remove_scratch(directory: Path) -> None:
    """Delete the scratch directories killed pulls left in a table's directory."""
    for leftover in directory.glob(f"{_SCRATCH_PREFIX}*"):
        shutil.rmtree(leftover, ignore_errors=True)


def finish_interrupted_bulk(vendor_root: str | Path, code: str) -> None:
    """Remove what a killed bulk pull left: its scratch and the pulls it superseded.

    Readers ignore both, so this only frees the disk. Every pull of a table
    starts here, since a table refreshed by windows may not see another bulk.

    Examples
    --------
    >>> import tempfile
    >>> finish_interrupted_bulk(tempfile.mkdtemp(), "sep")
    """
    _remove_scratch(raw_table_dir(vendor_root, code))
    _remove_superseded(vendor_root, code)


def publish_bulk(
    vendor_root: str | Path, code: str, pulled_at: datetime, rows: pl.LazyFrame, scratch: Path
) -> Path:
    """Commit a table's bulk pull with one rename, then clean up after it.

    Once the file is renamed, readers take it and ignore every older pull, so
    nothing after the rename has to happen for the data to read correctly. A
    TICKERS pull keeps a snapshot of itself (``tables.tickers_snapshot_file``)
    first, so a committed pull always has one to be folded. The rest is
    repeatable: older pulls are deleted and the TICKERS history is folded and
    pruned. The bulk file is durable when this returns, so the caller may move
    the watermark.

    Parameters
    ----------
    vendor_root : str or Path
        ``<download-dir>/sharadar``.
    code : str
        The table's code.
    pulled_at : datetime
        When the pull started (UTC), so a pull made after that overlays it.
    rows : polars.LazyFrame
        The table's rows.
    scratch : Path
        This pull's ``bulk_scratch``.

    Returns
    -------
    Path
        The bulk file.

    Examples
    --------
    >>> import tempfile
    >>> from datetime import UTC
    >>> from quantlab.dataset.sharadar.tables import pull_time
    >>> root = tempfile.mkdtemp()
    >>> pulled = datetime(2024, 1, 11, 8, 30, tzinfo=UTC)
    >>> rows = pl.LazyFrame({"ticker": ["AAPL"], "close": [185.0]})
    >>> with bulk_scratch(root, "sep") as scratch:
    ...     path = publish_bulk(root, "sep", pulled, rows, scratch)
    >>> path.name, pull_time(path) == pulled
    ('bulk_20240111T083000000000.parquet', True)
    """
    target = bulk_file(vendor_root, code, pulled_at)
    finished = scratch / target.name
    rows.sink_parquet(finished)
    if code == "tickers":
        snapshot = tickers_snapshot_file(vendor_root, pulled_at)
        shutil.copyfile(finished, scratch / snapshot.name)
        move_durably(scratch / snapshot.name, snapshot)
    move_durably(finished, target)  # the commit
    _remove_superseded(vendor_root, code)
    # A snapshot about to be pruned is kept in the TICKERS history.
    fold_tickers_history(vendor_root)
    prune_tickers_snapshots(vendor_root)
    return target


def _remove_superseded(vendor_root: str | Path, code: str) -> None:
    """Delete a table's pulls the readers no longer take: older bulks and what they supersede."""
    current = set(raw_files(vendor_root, code))
    directory = raw_table_dir(vendor_root, code)
    for prefix in (BULK_PREFIX, WINDOW_PREFIX, UPDATED_PREFIX):
        for path in directory.glob(f"{prefix}*.parquet"):
            if path not in current:
                path.unlink()


def fold_tickers_history(vendor_root: str | Path) -> Path | None:
    """Fold every TICKERS snapshot into the TICKERS history, and return its path.

    The history holds one row per table, permaticker and ticker ever seen in
    a snapshot: TICKERS' columns as of the latest snapshot holding it, plus
    ``first_seen`` and ``last_seen``, the pull times (UTC) of the first and
    last. Folding is idempotent and keeps what earlier folds saw, so a
    snapshot pruned after it is folded (``prune_tickers_snapshots``) loses
    nothing. Every bulk pull folds before it prunes. Read back by
    ``quantlab.dataset.sharadar.tables.tickers_history``.

    Parameters
    ----------
    vendor_root : str or Path
        ``<download-dir>/sharadar``.

    Returns
    -------
    Path or None
        The history (``tickers_history_file``), or ``None`` when there is
        no snapshot and no history, and nothing was written.

    Examples
    --------
    Needs the Sharadar raw tier::

        fold_tickers_history("/data/downloads/sharadar")
        # PosixPath('/data/downloads/sharadar/tickers/_history.parquet')
    """
    path = tickers_history_file(vendor_root)
    schema = TABLES["tickers"].schema
    frames = [] if not path.exists() else [pl.read_parquet(path)]
    for snapshot in tickers_snapshots(vendor_root):
        seen = pull_time(snapshot)
        frames.append(
            pl.read_parquet(snapshot)
            .select([pl.col(name).cast(dtype) for name, dtype in schema.items()])
            .with_columns(pl.lit(seen).alias("first_seen"), pl.lit(seen).alias("last_seen"))
        )
    if not frames:
        return None
    rows = pl.concat(frames, how="vertical_relaxed").drop_nulls(_HISTORY_KEY)
    folded = (
        rows.sort("last_seen")
        .group_by(_HISTORY_KEY, maintain_order=True)
        .agg(
            *(pl.col(name).last() for name in schema if name not in _HISTORY_KEY),
            pl.col("first_seen").min(),
            pl.col("last_seen").max(),
        )
        .select(*schema, "first_seen", "last_seen")
        .sort(_HISTORY_KEY)
    )
    write_parquet_atomically(path, folded)
    return path


def prune_tickers_snapshots(vendor_root: str | Path) -> list[Path]:
    """Delete the TICKERS snapshots no raw file maps with, and return them.

    The newest snapshot is always kept (the next pulls of its run map with
    it); an older one is kept while a raw file of any table pairs with it
    (``snapshot_for``). A bulk pull deletes its table's windows, so the
    snapshots only they used go too. Fold the snapshots into the TICKERS
    history first (``fold_tickers_history``), as a bulk pull does, so a
    deleted one loses nothing.

    Examples
    --------
    Needs the Sharadar raw tier::

        prune_tickers_snapshots("/data/downloads/sharadar")
        # [PosixPath('/data/downloads/sharadar/tickers/snapshot_20240111T082900000000.parquet')]
    """
    snapshots = tickers_snapshots(vendor_root)
    if len(snapshots) < 2:
        return []
    used = {snapshots[-1]}
    for spec in TABLES.values():
        if spec.code == "tickers" or "ticker" not in spec.schema:
            continue
        for path in raw_files(vendor_root, spec.code):
            paired = snapshot_for(vendor_root, pull_time(path))
            if paired is not None:
                used.add(paired)
    removed = [path for path in snapshots if path not in used]
    for path in removed:
        path.unlink()
    return removed
