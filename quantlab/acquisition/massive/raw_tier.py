"""The writers of Massive's raw tier: REST pages, whole pulls, the condition table and watermarks.

``MassiveClient`` (``quantlab.acquisition.massive.client``) downloads day
files, and the pull shapes (``quantlab.acquisition.massive.shapes``) pull REST
endpoints, into ``<download-dir>/massive/`` and record them with these writers.
The layout (``raw_file``, ``rest_page_file``, ``rest_pending_dir``, ...) and
every reader of it (``read_watermark``,
``read_rest_watermark``, ``read_rest_records``, ``rest_frame``, ...) stay in
``quantlab.dataset.massive.raw``: the dataset layer reads the raw tier, only
the acquisition layer writes it.

Both watermarks are dated watermarks (``quantlab.utils.watermark``) labelled
here: a day-file type's (``<massive>/<type>/_watermark.json``) holds
``data_type`` and ``through``, a REST endpoint's
(``<massive>/rest/<endpoint>/_watermark.json``) ``endpoint``, ``field`` and
``through``; ``field`` is ``"snapshot"`` for a snapshot endpoint. The client
resumes either the day after it (``quantlab.utils.watermark.resume_from``)
and never moves a REST watermark back.

A snapshot or as-of pull appears in one step: ``stage_snapshot`` and
``stage_as_of`` keep its pages in the endpoint's pending directory and
``StagedPull.commit`` publishes them with one directory rename, so a reader
sees the old whole data or the new, after a process kill or a power loss. The
next pull of the endpoint clears what an interrupted one left. One pull of an
endpoint runs at a time.

Examples
--------
>>> import tempfile
>>> from datetime import date
>>> from quantlab.dataset.massive.raw import read_rest_watermark
>>> root = tempfile.mkdtemp()
>>> write_rest_watermark(root, "splits", "execution_date", date(2024, 3, 14))
>>> read_rest_watermark(root, "splits")
datetime.date(2024, 3, 14)
"""

from __future__ import annotations

import shutil
from datetime import date, datetime
from pathlib import Path

from quantlab.dataset.massive.raw import (
    conditions_file,
    rest_as_of_pull_dir,
    rest_page_file,
    rest_pending_dir,
    rest_pull_page_name,
    rest_pull_pages,
    rest_snapshot_dir,
    rest_watermark_file,
    watermark_file,
)
from quantlab.utils.atomic import (
    make_directories_durably,
    sync_directory,
    write_json_atomically,
    write_jsonl_atomically,
)
from quantlab.utils.watermark import write_watermark


def write_day_file_watermark(vendor_root: str | Path, data_type: str, through: date) -> None:
    """Record that a data type's downloads are complete through ``through``.

    Read back by ``quantlab.dataset.massive.raw.read_watermark``.

    Examples
    --------
    Writes ``/data/quantlab/downloads/massive/trades/_watermark.json``::

        write_day_file_watermark("/data/quantlab/downloads/massive", "trades", date(2024, 11, 29))
    """
    write_watermark(watermark_file(vendor_root, data_type), through, {"data_type": data_type})


def write_rest_page(path: str | Path, records: list[dict]) -> None:
    """Write one page's records, one JSON object per line, atomically.

    Examples
    --------
    Writes one line per record::

        write_rest_page(rest_page_file(root, "splits", pulled_at, 1), page["results"])
    """
    write_jsonl_atomically(path, records)


class PageFiles:
    """The page files of one pull of an endpoint kept by date, each landing as it is written.

    A reader may see the first pages of such a pull: the watermark, written
    after them, is what says how far it reached.

    Parameters
    ----------
    vendor_root : str or Path
        ``<download-dir>/massive``.
    endpoint : str
        The endpoint's name.
    pulled_at : datetime
        When the pull started; it names the pages.

    Examples
    --------
    >>> import tempfile
    >>> from datetime import UTC
    >>> pages = PageFiles(tempfile.mkdtemp(), "splits", datetime(2026, 10, 16, tzinfo=UTC))
    >>> pages.add_page([{"id": "S1"}])
    >>> len(pages.written)
    1
    """

    def __init__(self, vendor_root: str | Path, endpoint: str, pulled_at: datetime):
        self._vendor_root = vendor_root
        self._endpoint = endpoint
        self._pulled_at = pulled_at
        self._written: list[Path] = []

    def add_page(self, records: list[dict]) -> None:
        """Write one page's records, the next in the pull.

        Examples
        --------
        >>> import tempfile
        >>> from datetime import UTC
        >>> PageFiles(tempfile.mkdtemp(), "splits", datetime(2026, 10, 16, tzinfo=UTC)).add_page([{"id": "S1"}])
        """
        path = rest_page_file(self._vendor_root, self._endpoint, self._pulled_at, len(self._written) + 1)
        write_rest_page(path, records)
        self._written.append(path)

    @property
    def written(self) -> tuple[Path, ...]:
        """The page files written so far, in order.

        Examples
        --------
        >>> import tempfile
        >>> from datetime import UTC
        >>> PageFiles(tempfile.mkdtemp(), "splits", datetime(2026, 10, 16, tzinfo=UTC)).written
        ()
        """
        return tuple(self._written)


class StagedPull:
    """The pages of one pull, out of the readers' sight until ``commit``.

    Built by ``stage_snapshot`` or ``stage_as_of``.

    Parameters
    ----------
    pending : Path
        The endpoint's pending directory; what is in it is cleared first.
    final : Path
        The directory the pull is published as.

    Examples
    --------
    Publishes ``/d/massive/rest/float/2026/20261016T120000000000/``::

        staged = stage_snapshot("/d/massive", "float", pulled_at)
        staged.add_page(page["results"])
        staged.commit()
    """

    def __init__(self, pending: Path, final: Path):
        self._pending = pending
        self._staged = pending / final.name
        self._final = final
        self._pages = 0
        shutil.rmtree(pending, ignore_errors=True)  # what an interrupted pull left

    def add_page(self, records: list[dict]) -> None:
        """Keep one page's records, the next in the pull.

        Examples
        --------
        >>> import tempfile
        >>> from datetime import UTC
        >>> staged = stage_snapshot(tempfile.mkdtemp(), "float", datetime(2026, 10, 16, tzinfo=UTC))
        >>> staged.add_page([{"ticker": "AAPL"}])
        """
        self._pages += 1
        write_rest_page(self._staged / rest_pull_page_name(self._pages), records)

    def commit(self) -> tuple[Path, ...]:
        """Publish the pull with one rename and return its page files in order.

        A pull with no pages publishes nothing. Once this returns, the pull
        survives a power loss, so a watermark may follow it.

        Returns
        -------
        tuple of Path
            The published page files.

        Raises
        ------
        OSError
            If the pull cannot be published; the readers see no part of it.

        Examples
        --------
        >>> import tempfile
        >>> from datetime import UTC
        >>> staged = stage_snapshot(tempfile.mkdtemp(), "float", datetime(2026, 10, 16, tzinfo=UTC))
        >>> staged.add_page([{"ticker": "AAPL"}])
        >>> [path.name for path in staged.commit()]
        ['000001.jsonl']
        """
        if self._pages:
            make_directories_durably(self._final.parent)
            self._staged.replace(self._final)
            sync_directory(self._final.parent)
        shutil.rmtree(self._pending, ignore_errors=True)
        return tuple(rest_pull_pages(self._final))


def stage_snapshot(vendor_root: str | Path, endpoint: str, pulled_at: datetime) -> StagedPull:
    """Start a snapshot pull of ``endpoint``, published by ``StagedPull.commit``.

    Parameters
    ----------
    vendor_root : str or Path
        ``<download-dir>/massive``.
    endpoint : str
        The endpoint's name.
    pulled_at : datetime
        When the pull started; it names the snapshot.

    Returns
    -------
    StagedPull
        The pull to add pages to; what an interrupted pull left is gone.

    Examples
    --------
    >>> import tempfile
    >>> from datetime import UTC
    >>> from quantlab.dataset.massive.raw import read_rest_snapshot
    >>> root = tempfile.mkdtemp()
    >>> staged = stage_snapshot(root, "float", datetime(2026, 10, 16, 12, tzinfo=UTC))
    >>> staged.add_page([{"ticker": "AAPL"}])
    >>> _ = staged.commit()
    >>> read_rest_snapshot(root, "float")
    [{'ticker': 'AAPL'}]
    """
    return StagedPull(rest_pending_dir(vendor_root, endpoint), rest_snapshot_dir(vendor_root, endpoint, pulled_at))


def stage_as_of(vendor_root: str | Path, endpoint: str, day: date, pulled_at: datetime) -> StagedPull:
    """Start a pull of ``endpoint`` as of ``day``, published by ``StagedPull.commit``.

    Parameters
    ----------
    vendor_root : str or Path
        ``<download-dir>/massive``.
    endpoint : str
        The endpoint's name.
    day : date
        The session the table is as of.
    pulled_at : datetime
        When the pull started; it names the pull.

    Returns
    -------
    StagedPull
        The pull to add pages to; what an interrupted pull left is gone.

    Examples
    --------
    >>> import tempfile
    >>> from datetime import UTC
    >>> from quantlab.dataset.massive.raw import read_rest_as_of
    >>> root = tempfile.mkdtemp()
    >>> staged = stage_as_of(root, "tickers", date(2024, 3, 11), datetime(2024, 3, 12, tzinfo=UTC))
    >>> staged.add_page([{"ticker": "AAPL"}])
    >>> _ = staged.commit()
    >>> read_rest_as_of(root, "tickers", date(2024, 3, 11))
    [{'ticker': 'AAPL'}]
    """
    final = rest_as_of_pull_dir(vendor_root, endpoint, day, pulled_at)
    return StagedPull(rest_pending_dir(vendor_root, endpoint), final)


def write_conditions(vendor_root: str | Path, pulled_at: datetime, records: list[dict]) -> Path:
    """Write the condition table, the vendor's records verbatim, as one file.

    Parameters
    ----------
    vendor_root : str or Path
        ``<download-dir>/massive``.
    pulled_at : datetime
        When the pull started; it names the file.
    records : list of dict
        Every record of the table.

    Returns
    -------
    Path
        The file (``quantlab.dataset.massive.raw.conditions_file``).

    Examples
    --------
    >>> import tempfile
    >>> from datetime import UTC
    >>> write_conditions(tempfile.mkdtemp(), datetime(2026, 10, 9, 12, tzinfo=UTC), [{"id": 1}]).name
    'conditions_20261009T120000000000.json'
    """
    path = conditions_file(vendor_root, pulled_at)
    write_json_atomically(path, records)
    return path


def write_rest_watermark(vendor_root: str | Path, endpoint: str, field: str, through: date) -> None:
    """Record that a REST endpoint's rows are pulled through ``through`` on ``field``.

    Read back by ``quantlab.dataset.massive.raw.read_rest_watermark``.

    Examples
    --------
    Writes ``/data/quantlab/downloads/massive/rest/splits/_watermark.json``::

        write_rest_watermark("/data/quantlab/downloads/massive", "splits", "execution_date", date(2024, 3, 14))
    """
    write_watermark(rest_watermark_file(vendor_root, endpoint), through, {"endpoint": endpoint, "field": field})
