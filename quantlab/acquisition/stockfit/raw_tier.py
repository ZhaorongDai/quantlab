"""The writer of StockFit's raw tier: one pull staged, then published by one rename or dropped.

The layout and every reader of it stay in ``quantlab.dataset.stockfit.raw``:
the dataset layer reads the raw tier, only the acquisition layer writes it,
and this module takes every path from there.

A pull of one endpoint for one entity is staged page by page in the entity's
pending directory (``StagedPull``). ``commit`` then compares the content hash
of the staged pages with the entity's newest published pull: equal, the
staged pull is dropped; different (or no pull yet), it is published with one
directory rename, so a reader sees the old whole pull or the new one, after a
process kill or a power loss. Either way the entity's watermark then moves to
the pull's UTC day. The next pull of the entity clears what an interrupted one
left. One pull of an entity runs at a time.

Examples
--------
>>> import tempfile
>>> from datetime import UTC, datetime
>>> from quantlab.dataset.stockfit.raw import read_pull
>>> root = tempfile.mkdtemp()
>>> staged = StagedPull(root, "symbols", "operating", datetime(2026, 11, 16, tzinfo=UTC), records="data")
>>> staged.add_page({"page": 1, "totalPages": 1, "data": [{"cik": 1093691}]})
>>> staged.commit().published
True
>>> read_pull(root, "symbols", "operating")
[{'cik': 1093691}]
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

from quantlab.dataset.stockfit.raw import (
    MANIFEST_FILE,
    endpoint_watermark_file,
    newest_pull_dir,
    page_name,
    pending_dir,
    pull_dir,
    pull_stamp,
    read_manifest,
    watermark_file,
)
from quantlab.utils.atomic import (
    make_directories_durably,
    sync_directory,
    write_json_atomically,
)
from quantlab.utils.watermark import write_watermark


@dataclass(frozen=True)
class Committed:
    """What ``StagedPull.commit`` did: the pull's directory, and whether it was published.

    ``published`` is False when the pull equalled the newest published one
    and was dropped; ``directory`` is then that newest pull's.
    """

    directory: Path
    published: bool


class StagedPull:
    """The pages of one pull of an entity, out of the readers' sight until ``commit``.

    Parameters
    ----------
    vendor_root : str or Path
        ``<download-dir>/stockfit``.
    endpoint, entity : str
        The endpoint's and the entity's directory names.
    pulled_at : datetime
        When the pull started; it names the pull and dates the watermark.
    records : str or None, default None
        The key a paged body keeps its records under (``"data"``), written to
        the manifest so the readers join the pages; ``None`` for a one-page
        body read whole.
    query : dict or None, default None
        The query parameters of the pull, recorded in the manifest. Never
        holds a credential: StockFit takes its key in a header.

    Examples
    --------
    See the module docstring.
    """

    def __init__(
        self,
        vendor_root: str | Path,
        endpoint: str,
        entity: str,
        pulled_at: datetime,
        *,
        records: str | None = None,
        query: dict | None = None,
    ):
        self._root = Path(vendor_root)
        self._endpoint = endpoint
        self._entity = entity
        self._pulled_at = pulled_at
        self._records = records
        self._query = dict(query or {})
        self._pending = pending_dir(vendor_root, endpoint, entity)
        self._final = pull_dir(vendor_root, endpoint, entity, pulled_at)
        self._staged = self._pending / self._final.name
        self._hash = hashlib.sha256()
        self._pages = 0
        shutil.rmtree(self._pending, ignore_errors=True)  # what an interrupted pull left

    def add_page(self, body: object) -> None:
        """Keep one page's response body, the next in the pull.

        Examples
        --------
        See the module docstring.
        """
        self._pages += 1
        canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
        self._hash.update(canonical.encode())
        self._hash.update(b"\n")
        write_json_atomically(self._staged / page_name(self._pages), body)

    def commit(self) -> Committed:
        """Publish the pull with one rename, or drop it if it equals the newest pull; then move the watermark.

        Returns
        -------
        Committed
            The published pull's directory, or the newest pull's when this
            one was dropped as unchanged.

        Raises
        ------
        ValueError
            If no page was added: an empty pull is not a pull.
        OSError
            If the pull cannot be published; the readers see no part of it,
            and the watermark has not moved.

        Examples
        --------
        See the module docstring.
        """
        if not self._pages:
            raise ValueError(f"StockFit {self._endpoint}/{self._entity}: a pull needs at least one page.")
        digest = self._hash.hexdigest()
        newest = newest_pull_dir(self._root, self._endpoint, self._entity)
        if newest is not None and read_manifest(newest).get("sha256") == digest:
            shutil.rmtree(self._pending, ignore_errors=True)
            committed = Committed(newest, False)
        else:
            write_json_atomically(self._staged / MANIFEST_FILE, {
                "endpoint": self._endpoint,
                "entity": self._entity,
                "pulled_at": pull_stamp(self._pulled_at),
                "query": self._query,
                "pages": self._pages,
                "records": self._records,
                "sha256": digest,
            }, indent=1, sort_keys=True)
            make_directories_durably(self._final.parent)
            self._staged.replace(self._final)
            sync_directory(self._final.parent)
            shutil.rmtree(self._pending, ignore_errors=True)
            committed = Committed(self._final, True)
        write_entity_watermark(self._root, self._endpoint, self._entity, self._pulled_at)
        return committed


def write_entity_watermark(vendor_root: str | Path, endpoint: str, entity: str, pulled_at: datetime) -> None:
    """Record that an entity was pulled at ``pulled_at``, whether or not a pull was published.

    Read back by ``quantlab.dataset.stockfit.raw.read_watermark`` (the UTC
    day); the file also holds the pull's stamp.

    Examples
    --------
    >>> import tempfile
    >>> from datetime import UTC
    >>> from quantlab.dataset.stockfit.raw import read_watermark
    >>> root = tempfile.mkdtemp()
    >>> write_entity_watermark(root, "symbols", "operating", datetime(2026, 11, 16, 14, tzinfo=UTC))
    >>> read_watermark(root, "symbols", "operating")
    datetime.date(2026, 11, 16)
    """
    write_watermark(
        watermark_file(vendor_root, endpoint, entity),
        pulled_at.astimezone(UTC).date(),
        {"endpoint": endpoint, "entity": entity, "pulled_at": pull_stamp(pulled_at)},
    )


def write_endpoint_watermark(vendor_root: str | Path, endpoint: str, through: date) -> None:
    """Record that a refresh on ``through`` (UTC) found none of the endpoint's entities due.

    Read back through ``quantlab.dataset.stockfit.raw.newest_watermark``.

    Examples
    --------
    >>> import tempfile
    >>> from quantlab.dataset.stockfit.raw import newest_watermark
    >>> root = tempfile.mkdtemp()
    >>> write_endpoint_watermark(root, "income_statement_quarter", date(2026, 11, 17))
    >>> newest_watermark(root, "income_statement_quarter")
    datetime.date(2026, 11, 17)
    """
    write_watermark(endpoint_watermark_file(vendor_root, endpoint), through, {"endpoint": endpoint})
