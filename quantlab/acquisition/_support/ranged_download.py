"""One resumable download of a file in parallel byte ranges.

A vendor's big file (Sharadar's bulk zip of a table, Massive's day file of
trades) is fetched as ranges of ``part_bytes`` by ``workers`` threads at once,
each written at its offset of ``<name>.part``. The vendor supplies only how
one range is fetched: ``fetch_range(start, stop)`` returns the bytes
``start`` up to ``stop`` (exclusive) as an iterable of chunks, from an HTTP
``Range`` GET or an S3 ranged ``get_object``. A ``TransientError`` raised
while the chunks are read (a dropped stream), or a body that ends short, is
the range breaking:

* a break after new bytes resumes at once from the first byte still
  missing, and the range's retry budget starts again, because the vendors'
  storage drops long streams and a range may need many resumed requests;
* a break that delivered nothing waits the policy's back-off (the error's
  ``retry_after``, else ``backoff * 2 ** attempt``) and is retried, at most
  ``policy.retries`` times, after which the last error is raised.

Any other error from ``fetch_range`` (a refusal, an unexpected status) is
raised at once. A range that fails, an ``abort`` event that is set, or an
interrupt stops every other range at its next chunk. The ``.part`` file is
kept with a sidecar ``<name>.part.json`` recording the file's size, the
vendor's ``identity`` of the object (an S3 ETag) and the ranges already
complete, so the next call fetches only the missing ranges; a sidecar whose
size, identity or ``part_bytes`` differs (the object changed), one that is not
the shape written here or names anything but this download's range starts,
and any call without an ``identity`` (a same-sized replacement could not be
told apart) start the file over. The file appears at ``path`` only once every range is in, its size is
checked and ``verify`` (when given) accepted it, by one atomic rename; a file
``verify`` refuses is removed with its sidecar.

Examples
--------
>>> import tempfile
>>> from pathlib import Path
>>> from quantlab.acquisition._support.request_policy import RequestPolicy
>>> body = bytes(range(256)) * 4
>>> target = Path(tempfile.mkdtemp()) / "file.bin"
>>> download_ranges(target, len(body), lambda start, stop: [body[start:stop]],
...                 workers=4, part_bytes=300, policy=RequestPolicy({}), what="example")
4
>>> target.read_bytes() == body
True
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Callable, Iterable
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path

from joblib import Parallel, delayed
from loguru import logger

from quantlab.acquisition._support.request_policy import RequestPolicy, TransientError
from quantlab.utils.atomic import write_json_atomically

#: ``fetch_range(start, stop)``: the bytes ``start`` up to ``stop`` (exclusive), in chunks.
FetchRange = Callable[[int, int], Iterable[bytes]]


class DownloadAborted(Exception):
    """The download was stopped: another range failed, or the caller set ``abort``."""


def download_ranges(
    path: str | Path,
    size: int,
    fetch_range: FetchRange,
    *,
    workers: int,
    part_bytes: int,
    policy: RequestPolicy,
    what: str,
    error: type[TransientError] = TransientError,
    verify: Callable[[Path], None] | None = None,
    fetch_lock: AbstractContextManager | None = None,
    abort: threading.Event | None = None,
    identity: str | None = None,
) -> int:
    """Download ``size`` bytes into ``path`` in parallel, resumable byte ranges.

    Parameters
    ----------
    path : str or Path
        Where the file lands; its directory must exist. The bytes are
        written to ``<path>.part`` first: a fresh start truncates it to
        ``size``; a sidecar matching this call keeps it and its completed
        ranges, and only the others are fetched.
    size : int
        The file's size in bytes, as the vendor gives it.
    fetch_range : callable
        ``fetch_range(start, stop)`` returns the bytes ``start`` up to
        ``stop`` (exclusive) as an iterable of chunks; extra bytes are cut.
        A ``TransientError`` while the chunks are read is the range
        breaking; it is resumed (see the module docstring).
    workers : int
        Ranges fetched at once (joblib threads).
    part_bytes : int
        Size of one range.
    policy : RequestPolicy
        Gives each range's retry budget (``retries``), its back-off and the
        scrubbing of every logged message.
    what : str
        Names the file in messages (never a URL with its query).
    error : type, default TransientError
        Raised (after the retries) for a body that ends short; the vendor's
        own ``TransientError`` subclass.
    verify : callable or None, default None
        Called with the whole ``.part`` file before the rename; whatever it
        raises propagates, and the ``.part`` is removed.
    fetch_lock : context manager or None, default None
        Held while the ranges are fetched, not while verifying: a caller
        fetching one file at a time verifies one while the next is fetched.
    abort : threading.Event or None, default None
        Set by the caller to stop every range at its next chunk.
    identity : str or None, default None
        The vendor's identity of the remote object (an S3 ETag), recorded in
        the sidecar: a ``.part`` left for another identity is started over,
        and without one (``None``) every call starts over.

    Returns
    -------
    int
        The number of ranges.

    Raises
    ------
    TransientError
        The last break of a range that delivered nothing ``policy.retries``
        times running.
    DownloadAborted
        If ``abort`` was set.
    ValueError
        If ``workers`` or ``part_bytes`` is below 1.

    Examples
    --------
    See the module docstring.
    """
    if workers < 1:
        raise ValueError(f"download_ranges needs workers of at least 1; got {workers}.")
    if part_bytes < 1:
        raise ValueError(f"download_ranges needs part_bytes of at least 1; got {part_bytes}.")
    target = Path(path)
    partial = target.with_name(target.name + ".part")
    parts = [(first, min(first + part_bytes, size)) for first in range(0, size, part_bytes)]
    resume = _ResumeState.load(
        partial, {"size": size, "identity": identity, "part_bytes": part_bytes}, {first for first, _ in parts}
    )
    pending = [(first, stop) for first, stop in parts if first not in resume.done]
    if resume.done:
        logger.info(f"{what}: resuming, {len(parts) - len(pending)} of {len(parts)} range(s) already complete.")
    fanout = _RangeFanout(partial, fetch_range, policy, what, error, abort, resume)
    verifying = False
    try:
        if not resume.done:
            resume.start_over()
        with fetch_lock if fetch_lock is not None else nullcontext():
            fanout.run(pending, workers)
        verifying = True
        _land(partial, target, size, what, error, verify)
        resume.sidecar.unlink(missing_ok=True)
    except BaseException as raised:
        fanout.stop()
        if verifying:
            # A whole file that is wrong is not resumed: the next call starts over.
            partial.unlink(missing_ok=True)
            resume.sidecar.unlink(missing_ok=True)
        # The range that failed first, not one it stopped.
        if isinstance(raised, DownloadAborted) and fanout.failures:
            raise fanout.failures[0] from None
        raise
    return len(parts)


class _ResumeState:
    """The ``.part`` file's sidecar: this download's key and the ranges already complete.

    ``done`` holds the first bytes of the complete ranges; ``record`` adds
    one and rewrites the sidecar under a lock, so ranges finishing at once
    on different threads each land in it.
    """

    def __init__(self, partial: Path, key: dict, done: set[int]):
        """Hold the sidecar beside ``partial``, written for ``key``, with ``done`` complete."""
        self.partial = partial
        self.sidecar = Path(f"{partial}.json")
        self.key = key
        self.done = done
        self._lock = threading.Lock()

    @classmethod
    def load(cls, partial: Path, key: dict, starts: set[int]) -> _ResumeState:
        """The state a sidecar left for ``key``; nothing complete when there is none to resume."""
        return cls(partial, key, _completed(partial, Path(f"{partial}.json"), key, starts))

    def start_over(self) -> None:
        """Truncate the ``.part`` file to the download's size and record no range complete."""
        with self.partial.open("wb") as handle:
            handle.truncate(self.key["size"])
        _write_sidecar(self.sidecar, self.key, self.done)

    def record(self, first: int) -> None:
        """Mark the range starting at byte ``first`` complete in the sidecar."""
        with self._lock:
            self.done.add(first)
            _write_sidecar(self.sidecar, self.key, self.done)


class _RangeFanout:
    """Fetches ranges on joblib threads; the first failure stops the others at their next chunk.

    ``failures`` keeps every failure but the ``DownloadAborted`` of a range
    that was stopped, in the order they happened.
    """

    def __init__(
        self,
        partial: Path,
        fetch_range: FetchRange,
        policy: RequestPolicy,
        what: str,
        error: type[TransientError],
        abort: threading.Event | None,
        resume: _ResumeState,
    ):
        """Write into ``partial`` through ``fetch_range``, recording each finished range in ``resume``."""
        self.partial = partial
        self.fetch_range = fetch_range
        self.policy = policy
        self.what = what
        self.error = error
        self.abort = abort
        self.resume = resume
        # Set when a range fails or the fan-out is interrupted: the others stop.
        self._stopped = threading.Event()
        self.failures: list[BaseException] = []

    def stop(self) -> None:
        """Stop every range still running at its next chunk."""
        self._stopped.set()

    def aborted(self) -> bool:
        """Whether a range failed, the fan-out was stopped or the caller set ``abort``."""
        return self._stopped.is_set() or (self.abort is not None and self.abort.is_set())

    def run(self, pending: list[tuple[int, int]], workers: int) -> None:
        """Fetch the ``(first, stop)`` ranges in ``pending``, ``workers`` at once."""
        if pending:
            # Threads, as every fan-out in quantlab: the ranges wait on the network.
            Parallel(n_jobs=min(workers, len(pending)), backend="threading", batch_size=1)(
                delayed(self._fetch_and_record)(first, stop) for first, stop in pending
            )

    def _fetch_and_record(self, first: int, stop: int) -> None:
        """Fetch one range and record it complete; a failure stops the others."""
        try:
            _fetch_one(
                self.partial, first, stop, self.fetch_range, self.policy, self.what, self.error, self.aborted
            )
        except BaseException as failure:
            if not isinstance(failure, DownloadAborted):
                self.failures.append(failure)
            self._stopped.set()
            raise
        self.resume.record(first)


def _land(
    partial: Path,
    target: Path,
    size: int,
    what: str,
    error: type[TransientError],
    verify: Callable[[Path], None] | None,
) -> None:
    """Check the whole ``partial`` file's size, run ``verify`` on it and rename it to ``target``."""
    if partial.stat().st_size != size:
        raise error(f"{what}: the downloaded file holds {partial.stat().st_size} bytes, not {size}")
    if verify is not None:
        verify(partial)
    os.replace(partial, target)


def _completed(partial: Path, sidecar: Path, key: dict, starts: set[int]) -> set[int]:
    """The first bytes of the ranges already complete in ``partial``; empty to start over.

    Anything but a sidecar written for ``key`` whose ``done`` lists only
    ``starts`` (this download's range starts) is no resume state.
    """
    if key["identity"] is None or not (partial.exists() and sidecar.exists()):
        return set()
    try:
        state = json.loads(sidecar.read_text())
    except (OSError, ValueError):
        return set()
    if not isinstance(state, dict) or {name: state.get(name) for name in key} != key:
        return set()
    done = state.get("done")
    # bool is an int to Python; JSON true is no range start.
    if not isinstance(done, list) or not all(type(first) is int and first in starts for first in done):
        return set()
    if partial.stat().st_size != key["size"]:
        return set()
    return set(done)


def _write_sidecar(sidecar: Path, key: dict, done: set[int]) -> None:
    """Record ``key`` and the completed ranges' first bytes, replacing the sidecar atomically.

    Through ``write_json_atomically``, whose scratch file is created afresh
    (``mkstemp``: exclusive, no link followed), so nothing planted in the
    directory is written through.
    """
    write_json_atomically(sidecar, {**key, "done": sorted(done)})


def _fetch_one(
    partial: Path,
    first: int,
    stop: int,
    fetch_range: FetchRange,
    policy: RequestPolicy,
    what: str,
    error: type[TransientError],
    aborted: Callable[[], bool],
) -> None:
    """Write bytes ``first`` up to ``stop`` at their offset of ``partial``, resuming breaks."""
    position = first
    attempt = 0
    with partial.open("r+b") as handle:
        while position < stop:
            if aborted():
                raise DownloadAborted(what)
            before = position
            try:
                handle.seek(position)
                chunks = iter(fetch_range(position, stop))
                try:
                    for chunk in chunks:
                        if aborted():
                            raise DownloadAborted(what)
                        chunk = chunk[: stop - position]
                        handle.write(chunk)
                        position += len(chunk)
                        if position >= stop:
                            break
                finally:
                    # Release the connection of a body left unread.
                    close = getattr(chunks, "close", None)
                    if close is not None:
                        close()
                if position < stop:
                    raise error(
                        f"{what}: bytes {first}-{stop - 1} ended after {position - first} of "
                        f"{stop - first} bytes"
                    )
            except TransientError as broken:
                if position > before:
                    logger.warning(
                        policy.scrub(f"{what}: {broken}; resuming at byte {position} of {first}-{stop - 1}.")
                    )
                    attempt = 0
                    continue
                if attempt >= policy.retries or aborted():
                    raise
                policy.back_off(
                    attempt,
                    lambda wait, broken=broken, attempt=attempt, position=position: (
                        f"{what}: {broken}; retry {attempt + 1}/{policy.retries} of bytes "
                        f"{position}-{stop - 1} in {wait:g}s."
                    ),
                    broken.retry_after,
                )
                attempt += 1
