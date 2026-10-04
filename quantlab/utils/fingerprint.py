"""Content fingerprint of the data a backtest run read.

A stored backtest is reproducible only if the data under it has not changed,
and datasets do change: stores are appended to and adjusted prices are
re-based retroactively. ``dataset_fingerprint`` records, for one dataset, the
time range, the axis sizes and a sha256 digest over the values of the
variables a run consumed. A rebuild from a persisted config computes the same
record and compares it against the stored one.

NaN has many bit patterns, so every NaN is rewritten to one canonical NaN and
every ``-0.0`` to ``0.0`` before hashing; otherwise two reads of identical data
could report different digests.

A ``DataRecorder`` is the context a run opens to learn what it read. The two
read seams, ``BaseDataset.panel`` and ``Factor.read``, call ``record_read``;
inside an open recorder the read is logged as a request (dataset, start, end,
symbols, variables), outside one it returns at once. When the recorder closes,
each distinct request is read again once and hashed with
``dataset_fingerprint``, and the records are compared with an expected record
when one is given. This module has no project-internal imports.
"""

import contextvars
import hashlib
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager

import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger

__all__ = [
    "PARTIAL_NOTE",
    "DataRecorder",
    "active_recorder",
    "dataset_fingerprint",
    "record_read",
    "unrecorded",
]

#: Tail of every warning of a failure-path comparison, in place of the usual
#: "continuing". Log readers and tests find partial comparisons by the
#: substring ``"comparison is PARTIAL"``, so keep it when rewording.
PARTIAL_NOTE = (
    "this comparison is PARTIAL: the run failed before it finished reading, so "
    "a differing digest may reflect the interrupted read rather than a data "
    "change; the original error follows"
)

#: The innermost open recorder, or ``None`` outside every recorder (and while
#: a closing recorder re-reads its requests).
_ACTIVE: contextvars.ContextVar["DataRecorder | None"] = contextvars.ContextVar(
    "quantlab_data_recorder", default=None
)


def _iso(value) -> str:
    """Return ``value`` as an ISO-8601 timestamp string."""
    return pd.Timestamp(value).isoformat()


def dataset_fingerprint(ds: xr.Dataset, variables: list[str]) -> dict:
    """Fingerprint ``variables`` of a ``(timestamp, symbol)`` dataset.

    The dataset is sorted by timestamp and symbol first, so the digest does not
    depend on axis order. The hash covers, in order: the int64 nanosecond
    timestamps, the NUL-joined symbol names, then for each variable in sorted
    order its name and its float64 values on ``(timestamp, symbol)`` after NaN
    and signed-zero canonicalisation.

    Parameters
    ----------
    ds : xr.Dataset
        A panel indexed by ``timestamp`` and ``symbol``.
    variables : list[str]
        Names of the data variables to include.

    Returns
    -------
    dict
        A dict with keys ``algorithm`` (``"sha256"``), ``digest``, ``variables``
        (sorted), ``start`` and ``end`` (ISO strings, ``None`` when the
        timestamp axis is empty), ``n_timestamps`` and ``n_symbols``.

    Raises
    ------
    ValueError
        If any requested variable is not in ``ds``.

    Examples
    --------
    >>> record = dataset_fingerprint(prices, ["open", "close"])
    >>> record["digest"] == stored_record["digest"]
    True
    """
    names = sorted(variables)
    missing = [name for name in names if name not in ds.data_vars]
    if missing:
        raise ValueError(
            f"cannot fingerprint variables {missing}: not in the dataset "
            f"(data variables: {sorted(ds.data_vars)})"
        )

    ds = ds.sortby(["timestamp", "symbol"])
    timestamps = ds["timestamp"].values.astype("datetime64[ns]")
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(timestamps.astype("int64")).tobytes())
    digest.update("\x00".join(map(str, ds["symbol"].values.tolist())).encode())
    for name in names:
        values = np.asarray(
            ds[name].transpose("timestamp", "symbol").values, dtype=np.float64
        )
        values = np.where(np.isnan(values), np.nan, values)  # one NaN bit pattern
        values = values + 0.0  # -0.0 becomes 0.0
        digest.update(name.encode())
        digest.update(np.ascontiguousarray(values).tobytes())

    return {
        "algorithm": "sha256",
        "digest": digest.hexdigest(),
        "variables": names,
        "start": _iso(timestamps[0]) if timestamps.size else None,
        "end": _iso(timestamps[-1]) if timestamps.size else None,
        "n_timestamps": int(ds.sizes["timestamp"]),
        "n_symbols": int(ds.sizes["symbol"]),
    }


def active_recorder() -> "DataRecorder | None":
    """Return the innermost open ``DataRecorder``, or ``None`` outside every one.

    Examples
    --------
    >>> active_recorder() is None
    True
    >>> with DataRecorder() as recorder:
    ...     active_recorder() is recorder
    True
    """
    return _ACTIVE.get()


@contextmanager
def unrecorded() -> Iterator[None]:
    """Keep the reads inside out of every open recorder, and hash nothing.

    For reads that belong to another record than the run's around them:
    a backtest's training step, whose reads are the trained unit's. A
    recorder opened inside still records.

    Examples
    --------
    >>> with DataRecorder() as recorder:
    ...     with unrecorded():
    ...         _ = prices.panel("2024-01-02", "2024-01-31")
    >>> recorder.records
    {}
    """
    token = _ACTIVE.set(None)
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def record_read(
    source,
    start,
    end,
    *,
    symbols=None,
    variables=None,
    reread: Callable[[], xr.Dataset],
    store: str | None = None,
) -> None:
    """Log a read of ``source`` to the innermost open recorder, if there is one.

    Called by the read seams only (``BaseDataset.panel`` on a leaf dataset,
    ``Factor.read``) after a successful read. Outside a recorder it returns at
    once; nothing is kept and nothing is hashed.

    Parameters
    ----------
    source : BaseDataset or Factor
        The leaf dataset or the factor whose store was read.
    start, end : str, datetime.date or pd.Timestamp
        The requested range, as given to the seam.
    symbols : sequence, optional
        The requested symbols; ``None`` for every symbol.
    variables : sequence of str, optional
        The requested variables; ``None`` for every variable.
    reread : callable
        Returns the same panel again; the recorder calls it once per distinct
        request when it closes, to hash it.
    store : str, optional
        The store actually read, for the fallback key of an unmapped source;
        defaults to ``source.store_path``. A resampled dataset reading its
        source store passes that store.

    Examples
    --------
    A leaf dataset's ``panel`` ends with (``prices`` is a ``StockDataset``)::

        record_read(
            self, start, end, symbols=symbols, variables=variables,
            reread=lambda: self.panel(start, end, symbols, variables),
        )
    """
    recorder = _ACTIVE.get()
    if recorder is not None:
        recorder._log(source, start, end, symbols, variables, reread, store)


def _text(value) -> str:
    """Return a range end as recorded: strings as given, anything else ISO."""
    return value if isinstance(value, str) else pd.Timestamp(value).isoformat()


def _plain(label):
    """Return a symbol label as a plain Python value (numpy ints become int)."""
    return label.item() if hasattr(label, "item") else label


def _describe_request(request: dict) -> str:
    """Return ``start..end`` with the symbols and variables, for warnings."""
    text = f"{request['start']}..{request['end']}"
    if request["symbols"] is not None:
        text += f", {len(request['symbols'])} symbol(s)"
    if request["variables"] is not None:
        text += f", variables {request['variables']}"
    return text


def _extent(entry: dict) -> str:
    """Return the range and sizes an entry covered, the explanation of a warning."""
    return (
        f"{entry.get('start')} to {entry.get('end')}, "
        f"{entry.get('n_timestamps')} timestamps x {entry.get('n_symbols')} symbols"
    )


def _request_id(request: dict) -> tuple:
    """Return the hashable identity of a request."""
    return (
        request["start"],
        request["end"],
        None if request["symbols"] is None else tuple(map(repr, request["symbols"])),
        None if request["variables"] is None else tuple(request["variables"]),
    )


class DataRecorder:
    """Record the data a run reads, hash it once per request, compare on close.

    A run opens one around everything it reads: a trained unit, a backtest
    run, a ``run_cv`` fold. While it is the innermost open recorder, every
    read through ``BaseDataset.panel`` and ``Factor.read`` is logged as a
    request; nested recorders log to the innermost only. Only leaf datasets
    record (a store or a held panel): a ``MergedDataset`` records nothing
    itself, its inputs do.

    When the context closes, each distinct request is read once more and
    hashed with ``dataset_fingerprint`` over the requested variables (every
    variable of the panel when none were requested). ``records`` then maps each
    key to its requests, in the order first read; an in-memory dataset hashes
    the panel it holds. With ``expected`` given, the records are compared with
    it per key and request by ``digest`` alone; a changed, missing or extra
    request or key logs a warning showing the ranges and sizes as explanation,
    and nothing is raised.

    If the body raises, what was read so far is hashed and compared partially:
    a request or key not read yet is not reported, every warning carries
    ``PARTIAL_NOTE``, and the original error propagates. The diagnostic never
    raises and never replaces that error: a failure of the hashing or the
    comparison itself only logs a warning, and ``records`` keeps what was
    hashed before it.

    Parameters
    ----------
    keys : iterable of (object, str) pairs, optional
        The key each dataset or factor is recorded under, matched by identity;
        a source given twice takes its first key. An unmapped source is
        recorded under ``"<ClassName>:<store path>"``, so two unmapped
        in-memory datasets share ``"FrameDataset:None"``; map them.
    expected : mapping, optional
        A ``records`` of an earlier run to compare with on close.
    owner : str
        The name that opens every warning.

    Attributes
    ----------
    records : dict
        ``{key: [entry, ...]}``, filled when the context closes. Each entry is
        a ``dataset_fingerprint`` record plus ``request``: the requested
        ``start``, ``end``, ``symbols`` and ``variables`` (``None`` for all).

    Examples
    --------
    ``prices`` is a ``StockDataset`` over a business-day store from 2024-01-01:

    >>> with DataRecorder(keys=[(prices, "price_dataset")]) as recorder:
    ...     _ = prices.panel("2024-01-02", "2024-01-31", variables=["adjClose"])
    >>> entry = recorder.records["price_dataset"][0]
    >>> entry["request"]["variables"], entry["n_timestamps"]
    (['adjClose'], 22)

    Comparing a rebuilt run whose store was restated (``restated`` holds the
    same symbols and bars, other prices) logs, through loguru::

        with DataRecorder(keys=[(restated, "price_dataset")], expected=recorder.records):
            restated.panel("2024-01-02", "2024-01-31", variables=["adjClose"])

        DataRecorder: data fingerprint mismatch for 'price_dataset', request
        2024-01-02..2024-01-31, variables ['adjClose']: digest differs (expected
        2024-01-02T00:00:00 to 2024-01-31T00:00:00, 22 timestamps x 6 symbols, got
        ...). The data changed since the expected run; continuing
    """

    def __init__(
        self,
        keys: Iterable[tuple[object, str]] = (),
        expected: Mapping | None = None,
        owner: str = "DataRecorder",
    ):
        """Initialize the recorder; see the class docstring for parameters."""
        self._keys: dict[int, str] = {}
        self._held: list = []  # keeps keyed sources alive, so ids stay unique
        for source, key in keys:
            if id(source) not in self._keys:
                self._keys[id(source)] = key
                self._held.append(source)
        self.expected = expected
        self.owner = owner
        self.records: dict[str, list[dict]] = {}
        self._requests: dict[str, dict[tuple, tuple[dict, Callable]]] = {}
        self._token: contextvars.Token | None = None

    def __enter__(self) -> "DataRecorder":
        """Become the innermost open recorder."""
        self._token = _ACTIVE.set(self)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        """Hash the requests and compare them; never swallow the body's error."""
        _ACTIVE.reset(self._token)
        try:
            self._close(partial=exc_type is not None)
        except BaseException as error:  # noqa: BLE001 - never replace the error
            try:
                tail = "continuing" if exc_type is None else "the original error follows"
                logger.warning(
                    f"{self.owner}: the data fingerprint diagnostic itself "
                    f"raised {type(error).__name__}: {error!r}; it is skipped; "
                    f"{tail}"
                )
            except BaseException:  # noqa: BLE001 - a broken log sink must not raise
                pass
        return False

    def _key(self, source, store: str | None) -> str:
        """Return the key ``source`` is recorded under."""
        key = self._keys.get(id(source))
        if key is not None:
            return key
        if store is None:
            try:
                store = source.store_path
            except Exception:  # noqa: BLE001 - a view without a store
                pass
        return f"{type(source).__name__}:{store}"

    def _log(self, source, start, end, symbols, variables, reread, store) -> None:
        """Log one request of ``source``; a repeated request is kept once."""
        request = {
            "start": _text(start),
            "end": _text(end),
            "symbols": None if symbols is None else [_plain(s) for s in symbols],
            "variables": None if variables is None else sorted(variables),
        }
        requests = self._requests.setdefault(self._key(source, store), {})
        requests.setdefault(_request_id(request), (request, reread))

    def _close(self, *, partial: bool) -> None:
        """Hash every logged request once, then compare with ``expected``."""
        token = _ACTIVE.set(None)  # re-reads are bookkeeping, not run reads
        try:
            for key, requests in self._requests.items():
                entries = self.records.setdefault(key, [])
                for request, reread in requests.values():
                    panel = reread()
                    names = request["variables"] or list(panel.data_vars)
                    entries.append(
                        {"request": request, **dataset_fingerprint(panel, names)}
                    )
        finally:
            _ACTIVE.reset(token)
        if self.expected is not None:
            self._compare(partial=partial)

    def _warn(self, key: str, request: dict | None, problem: str, tail: str) -> None:
        """Log one mismatch warning."""
        where = f"{key!r}" if request is None else (
            f"{key!r}, request {_describe_request(request)}"
        )
        logger.warning(
            f"{self.owner}: data fingerprint mismatch for {where}: {problem}; {tail}"
        )

    def _compare(self, *, partial: bool) -> None:
        """Compare ``records`` with ``expected`` by digest, warning per difference.

        On the failure path (``partial``) a key or request expected but not
        read yet is skipped, since "not read yet" is not "not read".
        """
        tail = PARTIAL_NOTE if partial else "continuing"
        expected, actual = self.expected or {}, self.records
        for key in list(dict.fromkeys([*expected, *actual])):
            if key not in actual:
                if not partial:
                    self._warn(key, None, "in the expected record but not read by this run", tail)
                continue
            if key not in expected:
                self._warn(key, None, "read by this run but absent from the expected record", tail)
                continue
            wanted = {_request_id(e["request"]): e for e in expected[key]}
            got = {_request_id(e["request"]): e for e in actual[key]}
            for request_id in list(dict.fromkeys([*wanted, *got])):
                if request_id not in got:
                    if not partial:
                        entry = wanted[request_id]
                        self._warn(
                            key, entry["request"],
                            f"in the expected record ({_extent(entry)}) but not "
                            f"read by this run", tail,
                        )
                    continue
                if request_id not in wanted:
                    entry = got[request_id]
                    self._warn(
                        key, entry["request"],
                        f"read by this run ({_extent(entry)}) but absent from "
                        f"the expected record", tail,
                    )
                    continue
                old, new = wanted[request_id], got[request_id]
                if old.get("digest") != new.get("digest"):
                    self._warn(
                        key, new["request"],
                        f"digest differs (expected {_extent(old)}, got "
                        f"{_extent(new)}). The data changed since the expected run",
                        tail,
                    )
