"""Map the tickers of Sharadar's raw files to permatickers, as of each file's pull.

Sharadar keys every table but TICKERS by ticker, and renames a security's
whole history when its ticker changes (LESL became LESLQ: every LESL row
reads LESLQ from then on). Each raw file therefore names its securities by
the tickers *of its own pull*: a bulk file pulled before a change holds the
old ticker, a window pulled after it the new one. TICKERS, pulled whole every
run, holds only each permaticker's current ticker, so the old ticker of a
file pulled before a change is not in it.

``PermatickerResolver`` maps one file at a time (``annotate``, the hook of
``scan_raw_table``), each ticker in this order:

1. the TICKERS snapshot of the file's own run (``tables.snapshot_for``),
   which knows exactly the tickers of that pull. A ticker the snapshot gives
   to two permatickers is refused, as TICKERS always was.
2. otherwise the ticker's *holders*, the securities known to have used it:
   the permaticker current TICKERS gives it to (a ticker current TICKERS
   gives to two permatickers is refused); the securities the ACTIONS
   ``tickerchangefrom`` chain says used it, each from its change into the
   ticker (or its first price) until its change away
   (``ticker_changes``); and, for a store with one, the securities its ticker
   sidecar (``<store>.sharadar_tickers.json``) names by it, the sidecar
   having been written from an earlier TICKERS. A ticker with a single
   holder maps to it, whatever the dates (so a vendor slow to rename a
   security still maps). Several holders (the ticker was reused) are told
   apart by the file's pull date: the holder whose span covers it, the
   latest-starting one if several do, or the latest to have used it if none
   does.

A row whose ticker none of them maps is left out rather than refused:
``left_out`` drops it, logs a warning and writes the report
``<store>.unmapped.json`` (``UNMAPPED_SUFFIX``) naming each ticker's first
and last date, row count and raw files, so a daily update completes. So does
a ticker whose holders tie (two securities starting on the same day).

Examples
--------
>>> resolver = PermatickerResolver("/data/downloads/sharadar", "sep")
>>> prices = scan_raw_table("/data/downloads/sharadar", "sep", annotate=resolver.annotate).collect()
>>> prices = resolver.left_out(prices, owner="demo", report_path="/data/zarrs/sep.zarr.unmapped.json")
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import polars as pl
from loguru import logger

from quantlab.dataset.sharadar.tables import (
    permaticker_mapping,
    pull_time,
    scan_raw_table,
    snapshot_for,
    table,
)
from quantlab.utils.atomic import write_json_atomically

__all__ = [
    "TICKER_CHANGE",
    "UNMAPPED_SUFFIX",
    "PermatickerResolver",
    "map_raw_table",
    "ticker_changes",
]

#: The ACTIONS type recording a ticker change: ``contraticker`` became ``ticker``.
TICKER_CHANGE = "tickerchangefrom"

#: Suffix of the report of raw rows left out for want of a permaticker.
UNMAPPED_SUFFIX = ".unmapped.json"

#: How the vendor fills an unused contra column.
_NOT_APPLICABLE = "N/A"

#: Number of offending keys a message lists.
_SAMPLE = 5


def _text_or_none(value) -> str | None:
    """Return ``value`` as text, or ``None`` for a null, blank or ``N/A``."""
    if value is None:
        return None
    text = str(value).strip()
    return None if not text or text == _NOT_APPLICABLE else text


def ticker_changes(vendor_root: str | Path, code: str) -> dict[int, list[tuple[str, str, str | None]]]:
    """Return each permaticker's ticker changes, ``(date, old ticker, old company)``, by date.

    A ``tickerchangefrom`` row of ACTIONS says that on its ``date`` the
    security trading as ``contraticker`` (company ``contraname``) became
    ``ticker``. If a security changed away from that ticker later (``AAX``
    -> ``BBX`` -> ``BBB``: the change into ``BBX``), the row is that
    security's, the one whose change away from it comes first after the
    row's date, even when another company trades under the ticker now.
    Otherwise the row is the security the TICKERS rows of ``code`` give the
    ticker to. A row that maps to nothing, or to several permatickers, is
    left out.

    Parameters
    ----------
    vendor_root : str or Path
        ``<download-dir>/sharadar``, holding TICKERS and ACTIONS.
    code : str
        The table whose TICKERS rows own the tickers (``"sep"``).

    Returns
    -------
    dict
        ``{permaticker: [(date as YYYY-MM-DD, old ticker, old company), ...]}``.

    Examples
    --------
    >>> ticker_changes("/data/downloads/sharadar", "sep")[194817]
    [('2022-06-09', 'FB', 'FACEBOOK INC')]
    """
    rows = (
        scan_raw_table(vendor_root, "actions")
        .filter(pl.col("action") == TICKER_CHANGE)
        .select("date", "ticker", "contraticker", "contraname")
        .collect()
        .sort("date", descending=True)
    )
    owners: dict[str, set[int]] = {}
    for ticker, permaticker in permaticker_mapping(vendor_root, code).iter_rows():
        owners.setdefault(str(ticker), set()).add(int(permaticker))
    # Latest first, so a later change has resolved the ticker an earlier one
    # changed into before that earlier one is looked at.
    resolved: dict[int, list[tuple[str, str, str | None]]] = {}
    later: dict[str, list[tuple[str, int]]] = {}
    for day, ticker, old, old_company in rows.iter_rows():
        old = _text_or_none(old)
        if old is None or ticker is None:
            continue
        day = str(day)[:10]
        left = [(since, perm) for since, perm in later.get(str(ticker), []) if since > day]
        if left:
            permaticker = min(left)[1]
        else:
            candidates = owners.get(str(ticker), set())
            if len(candidates) != 1:
                continue
            (permaticker,) = candidates
        resolved.setdefault(permaticker, []).append((day, old, _text_or_none(old_company)))
        later.setdefault(old, []).append((day, permaticker))
    return {perm: sorted(changes) for perm, changes in resolved.items()}


@dataclass(frozen=True)
class _Span:
    """One security's use of one ticker: from ``start`` (``None``: always) to ``end`` (``None``: still)."""

    permaticker: int
    start: date | None
    end: date | None

    def covers(self, day: date) -> bool:
        """Return whether the span holds ``day``."""
        return (self.start is None or self.start <= day) and (self.end is None or day < self.end)


def _day(value) -> date | None:
    """Return a ``YYYY-MM-DD`` text, a date or ``None`` as a date or ``None``."""
    if value is None or value == "":
        return None
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


class PermatickerResolver:
    """Map each raw file's tickers to permatickers as of the file's pull.

    See the module docstring for the order. One resolver serves one
    mapping table (``code``: the TICKERS rows of ``tables.SharadarTable.mapping_labels``),
    and may annotate the files of several raw tables mapped through it (a
    price table and ACTIONS).

    Parameters
    ----------
    vendor_root : str or Path
        ``<download-dir>/sharadar``.
    code : str
        The table whose TICKERS rows map the tickers (``"sep"``, ``"sf1"``).
    sidecar_path : str or Path, optional
        The store's ticker sidecar (``<store>.sharadar_tickers.json``): the
        securities it names keep their tickers as of when it was written,
        including the former ones it carries. Missing or unreadable, it is
        ignored.

    Examples
    --------
    >>> resolver = PermatickerResolver("/data/downloads/sharadar", "sep")
    >>> frame = scan_raw_table("/data/downloads/sharadar", "sep", annotate=resolver.annotate)
    >>> "permaticker" in frame.collect_schema().names()
    True
    """

    def __init__(
        self, vendor_root: str | Path, code: str, *, sidecar_path: str | Path | None = None
    ) -> None:
        """Initialize without reading anything; see the class docstring."""
        self.vendor_root = Path(vendor_root)
        self.code = table(code).code
        self.sidecar_path = None if sidecar_path is None else Path(sidecar_path)
        self._holders: dict[str, list[_Span]] | None = None
        self._current: dict[str, set[int]] | None = None
        self._snapshots: dict[Path, dict[str, set[int]]] = {}
        #: Per raw file, the tickers it names that were not mapped, and why.
        self.unresolved: dict[Path, dict[str, str]] = {}

    # -- the sources ----------------------------------------------------------

    def _pairs(self, frame: pl.LazyFrame) -> dict[str, set[int]]:
        """Return ``{ticker: permatickers}`` from TICKERS-shaped rows of this table."""
        pairs: dict[str, set[int]] = {}
        rows = (
            frame.filter(pl.col("table").is_in(table(self.code).mapping_labels))
            .select("ticker", "permaticker")
            .unique()
            .collect()
        )
        for ticker, permaticker in rows.iter_rows():
            if ticker is not None and permaticker is not None:
                pairs.setdefault(str(ticker), set()).add(int(permaticker))
        return pairs

    def _snapshot(self, path: Path) -> dict[str, set[int]]:
        """Return the ``{ticker: permatickers}`` of one TICKERS snapshot, read once."""
        if path not in self._snapshots:
            self._snapshots[path] = self._pairs(pl.scan_parquet(path))
        return self._snapshots[path]

    def _current_pairs(self) -> dict[str, set[int]]:
        """Return the ``{ticker: permatickers}`` of current TICKERS, read once."""
        if self._current is None:
            self._current = self._pairs(scan_raw_table(self.vendor_root, "tickers"))
        return self._current

    def _sidecar_spans(self) -> list[tuple[str, int, date | None, date | None]]:
        """Return ``(ticker, permaticker, start, end)`` of every span the store's sidecar names.

        A span runs to the next span's start, the last one open-ended; a
        former ticker (``former``) runs from its recorded start, open-ended.
        """
        if self.sidecar_path is None or not self.sidecar_path.exists():
            return []
        try:
            payload = json.loads(self.sidecar_path.read_text(encoding="utf-8"))
            spans = []
            for permaticker, entries in payload.get("intervals", {}).items():
                starts = [_day(entry.get("start")) for entry in entries]
                for k, entry in enumerate(entries):
                    end = starts[k + 1] if k + 1 < len(entries) else None
                    spans.append((str(entry["ticker"]), int(permaticker), starts[k], end))
            for permaticker, entries in payload.get("former", {}).items():
                for entry in entries:
                    spans.append((str(entry["ticker"]), int(permaticker), _day(entry.get("start")), None))
            return spans
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            logger.warning(
                f"{type(self).__name__}: the ticker sidecar {str(self.sidecar_path)!r} is "
                f"unusable ({type(exc).__name__}: {exc}); tickers are mapped without it."
            )
            return []

    def _ticker_holders(self) -> dict[str, list[_Span]]:
        """Return each ticker's holders, one span per permaticker, built once.

        A span's start is the security's change into the ticker (ACTIONS),
        else the sidecar's start, else its first price date (TICKERS); its
        end the change away (ACTIONS), else the sidecar's next span.
        """
        if self._holders is not None:
            return self._holders
        first_price = {
            int(permaticker): first
            for permaticker, first in scan_raw_table(self.vendor_root, "tickers")
            .filter(pl.col("table").is_in(table(self.code).mapping_labels))
            .group_by("permaticker")
            .agg(pl.col("firstpricedate").min())
            .collect()
            .iter_rows()
        }
        chain: dict[tuple[str, int], tuple[date | None, date | None]] = {}
        current_of = {perm: ticker for ticker, perms in self._current_pairs().items() for perm in perms}
        try:
            changes_by_permaticker = ticker_changes(self.vendor_root, self.code)
        except FileNotFoundError:
            # No ACTIONS pulled (a raw tier of prices and holdings only).
            changes_by_permaticker = {}
        for permaticker, changes in changes_by_permaticker.items():
            start = None
            for day, old, _ in changes:
                chain[(old, permaticker)] = (start, _day(day))
                start = _day(day)
            current = current_of.get(permaticker)
            if current is not None and start is not None:
                chain[(current, permaticker)] = (start, None)
        sidecar = {}
        for ticker, permaticker, start, end in self._sidecar_spans():
            sidecar.setdefault((ticker, permaticker), (start, end))
        keys = set(chain) | set(sidecar)
        keys |= {(t, p) for t, perms in self._current_pairs().items() for p in perms}
        holders: dict[str, list[_Span]] = {}
        for ticker, permaticker in keys:
            chain_start, chain_end = chain.get((ticker, permaticker), (None, None))
            side_start, side_end = sidecar.get((ticker, permaticker), (None, None))
            start = chain_start or side_start or first_price.get(permaticker)
            if (ticker, permaticker) in chain:
                end = chain_end
            elif ticker in self._current_pairs() and permaticker in self._current_pairs()[ticker]:
                end = None
            else:
                end = side_end
            holders.setdefault(ticker, []).append(_Span(permaticker, start, end))
        self._holders = holders
        return holders

    # -- mapping --------------------------------------------------------------

    def _refuse_ambiguous(self, pairs: dict[str, set[int]], tickers: Iterable[str], where: str) -> None:
        """Raise if one of ``tickers`` maps to several permatickers in ``pairs``."""
        ambiguous = {t: sorted(pairs[t]) for t in sorted(tickers) if len(pairs.get(t, ())) > 1}
        if ambiguous:
            sample = dict(list(ambiguous.items())[:_SAMPLE])
            raise ValueError(
                f"{len(ambiguous)} {self.code!r} ticker(s) map to several "
                f"permatickers in {where}, first {sample}. Refusing rather than "
                f"guessing which company a row belongs to; re-pull TICKERS."
            )

    def resolve(self, path: str | Path, tickers: Iterable[str]) -> dict[str, int]:
        """Return the permaticker of each of ``tickers`` as named in the raw file ``path``.

        Tickers left unmapped are recorded in ``unresolved`` under ``path``.

        Raises
        ------
        ValueError
            If the file's TICKERS snapshot, or current TICKERS, gives one of
            the tickers to several permatickers.

        Examples
        --------
        >>> resolver.resolve("/data/downloads/sharadar/sep/sep.parquet", ["LESL"])
        {'LESL': 632479}
        """
        path = Path(path)
        pulled = pull_time(path)
        left = {str(t) for t in tickers if t is not None}
        mapped: dict[str, int] = {}
        snapshot = snapshot_for(self.vendor_root, pulled)
        if snapshot is not None:
            pairs = self._snapshot(snapshot)
            known = left & set(pairs)
            self._refuse_ambiguous(pairs, known, f"the TICKERS snapshot {snapshot.name}")
            mapped.update({t: next(iter(pairs[t])) for t in known})
            left -= known
        if left:
            self._refuse_ambiguous(self._current_pairs(), left, "TICKERS")
        holders = self._ticker_holders() if left else {}
        day = pulled.date()
        unresolved: dict[str, str] = {}
        for ticker in left:
            spans = holders.get(ticker, [])
            permatickers = {span.permaticker for span in spans}
            if len(permatickers) == 1:
                mapped[ticker] = spans[0].permaticker
                continue
            if not spans:
                unresolved[ticker] = "no permaticker"
                continue
            chosen = self._holder_on(spans, day)
            if chosen is None:
                unresolved[ticker] = f"several securities used it ({sorted(permatickers)})"
            else:
                mapped[ticker] = chosen
        if unresolved:
            self.unresolved.setdefault(path, {}).update(unresolved)
        return mapped

    @staticmethod
    def _holder_on(spans: list[_Span], day: date) -> int | None:
        """Return the holder of a reused ticker on ``day``, or ``None`` on a tie.

        The span covering ``day``, the latest-starting one if several do;
        with none covering it, the one that ended last on or before it.
        """
        covering = [s for s in spans if s.covers(day)]
        if covering:
            best = max((s.start or date.min) for s in covering)
            chosen = {s.permaticker for s in covering if (s.start or date.min) == best}
        else:
            ended = [s for s in spans if s.end is not None and s.end <= day]
            if not ended:
                return None
            best = max(s.end for s in ended)
            chosen = {s.permaticker for s in ended if s.end == best}
        return next(iter(chosen)) if len(chosen) == 1 else None

    def annotate(self, path: Path, frame: pl.LazyFrame) -> pl.LazyFrame:
        """Return one raw file's scan with a ``permaticker`` column, null where none maps.

        The hook ``scan_raw_table(..., annotate=resolver.annotate)`` calls for
        every file; it reads the file's distinct tickers.
        """
        tickers = frame.select(pl.col("ticker").unique()).collect().get_column("ticker").to_list()
        mapped = self.resolve(path, tickers)
        pairs = pl.DataFrame(
            {"ticker": list(mapped), "permaticker": list(mapped.values())},
            schema={"ticker": pl.String, "permaticker": pl.Int64},
        )
        return frame.join(pairs.lazy(), on="ticker", how="left")

    def left_out(
        self,
        frame: pl.DataFrame,
        *,
        owner: str,
        report_path: str | Path | None = None,
        date_column: str = "date",
        quiet: bool = False,
    ) -> pl.DataFrame:
        """Drop the rows no permaticker was found for, report them, and return the rest.

        Parameters
        ----------
        frame : pl.DataFrame
            Annotated rows (``annotate``), with ``ticker`` and ``date_column``.
        owner : str
            Named in the warning.
        report_path : str or Path, optional
            Where to write the report (``<store>.unmapped.json``), atomically:
            ``{"table", "owner", "unmapped": [{"ticker", "first_date",
            "last_date", "rows", "raw_files", "reason"}, ...]}``. An existing
            report is deleted when nothing is left out, so its presence
            means bars are missing. ``None`` writes nothing.
        date_column : str, default "date"
            The column whose first and last values the report gives.
        quiet : bool, default False
            Log at ``info`` instead of ``warning``, for a table where rows
            of securities outside the mapping are expected (SF3A holds funds).

        Returns
        -------
        pl.DataFrame
            ``frame`` without the rows whose ``permaticker`` is null.

        Examples
        --------
        >>> kept = resolver.left_out(frame, owner="demo", report_path="sep.zarr.unmapped.json")
        """
        missing = frame.filter(pl.col("permaticker").is_null())
        if missing.height == 0:
            if report_path is not None:
                Path(report_path).unlink(missing_ok=True)
            return frame
        summary = (
            missing.group_by("ticker")
            .agg(
                pl.col(date_column).min().alias("first_date"),
                pl.col(date_column).max().alias("last_date"),
                pl.len().alias("rows"),
            )
            .sort("ticker")
        )
        entries = []
        for ticker, first, last, rows in summary.iter_rows():
            files = sorted(p.name for p, names in self.unresolved.items() if ticker in names)
            reasons = sorted({names[ticker] for names in self.unresolved.values() if ticker in names})
            entries.append(
                {
                    "ticker": ticker,
                    "first_date": None if first is None else str(first)[:10],
                    "last_date": None if last is None else str(last)[:10],
                    "rows": int(rows),
                    "raw_files": files,
                    "reason": "; ".join(reasons) or "no permaticker",
                }
            )
        if report_path is not None:
            write_json_atomically(
                report_path,
                {"table": self.code, "owner": owner, "unmapped": entries},
                indent=2,
                sort_keys=True,
            )
        log = logger.info if quiet else logger.warning
        where = f"; listed in {report_path}" if report_path is not None else ""
        log(
            f"{owner}: {missing.height} raw row(s) of {len(entries)} ticker(s) have "
            f"no permaticker (as of their own pull, through TICKERS, the ACTIONS "
            f"ticker changes and the store's ticker sidecar) and are left out"
            f"{where}, first {[e['ticker'] for e in entries[:_SAMPLE]]}."
        )
        return frame.filter(pl.col("permaticker").is_not_null())


def map_raw_table(
    vendor_root: str | Path,
    code: str,
    *,
    owner: str,
    query=None,
    store_path: str | Path | None = None,
    sidecar_path: str | Path | None = None,
    date_column: str = "date",
    quiet: bool = False,
) -> pl.DataFrame:
    """Read one raw table with each row's permaticker, leaving out and reporting the unmapped rows.

    The rows are mapped through the TICKERS rows of ``code``'s mapping
    labels (``tables.SharadarTable.mapping_labels``), each raw file as of its
    own pull (``PermatickerResolver``).

    Parameters
    ----------
    vendor_root : str or Path
        ``<download-dir>/sharadar``.
    code : str
        The raw table to read (``"sf1"``, ``"daily"``, ``"events"``, ...).
    owner : str
        Named in messages.
    query : callable, optional
        Applied to the annotated scan before it is collected (filters, a
        column selection keeping ``ticker``, ``permaticker`` and
        ``date_column``).
    store_path : str or Path, optional
        The store the rows feed: the report goes to
        ``<store_path>.unmapped.json`` (``UNMAPPED_SUFFIX``). ``None`` writes
        no report.
    sidecar_path : str or Path, optional
        A ticker sidecar to map old tickers with (``PermatickerResolver``).
    date_column : str, default "date"
        The column the report dates rows by.
    quiet : bool, default False
        Log the left-out rows at ``info`` (``PermatickerResolver.left_out``).

    Returns
    -------
    pl.DataFrame
        The rows with a ``permaticker`` column, unmapped ones left out.

    Raises
    ------
    ValueError
        If a ticker maps to several permatickers in TICKERS or in its file's
        TICKERS snapshot.

    Examples
    --------
    >>> rows = map_raw_table("/data/downloads/sharadar", "daily", owner="demo",
    ...                      store_path="/data/zarrs/sharadar_daily_1d.zarr")
    >>> rows.columns[-1]
    'permaticker'
    """
    resolver = PermatickerResolver(vendor_root, code, sidecar_path=sidecar_path)
    frame = scan_raw_table(vendor_root, code, annotate=resolver.annotate)
    if query is not None:
        frame = query(frame)
    report = None if store_path is None else f"{store_path}{UNMAPPED_SUFFIX}"
    return resolver.left_out(
        frame.collect(), owner=owner, report_path=report, date_column=date_column, quiet=quiet
    )
