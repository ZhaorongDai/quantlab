"""The ticker sidecar of a Sharadar store: permatickers shown as the ticker in use that day.

A Sharadar panel's ``symbol`` axis is the permaticker, Sharadar's unchanging
integer id of one share class: right for code, but not what a person reads in
a backtest's Holdings tab or its ``settlements.json``. Sharadar renames a
security's whole history to its current ticker, so the price table alone
never says that permaticker 194817 traded as FB before it was META. The
names come from two raw tables:

- TICKERS gives each permaticker its current ticker and company, from the
  rows labelled with the store's table (``SEP`` or ``stocks`` for SEP), so a
  fund's row never names a stock;
- ACTIONS gives each ticker change as a ``tickerchangefrom`` row: on its
  ``date`` the security, then trading as ``contraticker`` (company
  ``contraname``), became ``ticker``.

``ticker_sidecar_payload`` builds, for given permatickers, a list of spans
per permaticker (the first day each ticker was in use, ``null`` for the
first), and ``SharadarTickerLookup`` answers "which ticker and company did
this permaticker have on this day" from the JSON file
``<store>.sharadar_tickers.json`` beside the store. A conversion of
``SharadarStockDataset`` writes the file; ``write_ticker_sidecar`` and
``scripts/sharadar/ticker_sidecar.py`` write it for an existing store.

The lookup is for display: ``names`` never raises. A permaticker the file
does not know reads as its own id; so does every permaticker when the file is
missing or damaged, with one warning per lookup object.

Examples
--------
>>> lookup = SharadarTickerLookup("/data/zarrs/sharadar_sep_1d.zarr.sharadar_tickers.json")
>>> lookup.label([194817], date(2022, 6, 8)), lookup.label([194817], date(2022, 6, 9))
(['FB'], ['META'])
"""

from __future__ import annotations

import bisect
import json
from collections.abc import Iterable, Sequence
from datetime import date
from pathlib import Path

import polars as pl
from loguru import logger

from quantlab.dataset.base import SymbolName, TickerLookup
from quantlab.dataset.sharadar.tables import permaticker_mapping, scan_raw_table, table

__all__ = ["TICKER_SIDECAR_SUFFIX", "SharadarTickerLookup", "ticker_sidecar_payload"]

#: Suffix of the ticker sidecar written beside a Sharadar store.
TICKER_SIDECAR_SUFFIX = ".sharadar_tickers.json"

#: The ACTIONS type recording a ticker change: ``contraticker`` became ``ticker``.
TICKER_CHANGE = "tickerchangefrom"

#: How the vendor fills an unused contra column.
_NOT_APPLICABLE = "N/A"


def _text_or_none(value) -> str | None:
    """Return ``value`` as text, or ``None`` for a null, blank or ``N/A``."""
    if value is None:
        return None
    text = str(value).strip()
    return None if not text or text == _NOT_APPLICABLE else text


def _current_names(vendor_root: str | Path, code: str, permatickers: set[int]) -> dict[int, tuple[str, str | None]]:
    """Return each permaticker's current ticker and company from its TICKERS rows of ``code``.

    A permaticker with several rows (an old ticker listed beside the new one)
    takes the row with the latest ``lastpricedate``, then ``lastupdated``.
    """
    rows = (
        scan_raw_table(vendor_root, "tickers")
        .filter(
            pl.col("table").is_in(table(code).mapping_labels)
            & pl.col("permaticker").is_in(list(permatickers))
        )
        .select("permaticker", "ticker", "name", "lastpricedate", "lastupdated")
        .collect()
        .sort(["permaticker", "lastpricedate", "lastupdated", "ticker"], nulls_last=False)
    )
    current: dict[int, tuple[str, str | None]] = {}
    for permaticker, ticker, name, _, _ in rows.iter_rows():
        current[int(permaticker)] = (str(ticker), _text_or_none(name))
    return current


def _ticker_changes(vendor_root: str | Path, code: str) -> dict[int, list[tuple[str, str, str | None]]]:
    """Return each permaticker's ticker changes, ``(date, old ticker, old company)``, by date.

    A change row's ``ticker`` is the one the security took on its date. If
    a security changed away from that ticker later (``AAX`` -> ``BBX`` ->
    ``BBB``: the change into ``BBX``), the row is that security's, the one
    whose change away from it comes first after the row's date, even when
    another company trades under the ticker now. Otherwise the row is the
    security the TICKERS rows of ``code`` give the ticker to. A row that
    maps to nothing, or to several permatickers, names nobody and is left
    out: the sidecar is for display, and a refusal would cost a backtest
    its names over one stray row.
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


def ticker_sidecar_payload(vendor_root: str | Path, code: str, permatickers: Iterable[int]) -> dict:
    """Return the ticker sidecar of ``permatickers`` from the raw TICKERS and ACTIONS tables.

    Parameters
    ----------
    vendor_root : str or Path
        ``<download-dir>/sharadar``, holding TICKERS and ACTIONS.
    code : str
        The store's price table (``"sep"`` or ``"sfp"``), whose TICKERS rows
        map tickers to permatickers.
    permatickers : iterable of int
        The store's symbols. A permaticker without a TICKERS row of the
        table is left out of the sidecar, and reads as its id.

    Returns
    -------
    dict
        ``{"table": code, "intervals": {permaticker: [span, ...]}}``, each
        span ``{"start", "ticker", "company"}``: the first day the ticker was
        in use (``None`` for the first span), the ticker and the company
        (``None`` when the tables record none; an earlier ticker's company is
        the ``contraname`` of its change). Spans are in date order and
        permatickers in numeric order.

    Examples
    --------
    >>> payload = ticker_sidecar_payload("/data/downloads/sharadar", "sep", [194817])
    >>> [span["ticker"] for span in payload["intervals"]["194817"]]
    ['FB', 'META']
    """
    wanted = {int(p) for p in permatickers}
    current = _current_names(vendor_root, code, wanted)
    changes = _ticker_changes(vendor_root, code)
    intervals: dict[str, list[dict]] = {}
    for permaticker in sorted(current):
        ticker, company = current[permaticker]
        steps = changes.get(permaticker, [])
        spans = []
        start = None
        for day, old, old_company in steps:
            spans.append({"start": start, "ticker": old, "company": old_company})
            start = day
        spans.append({"start": start, "ticker": ticker, "company": company})
        intervals[str(permaticker)] = spans
    return {"table": code, "intervals": intervals}


class SharadarTickerLookup(TickerLookup):
    """The ticker and company a permaticker had on a day, from one ``<store>.sharadar_tickers.json``.

    The file is read on first use, so a lookup costs nothing until a name is
    asked for.

    Parameters
    ----------
    sidecar_path : str or Path
        The sidecar. ``SharadarStockDataset.ticker_lookup()`` builds one
        over its own.

    Examples
    --------
    >>> lookup = SharadarTickerLookup("/data/zarrs/sharadar_sep_1d.zarr.sharadar_tickers.json")
    >>> lookup.names([194817, 1], date(2022, 6, 9))
    [SymbolName(ticker='META', company='META PLATFORMS INC'), SymbolName(ticker='1', company=None)]
    """

    def __init__(self, sidecar_path: str | Path) -> None:
        """Initialize the lookup without reading the file; see the class docstring."""
        self.sidecar_path = Path(sidecar_path)
        self._spans: dict[str, tuple[list[str], list[tuple[str, str | None]]]] | None = None

    def __repr__(self) -> str:
        """Return ``SharadarTickerLookup('<sidecar path>')``."""
        return f"SharadarTickerLookup({str(self.sidecar_path)!r})"

    def _read_spans(self) -> dict[str, tuple[list[str], list[tuple[str, str | None]]]]:
        """Return the spans per permaticker, as start labels and names, read once.

        A missing or damaged file gives an empty table and one warning, so
        every permaticker reads as its id.
        """
        if self._spans is None:
            self._spans = {}
            try:
                payload = json.loads(self.sidecar_path.read_text(encoding="utf-8"))
                for permaticker, spans in payload["intervals"].items():
                    # A null start sorts before every day.
                    starts = [str(span["start"] or "")[:10] for span in spans]
                    names = [(str(span["ticker"]), span.get("company")) for span in spans]
                    self._spans[str(permaticker)] = (starts, names)
            except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
                self._spans = {}
                logger.warning(
                    f"{type(self).__name__}: the ticker sidecar {str(self.sidecar_path)!r} "
                    f"is unusable ({type(exc).__name__}: {exc}); permatickers are shown "
                    f"as their ids. Write it with scripts/sharadar/ticker_sidecar.py."
                )
        return self._spans

    def names(self, symbols: Sequence, day: date) -> list[SymbolName]:
        """Return the ticker and company of each of ``symbols`` on ``day``, in order.

        Never raises: a symbol the sidecar does not know, or any symbol when
        the sidecar is missing or damaged, gets its own id as ticker.

        Parameters
        ----------
        symbols : Sequence
            Permatickers as on the panel's ``symbol`` axis.
        day : datetime.date
            The date whose names to use; a span starts on its ``start``.

        Returns
        -------
        list of SymbolName
            One name per input, in the input's order.

        Examples
        --------
        >>> lookup.names([194817], date(2022, 6, 8))
        [SymbolName(ticker='FB', company='FACEBOOK INC')]
        """
        spans = self._read_spans()
        as_of = str(day)[:10]
        names = []
        for symbol in symbols:
            try:
                key = str(int(symbol))
            except (TypeError, ValueError):
                key = str(symbol)
            found = spans.get(key)
            k = -1 if found is None else bisect.bisect_right(found[0], as_of) - 1
            if k < 0:
                names.append(SymbolName(key))
            else:
                ticker, company = found[1][k]
                names.append(SymbolName(ticker, company))
        return names
