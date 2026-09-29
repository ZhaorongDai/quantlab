"""Read the ticker sidecar to show a CRSP PERMNO as a readable ticker.

CRSP (the Center for Research in Security Prices) is a US stock database. A
PERMNO is its permanent integer id for one security. The CRSP price panel
uses the PERMNO as its ``symbol`` axis: the right id for code, but not what
a person wants to read in a log line, a ``liquidations.json`` entry or a
coverage report. The readable names live in a JSON *sidecar* next to the
Zarr store, ``{zarr}.crsp_tickers.json``, as date intervals per PERMNO. One
name per PERMNO would not do: FB and META are the same PERMNO 13407, so a
single name would be wrong for half its history. The CRSP conversion writes
the sidecar, and ``CrspTickerLookup`` answers "which ticker did this PERMNO
have on this day" from it.

The lookup has three entry points, each with a deliberate way of handling a
sidecar that is missing, unparseable or wrongly shaped:

- ``as_of(permno, day)`` is strict. It raises a clear error (naming the
  class, the path and the rebuild that fixes it) in all three cases, so
  "the file could not be read" is never mistaken for "that PERMNO had no
  name that day".
- ``product_end`` is strict about the file and its top level, but not about
  the interval table. A sidecar with broken intervals can still say which
  CRSP data version it was built from, and one that records no version
  returns ``None``.
- ``label(permnos, day)`` is the display entry point and never raises. With
  an unusable sidecar it falls back to the PERMNO digits, and logs one
  warning per lookup object, so a broken sidecar can be told apart from a
  store that never had one.

At import time the module loads only the standard library and ``loguru``,
so any layer may import it. The sidecar suffix constant is imported inside
``beside_store``, so that a log call never loads the whole converter.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import date
from pathlib import Path

from loguru import logger

__all__ = ["CrspTickerLookup"]

#: The exceptions an unusable sidecar raises. Every file problem goes
#: through ``_malformed`` (``ValueError``) or the ``payload`` property
#: (``FileNotFoundError`` for a missing file, ``ValueError`` for content that
#: does not parse), so these two types cover all damage read from disk.
#: ``KeyError``, ``AttributeError`` and ``TypeError`` are left out on
#: purpose: after the shape checks they can only be a bug in this module,
#: and ``label`` must let such a bug reach the caller rather than print
#: digits that look like a real answer.
_UNUSABLE = (FileNotFoundError, ValueError)


class CrspTickerLookup:
    """Look up the ticker a PERMNO had on a given day, from one ``{zarr}.crsp_tickers.json``.

    The file is read lazily, on first use. The callers are display code deep
    inside a backtest or a log call that only has a store path; if the
    constructor read the disk, creating a lookup "just in case" would cost a
    file read at every such place.

    Parameters
    ----------
    sidecar_path : str or Path
        Path of the ticker sidecar. Use ``beside_store`` to derive it from
        the store path.

    Attributes
    ----------
    sidecar_path : Path
        The sidecar path.

    Examples
    --------
    >>> from datetime import date
    >>> from quantlab.dataset.crsp.tickers import CrspTickerLookup
    >>> lookup = CrspTickerLookup.beside_store("data/data/us_equity/1d/crsp.zarr")
    >>> lookup.as_of(13407, date(2022, 6, 8)), lookup.as_of(13407, date(2022, 6, 9))
    ('FB', 'META')
    >>> lookup.label([13407, 14593, 99999], date(2020, 1, 1))
    ['FB', 'AAPL', '99999']
    """

    def __init__(self, sidecar_path: str | Path) -> None:
        """Initialize the lookup without reading the file; see the class docstring."""
        self.sidecar_path = Path(sidecar_path)
        self._payload: dict | None = None
        #: Whether this object already warned that its sidecar is unusable.
        #: It only limits logging (see ``_degrade``) and never affects a
        #: result. It is per object, not per module, so one broken store does
        #: not silence the warning for other stores in the same process.
        self._degraded = False

    def __repr__(self) -> str:
        """Return ``CrspTickerLookup('<sidecar path>')``."""
        return f"CrspTickerLookup({str(self.sidecar_path)!r})"

    @classmethod
    def beside_store(cls, zarr_file_path: str | Path) -> "CrspTickerLookup":
        """Return the lookup for the sidecar written next to ``zarr_file_path``.

        This is the one place on the read side that appends the sidecar
        suffix, so display code does not repeat ``".crsp_tickers.json"``.

        The import is inside the function on purpose.
        ``quantlab.dataset.crsp`` defines the constant and also loads polars
        and the whole converter, which a log line does not need. It is also
        this module's own parent package, and importing it is only safe at
        call time, when the parent is fully initialized.

        Parameters
        ----------
        zarr_file_path : str or Path
            Path of the CRSP Zarr store.

        Returns
        -------
        CrspTickerLookup
            A lookup over ``<zarr_file_path>.crsp_tickers.json``.

        Examples
        --------
        >>> CrspTickerLookup.beside_store("data/data/us_equity/1d/crsp.zarr")
        CrspTickerLookup('data/data/us_equity/1d/crsp.zarr.crsp_tickers.json')
        """
        from quantlab.dataset.crsp import TICKER_SIDECAR_SUFFIX

        return cls(str(zarr_file_path) + TICKER_SIDECAR_SUFFIX)

    # -- the file -----------------------------------------------------------

    @property
    def payload(self) -> dict:
        """Return the sidecar as parsed JSON, read at most once per instance.

        Raises
        ------
        FileNotFoundError
            If the sidecar is missing. The message names the class, the path
            and the fix (a rebuild), because the file is written by a
            conversion and cannot be created by hand.
        ValueError
            If the file cannot be read or parsed. This includes a
            ``RecursionError`` from JSON nested deeper than the parser
            allows, which would otherwise escape as a ``RuntimeError``
            without the explanatory message.

        Examples
        --------
        >>> sorted(lookup.payload)
        ['generated_from', 'intervals', 'vintage_product_end']
        """
        if self._payload is None:
            if not self.sidecar_path.exists():
                raise FileNotFoundError(
                    f"{type(self).__name__}: no ticker sidecar at "
                    f"{str(self.sidecar_path)!r}. It is written by the CRSP "
                    f"conversion next to the store it describes, and an append "
                    f"to an existing store never creates one afterwards. "
                    f"Delete the store together with its "
                    f"'.crsp_*.json' sidecars and re-convert to get it."
                )
            try:
                self._payload = json.loads(
                    self.sidecar_path.read_text(encoding="utf-8")
                )
            except (OSError, ValueError, RecursionError) as exc:
                raise ValueError(
                    f"{type(self).__name__}: the ticker sidecar "
                    f"{str(self.sidecar_path)!r} could not be read "
                    f"({type(exc).__name__}: {exc}). It records which ticker "
                    f"each of the store's PERMNOs had over which dates, so "
                    f"without it the panel's ids cannot be shown as tickers. "
                    f"Delete the store together with its '.crsp_*.json' "
                    f"sidecars and rebuild."
                ) from exc
        return self._payload

    def _malformed(self, detail: str) -> ValueError:
        """Build the error for a sidecar that parses but has the wrong shape.

        It is returned rather than raised, so each shape check reads
        ``raise self._malformed(...)`` and the message (class, path, what is
        wrong, how to fix it) is written in one place.

        Parameters
        ----------
        detail : str
            What is wrong with the file.

        Returns
        -------
        ValueError
            The error to raise.
        """
        return ValueError(
            f"{type(self).__name__}: the ticker sidecar "
            f"{str(self.sidecar_path)!r} parsed as JSON but is not shaped like "
            f"a ticker sidecar ({detail}). It is written by the CRSP "
            f"conversion and is not meant to be edited by hand. Delete the "
            f"store together with its '.crsp_*.json' sidecars and re-convert "
            f"to get a well-formed one."
        )

    def _degrade(self, exc: BaseException) -> None:
        """Warn, once per object, that the sidecar is unusable.

        The digits ``label`` falls back to are also the normal output for a
        store with no sidecar at all, so without this warning a broken
        sidecar would go unnoticed. It also warns when the file is missing,
        for the same reason.

        It warns only once per object because one caller passes a whole batch
        of forced-liquidation records to a single ``label`` call and would
        otherwise log hundreds of identical lines. The ``_degraded`` flag
        never affects a return value, so a race between threads can at most
        cause a duplicate log line.

        Parameters
        ----------
        exc : BaseException
            The error that made the sidecar unusable.
        """
        if not self._degraded:
            self._degraded = True
            logger.warning(
                f"{type(self).__name__}: the ticker sidecar "
                f"{str(self.sidecar_path)!r} is unusable, falling back to "
                f"PERMNO digits for this store ({type(exc).__name__}: {exc}). "
                f"The digits are also what a store with no sidecar prints, so "
                f"this line is the only thing that tells the two apart. "
                f"Logged once per lookup, not once per name."
            )

    def _object_payload(self) -> dict:
        """Return the payload after checking that it is a JSON object.

        This is the one place the top level is checked. ``_intervals`` and
        ``product_end`` both read through it, so a ``[]`` sidecar gets the
        same clear error from every entry point instead of a bare
        ``AttributeError``. Errors from ``payload`` pass through unchanged.
        """
        payload = self.payload
        if not isinstance(payload, dict):
            raise self._malformed(
                f"its top level is a {type(payload).__name__}, not a JSON "
                f"object"
            )
        return payload

    def _intervals(self) -> dict:
        """Return the ``{PERMNO: [span, ...]}`` table after checking its type.

        A missing ``intervals`` key is not an error: it means the sidecar
        knows no names, and both entry points handle that. Only a key of the
        wrong type is refused.
        """
        intervals = self._object_payload().get("intervals", {})
        if not isinstance(intervals, dict):
            raise self._malformed(
                f"its 'intervals' is a {type(intervals).__name__}, not a JSON "
                f"object keyed by PERMNO"
            )
        return intervals

    @property
    def product_end(self) -> date | None:
        """Return the last date of the CRSP data the sidecar was built from, or ``None``.

        CRSP publishes data in versions, each ending on a *product end*
        date. ``None`` means the sidecar records no version, which is a
        valid state, not damage. Only a payload whose top level is not a
        JSON object is refused, with the same error ``as_of`` gives; a broken
        interval table does not affect this property.

        Raises
        ------
        FileNotFoundError
            If the sidecar is absent.
        ValueError
            If it cannot be parsed or its top level is not an object.

        Examples
        --------
        >>> lookup.product_end
        datetime.date(2025, 12, 31)
        """
        recorded = self._object_payload().get("vintage_product_end")
        if not recorded:
            return None
        return date.fromisoformat(str(recorded)[:10])

    # -- queries ------------------------------------------------------------

    def as_of(self, permno: int, day: date) -> str | None:
        """Return the ticker ``permno`` had on ``day``, or ``None``.

        Both ends of an interval are inclusive. PERMNO 13407's FB interval
        ends on 2022-06-08 and its META interval starts on 2022-06-09, so
        each day belongs to exactly one of them. ``None`` means either "no
        interval covers this day" or "this sidecar does not know this
        PERMNO"; no caller needs to tell the two apart.

        The intervals are searched one by one: a security's whole naming
        history is only a handful of intervals, so an index would cost more
        than it saves.

        Parameters
        ----------
        permno : int
            The PERMNO, as an int or anything ``int()`` accepts.
        day : date
            The date to look up; anything whose ``str()`` starts with
            an ISO date works.

        Raises
        ------
        FileNotFoundError
            If the sidecar is absent.
        ValueError
            If it cannot be parsed or has the wrong shape (top level,
            ``intervals``, or an interval without ``start``, ``end`` or
            ``ticker``).

        Examples
        --------
        >>> lookup.as_of(13407, date(2022, 6, 8))
        'FB'
        >>> lookup.as_of(13407, date(2000, 1, 1)) is None
        True
        """
        spans = self._intervals().get(str(int(permno)))
        if not spans:
            return None
        if not isinstance(spans, (list, tuple)):
            raise self._malformed(
                f"the entry for PERMNO {int(permno)} is a "
                f"{type(spans).__name__}, not a list of spans"
            )
        as_of = str(day)[:10]
        for span in spans:
            if not isinstance(span, dict):
                raise self._malformed(
                    f"a span of PERMNO {int(permno)} is a "
                    f"{type(span).__name__}, not a JSON object"
                )
            absent = [key for key in ("start", "end", "ticker") if key not in span]
            if absent:
                raise self._malformed(
                    f"a span of PERMNO {int(permno)} is missing "
                    f"{', '.join(repr(key) for key in absent)}"
                )
            if str(span["start"])[:10] <= as_of <= str(span["end"])[:10]:
                return str(span["ticker"])
        return None

    def label(self, permnos: Sequence, day: date) -> list[str]:
        """Return a readable label for each of ``permnos``, in order.

        This is the entry point for display code. It takes a batch because
        every caller prints a list (a missing-member report, a dropped-symbol
        warning, the liquidation records of one date).

        It never raises. An unknown PERMNO falls back to its own digits, and
        so does every PERMNO when the sidecar is missing, unparseable or
        wrongly shaped. A store from a vendor with no sidecar therefore
        prints plain ids. Every fallback caused by the file goes through
        ``_degrade``, which warns once per object. A value that is not an
        integer at all (a string symbol from another vendor) is passed
        through unchanged and does not count as a fallback.

        Errors are caught in two places because a damaged file can fail at
        two points: ``{"intervals": [1, 2, 3]}`` fails while the table is
        read, while ``{"intervals": {"13407": [{"ticker": "FB"}]}}`` reads
        as a good table and only fails inside ``as_of``. Both places catch
        ``_UNUSABLE`` and nothing wider. ``except Exception`` is avoided on
        purpose, so a real bug in this module still reaches the caller
        instead of being printed as digits that look like a genuine "no
        name on that day" answer.

        Parameters
        ----------
        permnos : Sequence
            PERMNOs as ints or digit strings, or plain string symbols.
        day : date
            The date whose tickers to use.

        Returns
        -------
        list of str
            One label per input, in the input's order.

        Examples
        --------
        >>> lookup.label([13407, 14593, 99999], date(2020, 1, 1))
        ['FB', 'AAPL', '99999']
        >>> lookup.label(["AAPL", "MSFT"], date(2020, 1, 1))
        ['AAPL', 'MSFT']
        """
        try:
            intervals = self._intervals()
        except _UNUSABLE as exc:
            self._degrade(exc)
            intervals = {}

        labels: list[str] = []
        for value in permnos:
            spelled = str(value)
            if not intervals:
                labels.append(spelled)
                continue
            try:
                permno = int(value)
            except (TypeError, ValueError):
                # Not a PERMNO but a readable string symbol from another
                # vendor, so pass it through. The sidecar is fine, so there
                # is no `_degrade` call; warning here would fire on every
                # ticker-keyed panel.
                labels.append(spelled)
                continue
            try:
                labels.append(self.as_of(permno, day) or spelled)
            except _UNUSABLE as exc:
                self._degrade(exc)
                labels.append(spelled)
        return labels
