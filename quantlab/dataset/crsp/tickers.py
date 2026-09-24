"""Read side of the ticker sidecar: spell a PERMNO as a ticker for a human.

The CRSP price panel's ``symbol`` axis is the integer PERMNO. That is the
right identity for a machine and the wrong one for a log line, a
``liquidations.json`` entry or a coverage report. The names live in a JSON
sidecar next to the store, ``{zarr}.crsp_tickers.json``, as intervals per
PERMNO (FB and META are the same PERMNO 13407, so one name per PERMNO would
be wrong for half its history). The conversion writes the sidecar;
``CrspTickerLookup`` is the as-of query over it.

The lookup has three entry points with three deliberate postures towards a
sidecar that is missing, unparseable or wrongly shaped:

- ``as_of(permno, day)`` is strict. It raises a shaped error (naming the
  class, the path and the rebuild that fixes it) for all three failures,
  because "the file could not be read" must not be rounded down to "that
  PERMNO had no name that day".
- ``product_end`` is strict about the file and its top level, but not about
  the interval table: a sidecar whose intervals are broken can still say
  which CRSP vintage it was read against, and one that records no vintage
  answers ``None``.
- ``label(permnos, day)`` is the display entry point and never raises. An
  unusable sidecar degrades to the PERMNO digits, which is exactly what the
  messages printed before the sidecar existed, and the degradation is logged
  once per lookup instance so a broken sidecar can be told apart from a
  store that never had one.

The module imports only the standard library and ``loguru`` at module scope,
so any layer may import it. The one project constant it needs, the sidecar
suffix, is imported inside ``beside_store`` so that a log line never pulls
the whole converter onto its import path.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import date
from pathlib import Path

from loguru import logger

__all__ = ["CrspTickerLookup"]

#: What an unusable sidecar raises. Every structural defect is funnelled
#: through ``_malformed`` (``ValueError``) or the ``payload`` property
#: (``FileNotFoundError`` for an absent file, ``ValueError`` for bytes that
#: do not parse), so these two types are exhaustive for damage that came off
#: the disk. ``KeyError``, ``AttributeError`` and ``TypeError`` are
#: deliberately not included: after the structural guards they can only be a
#: bug in this module, and ``label()`` must let such a bug reach the caller
#: rather than print digits that look like a legitimate answer.
_UNUSABLE = (FileNotFoundError, ValueError)


class CrspTickerLookup:
    """As-of PERMNO-to-ticker lookup over one ``{zarr}.crsp_tickers.json``.

    Constructed from a path and read lazily on first use. The callers are
    display sites deep inside a backtest or a log call: they hold a store
    path and nothing else, and a constructor that hit the disk would make
    "build a lookup just in case" cost an I/O per call site.

    Example:
        >>> from datetime import date
        >>> from quantlab.dataset.crsp.tickers import CrspTickerLookup
        >>> lookup = CrspTickerLookup.beside_store("data/data/us_equity/1d/crsp.zarr")
        >>> lookup.as_of(13407, date(2022, 6, 8)), lookup.as_of(13407, date(2022, 6, 9))
        ('FB', 'META')
        >>> lookup.label([13407, 14593, 99999], date(2020, 1, 1))
        ['FB', 'AAPL', '99999']
    """

    def __init__(self, sidecar_path: str | Path) -> None:
        """Bind a sidecar path without reading it."""
        self.sidecar_path = Path(sidecar_path)
        self._payload: dict | None = None
        #: Whether this instance has already warned that its sidecar is
        #: unusable. A log throttle only (see ``_degrade``), never read by a
        #: query. Per instance rather than per module so that one broken
        #: store does not silence the warning for every other store in the
        #: same process.
        self._degraded = False

    def __repr__(self) -> str:
        """Return ``CrspTickerLookup('<sidecar path>')``."""
        return f"CrspTickerLookup({str(self.sidecar_path)!r})"

    @classmethod
    def beside_store(cls, zarr_file_path: str | Path) -> "CrspTickerLookup":
        """Return the lookup for the sidecar written beside ``zarr_file_path``.

        This is the one place on the read side where the sidecar suffix is
        appended, so display sites do not each spell ``".crsp_tickers.json"``.

        The import is function-local on purpose: ``quantlab.dataset.crsp``
        owns the constant and pulls in polars and the whole converter, none
        of which a log line needs. It also imports this module's own parent
        package, which is safe only because it runs at call time, when the
        parent is fully initialised.

        Example:
            >>> CrspTickerLookup.beside_store("data/data/us_equity/1d/crsp.zarr")
            CrspTickerLookup('data/data/us_equity/1d/crsp.zarr.crsp_tickers.json')
        """
        from quantlab.dataset.crsp import TICKER_SIDECAR_SUFFIX

        return cls(str(zarr_file_path) + TICKER_SIDECAR_SUFFIX)

    # -- the file -----------------------------------------------------------

    @property
    def payload(self) -> dict:
        """Return the sidecar as parsed JSON, read at most once per instance.

        Raises:
            FileNotFoundError: If the sidecar is absent. The message names
                the class, the path and the remedy (a rebuild), because the
                file is written by a conversion and cannot be created by hand.
            ValueError: If the bytes cannot be read or parsed. This includes
                ``RecursionError`` from JSON nested deeper than the parser's
                stack, which is a ``RuntimeError`` and would otherwise escape
                both entry points unshaped.

        Example:
            >>> sorted(lookup.payload)
            ['generated_from', 'intervals', 'vintage_product_end']
        """
        if self._payload is None:
            if not self.sidecar_path.exists():
                raise FileNotFoundError(
                    f"{type(self).__name__}: no ticker sidecar at "
                    f"{str(self.sidecar_path)!r}. It is written by the CRSP "
                    f"conversion beside the store it describes, and the "
                    f"store-exists guard means an append never creates one "
                    f"after the fact. Delete the store together with its "
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
                    f"each of the store's PERMNOs wore over which dates, so "
                    f"without it the panel's numbers cannot be spelled. "
                    f"Delete the store together with its '.crsp_*.json' "
                    f"sidecars and rebuild."
                ) from exc
        return self._payload

    def _malformed(self, detail: str) -> ValueError:
        """Build the error for a sidecar that parsed but is shaped wrong.

        Returned rather than raised, so every structural check reads
        ``raise self._malformed(...)`` and the four-part message (which class,
        which path, what is wrong, how to fix it) is written once.
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
        """Warn once per instance that the sidecar is unusable.

        The digits ``label()`` falls back to are also the normal output of a
        store that has no sidecar at all, so without this line a broken
        sidecar would be invisible on a console. It warns on the absent case
        too, for the same reason.

        Once per instance, because the worst caller hands a whole
        forced-liquidation batch to a single ``label()`` call and would
        otherwise log hundreds of identical lines. The ``_degraded`` flag is
        written here and in ``__init__`` and read only here; it never
        influences a return value, so a race between threads can at most
        cost a duplicate log line.
        """
        if not self._degraded:
            self._degraded = True
            logger.warning(
                f"{type(self).__name__}: the ticker sidecar "
                f"{str(self.sidecar_path)!r} is unusable, falling back to "
                f"PERMNO digits for this store ({type(exc).__name__}: {exc}). "
                f"The digits are also what a store with no sidecar prints, so "
                f"this line is the only thing that tells the two apart. Said "
                f"once per lookup, not once per name."
            )

    def _object_payload(self) -> dict:
        """Return the payload once it is known to be a JSON object.

        The one place the top-level shape is checked; ``_intervals`` and
        ``product_end`` both read through it, so a ``[]`` sidecar gets the
        same shaped refusal from every entry point instead of a bare
        ``AttributeError``. ``FileNotFoundError`` and ``ValueError`` from
        ``payload`` pass through untouched.
        """
        payload = self.payload
        if not isinstance(payload, dict):
            raise self._malformed(
                f"its top level is a {type(payload).__name__}, not a JSON "
                f"object"
            )
        return payload

    def _intervals(self) -> dict:
        """Return the ``{PERMNO: [span, ...]}`` table, checking its shape.

        A missing ``intervals`` key is not damage: it means the sidecar knows
        no names, and both entry points have a good answer for that. Only a
        key present with the wrong type is refused.
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
        """Return the CRSP vintage the sidecar was read against, or ``None``.

        ``None`` means the sidecar records no vintage, which is a usable
        state, not damage. Only a payload whose top level is not a JSON
        object is refused, with the same shaped error ``as_of`` gives; a
        broken interval table does not affect this property.

        Raises:
            FileNotFoundError: If the sidecar is absent.
            ValueError: If it cannot be parsed or its top level is not an
                object.

        Example:
            >>> lookup.product_end
            datetime.date(2025, 12, 31)
        """
        recorded = self._object_payload().get("vintage_product_end")
        if not recorded:
            return None
        return date.fromisoformat(str(recorded)[:10])

    # -- queries ------------------------------------------------------------

    def as_of(self, permno: int, day: date) -> str | None:
        """Return the ticker ``permno`` wore on ``day``, or ``None``.

        Both ends of an interval are inclusive: PERMNO 13407's FB span ends
        2022-06-08 and its META span starts 2022-06-09, so each day belongs
        to exactly one of them. ``None`` covers both "no interval covers this
        day" and "this sidecar has never heard of this PERMNO"; no caller
        acts differently on the two.

        The spans are scanned linearly: a security's whole naming history is
        a handful of intervals, and an index would cost more than it saves.

        Args:
            permno: The PERMNO, as an int or anything ``int()`` accepts.
            day: The date to look up; anything whose ``str()`` starts with
                an ISO date works.

        Raises:
            FileNotFoundError: If the sidecar is absent.
            ValueError: If it cannot be parsed or is not shaped like a
                sidecar (top level, ``intervals``, or a span lacking
                ``start``, ``end`` or ``ticker``).

        Example:
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
        """Spell ``permnos`` for a human, one string per input, in order.

        This is the display entry point, and it is a batch call because every
        caller renders a list (a missing-member report, a dropped-symbol
        warning, a run of liquidation records on one date).

        It never raises. An unknown PERMNO falls back to its own digits, and
        so does every PERMNO when the sidecar is missing, unparseable or
        wrongly shaped. The digits are exactly what these messages printed
        before the sidecar existed, so a panel from a vendor with no sidecar
        reads as it always did. Every fall-back caused by the file goes
        through ``_degrade``, which warns once per instance. A value that is
        not an integer at all (a string symbol axis from another vendor) is
        passed through untouched and does not count as a degradation.

        The guard sits in two places because damage arrives by two routes:
        ``{"intervals": [1, 2, 3]}`` breaks while the table is being read,
        while ``{"intervals": {"13407": [{"ticker": "FB"}]}}`` reads as a
        good table and only breaks inside the per-PERMNO ``as_of``. Both
        sites catch ``_UNUSABLE`` and nothing wider. ``except Exception`` is
        deliberately not used, so that a genuine programming bug in this
        module still reaches the caller instead of being printed as digits
        that look like a legitimate "no name on that day" answer.

        Args:
            permnos: PERMNOs as ints, digit strings, or plain symbols.
            day: The date to spell them as of.

        Returns:
            One label per input, in the input's order.

        Example:
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
                # A non-integer label is not a PERMNO: it is a string symbol
                # axis from another vendor reaching a shared display path,
                # already readable, so it passes through untouched. This is
                # about the caller's argument, not the sidecar, which is why
                # there is no `_degrade` call here: nothing degraded, and
                # warning would fire on every panel that renders a symbol
                # list.
                labels.append(spelled)
                continue
            try:
                labels.append(self.as_of(permno, day) or spelled)
            except _UNUSABLE as exc:
                self._degrade(exc)
                labels.append(spelled)
        return labels
