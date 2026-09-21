"""Read side of `{zarr}.crsp_tickers.json`: a PERMNO spelled for a human (D-03).

The CRSP price panel's `symbol` axis is the int64 PERMNO (D-01). That is the
right identity for a machine -- it never collides, never needs a share-class
suffix, and never changes when a company renames itself -- and the wrong one
for a log line, a `liquidations.json` entry or a coverage report, none of which
a person can read as digits.

**Why the names are not in the panel.** A 2-D `ticker(timestamp, symbol)`
string variable is refused by the backend's symbol-dim dtype guards, and a 1-D
`ticker(symbol)` coord holds one name per PERMNO -- so PERMNO 13407 would be
"META" for its whole history and FB's decade would be filed under a name it did
not wear. The names therefore live in a SIDECAR, as INTERVALS, and this module
is the as-of query over them. `quantlab/dataset/crsp.py` writes the file;
nothing else reads it.

**Two entry points, deliberately different about failure.**

- `as_of(permno, day)` is the strict, single-value question. All THREE ways the
  sidecar can fail RAISE, and each refusal is shaped -- it names the class, the
  path and the rebuild that fixes it: the file is MISSING, its bytes DO NOT
  PARSE (they are not decodable as UTF-8, they are not JSON at all, or they are
  nested deeper than the parser's own stack -- see the `payload` property, where
  that last one is why `RecursionError` is caught alongside `OSError` and
  `ValueError`), or it parses and is STRUCTURALLY WRONG (the top level is not an object,
  `intervals` is not an object, a span is not an object or lacks
  `start`/`end`/`ticker`). The caller asked which name a specific security wore
  on a specific day, and "I could not read the file" is not an answer that may
  be silently rounded to `None` -- rounding it down would make "this sidecar is
  unreadable" and "that PERMNO had no name that day" the same answer.
- `label(permnos, day)` is the DISPLAY entry point, and it never raises -- for
  all three of those failures alike. The six human-visible points that call it
  -- the forced-liquidation log and `liquidations.json`, the model's
  missing/extra symbol lists, `UniverseMask.report()`'s missing-member list,
  `browse_zarr`'s refusal and the `--symbols` CLI help -- are all trying to make
  an EXISTING message readable. Breaking a backtest because an audit sidecar is
  absent or half-written would make the readability layer more fragile than the
  thing it annotates (T-03.11-30), so an unusable sidecar degrades to the
  digits, which is exactly what those messages printed before this sidecar
  existed.

A LEAF module: stdlib only, no project-internal imports at module scope, so any
layer may import it. The one name it needs from `crsp.py` -- the suffix -- is
imported inside `beside_store`, because appending a string must not drag the
whole converter (polars, xarray, the reference tier) into a display path.

Shaped after `crsp_reference.py:CrspReference.manifest` in four respects, and
the resemblance is on purpose -- this repo has one way of reading a JSON
sidecar and it is worth only having one: a `None` sentinel rather than a
`hasattr` dance, a `FileNotFoundError` that names the class, the path and the
remedy, `json.loads(path.read_text(encoding="utf-8"))` as the single read, and
derived values (`product_end`) behind a `@property` rather than recomputed per
call.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import date
from pathlib import Path

__all__ = ["CrspTickerLookup"]


class CrspTickerLookup:
    """As-of PERMNO -> ticker over one `{zarr}.crsp_tickers.json`.

    Constructed from a PATH and reads it lazily, like `CrspReference`, rather
    than from an already-parsed payload like `CrspSymbology`. The consumers are
    display sites deep inside a backtest or a log call: they have a store path
    in hand and no way to obtain a parsed frame, and a constructor that hit the
    disk would make "build a lookup just in case" cost an I/O per call site.
    """

    def __init__(self, sidecar_path: str | Path) -> None:
        self.sidecar_path = Path(sidecar_path)
        self._payload: dict | None = None

    def __repr__(self) -> str:
        return f"CrspTickerLookup({str(self.sidecar_path)!r})"

    @classmethod
    def beside_store(cls, zarr_file_path: str | Path) -> "CrspTickerLookup":
        """The lookup for the store at `zarr_file_path`.

        The ONE place the suffix is appended on the read side, so the display
        points do not each spell `".crsp_tickers.json"` for themselves -- a
        literal repeated at four call sites is a rename waiting to go half-done.

        The import is function-local on purpose: `crsp.py` owns the constant
        and pulls in polars, the reference tier and the whole converter with
        it, none of which a log line needs. This module stays a stdlib-only
        leaf for every caller that already has a sidecar path.
        """
        from quantlab.dataset.crsp import TICKER_SIDECAR_SUFFIX

        return cls(str(zarr_file_path) + TICKER_SIDECAR_SUFFIX)

    # -- the file -----------------------------------------------------------

    @property
    def payload(self) -> dict:
        """The sidecar as a dict, read at most once per instance.

        Raises the same shaped `FileNotFoundError` `CrspReference` raises for a
        missing manifest: which class is complaining, which path it looked at,
        and what to do -- because this file is written BY a conversion and
        cannot be created by hand, the remedy is a rebuild, not an edit.

        A failure of the READ or the PARSE becomes the second shaped refusal,
        and "the parse failed" includes the case where the parser itself runs
        out of stack: deeply nested JSON raises `RecursionError`, which is a
        `RuntimeError` subclass and therefore caught by neither `OSError` nor
        `ValueError`. It used to escape from here past every structural guard
        downstream and out of BOTH entry points -- unshaped out of `as_of` and,
        worse, out of `label`, which the display sites call bare (G-03.11-3 /
        WR-01). Re-raising it here is what makes this the single place where an
        unreadable sidecar turns into a refusal a reader can act on. Building
        the message after a `RecursionError` is safe: CPython restores stack
        headroom once the exception unwinds.
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
        """The refusal for a sidecar that PARSED but is shaped wrong.

        RETURNS the error rather than raising it, so every structural branch
        below reads `raise self._malformed(...)` and the four-part message --
        which class, which path, what is wrong, how to fix it -- is written
        ONCE. Five hand-written copies of the same sentence drift, and they
        drift invisibly: nobody diffs error strings.

        Same shape as the `payload` property's two refusals on purpose; a
        reader who has seen one has seen all of them.
        """
        return ValueError(
            f"{type(self).__name__}: the ticker sidecar "
            f"{str(self.sidecar_path)!r} parsed as JSON but is not shaped like "
            f"a ticker sidecar ({detail}). It is written by the CRSP "
            f"conversion and is not meant to be edited by hand. Delete the "
            f"store together with its '.crsp_*.json' sidecars and re-convert "
            f"to get a well-formed one."
        )

    def _intervals(self) -> dict:
        """The `{PERMNO: [span, ...]}` table, or a refusal naming what is off.

        The one place the payload's top-level shape is checked, so `as_of` can
        index it without a second thought.

        A MISSING `intervals` key is not damage -- `.get("intervals", {})` has
        always answered "this sidecar knows no names", and both entry points
        already have a good answer for that. Only a key present with the wrong
        TYPE is a refusal.

        `FileNotFoundError` and `ValueError` out of `self.payload` (absent file,
        unparseable bytes) travel through untouched: those two refusals are the
        existing contract and this method has nothing to add to them.
        """
        payload = self.payload
        if not isinstance(payload, dict):
            raise self._malformed(
                f"its top level is a {type(payload).__name__}, not a JSON "
                f"object"
            )
        intervals = payload.get("intervals", {})
        if not isinstance(intervals, dict):
            raise self._malformed(
                f"its 'intervals' is a {type(intervals).__name__}, not a JSON "
                f"object keyed by PERMNO"
            )
        return intervals

    @property
    def product_end(self) -> date | None:
        """The CRSP vintage the sidecar's intervals were read against.

        `None` when the sidecar does not record one. A derived value behind a
        `@property`, matching `CrspReference.product_end` -- the parse belongs
        beside the field it parses, not at each reader.
        """
        recorded = self.payload.get("vintage_product_end")
        if not recorded:
            return None
        return date.fromisoformat(str(recorded)[:10])

    # -- queries ------------------------------------------------------------

    def as_of(self, permno: int, day: date) -> str | None:
        """The ticker `permno` wore on `day`, or `None`.

        Both ends of an interval are INCLUSIVE, the convention
        `symbol_intervals()` and `_member_intervals` already use: 13407's FB
        span ends 2022-06-08 and its META span starts 2022-06-09, so the
        boundary day belongs to exactly one of them.

        `None` means "no interval covers this day" and "this sidecar has never
        heard of this PERMNO" alike. The two are not distinguished because no
        caller acts differently on them: both mean there is no name to print,
        and both arise from the same cause -- a store whose sidecar was written
        for a different roster or a different window.

        A linear scan of one PERMNO's spans, not a bisect: a security's whole
        naming history is a handful of intervals (four for AAPL's 45 years),
        and an index would cost more to build than every lookup it saves.
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
        """`permnos` spelled for a human, one string per input, in order.

        The single entry point the six display points use, and the reason it is
        BATCH: every one of them is rendering a LIST (a missing-member report,
        a dropped-symbol warning, a run of liquidation records on one date), so
        a per-item call would re-enter the payload once per name.

        **Never raises.** An unknown PERMNO falls back to its own digits, and
        so does every PERMNO when the sidecar is missing or CORRUPT -- where
        corrupt means both halves of it: bytes that never reach a shape at all
        (undecodable, not JSON, or nested past the parser's stack), and bytes
        that parse fine but are not shaped like a sidecar (`intervals` holding
        a list, a span with no `start`). The digits are precisely the
        output these messages produced before the sidecar existed, so a panel
        with no sidecar (a Tiingo or Alpaca store, or a CRSP store built before
        03.11-09) reads exactly as it did. `as_of` keeps the strict behaviour
        for callers who want the refusal.

        The guard is in TWO places because damage arrives by two routes:
        `{"intervals": [1, 2, 3]}` breaks while the table is being read, while
        `{"intervals": {"13407": [{"ticker": "FB"}]}}` reads a perfectly good
        non-empty table and only breaks inside the per-PERMNO `as_of`. A guard
        on the first alone leaves the second crashing -- and the worst caller,
        `base/model.py`'s WR-02 warning, is a BARE call on a happy path.

        `except Exception` is deliberately NOT used: the tuple is spelled out
        so a genuine programming bug in this module still reaches the caller
        instead of being swallowed by a display path.
        """
        unusable = (FileNotFoundError, ValueError, KeyError, AttributeError, TypeError)
        try:
            intervals = self._intervals()
        except unusable:
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
                # A non-integer label is not a PERMNO -- a string symbol axis
                # from another vendor reaching a shared display path. It is
                # already readable; pass it through untouched.
                labels.append(spelled)
                continue
            try:
                labels.append(self.as_of(permno, day) or spelled)
            except unusable:
                labels.append(spelled)
        return labels
