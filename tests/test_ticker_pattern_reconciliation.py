"""Pins the reconciliation between the two ends of the symbol lifecycle:
what the ROSTER BUILDER persists, and what the FETCH GUARD admits.

Quick task 260907-10t. Two separately-deliberate decisions were never pinned
against each other:

- 260906-eme decided that `us_all` deliberately RETAINS its 1,124 warrant /
  unit / right / when-issued lines -- a named, measured, carried-forward
  finding.
- 03.2-03 decided that `Acquisition._validate_symbols` admits at most ONE
  suffix segment, because a symbol becomes both a filesystem path segment and
  a comma-joined query value.

Each is correct alone. Together they mean `download()`'s whole-roster
pre-flight raises on `NXG-R-W` and kills a multi-hour full-market job before
one request is issued. Nothing in the suite could see it, because no test ever
fed the builder's OUTPUT to the guard.

That end-to-end assertion is
`test_every_symbol_either_roster_builds_is_accepted_by_the_fetch_guard` below.
"""

import inspect
import re
from pathlib import Path
from typing import Optional

import pytest

import base.acquisition as acquisition_module
import enums.data as enums_data


# ---------------------------------------------------------------------------
# Measured 2026-09-07 against data/data/reference/universe.parquet (25,709
# rows). Every literal in this module is a REAL symbol from that table.
# ---------------------------------------------------------------------------

#: Three-segment `ROOT-X-Y` symbols the roster builder deliberately KEEPS and
#: the pre-fix fetch guard refused. 77 such symbols in `us_all`, 4 in
#: `nasdaq_all`.
_RETAINED_MULTI_SUFFIX = (
    "NXG-R-W",  # the symbol the real full-market run actually halted on
    "ACP-R-W",
    "AVK-R-W",
    "DB-R-W",
    "C-WS-A",
    "GM-WS-B",
    "BAC-WS-A",
    "KODK-WS-A",
    "MIMO-W-A",
    "MT-WS-W",
    # Family evidence `['UA', 'UA-C', 'UA-C-W']`: `UA-C` is one of
    # 260906-eme's own docstring-named verified surviving class shares, so
    # `UA-C-W` is that class-C share's when-issued line. Shape `2-1-1`,
    # byte-identical to `DB-R-W`. A fixture literal precisely so this call is
    # re-checkable rather than buried in a plan.
    "UA-C-W",
)

#: `nasdaq_all`'s frozen preferred shares (Locked Decision A4 / D-02). Shape
#: `4-1-1` -- the SAME legitimate three-segment shape, so the widening admits
#: them. They are never dropped: the build-time filter's axis is
#: well-formedness, never security type.
_FROZEN_PREFERREDS = ("FITB-P-A", "FITB-P-I", "FITB-P-K", "FITB-P-M")

#: The real malformed literals. 6 in `us_all`, 7 in `nasdaq_all` (overlapping
#: on 4). Measured to be exactly and only what the widened pattern rejects.
_MALFORMED = (
    "CAPTW(EXP20260807)",
    "NXT(EXP20091224)",
    "DTV_1",
    "ETP-",
    # Family `['NSPR', 'NSPR-WS', 'NSPR-WSB']` -- no `NSPR-WS-B` exists, so
    # dropping this loses one microcap warrant series. Accepted deliberately:
    # admitting it means widening EVERY suffix segment from {1,2} to {1,3} for
    # all 14,485 symbols on the evidence of two outliers.
    "NSPR-WSB",
    # Family `['OXY', 'OXY-WS', 'OXY-WS-W', 'OXY-WSW']` -- the properly
    # delimited form of the same warrant is ALREADY in the roster, so dropping
    # the un-delimited variant loses no security at all.
    "OXY-WSW",
    "-P-HIZ",
    "ASRV 8.45 06-30-28",
    "CHNG 6",
)

#: The path-traversal / query-injection cases the guard exists for. These are
#: what makes the widening a widening and not a hole.
_UNSAFE = (
    "AAPL/../etc",
    "A/B",
    "../X",
    "A..B",
    "AAPL,MSFT",
    "AAPL MSFT",
    "aapl",
    "",
)


def _acquisition(acquisition_config, **config_kwargs):
    """A minimal CONCRETE `Acquisition` whose `_fetch_page` is never called.

    Same idiom as `tests/test_acquisition_batching.py:_acquisition`, and for
    the same reason: `_validate_symbols` belongs to the base class, so
    instantiating a vendor here would let a vendor override silently satisfy
    the assertion. `_validate_symbols` reads only `self.class_name` and
    `self.config.raw_data_dir_path`, so nothing here touches the network.
    """
    from base.acquisition import Acquisition

    class _Guard(Acquisition):
        VENDOR = "tiingo"
        RAW_COLUMNS = ("timestamp", "symbol", "vendor")

        def _fetch_page(self, symbols, start_date, end_date, page_token=None):
            raise AssertionError(
                "symbol validation must not issue a vendor request"
            )

    return _Guard(acquisition_config(**config_kwargs))


def test_the_fetch_guard_admits_the_multi_suffix_securities_the_roster_keeps(
    acquisition_config,
):
    """THE bug, stated as an assertion.

    Every literal here is a symbol the roster builder persists today. The
    pre-fix one-suffix pattern refuses all of them, which is what made
    `download()` abort a full-market run at pre-flight.

    Reddened by: narrowing `TRADEABLE_TICKER_PATTERN`'s suffix group back to
    `{0,1}` (the pre-fix pattern).
    """
    acq = _acquisition(acquisition_config)
    admitted = _RETAINED_MULTI_SUFFIX + _FROZEN_PREFERREDS

    # One call, so the failure names every refused symbol rather than only the
    # first -- a per-symbol loop would report `NXG-R-W` and hide the rest.
    assert acq._validate_symbols(admitted) == list(admitted)


def test_the_widened_guard_is_not_a_hole(acquisition_config):
    """The widening changes the suffix REPETITION COUNT and nothing else.

    The character class `[A-Z0-9]`, the delimiter class `[.-]` and both
    anchors are untouched, so a path separator, a parent reference, an
    embedded comma, a space and lowercase remain unrepresentable rather than
    merely unmatched (T-10t-01).

    Reddened by: replacing the pattern body with `.*`.
    """
    acq = _acquisition(acquisition_config)

    for symbol in _UNSAFE + _MALFORMED:
        with pytest.raises(ValueError, match="well-formed ticker pattern"):
            acq._validate_symbols([symbol])


def test_the_fetch_guard_binds_the_shared_pattern_object_not_a_copy():
    """IDENTITY, not equality -- a re-declaration must be impossible rather
    than merely discouraged (T-10t-02).

    Before this task `base/acquisition.py` carried its own standalone
    `re.compile` of the same literal while its comment and its
    `_validate_symbols` docstring BOTH claimed the pattern was imported from
    `acquisition/universe.py`. Two copies of one literal, already free to
    diverge, with the provenance comment asserting otherwise -- which is
    exactly how the two ends drifted without anything noticing.

    The shared object lives in `enums/data.py` because neither module may
    import the other: `base/acquisition.py` importing `acquisition.universe`
    inverts the layering, and `acquisition/universe.py` importing
    `base.acquisition` breaks `tests/test_volume_guard.py`'s structural
    assertion that the volume guard lives where no acquisition client can be
    constructed.

    Reddened by: restoring the standalone `re.compile` at
    `base/acquisition.py:34`.
    """
    assert (
        acquisition_module._TICKER_PATTERN is enums_data.TRADEABLE_TICKER_PATTERN
    ), "the fetch guard must BIND the shared object, not an equal copy"

    source = Path(inspect.getfile(acquisition_module)).read_text()
    offenders = [
        line
        for line in source.splitlines()
        if re.search(r"_TICKER_PATTERN\s*=\s*re\.compile", line)
    ]
    assert not offenders, offenders

    # And the module compiles no ticker regex of its own by any other name.
    uncommented = [
        line for line in source.splitlines() if not line.lstrip().startswith("#")
    ]
    assert not [line for line in uncommented if "re.compile" in line]
