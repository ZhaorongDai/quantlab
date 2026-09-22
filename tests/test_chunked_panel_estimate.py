"""The dense-panel estimator/guard pair is GONE — pinned in both directions.

Until phase 03.6 this module pinned the estimate/assert split that 03.5 D-10
introduced: `estimate_chunked_panel` completed its per-chunk loop and answered
for EVERY window (`fits`, a per-window `remedy`, `dense_bytes`), while
`assert_chunked_panel_fits` was the thin raising wrapper over it, and both
carried a keyword-only `bars_per_day: int = 1` whose default was the arithmetic
identity. Eighteen tests drove that pair: the budget boundary, the loop
completion, the chunk-window adjacency, the `390x` intraday factor, the remedy
sentence, the chunk report's rendering, and the `EXPLICIT_SYMBOLS_CATEGORY`
sentinel's reach into the chunked entry points.

Phase 03.6 SC-3 DELETED that pair — both halves, the estimating one as well as
the refusing one — together with `MAX_DENSE_PANEL_BYTES`, `estimate_dense_panel`
and `assert_dense_panel_fits`. The eighteen tests are not salvageable one by
one, because their subject no longer exists. Recording the history above in
prose is deliberate (03.6 D-18): a future reader must be able to see that the
coverage was RETIRED WITH ITS SUBJECT rather than quietly lost.

What replaces them is the inverse of what they did, and it is two-directional,
per 03.6 D-15 as amended 2026-09-12. D-15 originally asked these suites to pin
"no longer refuses"; the amendment makes the operative pin "no longer EXISTS"
(`not hasattr`), because this phase deletes the estimating half too, so there is
no object left to observe not-refusing:

- `test_the_dense_panel_estimator_and_guard_are_both_deleted` — SC-3. Red if any
  of the five comes back, and red if any test module anywhere still reaches for
  one in EXECUTABLE code (an AST walk, because four modules name them in the
  retirement prose D-18 requires and a text grep cannot tell prose from a call).
- `test_the_acquisition_volume_guard_was_not_collaterally_deleted` — SC-4. Red if
  a deletion that satisfies SC-3 takes one member too many with it.

The two together are why "the guard was REMOVED" stays distinguishable from "the
guard was BYPASSED": a bypassed guard still answers `hasattr`.
"""

import ast
from pathlib import Path

from quantlab.universe import UniverseCatalog

#: The five members phase 03.6 SC-3 deletes from `UniverseCatalog`.
_DELETED_DENSE_PANEL_MEMBERS = (
    "MAX_DENSE_PANEL_BYTES",
    "estimate_dense_panel",
    "assert_dense_panel_fits",
    "estimate_chunked_panel",
    "assert_chunked_panel_fits",
)

#: The eleven `UniverseCatalog` members of the acquisition-volume group that
#: ROADMAP 03.6 SC-4 declares UNTOUCHED: the five it names by hand, plus the six
#: constants those five read, without which the named five could not run.
#:
#: `_roster_window_profile`, `BARS_PER_DAY_BY_FREQUENCY`, `TRADING_DAYS_PER_YEAR`
#: and `CALENDAR_DAYS_PER_YEAR` are deliberately NOT listed here: they are pinned
#: NUMERICALLY, by the pre-cut golden literals in
#: `tests/test_volume_guard.py::test_the_acquisition_volume_arithmetic_survives_the_roster_profile_split`,
#: which is a strictly stronger claim than `hasattr`.
_SURVIVING_ACQUISITION_VOLUME_MEMBERS = (
    "_resolve_volume_knobs",
    "estimate_acquisition_volume",
    "_narrowing_that_fits",
    "_crossed_ceilings",
    "assert_acquisition_volume_fits",
    "ACQUISITION_CEILINGS",
    "MAX_RAW_BYTES",
    "MAX_ACQUISITION_REQUESTS",
    "MAX_ACQUISITION_WALL_CLOCK_HOURS",
    "BYTES_PER_RAW_ROW",
    "DEFAULT_RATE_LIMIT_PER_MIN",
)

#: The three `quantlab/utils/cli.py` helpers SC-4 names beside them.
_SURVIVING_CLI_HELPERS = (
    "volume_pricing",
    "add_volume_guard_args",
    "print_volume_estimate",
)


def test_the_dense_panel_estimator_and_guard_are_both_deleted():
    """SC-3: all five dense-panel members are GONE from `UniverseCatalog`.

    The accepted cost, in one sentence: an over-sized dense panel now reaches
    OOM instead of a legible refusal naming a finer `--chunk` — decided by the
    developer 2026-09-11, re-affirmed 2026-09-12 when the phase scope was
    narrowed, with quick task 260906-13w (~7.2 GiB grid, ~29.6M-row frame, 16
    GiB box) as the recorded precedent for the failure mode being accepted.

    `not hasattr` rather than "no longer raises" is the amended D-15 pin: a
    guard that was merely bypassed still answers `hasattr`, so only absence
    distinguishes removal from bypass.
    """
    present = [
        member
        for member in _DELETED_DENSE_PANEL_MEMBERS
        if hasattr(UniverseCatalog, member)
    ]
    assert not present, (
        f"phase 03.6 SC-3 deleted these, but they are back on UniverseCatalog: "
        f"{present}. The surviving pre-flight guard is "
        f"`assert_acquisition_volume_fits`, which bounds disk bytes, request "
        f"count and wall clock — not dense-panel RAM. If a RAM guard is wanted "
        f"again, that is a scope decision to re-open, not a member to re-add."
    )

    # Second arm: NO test module anywhere still reaches for one of the five in
    # EXECUTABLE code.
    #
    # An AST walk, never a text grep, and the distinction is the whole point.
    # Four test modules (this one included) name all five symbols in prose, on
    # purpose: 03.6 D-18 requires a falsified locked text to be annotated in
    # place rather than silently vanish, so the history of what was pinned and
    # why it was retired has to stay readable. A literal `grep -r` cannot tell
    # that prose from a live call site, and the tuple above — the subject of
    # this very test — would trip it too. So the claim is narrowed to what it
    # was always about: an attribute access, a bare name or an import.
    offenders: list[str] = []
    for path in sorted(Path(__file__).parent.glob("test_*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                found = node.attr
            elif isinstance(node, ast.Name):
                found = node.id
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    if alias.name in _DELETED_DENSE_PANEL_MEMBERS:
                        offenders.append(f"{path.name}:{node.lineno}:{alias.name}")
                continue
            else:
                continue
            if found in _DELETED_DENSE_PANEL_MEMBERS:
                offenders.append(f"{path.name}:{node.lineno}:{found}")
    assert not offenders, (
        f"these test sites still reach for a deleted dense-panel member in "
        f"executable code: {offenders}"
    )


def test_the_acquisition_volume_guard_was_not_collaterally_deleted():
    """SC-4, the fence around SC-3's cut, proved POSITIVELY.

    SC-3 and SC-4 are in direct tension: `estimate_acquisition_volume` — the
    heart of the group SC-4 declares untouched — read its symbol count, trading
    days, observed cells and density off the very estimator SC-3 deletes. A
    deletion plan that satisfies SC-3 by removing one member too many must fail
    HERE, in a test whose whole subject is the survivors, rather than in
    production on the day someone prices a full-market minute backfill.

    Absence of the deleted group is NOT evidence that the surviving group is
    intact; that is why this test asserts survival positively instead of
    inferring it.

    The end-to-end arm matters as much as the `hasattr` arms:
    `_explicit_symbol_catalog` is the synthetic view that lets an explicit
    `--symbols` list be priced on a machine with no `universe.parquet` on disk,
    and it used to get there through an `estimate_dense_panel` override. If the
    cut disabled that path, every `hasattr` above would still pass while the
    `--symbols` volume guard silently stopped guarding.
    """
    import quantlab.utils.cli as cli
    from quantlab.utils.cli import EXPLICIT_SYMBOLS_CATEGORY, _explicit_symbol_catalog

    missing = [
        member
        for member in _SURVIVING_ACQUISITION_VOLUME_MEMBERS
        if not hasattr(UniverseCatalog, member)
    ]
    assert not missing, (
        f"SC-4 declares the acquisition-volume group UNTOUCHED, but these are "
        f"gone from UniverseCatalog: {missing}. That group bounds money and "
        f"wall clock; a burned API quota is not recoverable, which is why it "
        f"was kept when the RAM guard beside it was deleted."
    )

    missing_helpers = [
        helper for helper in _SURVIVING_CLI_HELPERS if not hasattr(cli, helper)
    ]
    assert not missing_helpers, (
        f"SC-4 keeps these `quantlab/utils/cli.py` helpers beside the guard, "
        f"but they are gone: {missing_helpers}"
    )

    pricing = _explicit_symbol_catalog(500)
    estimate = pricing.assert_acquisition_volume_fits(
        EXPLICIT_SYMBOLS_CATEGORY,
        "2024-01-01",
        "2024-12-31",
        frequency="1d",
        batch_size=1,
    )
    # A year of daily bars for 500 hand-named symbols is far under every
    # ceiling, so the guard must RETURN its estimate rather than raise — and
    # the estimate must be the real one, reached with no roster file on disk.
    assert estimate["symbols"] == 500
    assert estimate["rows"] > 0
    assert estimate["raw_bytes"] < UniverseCatalog.MAX_RAW_BYTES
