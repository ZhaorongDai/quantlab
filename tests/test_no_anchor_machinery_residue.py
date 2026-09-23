"""A repository-level gate: the deleted WINDOW-ANCHOR machinery leaves NO
residue in `quantlab/` -- and residue COUNTS COMMENTS.

Phase 03.12 moved the CRSP adjustment anchor to each PERMNO's FIRST usable row
inside the store (backward adjustment). Under that rule the anchor does not
move when a store only grows forward, which removed the reason for an entire
mechanism: a sidecar recording the conversion WINDOW quadruple
(`start_date` / `end_date` / `product_end` / the rule's name), a cross-run gate
that compared it and refused on any difference, and the record-builder and
writer that fed it. Plan 03.12-01 switched the anchor; plan 03.12-02 deleted
the machinery. This file is what keeps it deleted.

**Why a NEW file rather than a second list inside
`tests/test_no_identity_residue.py`.** That module's docstring is, word for
word, about 03.11's ticker-identity machinery, and its filename is that list's
home. Dropping an unrelated list into it would make the filename a lie about
half its contents. 03.11 established the judgement "one list, one home"; this
phase follows it by opening a second home rather than widening the first.

**The hit test counts comments, and that is deliberate, not an oversight.**
A conventional residue gate strips comments and looks only at live code. This
one does not, for a reason specific to this repository: `quantlab/dataset/`
carries an unusually high docstring density and those docstrings cross-refer
constantly, so a comment pointing at a mechanism that no longer exists is not
clutter -- it is a false statement about how the system works, and it sends the
next reader hunting for something that is not there. This phase is the sharpest
case of it yet: the deletion was four DOCSTRING passages and a handful of
comments (`quantlab/dataset/crsp/__init__.py`'s module header, the
`_derivation()` gate paragraph, the `_raw_axes_in_range` "two writes" note, the
`_raw_data_to_xr` "latched omission" note) against six methods and constants.
More than half the surface being deleted here IS prose. So: no comment-stripping
pre-pass, no AST scan that discards docstrings. If a name below appears anywhere
in a `.py` file under `quantlab/`, this gate is red.

**The scan is `quantlab/` only.** `.planning/`, `example/` and `tests/` are
deliberately out of scope: a plan document has to be able to write
"`_assert_anchor_unchanged` was deleted because...", and the prose
documentation has to be able to say a mechanism was removed in 03.12. Naming
the dead is how you explain a deletion; the rule is only that the SHIPPING
PACKAGE may not.

**Membership rule for the list below.** A name goes on it only if it was
unique to the deleted machinery. The recorded exclusion, so the next reader
sees it was weighed rather than missed: the LITERAL `.crsp_adjustment.json` is
deliberately NOT listed. It must keep living in
`quantlab/dataset/crsp/rebuild.py:CRSP_SIDECAR_SUFFIXES` -- the CLEARING list
-- because a rebuild that stopped deleting it would leave a file describing a
deleted mechanism sitting beside a store it never described (R-05). Listing
that string here would make this gate cry wolf on a line that is deliberately
correct, and a gate that cries wolf is one people learn to skip.
"""

from __future__ import annotations

import re
from pathlib import Path

#: The shipping package. The sibling of `tests/`, resolved from this file so
#: the gate works from any working directory.
PACKAGE_ROOT = Path(__file__).resolve().parent.parent / "quantlab"

#: Deleted in 03.12-02. Must be zero NOW.
DELETED_IN_03_12_02: tuple[str, ...] = (
    # crsp/__init__.py -- the two class constants naming the sidecar and the
    # rule it recorded
    "ADJUSTMENT_SIDECAR_SUFFIX",
    "ADJUSTMENT_RULE",
    # crsp/__init__.py -- the sidecar's path, the record it held, the gate that
    # compared it across runs, and the writer that laid it down
    "adjustment_sidecar_path",
    "_adjustment_record",
    "_assert_anchor_unchanged",
    "_write_adjustment_record",
    # the VALUE `ADJUSTMENT_RULE` carried. Listed separately because it was
    # written into every sidecar on disk, so it also appeared as a bare string
    # in the gate's error message and in the record shape -- a name can survive
    # its constant.
    "total_return_backward_from_last_close",
)


def _sources() -> list[Path]:
    """Every `.py` file in the shipping package, sorted."""
    assert PACKAGE_ROOT.is_dir(), f"expected the package at {PACKAGE_ROOT}"
    sources = sorted(PACKAGE_ROOT.rglob("*.py"))
    assert len(sources) > 20, (
        f"only {len(sources)} module(s) walked -- the scan must not be "
        f"vacuous, or a name reintroduced tomorrow would pass unseen"
    )
    return sources


def _hits(names: tuple[str, ...]) -> list[str]:
    """`['path:line: text', ...]` for every occurrence of every name.

    Matched on WORD BOUNDARIES, so `_adjustment_record` does not fire on a
    local named `adjustment_record` and `ADJUSTMENT_RULE` would not fire inside
    `ADJUSTMENT_RULE_NAME`. Comments and docstrings are ordinary text here and
    are matched like any other line -- see this module's docstring for why.
    """
    pattern = re.compile(
        r"(?<![A-Za-z0-9_])(?:" + "|".join(re.escape(n) for n in names) + r")"
        r"(?![A-Za-z0-9_])"
    )
    found: list[str] = []
    for source in _sources():
        relative = source.relative_to(PACKAGE_ROOT.parent)
        for number, line in enumerate(
            source.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if pattern.search(line):
                found.append(f"{relative}:{number}: {line.strip()}")
    return found


def test_the_deleted_window_anchor_machinery_leaves_no_residue_in_quantlab():
    """Zero hits in `quantlab/`, comments included.

    The failure message names every file and line, because "there is residue"
    is not actionable and the whole hazard here is that the residue sits beside
    code that must NOT be touched -- `_assert_anchor_usable` and the three
    same-row predicates stayed, one section heading away from the four methods
    that went.
    """
    residue = _hits(DELETED_IN_03_12_02)

    assert residue == [], (
        "phase 03.12 deleted the window-anchor machinery, but "
        f"{len(residue)} reference(s) to it survive in quantlab/. A reference "
        "in a COMMENT counts and is listed here on purpose: these docstrings "
        "are the design record, and one pointing at a deleted mechanism sends "
        "the next reader looking for something that is not there.\n"
        + "\n".join(residue)
    )
