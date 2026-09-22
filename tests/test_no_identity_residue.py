"""A repository-level gate: the deleted ticker-identity machinery leaves NO
residue in `quantlab/` -- and residue COUNTS COMMENTS.

Phase 03.11 moved the CRSP price panel's symbol axis onto the int64 PERMNO
(D-01), which made five mechanisms unspellable rather than merely unused: the
`(date, symbol)` collision tie-break, the share-class respelling pass, the
delisting SYMBOL carry, the PERMNO seam with its NaN, and the symbology report
that recorded all four. Plan 03.11-03 disconnected them; plan 03.11-07 deleted
them, and plan 03.11-08 deleted the two config fields that were their last
surface (a per-PERMNO ticker pin and the seam-NaN switch). This file is what
keeps all of it deleted.

**It was TWO lists between 03.11-07 and 03.11-08, and is one again.** The
second held the names 08 was responsible for, in a STRICT expected-failure
group: red on purpose while the fields still existed, and an unexpected PASS
-- which strict mode reports as a failure -- the moment they were removed. The
point was to give the deletion checklist exactly ONE home, so plan 08 could
not finish without coming here, and so nobody had to re-derive the list from a
SUMMARY that had scrolled out of context. It worked; the group is gone and the
names sit on the single list below.

**The hit test counts comments, and that is deliberate, not an oversight.**
A conventional residue gate strips comments and looks only at live code. This
one does not, for a reason specific to this repository: `quantlab/dataset/`
carries an unusually high docstring density and those docstrings cross-refer
constantly -- one docstring in `crsp/__init__.py` described the delisting TYPE-column
verdict inheritance and the delisting TICKER carry side by side, in adjacent
sentences, and that adjacency is exactly how someone deletes the wrong half.
A comment pointing at a mechanism that no longer exists sends the next reader
hunting for it, and in a file where the comments are the design record, a
stale one is not clutter -- it is a false statement about how the system
works. So: no `grep -v '^#'`, no AST pass that discards docstrings. If a name
below appears anywhere in a `.py` file under `quantlab/`, this gate is red.

**The scan is `quantlab/` only.** `.planning/`, `example/` and `tests/` are
deliberately out of scope: a plan document has to be able to write
"`resolve_collisions` was deleted because..." and the prose documentation has
to be able to say a mechanism was removed in 03.11. Naming the dead is how you
explain a deletion; the rule is only that the SHIPPING PACKAGE may not.

**Membership rule for the list below.** A name goes on it only if it was
unique to the deleted machinery. `_as_date` was deleted from
`crsp/symbology.py` in 03.11-07 and is deliberately NOT listed: three other
modules define their own private helper of that name, so listing it would make
the gate cry wolf, and a gate that cries wolf is one people learn to skip.
"""

from __future__ import annotations

import re
from pathlib import Path

#: The shipping package. The sibling of `tests/`, resolved from this file so
#: the gate works from any working directory.
PACKAGE_ROOT = Path(__file__).resolve().parent.parent / "quantlab"

#: Deleted in 03.11-07. Must be zero NOW.
DELETED_IN_03_11_07: tuple[str, ...] = (
    # crsp/__init__.py -- the identity-resolution entry point and its sidecar
    "_resolve_identity",
    "symbology_report_path",
    "SYMBOLOGY_REPORT_SUFFIX",
    "_symbology_report",
    # crsp/symbology.py -- collision resolution and the class-suffix pass
    "resolve_collisions",
    "_class_collision_pass",
    "_collision_message",
    "_MAX_LISTED_COLLISIONS",
    "_member_spans",
    # crsp/symbology.py -- the daily labeller and its per-PERMNO report shape
    "label_rows",
    "_per_permno",
    # config fields deleted in 03.11-08, PROMOTED here out of the handover
    # group 03.11-07 parked them in. They were the ticker axis's last two
    # config fields: a per-PERMNO ticker PIN (the QQQ/QQQQ era) and the
    # NaN-at-a-PERMNO-seam switch. On the int64 PERMNO axis (D-01) neither
    # situation is expressible -- one instrument is one column for its whole
    # history, and no column ever changes company -- so 03.11-08 deleted them
    # outright under D-04, with no migration and no compatibility shim.
    #
    # The handover worked as designed. 03.11-07 parked the names in a STRICT
    # expected-failure group, so deleting the fields without coming here would
    # have turned that group into an unexpected PASS, which strict mode
    # reports as a failure. The list never had to be re-derived from a SUMMARY
    # that had scrolled out of context, and the group is gone now that the
    # names live on the list above.
    "symbol_overrides",
    "nan_adj_at_permno_seam",
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

    Matched on WORD BOUNDARIES, so `_per_permno` does not fire on the local
    variable `per_permno` and `_as_date` would not fire inside `_as_datetime`.
    Comments and docstrings are ordinary text here and are matched like any
    other line -- see this module's docstring for why.
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


def test_the_deleted_identity_machinery_leaves_no_residue_in_quantlab():
    """Zero hits in `quantlab/`, comments included.

    The failure message names every file and line, because "there is residue"
    is not actionable and the whole hazard here is that the residue sits
    beside code that must NOT be touched.
    """
    residue = _hits(DELETED_IN_03_11_07)

    assert residue == [], (
        "phase 03.11 deleted the ticker-identity machinery, but "
        f"{len(residue)} reference(s) to it survive in quantlab/. A reference "
        "in a COMMENT counts and is listed here on purpose: these docstrings "
        "are the design record, and one pointing at a deleted mechanism sends "
        "the next reader looking for something that is not there.\n"
        + "\n".join(residue)
    )


