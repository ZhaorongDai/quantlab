#!/usr/bin/env python3
"""Scan the repository for statements that assert RETIRED `_failures.json` semantics.

This is a REPORTER, not a gate: it always exits 0 and prints one line per
(file, pattern) hit as `path|Pnn|count`.  The gate is built by the caller, by
diffing this output against a pre-committed `scan-allowlist.txt` in BOTH
directions:

    scan | grep -Fxv -f scan-allowlist.txt      # nothing outside the allowlist
    scan | grep -Fxv -f - scan-allowlist.txt    # nothing missing from it

Why a script instead of `grep -c LITERAL`: three of the twelve sites are broken
across source lines, and one of those is additionally broken across adjacent
Python string literals.  A line-oriented grep cannot see any of them.  Each file
is therefore normalised twice before matching:

  * `flat`   — every run of whitespace collapsed to a single space, so a
               sentence split over several lines reads as one sentence;
  * `joined` — `flat` with adjacent string-literal seams (`" "` / `' '`)
               stitched shut, so implicit concatenation cannot hide a phrase.

The count reported for a (file, pattern) pair is the LARGER of the two, because
`joined` can merge two genuinely separate literals while `flat` can miss a
concatenated phrase; taking the max never under-reports.

Excluded paths are stated in the open rather than buried in a filter: only
`.planning/phases/` and `.planning/quick/` are skipped, both being dated
historical records that are allowed to describe what was true when written.
This plan, its SUMMARY and this scanner all live under `.planning/quick/`, so
the scanner cannot match itself.  `.planning/ROADMAP.md`, `.planning/STATE.md`
and `.planning/WINDOWS.md` are NOT excluded — two of the sites live in the first
two, and WINDOWS.md's exemption is recorded in the allowlist where it is visible.

Where these patterns come from
------------------------------

P00..P10 were copied VERBATIM out of the six variants an earlier verification
report happened to list.  That report was never an enumeration -- it was a list
of the sites somebody had already found -- and copying it is exactly how this
defect recurred a fourth time: every round fixed the wordings on the list and
left untouched the ones nobody had thought of.  Saying so here, in the file, is
the point.  If you arrived intending to "just add two more patterns", stop and
read the next paragraph first.

P11..P13 are DERIVED one at a time from the `retired_sentence` column of
`.planning/phases/03.4-data-source-registry/manifest-sentence-retired.tsv` --
sentences a human adjudicated STALE in 03.4-10, retired, and rewrote.  P14 and
P15 derive from no retired row at all: they are prophylactic entries named word
for word by item 4 of `gaps[].missing` in
`.planning/phases/03.4-data-source-registry/03.4-VERIFICATION.md`.  Every entry
added after P10 carries a trailing provenance comment saying which of those two
it is, and pointing at the retired row or the verification item it came from.  A
pattern with no provenance is a pattern of unknown origin, and the next person
has no way to judge whether it should stay.

The rule from here on: the only legitimate input to this list is
`manifest-sentence-retired.tsv`, that is, the product of reading candidate
sentences one at a time.  Never a list of bad sentences assembled from memory,
and never the findings of the round you happen to be standing in.  That
direction IS the defect; it is not a shortcut to the fix.

Subordinate to the sentence-level audit
---------------------------------------

This script is a fast tripwire and nothing more.  Authority lives in the
candidate-set enumeration of
`.planning/phases/03.4-data-source-registry/manifest_sentence_audit.py`, which
intersects two SEMANTIC CLASSES over sentence text and hands every hit to a
human to adjudicate.  A literal list can only make the RE-appearance of an
already-retired wording cheap to catch.  It can say nothing at all about a
wording nobody has written down yet.

That gap is measured, not hypothetical.  03.4-11 Task 1's second mutation
appended one new false sentence about per-run scope to `example/registry.md` and
ran both tools against the SAME working tree: this script's two-way allowlist
diff produced no lines in either direction (green), while the sentence-level
audit exited 1 with an UNADJUDICATED row naming `example/registry.md`.  Two
exit codes, one tree.  Read that result narrowly.  It shows this literal list is
NARROWER than the semantic intersection -- an existence claim, which one sample
settles.  It does NOT show the semantic intersection is complete: that sentence
was chosen by the plan, and its run-scope wording was written to hit a run-scope
term by construction.  Green here is therefore never a completeness claim about
anything.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

EXCLUDED_PREFIXES = (
    ".planning/phases/",
    ".planning/quick/",
)

# Order is the contract: the index IS the `Pnn` label in the output.
# Do not reorder; append only.
PATTERNS: list[str] = [
    r"set\(result\.failures\)\s*==\s*set",          # P00
    r"does hold on every exit path",                # P01
    r"always describes the LATEST run",             # P02
    r"an empty one is a meaningful statement",      # P03
    r"last run was clean",                          # P04
    r"describing the latest run",                   # P05
    r"最近一次\s*run",                                # P06
    r"每次\s*run\s*覆盖重写",                          # P07
    r"永远描述",                                      # P08
    r"上次跑干净",                                    # P09
    r"干净收尾",                                      # P10
    r"[Tt]he last run's",  # derived: quantlab/base/coverage.py + quantlab/acquisition/inspector.py (P11)
    r"上一次\s*run\s*的",  # derived: example/registry.md (P12)
    r"lie about what the last run",  # derived: quantlab/base/acquisition.py + tests/test_tiingo_quota.py (P13)
    r"上一轮的\s*_failures",  # named-by: 03.4-VERIFICATION missing #4 (P14)
    r"describes? the last run",  # named-by: 03.4-VERIFICATION missing #4 (P15)
]

COMPILED = [re.compile(p) for p in PATTERNS]


def tracked_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "*.md", "*.py"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    files = [line for line in out.splitlines() if line]
    return [f for f in files if not f.startswith(EXCLUDED_PREFIXES)]


def normalise(raw: str) -> tuple[str, str]:
    """Return (flat, joined) — see module docstring."""
    flat = re.sub(r"\s+", " ", raw)
    joined = re.sub(r"(['\"])\s*\1", "", flat)
    return flat, joined


def main() -> int:
    for path in sorted(tracked_files()):
        # Strict utf-8 on purpose: a decode error in a tracked .md/.py file is
        # itself a finding, not something to swallow.
        raw = Path(path).read_text(encoding="utf-8")
        flat, joined = normalise(raw)
        for index, pattern in enumerate(COMPILED):
            count = max(len(pattern.findall(flat)), len(pattern.findall(joined)))
            if count:
                print(f"{path}|P{index:02d}|{count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
