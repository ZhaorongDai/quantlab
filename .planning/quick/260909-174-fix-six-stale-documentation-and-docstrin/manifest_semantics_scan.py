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
