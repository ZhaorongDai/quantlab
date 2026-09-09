#!/usr/bin/env python3
"""Enumerate every SENTENCE that talks about the failure manifest AND about run scope.

This is the D-18 documentation gate, and it is deliberately built the OTHER way
round from the scanner it replaces.  `manifest_semantics_scan.py` matched a list
of eleven KNOWN-BAD literals; that list was copied out of a verification report
and was never an enumeration, so three rounds of "close the loop" each fixed the
sites on the list and left the ones nobody had thought of.  Four recurrences of
one defect, all of them the same mechanism: the word list came from a list of
known-bad sites.

So this tool does not decide anything.  It INTERSECTS two SEMANTIC CLASSES --
`MANIFEST_TERMS` (what object is being talked about) and `RUNSCOPE_TERMS` (what
run scope is being claimed) -- over normalised sentence text, and prints every
sentence that hits both.  That output is a WORKSHEET.  The verdict is a human
reading each sentence and writing a row in `manifest-sentence-ledger.tsv`.  The
ORDER is the whole point: enumerate the candidate set first, adjudicate second,
and only then may a literal word list be derived from the STALE verdicts.

Scope is stated in the open (`SCOPE_ROOTS`, `SCOPE_FILES`) rather than buried in
a filter.  This tool and its two TSVs live under `.planning/phases/`, which is
BY CONSTRUCTION outside that scope, so the audit cannot match itself.  That is
deliberate, not an oversight: the ledger quotes the offending sentences verbatim
and would otherwise be its own biggest source of candidates.

Usage:

    manifest_sentence_audit.py --report   # worksheet: digest<TAB>path<TAB>sentence
    manifest_sentence_audit.py --check    # gate (default): exit 1 on any finding
"""

from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from pathlib import Path

# Scope, in the open. `.md` and `.py` files under these roots, plus these two
# exact files. `.planning/phases/` and `.planning/quick/` are dated historical
# records and are outside scope by construction (see module docstring).
SCOPE_ROOTS = ("quantlab/", "example/", "tests/")
SCOPE_FILES = (".planning/ROADMAP.md", ".planning/STATE.md")

# These two lists exist to CAST THE NET WIDE, and nothing else. They carry no
# verdict: adjudication lives in `manifest-sentence-ledger.tsv`, one row per
# sentence, written by a human who read it. Treating either list as "the list of
# known-bad wordings" is exactly how this defect recurs a fifth time.
#
# Only SEMANTIC-CLASS terms belong here. A fragment lifted out of one particular
# sentence is not a semantic class -- it is the old "derive the word list from
# the known-bad sites" habit wearing a new hat. A draft of this file carried
# `lie about` in RUNSCOPE_TERMS; it is a shard of one line in
# `quantlab/base/acquisition.py`, not a run-scope concept, and measurement
# confirmed it was entirely redundant (identical candidate set, identical
# per-file counts, no file dropped out of coverage). It was removed. Before
# adding a term, ask: does this name a CLASS of meaning, or a sentence I happen
# to have just read? The latter never goes in.
MANIFEST_TERMS = [
    r"_failures\.json",
    r"failure manifest",
    r"failures manifest",
    r"manifest",
    r"失败清单",
    r"FAILURE_MANIFEST_NAME",
    r"read_failure_manifest",
    r"result\.failures",
]

RUNSCOPE_TERMS = [
    r"last run",
    r"latest run",
    r"previous run",
    r"this run",
    r"per-run",
    r"per run",
    r"each run",
    r"every run",
    r"cross-run",
    r"run's",
    r"一次\s*run",
    r"上一次",
    r"上一轮",
    r"上次",
    r"本轮",
    r"每次\s*run",
    r"最近",
    r"overwrit",
    r"覆盖重写",
    r"snapshot",
    r"快照",
    r"跨\s*run",
    r"durable",
    r"耐久",
    r"accumulat",
    r"累积",
    r"折回",
]

MANIFEST_RE = [re.compile(p, re.I) for p in MANIFEST_TERMS]
RUNSCOPE_RE = [re.compile(p, re.I) for p in RUNSCOPE_TERMS]

VERDICTS = ("OK", "NA", "STALE")

LEDGER_PATH = Path(__file__).with_name("manifest-sentence-ledger.tsv")

# A block ends at a blank line, and a new block starts BEFORE any markdown
# structural line: list item, table row, ATX heading, code fence. Without this a
# whole table or bullet list would fold into one "sentence".
_STRUCTURE_RE = re.compile(r"^\s*(?:[-*+]\s|\d+[.)]\s|\||#|```|~~~)")

# Sentence terminators come in two flavours and the rules DIFFER. Full-width
# `。！？；` terminate unconditionally -- Chinese does not put a space after
# them, so requiring trailing whitespace would mean never splitting. Half-width
# `.!?;` terminate ONLY before whitespace or end-of-string. That restriction is
# load-bearing, not an optimisation: without it `` `_failures.json` `` is cut in
# half by its own dot, and `_failures\.json` -- the single most important entry
# in MANIFEST_TERMS -- stops matching anywhere.
_SPLIT_RE = re.compile(r"(?<=[。！？；])|(?<=[.!?;])(?=\s|$)|\|")


def repo_root() -> Path:
    out = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return Path(out)


def scoped_files() -> list[str]:
    """Tracked `.md`/`.py` files inside SCOPE_ROOTS, plus SCOPE_FILES."""
    out = subprocess.run(
        ["git", "ls-files"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    files = [line for line in out.splitlines() if line]
    keep: list[str] = []
    for path in files:
        if not (path.endswith(".md") or path.endswith(".py")):
            continue
        if path.startswith(SCOPE_ROOTS) or path in SCOPE_FILES:
            keep.append(path)
    return sorted(keep)


def segment(raw: str) -> list[str]:
    """Split source text into sentences, normalising away line and literal seams."""
    blocks: list[list[str]] = [[]]
    for line in raw.splitlines():
        if not line.strip():
            blocks.append([])
            continue
        if _STRUCTURE_RE.match(line) and blocks[-1]:
            blocks.append([])
        blocks[-1].append(line)

    sentences: list[str] = []
    for block in blocks:
        if not block:
            continue
        # Every run of whitespace to one space, so a sentence broken over
        # several lines reads as one sentence.
        flat = re.sub(r"\s+", " ", " ".join(block)).strip()
        # Stitch adjacent string-literal seams shut, so Python's implicit
        # concatenation cannot hide a phrase in the middle of a sentence.
        joined = re.sub(r"(['\"])\s*\1", "", flat)
        for piece in _SPLIT_RE.split(joined):
            piece = piece.strip()
            if piece:
                sentences.append(piece)
    return sentences


def is_candidate(sentence: str) -> bool:
    if not any(p.search(sentence) for p in MANIFEST_RE):
        return False
    return any(p.search(sentence) for p in RUNSCOPE_RE)


def digest_of(path: str, sentence: str) -> str:
    payload = (path + "\x00" + sentence).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:12]


def candidates() -> list[tuple[str, str, str]]:
    """Return sorted `(digest, path, sentence)` for every candidate sentence."""
    root = repo_root()
    found: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str]] = set()
    for path in scoped_files():
        # Strict utf-8 on purpose: a decode error in a tracked .md/.py file is
        # itself a finding, not something to swallow.
        raw = (root / path).read_text(encoding="utf-8")
        for sentence in segment(raw):
            if not is_candidate(sentence):
                continue
            key = (path, sentence)
            if key in seen:
                continue
            seen.add(key)
            found.append((digest_of(path, sentence), path, sentence))
    found.sort(key=lambda row: (row[1], row[0]))
    return found


def load_ledger() -> dict[str, tuple[str, str, str]]:
    """`digest -> (verdict, path, sentence)` from the adjudication ledger."""
    if not LEDGER_PATH.exists():
        return {}
    ledger: dict[str, tuple[str, str, str]] = {}
    lines = LEDGER_PATH.read_text(encoding="utf-8").splitlines()
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 4:
            parts = parts + [""] * (4 - len(parts))
        digest, verdict, path, sentence = parts[0], parts[1], parts[2], parts[3]
        ledger[digest] = (verdict, path, sentence)
    return ledger


def cmd_report() -> int:
    for digest, path, sentence in candidates():
        print(f"{digest}\t{path}\t{sentence}")
    return 0


def cmd_check() -> int:
    rows = candidates()
    ledger = load_ledger()
    live = {digest for digest, _, _ in rows}

    unadjudicated: list[tuple[str, str, str]] = []
    stale: list[tuple[str, str, str]] = []
    for digest, path, sentence in rows:
        entry = ledger.get(digest)
        if entry is None:
            unadjudicated.append((digest, path, sentence))
        elif entry[0] == "STALE":
            stale.append((digest, path, sentence))

    # A digest that used to be adjudicated and no longer appears means the
    # sentence was deleted or quietly reworded. That is a RED, not a GREEN:
    # otherwise "make the finding disappear" is the cheapest way to pass.
    vanished = [(d, ledger[d][1]) for d in ledger if d not in live]
    vanished.sort()

    for digest, path, sentence in stale:
        print(f"STALE\t{digest}\t{path}\t{sentence}")
    for digest, path, sentence in unadjudicated:
        print(f"UNADJUDICATED\t{digest}\t{path}\t{sentence}")
    for digest, path in vanished:
        print(f"VANISHED\t{digest}\t{path}")

    if stale or unadjudicated or vanished:
        print(
            f"AUDIT RED: {len(stale)} STALE, "
            f"{len(unadjudicated)} UNADJUDICATED, {len(vanished)} VANISHED"
        )
        return 1

    print(f"AUDIT GREEN: {len(rows)} candidates, {len(ledger)} adjudicated, 0 STALE")
    return 0


def main(argv: list[str]) -> int:
    action = argv[1] if len(argv) > 1 else "--check"
    if action == "--report":
        return cmd_report()
    if action == "--check":
        return cmd_check()
    print(f"usage: {Path(argv[0]).name} [--report|--check]", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
