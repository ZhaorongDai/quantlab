---
created: "2026-09-07T00:00:00.000Z"
title: Re-point stale former-root paths in Python docstrings and comments
area: docs
severity: minor
status: pending
---

# Re-point stale former-root paths in Python docstrings and comments

## Problem

Quick task `260907-sm2` moved the twelve flat top-level packages under `quantlab/`.
It scoped its prose rewrite to `README.md` and `example/*.md` **deliberately and
explicitly** — so every backticked path inside `.py` docstrings and comments still
names the pre-migration layout.

None of it is executable and no structural guard reaches any of it, so nothing fails
and no `must_have` is violated. That is exactly why it will not fix itself.

It matters here more than it would in most repositories. quantlab's comment density
is unusually high and the comments carry real contract: guard rationales, measured
evidence, decision records, and cross-references that a reader is meant to follow.
`quantlab/enums/data.py`'s `RAW_HIVE_KEYS` comment, for instance, tells the reader
which writer and which reader import it *by path* — the whole point of that note is
that a reader can go check. A path that no longer resolves makes the contract itself
misleading, not merely untidy.

## Measured, 2026-09-07, after `260907-sm2` landed

Three distinct shapes, counted live rather than taken from the verification report
(whose figure of 277 was measured before the six `example/*.md` fixes and over a
slightly different pattern):

```bash
# 1. Slash form -- `base/data.py`, `quantlab/dataset/_support/cleaning.py:clean_membership_panel()`
grep -rInoE '`(acquisition|base|config|dataset|dl_model|enums|factor|label|ml_model|my_ops|utils|vecbt)/[^`]*`' \
  --include="*.py" quantlab/ tests/ ./*.py | wc -l          # -> 161

# 2. Dotted form -- `base.data.BaseDataset`, `config.stock_kline_config`
grep -rInoE '`(acquisition|base|config|dataset|dl_model|enums|factor|label|ml_model|my_ops|utils|vecbt)\.[A-Za-z_]' \
  --include="*.py" quantlab/ tests/ ./*.py | wc -l          # -> 143

# 3. Statement form -- `from config import set_data_root`, `import acquisition.alpaca`
grep -rInoE '`(from|import) (acquisition|base|config|dataset|dl_model|enums|factor|label|ml_model|my_ops|utils|vecbt)[. ]' \
  --include="*.py" quantlab/ tests/ ./*.py | wc -l          # -> 3

# Union, deduplicated by line
grep -rInE '`(acquisition|base|config|dataset|dl_model|enums|factor|label|ml_model|my_ops|utils|vecbt)([/.]|` )' \
  --include="*.py" quantlab/ tests/ ./*.py | wc -l          # -> 289 lines
```

Spread across at least: `quantlab/ml_model/backend.py`, `quantlab/config/__init__.py`,
`quantlab/acquisition/{tiingo,universe,alpaca}.py`,
`quantlab/dataset/{stock,constituent,masking,backend,cleaning}.py`,
`quantlab/enums/data.py`, `quantlab/utils/nautilus.py`, plus `tests/` and the
repo-root scripts (`cal.py:7` and `train_model.py:8` carry commented-out imports).

## Solution

TBD, but the shape is a mechanical prefix, the same edit `260907-sm2` applied to the
docs — with two cautions that task learned the hard way:

- **Do not enumerate the terminator.** That task's sweep excluded `:` and missed
  `` `base/model.py:BaseModel` ``; the fix admitted `:` but still terminated at `(`,
  so `` `pkg/file.py:Symbol()` `` stayed invisible and the gate reported clean over
  stale paths. Match to the closing backtick (`[^`]*`) instead.
- **The dotted form (shape 2) is the larger and subtler half.** A slash form is
  obviously a path; `` `base.data.BaseDataset` `` reads like prose and is easy to
  skip. Any gate must cover both, or it will report clean over 143 references.

Apply the repo's comment-text discipline while rewriting: do not quote a retired path
alongside its replacement to "explain" the change, or a negative-grep gate over
retired names self-defeats.

## Deliberately not in scope

Non-backticked mentions in flowing prose. The three shapes above are the ones a
reader treats as a reference to follow; unquoted English mentions of the word
`config` or `base` are not, and a sweep that tried to catch them would be
unanchorable.

## Discovered

2026-09-07, as a non-blocking advisory in `260907-sm2`'s verification. Recorded
rather than folded into that task's gap closure, because the plan had scoped the
prose rewrite to `README.md` and `example/*.md` and quietly widening it during a
gap fix is the silent scope growth those gates exist to prevent.
