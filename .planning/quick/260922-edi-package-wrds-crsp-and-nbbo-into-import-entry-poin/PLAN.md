---
phase: quick-260922-edi
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - quantlab/acquisition/wrds/__init__.py
  - quantlab/acquisition/wrds/crsp.py
  - quantlab/acquisition/wrds/crsp_reference.py
  - quantlab/acquisition/wrds/taq.py
  - quantlab/dataset/crsp/__init__.py
  - quantlab/dataset/crsp/membership.py
  - quantlab/dataset/crsp/rebuild.py
  - quantlab/dataset/crsp/reference.py
  - quantlab/dataset/crsp/symbology.py
  - quantlab/dataset/crsp/tickers.py
  - quantlab/dataset/nbbo/__init__.py
  - quantlab/dataset/nbbo/resample.py
  - quantlab/acquisition/inspector.py
  - quantlab/base/backtest.py
  - quantlab/base/config.py
  - quantlab/base/data.py
  - quantlab/base/rebuild.py
  - quantlab/dataset/constituent.py
  - quantlab/dataset/masking.py
  - quantlab/dataset/stock.py
  - quantlab/factor/universe_filter.py
  - quantlab/utils/symbol_axis.py
  - scripts/ingest_wrds_crsp.py
  - scripts/ingest_wrds_taq.py
  - tests/conftest.py
  - tests/crsp_fixtures.py
  - tests/wrds_fixtures.py
  - tests/test_chunked_ingest.py
  - tests/test_crsp_constituent.py
  - tests/test_crsp_dataset.py
  - tests/test_crsp_identity.py
  - tests/test_crsp_membership.py
  - tests/test_crsp_rebuild.py
  - tests/test_crsp_rebuild_measurements.py
  - tests/test_crsp_reference_tables.py
  - tests/test_crsp_symbology.py
  - tests/test_crsp_ticker_sidecar.py
  - tests/test_crsp_tracer.py
  - tests/test_extensibility_contract.py
  - tests/test_ingest_wrds_crsp.py
  - tests/test_ingest_wrds_taq.py
  - tests/test_model_predict_panel.py
  - tests/test_nbbo_dataset.py
  - tests/test_nbbo_resampler.py
  - tests/test_no_identity_residue.py
  - tests/test_source_registry.py
  - tests/test_universe_mask.py
  - tests/test_wrds_crsp_acquisition.py
  - tests/test_wrds_taq_acquisition.py
  - tests/test_wrds_vendor_seam.py
  - example/universe.md
  - example/wrds_crsp.md
  - example/wrds_taq.md
  - CLAUDE.md
  - .planning/STATE.md
  - .planning/todos/pending/2026-09-07-no-ticker-rename-mapping-between-membership-history-and-pric.md
  - .planning/todos/pending/2026-09-20-delisting-return-is-unreachable-through-adjclose-so-labels-miss-it.md
autonomous: true
requirements: [QT-260922-edi]

estimate:
  tokens: 150000
  raw_tokens: 150000
  tasks: 3
  confidence: low

must_haves:
  truths:
    - "`from quantlab.acquisition.wrds import WRDS_SOURCE`, `from quantlab.dataset.crsp import CrspStockDataset` and `from quantlab.dataset.nbbo import NbboPanelDataset` are spelled EXACTLY as they are today, and resolve to the same objects."
    - "The three import orders measured before this plan still enumerate every vendor in a fresh interpreter: taq-first, wrds-first and registry-first all print `['alpaca', 'tiingo', 'wrds']` with no traceback."
    - "Every one of the 12 moved files carries its git history: `git log --follow` on each new path reaches commits made under the old path."
    - "Zero stale references to the pre-move module paths remain anywhere in `quantlab/`, `tests/`, `scripts/`, `example/` or `CLAUDE.md` — not in imports, not in path strings, not in prose."
    - "The set of dangling `quantlab/**.py` path references in code and living docs is byte-for-byte the same 14-line pre-existing set it is today — no new one, and none of the 14 repaired by accident."
    - "The test suite fails exactly the same tests it fails today (expected: 55, pre-existing, D-03.11-12-A). Not one NEW failure."
    - "`quantlab/acquisition/wrds/__init__.py`'s docstring describes the package shape that now exists, states in its own words what runtime property the move traded away, and names no pre-move file path."
    - "`tests/test_wrds_vendor_seam.py` says, in the test that scans `taq.py`, whether it is asserting a SOURCE-TEXT claim or a runtime one."
    - "`CLAUDE.md`'s `__init__.py` convention text describes the three entry-point packages and still states that `quantlab/acquisition/__init__.py` is 0 bytes."
  artifacts:
    - quantlab/acquisition/wrds/__init__.py
    - quantlab/dataset/crsp/__init__.py
    - quantlab/dataset/nbbo/__init__.py
    - tests/test_wrds_vendor_seam.py
    - tests/test_crsp_tracer.py
    - CLAUDE.md
  key_links:
    - "`registry.py` bottom import (`from quantlab.acquisition import wrds as _wrds`, module-object form) -> `quantlab/acquisition/wrds/__init__.py` -> `from quantlab.acquisition.wrds import crsp, taq`"
    - "`quantlab/acquisition/wrds/crsp.py` -> `from quantlab.acquisition.wrds import taq as _wrds` -> `_wrds.WrdsSession.shared()` at CALL time -> patched by `tests/conftest.py` as the dotted string `quantlab.acquisition.wrds.taq.WrdsSession`"
    - "`quantlab/base/backtest.py` -> `from quantlab.dataset.crsp.tickers import CrspTickerLookup` -> now runs `quantlab/dataset/crsp/__init__.py` (measured +0.99s / +196 modules, accepted)"
---

<objective>
Turn three module families into packages whose `__init__.py` IS the original entry file,
with the `X_` prefix stripped from the submodules. 12 files move by `git mv`. The
entry-point imports operators and callers actually write
(`from quantlab.acquisition.wrds import WRDS_SOURCE`,
`from quantlab.dataset.crsp import CrspStockDataset`,
`from quantlab.dataset.nbbo import NbboPanelDataset`) are spelled identically before and
after; only submodule imports change shape
(`quantlab.dataset.crsp_tickers` -> `quantlab.dataset.crsp.tickers`).

This is a decided refactor, not a proposal. The scope, the file list, the absence of
compatibility shims and the accepted import-cost regression were settled by the operator
before this plan; nothing below re-opens them.

Purpose: `quantlab/acquisition/` and `quantlab/dataset/` currently spell one subsystem as
a flat cluster of sibling files whose shared prefix is the only thing grouping them. After
this change the grouping is the directory, the entry point is the package, and adding a
fourth CRSP submodule is a file rather than a naming convention.

Output: 12 moved files with history preserved, 200 references rewritten across 52 files,
a rewritten `wrds` package docstring that tells the truth about what the move cost, two
seam tests that say which kind of claim they are making, updated living docs, and a
suite whose failure list is unchanged.
</objective>

<execution_context>
@~/.claude/gsd-core/workflows/execute-plan.md
@~/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@CLAUDE.md
@quantlab/acquisition/wrds.py
@quantlab/acquisition/registry.py
@tests/test_wrds_vendor_seam.py
@tests/test_crsp_tracer.py

## The move, in full (the ONLY 12 files that move)

```
quantlab/acquisition/wrds.py                -> quantlab/acquisition/wrds/__init__.py
quantlab/acquisition/wrds_crsp.py           -> quantlab/acquisition/wrds/crsp.py
quantlab/acquisition/wrds_crsp_reference.py -> quantlab/acquisition/wrds/crsp_reference.py
quantlab/acquisition/wrds_taq.py            -> quantlab/acquisition/wrds/taq.py

quantlab/dataset/crsp.py                    -> quantlab/dataset/crsp/__init__.py
quantlab/dataset/crsp_membership.py         -> quantlab/dataset/crsp/membership.py
quantlab/dataset/crsp_rebuild.py            -> quantlab/dataset/crsp/rebuild.py
quantlab/dataset/crsp_reference.py          -> quantlab/dataset/crsp/reference.py
quantlab/dataset/crsp_symbology.py          -> quantlab/dataset/crsp/symbology.py
quantlab/dataset/crsp_tickers.py            -> quantlab/dataset/crsp/tickers.py

quantlab/dataset/nbbo.py                    -> quantlab/dataset/nbbo/__init__.py
quantlab/dataset/nbbo_resample.py           -> quantlab/dataset/nbbo/resample.py
```

These are the only `X.py` + `X_*.py` families in `acquisition/` and `dataset/`, verified by
enumeration. `sql_volume.py` and `session_calendar.py` are single modules, not families.

**Nothing else is renamed.** Not `example/wrds_crsp.md`, not `example/wrds_taq.md`, not
`tests/test_wrds_taq_acquisition.py`, not `tests/crsp_fixtures.py`, not
`tests/wrds_fixtures.py`, not the `wrds_crsp_sp500_1d.zarr` data-store names, not the
`.crsp_tickers.json` sidecar constant. Those names contain a moved module's spelling but
are not module paths.

`quantlab/acquisition/__init__.py` and `quantlab/dataset/__init__.py` stay 0 bytes.
`quantlab/acquisition/registry.py:728-737` documents in prose why the acquisition one must
stay empty (a non-empty package `__init__` would run on every
`import quantlab.acquisition.universe` and erode the volume guard's structural arm
silently). Nothing in this plan touches it.

## Measured facts — established before this plan. DO NOT RE-MEASURE.

- **No import cycle results.** The wrds half was simulated in a scratch copy and all three
  import orders were run in fresh interpreters with the editable-install finder removed:
  `quantlab.acquisition.wrds.taq` first, `quantlab.acquisition.wrds` first, and
  `quantlab.acquisition.registry` first. All three printed `['alpaca', 'tiingo', 'wrds']`
  with no traceback. It holds because `from package import submodule` is safe during
  partial init, and `registry.py`'s bottom import binds the MODULE OBJECT rather than an
  attribute.
- **What the refactor does change, measured:** `import <a wrds provider>` today does NOT
  load the registry or `dataset.crsp` (both `False`, 1457 modules, 0.87s). After the
  refactor both are `True` (1600 modules, 0.85s). The runtime property "the providers stay
  free of the registry" is genuinely lost; the time cost is nil and the module cost
  is +143.
- **`base/backtest.py` cost:** it imports `CrspTickerLookup`, today from a module with zero
  `quantlab` imports. After the refactor that import runs the whole `crsp` package
  `__init__`: measured incremental cost on top of the already-3.13s/3118-module backtest
  import is **+0.99s / +196 modules**. Accepted by the operator.
- **No dataset submodule imports its package top level**, so nothing there can cycle.
- **Scope, re-verified while writing this plan:** **200 references across 52 files** in
  `quantlab/` + `tests/` + `scripts/` + `example/` + `CLAUDE.md`, measured with the Task 3
  sweep. They partition with zero overlap and zero leakage into: **99** wrds-family
  references across 29 files (Task 1's half-pattern), **89** dataset-family references
  across 33 files (Task 2's half-pattern), and **12** bare prose spellings of `crsp.py` /
  `wrds.py` that neither half-pattern reaches and only the Task 3 full sweep catches. Some
  files appear in more than one group. `CLAUDE.md` contributes 0 of the 200: it predates
  these modules entirely and names none of them — its only
  relevant line is the `__init__.py` convention at :176. Do not go hunting for CLAUDE.md
  component-table entries naming the old paths; there are none.

## The dangerous part

The import statements are safe: a mistake is an immediate `ImportError`. The string-form
references — `Path(...) / "wrds_taq.py"`, doc prose, `monkeypatch.setattr` dotted targets,
grep/ast assertions inside tests — are not. A miss becomes a `FileNotFoundError`, or worse
a source scan that silently reads the wrong file and passes. Both gates in Task 3 exist
for this and only this.

Known string-form references that a naive import-only rewrite will miss, named here so
they cannot be forgotten:

| Location | Today | Kind |
|---|---|---|
| `tests/conftest.py:156` | `sys.modules.get("quantlab.acquisition.wrds_taq")` | dotted string, autouse fixture |
| `tests/conftest.py:1228,1263` | `importlib.util.find_spec("quantlab.acquisition.wrds_taq")` | dotted string |
| `tests/conftest.py:1230,1265` | `monkeypatch.setattr("quantlab.acquisition.wrds_taq.WrdsSession", ...)` | dotted patch target |
| `tests/test_wrds_vendor_seam.py:46` | `REPO_ROOT / "quantlab" / "acquisition" / "wrds_taq.py"` | `Path` construction |
| `tests/test_wrds_taq_acquisition.py:33` | same shape | `Path` construction |
| `tests/test_crsp_tracer.py:41` | `REPO_ROOT / "quantlab" / "acquisition" / "wrds_crsp.py"` | `Path` construction |
| `tests/test_crsp_identity.py:1970` | multi-line `Path` ending `/ "crsp.py"` | `Path` construction, NOT caught by the line-based sweep |
| `tests/test_wrds_vendor_seam.py:235-236,260-261,266` | forbidden-prefix / parametrize literals | ast-scan string literals |
| `tests/test_crsp_tracer.py:256,264` | `node.module == "quantlab.acquisition.wrds_taq"` and the failure message | ast-scan string literals |

## Documentation policy — decided, apply it as written

| Path | Action | Reason |
|---|---|---|
| `example/universe.md`, `example/wrds_crsp.md`, `example/wrds_taq.md` | **UPDATE** | living docs |
| `CLAUDE.md:176` | **UPDATE** | the convention this task deliberately changes |
| `.planning/STATE.md:308` | **UPDATE** the `quantlab/dataset/crsp_tickers.py:CrspTickerLookup` pointer only | a live-code pointer inside a historical record |
| `.planning/todos/pending/2026-09-07-...md:93` | **UPDATE** the `crsp_symbology.py` path | pending todo pointing at live code |
| `.planning/todos/pending/2026-09-20-...md:12` | **UPDATE** the `quantlab/dataset/crsp.py` path | pending todo pointing at live code |
| `.planning/phases/**` (71 files, 529 matches) | **DO NOT TOUCH** | historical phase records. A SUMMARY saying "modified `quantlab/dataset/crsp_tickers.py`" was TRUE when written; rewriting it falsifies the audit trail. Its absence from the sweep is correct, not a miss. |
| `.planning/WINDOWS.md` | **DO NOT TOUCH** | every entry the sweep matches (20, 22, 23, 25, 28) is status `fixed` — a closed record of a past deviation, same class as a phase SUMMARY. The one `open` entry (29) names a `.planning/phases/**` doc, not moved code. |
| `.planning/ROADMAP.md` | **DO NOT TOUCH** | every match is completed-phase narrative (`- [x] 03.11-07-PLAN.md — ...`) or the 03.11 W-slice plan text, all historical |
| `.planning/research/*.md` | **DO NOT TOUCH** | research for phases that already shipped; already stale in content (line-number citations against pre-migration code) |

## Constraints

- `uv` for everything. Test command:
  `uv run pytest -q --ignore=tests/test_factor_hierarchy.py --ignore=tests/test_crsp_rebuild_measurements.py`
- **The suite has 55 KNOWN pre-existing failures** (`ingest_*`/CLI path assumptions, filed
  as `D-03.11-12-A`; `.planning/WINDOWS.md:46` records that the prose describing them is
  slightly off but the per-file counts are right). The done-criterion is
  "same failures as the BEFORE baseline, no NEW ones", NOT green. Do not chase them. If the
  BEFORE baseline measured in this worktree is not 55, use the MEASURED number and say so.
- macOS: `tests/conftest.py` sets `OMP_NUM_THREADS=1` before any import (torch+xgboost
  libomp). Unchanged by this plan.
- **Stale bytecode**: this repo was burned by it (gap G-03.11-4,
  `tests/test_stale_bytecode_lesson.py`). After moving files, ALWAYS clear caches before a
  verifying run:
  `find . -path ./.venv -prune -o -name __pycache__ -type d -print0 | xargs -0 rm -rf`
- `git mv` for every move so history survives. Never `rm` + `Write`.
- No backward-compat shims, no deprecation aliases, no re-export stubs at the old paths.
- Commit with explicit paths. Never `git add -A` / `git add .` / `git commit -a`, and never
  `git stash`.
- **Every command below is relative to the repo root.** None may be rewritten to an
  absolute `/Users/...` path: execution happens in an isolated git worktree, and an
  absolute main-tree path would exercise UNCHANGED code and report a false green.
- The sweep commands scan `quantlab tests scripts example CLAUDE.md` and deliberately do
  NOT scan `.planning/`. This plan file itself spells the pre-move paths many times; it is
  outside the scanned set, so it cannot invalidate its own gates.

## The rewrite table — apply longest-match first

Dotted module form (also valid for the `/` path form, substituting `/` for `.`):

| From | To |
|---|---|
| `quantlab.acquisition.wrds_crsp_reference` | `quantlab.acquisition.wrds.crsp_reference` |
| `quantlab.acquisition.wrds_crsp` | `quantlab.acquisition.wrds.crsp` |
| `quantlab.acquisition.wrds_taq` | `quantlab.acquisition.wrds.taq` |
| `quantlab.dataset.crsp_membership` | `quantlab.dataset.crsp.membership` |
| `quantlab.dataset.crsp_rebuild` | `quantlab.dataset.crsp.rebuild` |
| `quantlab.dataset.crsp_reference` | `quantlab.dataset.crsp.reference` |
| `quantlab.dataset.crsp_symbology` | `quantlab.dataset.crsp.symbology` |
| `quantlab.dataset.crsp_tickers` | `quantlab.dataset.crsp.tickers` |
| `quantlab.dataset.nbbo_resample` | `quantlab.dataset.nbbo.resample` |

Entry points are UNCHANGED: `quantlab.acquisition.wrds`, `quantlab.dataset.crsp`,
`quantlab.dataset.nbbo` keep their exact spelling everywhere they appear as an import.

Entry-file PATH strings do change: `quantlab/acquisition/wrds.py` ->
`quantlab/acquisition/wrds/__init__.py`, `quantlab/dataset/crsp.py` ->
`quantlab/dataset/crsp/__init__.py`, `quantlab/dataset/nbbo.py` ->
`quantlab/dataset/nbbo/__init__.py`.

The seven `from <package> import <module>` lines need their own handling because the
rewrite table's dotted form does not reach them:

| File:line | From | To |
|---|---|---|
| `quantlab/acquisition/wrds.py:32` | `from quantlab.acquisition import wrds_crsp, wrds_taq` | `from quantlab.acquisition.wrds import crsp, taq` |
| `quantlab/acquisition/wrds_crsp.py:57` | `from quantlab.acquisition import wrds_taq as _wrds` | `from quantlab.acquisition.wrds import taq as _wrds` |
| `quantlab/acquisition/wrds_crsp.py:31` | the same line quoted in prose | same |
| `tests/test_wrds_crsp_acquisition.py:50` | `from quantlab.acquisition import wrds_crsp, wrds_taq` | `from quantlab.acquisition.wrds import crsp, taq` |
| `tests/test_source_registry.py:430` | `from quantlab.acquisition import wrds_crsp` | `from quantlab.acquisition.wrds import crsp` |
| `tests/test_crsp_tracer.py:264` | the same line quoted in a failure message | same |
| `tests/test_crsp_membership.py:603` | `from quantlab.dataset import crsp_membership` | `from quantlab.dataset.crsp import membership` |

`quantlab/acquisition/registry.py:758` (`from quantlab.acquisition import wrds as _wrds`)
is **UNCHANGED** — it already binds the package entry point in the module-object form, and
that form is exactly what keeps the cycle closed.

Inside the packages, use ABSOLUTE imports (`from quantlab.acquisition.wrds import taq as
_wrds`), never relative (`from . import taq`). Two reasons: the repo convention is full
dotted paths, and the ast scans in `tests/test_wrds_vendor_seam.py` and
`tests/test_crsp_tracer.py` read `node.module`, which is `None` for a relative import — a
relative spelling would silently evade both gates.

In PROSE (docstrings, comments, `example/*.md`), spell a submodule relative to its package:
`wrds/taq.py`, `crsp/tickers.py`, `nbbo/resample.py`. Do not leave a bare `taq.py` (which
reads as a top-level module) and do not leave the pre-move spelling anywhere, including in
sentences explaining what the file used to be called. Git history carries the old names; the
sweep in Task 3 is a hard zero.

## Bulk-edit discipline

200 references across 52 files: use scripted, anchored `s///g` substitutions, not 52 manual
passes. Rules:
- Count matches before and after each scripted pass and confirm the delta is what you
  intended. A pass that changes a different number of lines than it matched is a bug.
- NEVER use `sed` range deletes on any file. Only targeted substitutions.
- macOS `sed` is BSD: use `sed -i ''` or, preferably, `perl -pi -e`.
- Hand-edit, do not script, the six files that need judgement:
  `quantlab/acquisition/wrds/__init__.py`, `quantlab/acquisition/wrds/taq.py`,
  `quantlab/acquisition/wrds/crsp.py`, `tests/test_wrds_vendor_seam.py`,
  `tests/test_crsp_tracer.py`, `CLAUDE.md`.
</context>

<tasks>

<task type="tracer">
  <name>Task 1: move the wrds family end-to-end and prove the package shape holds</name>
  <files>quantlab/acquisition/wrds/__init__.py, quantlab/acquisition/wrds/crsp.py, quantlab/acquisition/wrds/crsp_reference.py, quantlab/acquisition/wrds/taq.py, quantlab/base/data.py, quantlab/dataset/crsp.py, quantlab/dataset/crsp_reference.py, quantlab/dataset/crsp_symbology.py, quantlab/dataset/stock.py, scripts/ingest_wrds_crsp.py, scripts/ingest_wrds_taq.py, tests/conftest.py, tests/crsp_fixtures.py, tests/wrds_fixtures.py, tests/test_chunked_ingest.py, tests/test_crsp_constituent.py, tests/test_crsp_dataset.py, tests/test_crsp_identity.py, tests/test_crsp_reference_tables.py, tests/test_crsp_tracer.py, tests/test_ingest_wrds_crsp.py, tests/test_ingest_wrds_taq.py, tests/test_nbbo_dataset.py, tests/test_source_registry.py, tests/test_wrds_crsp_acquisition.py, tests/test_wrds_taq_acquisition.py, tests/test_wrds_vendor_seam.py</files>
  <action>
This is the tracer: one family, all the way through — move, imports, string references,
seam tests, three fresh-interpreter import orders. It touches every layer the whole task
touches (package `__init__` as entry point, sibling submodule import, dotted patch target,
ast source scan, registry bottom import). If the shape is wrong, it is wrong after 4 moved
files instead of 12.

**0. Capture the BEFORE baseline first, before touching anything.**
`mkdir -p /tmp/quantlab-pkg-refactor` (a scratch directory for this run only; never
committed, and not a repo path, so it carries no false-green risk). Then:
  - `uv run pytest -q --ignore=tests/test_factor_hierarchy.py --ignore=tests/test_crsp_rebuild_measurements.py 2>&1 | tail -80 > /tmp/quantlab-pkg-refactor/before.txt`
  - extract the sorted list of failing node IDs into
    `/tmp/quantlab-pkg-refactor/before-ids.txt` and the final `N failed, M passed` summary
    line into `/tmp/quantlab-pkg-refactor/before-summary.txt`.
  - Report the failure count. Expected 55; if it differs, that MEASURED number is the
    baseline for the rest of this plan and must be stated in the SUMMARY.
  - Also capture the dangling-path baseline with the script in Task 3's `<verify>`, into
    `/tmp/quantlab-pkg-refactor/before-dangling.txt`. Expected: 14 lines, all pre-existing
    (`factor_polars.py`, `xxx.py` placeholders, `quantlab/vecbt/bt.py`, etc.).

**1. Move the four files with `git mv`:**
```
mkdir -p quantlab/acquisition/wrds
git mv quantlab/acquisition/wrds.py                quantlab/acquisition/wrds/__init__.py
git mv quantlab/acquisition/wrds_crsp.py           quantlab/acquisition/wrds/crsp.py
git mv quantlab/acquisition/wrds_crsp_reference.py quantlab/acquisition/wrds/crsp_reference.py
git mv quantlab/acquisition/wrds_taq.py            quantlab/acquisition/wrds/taq.py
```
Do not create a separate empty `quantlab/acquisition/wrds/__init__.py` — the moved
`wrds.py` IS it.

**2. Rewrite the wrds-family references repo-wide** per the rewrite table in `<context>`,
across `quantlab tests scripts example`. This is the scripted bulk pass. The wrds-half
reference set is 99 matches in 29 files; enumerate it before and after with:
```
WRDS_PAT='quantlab[/.]acquisition[/.]wrds_(crsp_reference|crsp|taq)|(^|[^A-Za-z0-9_])(wrds_crsp_reference|wrds_crsp|wrds_taq)\.py|acquisition import [^\n]*(wrds_crsp_reference|wrds_crsp|wrds_taq)|quantlab/acquisition/wrds\.py'
grep -rnE "$WRDS_PAT" quantlab tests scripts example CLAUDE.md --exclude-dir=__pycache__
```
Leave the dataset-family spellings alone; Task 2 owns them. That means
`quantlab/acquisition/wrds/crsp_reference.py:56` keeps
`from quantlab.dataset.crsp_reference import (...)` for now — correct at this point, and
Task 2 changes it. Expect that one line to be edited twice across the two tasks.

**3. Hand-edit `quantlab/acquisition/wrds/__init__.py`:**
  - `from quantlab.acquisition import wrds_crsp, wrds_taq` becomes
    `from quantlab.acquisition.wrds import crsp, taq`; the two body references become
    `taq.WrdsTaqNbboAcquisition` and `crsp.WrdsCrspDailyAcquisition`.
  - The two `from quantlab.dataset...` imports at :38-39 are UNCHANGED.
  - **Rewrite the module docstring.** Its current second and third paragraphs argue that
    the descriptor sits in a neutral module so "The providers therefore stay free of the
    registry" and "No import order can cycle". After the move the cycle claim still holds
    and the freedom claim does not. Do not delete the paragraph — the reasoning is
    valuable, it just has to become accurate. The rewritten docstring must say, in its own
    words and without naming any pre-move file path:
      (a) The descriptor lives in the package entry point, which imports the provider
          submodules. One WRDS account, several products, one descriptor — that part is
          unchanged, and it is still true that a registration inside a provider would have
          to name a sibling's classes.
      (b) No import order cycles, and why: `registry.py`'s bottom import binds the MODULE
          OBJECT, and `from package import submodule` is safe during partial init. Cite the
          measurement: three fresh interpreters, taq-first / wrds-first / registry-first,
          all printing `['alpaca', 'tiingo', 'wrds']`.
      (c) **What was traded away, plainly.** Importing any WRDS provider now runs this
          package `__init__`, so it loads the registry, `quantlab.dataset.crsp` and
          `quantlab.dataset.nbbo`. Before the move it did not. Give the measured numbers:
          1457 -> 1600 modules, 0.87s -> 0.85s. Say that the "providers stay free of the
          registry" property is a SOURCE-TEXT rule now, enforced by
          `tests/test_wrds_vendor_seam.py`, not a runtime fact.
      (d) The claim that the providers import each other not at all was ALREADY false
          before this move: the CRSP provider reaches the shared session through the `taq`
          module attribute. State the real, narrower rule: no provider imports the registry
          or this descriptor.

**4. Hand-edit `quantlab/acquisition/wrds/crsp.py`:** the module-attribute seam becomes
`from quantlab.acquisition.wrds import taq as _wrds`, still read as `_wrds.WrdsSession`
at call time, never bound by name. Update the four docstring passages at :21, :31, :33, :40
that name the pre-move layout and the pre-move patch target. The `quantlab/dataset/crsp.py`
mention at :21 belongs to Task 2.

**5. Hand-edit `quantlab/acquisition/wrds/taq.py`:** no import changes. Its docstring
paragraph at :29-36 names the pre-move descriptor path and argues the anti-cycle rule;
bring it in line with (c) and (d) above, keeping it short — the long version lives in the
package docstring.

**6. `tests/conftest.py`** — five dotted-string targets move to
`quantlab.acquisition.wrds.taq`:
`sys.modules.get(...)` at :156 and the `find_spec` guard + `monkeypatch.setattr` target in
each of `mock_wrds_session` (:1228-1232) and `mock_crsp_session` (:1263-1267).
**Behaviour change to record, not to fix:** `find_spec("quantlab.acquisition.wrds.taq")`
imports the PARENT package to find the submodule, so those two fixtures now pull in the
registry, `dataset.crsp` and `dataset.nbbo` at fixture-setup time; today's
`find_spec("quantlab.acquisition.wrds_taq")` only imports the 0-byte
`quantlab.acquisition`. `_close_shared_wrds_sessions` (autouse) uses `sys.modules.get`, so
it still adds no import and its docstring stays true. Add one sentence to each of the two
fixture docstrings recording the new `find_spec` side effect. Watch `isolated_registry`
and `tests/test_volume_guard.py` in the verify run — if anything there goes red, report it
rather than working around it.

**7. `tests/test_wrds_vendor_seam.py`** — this is the file the operator singled out:
  - `WRDS_TAQ_SOURCE` becomes `REPO_ROOT / "quantlab" / "acquisition" / "wrds" / "taq.py"`.
  - `test_wrds_taq_registers_nothing_and_imports_no_registry`: **decide and state in the
    docstring that this is a SOURCE-TEXT claim about `taq.py`, not a runtime claim about
    `sys.modules`.** The runtime property did not survive the move; the text rule did, and
    it is still worth enforcing because it is what keeps a registration from drifting back
    into a provider. Keep the `ast` walk; keep the forbidden set
    (`quantlab.acquisition.registry`, `quantlab.acquisition.wrds` and anything under
    either, which now also covers sibling submodules). ADD an assertion that every
    `ast.ImportFrom` in `taq.py` has `node.level == 0`, so the absolute-path scan cannot be
    evaded by a relative import — inside a package that is now a real possibility, and it
    was not before.
  - `test_registry_bottom_import_names_the_neutral_module`: rename to say "package entry
    point" rather than "neutral module". The `"quantlab.acquisition.wrds_taq" not in
    from_registry` assertion becomes: `registry.py` names the PACKAGE and no submodule of
    it.
  - `test_enumeration_survives_either_wrds_import_order`: the parametrize list moves to the
    new names and gains the third measured order —
    `["quantlab.acquisition.wrds.taq", "quantlab.acquisition.wrds",
    "quantlab.acquisition.registry"]`. Update the test name and docstring to say three
    orders. This turns the operator's one-off measurement into a permanent gate.
  - `live_session` (:118) and the helper-scrub test (:368) import from the new path.
  - Rewrite the module docstring's bullets at :10-20 to describe the package shape.

**8. `tests/test_crsp_tracer.py`:**
  - `WRDS_CRSP_SOURCE` becomes `REPO_ROOT / "quantlab" / "acquisition" / "wrds" / "crsp.py"`.
  - `test_wrds_crsp_reaches_the_session_only_through_the_wrds_taq_module`: rename to drop
    the pre-move spelling. The by-name check's `node.module` becomes
    `"quantlab.acquisition.wrds.taq"`; the module-import check's `node.module` becomes
    `"quantlab.acquisition.wrds"` with alias `"taq"`. Update the failure message and the
    docstring's quoted patch target.

**9. `tests/test_wrds_taq_acquisition.py:33`** — the `Path` construction moves to
`"wrds" / "taq.py"`.
  </action>
  <verify>
    <automated>find . -path ./.venv -prune -o -name __pycache__ -type d -print0 | xargs -0 rm -rf; for m in quantlab.acquisition.wrds.taq quantlab.acquisition.wrds quantlab.acquisition.registry; do echo "-- first import: $m"; uv run python -c "import $m; from quantlab.acquisition.registry import DataSourceRegistry; print(sorted(d.vendor for d in DataSourceRegistry.all()))" || exit 1; done</automated>
    <automated>uv run python -c "from quantlab.acquisition.wrds import WRDS_SOURCE; from quantlab.acquisition.registry import DataSourceRegistry; assert WRDS_SOURCE is DataSourceRegistry.get('wrds'); print('entry point intact')"</automated>
    <automated>for f in __init__ crsp crsp_reference taq; do git log --follow --oneline -- "quantlab/acquisition/wrds/$f.py" > /tmp/quantlab-pkg-refactor/hist.txt || exit 1; N=$(wc -l < /tmp/quantlab-pkg-refactor/hist.txt); printf '%-16s %s commits\n' "wrds/$f" "$N"; test "$N" -gt 1 || { echo "HISTORY LOST for wrds/$f.py"; exit 1; }; done</automated>
    <automated>WRDS_PAT='quantlab[/.]acquisition[/.]wrds_(crsp_reference|crsp|taq)|(^|[^A-Za-z0-9_])(wrds_crsp_reference|wrds_crsp|wrds_taq)\.py|acquisition import [^\n]*(wrds_crsp_reference|wrds_crsp|wrds_taq)|quantlab/acquisition/wrds\.py'; N=$(grep -rnE "$WRDS_PAT" quantlab tests scripts example CLAUDE.md --exclude-dir=__pycache__ | tee /tmp/quantlab-pkg-refactor/wrds-residue.txt | wc -l); echo "WRDS-FAMILY STALE REFERENCES REMAINING: $N (was 99, must now be 0)"; test "$N" -eq 0</automated>
    <automated>find . -path ./.venv -prune -o -name __pycache__ -type d -print0 | xargs -0 rm -rf; uv run pytest -q tests/test_wrds_vendor_seam.py tests/test_wrds_taq_acquisition.py tests/test_wrds_crsp_acquisition.py tests/test_crsp_tracer.py tests/test_source_registry.py tests/test_nbbo_dataset.py tests/test_volume_guard.py</automated>
  </verify>
  <done>
The four wrds files live under `quantlab/acquisition/wrds/` with `git log --follow`
reaching pre-move commits for each. All three fresh-interpreter import orders print
`['alpaca', 'tiingo', 'wrds']` with no traceback. `WRDS_SOURCE` is still the registered
descriptor, reached by the unchanged `from quantlab.acquisition.wrds import WRDS_SOURCE`.
The wrds-family stale-reference count is 0 (reported, from 99). The six named test files
pass. The package docstring states what the move traded away, with the measured numbers,
and names no pre-move path. `tests/test_wrds_vendor_seam.py` says explicitly which of its
claims is about source text and which is about runtime.
  </done>
</task>

<task type="auto">
  <name>Task 2: move the crsp and nbbo families</name>
  <files>quantlab/dataset/crsp/__init__.py, quantlab/dataset/crsp/membership.py, quantlab/dataset/crsp/rebuild.py, quantlab/dataset/crsp/reference.py, quantlab/dataset/crsp/symbology.py, quantlab/dataset/crsp/tickers.py, quantlab/dataset/nbbo/__init__.py, quantlab/dataset/nbbo/resample.py, quantlab/acquisition/inspector.py, quantlab/acquisition/wrds/crsp.py, quantlab/acquisition/wrds/crsp_reference.py, quantlab/acquisition/wrds/taq.py, quantlab/base/backtest.py, quantlab/base/config.py, quantlab/base/rebuild.py, quantlab/dataset/constituent.py, quantlab/dataset/masking.py, quantlab/factor/universe_filter.py, quantlab/utils/symbol_axis.py, scripts/ingest_wrds_crsp.py, tests/crsp_fixtures.py, tests/test_crsp_dataset.py, tests/test_crsp_identity.py, tests/test_crsp_membership.py, tests/test_crsp_rebuild.py, tests/test_crsp_rebuild_measurements.py, tests/test_crsp_reference_tables.py, tests/test_crsp_symbology.py, tests/test_crsp_ticker_sidecar.py, tests/test_extensibility_contract.py, tests/test_model_predict_panel.py, tests/test_nbbo_resampler.py, tests/test_no_identity_residue.py, tests/test_universe_mask.py</files>
  <action>
Same shape as Task 1, now that the shape is proven. The dataset half is simpler: no
submodule imports its own package at module level, so nothing here can cycle.

**1. Move the eight files with `git mv`:**
```
mkdir -p quantlab/dataset/crsp quantlab/dataset/nbbo
git mv quantlab/dataset/crsp.py            quantlab/dataset/crsp/__init__.py
git mv quantlab/dataset/crsp_membership.py quantlab/dataset/crsp/membership.py
git mv quantlab/dataset/crsp_rebuild.py    quantlab/dataset/crsp/rebuild.py
git mv quantlab/dataset/crsp_reference.py  quantlab/dataset/crsp/reference.py
git mv quantlab/dataset/crsp_symbology.py  quantlab/dataset/crsp/symbology.py
git mv quantlab/dataset/crsp_tickers.py    quantlab/dataset/crsp/tickers.py
git mv quantlab/dataset/nbbo.py            quantlab/dataset/nbbo/__init__.py
git mv quantlab/dataset/nbbo_resample.py   quantlab/dataset/nbbo/resample.py
```

**2. Rewrite the dataset-family references repo-wide** per the rewrite table, across
`quantlab tests scripts example`. 89 matches across 33 files. Enumerate before and after:
```
DS='crsp_membership|crsp_rebuild|crsp_reference|crsp_symbology|crsp_tickers|nbbo_resample'
DATASET_PAT="quantlab[/.]dataset[/.]($DS)|(^|[^A-Za-z0-9_])($DS)\.py|dataset import [^\n]*($DS)|quantlab/dataset/(crsp|nbbo)\.py"
grep -rnE "$DATASET_PAT" quantlab tests scripts example CLAUDE.md --exclude-dir=__pycache__
```

**3. The imports that must NOT change**, because they already name the entry point:
  - `quantlab/acquisition/wrds/__init__.py:38-39` — `from quantlab.dataset.crsp import
    CrspStockDataset`, `from quantlab.dataset.nbbo import NbboPanelDataset`
  - `quantlab/acquisition/inspector.py:544` — `from quantlab.dataset.crsp import
    TICKER_SIDECAR_SUFFIX`
  - `quantlab/dataset/crsp/rebuild.py:110` — `from quantlab.acquisition.wrds import
    WRDS_SOURCE`
  - `quantlab/dataset/crsp/tickers.py:165` — `from quantlab.dataset.crsp import
    TICKER_SIDECAR_SUFFIX`, which now imports the module's own parent package. It is
    function-local, so the parent is fully loaded by call time; the docstring at :160
    already explains why the import is function-local and that reasoning survives the move
    intact. Say so there rather than deleting it.
  - every `from quantlab.dataset.crsp import ...` / `from quantlab.dataset.nbbo import ...`
    in `tests/` and `scripts/`

**4. The imports that DO change inside the moved files:**
  - `quantlab/dataset/crsp/__init__.py` :95 `reference`, :96 `symbology` (module level);
    :485 and :769 `membership` (function level)
  - `quantlab/dataset/crsp/membership.py:62` -> `quantlab.dataset.crsp.reference`
  - `quantlab/dataset/nbbo/__init__.py:39` -> `quantlab.dataset.nbbo.resample`

**5. The callers outside the moved families:**
  - `quantlab/base/backtest.py:15` -> `from quantlab.dataset.crsp.tickers import
    CrspTickerLookup`. This is the measured +0.99s / +196 modules. Leave a one-line comment
    at that import recording the cost and that it is deliberate, so the next reader of a
    slow backtest import finds the answer at the import rather than by bisecting.
  - `quantlab/dataset/constituent.py:41-42`, `quantlab/dataset/masking.py:23`,
    `quantlab/acquisition/wrds/crsp_reference.py:56` (the line Task 1 left on the old
    dataset spelling), `scripts/ingest_wrds_crsp.py:112-113`

**6. `tests/test_crsp_identity.py`** — the multi-line `Path` at :1965-1971 ends
`/ "crsp.py"`. It becomes `/ "crsp" / "__init__.py"`. **Task 2's half-pattern above does
NOT match it** — the pattern needs the `"dataset"` component, which sits on a previous
line — so the step-2 count reaching 0 does not mean this line is done. It is named here for
that reason. Task 3's full sweep does catch it (on the bare `"crsp.py"` spelling), and if
both were missed `read_text` would raise `FileNotFoundError` and the test would error
loudly, which is the safe direction. Fix it here anyway.

**7. `tests/test_crsp_rebuild_measurements.py`** is on the `--ignore` list of the standard
test command, so a mistake in it will not show up in the suite run. Fix its imports and
verify it separately with `--collect-only` (in `<verify>`), so it is not left as a landmine.

**8. Prose-only mentions of the pre-move basenames** in
`tests/test_no_identity_residue.py` (:26, :59, :64, :70), `tests/test_extensibility_contract.py:40`,
`tests/test_chunked_ingest.py:2715`, `quantlab/base/config.py` (:146, :149),
`quantlab/utils/symbol_axis.py` (:13), `quantlab/base/rebuild.py:208`,
`quantlab/factor/universe_filter.py:36`, `quantlab/dataset/crsp/rebuild.py:43`,
`quantlab/acquisition/inspector.py:541`, `quantlab/dataset/crsp/tickers.py` (:73, :160):
rewrite to the package-relative spelling (`crsp/symbology.py`, `crsp/__init__.py`, ...).
These carry no runtime risk on their own; they are in scope because a doc that points at a
file that does not exist is how the next reader loses an hour.

`tests/test_no_identity_residue.py` and `tests/test_backtest_contracts.py` both discover
sources with `rglob("*.py")`, which descends into the new packages — they keep covering
every moved file with no change to the globs. Confirm this in the verify run rather than
assuming it.
  </action>
  <verify>
    <automated>find . -path ./.venv -prune -o -name __pycache__ -type d -print0 | xargs -0 rm -rf; uv run python -c "
from quantlab.dataset.crsp import CrspStockDataset, TICKER_SIDECAR_SUFFIX, SECURITY_FILTER_PRESETS
from quantlab.dataset.nbbo import NbboPanelDataset
from quantlab.dataset.crsp.tickers import CrspTickerLookup
from quantlab.dataset.crsp.membership import CrspMembership
from quantlab.dataset.crsp.reference import CrspReference
from quantlab.dataset.crsp.symbology import CrspSymbology
from quantlab.dataset.crsp.rebuild import CrspStoreRebuilder
from quantlab.dataset.nbbo.resample import NbboFilterPolicy, NbboResampler
print('dataset entry points and submodules intact')"</automated>
    <automated>for p in crsp/__init__ crsp/membership crsp/rebuild crsp/reference crsp/symbology crsp/tickers nbbo/__init__ nbbo/resample; do git log --follow --oneline -- "quantlab/dataset/$p.py" > /tmp/quantlab-pkg-refactor/hist.txt || exit 1; N=$(wc -l < /tmp/quantlab-pkg-refactor/hist.txt); printf '%-18s %s commits\n' "$p" "$N"; test "$N" -gt 1 || { echo "HISTORY LOST for quantlab/dataset/$p.py"; exit 1; }; done</automated>
    <automated>DS='crsp_membership|crsp_rebuild|crsp_reference|crsp_symbology|crsp_tickers|nbbo_resample'; DATASET_PAT="quantlab[/.]dataset[/.]($DS)|(^|[^A-Za-z0-9_])($DS)\.py|dataset import [^\n]*($DS)|quantlab/dataset/(crsp|nbbo)\.py"; N=$(grep -rnE "$DATASET_PAT" quantlab tests scripts example CLAUDE.md --exclude-dir=__pycache__ | tee /tmp/quantlab-pkg-refactor/dataset-residue.txt | wc -l); echo "DATASET-FAMILY STALE REFERENCES REMAINING: $N (was 89, must now be 0)"; test "$N" -eq 0</automated>
    <automated>uv run pytest -q --collect-only tests/test_crsp_rebuild_measurements.py | tail -3</automated>
    <automated>find . -path ./.venv -prune -o -name __pycache__ -type d -print0 | xargs -0 rm -rf; uv run pytest -q tests/test_crsp_dataset.py tests/test_crsp_identity.py tests/test_crsp_membership.py tests/test_crsp_rebuild.py tests/test_crsp_reference_tables.py tests/test_crsp_symbology.py tests/test_crsp_ticker_sidecar.py tests/test_crsp_tracer.py tests/test_crsp_constituent.py tests/test_nbbo_dataset.py tests/test_nbbo_resampler.py tests/test_no_identity_residue.py tests/test_backtest_contracts.py tests/test_universe_mask.py tests/test_extensibility_contract.py</automated>
  </verify>
  <done>
The eight crsp/nbbo files live under `quantlab/dataset/crsp/` and `quantlab/dataset/nbbo/`
with `git log --follow` reaching pre-move commits for each. Every entry-point import
(`from quantlab.dataset.crsp import CrspStockDataset`, `from quantlab.dataset.nbbo import
NbboPanelDataset`) is spelled exactly as before and still resolves. The dataset-family
stale-reference count is 0 (reported, from 89). `tests/test_crsp_rebuild_measurements.py`
collects cleanly despite sitting off the standard test command. The 15 named test files
pass, including the two `rglob`-based contract tests, which now descend into the new
packages with no change to their globs.
  </done>
</task>

<task type="auto">
  <name>Task 3: sweep to zero, update the docs, and prove no NEW test failure</name>
  <files>CLAUDE.md, example/universe.md, example/wrds_crsp.md, example/wrds_taq.md, .planning/STATE.md, .planning/todos/pending/2026-09-07-no-ticker-rename-mapping-between-membership-history-and-pric.md, .planning/todos/pending/2026-09-20-delisting-return-is-unreachable-through-adjclose-so-labels-miss-it.md</files>
  <action>
**1. Run the full sweep and drive it to zero.** Tasks 1 and 2 each cleared their own half;
this pass catches what the two half-patterns straddled and the bare prose spellings neither
enumerated:
```
OLD='wrds_crsp_reference|wrds_crsp|wrds_taq|crsp_membership|crsp_rebuild|crsp_reference|crsp_symbology|crsp_tickers|nbbo_resample'
PAT="quantlab[/.](acquisition|dataset)[/.]($OLD)|(^|[^A-Za-z0-9_])($OLD)\.py|(acquisition|dataset) import [^\n]*($OLD)|quantlab/(acquisition/wrds|dataset/(crsp|nbbo))\.py|(^|[^A-Za-z0-9_/])(wrds|crsp|nbbo)\.py"
EXC='"wrds" +/ +"crsp\.py"'
grep -rnE "$PAT" quantlab tests scripts example CLAUDE.md --exclude-dir=__pycache__ | grep -vE "$EXC" | wc -l
```
**Report the count.** Measured at 200 across 52 files while this plan was written: the 188
the two half-sweeps covered (99 + 89, no overlap), plus 12 bare prose spellings of
`crsp.py` / `wrds.py` that neither half-pattern reaches —
`quantlab/acquisition/inspector.py:541`, `quantlab/base/rebuild.py:208`,
`quantlab/dataset/crsp/rebuild.py:43`, `quantlab/dataset/crsp/tickers.py:73,160`,
`tests/test_crsp_identity.py:1954`, `tests/test_crsp_rebuild_measurements.py:570,656`,
`tests/test_no_identity_residue.py:26,59`, `tests/test_wrds_vendor_seam.py:204` and the
`/ "crsp.py"` line of `tests/test_crsp_identity.py:1970` that Task 2 owns. It must be **0**
when this task is done.

Two details of that command are load-bearing and were validated against both the pre-move
tree (200 matches) and a probe of the lines that will exist AFTER the move (0 matches):
  - the last alternative excludes a preceding `/`, so `wrds/taq.py`, `crsp/__init__.py` and
    `nbbo/resample.py` in prose do not trip it;
  - `EXC` excludes exactly one legitimate post-move spelling,
    `REPO_ROOT / "quantlab" / "acquisition" / "wrds" / "crsp.py"` in
    `tests/test_crsp_tracer.py`, where a quoted `"crsp.py"` is correct because the
    preceding quoted component is `"wrds"`. It is the ONLY exclusion; do not widen it. If a
    second line needs excluding, that is a signal the rewrite is wrong, not the filter.

**2. `example/*.md`** — living docs, update: `example/wrds_crsp.md` (13 matches),
`example/wrds_taq.md` (2), `example/universe.md` (1: the `quantlab/dataset/crsp.py`
reference in the `security_filter` paragraph). These are Chinese-language docs; match the
surrounding language and style and change only the paths. Do NOT rename the doc files
themselves.

**3. `CLAUDE.md:176`** — the convention this task deliberately breaks. It reads today:
"**No `__init__.py` re-exports:** Every package's `__init__.py` (`base/`, `dataset/`,
`factor/`, ...) is empty — all imports use full dotted paths ...". Rewrite it to state both
halves of the truth that now exists:
  - the layer packages (`base/`, `factor/`, `label/`, `dl_model/`, `ml_model/`, `my_ops/`,
    `utils/`, `enums/`, and `acquisition/` and `dataset/` THEMSELVES) still have empty
    `__init__.py` files and are still imported by full dotted path;
  - three packages are different — `quantlab/acquisition/wrds/`, `quantlab/dataset/crsp/`
    and `quantlab/dataset/nbbo/` — where `__init__.py` IS the entry module, not a re-export
    shim. Say what that buys (`from quantlab.dataset.crsp import CrspStockDataset` is one
    name for one subsystem, and submodules group by directory rather than by prefix) and
    what it costs (importing any submodule runs the entry module; measured +0.99s /
    +196 modules on `quantlab/base/backtest.py`);
  - `quantlab/acquisition/__init__.py` is still 0 bytes, and
    `quantlab/acquisition/registry.py:728-737` explains why it must stay that way.
Do not add a re-export list to any `__init__.py`. The distinction being adopted is "the
`__init__` IS the module", never "the `__init__` re-exports other modules".
CLAUDE.md contains no component/layer-table rows naming these modules — it predates them
and matches none of the sweep patterns. Line 176 is the only edit it needs; do not go
hunting.

**4. The three live `.planning` ledger entries** in the documentation-policy table in
`<context>`, and ONLY those three. Everything else under `.planning/` is history and stays
byte-for-byte as written — `.planning/phases/**` (71 files, 529 matches),
`.planning/WINDOWS.md`, `.planning/ROADMAP.md`, `.planning/research/*.md`. Their absence
from the sweep scope is deliberate, not an oversight; record that in the SUMMARY so a later
reader does not file it as a miss. Do not use `sed` range deletes on any planning document:
use an exact anchored replacement and confirm the file's line count is unchanged after.

**5. Run the full suite and compare against the BEFORE baseline** captured in Task 1, after
clearing bytecode caches. Report three things in the SUMMARY: the baseline failure count,
the after failure count, and the set difference of failing node IDs in BOTH directions
(each must be empty). A test that moved fail -> pass is as much a signal as one that moved
pass -> fail; report it rather than quietly banking it.

**6. Confirm no new dangling path reference.** The dangling-path check in `<verify>` is the
gate for the whole class of string-form misses, including the multi-line `Path`
constructions the line-based sweep cannot see. It is **not zero** today: there are 14
pre-existing dangling references (`quantlab/base/factor_polars.py`, the
`quantlab/factor/xxx.py` and `quantlab/dl_model/xxx.py` placeholders,
`quantlab/label/spot.py`, `quantlab/ingest_alpaca.py`, `quantlab/tests/test_page_ledger.py`,
`quantlab/acquisition/progress.py`, `quantlab/dataset/chunking.py`, `quantlab/vecbt/bt.py`).
The gate is that the AFTER list is IDENTICAL to the BEFORE list. Do not repair the
pre-existing 14 — they are out of scope, and repairing them inside this diff would hide a
regression.

**7. Commit and push.** Explicit paths only, never `git add -A` / `.` / `-a`, never
`git stash`. Suggested subject: `refactor: package wrds, crsp and nbbo into import entry
points`. The body should record that 12 files moved via `git mv`, that entry-point imports
are unchanged, and that "the providers stay free of the registry" became a source-text rule
rather than a runtime property. Push immediately after committing; if the push fails,
report it honestly and never force.
  </action>
  <verify>
    <automated>OLD='wrds_crsp_reference|wrds_crsp|wrds_taq|crsp_membership|crsp_rebuild|crsp_reference|crsp_symbology|crsp_tickers|nbbo_resample'; PAT="quantlab[/.](acquisition|dataset)[/.]($OLD)|(^|[^A-Za-z0-9_])($OLD)\.py|(acquisition|dataset) import [^\n]*($OLD)|quantlab/(acquisition/wrds|dataset/(crsp|nbbo))\.py|(^|[^A-Za-z0-9_/])(wrds|crsp|nbbo)\.py"; EXC='"wrds" +/ +"crsp\.py"'; grep -rnE "$PAT" quantlab tests scripts example CLAUDE.md --exclude-dir=__pycache__ | grep -vE "$EXC" > /tmp/quantlab-pkg-refactor/final-residue.txt; N=$(wc -l < /tmp/quantlab-pkg-refactor/final-residue.txt); echo "STALE PRE-MOVE REFERENCES REMAINING: $N (was 200, must be 0)"; cat /tmp/quantlab-pkg-refactor/final-residue.txt; test "$N" -eq 0</automated>
    <automated>for p in quantlab/acquisition/wrds.py quantlab/acquisition/wrds_crsp.py quantlab/acquisition/wrds_crsp_reference.py quantlab/acquisition/wrds_taq.py quantlab/dataset/crsp.py quantlab/dataset/crsp_membership.py quantlab/dataset/crsp_rebuild.py quantlab/dataset/crsp_reference.py quantlab/dataset/crsp_symbology.py quantlab/dataset/crsp_tickers.py quantlab/dataset/nbbo.py quantlab/dataset/nbbo_resample.py; do test ! -e "$p" || { echo "STILL PRESENT: $p"; exit 1; }; done; echo "all 12 pre-move paths are gone"</automated>
    <automated>test ! -s quantlab/acquisition/__init__.py && test ! -s quantlab/dataset/__init__.py && echo "acquisition/ and dataset/ __init__.py still 0 bytes"</automated>
    <automated>uv run python -c "
import pathlib, re
root = pathlib.Path('.')
files = sorted(set(root.rglob('quantlab/**/*.py')) | set(root.glob('tests/*.py')) | set(root.glob('scripts/*.py')) | set(root.glob('example/*.md')))
missing = []
for p in files:
    if '__pycache__' in p.parts: continue
    for m in re.finditer(r'quantlab/[A-Za-z0-9_/]+[.]py', p.read_text(encoding='utf-8')):
        if not (root / m.group(0)).exists(): missing.append(f'{p}:{m.group(0)}')
print(f'DANGLING quantlab/*.py PATH REFERENCES: {len(missing)}')
for x in missing: print('  ', x)
" > /tmp/quantlab-pkg-refactor/after-dangling.txt; cat /tmp/quantlab-pkg-refactor/after-dangling.txt; diff /tmp/quantlab-pkg-refactor/before-dangling.txt /tmp/quantlab-pkg-refactor/after-dangling.txt && echo "DANGLING SET UNCHANGED (14 pre-existing, none added, none repaired)"</automated>
    <automated>find . -path ./.venv -prune -o -name __pycache__ -type d -print0 | xargs -0 rm -rf; uv run pytest -q --ignore=tests/test_factor_hierarchy.py --ignore=tests/test_crsp_rebuild_measurements.py 2>&1 | tail -80 > /tmp/quantlab-pkg-refactor/after.txt; tail -5 /tmp/quantlab-pkg-refactor/after.txt; echo "--- BEFORE ---"; cat /tmp/quantlab-pkg-refactor/before-summary.txt</automated>
    <automated>grep -oE '^(FAILED|ERROR) [^ ]+' /tmp/quantlab-pkg-refactor/after.txt | sort -u > /tmp/quantlab-pkg-refactor/after-ids.txt; echo "NEW failures (in AFTER, not in BEFORE) -- must be empty:"; comm -13 /tmp/quantlab-pkg-refactor/before-ids.txt /tmp/quantlab-pkg-refactor/after-ids.txt; echo "FIXED (in BEFORE, not in AFTER) -- report if non-empty:"; comm -23 /tmp/quantlab-pkg-refactor/before-ids.txt /tmp/quantlab-pkg-refactor/after-ids.txt</automated>
  </verify>
  <done>
The full sweep reports 0 stale pre-move references across `quantlab/`, `tests/`,
`scripts/`, `example/` and `CLAUDE.md`, down from 200, with the count printed. None of the
12 pre-move paths exists on disk. `quantlab/acquisition/__init__.py` and
`quantlab/dataset/__init__.py` are still 0 bytes. The dangling-path set is the identical
14-line pre-existing set. The suite's failing node-ID set is unchanged in both directions
against the Task 1 baseline. `CLAUDE.md` describes the three entry-point packages and the
cost they carry. `example/*.md` and the three live `.planning` ledger entries point at
paths that exist; `.planning/phases/**`, `WINDOWS.md`, `ROADMAP.md` and `research/` are
untouched and the SUMMARY says why. The work is committed with explicit paths and pushed.
  </done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| repo working tree -> git history | 12 `git mv` operations. A `rm` + re-`Write` would silently sever `--follow` and lose the provenance of ~2,900 lines of reviewed code. |
| source text -> test assertions that read it | Six tests read module SOURCE by path (`ast.parse`, `read_text`) rather than importing it. A stale path makes the assertion read the wrong file, or no file. |
| test process -> WRDS network (Duo push) | `tests/conftest.py`'s autouse `_forbid_wrds_network` tripwire and the two dotted-string session patches. Retargeting those strings is the one edit in this plan that can silently disarm a safety mechanism. |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-edi-01 | Repudiation | the 12 moved files' git history | high | mitigate | `git mv` only, never `rm` + `Write`; Tasks 1 and 2 each verify with `git log --follow --oneline` per new path and report the commit count. |
| T-edi-02 | Tampering | `tests/conftest.py` dotted patch targets `quantlab.acquisition.wrds_taq.WrdsSession` | high | mitigate | A missed retarget makes `monkeypatch.setattr` patch nothing, the real `WrdsSession` reaches `psycopg2.connect`, and the autouse `_forbid_wrds_network` tripwire raises — loudly, in tests, before any Duo push. The failure is fail-closed by construction. Task 1 retargets all five strings and reruns every WRDS test file; `tests/test_crsp_tracer.py`'s ast gate independently pins that the provider never binds `WrdsSession` by name. |
| T-edi-03 | Spoofing | a source-scanning test reading a path that no longer exists, or the wrong file | high | mitigate | Two gates: the 200 -> 0 sweep with a reported count, and the dangling-`quantlab/*.py`-path check whose AFTER list must equal the 14-line BEFORE list. `tests/test_crsp_identity.py:1970` is the one reference Task 2's half-pattern cannot see (its `"dataset"` component is on the previous line), so it is named explicitly in the task text and is additionally covered by the Task 3 full sweep. |
| T-edi-04 | Tampering | stale `__pycache__` after 12 file moves | high | mitigate | Cache clear (`find . -path ./.venv -prune -o -name __pycache__ -type d -print0 \| xargs -0 rm -rf`) before every verifying run in all three tasks. This repo has already been burned once (gap G-03.11-4, `tests/test_stale_bytecode_lesson.py`). |
| T-edi-05 | Spoofing | a verify command rewritten to an absolute `/Users/...` path | high | mitigate | Every command in this plan is repo-root-relative. Execution happens in an isolated worktree; an absolute main-tree path would exercise unchanged code and report a false green. Stated in `<context>` and in `<verification>`. |
| T-edi-06 | Tampering | unrelated working-tree files swept into the commit | medium | mitigate | Explicit paths on every `git add`; no `-A` / `.` / `-a`, no `git stash`. Confirm with `git status --short` before committing. |
| T-edi-07 | Repudiation | `.planning/phases/**` audit trail rewritten by an over-broad sweep | medium | mitigate | The sweep's path list is `quantlab tests scripts example CLAUDE.md` and never `.planning`. The documentation-policy table names the three live ledger entries that may change and marks everything else DO NOT TOUCH. |
| T-edi-08 | Denial of Service | import-time regression on `quantlab/base/backtest.py` (+0.99s / +196 modules) | low | accept | Measured and accepted by the operator before this plan. Task 2 records it as a comment at the import so the next reader of a slow backtest import does not have to bisect for it. |
| T-edi-09 | Elevation of Privilege | a registration drifting back into a provider module now that they are siblings in one package | low | mitigate | `tests/test_wrds_vendor_seam.py`'s ast scan survives as a SOURCE-TEXT gate and gains a `node.level == 0` assertion, so a relative import cannot evade it — a new risk created precisely by the move into a package. |
| T-edi-SC | Tampering | npm/pip/cargo installs | high | mitigate | Not applicable: this plan installs no package and adds no dependency. If any task turns out to need one, stop and run the package-legitimacy gate first. |
</threat_model>

<verification>
- Three fresh interpreters, all printing `['alpaca', 'tiingo', 'wrds']` with no traceback:
  first import `quantlab.acquisition.wrds.taq`, then `quantlab.acquisition.wrds`, then
  `quantlab.acquisition.registry`.
- `git log --follow --oneline` reaches pre-move commits for each of the 12 new paths.
- Stale pre-move references: 200 -> 0, with the count printed and the residue file shown.
- Dangling `quantlab/**.py` path references: identical to the 14-line pre-existing baseline.
- `uv run pytest -q --ignore=tests/test_factor_hierarchy.py --ignore=tests/test_crsp_rebuild_measurements.py`
  — failing node-ID set identical to the BEFORE baseline in both directions. Expected 55
  pre-existing failures (`D-03.11-12-A`); the MEASURED baseline governs if it differs.
- `uv run pytest -q --collect-only tests/test_crsp_rebuild_measurements.py` collects, even
  though that file is off the standard command.
- Caches cleared (`find . -path ./.venv -prune -o -name __pycache__ -type d -print0 |
  xargs -0 rm -rf`) immediately before every verifying pytest or import probe.
- Every command above is repo-root-relative; none may be rewritten to an absolute
  `/Users/...` path, or an isolated-worktree run would exercise unchanged code and report a
  false green. The only absolute paths permitted anywhere are the
  `/tmp/quantlab-pkg-refactor/` scratch files, which hold run logs and are never committed.
</verification>

<success_criteria>
- 12 files moved by `git mv`, history intact on every one, and none of the 12 pre-move
  paths exists.
- `from quantlab.acquisition.wrds import WRDS_SOURCE`,
  `from quantlab.dataset.crsp import CrspStockDataset` and
  `from quantlab.dataset.nbbo import NbboPanelDataset` are spelled exactly as before and
  resolve to the same objects. No compatibility shim, alias or re-export stub exists at any
  old path.
- All three measured import orders still enumerate every vendor in a fresh interpreter, and
  `tests/test_wrds_vendor_seam.py` now pins all three permanently.
- Zero stale pre-move references in `quantlab/`, `tests/`, `scripts/`, `example/` and
  `CLAUDE.md` — reported as a count, from 200 to 0.
- The dangling-path set is the identical 14-line pre-existing set: nothing added, nothing
  incidentally repaired.
- The suite fails exactly the tests it failed before (expected 55, `D-03.11-12-A`), with the
  node-ID set difference empty in both directions.
- `quantlab/acquisition/wrds/__init__.py`'s docstring describes the package shape that
  exists, gives the measured cost of what the move traded away (1457 -> 1600 modules;
  registry and both datasets now loaded by any provider import), corrects the
  already-false "providers import each other not at all" claim, and names no pre-move path.
- `tests/test_wrds_vendor_seam.py`'s `taq.py` scan states that it is a SOURCE-TEXT claim,
  not a runtime one, and cannot be evaded by a relative import.
- `CLAUDE.md` documents the three entry-point packages, keeps the empty-`__init__` rule for
  every other package, and still records that `quantlab/acquisition/__init__.py` is 0 bytes.
- `example/*.md` and the three live `.planning` ledger entries point at paths that exist;
  `.planning/phases/**`, `WINDOWS.md`, `ROADMAP.md` and `research/` are byte-for-byte
  unchanged, and the SUMMARY says why that is correct rather than a miss.
- Committed with explicit paths and pushed.
</success_criteria>

<output>
Create `.planning/quick/260922-edi-package-wrds-crsp-and-nbbo-into-import-entry-poin/SUMMARY.md` when done.
</output>