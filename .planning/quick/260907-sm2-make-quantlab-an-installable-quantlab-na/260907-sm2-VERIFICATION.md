---
phase: quick-260907-sm2
verified: 2026-09-07T00:00:00Z
status: passed
score: 9/9 declared must-have truths verified (1 additional plan done-condition FAILED)
covered_files:
  - ".planning/quick/260907-sm2-make-quantlab-an-installable-quantlab-na/260907-sm2-PLAN.md"
  - ".planning/quick/260907-sm2-make-quantlab-an-installable-quantlab-na/260907-sm2-SUMMARY.md"
  - "README.md"
  - "get_binance_instruments.py"
  - "pyproject.toml"
  - "quantlab/__init__.py"
  - "quantlab/config/__init__.py"
  - "quantlab/utils/module.py"
  - "quantlab/utils/nautilus.py"
  - "quantlab/utils/paths.py"
  - "quantlab/vecbt/__init__.py"
  - "tests/conftest.py"
  - "tests/test_config_paths.py"
  - "tests/test_data_dir_cli.py"
  - "tests/test_volume_guard.py"
covered_digest: "v1:sha256:22003c465f5c8c42e73413a21d9a195aaacea11077661918107c54bbc2a12c05"
behavior_unverified: 0
overrides_applied: 0
gaps:
  - truth: "Task 3 done-condition: no stale backticked directory path remains in README.md or example/*.md"
    status: partial
    reason: >-
      The executor's Deviation 1 widened the doc-path sweep's character class to
      include ':' and reported it clean. The widened class still stops at '(',
      so every backticked reference of the form `pkg/file.py:Symbol()` remained
      invisible. Six such references survive in example/*.md, each naming a
      directory that no longer exists. README.md is clean; the gap is confined
      to example/. The SUMMARY's claim "Residual sweeps (... doc paths): all
      empty" is contradicted by a re-run with the terminator relaxed to [^`].
    artifacts:
      - path: "example/backend.md"
        issue: "line 264 `base/data.py:BaseDataset.head()`; line 294 `base/data.py:BaseDataset.from_raw_data_chunked()`"
      - path: "example/chunking.md"
        issue: "line 3 `base/data.py:BaseDataset.from_raw_data_chunked()` (sitting on the same line as an already-corrected `quantlab/base/chunking.py`); line 5 `acquisition/universe.py:assert_chunked_panel_fits()`"
      - path: "example/dataset.md"
        issue: "line 84 `dataset/cleaning.py:dedup_raw_frame(keep=\"last\")`; line 510 `dataset/cleaning.py:clean_membership_panel()`"
    missing:
      - "Prefix the six surviving paths with `quantlab/`."
      - "Re-run the doc sweep with the closing terminator relaxed from a character class to [^`]+ so a trailing call form cannot hide a stale path again: grep -rInoE '`(acquisition|base|config|dataset|dl_model|enums|factor|label|ml_model|my_ops|utils|vecbt)/[^`]*`' README.md example/*.md"
advisory:
  - finding: "277 backticked references to bare former-root module/file paths survive in .py docstrings and comments across quantlab/ and tests/ (e.g. quantlab/utils/cli.py:20 `from config import set_data_root`, quantlab/dataset/cleaning.py:281 `base/data.py:...`, tests/conftest.py:17 `import acquisition.alpaca`, cal.py:7 and train_model.py:8 commented-out imports of `dl_model.transformer` / `vecbt.bt`)."
    category: other
    reason: >-
      None is executable and none is reachable by any structural guard, so no
      truth fails. The plan scoped the prose rewrite to README.md and example/*.md
      only, so this is out-of-contract drift rather than a plan violation --
      raised because it is the same class of staleness the task existed to fix
      and it will mislead a reader. No deterministic evidence of harm.
    evidence_status: "none provided"
---

# Quick Task 260907-sm2: Make quantlab an installable `quantlab.` namespace package — Verification Report

**Task Goal:** Move the twelve flat top-level packages under a single `quantlab/`
package so the out-of-repo `quantlab-console` repo can depend on it as an
editable path dependency; rewrite every first-party import; declare a build
backend and an explicit package set; keep root CLI scripts at the repo root;
never move or package `data/`.

**Verified:** 2026-09-07
**Status:** gaps_found (one plan done-condition, documentation-only)
**Re-verification:** No — initial verification

Every check below was executed in this session. SUMMARY.md claims were treated
as hypotheses, not evidence; where a claim is reproduced here it is because I
re-ran the command.

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| NS-01 | Every layer resolves under the single `quantlab.` prefix; the twelve former roots no longer exist as importable roots | ✓ VERIFIED | `uv run python -c "import quantlab, quantlab.base.config, quantlab.base.model, quantlab.acquisition.universe, quantlab.utils.module, quantlab.utils.nautilus, quantlab.utils.paths, quantlab.config, quantlab.dataset.stock, quantlab.factor.alpha101, quantlab.vecbt.bt"` → `IMPORTS OK /Users/daizhaorong/projects/quantlab/quantlab/__init__.py`. With `sys.path` containing the repo root, `importlib.util.find_spec` on all twelve bare names returns `LEAKED ROOTS: []`. Filesystem sweep for residual top-level dirs: none. |
| NS-02 | Built wheel contains exactly one top-level dir plus `.dist-info`; repo-root `data/` absent | ✓ VERIFIED | I ran `uv build --wheel` myself. `TOPLEVEL ['quantlab', 'quantlab-0.1.0.dist-info']`; `DATA HITS []`; 57 members; all 12 subpackages + 13 `__init__.py` present, including the formerly `__init__`-less `vecbt` and the flat-layout-excluded `utils`; `quantlab/config/instruments.yaml` shipped via `[tool.setuptools.package-data]`. |
| NS-03 | Zero residual first-party imports of the bare names in `quantlab/`, `tests/`, `scripts/`, `example/`, repo-root scripts; `.planning/` not rewritten | ✓ VERIFIED | The plan's quote-tolerant sweep returns empty over `quantlab tests scripts ./*.py ./*.ipynb`. Doc import sweep over `README.md example/*.md` empty. Notebook checked by parsing the JSON cell sources directly (not by regex): both first-party imports are `from quantlab.dataset.spot ...` / `from quantlab.config ...`. `scripts/` carries no first-party import. No `.planning/` file appears in any of the three task commits. |
| NS-04 | `uv run pytest tests/ -q` → 537 passed, 0 failed, 0 errors | ✓ VERIFIED | Run in this session: `537 passed, 143 warnings in 29.45s`. Zero skipped. |
| NS-04a | `get_data_root()` defaults to REPO-root `data/`, asserted by two witnesses with independent derivations | ✓ VERIFIED (behavioral) | Runtime: `DATA ROOT /Users/daizhaorong/projects/quantlab/data` (expected root read from `git rev-parse --show-toplevel`, not from the package under test). Derivations confirmed independent by reading both files: `tests/test_config_paths.py:31 _REPO_ROOT = Path(__file__).resolve().parent.parent` and `tests/test_data_dir_cli.py:24 REPO_ROOT = Path(__file__).resolve().parent.parent` — neither uses `config.__file__`. **Mutation-proved**: I reintroduced the shallower `.parent` walk in `quantlab/config/__init__.py:86`; `test_config_paths.py` → 2 failed, `test_data_dir_cli.py` → 3 failed (5 total, matching the SUMMARY). Restored; `git diff` clean. |
| NS-04b | The volume guard's structural arm goes RED for all four new-layout spellings, and does so THROUGH the resolver | ✓ VERIFIED (behavioral) | I re-ran the RED proof myself on the plan's four spellings **plus three I invented** (`from . import tiingo`, `from .alpaca import AlpacaAcquisition`, `from ..acquisition import tiingo`): **7/7 RED-VIA-RESOLVER**, each failure carrying `FORBIDDEN-IMPORT-RESOLVED` under `--tb=line`. Token is unique to one file (`tests/test_volume_guard.py:703`). Green on the unmodified module. `universe.py` byte-identical after every probe. **Token attribution independently validated**: I injected a bound `Acquisition` subclass via `importlib.import_module` (invisible to the AST resolver); the test went red at `tests/test_volume_guard.py:814: AssertionError: ['_LeakedClient']` with the token **absent** — so the token check genuinely discriminates the resolver arm from the neighbouring bound-clients scan rather than passing on any failure. |
| NS-04c | Repo-root scripts and `scripts/` resolve their non-import filesystem references; the one packaged data file is located through the package | ✓ VERIFIED | Path-literal sweep over `./*.py ./*.ipynb scripts` returns empty. `get_binance_instruments.py` carries no `.yaml` literal; both the function default (line 17) and the argparse default (line 134) read `INSTRUMENTS_CONFIG_PATH` from `quantlab.utils.paths`. `quantlab/utils/nautilus.py:_load_instrument_config` reads the same constant. The constant is `Path(__file__).resolve().parent.parent / "config" / "instruments.yaml"` — stdlib-only leaf, no cycle. Confirmed resolving from the console repo's process too. |
| NS-05 | The `get_cls_from_path` behaviour change is written down as a decision with its evidence | ✓ VERIFIED | `quantlab/utils/module.py` carries both a module docstring and a 30-line `get_cls_from_path` docstring stating the hard break, the absence of a legacy alias table, the `import_path` → `config.name` → JSON chain, and the census. Anchored with `260907-sm2`. **Evidence independently re-derived**: `find` over the tree (excluding `.git`/`.venv`) returns zero `*.pth` and zero `*.joblib`; `git log --all --diff-filter=A` returns zero of either extension in history; the only `config.json` is `.planning/config.json`. |
| NS-06 | `uv sync` in `quantlab-console` installs `quantlab` editable and `import quantlab` there resolves into this tree | ✓ VERIFIED | `uv --directory /Users/daizhaorong/projects/quantlab-console sync` → `Resolved 145 packages / Audited 123 packages`, exit 0. `uv --directory ... run python -c "import quantlab, quantlab.base.config, quantlab.acquisition.universe"` → `/Users/daizhaorong/projects/quantlab/quantlab/__init__.py`. Console `site-packages` holds `__editable__.quantlab-0.1.0.pth`, `__editable___quantlab_0_1_0_finder.py`, `quantlab-0.1.0.dist-info` — an editable install, not a copied build. `quantlab.utils.paths.INSTRUMENTS_CONFIG_PATH` resolves from that process to the tracked yaml. |
| — | **Derived from Task 3 `<done>` / plan `<verification>`:** no stale backticked directory path remains in `README.md` or `example/*.md` | ✗ FAILED | Six survivors in `example/*.md` (see Gaps). README.md clean. |

**Score:** 9/9 declared `must_haves.truths` verified. One additional plan-stated
done-condition failed.

### Required Artifacts

| Artifact | Expected | Status | Details |
|---|---|---|---|
| `quantlab/__init__.py` | package root marker | ✓ VERIFIED | tracked, in wheel |
| `quantlab/vecbt/__init__.py` | new marker so `find_packages` sees `vecbt` | ✓ VERIFIED | added in `69fb808`; `quantlab/vecbt/bt.py` present in the wheel — proves the marker is load-bearing and worked |
| `quantlab/config/instruments.yaml` | packaged data | ✓ VERIFIED | in wheel via `[tool.setuptools.package-data]`; loads and serves a venue lookup |
| `quantlab/utils/paths.py` | stdlib-only leaf owning one package-derived constant | ✓ VERIFIED | 36 lines, substantive; imports only `pathlib`; docstring records the DECISION; two consumers wired |
| `pyproject.toml` | `[build-system]` + explicit package set | ✓ VERIFIED | `setuptools>=77` / `setuptools.build_meta`; `include = ["quantlab*"]`, `namespaces = false`; `[project.scripts]` absent (no `console_scripts` introduced); `pythonpath = ["."]` unchanged |
| `quantlab/utils/module.py` | decision record | ✓ VERIFIED | see NS-05 |

### Key Link Verification

| From | To | Via | Status | Details |
|---|---|---|---|---|
| `quantlab/config/__init__.py:get_data_root()` | `test_data_dir_cli.py:REPO_ROOT` **and** `test_config_paths.py:_REPO_ROOT` | two independent `Path(__file__)` walks from the test files | ✓ WIRED | Neither derives from `config.__file__`. Mutation-proved: both go red on the depth bug. |
| `test_data_dir_cli.py:_path_consuming_names()` | root scripts' `from quantlab.config import ...` | full-dotted-name equality against `CONFIG_MODULE = "quantlab.config"` (line 30/347) | ✓ WIRED | I executed the function against all five scripts: deny sets non-empty for every one (`ingest_alpaca` 4 names, `ingest_binance_spot` 2, `ingest_tiingo` 4, `ingest_us_equity` 3, `refresh_us_equity_universe` 1). The built-in "asserting nothing" tripwire is therefore silent **for the right reason**, not by accident. |
| `test_volume_guard.py` structural arm | `quantlab/acquisition/universe.py` imports | `ast` `Import`/`ImportFrom` resolver with relative-level handling | ✓ WIRED | 7/7 RED-via-token; discrimination probe confirms the token is not satisfied by a neighbouring arm. |
| residual top-level dirs | `tests/conftest.py` `find_spec` probe | implicit-namespace-package failure mode | ✓ WIRED | Zero residual dirs. `find_spec("quantlab.acquisition.alpaca")` returns a real `ModuleSpec` with `origin=.../quantlab/acquisition/alpaca.py`, so the gated Alpaca patch is applied, not silently skipped. All other monkeypatch string targets re-prefixed (`quantlab.acquisition.universe.requests.get`, `quantlab.acquisition.tiingo.TiingoClient`, `quantlab.base.data.BaseDataset.head`). |
| `get_cls_from_path()` | `import_path` → persisted `config.name` | `importlib.import_module` | ✓ WIRED | Hard break documented with independently re-derived evidence. |
| `[tool.setuptools.packages.find] include` | repo-root `data/` | exclusion by `include = ["quantlab*"]` | ✓ WIRED | Wheel `DATA HITS []`. |
| `quantlab/utils/paths.py` | `instruments.yaml` → nautilus loader **and** Binance CLI | one shared constant | ✓ WIRED | No yaml path literal in either consumer. |
| `[tool.setuptools.package-data]` | yaml in the installed wheel | explicit declaration | ✓ WIRED | asserted against the built wheel's namelist, not the source tree. |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|---|---|---|---|
| Full suite holds at baseline | `uv run pytest tests/ -q` | 537 passed, 0 failed, 0 errors, 0 skipped | ✓ PASS |
| Console editable install | `uv --directory .../quantlab-console sync` then `import quantlab` | resolves to `/Users/daizhaorong/projects/quantlab/quantlab/__init__.py` | ✓ PASS |
| Wheel top-level + data exclusion | `uv build --wheel` + zipfile namelist | `['quantlab', 'quantlab-0.1.0.dist-info']`, no `data/` | ✓ PASS |
| Storage-root depth bites | reintroduce shallower `.parent`, run both witness files | 2 + 3 failures, restored clean | ✓ PASS |
| Resolver RED proof (7 spellings) | inject import at top of `universe.py`, `pytest --tb=line`, require token | 7/7 RED-VIA-RESOLVER | ✓ PASS |
| Token discriminates arms | bind an `Acquisition` subclass via `importlib` (no AST-visible import) | red at the bound-clients scan, token **absent** | ✓ PASS |
| Deny set populated | execute `_path_consuming_names` on all five scripts | non-empty for all five | ✓ PASS |
| Data root at runtime | `get_data_root()` with override cleared and env unset | repo-root `data/`, exists | ✓ PASS |

### Commit Attribution (re-split by the developer after execution)

I checked the split rather than taking it on trust. It is correct.

| Commit | Message | Contents | Verdict |
|---|---|---|---|
| `69fb808` | move the twelve flat packages | 50 renames (`--find-renames` confirms `R100`/`R09x`, i.e. history records renames, not delete+add) + 3 adds (`quantlab/utils/__init__.py`, `quantlab/utils/paths.py`, `quantlab/vecbt/__init__.py`) + `pyproject.toml`, 7 root scripts, the notebook, 34 test files | ✓ matches Task 1 |
| `509c387` | re-derive the two guards | exactly 11 files — the guard tests plus the two package comments that described the guard as a substring scan | ✓ matches Task 2 (and the 11 files the executor reported as swept up) |
| `db51ef3` | record the break, re-point the docs | `quantlab/utils/module.py` + `README.md` + 9 `example/*.md` | ✓ matches Task 3 |
| `c6c5e46` | docs(todo) | one file: `.planning/todos/pending/2026-09-07-...md` | ✓ carries **only** the unrelated todo |

No `.planning/` file is touched by any of the three task commits. Nothing is
misattributed. The two package-comment fixes (`quantlab/base/acquisition.py`,
`quantlab/enums/data.py`) that SUMMARY.md lists under Task 3's deviations sit in
the Task 2 commit; that is where they belong by subject (both describe the
volume guard), so it is not a misattribution.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|---|---|---|---|---|
| lines added by `69fb808..db51ef3` | — | `TBD`/`FIXME`/`XXX`/`TODO`/`HACK`/`PLACEHOLDER` | — | **Zero.** Count over added lines (excluding `.planning/`) is 0. |
| `example/backend.md`, `example/chunking.md`, `example/dataset.md` | 6 sites | stale backticked path naming a moved directory | ⚠️ Warning | See Gaps. |
| `quantlab/**`, `tests/**`, `cal.py`, `train_model.py` | 277 sites | stale backticked module/file path in docstrings and comments | 📋 Advisory | Non-executable; out of the plan's declared prose scope. See `advisory:`. |

### Success Criteria (plan `<success_criteria>`)

| Criterion | Status | Evidence |
|---|---|---|
| NS-01..NS-06 hold | ✓ | table above |
| Root CLI scripts remain at repo root; no `console_scripts` | ✓ | `cal.py`, `train_model.py`, `ingest_*.py`, `refresh_us_equity_universe.py`, `get_binance_instruments.py`, `main.py`, `read_mock_data_sink.py` all at root; `[project.scripts]` absent from `pyproject.toml` |
| `tests/` at repo root, `pythonpath` unchanged | ✓ | `tests/` at root; `[tool.pytest.ini_options] pythonpath = ["."]` |
| Rename appears in git history as a rename | ✓ | 50 `R`-status entries under `--find-renames` |
| Nothing under `.planning/` rewritten | ✓ | no `.planning/` path in `69fb808`, `509c387`, `db51ef3` |

### Human Verification Required

None. Every truth was exercised by a command in this session; no visual, UX, or
external-service behaviour is in scope.

### Gaps Summary

The task's goal is achieved. Both acceptance gates pass under independent
re-run, all nine declared `must_haves.truths` hold, and the two items flagged
for scrutiny both survive adversarial probing — the volume guard's `ast`
resolver is genuinely the arm that fires (proved by a counter-probe that makes
the *neighbouring* arm fail and confirms the token is then absent), and
`test_config_paths.py`'s repo root is now an independent witness that goes red
when the depth bug is reintroduced.

The one gap is documentation, and it is a residue of the executor's own
Deviation 1. Having discovered that the plan's doc sweep could not see
`` `base/model.py:BaseModel` ``, the executor widened the character class by
adding `:` — but the class still terminates at `(`, so every reference written
as `` `pkg/file.py:Symbol()` `` stayed invisible. Six of those survive in
`example/*.md`. `example/chunking.md:3` is the clearest witness: a corrected
`` `quantlab/base/chunking.py` `` and a stale
`` `base/data.py:BaseDataset.from_raw_data_chunked()` `` sit on the same line.
SUMMARY.md's "Residual sweeps ... doc paths: all empty" is therefore false as
stated.

This blocks nothing about installability and costs about five minutes: prefix
the six paths and re-run the sweep with the terminator relaxed to `[^`]*`
rather than an enumerated character class, so no trailing call form can hide a
stale path a third time.

---

_Verified: 2026-09-07_
_Verifier: Claude (gsd-verifier)_


---

## Gap closure (orchestrator, after this report was written)

The single gap above — six stale backticked paths in `example/*.md` — was closed
directly rather than by re-running an executor: it is six prefix completions with an
unambiguous correct value, and the plan already declared a doc-path gate, so this is
that gate finishing its job rather than new scope.

Applied to the exact six lines this report names. The re-sweep the report recommends,
with the terminator relaxed rather than enumerated, now returns empty over
`README.md` and `example/*.md`:

```
grep -rInoE '`(acquisition|base|config|dataset|dl_model|enums|factor|label|ml_model|my_ops|utils|vecbt)/[^`]*`' README.md example/*.md
```

`status` was flipped from `gaps_found` to `passed` on that basis. Everything above this
section is the verifier's own report, unedited — including its finding that
SUMMARY.md's "residual doc sweeps: all empty" was false as stated, which stands as a
record of the defect regardless of the fix.

The report's **advisory** — 277 backticked references to former-root paths in `.py`
docstrings and comments — is NOT closed here. It is out of the plan's declared scope
(the prose rewrite was scoped to `README.md` and `example/*.md`), and folding 277
edits into a gap closure would be exactly the silent scope growth this workflow's
gates exist to prevent. Carried to its own todo instead.
