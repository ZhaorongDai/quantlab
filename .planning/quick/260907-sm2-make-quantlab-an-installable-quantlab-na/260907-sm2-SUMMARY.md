---
phase: quick-260907-sm2
plan: 01
subsystem: packaging
tags: [packaging, namespace, setuptools, uv, editable-install, test-guards]
status: complete
requires: []
provides:
  - "quantlab.* single top-level package"
  - "installable distribution (setuptools build backend, explicit package set)"
  - "quantlab.utils.paths.INSTRUMENTS_CONFIG_PATH package-derived data location"
  - "editable path dependency consumable by quantlab-console"
affects:
  - "every first-party import in the repo"
  - "persisted config.name dotted paths (hard break, documented)"
tech-stack:
  added:
    - "setuptools>=77 as an explicit [build-system] requirement (was already the implicit backend)"
  patterns:
    - "one stdlib-only leaf module owns a package-derived path constant shared by two callers"
    - "structural test guards resolve imports with ast, not substring scans over source text"
key-files:
  created:
    - quantlab/__init__.py
    - quantlab/vecbt/__init__.py
    - quantlab/utils/paths.py
  modified:
    - pyproject.toml
    - quantlab/config/__init__.py
    - quantlab/utils/nautilus.py
    - quantlab/utils/module.py
    - get_binance_instruments.py
    - tests/test_volume_guard.py
    - tests/test_config_paths.py
    - tests/test_data_dir_cli.py
    - tests/conftest.py
    - README.md
decisions:
  - "Hard break for pre-migration persisted config.name values; no legacy alias table (evidence: zero checkpoints of either format anywhere in tree or history)"
  - "Instrument metadata location is package-derived, not working-directory-relative"
  - "The shared path constant lives in utils/, not the config package, because the config package imports the dataset layer (cycle)"
  - "The volume guard's structural arm is an ast import resolver, not a substring scan"
metrics:
  duration: ~50 min
  completed: 2026-09-07
  tasks: 3
actuals:
  tokens: 96000
  tasks: 3
  commits: 3
plan_head_before: 5380ebcc1a2ff811c226971606d76315c6767491
---

# Quick Task 260907-sm2: Make quantlab an installable `quantlab.` namespace package — Summary

Moved twelve flat top-level packages under a single `quantlab/` package, rewrote every
first-party import, re-rooted and re-proved every structural test guard, and flipped
`uv sync` in the sibling `quantlab-console` repo from failing outright to installing
`quantlab` as an editable path dependency — with the test suite held at exactly 537.

## What Shipped

**Task 1 — the move (`69fb808`).** `git mv` of `acquisition`, `base`, `config`,
`dataset`, `dl_model`, `enums`, `factor`, `label`, `ml_model`, `my_ops`, `utils`,
`vecbt` into `quantlab/`; 50 tracked files, recorded as 50 renames in history. Added
`quantlab/__init__.py` and the `quantlab/vecbt/__init__.py` that package never had
(without it, `find_packages` semantics would have silently skipped `vecbt`). Rewrote
first-party imports across 71 files — packages, tests, root scripts, and the two
imports inside `test_nt.ipynb`'s JSON cell sources. Import *form* was preserved
(`import config` → `import quantlab.config as config`), which is what kept ~200
downstream call sites and the `pkgutil` walk in `test_acquisition_batching.py`
untouched.

Two path defects the rename created or exposed:

- `get_data_root()` resolved its default as `parent.parent / "data"`, which meant the
  repo root when the file was `config/__init__.py` and would now mean `quantlab/data/` —
  a directory that does not exist. Every acquisition run would have reported one root
  and written to another. Fixed with one more `.parent` hop.
- `get_binance_instruments.py` located `instruments.yaml` against the *current working
  directory*, in a function default and again in an argparse default. That was only ever
  correct when the process started at the repo root, and the move broke it outright. The
  new stdlib-only `quantlab/utils/paths.py` exports one package-derived
  `INSTRUMENTS_CONFIG_PATH`; the Nautilus loader and both CLI defaults now read that
  single constant, so no yaml path literal survives in either and the two cannot drift
  apart again.

`pyproject.toml` gained an explicit `[build-system]` (setuptools was already the implicit
backend — the flat-layout error that motivated this task is setuptools' own message), a
`[tool.setuptools.packages.find]` restricted to `quantlab*` with namespaces disabled, and
a `[tool.setuptools.package-data]` entry for the yaml.

**Task 2 — the guards (committed in `9423539`, see Anomaly below).** The import rewrite
cannot reach a module path written as a string, so these were swept separately: 15
monkeypatch string targets, the `conftest` `find_spec` probe, and three sets of
CWD-relative source paths. Two guards needed real repair rather than re-prefixing, and
both are the "passes for the wrong reason" class this plan was written to catch.

**Task 3 — the record and the payoff (`b6bc175`).** The persisted-config decision with its
evidence, 104 backticked doc paths and every fenced-block import re-pointed, and the
console install proven.

## The Two Guards That Would Have Passed For The Wrong Reason

**`test_config_paths.py` derived its repo root from `config.__file__` by walking up two
levels — the same expression as the storage-root default.** After the move both would be
wrong by the same directory, so the test would have agreed with the bug and stayed green.
Re-derived from the test file's own location, matching the independent witness
`test_data_dir_cli.py` already used. Proven: reintroducing the shallower `.parent` walk
fails **5 tests across both files**; restoring it returns all 41 to green and leaves
`config/__init__.py` byte-identical.

**The volume guard's structural arm asserted three dotted literals were absent from the
universe module's source text.** Airtight while every package was a separate root, because
an acquisition module could then only be named absolutely. With all twelve under one
parent, a *relative* import reaches a client while producing none of those substrings —
and naively re-prefixing the literals makes it strictly worse, since a prefixed literal
cannot match a relative spelling at all. Replaced with an `ast` resolver that resolves
relative levels against the module's own package.

RED-proven against all four spellings the new layout admits, each attributed to the
resolver by a repo-unique token (`FORBIDDEN-IMPORT-RESOLVED`) rather than merely by a
non-zero exit — two of the four also trip the neighbouring bound-clients scan, so an
exit-code-only probe would have reported them as proven while testing nothing:

| Spelling | Result |
|---|---|
| `from quantlab.base.acquisition import Acquisition` | RED via resolver |
| `from ..base.acquisition import Acquisition` | RED via resolver |
| `from ..base import acquisition` | RED via resolver |
| `import quantlab.acquisition.tiingo` | RED via resolver |

Green on the unmodified module, and `universe.py` left byte-identical after the probes.

**Deny-set derivation in `test_data_dir_cli.py`** selected imports by the first
dot-separated segment of the module name. Post-rewrite that segment is `quantlab` for
every first-party import, so the comparison matched either nothing or everything. Now
matched by full dotted name against a `CONFIG_MODULE` constant. Confirmed populated for
all five scripts for the right reason, not assumed:

```
ingest_alpaca.py               ['_build_configs', 'stock_acquisition_config', 'stock_kline_config', 'universe_config']
ingest_binance_spot.py         ['_build_dataset_config', 'spot_kline_config']
ingest_tiingo.py               ['_build_configs', 'stock_acquisition_config', 'stock_kline_config', 'universe_config']
ingest_us_equity.py            ['stock_acquisition_config', 'stock_kline_config', 'universe_config']
refresh_us_equity_universe.py  ['universe_config']
```

Mutation-verified: adding a module-scope config import to `quantlab/utils/cli.py` fails
the guard for **both** import forms (`from quantlab.config import ...` and
`import quantlab.config as config`).

## Acceptance Gates — Real Output

```
537 passed, 143 warnings in 26.39s
SUITE OK 537 collected, 0 skipped        # junit: failures=0 errors=0, tests-skipped == 537
```

```
CONSOLE EDITABLE OK /Users/daizhaorong/projects/quantlab/quantlab/__init__.py
```

Other gates, all passing:

```
IMPORTS OK /Users/daizhaorong/projects/quantlab/quantlab/__init__.py
DATA ROOT OK /Users/daizhaorong/projects/quantlab/data          # root read from git, not cwd
INSTRUMENTS PATH OK .../quantlab/config/instruments.yaml        # loaded + venue lookup from a foreign cwd
WHEEL OK 57   top-level set: ['quantlab', 'quantlab-0.1.0.dist-info']
RED PROOF OK 4/4 VIA RESOLVER
DECISION RECORDED 1980 chars
```

Residual sweeps (imports, path literals, residual directories, doc imports, doc paths):
all empty. `data/` — real market data — is absent from the wheel.

## Truths

| ID | Status | Evidence |
|---|---|---|
| NS-01 | met | Every layer imports under `quantlab.`; no former root resolves into this repo |
| NS-02 | met | Wheel top-level set is exactly `quantlab` + `.dist-info`; `data/` absent |
| NS-03 | met | Sweeps empty over `quantlab/`, `tests/`, `scripts/`, root scripts, notebook, docs. `.planning/` deliberately untouched |
| NS-04 | met | 537 passed, 0 failed, 0 errors, 0 skipped — same count as baseline |
| NS-04a | met | Both witnesses independent; both fail when the depth bug is reintroduced |
| NS-04b | met | 4/4 spellings RED **through the resolver**, token-attributed |
| NS-04c | met | Zero path literals naming a moved dir in root scripts / `scripts/`; the yaml resolves through the package |
| NS-05 | met | Decision + evidence + `260907-sm2` in `get_cls_from_path.__doc__` |
| NS-06 | met | `uv sync` installs; `quantlab.__file__` resolves into this tree |

Also: rename visible as a rename (50), root CLI scripts still at repo root, no
`console_scripts`, `tests/` at root, `pythonpath` unchanged, `.planning/` not rewritten.

## Deviations from Plan

**1. [Rule 2 — missing critical functionality] Doc paths with a `:Symbol` suffix.**
The plan's doc sweep regex stops at `[A-Za-z0-9_./*-]`, so backticked references of the
form `` `base/model.py:BaseModel` `` were invisible to it. Roughly 40 such references
across `README.md` and `example/*.md` would have been left naming directories that no
longer exist while the gate reported clean. Widened the character class to include `:`
and re-ran. Committed in `b6bc175`.

**2. [Advisory, folded in as instructed] Root-script docstring prose.** Re-derived live
rather than trusting the supplied line numbers; found 13 backticked references across
`ingest_alpaca.py`, `ingest_us_equity.py`, `ingest_tiingo.py`, `ingest_binance_spot.py`
and `refresh_us_equity_universe.py`. Updated. Deliberately **not** touched: `config.kwargs`
and `config.get_data_root()`, which are attribute accesses on a still-correctly-bound
name, not module paths. Also updated two package comments (`quantlab/enums/data.py`,
`quantlab/base/acquisition.py`) that described the volume guard as a substring scan —
now inaccurate as well as stale. The `#`-prefix constraint that
`test_ticker_pattern_reconciliation.py` depends on was preserved and verified.

**3. Comment-text discipline observed throughout.** No rewritten prose quotes a retired
spelling beside its replacement — doing so would plant a permanent failure in the very
sweeps that prove the rewrite complete.

**4. `test.py` — plan/orchestrator note corrected.** The orchestrator stated `test.py`
carries first-party imports the sweep would match. Re-derived: it imports only `polars`.
Nothing to rewrite; its pre-existing uncommitted diff was left untouched and unstaged
throughout, verified after every commit.

**5. README additions beyond the literal scope.** Added a "Consuming quantlab from
another project" subsection and listed the new `paths.py`. The README tours the package
directories and documents installation; leaving it silent about the one capability this
task existed to create would have been a stale doc of the same class the task was fixing.

## Anomaly — Concurrent Commit (needs your attention)

**Task 2's code changes are committed, but under commit `9423539`, whose message
describes something else.** While Task 2's files were staged and the suite was running, a
concurrent process committed `9423539 docs(todo): guard the append dim against
overlapping timestamps` — and swept my 11 staged files into it alongside its own
`.planning/todos/` doc. My `git commit` then found an empty index and reported "no changes
added to commit".

Verified before continuing: **nothing was lost or altered.** All Task 2 edits are present
and correct in `HEAD` (resolver token, `CONFIG_MODULE`, the re-derived `_REPO_ROOT`, both
re-rooted source-path constants), the working tree matches `HEAD` except the untouched
`test.py`, and the full gate re-ran clean afterwards.

**I did not repair this by rewriting history.** `main` is a protected branch with another
actor committing into it concurrently; `git reset` on that ref is exactly the
destroys-concurrent-work operation the execution rules forbid self-healing with. The cost
is traceability only — Task 2's changes are findable but not described by their commit
message. If you want them split out cleanly, that is a `git reset --soft 69fb808` plus two
commits, and it is safe **only** once you know nothing else is committing.

Consequently `commits: 3` counts `9423539` even though one of its 12 files
(`.planning/todos/...`) is not this task's work.

## Note on Branch

Committed directly to `main`. The repo's `.planning/config.json` sets
`git.branching_strategy: "none"`, the orchestrator directed sequential execution on the
main working tree, and all five prior GSD quick tasks committed here. Flagged because
`git.allow_default_branch_commits` is not set in config, so the standard protected-branch
assertion would otherwise halt; the override knob was not added, since editing project
config was not part of this task.

## Known Stubs

None. No stub, placeholder, skipped test, or unrun `<verify>` was introduced.

## Threat Flags

None. No new network endpoint, auth path, file-access pattern, or trust-boundary schema
change. T-sm2-01 (data directory in a distribution) is closed by the wheel assertion;
T-sm2-02 (silently-skipped vendor patch) by the residual-directory sweep plus the held
537; T-sm2-03 by the resolver's RED proof; T-sm2-04 by the two independent witnesses;
T-sm2-06 by the shared package-derived constant.

## Self-Check: PASSED

See the self-check block appended below.

```
FOUND: quantlab/__init__.py
FOUND: quantlab/vecbt/__init__.py
FOUND: quantlab/utils/paths.py
FOUND: quantlab/config/instruments.yaml
FOUND: pyproject.toml
FOUND: quantlab/utils/module.py
FOUND: 69fb808   refactor(quick-260907-sm2): move the twelve flat packages under quantlab/
FOUND: 9423539   (carries Task 2 -- see Anomaly)
FOUND: b6bc175   docs(quick-260907-sm2): record the persisted-config break and re-point the docs
```

`test.py`'s pre-existing uncommitted diff (3 insertions, 37 deletions) is intact and was
never staged.
