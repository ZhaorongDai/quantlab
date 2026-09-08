---
phase: quick-260907-rjq
plan: 01
subsystem: config / cli
status: complete
tags: [cli, config, storage-paths, ast-guard, mutation-testing]
requires:
  - config/__init__.py storage-path factories (02-CONTEXT.md D-01/D-02)
  - utils/cli.py shared argument-group helpers (03.2 D-14)
provides:
  - config.get_data_root() / config.set_data_root() -- the one storage-root resolver, three levels
  - utils.cli.add_data_dir_arg() / apply_data_dir() -- the shared flag and its explicit applicator
  - "--data-dir on all five data-acquisition entry points"
affects:
  - every path any run reads or writes (raw downloads, watermarks, Zarr stores, nautilus catalog, universe table)
tech-stack:
  added: []
  patterns:
    - "process-level override + explicit per-script applicator, not an argparse action side effect (D-03)"
    - "AST ordering guard over a script list DERIVED from a repo glob, so a new entry point cannot opt out by omission"
    - "deny set derived by transitive reachability to a `from config import` name, not by a name prefix"
key-files:
  created:
    - tests/test_data_dir_cli.py
  modified:
    - config/__init__.py
    - utils/cli.py
    - refresh_us_equity_universe.py
    - ingest_tiingo.py
    - ingest_alpaca.py
    - ingest_us_equity.py
    - ingest_binance_spot.py
    - README.md
    - tests/conftest.py
    - tests/test_config_paths.py
    - tests/test_factor_kunquant.py
decisions:
  - "D-02 applied as a rename, not an alias: `_data_root()` is gone from the repo, all 14 in-file call sites plus the three prose mentions moved in the same plan."
  - "The plan's `_build*`-prefix deny rule was replaced by transitive reachability to a `from config import` name -- the prefix rule as written denies `_build_arg_parser()`, which every script calls before `parse_args()`, making the ordering guard unsatisfiable."
  - "The derived-script-list lock was rewritten after a mutation escaped it: a substring scan for `add_data_dir_arg` still matched a script whose parser call had been deleted, because the import stayed behind."
metrics:
  duration: 34 min
  completed: 2026-09-07
actuals:
  tokens: 38000
  tasks: 3
  commits: 3
plan_head_before: 875fc540f567fc1e1f4c3fff8820d498d6e8dd64
---

# Quick Task 260907-rjq: `--data-dir` CLI Parameter Summary

A per-run storage root on all five data-acquisition entry points, resolved
through one knob with three levels -- `--data-dir` > `QUANTLAB_DATA_DIR` >
repo-root `data/` -- with the ordering that makes an override actually reach
disk enforced structurally rather than by convention.

## What Was Built

**`config/__init__.py`.** `_data_root()` was RENAMED to `get_data_root()`
(D-02, no alias) and given the three-level precedence body, and
`set_data_root(path)` was added beside it with a module-level
`_DATA_ROOT_OVERRIDE`. The setter `expanduser()`s but deliberately does not
`resolve()` (D-07 -- the env knob does not resolve either, and diverging would
change behaviour on a symlinked root), neither creates nor requires the
directory (D-08), clears on `None` (D-06), raises on an empty or
whitespace-only value, and returns the stored `Path`. All 14 in-file path
expressions moved to the new name in the same commit; the two factory
docstrings claiming `QUANTLAB_DATA_DIR` was "the only path knob" now state the
precedence while keeping the substance of the original claim (one root, no
hardcoded volume, `subdir`/`store_name` still beneath it).

**`utils/cli.py`.** `add_data_dir_arg(parser)` and `apply_data_dir(args)`, in
the module's existing shared-helper tradition. The `config` import inside
`apply_data_dir` is deferred to call time (D-04) so the dependency-light CLI
module does not drag `dataset.backend`/`dataset.spot`/`dataset.stock`/
`base.config` into every import of it; the module docstring's
dependency-surface paragraph now states that where the promise is made.

**All five entry points** call `add_data_dir_arg(parser)` in their parser build
and `apply_data_dir(args)` in `__main__` immediately after `parse_args()`,
ahead of everything that can reach a `config/` factory -- including the
indirect reachers `_build_configs` (tiingo, alpaca) and `_build_dataset_config`
(binance). `refresh_us_equity_universe.py` grew a module-level
`_build_arg_parser()` matching the other four so tests drive the same parser
`__main__` does. Each call carries a one-line comment saying why its position
matters.

**`ingest_binance_spot.py` composition (D-05).** `--data-dir` and
`--raw-data-dir` operate at different levels and neither supersedes the other.
The module docstring, the `--raw-data-dir` help text and a combined usage
example all say so, and a test drives the real parser and config builder to
prove `--data-dir D --raw-data-dir R` reads CSVs from `R` while writing the
Zarr under `D`.

**The locks.** `tests/test_config_paths.py` gained the four-cell precedence
matrix asserted through real factory paths (a resolver that answers correctly
while no factory consults it would pass a resolver-only test), the four-field
reach across both markets and both constituent panels, a by-construction check
that reads `config/__init__.py`'s own source and asserts no factory path
argument bypasses `get_data_root()` or its two derived roots, a vacuity guard
on that scan, and the env-user/default-user no-regression pair.
`tests/test_data_dir_cli.py` carries the precedence/setter/helper unit cases,
the autouse-reset contract, and the AST ordering guard.
`tests/conftest.py` gained an autouse fixture clearing the override before AND
after every test.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] The plan's `_build*`-prefix deny rule makes the ordering guard unsatisfiable**
- **Found during:** Task 3
- **Issue:** The plan specified the per-script deny set as "the names that
  script imports `from config import ...`, plus any local call whose name
  starts with `_build`". Every one of the five scripts opens `__main__` with
  `parser = _build_arg_parser()`, which must run before `parse_args()` and
  therefore before `apply_data_dir(args)`. Under the literal rule, all five
  scripts fail the guard and no correct arrangement exists.
- **Fix:** The deny set is derived by transitive reachability instead -- the
  names imported `from config import ...`, plus every module-level function
  that calls one of them, transitively. `_build_configs` and
  `_build_dataset_config` are caught by derivation rather than by spelling;
  `_build_arg_parser` is correctly not denied, because it reaches nothing. This
  is what the prefix rule was a proxy for, and it is strictly stronger: a
  factory-reaching helper named without a `_build` prefix would escape the
  prefix rule and does not escape this one. Documented in the helper's
  docstring.
- **Files modified:** tests/test_data_dir_cli.py
- **Commit:** 4552b0c

**2. [Rule 1 - Bug] The derived-script-list lock escaped its own mutation**
- **Found during:** Task 3, mutation 2
- **Issue:** `_scripts_offering_the_flag()` was first written as a substring
  scan for `add_data_dir_arg`. Deleting `add_data_dir_arg(parser)` from
  `ingest_alpaca.py` left the name in that module's `from utils.cli import
  (...)` block, so the scan still matched, the derived list still equalled the
  expected five, and the entire suite stayed green -- while `--data-dir` was
  dead on that script (the flag unregistered, so `apply_data_dir` silently
  no-ops on a missing attribute).
- **Fix:** Registration is now detected by AST -- a *Call* to
  `add_data_dir_arg`, never the mere presence of the name -- and the per-script
  guard additionally imports the module and asserts
  `_build_arg_parser().parse_args([])` actually carries a `data_dir`
  attribute, so a registration moved into a branch that never runs also fails.
  Re-running mutation 2 against the strengthened lock now fails two tests.
- **Files modified:** tests/test_data_dir_cli.py
- **Commit:** 4552b0c

### Adjusted expectations (not code changes)

The first draft of `test_omitting_data_dir_leaves_the_universe_paths_untouched`
expected `<root>/reference/universe.parquet`. Today's actual path is
`<root>/data/reference/universe.parquet` -- the storage ROOT defaults to the
repo's `data/` directory and every factory nests its own `data/`/`downloads/`
subtree beneath whatever root answered, so the repo default doubles the
segment. DDIR-03 is about reproducing today's paths exactly, so the test was
corrected to the real literal rather than the code being "fixed" to match a
prettier one. The doubling is called out in a comment so a later reader does
not read it as a typo.

## Verification

- `uv run pytest tests/ -q` -- **537 passed** (505 planning-time baseline + 32
  new). Zero failures, zero skips introduced.
- `--data-dir` present in `--help` on all five entry points, each importing
  cleanly with no credentials in the environment.
- `grep -rnE '(^|[^A-Za-z0-9_])_data_root' --include='*.py' --include='*.md' .`
  (excluding `.planning/`) -- zero hits.
- `grep -rni 'only path knob' --include='*.py' --include='*.md' .` (excluding
  `.planning/`) -- zero hits.

### Mutation verification

All four plan-specified mutations were run, and one extra. Each was reverted.

| # | Mutation | Result |
|---|----------|--------|
| 1 | Move `apply_data_dir(args)` after `_build_dataset_config(args)` in `ingest_binance_spot.py` | RED -- ordering guard failed for that script |
| 2 | Delete `add_data_dir_arg(parser)` from `ingest_alpaca.py` | **SURVIVED on first attempt** (see Deviation 2), then RED on 2 tests after the lock was strengthened |
| 3 | Invert precedence in `get_data_root()` so the env var beats the override | RED -- 3 tests across both files |
| 4 | Make `set_data_root(None)` a no-op instead of clearing | RED -- 6 tests across both files |
| 5 (extra) | Point `universe_config().output_path` at a hardcoded `/Volumes/BigDisk` | RED -- the by-construction root check and the reach test both failed |

## Success Criteria

- **DDIR-01** met -- `--data-dir /X` roots `raw_data_dir_path`,
  `zarr_file_path`, `watermark_path` and `output_path` under `/X`, asserted
  through real factory calls across both markets and both constituent panels.
- **DDIR-02** met -- all four precedence cells pass, and the by-construction
  scan proves no factory path argument bypasses the single root.
- **DDIR-03** met -- 505 pre-existing tests still pass; the env-user and
  default-user no-regression pair pins today's literals.
- **DDIR-04** met -- `apply_data_dir` precedes every factory-reaching call in
  every `__main__`, enforced by a derived, mutation-verified AST guard whose
  script list comes from a repo glob.
- **DDIR-05** met -- `--raw-data-dir` still redirects the raw CSV directory
  alone and composes with `--data-dir`, documented in the module docstring, the
  help text and a test.

## Threat Mitigations Applied

- **T-rjq-02** (medium, mitigate) -- ordering misdirection. Mitigated by the
  AST guard, derived script list, and the mutation that moves the call.
- **T-rjq-03** (medium, mitigate) -- process-global leakage between tests.
  Mitigated by the autouse `conftest.py` fixture clearing on both sides;
  mutation 4 confirms the clear is load-bearing.
- **T-rjq-04** (low, mitigate) -- `apply_data_dir` prints nothing, reads no
  credential, and adds no field to `AcquisitionConfig.to_dict()`.

No new packages installed; `pyproject.toml` unchanged.

## Known Stubs

None.

## Self-Check: PASSED

- `config/__init__.py`, `utils/cli.py`, `refresh_us_equity_universe.py`,
  `ingest_tiingo.py`, `ingest_alpaca.py`, `ingest_us_equity.py`,
  `ingest_binance_spot.py`, `README.md`, `tests/conftest.py`,
  `tests/test_data_dir_cli.py`, `tests/test_config_paths.py`,
  `tests/test_factor_kunquant.py` -- all present.
- Commits `aaefe11`, `a772d21`, `4552b0c` -- all present in `git log`.
- `git rev-list --count 875fc54..HEAD` = 3, matching `commits: 3`.
