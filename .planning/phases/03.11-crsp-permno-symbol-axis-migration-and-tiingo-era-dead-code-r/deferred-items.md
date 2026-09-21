# Phase 03.11 — Deferred Items

Out-of-scope discoveries logged during execution. Per the executor scope
boundary, these are NOT fixed by the plan that found them.

---

## D-03.11-12-A — 55 pre-existing full-suite failures: tests still import the
moved top-level `ingest_*.py` shells

**Found during:** Plan 03.11-12, Task 3 (full regression)
**Status:** open — pre-existing at this phase's base commit, untouched by 03.11-12

`uv run pytest -q --ignore=tests/test_factor_hierarchy.py
--ignore=tests/test_crsp_rebuild_measurements.py` reports
**55 failed, 1675 passed, 1 skipped**. Every one of the 55 is a
`ModuleNotFoundError: No module named 'ingest_tiingo'` /
`'ingest_binance_spot'`, or the `FileNotFoundError` twin of it raised by tests
that read the shell's SOURCE text to assert on its argparse wiring.

| Test file | Failures |
|---|---|
| `tests/test_ingest_shells.py` | 14 |
| `tests/test_ingest_tiingo_universe_wiring.py` | 11 |
| `tests/test_data_dir_cli.py` | 10 |
| `tests/test_ingest_conversion_gate.py` | 6 |
| `tests/test_volume_guard.py` | 5 |
| `tests/test_factor_kunquant.py` | 3 |
| `tests/test_chunked_ingest.py` | 2 |
| `tests/test_spot_dataset.py` | 2 |
| `tests/test_entry_point_contracts.py` | 1 |
| `tests/test_universe.py` | 1 |

**Cause.** Commit `d07f06e "modify docs and move scripts to dir"` moved the
top-level ingest shells into a directory. The tests above still import them by
their old top-level module name, or open `'ingest_tiingo.py'` relative to the
repo root. `git ls-tree --name-only e95ad6b | grep -i ingest` is EMPTY at this
phase's base commit, so the breakage predates every 03.11 plan.

**Why 03.11-12 did not fix it.** None of the ten files reference
`CrspTickerLookup`, `_spell` or `predict_panel`'s guard — the three files
03.11-12 changed (`quantlab/dataset/crsp_tickers.py`, plus a docstring in
`quantlab/base/model.py` and one line of `example/wrds_crsp.md`) cannot reach
them. Repointing ten test files at relocated CLI shells is a distinct piece of
work with its own blast radius, and folding it into a gap-closure plan about a
ticker sidecar would hide it inside an unrelated commit.

**Verification that 03.11-12 is green on its own surface:**
`uv run pytest -q tests/test_crsp_ticker_sidecar.py tests/test_model_predict_panel.py
-p no:cacheprovider` -> **56 passed**.
