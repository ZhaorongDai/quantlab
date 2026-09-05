---
phase: 3
slug: factor-computation-kunquant-polars
status: draft
nyquist_compliant: true
wave_0_complete: false
created: 2026-09-05
---

# Phase 3 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 9.1.1 (already installed via Phase 2's `[dependency-groups] dev`) |
| **Config file** | `pyproject.toml` `[tool.pytest.ini_options]` (`pythonpath = ["."]`) — already present, reused as-is |
| **Quick run command** | `uv run pytest tests/ -x -q` |
| **Full suite command** | `uv run pytest tests/ -q` |
| **Estimated runtime** | ~5-10 seconds (no network, no large data; KunQuant compilation is the only slow step and is already exercised by existing Phase-1/2 test infra patterns) |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest tests/ -x -q`
- **After every plan wave:** Run `uv run pytest tests/ -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------------|-----------|-------------------|-------------|--------|
| 03-0X-0Y | TBD | 0 | Wave 0 | N/A | scaffold | promote `_write_stock_pqt`-equivalent helper to `tests/conftest.py` | ❌ W0 | ⬜ pending |
| 03-0X-0Y | TBD | TBD | FACTOR-01 | N/A | unit/regression | `uv run pytest tests/test_factor_kunquant.py -k spot -x` | ❌ W0 | ⬜ pending |
| 03-0X-0Y | TBD | TBD | FACTOR-01 (new Alpha158Stock) | Correct `amount = volume * close` proxy, no `RuntimeError` | unit | `uv run pytest tests/test_factor_kunquant.py -k stock -x` | ❌ W0 | ⬜ pending |
| 03-0X-0Y | TBD | TBD | FACTOR-01 (Alpha101Stock bugfix, pre-existing bug found by research) | `Alpha101Stock.cal()` no longer raises `RuntimeError: Bad inputs` | regression | `uv run pytest tests/test_factor_kunquant.py -k alpha101_stock_bugfix -x` | ❌ W0 | ⬜ pending |
| 03-0X-0Y | TBD | TBD | FACTOR-02 | `cal_stream()` batch-replay produces incremental updates with no exception | unit/smoke | `uv run pytest tests/test_factor_stream.py -x` | ❌ W0 | ⬜ pending |
| 03-0X-0Y | TBD | TBD | FACTOR-03 | Example `Momentum(FactorPolars)` factor produces `xr.Dataset` with only dynamically-resolved factor columns | unit | `uv run pytest tests/test_factor_polars.py -x` | ❌ W0 | ⬜ pending |
| 03-0X-0Y | TBD | TBD | FACTOR-03 (laziness contract, D-04) | `_get_factor_lazyframe()` never calls `.collect()` internally | unit (introspection) | `uv run pytest tests/test_factor_polars.py -k laziness -x` | ❌ W0 | ⬜ pending |
| 03-0X-0Y | TBD | TBD | FACTOR-04 | No public `Factor`/`FactorKunQuant`/`FactorPolars` method accepts/returns a bare DataFrame — only `xr.Dataset` at the boundary | unit (signature introspection) | `uv run pytest tests/test_factor_hierarchy.py -k boundary_contract -x` | ❌ W0 | ⬜ pending |
| 03-0X-0Y | TBD | TBD | D-03 interchangeability | `DLConfig(factors=[kunquant_instance, polars_instance])` works uniformly, no `isinstance` branching in `base/model.py` | integration | `uv run pytest tests/test_factor_hierarchy.py -k interchangeability -x` | ❌ W0 | ⬜ pending |

*Exact Task IDs are filled in by the planner once PLAN.md files exist — this table's rows are the required coverage set, not final IDs.*

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] Promote the private `_write_stock_pqt`-style helper (currently duplicated in `tests/test_stock_dataset.py`) to `tests/conftest.py` so new `Alpha158Stock`/`Alpha101Stock` tests reuse it instead of duplicating it a third time
- [ ] `tests/test_factor_kunquant.py`, `tests/test_factor_stream.py`, `tests/test_factor_polars.py`, `tests/test_factor_hierarchy.py` — all net-new
- [ ] No new framework/dependency install needed — pytest, polars, and KunQuant are all already present in the project

---

## Manual-Only Verifications

*All phase behaviors have automated verification — no manual-only checks required for this phase (unlike Phase 2, this phase introduces no new external data sources or credentials).*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
