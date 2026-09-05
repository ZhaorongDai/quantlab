---
phase: 3
slug: factor-computation-kunquant-polars
status: planned
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
| **Estimated runtime** | ~25-40 s for the full suite once Phase 3 lands. Measured during planning: each KunQuant graph compilation costs ~1.5 s (batch) / ~1.2 s (stream) and the phase adds ~7 compiling tests; everything else is milliseconds. Per-file commands stay ~2-6 s. Cost is held down by passing an explicit 1-3 element `factor_names` list and `njobs=4` in every factor test. |

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
| 03-01-T1 | 03-01 | 1 | FACTOR-01 (BUG-01: `my_ops` decompose() signature drift — every batch `cal()` raises `TypeError` today) | Correct op decomposition; no silent value change | tracer / end-to-end | `uv run pytest tests/test_factor_kunquant.py -x -q` | ❌ W0 | ⬜ pending |
| 03-01-T2 | 03-01 | 1 | Wave 0 | N/A | scaffold | `uv run pytest tests/ -q` | ❌ W0 | ⬜ pending |
| 03-02-T1 | 03-02 | 2 | D-03 (config split + DLConfig/MLConfig widening) | No import cycle introduced | unit / introspection | `uv run pytest tests/ -q` | ✅ after 03-01 | ⬜ pending |
| 03-02-T2 | 03-02 | 2 | FACTOR-01 (behavior preservation through the `Factor` ABC extraction, incl. the previously-untested `label/spot.py` classes) | `Factor` never reads `config.mode` (runtime `AttributeError` guard) | regression | `uv run pytest tests/test_factor_kunquant.py tests/test_factor_hierarchy.py tests/test_extensibility_contract.py -x -q` | ✅ after 03-01 | ⬜ pending |
| 03-02-T3 | 03-02 | 2 | FACTOR-04 + D-03 + D-07 + Hazard 1 (`__init__` ordering) | No public method accepts/returns a bare DataFrame; `self.config` is assigned before the storage backend and no setter-reachable method reads it | unit (signature + source introspection) | `uv run pytest tests/test_factor_hierarchy.py -q` | ✅ after 03-01 | ⬜ pending |
| 03-03-T1 | 03-03 | 3 | FACTOR-01 (D-02 `amount` proxy; Alpha101Stock `RuntimeError: Bad inputs` regression) | Proxy is double-guarded and inert when not requested | unit + regression | `uv run pytest tests/test_factor_kunquant.py -q` | ✅ after 03-01 | ⬜ pending |
| 03-03-T2 | 03-03 | 3 | FACTOR-01 (D-01 new `Alpha158Stock`) + NORM-01/D-09 normalization matrix (both US-equity classes raw) | Normalization semantics recorded, not silent | unit + source introspection | `uv run pytest tests/test_factor_kunquant.py -q` | ✅ after 03-01 | ⬜ pending |
| 03-03-T3 | 03-03 | 3 | NORM-01 + D-09 | The locked matrix and the Phase-6 deferral are recoverable from `03-03-SUMMARY.md` | documentation + source introspection | `uv run pytest tests/test_factor_kunquant.py -k normalization -q` | ✅ after 03-01 | ⬜ pending |
| 03-04-T1 | 03-04 | 3 | FACTOR-03 + D-07 (`FactorPolars` has no streaming surface) | Core-layer purity of `base/factor_polars.py` | unit / introspection | `uv run pytest tests/test_extensibility_contract.py -q` | ✅ after 03-01 | ⬜ pending |
| 03-04-T2 | 03-04 | 3 | FACTOR-03 (D-08 `Momentum` example factor) | Per-symbol window, not global shift | unit | `uv run pytest tests/ -q` | ✅ after 03-01 | ⬜ pending |
| 03-04-T3 | 03-04 | 3 | FACTOR-03 + FACTOR-04 + D-04 laziness + D-05 dynamic names | `_get_factor_lazyframe()` materializes nothing (proved by monkeypatching `pl.LazyFrame.collect` to raise) | unit | `uv run pytest tests/test_factor_polars.py -q` | ✅ after 03-01 | ⬜ pending |
| 03-05-T1 | 03-05 | 4 | FACTOR-02 (BUG-02: hardcoded x86 SIMD block width crashes streaming compilation on aarch64) | x86_64 behavior byte-identical after the fix | unit | `uv run pytest tests/test_factor_kunquant.py -q` | ✅ after 03-01 | ⬜ pending |
| 03-05-T2 | 03-05 | 4 | FACTOR-02 | Replay raises no exception; successive updates differ (not a constant) | smoke / replay | `uv run pytest tests/test_factor_stream.py -q` | ✅ after 03-01 | ⬜ pending |
| 03-05-T3 | 03-05 | 4 | D-03 interchangeability + FACTOR-03 + FACTOR-04 | Test body contains no `isinstance`/`hasattr` branching | integration | `uv run pytest tests/test_factor_hierarchy.py -q` | ✅ after 03-01 | ⬜ pending |

*Task IDs assigned 2026-09-05 during phase planning. Two rows (03-01-T1 / 03-05-T1) cover blocking defects discovered by executing the real code during planning and NOT present in 03-RESEARCH.md: `my_ops/preprocess.py`'s `decompose()` signature drift against installed KunQuant 0.1.11, which breaks every batch `cal()`; and `FactorKunQuant._make_stream()`'s hardcoded x86 SIMD block width, which makes streaming compilation raise on aarch64. 03-RESEARCH.md's `[ASSUMED]` A1 (`num_stock` must be a multiple of 8) is RESOLVED: replays at 4, 5, 6, 8 and 16 symbols all succeeded with correct shapes.*

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] Promote the private `_write_stock_pqt`-style helper (currently duplicated in `tests/test_stock_dataset.py`) to `tests/conftest.py` so new `Alpha158Stock`/`Alpha101Stock` tests reuse it instead of duplicating it a third time
- [ ] `tests/test_factor_kunquant.py`, `tests/test_factor_stream.py`, `tests/test_factor_polars.py`, `tests/test_factor_hierarchy.py` — all net-new, each landing with one immediately-passing infrastructure self-test so per-file pytest commands are meaningful rather than exit-code-5
- [ ] `spot_kline_zarr` and `stock_zarr` factory fixtures in `tests/conftest.py` — synthetic, seeded, strictly-positive Binance-shaped and Tiingo-shaped Zarr stores written to `tmp_path`. Phase 2's `write_binance_csv` fixture is insufficient: it emits 2 rows for 1 symbol, whereas factor windows need ~60 timestamps and the streaming replay needs 8 symbols. Fixtures must write the store to disk BEFORE returning a `DatasetConfig` carrying a non-None `symbols`, because `Dataset.config`'s setter reads the store
- [ ] No new framework/dependency install needed — pytest, polars, and KunQuant are all already present in the project

---

## Manual-Only Verifications

**None.** The phase has no blocking human checkpoint and no manual-only verification.

03-03 Task 3 was previously a blocking `checkpoint:human-verify` asking the user to confirm the four-class normalization matrix. That domain judgement (时序 vs 截面 strategy type) has since been ruled on directly by the user and locked as **D-09 in `03-CONTEXT.md`** — both US-equity classes emit raw values, both crypto-spot classes apply the rolling time-series z-score. With the answer settled before execution, the task is now `type="auto"`: it writes the locked matrix and the Phase-6 ARCH-01/ARCH-02 deferral into `03-03-SUMMARY.md`, verified by `test_normalization_matrix_matches_recorded_strategy_types`. Keeping it blocking would have stalled wave 4 (`03-05` depends on `03-03`) on a human re-answering a settled question.

This phase introduces no new external data source, credential or network call, so nothing else requires manual verification either.

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies — no exceptions; every task in the phase, including 03-03 Task 3, now carries an automated command
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references (03-01 Tasks 1-2)
- [x] No watch-mode flags
- [x] Feedback latency < 15s for the per-file commands each task verifies with
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
