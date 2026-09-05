---
phase: 2
slug: multi-market-data-foundation
status: draft
nyquist_compliant: true
wave_0_complete: false
created: 2026-09-05
---

# Phase 2 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (not yet installed — no `pytest` in `pyproject.toml`, no `tests/` directory; confirmed absent by RESEARCH.md) |
| **Config file** | none — Wave 0 installs |
| **Quick run command** | `uv run pytest tests/ -x -q` |
| **Full suite command** | `uv run pytest tests/ -q` |
| **Estimated runtime** | ~10 seconds (small synthetic fixtures, no network calls, no large data) |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest tests/ -x -q`
- **After every plan wave:** Run `uv run pytest tests/ -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 02-0X-0Y | TBD | 0 | Wave 0 | — | N/A | scaffold | `uv add --dev pytest` | ❌ W0 | ⬜ pending |
| 02-0X-0Y | TBD | TBD | DATA-01 | T-02-01 | `TIINGO_API_KEY` never serialized into `AcquisitionConfig`/checkpoint JSON | unit (mocked `TiingoClient`) + integration (real Zarr roundtrip on tmp path) | `uv run pytest tests/test_tiingo_acquisition.py tests/test_stock_dataset.py -x` | ❌ W0 | ⬜ pending |
| 02-0X-0Y | TBD | TBD | DATA-02 | — | N/A | unit (synthetic CSV fixtures matching `BinanceCSVHeaders.SPOT`) | `uv run pytest tests/test_spot_dataset.py -x` | ❌ W0 | ⬜ pending |
| 02-0X-0Y | TBD | TBD | DATA-03 | — | N/A | automated proof (`FakeDataset` subclass exercising full lifecycle) + manual design review | `uv run pytest tests/test_extensibility_contract.py -x` | ❌ W0 | ⬜ pending |
| 02-0X-0Y | TBD | TBD | DATA-04 | T-02-02 | Anomalies flagged not deleted/corrected; schema validation rejects malformed input before persistence | unit (crafted DataFrames/xr.Datasets per rule: dedup, NaN-gap, anomaly-flag, schema) | `uv run pytest tests/test_cleaning.py -x` | ❌ W0 | ⬜ pending |
| 02-0X-0Y | TBD | TBD | Pitfall regression | — | N/A | regression (reproduces `save()` called twice against same Zarr path) | `uv run pytest tests/test_backend_overwrite.py -x` | ❌ W0 | ⬜ pending |

*Exact Task IDs are filled in by the planner once PLAN.md files exist — this table's rows are the required coverage set, not final IDs.*

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `uv add --dev pytest` — no test framework installed anywhere in the repo today
- [ ] `tests/conftest.py` — shared fixtures: synthetic Binance-CSV fixture (matching `BinanceCSVHeaders.SPOT` columns), synthetic Tiingo-JSON fixture (matching the confirmed field set — see RESEARCH.md Assumptions Log A1 on the `columns=` ambiguity), and a mocked `TiingoClient` fixture (no real network calls in unit tests)
- [ ] `tests/test_tiingo_acquisition.py`, `tests/test_stock_dataset.py`, `tests/test_spot_dataset.py`, `tests/test_cleaning.py`, `tests/test_extensibility_contract.py`, `tests/test_backend_overwrite.py` — all net-new stub files; zero existing test infrastructure to extend (confirmed by RESEARCH.md: zero `tests/`, zero pytest config anywhere in the repo)
- [ ] A regression test locking in the `XrBackend.write()` `mode="w"` fix (RESEARCH.md Pitfall 1: re-running `save()` against an existing Zarr path currently raises `FileExistsError`) — this fix is a hard prerequisite for the incremental/daily-refresh requirement (CONTEXT.md D-10)

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Adding a new market/frequency requires only a new `Dataset` subclass + config, no changes to factor/model/backtest code | DATA-03 / ROADMAP Success Criterion 4 | The automated `FakeDataset` proof (above) covers the mechanical contract, but confirming "no changes needed in factor/model/backtest code" for a *real* hypothetical third source is a design/code-review judgment call, not something a single automated test can fully prove | Reviewer reads `base/factor.py`, `base/model.py`, `base/backend.py` and confirms none of them reference `SpotKlineDataset`/`StockDataset` by name or by market-specific branching logic — only through the `Dataset`/`DataBackend` ABC interface |
| Tiingo EOD API's actual default JSON field set (with vs. without explicit `columns=`) | DATA-01 | RESEARCH.md Open Question 2 — requires a live API call against the real Tiingo API with a real key; cannot be resolved by static analysis or unit tests with mocked responses | During Wave 0 or the first acquisition task, make one live `client.get_ticker_price(<symbol>, fmt="json", columns=None)` call (using the user's own rotated/active `TIINGO_API_KEY`) and diff the returned field set against the explicit `columns=` list assumed by `StockDataset._to_kunquant`; document the result in the plan's SUMMARY.md |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
