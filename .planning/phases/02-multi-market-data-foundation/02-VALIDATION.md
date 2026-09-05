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
| 02-01-01 | 02-01 | 1 | Wave 0 | T-02-01-SC | N/A | scaffold | `uv add --dev pytest` | ❌ W0 | ⬜ pending |
| 02-01-02 | 02-01 | 1 | Pitfall regression | T-02-01-01 | `XrBackend.write()` idempotent-by-overwrite via `kwargs.setdefault("mode", "w")` | regression (reproduces `save()` called twice against same Zarr path) | `uv run pytest tests/test_backend_overwrite.py -x` | ❌ W0 | ⬜ pending |
| 02-02-01 | 02-02 | 2 | DATA-01, DATA-02, DATA-03 | T-02-02-01 | `AcquisitionConfig` has no credential field | unit (dataclass field introspection) | `uv run python -c "from base.config import DatasetConfig, AcquisitionConfig; ..."` | ❌ W0 | ⬜ pending |
| 02-02-02 | 02-02 | 2 | DATA-01, DATA-02, DATA-03 | T-02-02-02 | Config factories derive paths only via `_data_root()`/`_market_data_root()`/`_market_downloads_root()` | unit + regression smoke-check (ast.parse on cal.py/train_model.py/backtest/test_strategy.py, JSON/cell-parse on test_nt.ipynb) | `uv run pytest tests/test_config_paths.py -x` | ❌ W0 | ⬜ pending |
| 02-03-01 | 02-03 | 2 | DATA-04 | — | `dedup_raw_frame()` deterministic keep="last" dedup | unit | `uv run pytest tests/test_cleaning.py -k dedup -x` | ❌ W0 | ⬜ pending |
| 02-03-02 | 02-03 | 2 | DATA-04 | T-02-03-01, T-02-03-02, T-02-03-03 | Anomalies flagged not deleted/corrected; schema validation rejects malformed input before persistence; no forward-fill anywhere | unit (crafted DataFrames/xr.Datasets per rule: NaN-gap, anomaly-flag, schema) | `uv run pytest tests/test_cleaning.py -x` | ❌ W0 | ⬜ pending |
| 02-04-01 | 02-04 | 3 | DATA-01 | — | `Acquisition(ABC)` config lifecycle + watermark I/O, decoupled from Dataset/Zarr | unit (abstractness check) | `uv run python -c "from base.acquisition import Acquisition; ..."` | ❌ W0 | ⬜ pending |
| 02-04-02 | 02-04 | 3 | DATA-01 | T-02-04-01, T-02-04-02, T-02-04-03, T-02-04-SC | `TIINGO_API_KEY` never serialized into `AcquisitionConfig`/checkpoint JSON; explicit `columns=` always passed; incremental refresh from watermark | unit (mocked `TiingoClient`) | `uv run pytest tests/test_tiingo_acquisition.py -x` | ❌ W0 | ⬜ pending |
| 02-05-01 | 02-05 | 3 | DATA-02 | T-02-05-01 | `dedup_raw_frame()` inserted before `SpotKlineDataset.to_xarray()` | unit (synthetic CSV fixtures matching `BinanceCSVHeaders.SPOT`) | `uv run pytest tests/test_spot_dataset.py -x` | ❌ W0 | ⬜ pending |
| 02-05-02 | 02-05 | 3 | DATA-02 | T-02-05-02 | `--raw-data-dir` override lets a user point at their real, already-downloaded CSV location with no filesystem migration; no new network call (D-03) | unit (`_build_dataset_config` override behavior) + manual (`--help` output) | `uv run pytest tests/test_spot_dataset.py -x` | ❌ W0 | ⬜ pending |
| 02-05-03 | 02-05 | 3 | DATA-02 | T-02-05-03 | Real BTCUSDT 1d sample (fetched via the user's existing `binance-data-downloader` tool, D-11) round-trips through `ingest_binance_spot.py` into Zarr; failure is honestly recorded, never fabricated | manual/integration (real external tool + real Zarr write, non-gating) | `test -f .planning/phases/02-multi-market-data-foundation/02-05-REAL-DATA-CHECK.md` | ❌ W0 | ⬜ pending |
| 02-06-01 | 02-06 | 3 | DATA-03 | — | Core layers contain no literal reference to a concrete `Dataset` subclass or market-specific literal | automated grep-style check | `uv run pytest tests/test_extensibility_contract.py -k purity -x` | ❌ W0 | ⬜ pending |
| 02-06-02 | 02-06 | 3 | DATA-03 | — | A genuinely novel `Dataset` subclass runs the full lifecycle with zero core-layer changes | automated proof (`FakeDataset` subclass exercising full lifecycle) | `uv run pytest tests/test_extensibility_contract.py -x` | ❌ W0 | ⬜ pending |
| 02-06-03 | 02-06 | 3 | DATA-03 | T-02-06-01 | No structural coupling (isinstance/hasattr dispatch, market/frequency-value branching) in `base/factor.py`/`base/model.py`/`base/backend.py` beyond what the automated grep catches | manual design/code review (`checkpoint:human-verify`, ROADMAP Success Criterion 4's literal requirement) | N/A (human review) | ❌ W0 | ⬜ pending |
| 02-07-01 | 02-07 | 4 | DATA-01 | T-02-07-01 | `dedup_raw_frame()` inserted before `StockDataset.to_xarray()` | unit | `uv run pytest tests/test_stock_dataset.py -k dedup -x` | ❌ W0 | ⬜ pending |
| 02-07-02 | 02-07 | 4 | DATA-01 | — | Full `TiingoAcquisition` -> `StockDataset` -> Zarr round trip, `anomaly_flag` present, no forward-fill | integration (mocked `TiingoClient`, real tmp-path Zarr roundtrip) | `uv run pytest tests/test_stock_dataset.py -x` | ❌ W0 | ⬜ pending |
| 02-07-03 | 02-07 | 4 | DATA-01 | T-02-07-02 | `ingest_tiingo.py` never prints/logs the raw `TIINGO_API_KEY` value | manual (`--help` output) + grep | `TIINGO_API_KEY=test uv run python ingest_tiingo.py --help` | ❌ W0 | ⬜ pending |

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
| Adding a new market/frequency requires only a new `Dataset` subclass + config, no changes to factor/model/backtest code | DATA-03 / ROADMAP Success Criterion 4 | The automated `FakeDataset` proof (above) covers the mechanical contract, but confirming "no changes needed in factor/model/backtest code" for a *real* hypothetical third source is a design/code-review judgment call, not something a single automated test can fully prove | Covered by 02-06 Task 3 (`checkpoint:human-verify`, task ID `02-06-03`): reviewer reads `base/factor.py`, `base/model.py`, `base/backend.py` and confirms none of them reference `SpotKlineDataset`/`StockDataset` by name or by market-specific branching logic, and no `isinstance`/`hasattr` dispatch on a concrete `Dataset` subclass or on `config.market`/`config.frequency` values exists — only interaction through the `Dataset`/`DataBackend` ABC interface |
| Real BTCUSDT 1d sample round-trips through `ingest_binance_spot.py` into Zarr | DATA-02 | Verifying against real market data (not just synthetic fixtures) requires an actual network fetch via an external tool outside quantlab's control — not something a hermetic unit test should depend on for gating | Covered by 02-05 Task 3 (`02-05-03`, non-gating): fetch one real month of `BTCUSDT` `1d` klines via the user's existing `binance-data-downloader` CLI tool, run `ingest_binance_spot.py` against it, record evidence (or an honest failure) in `02-05-REAL-DATA-CHECK.md`. Tasks 1-2's fixture-based automated tests remain the phase's gating DATA-02 verification regardless of this task's outcome. |
| Tiingo EOD API's actual default JSON field set (with vs. without explicit `columns=`) | DATA-01 | ~~RESEARCH.md Open Question 2 — requires a live API call against the real Tiingo API with a real key; cannot be resolved by static analysis or unit tests with mocked responses~~ **Mitigated by design, no manual verification required.** `TiingoAcquisition` (02-04) always passes an explicit `columns=` parameter (`TiingoColumns.EOD`) to `get_ticker_price()`, listing every field `StockDataset` needs — this sidesteps the ambiguity of Tiingo's undocumented default field set entirely, regardless of what that default actually is. See 02-RESEARCH.md's Open Question 2 (updated) and Pitfall 3 for the full rationale. | None — no live-API smoke test is required for this phase. If a future phase wants to confirm the assumption empirically anyway (e.g. before removing the explicit `columns=` for some reason), that would require a live `TIINGO_API_KEY` and is out of scope here. |

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies (the sole exception, 02-06's Task 3, is a `checkpoint:human-verify` task per ROADMAP Success Criterion 4's explicit design/code-review requirement — not an automation gap)
- [x] Sampling continuity: no 3 consecutive `type="auto"` tasks without automated verify
- [x] Wave 0 covers all MISSING references (02-01 stands up pytest + fixtures + the Zarr overwrite fix every later plan's `<verify><automated>` depends on)
- [x] No watch-mode flags
- [x] Feedback latency < 15s (estimated ~10s full-suite runtime on small synthetic fixtures, no network calls)
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** pending (execution not yet started — sign-off above reflects plan-set completeness, not a post-execution green suite)
