---
phase: quick-260915-p91
verified: 2026-09-15T20:20:00Z
status: passed
score: 11/11 must-haves verified
covered_files:
  - ".planning/quick/260915-p91-add-point-in-time-rule-based-universe-fi/260915-p91-PLAN.md"
  - ".planning/quick/260915-p91-add-point-in-time-rule-based-universe-fi/260915-p91-SUMMARY.md"
  - "example/README.md"
  - "example/universe.md"
  - "quantlab/factor/universe_filter.py"
  - "quantlab/utils/module.py"
  - "tests/test_universe_filtered_factor.py"
  - "tests/universe_fixtures.py"
covered_digest: "v1:sha256:51a3d42eb2d902edab1d510bf365e90bdc23c51bc3ca07535266c9560d5fba3a"
behavior_unverified: 0
overrides_applied: 0
---

# Quick 260915-p91: Point-in-time rule-based universe filter — Verification Report

**Goal:** Add a stock-universe filter as a Factor wrapper class (`UniverseFilteredFactor`) that wraps any KunQuant factor or label and drops in seamlessly wherever factor classes are used, with universe rules as constructor parameters rather than config fields.

**Verified:** 2026-09-15 (verifier re-ran every gate command; SUMMARY numbers were NOT taken on trust)
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth (plan `must_haves.truths`) | Status | Evidence (verifier-run, not SUMMARY-quoted) |
|---|---|---|---|
| 1 | Wrapper is a `FactorKunQuant`, drops into `MLConfig.factors/labels` and a backtest config; model/backtest layers untouched | ✓ VERIFIED | `class UniverseFilteredFactor(FactorKunQuant)` at universe_filter.py:126. Verifier ran `git diff --stat 4fa4b36 -- quantlab/base/model.py quantlab/base/backtest.py quantlab/base/config.py quantlab/ml_model quantlab/backtest` → **0 lines of output, exit 0**. Task diffstat 3c7fadc..HEAD touches only the 6 planned files. |
| 2 | LS-1 mask: ticker rule ∧ RAW `close[t] ≥ min_price` ∧ trailing-`window` mean RAW `close*volume ≥ min_dollar_volume`; incomplete/NaN window ⇒ out; adjusted columns never read; PIT | ✓ VERIFIED | `compute_universe_mask` (l.364-422) reads `PRICE_COLUMN="close"` / `VOLUME_COLUMN="volume"` only, `rolling(min_periods=window)`, `xr.where(in_u, 1.0, np.nan)`. Behavioral: 6 hand-panel matrix tests + parametrized `test_mask_is_point_in_time[6/10/14]` all passed in the verifier's own run. |
| 3 | LS-2 rewrite: `Div(v, Input("universe_mask"))` before every `CrossSectionalOp` input, fed in BOTH batch and stream; matches pandas reference; perturbation-invariant; stream == batch | ✓ VERIFIED | `_mask_cross_sectional_inputs` (l.86-123) mutates `op.inputs` for `isinstance(op, CrossSectionalOp)`. Both `_make` (factor.py:329) and `_make_stream` (factor.py:347) call `self._get_factor_func()`, which the wrapper overrides — so one override covers both layouts. Behavioral: `test_cross_sectional_rank_sees_only_in_universe_symbols`, `test_perturbing_never_in_universe_symbols_leaves_wrapped_outputs_unchanged` (with an explicit teeth assertion that the UNWRAPPED ranks DO move), `test_stream_outputs_match_batch_and_the_mask_row_matches` — all passed in the verifier's run. |
| 4 | LS-3 output masking at the panel's OWN timestamp; symbol axis drops ONLY ticker-rule symbols, threshold failures kept as NaN columns | ✓ VERIFIED | `_get_labels` = `_mask_panel(self.factor._get_labels(data))` (mask AFTER `shift`, l.620-628; `Return._get_labels` does `shift(timestamp=-n)`, fret.py:33). `_mask_panel` (l.593-615) has **no `dropna`** on the symbol axis; `sel` keeps only `is_common_ticker` survivors. Behavioral: `test_label_is_masked_at_its_own_timestamp_only`, `test_ticker_rule_symbols_absent_in_every_window_threshold_failures_kept[cal/read]` (asserts the two windows' axes are identical), `test_exclude_non_common_false_drops_no_symbol` — passed. |
| 5 | LS-4 backtester drop-out timing: `price_dataset` unfiltered; ineligible at the next rebalance, sold at the next bar's open | ✓ VERIFIED | `test_holding_that_leaves_the_universe_is_sold_at_the_next_rebalance` passed: weight > 0 at R1, == 0 at R2, a Sell order at `bars[R2+1]`, and **zero** sells in `(R1+1, R2]`. `test_backtester_runs_with_wrapped_factors_and_never_selects_junk` passed: penny/illiquid/warrant weights are exactly 0 on every rebalance row with a teeth assertion that the book is non-empty. |
| 6 | Config round trip: `get_config()` shape, loader rebuild for factor/model/backtester, identical `weights.zarr` | ✓ VERIFIED (see Deviation 3 note) | `get_config` (l.634-647) emits exactly `{name, factor, min_price, min_dollar_volume, window, exclude_non_common}`. Model-level round trip is asserted **exactly** (`rebuilt_config == original_config`) in the tracer. Backtest-level: `xr.testing.assert_identical` on `weights.zarr` plus exact equity equality; config equality is asserted with the run-re-derived dates normalised away. The verifier independently reproduced that this divergence is pre-existing and wrapper-independent — see Deviations below. |
| 7 | Non-KunQuant inner factor and double wrap refused with `TypeError` naming class + reason | ✓ VERIFIED | `__init__` l.231-253, three ordered refusals. `test_a_polars_factor_is_refused`, `test_double_wrapping_is_refused`, `test_a_zero_window_is_refused` passed. |
| 8 | Ticker rule measured against `universe.parquet` (14,481 `us_all` symbols); named survive/exclude locks hold | ✓ VERIFIED — verifier re-measured | Verifier's own script against the real parquet: **14,481 distinct `us_all` symbols**; per-group counts `nasdaq_fifth_letter 2310, six_char_warrant 13, delimited_suffix 1056, when_issued_or_called 71, zzzt 7, xtest 72, zxyz 1, preferred 0, baby_bond 0`; **union 3,519 / 14,481 = 24.3%, 10,962 survivors** — identical to the comment block above `NON_COMMON_TICKER_PATTERNS`. All 33 survive-locks pass, all 21 exclude-locks pass (verifier-computed: 0 violations either way; 31 survive-locks present in the roster). |
| 9 | Falsifiers run; index constituents never excluded; uncorroborated 5-letter W/R/U reviewed; allowlist decision recorded | ✓ VERIFIED — verifier re-measured | Falsifier 1 (verifier-run): all **nine** pattern groups × **966** distinct `sp500_constituent`+`nasdaq100_constituent` symbols → **0 matches**. Falsifier 2 (verifier-run): **284** uncorroborated, median span **1.87y**, **36** ≥ 5y. No `COMMON_TICKER_ALLOWLIST` exists in the module, consistent with the recorded "none is common". Verifier spot-checked the 15 longest-listed: every one ends in W or U (warrant/unit fifth character) and 10/15 have a 3-char root present in the roster — none has a common-stock shape. |
| 10 | DL heads work across windows by design (no documented limitation) | ✓ VERIFIED | `test_dl_head_predicts_across_windows_when_a_trained_symbol_leaves_the_universe` passed with a real `MLPRegressor`: DROPOUT stays on the axis as an all-NaN column, `predict_panel` raises nothing, its predictions are NaN, and a teeth assertion requires at least one finite prediction. `example/universe.md` states this as "不是一条限制". |
| 11 | Gated suites keep their pre-change result; the 3 known `amount` failures deselected by node ID | ✓ VERIFIED — verifier re-ran | Verifier ran the plan's exact Task 3 gate (22 files, 3 `--deselect`, no pipe): **391 passed, 3 deselected, 0 failed, 96.44s, exit 0**. Planning baseline was 305 passed / 3 failed on 20 files; the 3 known failures remain the only deselected items. `tests/test_universe_filtered_factor.py --collect-only` → **84 tests collected**, matching the SUMMARY's 84. |

**Score:** 11/11 truths verified (0 present-but-behavior-unverified).

### Required Artifacts

| Artifact | Expected | Status | Details |
|---|---|---|---|
| `quantlab/factor/universe_filter.py` | Wrapper class, mask, rewrite, masking, lookback widening, stream, get/from_config | ✓ VERIFIED | 683 lines; contains `class UniverseFilteredFactor(FactorKunQuant)`; every contracted method present (`compute_universe_mask`, `is_common_ticker`, `cal`, `read`, `init_stream`, `cal_stream`, `_mask_panel`, `_get_factor_func`, `get_config`, `from_config`, `__repr__`). Imported by fixtures, tests and (by dotted path) the loader. |
| `quantlab/utils/module.py` | `from_config` dispatch before the plain path | ✓ VERIFIED | l.103-111: deep-copy → `get_cls_from_path` FIRST → `callable(from_config)` dispatch → unchanged legacy path. Only `UniverseFilteredFactor` declares `from_config` repo-wide (verified by grep), so no existing class is silently re-routed; `test_config_roundtrip.py` and `test_extensibility_contract.py` stayed green in the verifier's gate run. |
| `tests/universe_fixtures.py` | Importable 16-symbol store writer, tiny KunQuant factor, wrapped-model factory | ✓ VERIFIED | 386 lines; `class RankCloseFactor(FactorKunQuant)` with `Rank` + two `WindowedAvg` outputs; `SYMBOLS16`, `write_universe_store`, `hand_panel`, `pandas_universe_mask` (an independent pandas restatement of LS-1, sharing no code with the implementation), `make_wrapped_model`. Configs built directly, no `quantlab/config` factory import (D-32 honoured). |
| `tests/test_universe_filtered_factor.py` | Full matrix | ✓ VERIFIED | 1,157 lines, **84 collected, 84 passing**. Covers every behavior block the plan lists. Several tests carry explicit anti-vacuity ("teeth") assertions. |
| `example/universe.md` | Chinese doc incl. corrected DL-head cost claim | ✓ VERIFIED | 239 lines. Contains 一句话, 怎么用, LS-1/2/3 语义, 标的轴规则 with the DL-training cost stated plainly ("masked cells are zero-filled by `_preprocess` and ARE trained on… pre-existing `DLModel` behaviour… prediction and selection unaffected"), LS-4 timing table, LS-5, the measured ticker table, 6 限制 including the T-p91-03 fingerprint gap, and the `UniverseMask` contrast. Cited line references spot-checked: `xgb.py` `_to_rows` finite filter ✓, `mlp.py` `torch.nan_to_num(nan=0.0)` ✓, `model.py:1069` label-tensor `_preprocess` ✓, `model.py:567` all-NaN → NaN prediction ✓. |
| `example/README.md` | Index row under 第三步 | ✓ VERIFIED | Row present at line 33, inside the 第三步 table between `factor.md` and `model.md`. |

### Key Link Verification

| From | To | Via | Status | Details |
|---|---|---|---|---|
| universe_filter.py | inner factor config | `config` property returns `self.factor.config` | ✓ WIRED | l.290-306, getter returns the same object; setter delegates and re-widens. Model `_reset_factors_config` and backtester `_redate_factors` writes therefore land on the inner — exercised live by the DL cross-window test and the backtest run. |
| universe_filter.py | KunQuant graph | `_get_factor_func` rewrites a FRESH inner graph | ✓ WIRED | l.340-349 calls `self.factor._get_factor_func()` every time (no caching), sets `_uses_mask`. Both `_make` and `_make_stream` route through it. |
| module.py | universe_filter.py | `load_factor_from_config` → `cls.from_config` → recursive `load_factor_from_config` | ✓ WIRED | Exercised end-to-end by the model rebuild and the backtester run-dir rebuild, which reproduced byte-identical weights. |
| universe_filter.py | model.py | all-NaN features → NaN predictions; only ticker-rule symbols leave the axis | ✓ WIRED | `is_common_ticker` gate in `_mask_panel`; `DLModel._align_prediction_symbols` (model.py:1176) never raises in the cross-window test. |
| universe_filter.py | acquisition/universe.py | imports `_PREFERRED_SHARE_PATTERN` / `_BABY_BOND_PATTERN` | ✓ WIRED | l.72-75 import, used as the last two entries of `NON_COMMON_TICKER_PATTERNS` (single source of truth, no copied regex). |

### Data-Flow Trace (Level 4)

| Artifact | Data variable | Source | Produces real data | Status |
|---|---|---|---|---|
| `compute_universe_mask` | `close`, `volume` | `config.dataset.get_xarray_dataset()` RAW columns | Yes — rule matrix asserts real 1.0/NaN patterns from hand-built values | ✓ FLOWING |
| `cal()` | `input_dict[MASK_INPUT]` | mask `.reindex(timestamp, symbol)` (label-aligned, never positional) → float32 into `kr.runGraph(..., 0, num_time)` | Yes — rank output matches an independent pandas reference within 1e-5 | ✓ FLOWING |
| `cal_stream()` | mask row | rolling deque of `close*volume`, pushed via `queryBufferHandle` before `run()` | Yes — stream mask row equals the pandas reference at every bar | ✓ FLOWING |
| `_get_features/_get_labels` | every variable | inner transform → `where(mask.notnull())` | Yes — in-universe cells finite, out-of-universe NaN, no hardcoded fallback | ✓ FLOWING |

### Behavioral Spot-Checks (verifier-run)

| Behavior | Command | Result | Status |
|---|---|---|---|
| Full plan gate (22 files, 3 deselected) | `uv run pytest … --deselect ×3 -q -p no:cacheprovider` | `391 passed, 3 deselected in 96.44s`, exit 0 | ✓ PASS |
| Untouched model/backtest/config layers | `git diff --stat 4fa4b36 -- quantlab/base/model.py quantlab/base/backtest.py quantlab/base/config.py quantlab/ml_model quantlab/backtest` | empty, exit 0 | ✓ PASS |
| Universe test count | `pytest tests/test_universe_filtered_factor.py --collect-only -q` | `84 tests collected` | ✓ PASS |
| Ticker rule re-measurement | verifier script over `data/data/reference/universe.parquet` | 14,481 / union 3,519 / 10,962 survivors; 0 lock violations | ✓ PASS |
| Falsifier 1 re-run | same script, 966 index constituents × 9 groups | 0 matches in every group | ✓ PASS |
| Falsifier 2 re-run | same script | 284 uncorroborated, median 1.87y, 36 ≥ 5y | ✓ PASS |
| Deviation-3 counter-probe (unwrapped run→rebuild) | verifier probe on plain `tests.backtest_fixtures` | `configs equal after a RUN? False`; saved factor `2024-02-12..2024-03-18` vs rebuilt `2024-01-01..2024-02-09`; dataset start `2024-02-07` vs `2023-12-27`; equal once factor/label dates normalised → `True` | ✓ PASS |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|---|---|---|---|---|
| — | — | `TODO`/`FIXME`/`XXX`/`TBD`/`HACK`/`placeholder`/`pytest.skip`/`xfail` | — | **None found** across all five new/modified source files. No test is skipped or xfailed; the 84 universe tests all execute. |

### Deviations Assessed (all five SUMMARY-declared deviations checked independently)

| # | Executor's claim | Verifier finding |
|---|---|---|
| 1 | The contracted `RuntimeError` could not fire; `get_features()` hit a bare `AttributeError` first, so a `_get_xarray_dataset()` override was added | **Accurate and acceptable.** `XrBackend.__init__` (backend.py:16) does not set `self.data`; it is first assigned in `read()` (l.25). The override (l.583-591) restores the contracted behaviour and `test_get_features_before_cal_raises` passes. Strictly better than the contract, not a weakening. |
| 2 | `expected_fill in set(sold.tolist())` could never pass because `.tolist()` on `datetime64[ns]` yields ints | **Accurate.** Verifier reproduced: `type(a.tolist()[0])` is `int` (`1704412800000000000`), membership `False`. The fix compares as `datetime64` and still asserts both the positive fill bar and the absence of early sells — the behavioural claim is not weakened. |
| 3 | Post-run config equality is not an invariant; compared with re-derived dates normalised away | **Accurate, and independently reproduced by the verifier on an UNWRAPPED backtester** (see the counter-probe row above): identical divergence shape with no wrapper anywhere. Corroborated structurally — `run()` calls `_redate_factors` (backtest.py:488) before `config.json` is written (l.1657), and the only existing test asserting full config equality (`test_rebuilt_backtester_has_the_same_class_config_class_and_config`) runs on a backtester that was never run. The load-bearing claim (identical `weights.zarr` + equal equity) is still asserted exactly. |
| 4 | Fixture additions (REENTRY, `hand_panel`, `perturb_never_in_universe`, `config_cls`/mode/dataset passthroughs) | **Accurate and within the plan's explicit allowance.** All four are present in `tests/universe_fixtures.py` and each is consumed by a named test. |
| 5 | Planner said 46 uncorroborated tickers ≥ 5y; measurement found 36, recorded as measured | **Accurate.** Verifier's independent measurement also returns **36** (median 1.87y over 284). Recording the measured number rather than the planner's estimate is the correct call. |

### Informational Notes (not gaps)

1. **One truth rests partly on human judgment.** "None of the 284 uncorroborated 5-letter W/R/U tickers is a common stock" is a reviewed judgment, not a mechanical result. The verifier reproduced the mechanical half (falsifier 1 = 0/966, the 284 set itself) and spot-checked the 15 longest-listed: all end in `W` or `U` and 10/15 have a 3-char root in the roster, consistent with SPAC warrants/units. No counter-example found. Recorded for awareness; the survive/exclude lock tests pin the objective part.
2. **T-p91-03 remains `accept`** by design: the backtest data fingerprint covers `data_columns`, not the RAW close/volume the mask reads. Documented in `example/universe.md` 限制 #3. Closing it would require editing the backtest layer, which this task forbids.
3. **The `read`-strategy caveat is real and documented**: `save()`/`update()` persist pre-mask values, so a `factor_data_strategy="read"` store must be written through the wrapper. Stated in both the module docstring and 限制 #2.

### Gaps Summary

None. Every must-have truth is verified against the codebase with evidence the verifier produced itself: the plan's exact regression gate (391 passed / 3 deselected / exit 0), the pinned untouched-layer `git diff` (empty), an independent re-measurement of the ticker rule and both falsifiers from the real `universe.parquet`, and a counter-probe proving the one weakened assertion is a pre-existing, wrapper-independent property of the model/backtest layers.

---

_Verified: 2026-09-15T20:20:00Z_
_Verifier: Claude (gsd-verifier)_
