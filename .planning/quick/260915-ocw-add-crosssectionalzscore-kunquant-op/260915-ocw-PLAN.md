---
phase: quick-260915-ocw
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - quantlab/my_ops/preprocess.py
  - tests/test_cross_sectional_zscore.py
  - example/factor.md
autonomous: true
requirements: [260915-ocw]

estimate:
  tokens: 70000
  raw_tokens: 70000
  tasks: 3
  confidence: low

must_haves:
  truths:
    - "In a KunQuant graph compiled with TS layout and run via `kr.runGraph(executor, module, inputs, 0, num_time)` (start=0), `CrossSectionalZScore(x)` matches the pandas reference `df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1, ddof=1), axis=0)` to within 2e-4 with an identical NaN pattern. This holds in three setups: x is a graph `Input`, x is an intermediate node (`WindowedAvg(close, 5)`), and the output feeds a time-series op (`WindowedAvg(CrossSectionalZScore(close), 3)`)."
    - "The same three outputs compiled with STREAM layout and driven bar by bar through `kr.StreamContext` (pushData / run / getCurrentBuffer) match the same pandas reference within 2e-4, with an identical NaN pattern."
    - "Degenerate cross-sections: an all-NaN row, a row with exactly one valid value, and a constant row each produce a fully NaN output row. A NaN input produces a NaN output in the same position. Every row with at least 2 valid, non-constant values has an output nanmean of about 0 and a sample std (ddof=1) of about 1."
    - "`CrossSectionalZScore` subclasses KunQuant `GenericCrossSectionalOp`, and therefore `CrossSectionalOp`. It is not a `CompositiveOp`, and `generate_body` never reads `self.attrs`."
    - "The class docstring, in Chinese, records the semantics and the four pitfalls: the KunQuant 0.1.11 `CrossSectionalDataHolder` start>0 bug, which is safe here because `FactorKunQuant.cal` always passes start=0; C++ dedup in `CodegenCpp.py` by class name plus layout, so no attrs and one class per parameter set; SIMD symbol-count alignment in both TS and STREAM layouts; and that this op and `WindowedZScore` normalize along different axes, a design choice rather than a replacement."
    - "No factor class and no factor-kunquant test changes. `git status --porcelain -- quantlab/factor quantlab/base/factor.py tests/test_factor_kunquant.py` is empty, and `uv run pytest tests/test_factor_kunquant.py` still gives exactly the 3 baseline failures with 7 passing."
    - "The '现在还没有的东西' section of `example/factor.md` no longer claims the repo has no cross-sectional Z-score operator. It now says the operator exists at `quantlab/my_ops/preprocess.py:CrossSectionalZScore`, that no factor class uses it by default (US-equity factors still emit raw values, D-09), and that wiring it in is still deferred to ARCH-02. The xarray snippet stays."
  artifacts:
    - path: quantlab/my_ops/preprocess.py
      provides: "class CrossSectionalZScore(GenericCrossSectionalOp): NaN-aware cross-sectional (x - mean) / sample std; generate_head returns an empty string, generate_body is the C++ loop"
      contains: "class CrossSectionalZScore(GenericCrossSectionalOp)"
    - path: tests/test_cross_sectional_zscore.py
      provides: "Two module-scoped compiled modules (TS batch, STREAM), each with three outputs, compared against pandas; degenerate-row checks; structural lock"
      contains: "CrossSectionalZScore"
    - path: example/factor.md
      provides: "Dated update to the '现在还没有的东西' paragraph"
      contains: "CrossSectionalZScore"
  key_links:
    - from: tests/test_cross_sectional_zscore.py
      to: quantlab/my_ops/preprocess.py
      via: "from quantlab.my_ops.preprocess import CrossSectionalZScore"
      pattern: "from quantlab.my_ops.preprocess import CrossSectionalZScore"
    - from: quantlab/my_ops/preprocess.py
      to: "KunQuant GenericCrossSectionalOp codegen (passes/CodegenCpp.py)"
      via: "generate_body() C++ snippet using T, num_stocks, input_0[i], output_0[i]"
      pattern: "def generate_body"
---

<objective>
Add a cross-sectional Z-score KunQuant operator, `CrossSectionalZScore`, to `quantlab/my_ops/preprocess.py`. Lock its semantics and known pitfalls with a new test file, and correct the one doc paragraph that says no such operator exists.

Purpose: US-equity factors are traded with cross-sectional strategies (D-09) and need a cross-sectional normalization primitive that runs inside the compiled KunQuant graph, in both batch and streaming mode. `WindowedZScore` is the time-series counterpart and stays as it is. Scope is locked by the user to the operator, its tests, and one doc note. No factor class is rewired, and the normalization-matrix test keeps its expectations.

Output: the `CrossSectionalZScore` class, `tests/test_cross_sectional_zscore.py`, and a dated update in `example/factor.md`.
</objective>

<execution_context>
@~/.claude/gsd-core/workflows/execute-plan.md
@~/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@CLAUDE.md
@quantlab/my_ops/preprocess.py
@tests/test_factor_kunquant.py
@example/factor.md

Reference implementation (validated this session, read it first; it is not in the repo):
/private/tmp/claude-501/-Users-daizhaorong-projects-quantlab/21e1e125-d880-439a-a635-3a6d94ada2f5/scratchpad/cszscore/proto.py
- Lines 13-41: the `CSZScore` class. Its `generate_body` C++ is the exact body to ship.
- Lines 44-51: the three-output graph (`z_raw`, `z_ma5`, `ma3_of_z`).
- Lines 54-61: the pandas reference.
- Lines 71-78: the seeded panel with the degenerate rows.
- Lines 82-86: batch compile and run. Lines 98-109: stream compile and push/run loop.
If that scratchpad path is gone, the prose in Task 1 and Task 2 is enough to rebuild it.

Verified facts (planner checked on 2026-09-15; no need to re-derive):
- `from KunQuant.ops import *`, already the first import family in preprocess.py, exports `GenericCrossSectionalOp`, `CrossSectionalOp` and `WindowedAvg`, so preprocess.py needs no new import.
- `GenericCrossSectionalOp` lives in `KunQuant/ops/MiscOp.py`. `CrossSectionalOp` and `CompositiveOp` live in `KunQuant/Op.py`. Upstream subclass example: `DiffWithWeightedSum` in MiscOp.py, whose `__init__` calls `super().__init__([v, w], None)`.
- Test compile conventions come from `quantlab/base/factor.py` `_make` and `_make_stream` (L323-354): `cfake.compileit([(module_name, Function, KunCompilerConfig(input_layout=..., output_layout=...))], lib_name, cfake.CppCompilerConfig())`, then `lib.getModule(module_name)`. Batch runs are always `kr.runGraph(executor, modu, input_dict, 0, num_time)` (L289). This is the start=0 guarantee the docstring cites.
- Baseline for `uv run pytest tests/test_factor_kunquant.py -q -p no:cacheprovider` at HEAD 38a9be1 is 3 failed, 7 passed. The failures are `test_stock_to_kunquant_synthesizes_amount_as_adjusted_dollar_volume`, `test_stock_to_kunquant_without_amount_leaves_arrays_unchanged` and `test_alpha101_stock_bugfix_batch_cal_returns_xarray_dataset`.
</context>

<tasks>

<task type="tracer" tdd="true">
  <name>Task 1: Tracer, CrossSectionalZScore op plus one batch start=0 pandas-match test end to end</name>
  <files>quantlab/my_ops/preprocess.py, tests/test_cross_sectional_zscore.py</files>
  <read_first>
    - /private/tmp/claude-501/-Users-daizhaorong-projects-quantlab/21e1e125-d880-439a-a635-3a6d94ada2f5/scratchpad/cszscore/proto.py (full file)
    - quantlab/my_ops/preprocess.py (docstring style of WindowedZScore)
    - tests/test_factor_kunquant.py lines 1-17 (module docstring style)
  </read_first>
  <behavior>
    - A TS-layout module whose graph is `Output(CrossSectionalZScore(Input("close")), "z_raw")`, run with start=0 on a seeded 40x16 float32 panel, matches the pandas cross-sectional z-score (ddof=1, computed in float64) within atol 2e-4 on finite cells, with an exactly equal NaN mask.
    - Before the class exists, the test module fails at import (red). After it is added, the test passes (green).
  </behavior>
  <action>
RED first: create `tests/test_cross_sectional_zscore.py`, run it, and confirm it fails with an ImportError on `CrossSectionalZScore`. Then implement. The plan type is execute, not tdd. If a GSD red-evidence gate asks for TAP anyway, project the real pytest run into TAP with a throwaway script in the scratchpad, never in the repo, and never synthesize counts.

Operator, in `quantlab/my_ops/preprocess.py`: append `class CrossSectionalZScore(GenericCrossSectionalOp)` after `WindowedZScore`, and leave `WindowedZScore` byte-identical. `__init__(self, v: OpBase) -> None` calls `super().__init__([v], None)`. `generate_head(self) -> str` returns an empty string. `generate_body(self) -> str` returns the C++ body from proto.py lines 23-41, copied verbatim. Pass 1 sums the non-NaN `input_0[i]` over `num_stocks` into `T sum` and counts them in a `size_t n`; `mean` is `sum / n`, or NAN when n == 0. Pass 2 accumulates the squared deviations from `mean` over the non-NaN values into `ss`; `sd` is `std::sqrt(ss / (n - 1))` when n > 1, else NAN. Pass 3 writes `output_0[i]` as NAN when the input is NaN or when `!(sd > 0)`, otherwise `(v - mean) / sd`. Pass 3 sends sd == 0 and a NaN sd to NaN, which is what yields whole-row NaN for constant rows and rows with fewer than 2 valid values. No new import: the existing `from KunQuant.ops import *` already exports `GenericCrossSectionalOp`. `generate_body` must not read `self.attrs`.

Class docstring, in Chinese, matching the WindowedZScore docstring tone. It must cover all of the following:
(a) 截面 Z 值标准化: at each time point, across all symbols, the NaN-aware mean and sample std (ddof=1, the same as pandas `.std()` and KunQuant `WindowedStddev`), with output `(x - mean) / sd`.
(b) Missing values: a NaN input gives a NaN output. If a cross-section has fewer than 2 valid values, or sd == 0 (a constant cross-section), the whole row is NaN. The op does no fillna; as with WindowedZScore, filling is the caller's explicit job.
(c) Why it is a `GenericCrossSectionalOp` and not a `CompositiveOp`: decompose only unfolds into time-series ops, so the op follows the route in upstream doc/NewOperators.md "Cross-Sectional Operators" and supplies a C++ loop body directly.
(d) Pitfall 1: in KunQuant 0.1.11, `CrossSectionalDataHolder` (cpp/Kun/LayoutMappers.hpp) computes `base_time` from `num_time` before `num_time` is assigned. As a result, `kr.runGraph(..., start>0, ...)` gives wrong, non-deterministic output for every `GenericCrossSectionalOp`. The built-in `Scale` is correct, and upstream main still has the bug. The only caller in this repo, `FactorKunQuant.cal` in quantlab/base/factor.py, always passes start=0, so it is safe. New callers must not pass a non-zero start.
(e) Pitfall 2: `passes/CodegenCpp.py` dedups the generated C++ function by class name plus layout only. `generate_body` therefore must not depend on attrs, and a parameterised cross-sectional op needs one class per parameter set.
(f) Pitfall 3: the symbol count must align with KunQuant's SIMD block width. This is a general KunQuant limit, not specific to this op, and it applies to both TS and STREAM layouts. On this aarch64 machine 16 symbols works and 13 does not, and `allow_unaligned` is unsupported on aarch64. See pitfall 1 under "常见坑" in example/factor.md. State only these measured facts; do not claim a block width.
(g) Axis: this op and WindowedZScore normalize along different axes (cross-sectional vs time-series). The choice follows strategy type; this op does not replace WindowedZScore, and the two must never be "aligned". No factor class uses it by default: US-equity factors still emit raw values per D-09, and wiring is deferred to ARCH-02. Cross-reference the example/factor.md section "标准化算子与截面 vs 时序".

Tests, in `tests/test_cross_sectional_zscore.py`, with English docstrings in the style of tests/test_factor_kunquant.py. The module docstring says what is locked, why there are only two compiled modules (compile takes several seconds each), why there are 16 symbols (SIMD alignment), and why only start=0 is exercised (the pitfall 1 bug is non-deterministic, so the file documents it instead of asserting it). Do not write any test that runs start>0.

Module constants: `_N_TIMES = 40`, `_N_SYMBOLS = 16`, and named row indices `_ALL_NAN_ROW = 7`, `_SINGLE_VALID_ROW = 9`, `_CONSTANT_ROW = 11`.

Helpers:
- `_panel()` uses `np.random.default_rng(0)` to build a float32 array `100 + cumsum(standard_normal((T, S)), axis=0)`, sets about 10% of cells NaN via `rng.random((T, S)) < 0.1`, then sets row 7 all NaN, row 9 columns 1: to NaN, and row 11 all 5.0. This mirrors proto.py lines 73-78.
- `_build_function()` uses a Builder with `close = Input("close")` and three Outputs: `CrossSectionalZScore(close)` as "z_raw", `CrossSectionalZScore(WindowedAvg(close, 5))` as "z_ma5", and `WindowedAvg(CrossSectionalZScore(close), 3)` as "ma3_of_z". It returns `Function(b.ops)`.
- `_pandas_reference(close)` returns a dict with the same three keys, computed on a float64 DataFrame with `mean(axis=1)` and `std(axis=1, ddof=1)`, plus `rolling(5).mean()` / `rolling(3).mean()` as in proto.py lines 54-61.
- `_assert_matches(got, want)` asserts `np.isnan(got)` equals `np.isnan(want)` exactly (`np.testing.assert_array_equal`), then runs `np.testing.assert_allclose` on the finite cells with atol=2e-4 and rtol=0.

Fixture: a module-scoped `batch_outputs` compiles ONE TS module holding all three outputs. Use module name "CrossSectionalZScoreBatch" and a unique lib name such as "test_cs_zscore_batch", with `KunCompilerConfig(input_layout="TS", output_layout="TS")` and `cfake.CppCompilerConfig()`. Create the executor with `kr.createMultiThreadExecutor(4)`, run `kr.runGraph(executor, modu, {"close": np.ascontiguousarray(close)}, 0, _N_TIMES)`, and return the panel plus a dict of copied numpy outputs.

Tracer test: `test_batch_start0_matches_pandas_for_graph_input` asserts `z_raw` against the reference.
  </action>
  <verify>
    <automated>cd /Users/daizhaorong/projects/quantlab && uv run pytest tests/test_cross_sectional_zscore.py -q -p no:cacheprovider && grep -c "class CrossSectionalZScore(GenericCrossSectionalOp)" quantlab/my_ops/preprocess.py && grep -c "CrossSectionalDataHolder" quantlab/my_ops/preprocess.py && grep -c "CodegenCpp" quantlab/my_ops/preprocess.py</automated>
  </verify>
  <done>The pytest run reports 1 passed and 0 failed. Each of the three grep counts is at least 1. `WindowedZScore` is unchanged (`git diff quantlab/my_ops/preprocess.py` shows only added lines). The work is committed with only the two task files staged by explicit path.</done>
</task>

<task type="auto">
  <name>Task 2: Expand tests to stream layout, intermediate-node input, the TS-op consumer, degenerate rows and a structural lock</name>
  <files>tests/test_cross_sectional_zscore.py</files>
  <read_first>
    - tests/test_cross_sectional_zscore.py (as left by Task 1)
    - /private/tmp/claude-501/-Users-daizhaorong-projects-quantlab/21e1e125-d880-439a-a635-3a6d94ada2f5/scratchpad/cszscore/proto.py lines 97-111 (stream loop)
  </read_first>
  <action>
Add a module-scoped `stream_outputs` fixture that compiles ONE STREAM module from the same `_build_function()`. Use module name "CrossSectionalZScoreStream", a unique lib name such as "test_cs_zscore_stream", and `KunCompilerConfig(input_layout="STREAM", output_layout="STREAM")`, the plain config validated in proto.py. The fixture creates `kr.createMultiThreadExecutor(4)` and `kr.StreamContext(executor, modu, _N_SYMBOLS)`, gets handles via `queryBufferHandle` for "close" and the three outputs, and loops over the 40 bars. Each bar does `pushData(h_close, np.ascontiguousarray(close[t]))`, then `run()`, then copies `getCurrentBuffer(h)[:_N_SYMBOLS]` into preallocated (T, S) float32 arrays. The copy matters because the buffer is reused. Keep the executor referenced for as long as the context lives, and return the dict.

Add these tests. Each gets an English docstring stating the property it locks.
1. `test_batch_start0_matches_pandas_for_intermediate_node_input`: `z_ma5` from batch_outputs matches the reference.
2. `test_batch_output_feeds_time_series_op`: `ma3_of_z` from batch_outputs matches the reference. This shows the cross-sectional output composes with a downstream `WindowedAvg`.
3. `test_stream_matches_pandas`: parametrized over ("z_raw", "z_ma5", "ma3_of_z") with those ids, comparing stream_outputs against the reference.
4. `test_degenerate_cross_sections_are_all_nan`: on batch `z_raw`, rows `_ALL_NAN_ROW`, `_SINGLE_VALID_ROW` and `_CONSTANT_ROW` are entirely NaN, and on every other row `np.isnan(output)` equals `np.isnan(input)`. Before asserting, the test checks that the panel really contains at least one non-degenerate row with a NaN cell, so the NaN-propagation branch cannot pass vacuously.
5. `test_valid_rows_have_zero_mean_unit_sample_std`: this checks the definition directly, independent of pandas. For every row of batch `z_raw` except the three degenerate ones, `np.nanmean` is within 1e-4 of 0 and `np.nanstd(ddof=1)` is within 1e-4 of 1. The panel is float32, so if 1e-4 is too tight on real output, widen to 5e-4 and state the measured max deviation in the docstring. Do not widen further without investigating.
6. `test_is_a_generic_cross_sectional_op_without_attrs`: a structural lock. `issubclass(CrossSectionalZScore, GenericCrossSectionalOp)` and `issubclass(CrossSectionalZScore, CrossSectionalOp)` hold, `issubclass(CrossSectionalZScore, CompositiveOp)` does not, and the substring `attrs` does not appear in `inspect.getsource(CrossSectionalZScore.generate_body)`. This locks pitfall 2. Import `GenericCrossSectionalOp` from `KunQuant.ops.MiscOp`, and `CrossSectionalOp` and `CompositiveOp` from `KunQuant.Op`.

The file totals exactly two `cfake.compileit` calls, one per fixture. Do not add tests that call runGraph with a non-zero start (pitfall 1 is non-deterministic, so it stays documented rather than asserted). Do not touch tests/test_factor_kunquant.py, any file under quantlab/factor, or quantlab/base. The parallel quick task 260915-o5y is editing quantlab/base/model.py and quantlab/base/backtest.py. Stage only tests/test_cross_sectional_zscore.py by explicit path; never use `git add -A`, `git commit -a` or `git stash`.
  </action>
  <verify>
    <automated>cd /Users/daizhaorong/projects/quantlab && uv run pytest tests/test_cross_sectional_zscore.py -q -p no:cacheprovider && test "$(grep -v '^\s*#' tests/test_cross_sectional_zscore.py | grep -c 'cfake.compileit(')" = "2" && uv run pytest tests/test_factor_kunquant.py -q -p no:cacheprovider 2>&1 | tail -5; test -z "$(git status --porcelain -- quantlab/factor quantlab/base/factor.py tests/test_factor_kunquant.py)" && echo UNTOUCHED_OK</automated>
  </verify>
  <done>`tests/test_cross_sectional_zscore.py` reports at least 9 passed and 0 failed: 1 tracer, 2 batch, 3 stream, 1 degenerate-rows, 1 moments, 1 structural. `tests/test_factor_kunquant.py` still reports exactly 3 failed and 7 passed, and the failing names are the three baseline names listed in context. UNTOUCHED_OK is printed. The work is committed with only the test file staged.</done>
</task>

<task type="auto">
  <name>Task 3: Dated doc update in example/factor.md, "现在还没有的东西"</name>
  <files>example/factor.md</files>
  <read_first>
    - example/factor.md lines 559-611 (the whole "标准化算子与截面 vs 时序" section; note the dated-correction convention, e.g. "**2026-09-07 已删除**" at L575)
    - quantlab/my_ops/preprocess.py (final CrossSectionalZScore docstring, so the doc matches it)
  </read_first>
  <action>
Edit only the "### 现在还没有的东西" subsection (around L599-610). Keep the heading. Replace the single paragraph at L601 with one paragraph that follows the file's dated-correction convention. It keeps the old claim visible as history, marked superseded with a bold dated tag "**2026-09-15 更新**", and then states the following:
- The operator now exists at `quantlab/my_ops/preprocess.py:CrossSectionalZScore`, a `GenericCrossSectionalOp` with a C++ loop body: NaN-aware sample std (ddof=1), and a fully NaN row when fewer than 2 values are valid or sd == 0.
- No factor class uses it by default. The four-cell matrix above is unchanged, US-equity factors still emit raw values (D-09), and wiring it into factor classes is still deferred to ARCH-01/ARCH-02.
- Before using it, read the class docstring's pitfalls, especially that a start>0 `runGraph` gives wrong results on KunQuant 0.1.11, and the symbol-count alignment.

Keep the existing "如果你现在就需要，在 xarray 层做..." sentence, the xarray snippet and the "（这段是示意...）" note exactly as they are. Add one short sentence after the snippet note: xarray `.std()` defaults to ddof=0, which differs from the operator (ddof=1) by a factor of sqrt(n/(n-1)), so pass `ddof=1` to match the operator. Do not edit any other section of example/factor.md, including the pitfalls list and the four-cell matrix.
  </action>
  <verify>
    <automated>cd /Users/daizhaorong/projects/quantlab && grep -c "CrossSectionalZScore" example/factor.md && grep -c "2026-09-15" example/factor.md && grep -cF 'cs_z = (panel - panel.mean(dim="symbol")) / panel.std(dim="symbol")' example/factor.md && grep -c "ddof=1" example/factor.md && git diff --stat -- example/factor.md</automated>
  </verify>
  <done>Four conditions hold in example/factor.md: `CrossSectionalZScore` appears at least once, `2026-09-15` appears at least once, the xarray snippet line appears exactly once (unchanged), and `ddof=1` appears at least once. `git diff` touches only lines inside the "现在还没有的东西" subsection, and the work is committed with only example/factor.md staged.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| Python op definition -> generated C++ | `generate_body()` returns a static string literal compiled by KunQuant's cfake JIT. No user or config input reaches the generated source. |
| Factor panel -> compiled kernel | Numeric float32 arrays produced by in-repo datasets. No external or network input is added by this plan. |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-260915-ocw-01 | Tampering | `CrossSectionalZScore.generate_body` codegen | low | mitigate | The body is a constant literal that never reads `self.attrs`, so no runtime value can be spliced into C++. Task 2's structural test asserts that `attrs` is absent from its source, which also prevents the CodegenCpp dedup collision (pitfall 2). |
| T-260915-ocw-02 | Information disclosure / integrity of results | `kr.runGraph` with start>0 (KunQuant 0.1.11 `CrossSectionalDataHolder` bug) | medium | mitigate | Documented in the class docstring and in example/factor.md. The only caller, `FactorKunQuant.cal`, passes start=0, and the tests lock the start=0 correctness against pandas. |
| T-260915-ocw-03 | Denial of service | Out-of-bounds read when the symbol count is not SIMD-aligned | low | accept | KunQuant rejects misaligned shapes at open ("Bad shape at open"). Documented as pitfall 3, and the tests use 16 symbols. |
| T-260915-ocw-SC | Tampering | Package installs | low | accept | No npm/pip/cargo installs. KunQuant, pandas and numpy are already locked dependencies, so no package-legitimacy gate applies. |
</threat_model>

<verification>
- `uv run pytest tests/test_cross_sectional_zscore.py -q -p no:cacheprovider` reports at least 9 passed and 0 failed.
- `uv run pytest tests/test_factor_kunquant.py -q -p no:cacheprovider` reports 3 failed and 7 passed, the same three baseline failure names with no new failures. The normalization-matrix test still passes and is unchanged.
- `git status --porcelain -- quantlab/factor quantlab/base/factor.py tests/test_factor_kunquant.py` is empty.
- The plan never uses `git stash` and never stages quantlab/base/model.py or quantlab/base/backtest.py, which the parallel task 260915-o5y owns.
</verification>

<success_criteria>
- `CrossSectionalZScore` computes the NaN-aware cross-sectional z-score (ddof=1) inside compiled KunQuant graphs, in both TS batch (start=0) and STREAM layouts, and matches pandas within 2e-4 with an identical NaN pattern.
- Its four pitfalls are written in its Chinese docstring. The ones that can be tested (degenerate rows, no attrs in codegen, start=0 correctness) are locked by tests.
- Scope stayed locked: no factor class, factor test or normalization-matrix expectation changed. `example/factor.md` now tells readers the operator exists but is not wired in.
</success_criteria>

<output>
Create `.planning/quick/260915-ocw-add-crosssectionalzscore-kunquant-op/260915-ocw-SUMMARY.md` when done
</output>
