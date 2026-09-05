---
phase: 03-factor-computation-kunquant-polars
plan: 05
subsystem: factor-layer
tags: [kunquant, streaming, simd, aarch64, polars, xarray, d-03, factor-02, readme]

# Dependency graph
requires:
  - phase: 03-factor-computation-kunquant-polars
    plan: 01
    provides: "The spot_kline_zarr 8-symbol synthetic fixture and the tests/test_factor_stream.py scaffold this plan fills, plus the my_ops decompose() fix without which no graph compiles at all"
  - phase: 03-factor-computation-kunquant-polars
    plan: 02
    provides: "The Factor/FactorKunQuant split that keeps init_stream/cal_stream/_make_stream on the KunQuant subclass, and the Factor-typed DLConfig.factors this plan's proof test populates with two backends"
  - phase: 03-factor-computation-kunquant-polars
    plan: 03
    provides: "The dual-market Alpha158 (Alpha158SpotKline/Alpha158Stock) the README's KunQuant example points at, and the prior observation from the execution side that KunQuant's SIMD block width leaks into what data shapes the pipeline accepts"
  - phase: 03-factor-computation-kunquant-polars
    plan: 04
    provides: "base/factor_polars.py:FactorPolars and factor/momentum.py:Momentum — the live Polars-backed object this plan's interchangeability proof pairs against a live KunQuant factor"
provides:
  - "A runnable KunQuant streaming path on aarch64: FactorKunQuant._make_stream() lets KunQuant select the SIMD block width per architecture (BUG-02 closed)"
  - "tests/test_factor_stream.py — the first execution of init_stream()/cal_stream() in this repository's history, as a 60-bar replay proving incremental updates (FACTOR-02)"
  - "tests/test_factor_hierarchy.py:test_kunquant_and_polars_factors_are_interchangeable_in_one_dlconfig — D-03 proven with two live objects driven through base/model.py's real call sequence, with zero type-based branching"
  - "README ## Factor Backends — the two-backend contract and step-by-step instructions for adding a factor to each"
affects: [04-return-model, 05-portfolio-optimization, live-data-feed, streaming-factors]

actuals:
  tokens: 6000
  tasks: 3
  commits: 3

tech-stack:
  added: []
  patterns:
    - "Architecture-dependent compiler arguments are left UNSET rather than pinned: KunQuant's own per-arch default is both correct on the new architecture and byte-identical on the old one, so deleting the argument is strictly safer than branching on platform.machine()"
    - "Behavioural gate over grep: an acceptance criterion that greps for the absence of an identifier is paired with a test that physically cannot pass on this machine unless the fix landed (mutation-verified by restoring the argument)"
    - "Replay-driven streaming tests source their per-bar inputs from the production Dataset.to_kunquant() adapter and the SAME Dataset instance the factor holds, so symbol ordering matches the stream context's by construction rather than by coincidence"
    - "An interchangeability proof's strongest assertion is the absence of code: no runtime type check anywhere in the test body, enforced by a grep over inspect.getsource()"

key-files:
  created: []
  modified:
    - base/factor.py
    - tests/test_factor_stream.py
    - tests/test_factor_hierarchy.py
    - README.md

key-decisions:
  - "The SIMD block width is deleted, not branched on: KunQuant's Driver.py already selects the correct value per architecture, so an explicit platform branch in quantlab would duplicate logic that upstream owns and would drift when KunQuant adds an architecture"
  - "The streaming test asserts inequality between two consecutive KMID snapshots, not merely the absence of an exception — a stale or constant buffer is the failure mode that looks like success (T-03-05-02)"
  - "The interchangeability proof replicates base/model.py's four factor interactions inline rather than instantiating BaseModel, which is abstract and would pull torch and wandb into a factor-layer test"
  - "Where Task 3's action text (say it in the docstring) contradicted its own acceptance grep (the identifier must not appear in inspect.getsource, which includes the docstring), the grep won — same resolution 03-04 reached for the KunQuant string, for the same reason: the grep is the enforceable half"

patterns-established:
  - "Mutation-checked fixes (continued from 03-02/03-03/03-04): restoring the deleted compiler argument was observed to fail exactly the two new streaming tests and nothing else, then reverted"
  - "README documents a layer's extension points as a numbered how-to per backend, pointing at a named worked-example file rather than inlining a code sample that would drift"

requirements-completed: [FACTOR-02, FACTOR-03, FACTOR-04]

coverage:
  - id: D1
    description: "FactorKunQuant._make_stream() compiles a streaming graph on aarch64 — the hardcoded x86-only SIMD block width is gone, and x86_64 behaviour is unchanged because 8 is KunQuant's own float default there"
    requirement: FACTOR-02
    verification:
      - kind: integration
        ref: "tests/test_factor_stream.py#test_init_stream_binds_a_buffer_handle_for_every_declared_name (mutation-checked: restoring the argument fails it with RuntimeError: Blocking length 8 is not supported for float on aarch64)"
        status: pass
      - kind: other
        ref: "uv run python -c \"'blocking_len' not in inspect.getsource(FactorKunQuant._make_stream) and 'partition_factor=8' in it and 'SIMD block width' in it\" -> ok"
        status: pass
    human_judgment: false
  - id: D2
    description: "cal_stream() accepts a full 60-bar historical replay one timestamp at a time without raising, and produces a (1, num_symbols) factor update per step (FACTOR-02, ROADMAP Success Criterion 2)"
    requirement: FACTOR-02
    verification:
      - kind: integration
        ref: "tests/test_factor_stream.py#test_cal_stream_replay_produces_incremental_factor_updates (sizes == {timestamp: 1, symbol: 8}; data_vars == [KMID, STD5, VOLUME0]; finite count > 0)"
        status: pass
    human_judgment: false
  - id: D3
    description: "Successive cal_stream() calls produce different factor values — the updates are genuinely incremental, not a constant or a stale buffer (T-03-05-02)"
    requirement: FACTOR-02
    verification:
      - kind: integration
        ref: "tests/test_factor_stream.py#test_cal_stream_replay_produces_incremental_factor_updates — np.array_equal(previous_kmid, last_kmid) is False"
        status: pass
    human_judgment: false
  - id: D4
    description: "One DLConfig holding a KunQuant factor and a Polars factor drives .cal().get_features(), ._get_factor_names(), .get_config() and ._reset_dataset_config() uniformly across both backends, and their outputs combine into one xr.Dataset (D-03)"
    requirement: FACTOR-04
    verification:
      - kind: integration
        ref: "tests/test_factor_hierarchy.py#test_kunquant_and_polars_factors_are_interchangeable_in_one_dlconfig"
        status: pass
      - kind: other
        ref: "uv run python -c \"'isinstance' not in inspect.getsource(test) and 'hasattr' not in it\" -> ok; strengthened check 'type(' not in it -> ok"
        status: pass
    human_judgment: false
  - id: D5
    description: "README documents both factor backends and how to add a new factor to each, pointing at factor/momentum.py and factor/alpha158.py as the worked examples"
    requirement: FACTOR-03
    verification:
      - kind: other
        ref: "grep -c 'FactorPolars' README.md -> 4; grep -c '_get_factor_lazyframe' README.md -> 1; new ## Factor Backends section with per-backend numbered how-tos"
        status: pass
    human_judgment: false

# Metrics
duration: 15 min
completed: 2026-09-05
status: complete
---

# Phase 3 Plan 05: Streaming Factors and the Live Interchangeability Proof Summary

**A one-argument deletion that unbreaks KunQuant streaming compilation on Apple Silicon, the first execution of `init_stream()`/`cal_stream()` in this repository's history as a 60-bar replay proving genuinely incremental updates, and D-03's interchangeability proven with two live factor objects from two different backends driven through `base/model.py`'s own call sequence with zero type-based branching.**

## Performance

- **Duration:** 15 min
- **Started:** 2026-09-05T18:47Z (approx.)
- **Completed:** 2026-09-05T19:01:50-04:00
- **Tasks:** 3
- **Files modified:** 4 (all modified, none created)

## Accomplishments

- **Closed BUG-02 and made the streaming interface runnable on this project's own dev machine.** `FactorKunQuant._make_stream()` pinned `blocking_len=8` — the x86_64 float default — into its `KunCompilerConfig`. `KunQuant/Driver.py` selects `{float: 8, double: 4}` with `simd_len ∈ {256, 512}` on x86_64 but `{float: 4, double: 2}` with `simd_len == {128}` on aarch64, and raises when the product misses. So `init_stream()`/`cal_stream()` raised `RuntimeError: Blocking length 8 is not supported for float on aarch64` immediately on Apple Silicon. Deleting the argument lets KunQuant pick per architecture; x86_64 behaviour is byte-identical, because 8 is exactly what it selects there. `_make()` had always omitted it, which is why the batch path worked while the streaming path was dead.

- **Executed `cal_stream()` for the first time in this repository's history — and proved it produces real incremental updates, not just an absence of exceptions.** A full-repo grep finds exactly two non-definition call sites, both in `backtest/test_strategy.py` and both commented out; graph construction, buffer wiring and the `pushData`/`run`/`getCurrentBuffer` loop had never run. `tests/test_factor_stream.py` now replays all 60 bars of the synthetic 8-symbol panel one timestamp at a time, sourcing each bar from the production `Dataset.to_kunquant()` adapter on the same `Dataset` instance the factor holds. It asserts the `(1, 8)` output shape, the exact `data_vars`, a finite-value count, **and inequality between two consecutive `KMID` snapshots** — the T-03-05-02 case where a stale or constant buffer would otherwise read as success. ROADMAP Phase 3 Success Criterion 2 is now satisfied by an executed replay rather than by the code's mere existence.

- **Made the fix's necessity a measured fact rather than a claim.** Restoring the deleted argument makes exactly the two new streaming tests fail with the aarch64 `RuntimeError`, and nothing else in the suite move (`2 failed, 1 passed` in that file; the 03-01 scaffold test is unaffected). The edit was then reverted with `git checkout`. The behavioural gate is real: the grep-style source criteria in Task 1 could not have caught a regression on their own, but this test cannot pass on aarch64 unless the fix is present.

- **Turned D-03 from a static claim into a live one.** 03-02 proved the *shape* was right — `Factor` carries the whole `base/model.py` call surface, `DLConfig.factors` is typed against it, the model layer names no backend. This plan proves the shape holds up: one real `DLConfig` holds an `Alpha158SpotKline` and a `Momentum`, and all four interactions `base/model.py:115-201` performs — `_reset_factors_config`'s date assignment plus `_reset_dataset_config()`, `_get_features_batch`'s `.cal().get_features()` into `xr.combine_by_coords`, `get_factor_names`'s `chain.from_iterable`, and `get_config`'s comprehension — run uniformly across both. The cross-backend combine flagged in the plan as the one real risk worked as measured during planning: both outputs merged into one `xr.Dataset` over shared `timestamp`/`symbol` dims carrying `KMID` and `momentum_5`, despite the `symbol` coordinate dtype differing between the two paths. No coordinate-normalization fix was needed in `FactorPolars.cal()`.

- **Gave a new contributor a way in.** README's new `## Factor Backends` section states the two-backend contract, the interchangeability guarantee, the xarray-only boundary, and — the actual point — numbered instructions for adding a factor to each backend, with `factor/momentum.py` and `factor/alpha158.py` named as the worked examples. It also records the streaming `Cannot find the buffer name` caveat, which is the trap a live-feed integration will hit first.

## Task Commits

Each task was committed atomically:

1. **Task 1: Let KunQuant select the architecture-correct SIMD block width** — `0a61d8e` (fix)
2. **Task 2: Batch-replay streaming smoke test — the first execution of `cal_stream()`** — `598abe9` (test)
3. **Task 3: Live two-backend interchangeability proof and README coverage** — `7647568` (test)

**Plan metadata:** see the `docs(03-05)` commit that carries this SUMMARY.

## Files Created/Modified

- `base/factor.py` — **modified.** One keyword argument deleted from `_make_stream()`'s `KunCompilerConfig`, plus an 11-line comment recording why it is absent. Deliberately phrased as "the SIMD block width" and never by the removed identifier, so Task 1's negative grep stays meaningful. `_make()`, `partition_factor=8`, the layouts and the `options` dict are untouched; nothing else in the file changed.
- `tests/test_factor_stream.py` — **modified.** Grew from 1 test to 3. The 03-01 scaffold self-test keeps its assertions verbatim (only its docstring's stale reference to the now-deleted identifier was reworded). Added: the module-level `_DATA_COLUMNS`/`_FACTOR_NAMES` pairing with the comment explaining why it is load-bearing, a `_stream_factor()` builder documenting the `symbols`-before-`Dataset` ordering rule, the `init_stream()` buffer-handle test and the 60-bar replay test.
- `tests/test_factor_hierarchy.py` — **modified.** Grew from 9 tests to 10. Added the live interchangeability proof plus two module constants; promoted `base.factor_polars`/`factor.momentum` to normal module-level imports now that both exist, and updated the module docstring's import-safety note accordingly.
- `README.md` — **modified.** New `## Factor Backends` section (backend comparison table, interchangeability guarantee, xarray-only boundary, "Adding a KunQuant factor", "Adding a Polars factor"). Three existing bullets brought in line with it: `base/` and `factor/` under Project Structure, and the factor-config entries under Configuration (`BaseFactorConfig`/`FactorConfig`/`PolarsFactorConfig`).

## Decisions Made

1. **Delete the SIMD block width; do not branch on `platform.machine()`.** The obvious alternative — pick 8 on x86_64 and 4 on aarch64 in quantlab — duplicates logic `KunQuant/Driver.py` already owns and would silently drift the day KunQuant adds an architecture or changes a default. Deleting the argument delegates to upstream, and is provably a no-op on x86_64 because 8 is what upstream selects there.

2. **The streaming test compares snapshots, not just exit status.** An incremental path returning a stale or constant buffer raises nothing and produces a correctly-shaped, all-finite result — it would pass every assertion a naive smoke test makes. Capturing `KMID` after the second-to-last and last steps (inside the loop, because each `cal_stream()` replaces the backend's data) and asserting inequality is what makes T-03-05-02 detectable.

3. **`BaseModel` is not instantiated in the interchangeability proof.** It is abstract, and importing it into a factor-layer test would make that test require torch and wandb. Replicating its four factor interactions inline is both cheaper and more legible — a reader sees exactly which calls are being claimed as interchangeable.

4. **The `data_columns`/`factor_names` pairing is documented in the test, not worked around.** `init_stream()` raises `RuntimeError: Cannot find the buffer name` for any declared input the selected outputs do not consume, because KunQuant prunes unreachable inputs. That is a property of the existing code (the production Alpha158 config uses the full factor set, which consumes all six inputs), so the test carries a comment citing the measurement rather than a defensive change to `init_stream()`.

5. **Where Task 3's action text and its own acceptance grep contradicted each other, the grep won.** Task 3 asks the docstring to say the test contains no `isinstance`/`hasattr`, while its acceptance criterion greps `inspect.getsource(...)` — which includes the docstring — for the absence of those exact strings. The docstring now states the property without spelling the built-ins, and explains in-line why it does not. This is the same resolution 03-04 reached for the `KunQuant` string in `base/factor_polars.py`, for the same reason: the machine-checkable half is the half that survives.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Task 3's action text contradicts its own acceptance criterion on the strings `isinstance`/`hasattr`**

- **Found during:** Task 3 (Step A)
- **Issue:** The action says of the no-type-branching property: "Say so in the docstring." The acceptance criterion runs `assert 'isinstance' not in s and 'hasattr' not in s` where `s = inspect.getsource(<the test function>)` — and `inspect.getsource` returns the docstring along with the body. Writing the sentence as specified makes the criterion fail against a fully correct test. Verified empirically: with the plain wording the criterion raised `AssertionError`, with the test itself passing.
- **Fix:** Reworded the docstring to state the property without naming the built-ins ("no runtime type check of any form — no built-in type predicate, no attribute probe — and no per-backend branch anywhere in its body"), with a parenthetical explaining that the names are omitted precisely because the acceptance grep covers the docstring. Two assertions that used `isinstance` incidentally (`isinstance(data, xr.Dataset)`, `isinstance(cfg, dict)`) were replaced with checks that prove the same thing structurally — reading `.dims`/`.data_vars` off the combined result, and `in` on the config mapping.
- **Files modified:** `tests/test_factor_hierarchy.py` (test only — no implementation change)
- **Verification:** the plan's criterion now prints `ok`; a strengthened variant also asserting `'type(' not in s` passes too, so the test contains no runtime type inspection of any form
- **Commit:** `7647568`

**2. [Rule 1 - Bug] Task 3's expected test count for `tests/test_factor_hierarchy.py` is off by two**

- **Found during:** Task 3
- **Issue:** The criterion states `uv run pytest tests/test_factor_hierarchy.py -q` shows **8 passed** ("03-01's scaffold test, 03-02's six, plus this one"). 03-02 in fact added **eight** tests to that file, not six — its own SUMMARY records "Grew from 1 test to 9" (one label-construction regression plus seven hierarchy/hazard/boundary locks). The file therefore held 9 tests before this plan, and holds **10 passed** after it.
- **Fix:** No code change. The substantive half of the criterion — the whole file green with the new test passing — is met. Recorded here so a future reader does not go looking for two deleted tests.
- **Files modified:** none
- **Verification:** `uv run pytest tests/test_factor_hierarchy.py -q` → **10 passed**; full suite → 66 passed
- **Commit:** n/a (documentation-only finding)

**3. [Rule 2 - Missing critical] Two README bullets outside the new section contradicted it until updated**

- **Found during:** Task 3 (Step B)
- **Issue:** Step B says not to restructure sections this plan is not adding to. But Project Structure's `base/` bullet named `FactorKunQuant` as *the* factor base class and its `factor/` bullet listed only three KunQuant factor sets — both written before the `Factor` ABC, `FactorPolars`, `Alpha158Stock` and `Momentum` existed. Leaving them would have put a direct contradiction two screens above the new `## Factor Backends` section, which is worse than the small edit.
- **Fix:** Updated three existing bullets in place (the `base/` and `factor/` entries under Project Structure, and the factor-config entries under Configuration to cover the `BaseFactorConfig`/`FactorConfig`/`PolarsFactorConfig` split 03-02 shipped). No section was moved, renamed, split or reordered.
- **Files modified:** `README.md`
- **Verification:** `grep -c "FactorPolars" README.md` → 4; `grep -c "_get_factor_lazyframe" README.md` → 1
- **Commit:** `7647568`

**4. [Rule 2 - Missing critical] The 03-01 scaffold docstring referenced the identifier Task 1 deleted**

- **Found during:** Task 2
- **Issue:** `test_stream_fixture_provides_eight_symbols`'s docstring justified the 8-symbol panel as matching "KunQuant's stream `blocking_len`/`partition_factor` of 8". After Task 1, `blocking_len` no longer appears in `base/factor.py` at all, so the docstring pointed at a setting that does not exist and implied the panel width was tied to a value the codebase had stopped pinning.
- **Fix:** Reworded to "the SIMD block width KunQuant selects for float on x86_64, and its stream `partition_factor` of 8". The test's assertions are byte-identical to what 03-01 wrote; only the docstring changed.
- **Files modified:** `tests/test_factor_stream.py`
- **Verification:** `uv run pytest tests/test_factor_stream.py -q` → 3 passed
- **Commit:** `598abe9`

---

**Total deviations:** 4 (2 × Rule 1 bugs in plan verification text, 2 × Rule 2 documentation-consistency gaps). **Zero deviations altered implementation behaviour** — Task 1's one-line deletion landed exactly as specified, and every change above is confined to tests, docstrings and README prose.

**Impact on plan:** None on scope or outcome. Every `<success_criteria>` item is met as written, and every acceptance criterion passed after the two corrections above.

## Verification Results

Plan-level `<verification>`, re-run after all three commits:

| Command | Expected | Actual |
|---|---|---|
| `uv run pytest tests/ -q` | zero failures | **66 passed** |
| **SC 1** — `uv run pytest tests/test_factor_kunquant.py -q` | batch Alpha158 → `xr.Dataset`, both markets | **9 passed** |
| **SC 2** — `uv run pytest tests/test_factor_stream.py -q` | streaming `cal_stream` incremental, no error | **3 passed** |
| **SC 3** — `uv run pytest tests/test_factor_polars.py -q` | Polars batch backend, same `xr.Dataset` contract | **5 passed** |
| **SC 4** — `uv run pytest tests/test_factor_hierarchy.py -q` | no plain DataFrame crosses a module boundary | **10 passed** |
| `'blocking_len' not in getsource(_make_stream)` + `'partition_factor=8' in it` | ok | **ok** |
| `'SIMD block width' in getsource(_make_stream)` | ok | **ok** |
| `pytest tests/test_factor_stream.py -q --durations=3` | no test above 20 s | **1.19 s / 0.84 s / 0.07 s** |
| `'isinstance'`/`'hasattr'` not in the proof test's source | ok | **ok** (strengthened: `'type('` also absent) |
| `grep -c "FactorPolars" README.md` | ≥ 1 | **4** |
| `grep -c "_get_factor_lazyframe" README.md` | ≥ 1 | **1** |

**Mutation check** (applied, observed, then reverted with `git checkout -- base/factor.py`):

| Mutation | Result |
|---|---|
| Restore `blocking_len=8` to `_make_stream()`'s `KunCompilerConfig` | **`2 failed, 1 passed`** in `tests/test_factor_stream.py` — both new tests raise `RuntimeError: Blocking length 8 is not supported for float on aarch64`; the 03-01 scaffold test is unaffected. The behavioural gate is real. |

The suite grew from 63 to 66 tests (2 streaming + 1 interchangeability).

## Known Stubs

None introduced by this plan. Every scaffold item 03-01 and 03-02 assigned to 03-05 is now filled:

| Item assigned to 03-05 | Status |
|---|---|
| `tests/test_factor_stream.py` — FACTOR-02 `cal_stream()` replay, BUG-02 aarch64 SIMD block width | **Done** (`598abe9`) |
| `tests/test_factor_hierarchy.py` — KunQuant/Polars interchangeability integration test | **Done** (`7647568`) |

`Factor._get_features`/`_get_labels` still `raise NotImplementedError`; those bodies are pre-existing and untouched by this plan.

## Issues Encountered

None blocking. Two inaccuracies in the plan's own verification text (an acceptance criterion that contradicted its action, and an off-by-two test count) were found and corrected in-flight — see Deviations. The one substantive risk the plan flagged, cross-backend coordinate drift in `xr.combine_by_coords`, did not materialize: the combine worked on the first run, so no `FactorPolars.cal()` normalization was needed.

## Forward Flags

Both flags the plan's `<output>` requires, plus one carried from the threat register.

1. **A real-time data feed wired to `cal_stream()` needs its own threat model.** Everything proven here replays *local synthetic* data: no live feed, no credential, no network call, no external integration is introduced (03-RESEARCH.md Security Domain; the register in this plan is deliberately minimal, not omitted). `cal_stream()` pushes raw `float32` buffers straight into native code sized once at `init_stream()` time — a shape or symbol-count mismatch is a native-side contract, not a Python one, and there is currently no validation between the caller and `pushData`. When a live feed becomes the caller, that input-validation boundary is new attack surface and must get a threat model of its own.

2. **`init_stream()` raises `RuntimeError: Cannot find the buffer name` whenever `config.data_columns` is wider than the selected `factor_names` actually consume.** KunQuant prunes declared inputs no reachable `Output(...)` uses, so `queryBufferHandle` finds no handle for them. Benign for full factor sets — the production Alpha158 config consumes all six inputs — but **a live-feed integration that subsets factors to keep per-bar latency down will hit this immediately**, and the error message does not explain why. Measured example: `factor_names=["KMID","VOLUME0","STD5"]` exposes handles for `close`/`open`/`volume` and the three factors, but not for `high`/`low`/`amount`. Documented in `tests/test_factor_stream.py`, in the README's streaming note, and asserted directly by `test_init_stream_binds_a_buffer_handle_for_every_declared_name`.

3. **Carried from 03-02, still open: `utils/module.py:18` hardcodes `FactorConfig(**config)`.** A checkpoint saved from a `PolarsFactorConfig`-backed factor cannot reload through that path (the persisted dict has no `mode`/`data_columns`). Deliberately out of Phase 3's scope; Phase 4 owns choosing the reload mechanism.

## User Setup Required

None — zero new packages installed, no external service configuration, no environment variable added. 03-RESEARCH.md records the Package Legitimacy Audit as "Not applicable" for this phase (threat T-03-05-SC: accepted, nothing installed).

## Next Phase Readiness

**Phase 3 is complete.** All five plans have SUMMARYs and all four ROADMAP Phase 3 success criteria are satisfied by executed tests rather than by inspection:

1. Alpha158 computable in batch mode, **both markets** → `tests/test_factor_kunquant.py` (9 passed).
2. Streaming `cal_stream()` producing incremental updates without error → `tests/test_factor_stream.py` (3 passed).
3. A new factor via the Polars batch backend on the same `xr.Dataset` contract → `tests/test_factor_polars.py` (5 passed).
4. No plain DataFrame crossing a module boundary → `tests/test_factor_hierarchy.py` (10 passed).

**Ready for Phase 4 (return model).** The factor layer now hands the model layer a stable, backend-agnostic contract: `DLConfig.factors` accepts any `Factor`, its outputs are always `xr.Dataset` over `[timestamp, symbol]`, and mixing backends within one config is a tested path rather than a hope.

**Requirement status:** FACTOR-02 is declared only by this plan and marks Complete immediately. FACTOR-03/FACTOR-04 are shared with sibling plans; with this SUMMARY, this plan is the last declarer for both, so the shared-ID gate should release them.

**Concerns:** none blocking. The three forward flags above are the only open items this plan creates or carries, and all three belong to later phases by decision.

---
*Phase: 03-factor-computation-kunquant-polars*
*Completed: 2026-09-05*

## Self-Check: PASSED

All 4 modified files verified present on disk (`base/factor.py`, `tests/test_factor_stream.py`, `tests/test_factor_hierarchy.py`, `README.md`); all three task commits (`0a61d8e`, `598abe9`, `7647568`) verified present in `git log --all`; full suite re-run green (66 passed).
