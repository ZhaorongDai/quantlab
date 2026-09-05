---
phase: 03-factor-computation-kunquant-polars
plan: 02
subsystem: factor-layer
tags: [refactor, abc, dataclass, factor, kunquant, polars, xarray, d-03, d-07]

# Dependency graph
requires:
  - phase: 03-factor-computation-kunquant-polars
    plan: 01
    provides: "Green batch KunQuant .cal() path plus tests/test_factor_kunquant.py and the tests/test_factor_hierarchy.py scaffold this plan uses as its behaviour-preservation and purity locks"
provides:
  - "base/factor.py:Factor(ABC) — the shared, backend-agnostic factor contract carrying the complete base/model.py call surface"
  - "base/factor.py:FactorKunQuant(Factor) — the KunQuant backend, holding every streaming/compiled-graph member and the three mode-branching overrides"
  - "base/config.py three-way dataclass split: BaseFactorConfig / FactorConfig(BaseFactorConfig) / PolarsFactorConfig(BaseFactorConfig), all kw_only"
  - "DLConfig/MLConfig factors+labels typed against the shared Factor type"
  - "Factor._maybe_resolve_factor_names() — the overridable factor-name-resolution seam 03-04's FactorPolars depends on (D-05)"
  - "Eight automated assertions locking D-03, D-07, Hazard 1, Hazard 2 and FACTOR-04's xarray-only boundary"
  - "The only test coverage label/spot.py has anywhere in the repository"
affects: [03-04, 03-05, 04-return-model, polars-factor-backend, model-checkpoint-reload]

actuals:
  tokens: 8157
  tasks: 3
  commits: 3

tech-stack:
  added: []
  patterns:
    - "Three-way member split for backend-divergent behaviour: batch-shaped body on the shared base, backend-branching two-line delegate override on the subclass (`super().<name>` works for property getters as well as ordinary methods)"
    - "Overridable resolution-timing hook (`_maybe_resolve_factor_names`) whose default is the pre-refactor inline code, so introducing the seam is a zero-behaviour-change edit"
    - "`@dataclass(kw_only=True)` inheritance for config splits — legal only because 100% of construction sites are keyword-only (audited); it lifts the defaults-before-mandatory-fields restriction"
    - "Source-introspection tests (`inspect.getsource` over `vars(cls)`) as the runtime guard for invariants that nothing currently fails on, paired with a mutation check proving the guard actually fires"

key-files:
  created: []
  modified:
    - base/config.py
    - base/factor.py
    - tests/test_factor_hierarchy.py

key-decisions:
  - "Factor's mode-free members are the batch-shaped bodies and FactorKunQuant restores today's exact behaviour via three two-line overrides — a naive hoist would have made PolarsFactorConfig raise AttributeError at runtime only"
  - "`_maybe_resolve_factor_names()` is introduced now with the eager default even though nothing overrides it yet, because it is the seam 03-04 needs; the default keeps every existing subclass byte-for-byte behaviour-identical"
  - "The Factor import in base/config.py stays inside `if TYPE_CHECKING:` — promoting it to a runtime import creates the one import cycle in the layer graph"
  - "No `Factor.config_cls` ClassVar was added: utils/module.py's checkpoint-reload path is Phase 4's problem and an unused attribute would violate QUAL-02"
  - "Task 1's field-annotation acceptance command and Task 3's typing test use `str(field.type)`, not `field.type` — base/config.py has no `from __future__ import annotations`, so annotations are evaluated GenericAlias objects, not source strings"

patterns-established:
  - "Mutation-checked introspection guards: every source-introspection assertion in this plan was proven to fail against a deliberately broken variant, then reverted — an introspection test that cannot fail is worse than no test"
  - "Refactor commits carry the behaviour-preservation proof in the commit body (03-01's .cal() tests pass unchanged; git diff base/model.py empty)"

requirements-completed: [FACTOR-01, FACTOR-04]

coverage:
  - id: D1
    description: "A shared abstract Factor base exists in base/factor.py and FactorKunQuant is a subclass of it, with cal and _get_factor_names abstract on the shared base (D-03)"
    requirement: FACTOR-04
    verification:
      - kind: unit
        ref: "tests/test_factor_hierarchy.py#test_factor_kunquant_subclasses_shared_factor_base"
        status: pass
    human_judgment: false
  - id: D2
    description: "Alpha101SpotKline and Alpha158SpotKline still return identical .cal() results through the new hierarchy — 03-01's end-to-end tests re-run unmodified"
    requirement: FACTOR-01
    verification:
      - kind: integration
        ref: "tests/test_factor_kunquant.py (2 passed, file unmodified by this plan)"
        status: pass
    human_judgment: false
  - id: D3
    description: "SpotReturn/SpotBinaryReturn still construct with config.factor_names resolved eagerly by the hoisted config setter and the new _maybe_resolve_factor_names() hook"
    requirement: FACTOR-01
    verification:
      - kind: integration
        ref: "tests/test_factor_hierarchy.py#test_label_classes_construct_through_the_refactored_hierarchy"
        status: pass
    human_judgment: false
  - id: D4
    description: "Factor carries every one of the eight members base/model.py invokes on a factor object, so a non-KunQuant backend is droppable into DLConfig.factors with zero base/model.py edits (D-03 interchangeability)"
    requirement: FACTOR-04
    verification:
      - kind: unit
        ref: "tests/test_factor_hierarchy.py#test_factor_base_carries_the_full_base_model_call_surface"
        status: pass
      - kind: other
        ref: "git diff 66ea1ba..HEAD -- base/model.py produces 0 bytes"
        status: pass
    human_judgment: false
  - id: D5
    description: "Every KunQuant-streaming-specific member stays on FactorKunQuant and is absent from Factor (D-07)"
    requirement: FACTOR-04
    verification:
      - kind: unit
        ref: "tests/test_factor_hierarchy.py#test_streaming_members_stay_on_the_kunquant_subclass"
        status: pass
    human_judgment: false
  - id: D6
    description: "No method or property defined on Factor reads config.mode, so a PolarsFactorConfig without a mode field cannot raise AttributeError at runtime (Hazard 2)"
    requirement: FACTOR-04
    verification:
      - kind: unit
        ref: "tests/test_factor_hierarchy.py#test_shared_factor_base_never_reads_config_mode (mutation-checked: adding a mode branch to Factor._auto_filter fails it)"
        status: pass
    human_judgment: false
  - id: D7
    description: "DLConfig.factors/.labels and MLConfig.factors/.labels are typed against the shared Factor type, not FactorKunQuant"
    requirement: FACTOR-04
    verification:
      - kind: unit
        ref: "tests/test_factor_hierarchy.py#test_model_configs_are_typed_against_the_shared_factor_base"
        status: pass
    human_judgment: false
  - id: D8
    description: "Factor.__init__ assigns self.config before self.data_backend, and no setter-reachable method reads the storage backend (Hazard 1)"
    requirement: FACTOR-04
    verification:
      - kind: unit
        ref: "tests/test_factor_hierarchy.py#test_factor_init_assigns_config_before_the_storage_backend (mutation-checked: swapping the two __init__ lines fails it)"
        status: pass
    human_judgment: false
  - id: D9
    description: "No public method of Factor or FactorKunQuant accepts or returns a bare pandas/polars DataFrame — xr.Dataset is the only exchange type at the public boundary (FACTOR-04)"
    requirement: FACTOR-04
    verification:
      - kind: unit
        ref: "tests/test_factor_hierarchy.py#test_public_factor_api_exchanges_only_xarray_datasets"
        status: pass
    human_judgment: false
  - id: D10
    description: "FactorConfig splits into BaseFactorConfig / FactorConfig / PolarsFactorConfig with the single to_dict() on the base, and every existing keyword-only config factory still constructs under kw_only=True"
    requirement: FACTOR-04
    verification:
      - kind: other
        ref: "uv run python -c 'from config import alpha101_config, alpha158_config, spot_label_config' -> ok; dataclasses.fields() field-set assertions -> ok; uv run pytest tests/ -q -> 52 passed"
        status: pass
    human_judgment: false

duration: 6 min
completed: 2026-09-05
status: complete
---

# Phase 3 Plan 02: Extract the Shared `Factor` Base Summary

**A pure, behaviour-preserving hierarchy refactor: `base/factor.py` now holds `Factor(ABC)` + `FactorKunQuant(Factor)`, `base/config.py` holds a three-way `BaseFactorConfig`/`FactorConfig`/`PolarsFactorConfig` split, and eight mutation-checked assertions lock the result — with `base/model.py` untouched, which is the actual proof of D-03's "无缝替换" interchangeability rather than the widened type hint.**

## Performance

- **Duration:** 6 min
- **Started:** 2026-09-05T18:24:02Z
- **Completed:** 2026-09-05T18:31:00Z
- **Tasks:** 3
- **Files modified:** 3 (all modified, none created)

## Accomplishments

- **Delivered D-03, the phase's core architectural requirement.** `Factor(ABC)` now carries the complete set of eight members `base/model.py:115-201` invokes on a factor or label object (`config` read+write, `_reset_dataset_config`, `cal`, `read`, `get_features`, `get_labels`, `_get_factor_names`, `get_config`). `cal()` and `_get_factor_names()` are abstract **on the shared base**, which is the mechanism that makes `factor.cal()` polymorphic across backends instead of a KunQuant call in disguise. `base/model.py` required zero edits — verified by `git diff 66ea1ba..HEAD -- base/model.py` producing 0 bytes.

- **Split the config dataclasses without breaking a single construction site.** `BaseFactorConfig` holds the nine backend-agnostic fields and the one `to_dict()`; `FactorConfig` adds the KunQuant-only `mode`/`data_columns`/`njobs`; `PolarsFactorConfig` adds nothing and deliberately has **no `mode`** (D-07: the Polars backend is batch-only). All three are `@dataclass(kw_only=True)`, which is safe precisely because 03-PATTERNS.md §3's audit found every `FactorConfig(...)` construction in the repository — `config/__init__.py:127/:154/:184`, `test.py:27`, and `utils/module.py:18`'s dict-splat — is 100% keyword-argument.

- **Neutralised all three refactor hazards, and proved each neutralisation with a test that was shown to fail.** `mode` is read by three members, not one; all three are now split (batch-shaped body on `Factor`, two-line `super()`-delegating override on `FactorKunQuant`). `Factor.__init__` preserves the load-bearing `self.config` → `self.data_backend` order. The config setter's eager factor-name resolution became the `_maybe_resolve_factor_names()` hook with its pre-refactor body as the default.

- **Gave `label/spot.py` its first test coverage in the repository's history.** `SpotReturn`/`SpotBinaryReturn` are direct `FactorKunQuant` subclasses, so constructing one runs precisely the code this plan rewrote — the hoisted `config` setter and the brand-new hook. The new regression builds a real `SpotReturn` against the `spot_kline_zarr` fixture with `factor_names=None` and asserts the setter still resolved `("ret_1",)` eagerly.

- **Turned FACTOR-04 from a claim into an enforced assertion.** No public method of either class exchanges a `DataFrame`, and `get_features`/`get_labels` are asserted to return `xr.Dataset` by identity (not by string match). Private helpers are excluded on purpose — `Factor._get_lazyframe()` legitimately returns a `pl.LazyFrame`; FACTOR-04 constrains the module boundary, not internals.

## Task Commits

Each task was committed atomically:

1. **Task 1: Split FactorConfig and widen DLConfig/MLConfig** — `05abddd` (refactor)
2. **Task 2: Extract Factor(ABC) from FactorKunQuant, behaviour-preserving** — `41b7c33` (refactor)
3. **Task 3: Lock the hierarchy shape, mode isolation and xarray-only boundary** — `9434230` (test)

**Plan metadata:** see the `docs(03-02)` commit that carries this SUMMARY.

## Files Created/Modified

- `base/config.py` — **modified.** `FactorConfig` (18 lines) became three `kw_only` dataclasses (~50 lines). The `TYPE_CHECKING` import changed from `FactorKunQuant` to `Factor`, and the four `list["FactorKunQuant"]` annotations on `DLConfig`/`MLConfig` became `list["Factor"]`. `DatasetConfig`, `AcquisitionConfig` and `UniverseConfig` are untouched.
- `base/factor.py` — **modified (restructured).** One class became two. Every hoisted body is byte-identical to what was on disk, Chinese docstrings and inline comments included. `Factor` gained one new method (`_maybe_resolve_factor_names`) and one new comment (the `__init__` ordering note); `FactorKunQuant` gained three `super()`-delegating overrides. Nothing else changed.
- `tests/test_factor_hierarchy.py` — **modified.** Grew from 1 test to 9: the 03-01 purity scaffold (untouched), Task 2's label-class construction regression, and Task 3's seven hierarchy/hazard/boundary locks, plus four module-level constants and one `_own_function_sources()` helper.

## Decisions Made

1. **`Factor`'s three `mode`-divergent members carry the batch-shaped body, and `FactorKunQuant` restores today's exact semantics via delegating overrides.** `_auto_filter`, `num_symbols` and `symbols` all branch on `config.mode`, which now lives only on `FactorConfig`. A naive hoist would have been invisible today and fatal for 03-04 — `AttributeError: 'PolarsFactorConfig' object has no attribute 'mode'`, at runtime, only on the code path someone happens to call. `super().num_symbols` works for property getters, so each override is two lines.

2. **`_maybe_resolve_factor_names()` ships now with the eager default, even though nothing overrides it yet.** Introducing the seam and preserving behaviour are the same edit: the default body is verbatim the three lines it replaced, so every existing subclass — including the two label classes — behaves identically. 03-04's `FactorPolars` overrides it to a no-op because Polars factor names need a disk read (D-05). Building the seam later would have meant re-touching the setter in a plan that has other things to prove.

3. **The `Factor` import in `base/config.py` stays inside `if TYPE_CHECKING:`.** `base/factor.py` imports from `base/config.py` at runtime; promoting the reverse import would close the cycle. The `from config import alpha101_config, ...` acceptance check exists specifically to fail loudly if a future edit does so.

4. **No `Factor.config_cls` ClassVar was added.** `utils/module.py:18` hardcodes `FactorConfig(**config)` and cannot rebuild a `PolarsFactorConfig`-backed factor, but that path is only reachable from Phase 4's checkpoint reload and no Phase-3 criterion touches it. Adding an unused attribute "to unblock Phase 4" would violate QUAL-02 and pre-decide a mechanism Phase 4 should choose for itself. Recorded as a forward flag below instead.

5. **Every source-introspection guard was mutation-checked before being trusted.** Both `test_shared_factor_base_never_reads_config_mode` and `test_factor_init_assigns_config_before_the_storage_backend` assert facts that nothing at runtime currently depends on — so a guard that silently could not fail would be worse than no guard. Each was run against a deliberately broken `base/factor.py`, observed to fail, and reverted via `git checkout -- base/factor.py`.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] `dataclasses.fields()` exposes evaluated annotations here, not source strings**

- **Found during:** Task 1 (acceptance criterion 2), carried into Task 3
- **Issue:** The plan states "(Verified: `dataclasses.fields()` exposes these as the literal source strings.)" and both Task 1's second acceptance command and Task 3's `test_model_configs_are_typed_against_the_shared_factor_base` rely on it. `base/config.py` has no `from __future__ import annotations`, so class-level annotations are evaluated at class-creation time and `field.type` is a `types.GenericAlias` (`list['Factor']`), not a `str`. `'Factor' in field.type` does **not** raise — `GenericAlias` forwards `__contains__` to `list` — it silently returns `False`. The criterion as literally written would have failed against fully correct code, and its `'FactorKunQuant' not in t` half would have passed vacuously against incorrect code.
- **Fix:** Wrapped the annotation in `str(...)` in both the acceptance command and the test: `str(field.type)` yields `"list['Factor']"`, which contains `Factor` and not `FactorKunQuant` exactly as the plan intends. Adding `from __future__ import annotations` to `base/config.py` was rejected as a larger, out-of-scope behaviour change.
- **Files modified:** `tests/test_factor_hierarchy.py` (test only — no implementation change)
- **Verification:** `test_model_configs_are_typed_against_the_shared_factor_base` passes and would fail on either half of the invariant
- **Commit:** `9434230`

**2. [Rule 2 - Missing critical] Hazard-1 guard extended to cover the whole setter-reachable set**

- **Found during:** Task 3
- **Issue:** The plan enumerates the setter-reachable methods as `Factor.config.fset`, `_maybe_resolve_factor_names`, `_reset_dataset_config` and any `FactorKunQuant` override of those three. But the setter's second line is `self._config.name = self.import_path` — `import_path` is genuinely setter-reachable and was outside the enumerated set, so the test's own docstring claim ("no setter-reachable method reads the storage backend") would have been only partially asserted. Similarly, the `mode` guard's spec says "every function defined directly in `vars(Factor)`", which by `inspect.isfunction` skips property getters/setters — and `num_symbols`/`symbols`, the two members most likely to regain a `mode` read, are properties.
- **Fix:** Added `vars(Factor)["import_path"].fget` to the setter-reachable set, and made `_own_function_sources()` walk property `fget`/`fset`/`fdel` in addition to plain functions. Both are strict strengthenings; neither weakens anything the plan asked for.
- **Files modified:** `tests/test_factor_hierarchy.py`
- **Verification:** both tests pass; both were mutation-checked against a broken `base/factor.py`
- **Commit:** `9434230`

**Total deviations:** 2 auto-fixed (1 × Rule 1 bug in a plan verification command, 1 × Rule 2 assertion-completeness gap). Both are confined to the test file; **zero deviations touched implementation code**.

**Impact on plan:** None on scope or outcome. Every `<success_criteria>` item is met as written, and every acceptance criterion passed after the two corrections above.

## Verification Results

Plan-level `<verification>`, re-run after all three commits:

| Command | Expected | Actual |
|---|---|---|
| `uv run pytest tests/ -q` | zero failures | **52 passed** |
| `git diff 66ea1ba..HEAD -- base/model.py` | empty | **0 bytes** |
| `uv run pytest tests/test_factor_hierarchy.py -q` | 9 passed | **9 passed** |
| `uv run pytest tests/test_factor_kunquant.py -x -q` | 2 passed (behaviour-preservation gate) | **2 passed** |
| `uv run pytest tests/test_extensibility_contract.py -q` | all passed (`base/factor.py` stayed pure) | **2 passed** |
| `uv run pytest tests/test_factor_hierarchy.py -k label -q` | 1 passed | **1 passed** |
| `inspect.isabstract(Factor)` + `issubclass(FactorKunQuant, Factor)` | ok | **ok** (`__abstractmethods__` = `_get_factor_names`, `cal`) |
| Streaming members in `FactorKunQuant.__dict__`, absent from `Factor.__dict__` | ok | **ok** |
| `.mode` absent from every `Factor`-own function source | ok | **ok** |
| Label classes subclass the hierarchy + expose the full call surface | ok | **ok** |
| `issubclass` / field-set checks on the three config dataclasses | ok | **ok** |
| `str(field.type)` on all four `DLConfig`/`MLConfig` fields | contains `Factor`, not `FactorKunQuant` | **`list['Factor']`** |
| `from config import alpha101_config, alpha158_config, spot_label_config` | ok (no import cycle, `kw_only` factories still construct) | **ok** |

**Mutation checks** (Task 3 acceptance criteria — each applied, observed, then reverted with `git checkout -- base/factor.py`):

| Mutation | Test | Result |
|---|---|---|
| Add `if self.config.mode != "batch": return` to `Factor._auto_filter` | `test_shared_factor_base_never_reads_config_mode` | **FAILED as required** |
| Swap the two assignments in `Factor.__init__` | `test_factor_init_assigns_config_before_the_storage_backend` | **FAILED as required** |

The suite grew from 44 to 52 tests (8 added: 1 label regression + 7 hierarchy locks).

## Known Stubs

None introduced by this plan.

`PolarsFactorConfig` is field-free by design, not by omission — D-04..D-07's scope needs no field beyond `BaseFactorConfig`, and adding a speculative one would violate QUAL-02. `Factor._get_features`/`_get_labels` still `raise NotImplementedError`; those bodies are pre-existing and were moved verbatim.

Two scaffold items in `tests/test_factor_hierarchy.py` remain, both already recorded in 03-01's SUMMARY and both assigned:

| Item | Status | Resolved by |
|---|---|---|
| KunQuant/Polars interchangeability integration test | Not written — `FactorPolars` does not exist yet | **03-05** |
| `base/factor_polars.py` added to `tests/test_extensibility_contract.py:CORE_LAYER_FILES` | Deliberately not added here — the file does not exist and the test would raise `FileNotFoundError` | **03-04** |

## Issues Encountered

None blocking. One plan-level inaccuracy (`dataclasses.fields()` annotation types) was found and corrected in-flight — see Deviations.

## Forward Flag for Phase 4 — Checkpoint Reload

`utils/module.py:18` hardcodes `FactorConfig(**config)`:

```python
def load_factor_from_config(config: dict):
    config["dataset"] = load_dataset_from_config(config["dataset"])
    return get_cls_from_path(config["name"])(FactorConfig(**config))
```

After 03-04 ships `FactorPolars`, a checkpoint saved from a `PolarsFactorConfig`-backed factor will not reload through this path: the persisted dict has no `mode`/`data_columns`/`njobs`, so `FactorConfig(**config)` raises `TypeError` on the missing mandatory `mode` and `data_columns`. This is 03-RESEARCH.md Open Question 1 and is **deliberately out of Phase 3's scope** — the path is only reachable from Phase 4's model-checkpoint reload flow, and no Phase-3 success criterion touches it.

**Phase 4 must choose a reload mechanism.** Candidate approaches, none pre-decided here: a `Factor.config_cls` ClassVar consulted by `load_factor_from_config`; persisting the config's own qualified class name alongside `name`; or a registry keyed on the factor's `import_path`. A ClassVar was deliberately NOT added in this plan — an unused attribute violates QUAL-02, and the choice belongs to the phase that has to live with it.

## User Setup Required

None — zero new packages installed, no external service configuration, no environment variable added. 03-RESEARCH.md records the Package Legitimacy Audit as "Not applicable" for this phase.

## Next Phase Readiness

**Ready for 03-03 and 03-04.** Wave 2 is complete and the seam 03-04 was blocked on now exists.

- **03-04** (`FactorPolars`, `factor/momentum.py`) can now subclass `Factor` directly, construct against `PolarsFactorConfig`, and override `_maybe_resolve_factor_names()` to a no-op for D-05's dynamic schema resolution. Every hazard that would have bitten it — a `mode` read on the shared base, a `__init__` ordering inversion, a streaming obligation inherited from the base — is now a failing test rather than a runtime surprise. Remember to add `base/factor_polars.py` to `tests/test_extensibility_contract.py:CORE_LAYER_FILES` in that plan.
- **03-03** (`Alpha158Stock`, D-02 `amount` proxy, D-09 normalization) is unaffected by this refactor and still has `stock_zarr` waiting. `Alpha101Stock` was deliberately not exercised here: it crashes at graph-construction time on the pre-existing missing-`amount` defect 03-03 fixes, so its behaviour preservation is asserted there. This plan asserted only that it still subclasses `FactorKunQuant` (via the class-level hierarchy tests) and that the full `base/model.py` call surface resolves.
- **03-05** (streaming) inherits an unchanged `FactorKunQuant` streaming path — `init_stream`/`cal_stream`/`_make_stream` were moved as a block, not rewritten — and can add the interchangeability integration test to the now-substantial `tests/test_factor_hierarchy.py`.

**Requirement status:** FACTOR-04 is satisfied and asserted. FACTOR-01 is declared by this plan and by siblings 03-03/03-05, so the shared-ID gate may hold it short of `Complete` until the last declaring plan produces its SUMMARY — expected behaviour, not a gap.

**Concerns:** none blocking. The Phase-4 checkpoint-reload flag above is the only forward-facing item this plan creates.

---
*Phase: 03-factor-computation-kunquant-polars*
*Completed: 2026-09-05*

## Self-Check: PASSED

All 3 modified files verified present on disk; all three task commits (`05abddd`, `41b7c33`, `9434230`) verified present in `git log --all`; full suite re-run green (52 passed).
