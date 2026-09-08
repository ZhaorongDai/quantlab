---
gsd_state_version: "1.0"
milestone: v1.0
current_phase: 3
current_phase_name: Factor Computation (KunQuant + Polars)
status: planning
stopped_at: Completed quick task 260907-rjq (--data-dir CLI parameter)
last_updated: "2026-09-08T00:26:24.132Z"
last_activity: 2026-09-07
last_activity_desc: "Completed quick task 260907-rjq: --data-dir CLI parameter for the acquisition scripts"
state_head: e34a84b974eb729dd00cfe383ac56e95c2c6e90e
progress:
  total_phases: 11
  completed_phases: 2
  total_plans: 30
  completed_plans: 30
milestone_name: milestone
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-09-04)

**Core value:** 一条打通的、config 驱动可复现的量化流水线（数据→因子→收益模型→组合优化→目标持仓→回测→结果），模块间用清晰的输入输出契约组合，任何一环都能独立替换/扩展而不需要推倒重来。
**Current focus:** Phase 03.1 — Index Historical Constituents Data Layer (INSERTED)

## Current Position

Phase: 3 — Factor Computation (KunQuant + Polars)
Plan: Not started
Status: Ready to plan
Last activity: 2026-09-07 - Completed quick task 260907-rjq: Add a --data-dir CLI parameter to the data acquisition scripts

Progress: [██████████] 100%

## Performance Metrics

**Velocity:**

- Total plans completed: 12
- Average duration: -
- Total execution time: 0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 03.2 | 7 | - | - |
| 03.1 | 5 | - | - |

**Recent Trend:**

- Last 5 plans: -
- Trend: -

*Updated after each plan completion*
**Per-Plan Metrics:**

| Plan | Duration | Tasks | Files |
|------|----------|-------|-------|
| Phase 03 P01 | 31 min | 2 tasks | 7 files |
| Phase 03 P02 | 6 min | 3 tasks | 3 files |
| Phase 03 P03 | 7 min | 3 tasks | 5 files |
| Phase 03 P04 | 8 min | 3 tasks | 4 files |
| Phase 03 P05 | 15 min | 3 tasks | 4 files |
| Phase 03.1 P01 | 6 min | 3 tasks | 8 files |
| Phase 03.1 P02 | 7 min | 2 tasks | 4 files |
| Phase 03.1 P03 | 15 min | 3 tasks | 7 files |
| Phase 03.1 P04 | 18 min | 3 tasks | 7 files |
| Phase quick-260906-0iy P01 | 15 min | 3 tasks | 10 files |
| Phase quick-260906-13w P01 | 18 min | 3 tasks | 10 files |
| Phase quick-260906-26o P01 | 42 min | 2 tasks | 5 files |
| Phase quick-260906-eme P01 | 9 min | 2 tasks | 3 files |
| Phase 03.2 P01 | 12 min | 2 tasks | 6 files |
| Phase 03.2 P02 | 47 min | 4 tasks | 19 files |
| Phase 03.2 P03 | 37 min | 3 tasks | 9 files |
| Phase 03.2 P04 | 16 min | 2 tasks | 2 files |
| Phase 03.2 P05 | 25 min | 2 tasks | 3 files |
| Phase 03.2 P06 | 17 min | 3 tasks | 5 files |
| Phase 03.2 P07 | 27 min | 3 tasks | 8 files |
| Phase quick-260906-usg P01 | 41 min | 3 tasks | 8 files |
| Phase quick-260906-w3t P01 | 18 min | 3 tasks | 7 files |
| Phase quick-260906-x2s P01 | 14 min | 3 tasks | 8 files |
| Phase 03.1 P05 | 8 min | 3 tasks | 5 files |
| Phase quick-260907-rjq P01 | 34 min | 3 tasks | 12 files |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- Roadmap: Phases follow the pipeline's natural sequential dependency (data → factor → model → portfolio → backtest) per the project's own Key Decision to deliver the full chain with baseline implementations at every stage before deepening any one stage.
- Roadmap: Phase 1 (cleanup/security) is a mandatory precondition — later phases assume a working `uv sync`, no leaked credentials, and accurate docs.
- Roadmap: QUAL-01/QUAL-02 (tests, code quality) placed in a dedicated final Phase 7 since they apply across all core modules built in Phases 2-6.
- [Phase 03]: my_ops composite ops accept and ignore `options` rather than reading it — Neither WindowedZScore nor WindowedRobustStandardization needs decomposition options; accepting-and-ignoring is a strict superset of prior behaviour and matches the KunQuant-shipped ops in KunQuant/ops/CompOp.py that do the same. Reading options would have been unverifiable new behaviour on a bug-fix commit.
- [Phase 03]: Phase-3 test fixtures write Zarr stores directly, bypassing raw CSV/parquet ingestion — Factor windows need ~60 timestamps x 8 symbols; driving that through SpotKlineDataset._raw_data_to_xr() would make every factor test also a CSV-parsing test. Writing the synthetic panel straight to Zarr keeps factor tests testing factors, at ~0.5s each.
- [Phase 03]: Each Wave-0 scaffold test file ships one real infrastructure self-test, never a placeholder — A file with zero tests makes its per-file pytest command exit 5 ("no tests ran"), which reads as green. Each scaffold instead asserts the exact fixture contract the plan that fills it depends on, so it has value alone and fails loudly if that contract drifts.
- [Phase 03]: Factor mode-divergent members (_auto_filter/num_symbols/symbols) carry the batch-shaped body on the shared base, with FactorKunQuant restoring exact behaviour via super()-delegating overrides — A naive hoist would be invisible today and fatal for 03-04: config.mode lives only on FactorConfig, so PolarsFactorConfig would raise AttributeError at runtime, only on the path someone happens to call. Test-time source-introspection enforcement replaces that runtime surprise.
- [Phase 03]: Factor._maybe_resolve_factor_names() ships now with the pre-refactor eager body as its default, even though nothing overrides it yet — Introducing the seam and preserving behaviour are the same edit; 03-04 FactorPolars overrides it to a no-op for D-05 dynamic schema resolution. Adding it later would mean re-touching the config setter in a plan with other things to prove.
- [Phase 03]: No Factor.config_cls ClassVar was added - utils/module.py hardcoded FactorConfig(**config) checkpoint reload is deferred to Phase 4 — The path is only reachable from Phase 4 model-checkpoint reload and no Phase-3 success criterion touches it; an unused attribute violates QUAL-02 and would pre-decide a mechanism Phase 4 should choose for itself.
- [Phase 03]: amount = volume * close is synthesized once centrally in StockDataset._to_kunquant(), double-guarded, not per factor class — The boundary method is the one place every KunQuant consumer of US-equity data passes through; per-class fixes would need one edit per factor class forever and each is a place to get adjusted-vs-raw wrong. The double guard (requested AND absent) keeps it inert and stops it overwriting a real vendor column if one ever appears (T-03-03-01).
- [Phase 03]: Alpha158Stock replicates Alpha158SpotKline verbatim (double-AllData build, six unused Input nodes); the normalization wrapper is the single deliberate divergence — D-01 fixes the invocation shape across markets, so tidying one class alone would silently diverge them and tidying both would edit working code for cosmetics. A vars() method-set equality assertion is the mechanical guard.
- [Phase 03]: The D-09 four-class normalization matrix is locked by one equality assertion against a literal, with docstrings on all four classes — Raw vs normalized factor values are indistinguishable to a downstream consumer at runtime (T-03-03-02). A per-market strategy choice that looks like an inconsistency must be impossible to 'align' silently; the assertion message routes a would-be changer to D-09 in 03-CONTEXT.md before the test literal.
- [Phase 03]: FactorPolars resolves factor names inside cal() from collect_schema().names(), never at construction time — Keeping the inherited eager _maybe_resolve_factor_names() would make merely constructing a Polars factor trigger a disk read, contradicting D-04's computation-starts-at-cal() contract. The documented cost is that _get_factor_names()/num_factors raise until cal()/read() has run - a gap base/model.py never hits.
- [Phase 03]: Polars factors are not portable across markets: Dataset.get_lazyframe() applies no per-market column normalization (03-RESEARCH.md Open Question 2), so factor/momentum.py is written against crypto-spot Title-Case Close — D-08 requires only one example factor and adding a get_lazyframe()-side rename layer was explicitly out of Phase 3 scope. Recorded in both docstrings a new factor author reads; revisit when a second Polars factor must span both markets.
- [Phase 03.1]: BaseFactorConfig.dataset is annotated "MarketDataset", not "BaseDataset" — FactorKunQuant calls dataset.to_kunquant(), a MarketDataset-only member, so widening the annotation would be a lie that type-checks. A future constituent-masking consumer takes a BaseDataset of its own; do not pre-widen.
- [Phase 03.1]: The pre-split name base.data.Dataset was retired, not kept as a permanent alias — Two live names for one class is exactly the ambiguity a later reader 'fixes' wrongly, and an unused alias violates QUAL-02. A transitional alias existed for exactly one commit so the split could land with a green suite; test 7 makes reintroducing it a test failure.
- [Phase 03.1]: _reset_symbols() became a named overridable seam on BaseDataset rather than an isinstance branch or a config boolean flag — It is called from the shared config setter, so pushing it down to MarketDataset would force the setter to split too. Overriding it to a no-op is proven to suppress both the store read and the from_raw_data() network fallback, which is what lets 03.1-02 construct a constituent dataset offline.
- [Phase 03.1]: BaseDataset.__init__ keeps its data_backend-first order, the deliberate inverse of base/factor.py:Factor.__init__ — The dataset config setter DOES reach the storage backend via _reset_symbols() -> read(), whereas no method reachable from the factor setter may. Both invariants are now locked by their own inverse guard tests so neither hierarchy can be silently harmonized onto the other.
- [Phase 03.1]: Nasdaq-100 anchor comes from stockanalysis.com with slickcharts.com documented in the class docstring as the fallback, deliberately not implemented as a second code path — Wikipedia's Nasdaq-100 page renders components through a navbox template with no parseable constituents table, so a commercial scraped source was unavoidable. Implementing both sources doubles the untested surface for a failure mode that has not happened; naming the alternative in the docstring is what a maintainer actually needs the day the primary dies.
- [Phase 03.1]: The Nasdaq-100 anchor is asserted at 102 rows, not 100, and fetch_anchor() carries its own MIN_ANCHOR_ROWS=50 structural-drift guard — The index carries multiple share classes for some issuers (GOOGL/GOOG, FOX/FOXA), so an ==100 assertion fails against correct data. Unlike the S&P anchor's hosted CSV, this scraped page has no cache-fallback path, so a truncated parse would close every unmentioned membership and silently reintroduce the survivorship bias the layer exists to remove (T-03.1-02-02).
- [Phase 03.1]: Only two message strings were parameterised via INDEX_LABEL/CATEGORY when hoisting fetch_changes(); both ValueError guard messages stayed word-for-word — The fallback logger.error and _load_cache's RuntimeError are the only index-specific text. Keeping the missing-columns and row-count-monotonicity messages verbatim means the S&P path renders byte-identical output post-extraction, so the refactor is provably behaviour-preserving rather than merely test-passing.
- [Phase 03.1]: The Nasdaq-100 panel gets its own nasdaq100_constituent.zarr store rather than sharing the S&P panel's — The two coverage starts are ~31 years apart (1976-07-01 vs 2007-02-01), and in a boolean panel a fabricated pre-coverage region is indistinguishable at read time from a genuine 'nobody was a member'. A consumer wanting both opens both and joins on the intersection of their timestamp axes -- a deliberate, visible step rather than an implicit and wrong union.
- [Phase 03.1]: UniverseCatalog's category list and coverage guard are driven by a MEMBERSHIP_FETCHERS registry, with NasdaqUniverseFetcher deliberately outside it — The pre-existing guard hardcoded sp500_constituent, so the newly-added nasdaq100_constituent category would have answered pre-2007 queries with a silently incomplete roster -- the exact failure DATA-05 exists to prevent. Deriving each boundary from the fetcher's own PIT_COVERAGE_START makes a boundary-less category impossible to create by omission. nasdaq_all stays out because it is a full-exchange roster with no membership-interval semantics and no coverage start (D-02); test_nasdaq_all_has_no_coverage_boundary makes that absence a tested property.
- [Phase 03.1]: DATA-06 was proved with a baseline SHA pinned before any edit, not a working-tree check — GSD commits per task, so 'git status --porcelain -- base/' alone prints nothing in exactly the world where base/ was edited and committed -- a proof that cannot fail. Pinning BASE_SHA into the git dir before Task 1 and gating on 'git diff BASE_SHA..HEAD -- base/' PLUS the working tree keeps the claim red after each task commit. Final result for 8be0adb: both halves empty.
- [Phase 03.1]: Chunked ingest: the symbol axis is pinned once whole-range before any window and every window is reindexed onto it (D-02), following base/constituent.py:_densify
- [Phase 03.1]: Per-chunk sizing uses the PINNED whole-range symbol count, never the chunk's own roster -- a chunk-scoped estimate understates the allocation and reopens the OOM
- [Phase 03.1]: assert_dense_panel_fits/MAX_DENSE_PANEL_BYTES kept unchanged; assert_chunked_panel_fits is a sibling that lifts the whole-range refusal and reports the total as a non-raising advisory
- [Phase 03.1]: XrBackend.append raises on a changed non-append coordinate or dtype -- both were measured to corrupt silently in raw zarr
- [Phase 03.1]: UniverseMask lives in dataset/ not base/: hook-free concrete composition, per the dataset/cleaning.py precedent
- [quick-260906-26o]: An un-stamped legacy watermark is SKIPPED but reported loudly with the exact stamping command -- never assigned an invented covered start (forbidden by D-04) and never re-fetched by default (would violate D-01). What made the original defect dangerous was the silence, not the skip.
- [quick-260906-26o]: QUOTA_STATUS_CODES is frozenset({429}) only; 403 is deliberately excluded because Tiingo also returns it for a plan-restricted single ticker, and treating that as global would let one restricted ticker abort a 15k-symbol run.
- [quick-260906-26o]: The quota abort is a threading.Event checked as _attempt FIRST statement -- joblib cannot cancel already-queued work, so the input-generator check is an optimisation, not the guarantee. The result generator is drained, never abandoned.
- [quick-260906-26o]: refresh() keeps the end-date-only coverage rule; only download() honours a widened --start-date, because refresh requests [watermark, end_date] per symbol and judging it against a widened config.start_date would re-fetch endlessly without ever closing the gap.
- [quick-260906-eme]: _WELL_FORMED_TICKER admits digits ([A-Z0-9]) though zero live cells carry one -- the plan's literal [A-Z] would have raised on the repo's own ADDED1/TEMP1/GONE1 synthetic fixture symbols (28 refs across 4 test files). A false positive here raises, falls back to the stale cache and blocks every refresh until someone edits Wikipedia; the malformations the guard exists to catch are all still rejected.
- [quick-260906-eme]: Wikipedia ticker cells are normalized -> logged -> validated -> raised-on, never bare-raised and never silently corrected. Only LEADING/TRAILING delimiters are stripped; an interior one raises because it means two cells were merged by a parser regression, not an editor typo.
- [quick-260906-eme]: EXCLUDE_NON_COMMON_SECURITY_TYPES defaults to False on TiingoRosterFetcher and is opted into by USEquityUniverseFetcher alone -- that default is the entire mechanism freezing nasdaq_all's semantics (Locked Decision A4/D-02). us_all is therefore NO LONGER a strict superset of nasdaq_all, documented as intentional rather than left as an inconsistency to 'fix'.
- [Phase 03.2]: mock_alpaca_client probes acquisition.alpaca with importlib.util.find_spec before monkeypatch.setattr, rather than relying on raising=False — monkeypatch.setattr on a dotted string IMPORTS the module; raising=False only tolerates a missing ATTRIBUTE, so the mock_tiingo_client pattern raised ImportError the moment the fixture was used. While acquisition/alpaca.py is absent there is nothing to patch AND nothing that could issue a real request, since _AlpacaMarketDataClient does not exist for any caller to construct. The guard is deliberately narrow (find_spec is not None, never except Exception) so a genuine ImportError from a broken module still surfaces instead of leaving the fixture silently unpatched.
- [Phase 03.2]: acquisition_config expresses the vendor purely through its paths; AcquisitionConfig.vendor is deferred to 03.2-02 Task 1 — The raw root terminates at the vendor segment and the watermark root is its sibling, which is everything a Wave-0 assertion needs. Introducing the dataclass field in the same commit that uses it beats faking it here behind a guarded dict splat that would have to be unwound one plan later.
- [Phase 03.2]: alpaca_bars_page RAISES on a project column name instead of passing it through, and test_raw_hive_layout.py ships a test asserting the BUG (the silent two-vendor merge), not only the fix — A fixture that accepted `close` would let a test be written against a shape Alpaca never emits, hiding the vendor-to-project field mapping where a swapped o/c reads as plausible data forever. Symmetrically, reproducing the measured merge offline is what keeps SC-7's basename assertion load-bearing rather than decorative -- if polars ever stops merging, that test fails and routes the reader to re-derive SC-7 rather than delete it.
- [Phase 03.2]: Wave-0 self-test names match a phase -k selector only where the self-test honestly covers that ground (vendor_isolation, resume); abort_is_first / no_data / rate_limit / idempotent / fingerprint / prun were deliberately NOT forced onto a name — A test name that matches a selector without testing that behaviour is the same green-but-empty lie the exit-5 trap produces -- it would make a later task's -k command pass before the behaviour exists. Those six selectors match zero tests today and their exit-5 must not be read as green.
- [Phase 03.2]: D-19 confirmed as-proposed: all nine one-way raw-tier on-disk contracts are locked — Raw hive root terminates at the vendor segment; the watermark root is its SIBLING (a .json inside the raw tree breaks pl.scan_parquet); shard names are fully deterministic; RAW_HIVE_KEYS is declared once in enums/data.py; tick's data_type= is a hive key rather than a new Frequency token; the intraday date= key is the US/Eastern SESSION date. date-key-for-daily, utc-day-key and watermarks-inside-vendor-dir were rejected on the record.
- [Phase 03.2]: A completed page ledger must NOT veto a re-fetch the symbol layer already decided on — download(resume=False) issued zero vendor requests because _fetch_batch returned early on ledger.is_complete(). The ledger answers 'where within this batch do I resume', never 'should this batch run' -- those are D-05's two separate layers. It now resets instead, which is safe because deterministic shard names make a redo an overwrite.
- [Phase 03.2]: Alpaca credential env-var NAMES live in module-level constants, never on the patchable transport class — AlpacaAcquisition._scrub read them off _AlpacaMarketDataClient, which tests/conftest.py replaces wholesale. A security control reachable through an indirection whose entire purpose is to be swapped out can be silently disabled by a test double.
- [Phase 03.2]: polars abbreviates long scan source lists, so a naive .pqt count in an .explain() pruning assertion reads exactly backwards — The plan called the pruning test the easiest to write wrongly. explain() renders '[first.pqt, ... 4 other sources]', so count('.pqt') returns 1 for an unpruned five-file scan and 2 for a pruned two-file one. _scan_source_count parses both forms and carries its own self-test.
- [Phase 03.2]: A -k selector that does not reach the one test covering a mechanism is a weaker guarantee than it appears — Deleting PageLedger._load's fingerprint check left -k fingerprint green, because the only covering test was named ..._for_a_different_roster_... Found by mutation testing; fixed by renaming the test into the selector and recording why the name is load-bearing.
- [Phase 03.2]: Orchestration hoisted to `Acquisition`, classification kept per-vendor: `QUOTA_STATUS_CODES` is asserted ABSENT from the base, because Tiingo's 429 is hourly allocation exhaustion while Alpaca's is a per-minute rate limit (D-02) — Hoisting the status set would abort every Alpaca run within seconds while logging an allocation message for a vendor that has no allocation concept. Mutation confirmed the mirror harm too: downgrading Tiingo's quota to `rate_limited` makes a 200-symbol run back off for ~50 minutes instead of stopping.
- [Phase 03.2]: `ConcurrentTiingoAcquisition` retired outright, with all 85 references rewritten in the same commit as the deletion and no transitional alias — The 03.1 D-03 precedent: two live names for one class is exactly the ambiguity a later reader resolves wrongly. The class docstring's "a shared seam is not worth introducing for one subclass" claim was conditioned on there being one subclass; Alpaca made two.
- [Phase 03.2]: `refresh()` packs requests by GROUPING pending symbols on identical recorded `last_date`, then chunking each group — request packing only, with D-06's window rule untouched — Most symbols in a routine refresh share one watermark, so grouping is near-free. The alternative — issue the earliest start for a mixed batch and let dedup absorb the overlap — is correct but re-fetches history nobody asked for and would make 03.2-04's volume guard systematically wrong.
- [Phase 03.2]: A lock that passes on arrival is mutation-verified rather than accepted, and a `-k` selector is verified by what it MATCHES rather than by its exit code — Twelve mutations were run against this plan's locks; one escaped. Replacing `_refresh_batches` with `_batches` inside `_run_once` deleted the whole grouping mechanism and left the entire suite green, because the grouping was proved as a function but never proved to be wired, and the only vendor whose refresh is covered end to end has batch_size 1 where grouping cannot fail.
- [Phase 03.2]: The volume guard's "refuse before the client is constructed" ordering is STRUCTURAL, not procedural: it lives on UniverseCatalog, in a module that imports no acquisition module and binds no Acquisition subclass — Relying on call order lets a later refactor silently invert it while every test still passes. Asserting that acquisition/universe.py cannot reach a client class means there is nothing here that COULD be constructed, whatever the order -- proved alongside a socket tripwire and the removal of every vendor credential from the environment.
- [Phase 03.2]: Volume rows come from estimate_dense_panel's observed_cells, never dense_cells; and tick REFUSES without an explicit rows_per_symbol_day rather than defaulting one — Over 2016-2026 the full US roster is only 36.8% listed, so a dense count overstates a daily backfill by ~2.7x -- a guard that overstates refuses fetches that would have been fine, which is how a guard gets deleted rather than obeyed. Symmetrically, the available ~100k-trades-per-symbol-day figure is unmeasured against Alpaca (RESEARCH A4), so encoding it as a default would make the guard confidently wrong in exactly the regime it exists for.
- [Phase 03.2]: A refusal names EVERY crossed ceiling and offers a narrowing that is RE-ESTIMATED for the shorter window, not scaled from the original figures — requests carries a ceil and a batch floor, so a figure divided by the overshoot ratio is a plausible-looking lie about what the narrowed fetch costs. And reporting only the first crossed ceiling ESCAPED the whole suite under mutation -- the three isolating scenarios each cross exactly one ceiling by construction, so the multi-crossing case was the only one that could catch it and was asserting on arithmetic instead. A caller who raises the one ceiling they were told about, only to hit the next, learns to distrust the message and reaches for force.
- [Phase 03.2]: 03.2-05 D-A: the no_data marker is OMITTED when false, never written as false — Absence becomes the default, so all pre-existing sidecars read back correctly as not-no-data (the old code only wrote a watermark after a successful fetch). Writing false would leave older files ambiguous between 'had data' and 'never said'.
- [Phase 03.2]: 03.2-05 D-B: the coverage classification rule stays blind to the no_data marker — A marked symbol is covered/widened/legacy by the ordinary rule, so a marked symbol whose recorded window is narrower than a later request is still re-fetched. The marker records what the vendor said about a WINDOW, never a permanent verdict about the symbol; the absence of a special case is the design and two tests keep one from being added.
- [Phase 03.2]: 03.2-05 D-C: a refresh may CLEAR a no_data marker but never assert a new one — A refresh queries [watermark, end_date] while stamping a sidecar that records [covered_start, end_date] -- a strictly wider window -- so it has no evidence about the earlier part of the range. Asserting absence there would launder 'no new rows this week' into 'nothing since 2020'. Same carry-through asymmetry start_date already follows (D-04).
- [Phase 03.2]: 03.2-05 D-D: the marker set is computed once per COMPLETED batch, gated on outcome.complete and a clear abort — Alpaca is symbol-major, so page 0 of a 100-symbol batch legitimately carries one symbol; a per-page difference would stamp the other 99 'no data' and skip them forever (RESEARCH Pitfall 4). The completion gate is locked by an injected incomplete BatchOutcome because no end-to-end test can move the flag today -- the mutation survived without it.
- [Phase 03.2]: Intraday `date=` hive key is the US/Eastern SESSION date (RESEARCH A8 resolved); timestamp values stay naive UTC and only the derived key converts — A UTC-derived key files the last ~4 hours of every US session (20:00-24:00 UTC) under the following day, making a one-trading-day query wrong at both edges in the shape that reads as sparse data rather than as a bug. `Acquisition._session_date` is an overridable seam that RAISES when SESSION_TIME_ZONE is undeclared, so a vendor that never considered the boundary fails at the first intraday write.
- [Phase 03.2]: A tick scan is ROOT-SCOPED to `data_type={quotes|trades}` rather than filtered on the hive key — Measured on polars 1.44.1: a `data_type` predicate prunes the query plan correctly and `.collect()` still raises SchemaError, because the expected schema is fixed from the first file discovered. Filtering to `quotes` appears to work only because it sorts before `trades`, so the bug is filename-ordering dependent. This is D-11's vendor-segment lesson one level deeper: a distinction only a predicate enforces is not isolation.
- [Phase 03.2]: Watermark/ledger/failure-manifest sidecars are namespaced by data type where the frequency partitions on one — Quotes and trades share one vendor raw root but their sidecars are `{symbol}.json` with no data_type key, so a completed quotes backfill told the trades run every symbol was covered; it skipped the whole roster and reported success. `1d`/`1m` sidecar paths are unchanged.
- [quick-260906-usg]: Polars factor names derive from the computation GRAPH, never from the factor store on disk -- a store written under n=5 read back under a config saying n=60 yields momentum_60 and fails loudly at lookup, instead of silently reporting the stale name the data happens to carry. Deriving from disk would match the data while ignoring what the config asked for.
- [quick-260906-usg]: The Gap 1 fix is a DELETED override, not an added one. base/factor.py already implements "explicit pin wins, else derive"; FactorPolars had disabled the derivation channel with a no-op _maybe_resolve_factor_names(), leaving only cal() to fill config.factor_names. Deleting it fixes read(), cal() and bare construction at once.
- [quick-260906-usg]: DataBackend.head(n) is an @abstractmethod rather than a limit= keyword on get_lazyframe(). ABC enforcement makes a backend that omits the bounded read impossible to construct; an optional keyword is satisfied by plain inheritance and only fails at whichever call site passes it. Accepted consequence: constructing a FactorPolars now performs a bounded disk read (~25-50 ms, flat in store size).
- [quick-260906-usg]: XrBackend.head must NOT write its slice back to self.data, unlike its in-place filter_by_date/filter_by_symbol siblings -- a mutating probe would silently leave cal() computing over 8 rows with nothing downstream able to tell. Locked by tests/test_backend_head.py::test_head_does_not_mutate_backend_state and its mutation.
- [quick-260906-w3t]: DataBackend.head takes the store PATH and opens the store itself (head(path, n), path first, mirroring read(path, **kwargs)). Reaching the store through read() ran BaseDataset._filter(), narrowing data_backend.data IN PLACE, and XrBackend.read()'s cache early-return made that narrowing permanent -- so the FactorPolars name probe, which fires BEFORE _reset_dataset_config() widens the window, silently dropped the factor's entire lookback (RV-01: 29 timestamps instead of 49, momentum_5 17.2% NaN, nothing raised).
- [quick-260906-w3t]: read() is UNCHANGED -- same signature, same meaning, no filter= parameter. The RV-01 fix removes a CALLER, not a method. The rejected alternative (filter= on read() plus clearing self.data afterwards) would have destroyed a caller's already-read data, failed to compose with the cache, and made correctness depend on one side effect cancelling another.
- [quick-260906-w3t]: XrBackend.head opens with xr.open_dataset, the same opener read() uses -- never xr.open_zarr. Two openers for one store in one class is a divergence waiting to bite; _assert_append_compatible's open_zarr is a separate, append-specific concern.
- [quick-260906-w3t]: RV-02 is deliberately NOT closed and stays filed in 03-VERIFICATION.md. head() carries read()'s explicit Path(path).exists() guard and its exact FileNotFoundError message on both backends, now locked by a test rather than prose. Accepted narrowing: a dataset whose data exists only in memory (the _reset_symbols() -> from_raw_data() fallback) was previously probeable via the cache early-return and now raises. Closing RV-02 means deciding whether name derivation may be DEFERRED, which re-opens D-05's construction-time-names contract -- a user design call, not a side effect of a mutation fix.
- [quick-260906-w3t]: Every pre-existing fixture left DatasetConfig.start_date/end_date unset, making _filter() a no-op -- which is exactly why 384 green tests could not see RV-01. Restoring .read() on the probe WITHOUT the new dated fixture leaves the suite green (verified as a mutation). The dated fixture IS the guard; the fix without it would be unprotected.
- [Phase 03.1]: RAISE rather than warn-and-clamp on a pre-coverage start_date in UniverseCatalog.get_symbols_in_range, with an INCLUSIVE boundary (strict `<`) — DATA-05 says 显式报错 literally; the sibling query already raises, and a different resolution would leave two membership queries with two contracts — the exact condition that produced this gap. Clamping would return the identical censored roster plus a log line, since every membership interval already starts at or after its own coverage start. Reversible: undo is one deleted call.
- [Phase 03.1]: The coverage boundary lives in ONE shared helper (UniverseCatalog._assert_within_coverage) called by both membership queries, derived from MEMBERSHIP_FETCHERS — A fifth index must inherit the boundary on BOTH queries by registration alone; neither query body contains a category literal. Two independent guards is how get_symbols_in_range silently drifted out of the contract in the first place.
- [Phase 03.1]: IndexConstituentDataset._clamp_coverage_start() keeps CLAMPING where the two catalog queries RAISE — a design choice, recorded in README.md and example/constituent.md so it is not 'aligned' away — The panel consumes a framework-supplied config default (enums/constant.py:Date.START_DATE) nobody typed, so raising would explode every default construction; a query date is one a caller actually asked, so refusing it is right.
- [Phase 3]: [quick-260907-rjq]: The AST ordering guard's deny set is derived by TRANSITIVE REACHABILITY to a `from config import` name, not by the planned `_build*` name prefix -- every script opens __main__ with `_build_arg_parser()`, which reaches no factory and must run before parse_args(), so the prefix rule made the guard unsatisfiable. Reachability is also strictly stronger: a factory-reaching helper named without the prefix escapes the prefix rule and not this one.
- [Phase 3]: [quick-260907-rjq]: The derived-script-list lock ESCAPED its own mutation as first written -- a substring scan for `add_data_dir_arg` still matched ingest_alpaca.py after the parser call was deleted, because the import stayed behind, leaving --data-dir dead on that script with the suite green. Registration is now detected as an AST Call plus an assertion that the flag reaches the parser __main__ actually uses.
- [Phase 3]: [quick-260907-rjq]: --data-dir sets a process-level override consulted by ONE resolver (config.get_data_root), precedence CLI > QUANTLAB_DATA_DIR > repo-root data/. It is an explicit apply_data_dir(args) call in each __main__, never an argparse action= side effect: utils/cli.py already made that call deliberately for the volume guard, and a root-relocating side effect hidden inside argument parsing is exactly the invisible action that comment exists to prevent.

### Pending Todos

- **[major/data]** No ticker-rename mapping between index-membership history and the price roster. Membership comes from Wikipedia change logs (period-correct tickers: `FB`); prices come from Tiingo, which rewrites a renamed security's whole history onto the CURRENT ticker (`META`, start_date = Facebook's IPO). 89 of 876 sp500_constituent symbols have no counterpart in `us_all`. Fails SILENTLY as "no data in window". Not reachable until a backtest walks membership through history (Phase 4+). `.planning/todos/pending/2026-09-07-no-ticker-rename-mapping-between-membership-history-and-pric.md`
- **[minor/data]** Class-share ticker delimiter differs between the two sources — Wikipedia writes `BRK.B`/`BF.B`, Tiingo writes `BRK-B`/`BF-B`; both securities exist on both sides, only the notation blocks the join. `TRADEABLE_TICKER_PATTERN` admits both, so this is a join problem, not a validation one. `.planning/todos/pending/2026-09-07-normalize-ticker-delimiter-between-membership-panel-and-pric.md`
- **[major/database]** New listings break cross-run incremental zarr appends — `XrBackend.append()`'s coordinate guard correctly refuses when the symbol union grows, but there is no path to widen an existing store's symbol axis, so a periodic refresh halts after any new listing. Guard must stay (it prevents silent history mis-attribution when the symbol count is unchanged but labels differ — measured). **DONE** (quick task 260906-x2s, commits 874cdb8/fa8e1d8/ea46887: both directions shipped — `XrBackend.widen_symbol_axis`/`widen_and_append` plus `--on-new-listing {refuse,rebuild,widen}`) — `.planning/todos/completed/2026-09-06-support-new-listings-in-incremental-zarr-appends.md`
- Broader test-code/temporary-script cleanup (`test.py`, `test_nt.ipynb`, `read_mock_data_sink.py` as CLEAN-02 "test code / temporary scripts" candidates beyond the Phase 1 hardcoded-path fix applied to `test.py`) is intentionally deferred to Phase 7 (QUAL-02: "no leftover test/temp/redundant code remains outside the established `tests/` structure"), not silently dropped from Phase 1. Phase 1 only fixes `test.py`'s hardcoded per-developer paths (01-01 Task 3); it does not relocate, rename, or remove these files.

### Blockers/Concerns

- Tiingo API key currently leaked in `scripts/download_stock_data_from_tiingo.py` and pushed to `origin/main` — user should revoke/rotate the key in the Tiingo dashboard independent of the git-history reset planned in Phase 1.

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 260906-x2s | Support new listings in incremental zarr appends: widen-in-place symbol axis path on XrBackend, then detect-and-rebuild at the Dataset layer | 2026-09-07 | ea46887 | [260906-x2s-support-new-listings-in-incremental-zarr](./quick/260906-x2s-support-new-listings-in-incremental-zarr/) |
| 260907-10t | Reconcile the fetch-time ticker guard with the us_all roster: drop malformed roster entries, widen the pattern to the multi-suffix securities the roster deliberately keeps | 2026-09-07 | 4d18336 | [260907-10t-reconcile-the-fetch-time-ticker-guard-wi](./quick/260907-10t-reconcile-the-fetch-time-ticker-guard-wi/) |
| 260907-1du | Fix three ingest-path defects: empty-store construction crash, double raw-to-xarray conversion, validate_schema false-positive null warnings | 2026-09-07 | 7bbd1f9 | [260907-1du-fix-three-ingest-path-defects-empty-stor](./quick/260907-1du-fix-three-ingest-path-defects-empty-stor/) |
| 260907-fl6 | Fix the defects surfaced while writing the example/ guides — model-layer correctness (early stopping, alphabetical tensor axis, eval-mode inference), the layer's first tests, three deletions, a persistent refit optimizer, and the `_*_one_epoch`→`_*_one_batch` rename | 2026-09-07 | f80e9fb | [260907-fl6-fix-the-defects-surfaced-while-writing-t](./quick/260907-fl6-fix-the-defects-surfaced-while-writing-t/) |
| 260907-rjq | Add a --data-dir CLI parameter to the data acquisition scripts to override the download/storage root | 2026-09-07 | 4552b0c | [260907-rjq-add-a-data-dir-cli-parameter-to-the-data](./quick/260907-rjq-add-a-data-dir-cli-parameter-to-the-data/) |

### Roadmap Evolution

- Phase 03.1 inserted after Phase 3: Index Historical Constituents Data Layer — daily point-in-time S&P 500 / Nasdaq-100 membership panels as xarray (URGENT)
- Phase 03.2 inserted after Phase 3: Multi-source acquisition abstraction: batched multi-symbol fetching + pagination in the Acquisition base, plus Alpaca source classes (daily bars, minute bars, corporate actions, high-frequency quotes/trades with a volume guard) (URGENT)
- Phase 03.3 inserted after Phase 03.2: Tick data storage: a non-dense event-axis Dataset and its raw-to-Zarr conversion. Committed follow-up deferred from 03.2 by D-18, not urgent work.
- Phase 03.4 inserted after Phase 3: Data Source Registry — one registered descriptor per data source, consumed by quantlab's own CLI and by the out-of-repo `quantlab-console` operator surface. The TUI/web console itself is a SEPARATE repository and is not on this roadmap. (URGENT)

## Deferred Items

Items acknowledged and carried forward from previous milestone close:

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| v2 | PLAT-01..06 (service API, multi-user, online factor/model editing, web frontend) | Deferred to v2 | Initial requirements definition |
| v2 | DATA-V2-01 (full tick-data production ingestion) | Deferred to v2 | Initial requirements definition |
| v2 | DATA-V2-02 (full minute-frequency historical backfill) | Deferred to v2 | Initial requirements definition |
| Phase 7 | `test.py`, `test_nt.ipynb`, `read_mock_data_sink.py` broader cleanup/relocation (CLEAN-02 test-code candidates) | Deferred to Phase 7 (QUAL-02) | Phase 1 planning revision, 2026-09-04 |

## Session Continuity

Last session: 2026-09-08T00:15:07.484Z
Stopped at: Completed quick task 260907-rjq (--data-dir CLI parameter)
rebuild the Zarr stores). NOTE: quick task 260906-26o Task 3 is still an OPEN blocking human
checkpoint (stamp legacy Tiingo watermarks) -- untouched by this task.
Resume file: None
</content>
