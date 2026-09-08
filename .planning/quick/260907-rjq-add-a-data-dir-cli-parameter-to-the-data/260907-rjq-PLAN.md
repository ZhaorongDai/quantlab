---
phase: quick-260907-rjq
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - config/__init__.py
  - utils/cli.py
  - refresh_us_equity_universe.py
  - ingest_tiingo.py
  - ingest_alpaca.py
  - ingest_us_equity.py
  - ingest_binance_spot.py
  - README.md
  - tests/conftest.py
  - tests/test_data_dir_cli.py
  - tests/test_config_paths.py
  - tests/test_factor_kunquant.py
autonomous: true
requirements: [DDIR-01, DDIR-02, DDIR-03, DDIR-04, DDIR-05]
estimate:
  tokens: 55000
  raw_tokens: 55000
  tasks: 3
  confidence: low

must_haves:
  truths:
    - "DDIR-01: `--data-dir /X` on any of the five acquisition entry points makes that run read and write under `/X` -- visible in `raw_data_dir_path`, `zarr_file_path`, `watermark_path` and the universe table's `output_path`."
    - "DDIR-02: Precedence is CLI override > `QUANTLAB_DATA_DIR` > repo-root `data/`, in that order, and it is one knob -- no second competing path root and no hardcoded volume anywhere."
    - "DDIR-03: Omitting `--data-dir` reproduces today's paths exactly, for both `QUANTLAB_DATA_DIR` users and repo-default users. The existing 505-test suite stays green."
    - "DDIR-04: In every script that offers the flag, the override is in effect BEFORE the first `config/` factory call in `__main__`, so no config object can be constructed carrying a stale root."
    - "DDIR-05: `ingest_binance_spot.py --raw-data-dir` still redirects the raw CSV directory alone and COMPOSES with `--data-dir` (root relocated, raw dir still pointed elsewhere)."
    - "No prose left in the repo claims `QUANTLAB_DATA_DIR` is the ONLY path knob."
  artifacts:
    - config/__init__.py
    - utils/cli.py
    - refresh_us_equity_universe.py
    - ingest_tiingo.py
    - ingest_alpaca.py
    - ingest_us_equity.py
    - ingest_binance_spot.py
    - tests/test_data_dir_cli.py
    - tests/test_config_paths.py
    - tests/conftest.py
  key_links:
    - "utils/cli.py:apply_data_dir() -> config.set_data_root() -- the ONE place the dependency-light CLI helper module reaches the config layer, via a call-time deferred import."
    - "config.get_data_root() -> every one of the 14 path expressions in config/__init__.py -- the single resolver the whole factory file derives from."
    - "each script's __main__: apply_data_dir(args) BEFORE the first path-consuming call -- locked by an AST guard whose script list is DERIVED from a repo glob, so a sixth entry point cannot opt out by omission."
    - "tests/conftest.py autouse reset -> the process-global override, cleared before and after every test, so the knob cannot leak into the other 505 tests."
---

<objective>
Add a `--data-dir` CLI parameter to the five data-acquisition entry points so the
download/storage root can be overridden per run from the command line, instead of
only via the `QUANTLAB_DATA_DIR` environment variable.

Today every storage path in this repo funnels through
`config/__init__.py:_data_root()`, which reads `QUANTLAB_DATA_DIR` and otherwise
falls back to a repo-root-relative `data/`. There is no way to say "put this run
on /Volumes/BigDisk" without exporting an env var first.

Purpose: a per-run storage root, with the existing env var and repo default
untouched beneath it.
Output: `config.get_data_root()`/`set_data_root()`, `utils.cli.add_data_dir_arg()`/
`apply_data_dir()`, the flag wired into all five entry points, and a test suite
locking precedence, reach, ordering and the no-regression guarantee.
</objective>

<execution_context>
@~/.claude/gsd-core/workflows/execute-plan.md
@~/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@CLAUDE.md

@config/__init__.py
@utils/cli.py
@tests/test_config_paths.py
</context>

<decisions>

Recorded here because each one has a rejected alternative that a later reader
would otherwise re-litigate.

**D-01 — One knob, three levels.** `--data-dir` sets a process-level override
consulted by the single resolver in `config/__init__.py`, with precedence
**CLI override > `QUANTLAB_DATA_DIR` > repo-root `data/`**. No second path root
is introduced and no volume is ever hardcoded, which keeps the 260906-0iy D-04
promise intact in substance: there is still exactly ONE data root, it simply now
has one more way to be set.

**D-02 — `_data_root()` is RENAMED to `get_data_root()`, not aliased.** The repo
has settled this twice (03.1 D-03, quick-260906-usg): two live names for one
thing is the ambiguity a later reader resolves wrongly. All 14 in-file call sites
move in the same commit as the definition; the three prose mentions elsewhere
(`README.md`, `ingest_us_equity.py`, `tests/test_factor_kunquant.py`) are fixed
inside this plan, each in the task that already owns that file.

**D-03 — `apply_data_dir(args)` is an EXPLICIT call in each `__main__`, not an
argparse `action=` side effect.** A custom argparse Action firing during
`parse_args()` would make the ordering structurally unbreakable, which is
tempting. It was rejected because `utils/cli.py` already made the opposite call
deliberately for the volume guard -- "each ingest script names
`assert_acquisition_volume_fits` at its own call site, so a reader of the script,
and a grep across the entry points, sees the guard where the decision is made,
rather than one level of indirection away." A root-relocating side effect hidden
inside argument parsing is exactly the kind of invisible action that comment
exists to prevent. The ordering is instead enforced by the AST guard in Task 3.

**D-04 — the `config` import inside `apply_data_dir` is DEFERRED to call time.**
`utils/cli.py`'s module docstring pins its module-scope project dependency
surface at `base.chunking` and `base.data`, and `_explicit_symbol_catalog`
already defers its `acquisition.universe` import for that stated reason. A
module-scope `from config import set_data_root` would drag `dataset.backend`,
`dataset.spot`, `dataset.stock` and `base.config` into every import of the
dependency-light CLI helper module.

**D-05 — `--data-dir` and `ingest_binance_spot.py --raw-data-dir` COMPOSE;
neither supersedes the other.** They operate at different levels.
`--data-dir` relocates the whole root (the Zarr store, the nautilus catalog AND
the default raw directory). `--raw-data-dir` points at one pre-existing directory
that does not follow the `data/{market}/{frequency}/...` convention at all -- its
whole purpose, per that script's docstring, is "no need to move, copy, or
symlink files into the project's convention path". So `--data-dir /Volumes/BigDisk
--raw-data-dir ~/Downloads/klines` reads CSVs from `~/Downloads/klines` and writes
the Zarr under `/Volumes/BigDisk`, which is the useful combination. Order is
unchanged from today: the root override applies before `spot_kline_config()`, the
narrow raw override mutates `raw_data_dir_path` after construction.
`--raw-data-dir` is NOT deprecated and NOT removed.

**D-06 — `set_data_root(None)` clears the override.** A process-global knob needs
a reset, or the first test that sets it silently redirects every later test in the
session. Clearing also makes DDIR-03 expressible as a positive property ("with the
override cleared, paths equal today's") rather than as an absence.

**D-07 — `expanduser()` yes, `resolve()` no; empty string RAISES.** The env path
is `Path(env_value)` with no resolution, so resolving only the CLI path would make
the two knobs behave differently on a symlinked root (on macOS `/tmp` resolves to
`/private/tmp`) and would defeat a user who passed a symlink deliberately.
`expanduser()` IS applied because a quoted `--data-dir '~/quantlab-data'` reaches
Python with a literal tilde and would otherwise create a directory named `~`. An
empty or whitespace-only value raises `ValueError`: an env var of `""` is already
falsy and falls through to the default, but a user who typed `--data-dir ""` on
the command line meant something, and silently meaning "repo default" is wrong.

**D-08 — `set_data_root` does not check existence and does not mkdir.** The
acquisition layer creates its own directories, and the env knob validates nothing
today; validating one knob and not the other is how two knobs drift apart.

</decisions>

<tasks>

<task type="tracer" tdd="true">
  <name>Task 1: End-to-end `--data-dir` on one script -- root override, shared CLI helper, one wired entry point</name>
  <files>config/__init__.py, utils/cli.py, refresh_us_equity_universe.py, tests/conftest.py, tests/test_data_dir_cli.py</files>
  <read_first>config/__init__.py (lines 19-45 for the resolver and its two derived roots; lines 85 and 133 for the "only path knob" docstring claims that this task falsifies), utils/cli.py (module docstring's dependency-surface promise, and `_explicit_symbol_catalog` at lines ~406-497 for the deferred-import precedent D-04 mirrors), refresh_us_equity_universe.py (all 44 lines), tests/conftest.py (fixture style only -- it has no autouse fixture today)</read_first>
  <behavior>
    Precedence (`config.get_data_root()`), per D-01:
    - override set, env unset -> override
    - override set, env set -> override (CLI wins)
    - override cleared, env set -> `Path(env)`
    - override cleared, env unset -> repo-root `data/` (`Path(config.__file__).resolve().parent.parent / "data"`)

    `config.set_data_root`, per D-06/D-07/D-08:
    - accepts `str | os.PathLike`, stores `Path(value).expanduser()`, does NOT `resolve()`, does NOT mkdir, does NOT check existence
    - `set_data_root(None)` clears the override and returns `None`
    - `set_data_root("")` and `set_data_root("   ")` raise `ValueError`
    - returns the `Path` it stored, so a caller can report it

    `utils.cli.add_data_dir_arg(parser)`:
    - registers `--data-dir` (`type=str`, `default=None`) and returns the parser
    - after it, `parser.parse_args([])` yields `args.data_dir is None`

    `utils.cli.apply_data_dir(args)`, per D-03/D-04:
    - `args.data_dir is None` (or the attribute is absent) -> returns `None` and leaves the override untouched
    - `args.data_dir` set -> calls `config.set_data_root` and returns the stored `Path`
    - imports `config` at CALL time, never at module scope

    End-to-end through `refresh_us_equity_universe.py`:
    - `--data-dir <tmp>` makes `universe_config().output_path` and `.cache_dir` land under `<tmp>`
    - no `--data-dir` leaves both byte-identical to today
  </behavior>
  <action>
Rename and extend the root resolver in `config/__init__.py`, add the two shared
CLI helpers to `utils/cli.py`, and wire ONE entry point end to end. This is the
thin vertical slice: argv -> `apply_data_dir` -> `set_data_root` -> resolver ->
a real config factory's path -> disk destination. It is production code, not a
prototype; Task 2 expands the same wiring horizontally to the other four scripts.

`config/__init__.py`:
- Add a module-level `_DATA_ROOT_OVERRIDE: Path | None = None`.
- Add `set_data_root(path)` per D-06/D-07/D-08. Its docstring states the D-01
  precedence, says the value is expanded but deliberately NOT resolved and why
  (D-07), and says `None` clears it (D-06).
- Rename `_data_root()` to `get_data_root()` per D-02 and give it the precedence
  body: override first, then `os.environ.get("QUANTLAB_DATA_DIR")`, then the
  repo-root `data/` default. Update every in-file caller -- the two derived roots
  `_market_data_root`/`_market_downloads_root` plus the `catalog_path`,
  `cache_dir`, `output_path`, `file_path` expressions. Grep the file afterwards
  for the old name; zero hits is the bar. **Do not name the retired symbol in any
  docstring, comment or migration note in this file** -- the gate greps this file
  itself, so a docstring saying "renamed from the old private helper" keeps it red
  forever. State the CURRENT name and its precedence; the rename lives in git
  history, not in prose.
- Update the two factory docstrings that assert `QUANTLAB_DATA_DIR` "remains the
  only path knob" (in `stock_kline_config` and `stock_acquisition_config`) to
  state the D-01 precedence instead. Keep the substance of the original claim:
  `subdir`/`store_name` are still subdirectories beneath the one root, and there
  is still no second competing root and no hardcoded volume -- the root simply has
  one more way to be set. Do not merely delete the sentence.

`utils/cli.py`:
- Add `add_data_dir_arg(parser)` alongside the existing `add_*_args` helpers, with
  help text that names the D-01 precedence explicitly, says the directory is
  created by the run rather than required to exist, and (for the reader landing
  from `ingest_binance_spot.py`) says it relocates the ROOT while
  `--raw-data-dir` points at one pre-existing raw directory.
- Add `apply_data_dir(args)` per D-03/D-04, with a docstring recording BOTH the
  deferred import's reason and the reason this is an explicit call rather than an
  argparse action -- pointing at the volume-guard call-site comment already in
  this file as the precedent it follows.
- Extend the module docstring's dependency-surface paragraph so the new call-time
  `config` import is stated where the promise is made, not left to be discovered.

`refresh_us_equity_universe.py`:
- `add_data_dir_arg(parser)` in the parser build, and `apply_data_dir(args)`
  immediately after `parse_args()` -- BEFORE `universe_config()` (DDIR-04). Add a
  one-line comment saying why the position matters: the factory snapshots paths at
  construction time, so an override applied after it silently does nothing.
- Extend the usage block with a `--data-dir` example.

`tests/conftest.py`:
- Add an `autouse=True` fixture that clears the override (`config.set_data_root(None)`)
  before AND after every test, importing `config` inside the fixture body so
  collection cost is unchanged. Its docstring says what it prevents: a
  process-global knob set by one test redirecting every later test's paths.
  Clearing on BOTH sides matters -- the "after" half contains a test that sets it,
  the "before" half contains anything that sets it outside a fixture.

`tests/test_data_dir_cli.py` (new):
- Write the `<behavior>` cases above as tests. Use `monkeypatch.delenv(
  "QUANTLAB_DATA_DIR", raising=False)` wherever the env layer must be absent, and
  `monkeypatch.setenv` where it must be present -- a developer machine may have it
  exported and the assertions must not depend on that.
- Include the end-to-end case: build `refresh_us_equity_universe`'s parser, parse
  `["--data-dir", str(tmp_path)]`, call `apply_data_dir`, then assert
  `universe_config().output_path` starts with `str(tmp_path)`. To parse the real
  parser, extract `refresh_us_equity_universe.py`'s parser construction into a
  module-level `_build_arg_parser()` matching the other four scripts' shape, so
  the test drives the SAME parser `__main__` uses rather than a replica.
  </action>
  <verify>
    <automated>uv run pytest tests/test_data_dir_cli.py -q &amp;&amp; uv run pytest tests/test_config_paths.py -q &amp;&amp; uv run python refresh_us_equity_universe.py --help | grep -q -- '--data-dir' &amp;&amp; ! grep -nE '(^|[^A-Za-z0-9_])_data_root' config/__init__.py</automated>
  </verify>
  <done>`get_data_root()` resolves override > env > default; `set_data_root(None)` clears; `--data-dir` on `refresh_us_equity_universe.py` relocates `universe_config().output_path`; omitting it changes nothing; `_data_root` no longer appears in `config/__init__.py`; the autouse reset fixture is in place.</done>
  <reversibility rating="reversible">A process-level override with an explicit clear; removing it is deleting one module global, one function and one flag. Nothing is written to disk in a new shape.</reversibility>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Expand the flag to the remaining four entry points, and fix the contradicted prose</name>
  <files>ingest_tiingo.py, ingest_alpaca.py, ingest_us_equity.py, ingest_binance_spot.py, README.md</files>
  <read_first>ingest_tiingo.py `__main__` (lines ~103-152: `parse_args` -> `validate_roster_args` -> `universe_config()` -> `_build_configs`), ingest_alpaca.py `__main__` (lines ~274-290) and `_build_configs` (lines ~143-190), ingest_us_equity.py module docstring (lines 1-20, the "ONLY path knob" claim) and `__main__` (lines ~338-390), ingest_binance_spot.py (all 89 lines), README.md lines 55-62 and 290-320</read_first>
  <behavior>
    - Each of the four scripts responds to `--help` with a `--data-dir` entry, with no credentials set in the environment.
    - In each `__main__`, `apply_data_dir(args)` runs before the first call that reaches a `config/` factory -- including the indirect reachers `_build_configs(args)` (tiingo, alpaca) and `_build_dataset_config(args)` (binance).
    - `ingest_binance_spot.py`: with `--data-dir D` alone, `raw_data_dir_path`, `zarr_file_path` and `catalog_path` all sit under `D`. With `--data-dir D --raw-data-dir R`, `raw_data_dir_path == R` while `zarr_file_path` stays under `D` (D-05).
    - No file in the repo still asserts `QUANTLAB_DATA_DIR` is the ONLY path knob, and no file still names `_data_root()`.
  </behavior>
  <action>
Wire the same two calls into the remaining four scripts, following exactly the
shape Task 1 proved. In each: `add_data_dir_arg(parser)` in the parser build
(next to the other shared `add_*` helpers), and `apply_data_dir(args)` in
`__main__` immediately after `parse_args()`, before anything that can reach a
`config/` factory.

`ingest_tiingo.py` and `ingest_alpaca.py`: the call goes after `parse_args()` and
before `validate_roster_args(...)`/`_validate_data_type(...)` and before the
`UniverseCatalog.load(universe_config())` line. Note both scripts reach factories
indirectly through `_build_configs`, so placing it after `parse_args` is what
covers both paths at once.

`ingest_us_equity.py`: the call goes after `parse_args()` and before
`UniverseCatalog.load(universe_config())`. Also rewrite the module docstring's
storage paragraph, which currently claims `QUANTLAB_DATA_DIR` is "the ONLY path
knob: this script hardcodes no volume and adds no competing setting". Replace it
with the D-01 precedence, preserving the part that is still true and still worth
saying -- this script hardcodes no volume and adds no competing root. Also fix
the `config/__init__.py:_data_root` reference to `get_data_root` (D-02), and the
`DEFAULT_SUBDIR`/`DEFAULT_STORE_NAME` comment that says both resolve "BENEATH
`QUANTLAB_DATA_DIR` (D-04)" -- they now resolve beneath the resolved root.

`ingest_binance_spot.py`: per D-05, `--data-dir` and `--raw-data-dir` compose.
`apply_data_dir(args)` goes in `__main__` after `parse_args()` and before
`_build_dataset_config(args)`; `_build_dataset_config` itself is unchanged in
structure -- `spot_kline_config()` then the existing post-construction
`raw_data_dir_path` mutation, which now overrides a root-relocated default rather
than a repo-relative one. Update the module docstring: state that the default raw
directory follows the resolved root, that `--data-dir` moves the whole root while
`--raw-data-dir` redirects only the raw CSV directory, and that combining them is
supported and what it means. Add a combined usage example. Extend the
`--raw-data-dir` help text with one sentence naming the composition, so a reader
who sees only `--help` learns the same thing.

`README.md`: update the `QUANTLAB_DATA_DIR` bullet (~line 293) to describe the
three-level precedence and name `--data-dir` as the per-run override available on
the acquisition entry points; update the paragraph at ~line 316 that says factory
paths derive from "`QUANTLAB_DATA_DIR` (or the repo-root `data/` default)"; and
update the `_data_root()` mention at ~line 60 to `get_data_root()` (D-02).

**Comment-text discipline for the prose rewrites in this task.** The gate
negative-greps the phrase "only path knob" and the retired symbol name across the
repo, so the replacement prose must state the NEW precedence in its own words and
must not quote the retired claim in order to correct it -- no "no longer the only
path knob", no "formerly `_data_root()`". Write what is true now.

Do not add the flag to `cal.py`, `train_model.py`, `test.py`,
`get_binance_instruments.py` or anything under `scripts/` -- those are not
acquisition entry points and are outside this task's scope.
  </action>
  <verify>
    <automated>for f in ingest_tiingo ingest_alpaca ingest_us_equity ingest_binance_spot refresh_us_equity_universe; do uv run python $f.py --help | grep -q -- '--data-dir' || { echo "MISSING --data-dir in $f"; exit 1; }; done &amp;&amp; ! (grep -rni 'only path knob' --include='*.py' --include='*.md' . | grep -v '\.planning/' | grep -v 'tests/test_config_paths.py') &amp;&amp; ! (grep -rnE '(^|[^A-Za-z0-9_])_data_root' --include='*.py' --include='*.md' . | grep -v '\.planning/' | grep -v 'tests/test_factor_kunquant.py')</automated>
  </verify>
  <done>All five entry points expose `--data-dir` in `--help`; `apply_data_dir` precedes every factory-reaching call in each `__main__`; the binance composition is documented in both the module docstring and the `--raw-data-dir` help; README and the `ingest_us_equity.py` docstring describe the new precedence; the only surviving `_data_root` mention is the one Task 3 fixes.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 3: Lock the contract -- precedence matrix, path reach, ordering guard, no-regression</name>
  <files>tests/test_config_paths.py, tests/test_data_dir_cli.py, tests/test_factor_kunquant.py</files>
  <read_first>tests/test_config_paths.py (all 173 lines -- extend its style, do not invent a second convention), tests/test_data_dir_cli.py (as written in Task 1), tests/test_ingest_tiingo_universe_wiring.py (the per-script wiring-test pattern this repo already uses), tests/test_factor_kunquant.py lines 318-322 (the one remaining `_data_root()` prose mention)</read_first>
  <behavior>
    In `tests/test_config_paths.py` (the path-convention half):
    - Precedence matrix, all four cells of D-01, asserted through a REAL factory path rather than through the resolver alone.
    - Reach: with the override set to `tmp_path`, every one of these lands under it -- `stock_acquisition_config(symbols=("AAPL",)).raw_data_dir_path`, the same config's `.watermark_path`, `stock_kline_config().zarr_file_path`, `universe_config().output_path`, `spot_kline_config().raw_data_dir_path`, `sp500_constituent_config().zarr_file_path`.
    - Reach is exhaustive by CONSTRUCTION, not by enumeration: assert that `config/__init__.py` contains no path literal that bypasses `get_data_root()` -- every `Path(`/string path expression in the module's factory bodies is rooted in `get_data_root()` or one of its two derived roots.
    - No-regression: with the override cleared and `QUANTLAB_DATA_DIR` set to `E`, paths are `E`-rooted; with both absent, `stock_acquisition_config` still ends `downloads/us_equity/1d/nasdaq_data/tiingo` and `stock_kline_config` still ends `data/us_equity/1d/stock.zarr` -- the same literals the existing `test_stock_config_defaults_are_byte_identical_without_the_new_arguments` pins.

    In `tests/test_data_dir_cli.py` (the CLI half):
    - Ordering guard (DDIR-04) over every entry point, AST-based.
    - The guarded script list is DERIVED: glob the repo root for `*.py` whose source contains `add_data_dir_arg`, and assert that set equals the five expected names. A sixth entry point that adds the flag is guarded automatically; a script that drops the flag fails here.
    - The path-consuming deny set per script is DERIVED too: the names that script imports `from config import ...`, plus any local call whose name starts with `_build`.
    - Each script's `__main__` contains exactly one `apply_data_dir` call, and its position precedes the first deny-set call.
    - Binance composition (D-05), driven through the real `_build_arg_parser()`/`_build_dataset_config()`: `--data-dir D` alone roots all three paths under `D`; `--data-dir D --raw-data-dir R` gives `raw_data_dir_path == R` with `zarr_file_path` still under `D`.
  </behavior>
  <action>
Write the locks that only become expressible once all five scripts are wired.

`tests/test_config_paths.py` -- extend, matching the existing file's style
(module-level test functions, docstrings that name the decision being pinned):
add the precedence matrix, the reach assertions and the no-regression pair from
`<behavior>`. Assert on the four path FIELDS the task set out to cover
(`raw_data_dir_path`, `zarr_file_path`, `watermark_path`, universe `output_path`)
through real factory calls. For the by-construction check, read
`config/__init__.py`'s own source and assert every factory-body path expression
is rooted in `get_data_root()` or one of the two derived roots -- an enumerated
list of factories would go stale the day a thirteenth factory is added, which is
the failure mode this repo has hit before (the `--universe` choices that drifted
from `UNIVERSE_CATEGORY_MAP`).

`tests/test_data_dir_cli.py` -- add the AST ordering guard. Parse each script with
`ast.parse`, find the module-level `ast.If` whose test compares `__name__` to
`"__main__"`, then walk it collecting every `ast.Call`'s resolved function name in
source order (sort by `(lineno, col_offset)`; resolve `ast.Name` and
`ast.Attribute` alike). Build the deny set from that module's own
`from config import (...)` names plus local `_build*` calls, then assert
`index(apply_data_dir) < min(index of any deny-set call)`. Derive the script list
by globbing the repo root for `*.py` containing `add_data_dir_arg` and assert it
equals the five expected filenames.

Write the failure messages to route a future reader to the DECISION, not just to
the assertion: the ordering message says the factories snapshot paths at
construction time so an override applied later silently does nothing (DDIR-04);
the derived-script-list message says a new entry point offering `--data-dir` must
be covered here rather than exempted.

Then verify the guard is not decorative. Mutate and confirm each mutation turns
the suite RED, then revert every mutation:
1. Move `apply_data_dir(args)` in one script to AFTER its first factory call ->
   the ordering test must fail.
2. Delete `add_data_dir_arg(parser)` from one script -> the derived-list test
   must fail.
3. Invert the precedence in `get_data_root()` so the env var beats the override ->
   the precedence matrix must fail.
4. Make `set_data_root(None)` a no-op instead of clearing -> a precedence or
   no-regression case must fail.
Record in the SUMMARY which mutations were run and that all four were caught. If
any mutation survives, the escaping lock is the bug -- strengthen the test rather
than noting it.

`tests/test_config_paths.py` also carries the LAST surviving "ONLY path knob"
claim, in `test_stock_config_subdir_and_store_name_redirect_under_the_same_root`'s
docstring (~line 157). Rewrite it to the D-01 precedence, keeping what that test
actually pins -- `subdir`/`store_name` select a location BENEATH the one root and
never a second root or a hardcoded volume. Same discipline as Task 2: state the
new truth, do not quote the retired phrase to correct it, or the repo-wide gate in
`<verification>` stays red.

`tests/test_factor_kunquant.py`: update the one docstring at line ~320 that names
the retired private helper as the source of the derived path, to name
`get_data_root()` instead (D-02). Prose only; the assertions are untouched, and
the retired name must not survive anywhere in the replacement sentence.
  </action>
  <verify>
    <automated>uv run pytest tests/ -q &amp;&amp; ! (grep -rni 'only path knob' --include='*.py' --include='*.md' . | grep -v '\.planning/') &amp;&amp; ! (grep -rnE '(^|[^A-Za-z0-9_])_data_root' --include='*.py' --include='*.md' . | grep -v '\.planning/')</automated>
  </verify>
  <done>The full suite passes with no fewer than the 505 tests that passed before this plan plus the new ones; the precedence matrix, the four-field reach, the by-construction root check, the no-regression pair, the AST ordering guard and the binance composition all pass; all four mutations were run, each turned the suite red, and each was reverted.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| argv -> filesystem root | A `--data-dir` value typed on the command line becomes the root every read and write in that run is derived from. |
| process-global state -> config factories | A module-level override read by 14 path expressions; whoever sets it last wins for the rest of the process. |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-rjq-01 | Tampering | `config.set_data_root` | low | accept | The value reaches `set_data_root` only from this process's own argv or an explicit programmatic call -- never from a network response, a fetched config, or a vendor payload. The operator typing the flag already holds the process's full filesystem rights, exactly as with `QUANTLAB_DATA_DIR`. No new authority is granted. |
| T-rjq-02 | Tampering | each script's `__main__` ordering | medium | mitigate | An override applied AFTER a factory constructs produces a run that REPORTS one root and writes to another -- silent misdirection, and the printed path would be the lie. Mitigated by the Task 3 AST ordering guard, whose script list is derived from a repo glob so a sixth entry point cannot opt out by omission, and which is mutation-verified. |
| T-rjq-03 | Tampering | `tests/conftest.py` global state | medium | mitigate | The process-global override leaking between tests would let one test redirect another's paths -- including onto a real data root. Mitigated by the autouse fixture clearing before AND after every test (Task 1), and by the mutation that makes `set_data_root(None)` a no-op being required to turn the suite red (Task 3). |
| T-rjq-04 | Information disclosure | `apply_data_dir` / script output | low | mitigate | `apply_data_dir` prints nothing itself. Paths are already on this repo's explicit safe-to-print list ("only symbol lists, date ranges and paths are ever printed" -- `ingest_tiingo.py`, `ingest_us_equity.py` docstrings), and no credential is read, joined into, or derived from a data-root path anywhere in this change. `AcquisitionConfig.to_dict()` gains no new field. |
| T-rjq-05 | Elevation of privilege | `set_data_root` validation | low | accept | No existence check and no `mkdir` (D-08): the acquisition layer creates its own directories under whatever root it is given, and the env knob validates nothing today. Adding a check to one knob and not the other is how two knobs drift apart. A nonexistent or unwritable root fails at the first write with the OS's own error, which names the path. |

**Supply chain:** this plan installs no packages -- no npm/pip/cargo install task exists, `pyproject.toml` gains no dependency, and every import used is already in the tree. No package-legitimacy gate applies.
</threat_model>

<verification>
- `uv run pytest tests/ -q` -- full suite green, at least 505 + the new tests.
- `for f in ingest_tiingo ingest_alpaca ingest_us_equity ingest_binance_spot refresh_us_equity_universe; do uv run python $f.py --help | grep -q -- '--data-dir'; done` -- the flag is on all five, and each script still imports cleanly with no credentials in the environment.
- `grep -rnE '(^|[^A-Za-z0-9_])_data_root' --include='*.py' --include='*.md' . | grep -v '\.planning/'` -- zero hits. The pattern is ANCHORED on purpose: a bare `_data_root` also matches `_market_data_root`, which survives the rename, so the naive gate could never go green.
- `grep -rni 'only path knob' --include='*.py' --include='*.md' . | grep -v '\.planning/'` -- zero hits.
- The four Task 3 mutations were each run, each turned the suite red, and each was reverted.
</verification>

<success_criteria>
- DDIR-01: `--data-dir /X` on any of the five entry points roots `raw_data_dir_path`, `zarr_file_path`, `watermark_path` and `output_path` under `/X`.
- DDIR-02: precedence is CLI > `QUANTLAB_DATA_DIR` > repo-root `data/`, proved in all four cells, with exactly one root and no hardcoded volume.
- DDIR-03: omitting `--data-dir` reproduces today's paths for env-var users and default users alike; the pre-existing 505 tests still pass.
- DDIR-04: `apply_data_dir` precedes every factory-reaching call in every `__main__`, enforced by a derived, mutation-verified AST guard.
- DDIR-05: `--raw-data-dir` still redirects only the raw CSV directory and composes with `--data-dir`, documented in the docstring, the help text and a test.
</success_criteria>

<output>
Create `.planning/quick/260907-rjq-add-a-data-dir-cli-parameter-to-the-data/260907-rjq-SUMMARY.md` when done.
</output>
</content>
</invoke>
