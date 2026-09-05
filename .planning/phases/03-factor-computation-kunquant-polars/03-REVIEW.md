---
phase: 03-factor-computation-kunquant-polars
reviewed: 2026-09-05T00:00:00Z
depth: standard
files_reviewed: 17
files_reviewed_list:
  - README.md
  - base/config.py
  - base/factor.py
  - base/factor_polars.py
  - config/__init__.py
  - dataset/stock.py
  - factor/alpha101.py
  - factor/alpha158.py
  - factor/momentum.py
  - my_ops/preprocess.py
  - tests/conftest.py
  - tests/test_extensibility_contract.py
  - tests/test_factor_hierarchy.py
  - tests/test_factor_kunquant.py
  - tests/test_factor_polars.py
  - tests/test_factor_stream.py
  - tests/test_stock_dataset.py
findings:
  critical: 2
  warning: 13
  info: 8
  total: 23
status: issues_found
---

# Phase 3: Code Review Report

**Reviewed:** 2026-09-05
**Depth:** standard
**Files Reviewed:** 17
**Status:** issues_found

## Summary

Phase 3 splits `FactorKunQuant` into a shared `Factor` base plus two sibling
backends, adds two US-equity KunQuant factor classes, a Polars backend with a
worked example, and a large test suite. The refactor itself is mechanically
sound — the `mode`-branching overrides, the `__init__` ordering hazard and the
`_auto_filter` batch guard are all preserved correctly, and the KunQuant
`decompose(self, options)` signature fix is real.

Two defects are shipping-blocking. First, the claimed "seamless
interchangeability" of the two backends holds only on the `cal()` path: a
`FactorPolars` loaded via `read()` never resolves `config.factor_names`, so
`base/model.py`'s `factor_data_strategy="read"` branch raises for Polars
factors and not for KunQuant ones — the exact coupling D-03/FACTOR-04 exist to
remove, and the one path the live interchangeability test never exercises.
Second, the D-02 `amount = volume * close` synthesis makes KunQuant's derived
`vwap` *identically equal to close*, so every VWAP-derived US-equity feature
collapses onto a close-price feature (`VWAP0` becomes a constant 1.0). The
factor values compute and look finite, which is exactly why nothing catches it.

Beyond those, the review found: a bar-count/calendar-day confusion in the
lookback extension that under-warms US-equity factors, two sources of truth for
the momentum horizon, a dead-and-broken `_get_lazyframe()` carried through the
refactor, ~160 lines of verbatim class duplication with no lock on the
"identical invocation" claim it rests on, and several tests whose assertions
ride on float32 rounding noise or on shared mutable config state rather than on
the property they claim to lock.

No security defects were found. Credential handling is correct
(`TIINGO_API_KEY` read from the environment with an explicit raise in both
`acquisition/tiingo.py` and `scripts/download_stock_data_from_tiingo.py`), no
secrets are hardcoded, and no injection/eval/deserialization surface exists in
the reviewed files.

## Critical Issues

### CR-01: Polars factors are not interchangeable on the `read()` path — `factor_data_strategy="read"` raises

**File:** `base/factor_polars.py:43-49`, `base/factor_polars.py:63-71`, `base/factor.py:126-129`

**Issue:** `base/factor_polars.py:45-46` states the precondition as *"only valid
AFTER `cal()` **or `read()`** has populated `config.factor_names`"*, and
`tests/test_factor_polars.py:103` repeats it. That is false: nothing populates
`factor_names` on the read path. `Factor.read()` (`base/factor.py:126-129`)
only calls `data_backend.read()` + `_auto_filter()`; the only assignment to
`config.factor_names` for a Polars factor is inside `FactorPolars.cal()`
(`base/factor_polars.py:110`), and `_maybe_resolve_factor_names()` is
deliberately a no-op (`base/factor_polars.py:55-61`).

Consequence, in `base/model.py`:

```python
# base/model.py:158-165 — a first-class, config-selectable branch
case "read":
    ds = factor.read().get_features()   # OK, returns data
...
# base/model.py:184-189 — called next, on the same factor
[factor._get_factor_names() for factor in self.config.factors]
# -> RuntimeError: Momentum: factor names are not resolved yet.
```

So a `DLConfig` with `factor_data_strategy="read"` works for every KunQuant
factor and raises for every Polars factor. That is precisely the
backend-dependent behaviour D-03 forbids and `README.md:34-40` claims is
impossible ("nothing downstream can tell which one produced a given factor
store"). It also means `BaseModel._save_model()` serializes
`"factor_names": null` into `config.json` for any Polars factor, so a
checkpoint cannot be reconstructed.

`tests/test_factor_hierarchy.py:430-522` does not catch this: it sets
`factor_data_strategy="cal"` and then hand-calls `factor.cal()` — the `read()`
half of the model's call surface is never driven for either backend.

**Fix:** resolve the names from the persisted store on the read path. The
factor store's non-index data variables *are* the factor names, symmetrically
with `cal()`:

```python
# base/factor_polars.py
def read(self) -> Self:
    super().read()
    if self.config.factor_names is None:
        self.config.factor_names = tuple(
            self._get_xarray_dataset().data_vars
        )
    return self
```

Then add a regression test that drives one KunQuant and one Polars factor
through `save()` -> fresh instance -> `read()` -> `_get_factor_names()`, and
correct the docstring/test comment if the `read()` claim is dropped instead.

---

### CR-02: `amount = volume * close` makes `vwap` identically equal to `close`, collapsing every VWAP-derived US-equity factor

**File:** `dataset/stock.py:73-74`

**Issue:** The D-02 synthesis is

```python
if "amount" in data_columns and "amount" not in data.data_vars:
    data = data.assign(amount=data["volume"] * data["close"])
```

KunQuant's `Alpha101.AllData` / `Alpha158.AllData` derive `vwap = amount /
volume` (this is stated in the phase's own docstrings — `config/__init__.py:169-172`
notes `AllData` "derives `vwap` from it"). Substituting the proxy:

```
vwap = amount / volume = (volume * close) / volume = close
```

`vwap` is therefore **exactly** the close series for every US-equity symbol and
timestamp, not an approximation. Downstream:

- `Alpha158Stock` builds the price block with `("VWAP", all_data.vwap)` over
  windows `[0,1,2,3,4]` (`factor/alpha158.py:161-172`). Qlib's `VWAPd` is
  `Ref(vwap, d) / close`, so `VWAP0 ≡ 1.0` — a zero-variance feature fed to the
  model — and `VWAP1..VWAP4` are bit-for-bit duplicates of `CLOSE1..CLOSE4`.
  Five of the emitted features carry zero incremental information.
- Every `Alpha101Stock` alpha that references `vwap` (alpha025, alpha028,
  alpha041, alpha050, alpha083, …) silently degenerates into a close-price
  variant of itself.

Nothing fails loudly: the arrays are finite, the dataset shape is right, and
`tests/test_factor_kunquant.py:222-247` asserts only on `KMID`/`VOLUME0`/`STD5`,
never on a VWAP feature. This is silent corruption of the US-equity feature set,
not a cosmetic approximation, and it is worse than the pre-fix crash it
replaced because the crash was visible.

**Fix:** use a proxy whose implied VWAP is not the close price — the standard
typical-price dollar volume — and make the degeneracy impossible to reintroduce
silently:

```python
# dataset/stock.py:_to_kunquant
if "amount" in data_columns and "amount" not in data.data_vars:
    typical = (data["high"] + data["low"] + data["close"]) / 3.0
    data = data.assign(amount=data["volume"] * typical)
```

Add a test asserting `input_dict["amount"] / input_dict["volume"]` is **not**
allclose to `input_dict["close"]`, and a test that `Alpha158Stock`'s `VWAP0`
output is not constant. If a real vendor dollar-volume column is obtainable
instead, prefer it and keep the proxy as the documented fallback.

## Warnings

### WR-01: `num_factors` / `get_factor_names()` bypass the guard and raise `TypeError`, not the documented `RuntimeError`

**File:** `base/factor.py:111-112`, `base/factor.py:161-162`, `base/factor_polars.py:63-71`

**Issue:** `get_factor_names()` returns `self.config.factor_names` directly; it
never calls `_get_factor_names()`. So `FactorPolars`'s carefully worded
`RuntimeError` (`base/factor_polars.py:65-70`) is unreachable from the public
surface: `num_factors` on an unresolved Polars factor evaluates
`len(None)` -> `TypeError: object of type 'NoneType' has no len()`, and
`get_factor_names()` silently returns `None` instead of raising.

`base/factor_polars.py:45` explicitly claims the guard covers "`_get_factor_names()`
-- and therefore `num_factors`". It does not.
`tests/test_factor_polars.py:110-111` asserts on the private
`factor._get_factor_names()`, which is the only caller that reaches the guard,
so the test gives false confidence about the public path.

**Fix:** route the public accessor through the hook so both backends share one
resolution path:

```python
# base/factor.py
def get_factor_names(self) -> tuple[str, ...]:
    if self.config.factor_names is None:
        return self._get_factor_names()
    return tuple(self.config.factor_names)
```

Then extend the test to `with pytest.raises(RuntimeError): factor.num_factors`.

### WR-02: lookback is extended in calendar days but every factor window is in bars

**File:** `base/factor.py:95-100`, `config/__init__.py:154-195`, `config/__init__.py:225-268`, `config/__init__.py:271-305`

**Issue:**

```python
start_date = start_date - pd.DateOffset(days=self._config.window)
```

`window` is a bar count everywhere it is consumed (`WindowedZScore(..., window)`,
Alpha158 rolling windows, `Momentum`'s `shift(n)`), but the lookback it buys is
measured in **calendar days**. That identity holds only for continuously-traded
crypto at `frequency="1d"` — the market this code was written against.

The new configs break the assumption:

- `stock_alpha101_config` / `stock_alpha158_config` default `market="us_equity"`,
  `frequency="1d"`: `window=128` calendar days buys ~88 trading bars, a ~31%
  shortfall against the declared window.
- `momentum_config` exposes `market` as a parameter
  (`config/__init__.py:277`), so `momentum_config(n=20, market="us_equity")`
  buys ~14 bars for a 20-bar shift — the first ~6 timestamps of the requested
  range come back as silent NaN rather than an error.
- `frequency` is likewise a parameter and accepts `"1m"`/`"tick"`
  (`enums/data.py`), where the mismatch is orders of magnitude.

Failure mode is silent NaN/under-warmed factor values at the head of every
requested range, not an exception.

**Fix:** derive the offset from the dataset's own bar interval rather than
assuming one bar per day — `base/data.py:44-51` already exposes
`Dataset.time_interval` for exactly this:

```python
def _reset_dataset_config(self):
    start = pd.to_datetime(self._config.start_date)
    try:
        bar = pd.Timedelta(self._config.dataset.time_interval)
    except Exception:
        bar = pd.Timedelta(days=1)
    # calendar-day padding for non-trading days on a session-based market
    start = start - bar * self._config.window * 1.5
```

At minimum, document the assumption on `BaseFactorConfig.window` and assert
that the materialized panel has at least `window` bars before the requested
`start_date`, so an under-warmed run fails instead of emitting NaN.

### WR-03: the momentum horizon has two unvalidated sources of truth

**File:** `factor/momentum.py:44-47`, `config/__init__.py:293-305`

**Issue:** `momentum_config()` writes the horizon twice — into `window` (which
drives the lookback) and into `kwargs["n"]` (which drives the actual shift) —
and `Momentum.horizon` reads only the latter, falling back to
`_DEFAULT_HORIZON = 20` when the key is absent. Nothing checks they agree. A
hand-built `PolarsFactorConfig(window=5, kwargs={"n": 60})` (or a config with
`window=60` and no `kwargs`, which silently becomes `n=20`) computes a horizon
the lookback does not cover, producing NaN-heavy output with no diagnostic.

**Fix:** make `window` the single source of truth and delete the duplicate, or
validate in `Momentum.__init__`:

```python
def __init__(self, factor_config: PolarsFactorConfig):
    super().__init__(factor_config)
    if self.horizon > self.config.window:
        raise ValueError(
            f"{self.class_name}: horizon n={self.horizon} exceeds the "
            f"configured lookback window={self.config.window}"
        )
```

### WR-04: every momentum horizon writes to the same `momentum.zarr`

**File:** `config/__init__.py:294`

**Issue:** `file_path` is a fixed `.../factor/momentum.zarr` regardless of `n`,
even though the emitted column is horizon-dependent (`momentum_{n}`). Computing
`momentum_config(n=5)` and then `momentum_config(n=20)` targets one store;
`Factor.save()` defaults to `mode="a"` (`base/factor.py:131`), so the two runs
either collide or interleave, and a subsequent `read()` picks up whichever
horizon last landed while the config still says something else. The KunQuant
factories get away with a fixed path because their factor set is fixed; this
one is parameterized.

**Fix:** derive the path from the horizon, mirroring the parameterization:

```python
file_path=str(_data_root() / "data" / "factor" / f"momentum_{n}.zarr"),
```

### WR-05: `Factor._get_lazyframe()` is dead code and is broken as written

**File:** `base/factor.py:141-144`

**Issue:** Zero callers repo-wide (the Polars backend correctly uses
`Dataset.get_lazyframe()` at `base/factor_polars.py:104`). It is also
non-functional: `xr.Dataset.to_pandas()` only supports datasets with one or
fewer dimensions, and the canonical factor dataset is 2-D `[timestamp, symbol]`,
so the first line raises `ValueError` on any real input. It survived the 03-02
hoist into the shared base and is now enshrined by
`tests/test_factor_hierarchy.py:394` as a legitimate private helper, which will
mislead the next reader into calling it. It is also the only reason
`base/factor.py` imports `polars`.

**Fix:** delete the method, the `import polars as pl` on `base/factor.py:9`, and
the reference in the test docstring.

### WR-06: `FactorPolars.cal()` does not enforce the output contract it documents

**File:** `base/factor_polars.py:103-127`

**Issue:** The hook contract (`base/factor_polars.py:84-88`) says the returned
frame must carry *only* `timestamp`, `symbol` and factor columns, and warns that
"a surviving raw price or volume column would be persisted as if it were a
factor". `cal()` implements the naming rule (`everything not in _INDEX_COLUMNS`
is a factor) but never validates it. A subclass that forgets the final
`select(...)` silently persists `Close`/`Volume`/`anomaly_flag` into the factor
store as factors, and they then flow into `DLConfig.factors` as model features.
A subclass that omits `timestamp` or `symbol` fails later with an opaque pandas
`KeyError` from `set_index`, not a contract error.

**Fix:** enforce both halves at the boundary:

```python
names = factor_lf.collect_schema().names()
missing = [c for c in _INDEX_COLUMNS if c not in names]
if missing:
    raise ValueError(
        f"{self.class_name}._get_factor_lazyframe() must return "
        f"{_INDEX_COLUMNS}; missing {missing}"
    )
factor_names = tuple(n for n in names if n not in _INDEX_COLUMNS)
if not factor_names:
    raise ValueError(f"{self.class_name} produced no factor columns")
```

### WR-07: ~160 lines duplicated verbatim across the four factor classes, with nothing locking the "identical invocation" claim

**File:** `factor/alpha158.py:107-211`, `factor/alpha101.py:73-137`

**Issue:** `Alpha158Stock` is a character-for-character copy of
`Alpha158SpotKline` except for one `Output(...)` line; `Alpha101Stock` likewise.
The 20-line `all_data.build({...})` category dict — the actual definition of the
factor set — now exists in two places, and the Alpha101 name enumeration in two
more. The docstring (`factor/alpha158.py:128-137`) defends this on the grounds
that "divergence in one class only would silently break D-01's
identical-invocation guarantee" — but nothing enforces that guarantee. The only
lock,
`tests/test_factor_kunquant.py:291-314`, greps for the string `"WindowedZScore"`
in each class's emitting method; it would pass unchanged if someone edited
`Alpha158Stock`'s `windows` list from `[5,10,20,30,60]` to `[5,10]`, silently
giving the two markets different factor sets under the same name.

**Fix:** collapse to one class per factor family with a single-line
normalization seam, which keeps D-09's per-market split explicit while removing
the duplication:

```python
class Alpha158Base(FactorKunQuant):
    def _normalize(self, op):          # crypto: time-series z-score
        return WindowedZScore(op, self.config.window)

class Alpha158Stock(Alpha158Base):
    def _normalize(self, op):          # US equities: raw (D-09)
        return op
```

If the duplication is kept deliberately, add a test asserting
`inspect.getsource(Alpha158Stock._get_func_names) ==
inspect.getsource(Alpha158SpotKline._get_func_names)` so the claimed invariant
is actually locked.

### WR-08: `WindowedZScore` divides by the rolling std with no zero guard

**File:** `my_ops/preprocess.py:12-25`

**Issue:** `Div(diff, rolling_std)` has no epsilon, while its sibling
`WindowedRobustStandardization` explicitly adds `1e-8`
(`my_ops/preprocess.py:53-54`) for exactly this reason. A symbol whose input is
constant across the window — a trading halt, a stale feed, a stablecoin pair, a
zero-volume stretch — yields `0/0 = NaN` or `x/0 = ±inf` in every crypto-spot
Alpha101/Alpha158 factor for that symbol/window. Since this op wraps *every*
`Output(...)` on both crypto classes, one degenerate symbol poisons the whole
feature row.

**Fix:** mirror the robust op's guard:

```python
std_safe = Add(rolling_std, ConstantOp(1e-8))
z_score = Div(diff, std_safe)
```

### WR-09: the synthetic fixtures make `KMID` mathematically constant, so two "real values"/"incremental update" assertions measure float32 rounding noise

**File:** `tests/conftest.py:380-387`, `tests/conftest.py:514-525`, `tests/test_factor_kunquant.py:88`, `tests/test_factor_stream.py:171-176`

**Issue:** Both Zarr fixtures build OHLC as exact scalar multiples of one series:

```python
"Open": base * 0.99, "High": base * 1.02, "Low": base * 0.98, "Close": base,
```

Alpha158's `KMID = (close - open) / open` is therefore the constant
`1/0.99 - 1 ≈ 0.0101` at every timestamp and every symbol, by construction.
Consequences:

- `tests/test_factor_kunquant.py:88` — `np.isfinite(result["KMID"]).sum() > 0`
  on `Alpha158SpotKline`, where `KMID` is z-scored: the numerator and the rolling
  std are both zero in exact arithmetic, so whether this passes depends entirely
  on float32 rounding in `base * 0.99`. The test is one dtype change away from
  flipping to all-NaN.
- `tests/test_factor_stream.py:171-176` — the load-bearing assertion
  `not np.array_equal(previous_kmid, last_kmid)`, whose stated purpose is to
  prove "the stream is not returning a stale or constant buffer", is comparing
  two snapshots of a signal that is constant by design. It proves rounding noise
  differs, not that the stream advanced.

The same construction makes every K-bar feature (`KLEN`, `KUP`, `KLOW`, `KSFT`)
degenerate, and makes `Alpha158Stock`'s un-normalized `KMID` a trivially finite
constant so `tests/test_factor_kunquant.py:247` asserts nothing.

**Fix:** give each OHLC series independent noise so the K-bar family varies:

```python
o = base * (0.99 + rng.normal(0, 0.004, base.shape))
c = base
h = np.maximum(o, c) * (1.0 + np.abs(rng.normal(0, 0.004, base.shape)))
l = np.minimum(o, c) * (1.0 - np.abs(rng.normal(0, 0.004, base.shape)))
```

and strengthen the stream assertion to compare a factor with genuine
time-variation (e.g. `STD5`) in addition to `KMID`.

### WR-10: the interchangeability test shares one mutable `DatasetConfig` between two `Dataset` instances

**File:** `tests/test_factor_hierarchy.py:458-478`

**Issue:** `SpotKlineDataset(dataset_config)` is constructed twice from the
*same* `DatasetConfig` object, and `Dataset.config`'s setter stores the
reference without copying (`base/data.py:70`). `Factor._reset_dataset_config()`
then mutates `self._config.dataset.config.start_date` — so both factors write to
one shared object, and the second factor's lookback silently overwrites the
first's. The test only passes cleanly because both configs use `window=10`.

This directly undermines what the test claims to prove: if the two backends
computed *different* lookbacks (which WR-02 makes plausible the moment markets
or frequencies differ), the aliasing would hide the divergence rather than
expose it. It is also a live production hazard for any caller that reuses one
`DatasetConfig` across two factors.

**Fix:** give each factor its own dataset config in the test
(`dataclasses.replace(dataset_config)` or two `spot_kline_zarr()` calls), and
consider defensively copying in `Dataset.config`'s setter so config objects are
not silently shared across instances.

### WR-11: `XrBackend.read()` caching plus monotonic filtering makes a second `cal()` silently return truncated data

**File:** `base/factor_polars.py:104`, `base/factor.py:252-255`, `dataset/backend.py:15-22`

**Issue:** `XrBackend.read()` returns early whenever `self.data` already exists,
and `Dataset._filter()` only ever narrows the in-memory data. The model layer's
documented flow is `_reset_factors_config()` (mutates the factor's date range)
followed by `cal()`. If the second range is *wider* than the first — e.g. a CV
fold, or the predict-over-a-longer-range flow in `train_model.py` — the dataset
is never re-read from Zarr and the already-narrowed data is filtered again. The
factor is computed over a silently truncated panel with no warning.

This is pre-existing on the KunQuant path but is newly inherited by
`FactorPolars.cal()` (`base/factor_polars.py:104`), which calls
`dataset.read()` on every invocation and therefore looks like it re-reads.

**Fix:** make `read()` honour the current config rather than short-circuiting on
presence of data — e.g. cache on `(path, start_date, end_date, symbols)` and
re-read when the key changes, or have `Dataset.read()` pass `overwrite=True`
whenever its config dates changed since the last read.

### WR-12: six unused duplicate `Input(...)` nodes are registered into every Alpha158 graph

**File:** `factor/alpha158.py:84-90`, `factor/alpha158.py:191-197`

**Issue:** `_get_func_stream()` declares `close/low/high/vopen/amount/vol`
inside the `with builder:` block and never uses any of them — the operator graph
is built by `self._get_func_names()` on the next line, which constructs its own
six `Input(...)` nodes. The result is twelve `Input` nodes for six logical
inputs registered into the same builder. They survive today only because
KunQuant prunes unreachable inputs, which is the same pruning that produces the
`RuntimeError: Cannot find the buffer name` trap documented at `README.md:66-69`
and `tests/test_factor_stream.py:32-42`. Carrying dead graph nodes into a
compiled artefact while simultaneously documenting a pruning footgun is a
maintenance liability, and 03-03 duplicated it into `Alpha158Stock` rather than
removing it.

**Fix:** delete lines 84-90 and 191-197; `_get_func_names()` already builds
everything the graph needs.

### WR-13: `FactorKunQuant.cal()` defeats its own compile cache

**File:** `base/factor.py:261-269`

**Issue:**

```python
if self._lib is None:
    self._lib = self._make()
modu = self._lib.getModule(...)
...
self._lib = None        # line 269
```

`_lib` is cleared at the end of every call, so the `if self._lib is None`
guard is always true and the graph is recompiled from scratch on every `cal()`
— the cache it implies never engages. Either the guard or the reset is wrong;
as written the code reads as if compilation is cached when it is not, which will
mislead anyone profiling a multi-factor `collect()`.

**Fix:** decide the intent explicitly. To cache, drop line 269. To release the
library deliberately, drop the `if` and add a comment explaining why the
compiled artefact must not be retained across calls.

## Info

### IN-01: unused imports in `dataset/stock.py`

**File:** `dataset/stock.py:1-14`
**Issue:** `Decimal`, `Path`, `Parallel`, `delayed`, `BinanceCSVHeaders` and
`file_date_filter` each appear exactly once — on their import line. They are
leftovers from the `SpotKlineDataset` template this class was copied from.
**Fix:** remove them; keep `numpy`, `polars`, `xarray`, `tqdm`, `get_pqt_files`,
`dedup_raw_frame`, `Timer`, `Dataset`, `DatasetConfig`.

### IN-02: `__class__.__name__` instead of `self.class_name` in `_get_labels`

**File:** `factor/momentum.py:70`, `factor/alpha101.py:67`, `factor/alpha101.py:134`, `factor/alpha158.py:101`, `factor/alpha158.py:208`
**Issue:** `f"{__class__.__name__} does not support get_label()"` resolves via
the implicit class cell to the *defining* class, so a subclass reports its
parent's name in the error. `Factor.class_name` (`base/factor.py:122-124`)
already exists for this.
**Fix:** `raise RuntimeError(f"{self.class_name} does not support get_labels()")`
— and note the message names a method (`get_label()`) that does not exist; the
public accessor is `get_labels()`.

### IN-03: `stock_alpha158_config` hardcodes `window=128` while its Alpha101 sibling parameterizes it

**File:** `config/__init__.py:225-268` vs `config/__init__.py:154-195`
**Issue:** The two factories are documented as "identical in shape", but
`stock_alpha101_config` exposes `window: int = 128` and `stock_alpha158_config`
does not, so the Alpha158 lookback cannot be tuned without editing the factory.
(The crypto pair has the same asymmetry, which was copied.)
**Fix:** add `window: int = 128` to the signature and pass it through.

### IN-04: default data root produces a doubled `data/data/...` path

**File:** `config/__init__.py:18-40`
**Issue:** `_data_root()` returns `<repo>/data`, and `_market_data_root()` /
every factor `file_path` then append another `"data"` segment, so the default
store lands at `<repo>/data/data/crypto_spot/1d/klines.zarr` while raw downloads
land at `<repo>/data/downloads/...`. Confusing and undocumented in
`README.md:171-173`.
**Fix:** drop the redundant `"data"` segment, or rename `_data_root()` to
`_workspace_root()` and document the layout in the README.

### IN-05: README's config-factory list is stale

**File:** `README.md:191-194`
**Issue:** Lists only `spot_kline_config`, `alpha101_config`, `alpha158_config`,
`spot_label_config`; `config/__init__.py` now also exports
`stock_kline_config`, `stock_acquisition_config`, `universe_config`,
`stock_alpha101_config`, `stock_alpha158_config` and `momentum_config` — the
last three delivered by this phase and documented nowhere in the README.
**Fix:** update the list, or point at the module instead of enumerating.

### IN-06: `factor_names` is typed `list` but assigned tuples throughout

**File:** `base/config.py:78`, `base/factor.py:93`, `base/factor_polars.py:110`, `base/factor.py:161`
**Issue:** `BaseFactorConfig.factor_names: list | None`, but every writer
assigns a `tuple` and `get_factor_names()` is annotated `-> tuple[str, ...]`
while returning whatever the config holds (possibly a caller-supplied `list`, or
`None`). The declared and actual types never agree.
**Fix:** type the field `tuple[str, ...] | None` and normalize on assignment in
`_maybe_resolve_factor_names()`.

### IN-07: a typo in `factor_names` silently produces an empty factor graph

**File:** `factor/alpha101.py:54-59`, `factor/alpha101.py:121-126`, `factor/alpha158.py:92-94`, `factor/alpha158.py:199-201`
**Issue:** The emitting loops filter with `if alpha.__name__ in factor_names`
and never check that every requested name matched. `factor_names=["alpha01"]`
(one digit short) compiles a graph with zero `Output(...)` nodes rather than
raising.
**Fix:** validate before building —
`unknown = set(factor_names) - set(self._get_factor_names())` and raise if
non-empty.

### IN-08: the normalization-matrix lock is a source-text grep

**File:** `tests/test_factor_kunquant.py:304-314`
**Issue:** `"WindowedZScore" in inspect.getsource(...)` flips on a comment or a
renamed import alias, and cannot see a z-score applied through an indirection.
Given how much weight the phase puts on D-09, a behavioural assertion would be
stronger.
**Fix:** additionally assert on values — e.g. that a `Alpha158SpotKline` output
column has near-zero rolling mean while the `Alpha158Stock` equivalent does not.

---

_Reviewed: 2026-09-05_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
