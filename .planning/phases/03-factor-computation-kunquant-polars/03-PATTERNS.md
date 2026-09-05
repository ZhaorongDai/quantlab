# Phase 3: Factor Computation (KunQuant + Polars) — Pattern Map

**Mapped:** 2026-09-05
**Scope:** Files planned for creation/modification in Phase 3, their roles, closest existing analogs, and the EXACT current source of every method the refactor touches.

> **Why the verbatim source dumps below:** this is a refactor-heavy phase. The planner must write surgical before/after diffs against `base/factor.py`, `base/config.py`, and `dataset/stock.py`. Every code block in the "Exact Current Source" sections is a byte-accurate copy of the file on disk as of 2026-09-05 (with the original line numbers noted) — do not re-derive current state from memory or from RESEARCH.md's paraphrases.

---

## 1. File Manifest (verified against 03-CONTEXT.md + 03-RESEARCH.md)

| # | File | Action | Role | Data flow | Closest existing analog |
|---|------|--------|------|-----------|-------------------------|
| 1 | `base/factor.py` | **Modify (refactor)** | Abstract factor-layer contract | in: `Dataset` → out: `xr.Dataset` via `XrBackend` | `base/data.py:Dataset` (the sibling ABC that already has the same `config`-property / `read()` / `save()` / `_filter()` lifecycle shape) |
| 2 | `base/factor_polars.py` | **Create** (planner's call; may instead go in `base/factor.py`) | New `FactorPolars(Factor)` ABC | in: `Dataset.get_lazyframe()` → `pl.LazyFrame` → out: `xr.Dataset` | `base/factor.py:FactorKunQuant` (structure), `dataset/backend.py:PlBackend` (Polars idioms) |
| 3 | `base/config.py` | **Modify** | Dataclass config bag | pure in-process | `DatasetConfig` / `AcquisitionConfig` / `UniverseConfig` in the same file |
| 4 | `factor/alpha158.py` | **Modify** (add `Alpha158Stock`) | Concrete KunQuant factor set, US-equity market | in: `StockDataset` → out: `xr.Dataset` | `factor/alpha158.py:Alpha158SpotKline` (byte-for-byte structural template), `factor/alpha101.py:Alpha101Stock` (market-pairing pattern) |
| 5 | `factor/alpha101.py` | **Modify** (bugfix) | Concrete KunQuant factor set | same | `Alpha101SpotKline` in the same file — the working sibling `Alpha101Stock` must be made to match |
| 6 | `dataset/stock.py` | **Modify** (`_to_kunquant`) | Dataset→KunQuant input adapter | in: `xr.Dataset` → out: `dict[str, np.ndarray]` | `dataset/spot.py:SpotKlineDataset._to_kunquant()` (the version that *does* supply `amount`) |
| 7 | `factor/momentum.py` | **Create** | Example Polars factor (D-08) | in: `pl.LazyFrame` → out: `pl.LazyFrame` (factor cols only) | `factor/alpha101.py:Alpha101SpotKline` (class shape), `dataset/cleaning.py` (Polars-internal / xarray-boundary precedent) |
| 8 | `config/__init__.py` | **Modify** | Config factory functions | pure in-process | `alpha101_config()` / `alpha158_config()` in the same file |
| 9 | `base/model.py` | **Unchanged consumer** | Must keep working with zero edits | consumes `list[Factor]` | — (this is the interchangeability proof, see §7) |
| 10 | `tests/test_factor_*.py` | **Create** | Wave-0 test gaps | — | `tests/test_stock_dataset.py`, `tests/test_extensibility_contract.py` |
| 11 | `utils/module.py` | **Explicitly NOT modified** | Checkpoint reload | — | Deferred to Phase 4 (RESEARCH Open Question #1) |

---

## 2. Exact Current Source — `base/factor.py` (282 lines, whole file)

Class declaration and constructor (lines 21–30):

```python
class FactorKunQuant(ABC):
    def __init__(self, config: FactorConfig):
        self.config = config
        self.data_backend = XrBackend()
        self._stream_context: kr.StreamContext = None
        self._lib = None
        self._buffer_name_to_id = dict()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(config={self.config})"
```

> Note the ordering hazard: `self.config = config` runs the **setter** (line 48), which calls `self._get_factor_names()` and `self._reset_dataset_config()`, **before** `self.data_backend` exists. Any hoisted `Factor.__init__` must preserve this exact order, or must be careful that no hoisted setter path touches `self.data_backend`. (Currently none does — verified.)

Imports (lines 1–18):

```python
from abc import ABC, abstractmethod

# from prefect import task, flow
from typing import Literal, Self

import KunQuant.runner.KunRunner as kr
import numpy as np
import pandas as pd
import polars as pl
import xarray as xr
from KunQuant.Driver import KunCompilerConfig
from KunQuant.jit import cfake
from KunQuant.Stage import Function

from base.config import FactorConfig
from dataset.backend import XrBackend
from enums.constant import Date
from utils.timer import Timer
```

### 2a. Methods classified "(a) generic → hoist to `Factor` as-is"

```python
    # lines 131-134
    def read(self) -> Self:
        self.data_backend.read(self.config.file_path)
        self._auto_filter()
        return self

    # lines 136-144
    def save(self, mode: Literal["a", "w"] = "a", **kwargs) -> Self:
        with Timer(f"{self.__class__.__name__}: save"):
            self._auto_filter()
            self.data_backend.write(
                self.config.file_path,
                mode=mode,
                **kwargs,
            )
            return self

    # lines 146-149
    def _get_lazyframe(self) -> pl.LazyFrame:
        df = self.data_backend.get_xarray_dataset().to_pandas()  # type: ignore
        df = pl.LazyFrame(df.reset_index())
        return df

    # lines 151-152
    def _get_xarray_dataset(self) -> xr.Dataset:
        return self.data_backend.get_xarray_dataset()  # type: ignore

    # lines 171-184
    def _get_features(self, data: xr.Dataset) -> xr.Dataset:
        raise NotImplementedError

    def get_features(self) -> xr.Dataset:
        return self._get_features(self._get_xarray_dataset())

    def _get_labels(self, data: xr.Dataset) -> xr.Dataset:
        raise NotImplementedError

    def get_labels(self) -> xr.Dataset:
        return self._get_labels(self._get_xarray_dataset())

    def get_factor_names(self) -> tuple[str, ...]:
        return self.config.factor_names

    # lines 186-190
    def get_config(self) -> dict:
        ds_config = self.config.dataset.get_config()
        cfg = self.config.to_dict()
        cfg["dataset"] = ds_config  # type: ignore
        return cfg  # type: ignore

    # lines 72-81
    def _reset_dataset_config(self):
        # 时间
        start_date = pd.to_datetime(self._config.start_date)
        start_date = start_date - pd.DateOffset(days=self._config.window)
        self._config.dataset.config.start_date = start_date.strftime("%Y-%m-%d")
        self._config.dataset.config.end_date = self._config.end_date

        # symbol
        # 可以不reset, 因为xarray缺失的数据设为null 但是.symbol还是存在
        # self._config.dataset._reset_symbols()

    # lines 92-98, 109-111
    @property
    def num_factors(self) -> int:
        return len(self.get_factor_names())

    @property
    def import_path(self) -> str:
        return f"{self.__class__.__module__}.{self.__class__.__qualname__}"

    @property
    def class_name(self) -> str:
        return self.__class__.__name__
```

### 2b. Methods classified "(c) hoist WITH a `FactorKunQuant` override"

`config` property + setter (lines 44–70) — **note the Chinese docstring and inline comments must be preserved verbatim when hoisted** (see §8 conventions):

```python
    @property
    def config(self) -> FactorConfig:
        return self._config

    @config.setter
    def config(self, config: FactorConfig):
        """设置因子配置文件 使用因子配置覆盖数据集配置

        Args:
            config (FactorConfig): 配置类
        """
        self._config = config
        self._config.name = self.import_path

        # 初始化时间
        if self._config.start_date is None:
            self._config.start_date = Date.START_DATE
        if self._config.end_date is None:
            self._config.end_date = Date.END_DATE

        # 初始化因子名
        if self._config.factor_names is None:
            self._config.factor_names = self._get_factor_names()

        # 重置数据集配置
        # batch模式下, 数据集实例化时会初始化数据集文件(若不存在), 存在则会读取
        self._reset_dataset_config()
```

The three lines `if self._config.factor_names is None: ... = self._get_factor_names()` are the ones RESEARCH.md's "timing split" replaces with a `self._maybe_resolve_factor_names()` call.

`_auto_filter()` (lines 32–42) — **currently wraps its entire body in `if self.config.mode == "batch":`**, i.e. it is a silent no-op in stream mode:

```python
    def _auto_filter(self):
        if self.config.mode == "batch":
            self.data_backend.filter_by_date(
                col="timestamp",
                start_date=self.config.start_date,
                end_date=self.config.end_date,
            )
            if self.config.symbols is not None:
                self.data_backend.filter_by_symbol(
                    "symbol", self.config.symbols
                )
```

`num_symbols` / `symbols` (lines 83–107) — both branch on `self.config.mode`:

```python
    @property
    def num_symbols(self) -> int:
        if self.config.mode == "batch":
            return self.config.dataset.num_symbols
        elif self.config.mode == "stream":
            return len(self.config.dataset.config.symbols)
        else:
            raise ValueError(f"mode {self.config.mode} is not supported")

    @property
    def symbols(self) -> list[str]:
        if self.config.mode == "batch":
            return self.config.dataset.symbols
        elif self.config.mode == "stream":
            return list(self.config.dataset.config.symbols)
        else:
            raise ValueError(f"mode {self.config.mode} is not supported")
```

> **Refactor hazard:** `mode` is the field that moves to `FactorConfig` (KunQuant-only) and does NOT exist on `PolarsFactorConfig`. Therefore all three of `_auto_filter`/`num_symbols`/`symbols` **must** be split (generic body on `Factor`, mode-branching override on `FactorKunQuant`) — a naive hoist of the current bodies onto `Factor` would make `FactorPolars` raise `AttributeError: 'PolarsFactorConfig' object has no attribute 'mode'`.

### 2c. Methods classified "(b) KunQuant-only → stay on `FactorKunQuant`"

```python
    # lines 113-129
    def init_stream(self) -> Self:
        with Timer(f"{self.__class__.__name__}: init stream"):
            lib = self._make_stream()
            modu = lib.getModule(f"{self.__class__.__name__}_stream")  # type: ignore

            executor = kr.createMultiThreadExecutor(self.config.njobs)
            stream = kr.StreamContext(executor, modu, self.num_symbols)

            buffer_name_to_id = {}
            for name in self.config.data_columns:
                buffer_name_to_id[name] = stream.queryBufferHandle(name)
            for name in self.config.factor_names:
                buffer_name_to_id[name] = stream.queryBufferHandle(name)

            self._stream_context = stream
            self._buffer_name_to_id = buffer_name_to_id
            return self

    # lines 154-169
    def _to_xarray_dataset(
        self,
        raw_factor: dict[str, np.ndarray],
        timestamps: np.ndarray,
        symbols: np.ndarray,
    ):
        ds = xr.Dataset(
            {k: (["timestamp", "symbol"], v) for k, v in raw_factor.items()},
            coords={
                "timestamp": timestamps,
                "symbol": symbols,
            },
        )
        self.data_backend.to_internal(ds)
        self._auto_filter()
        return self

    # lines 192-196
    @abstractmethod
    def _get_factor_func(self) -> Function: ...

    @abstractmethod
    def _get_factor_names(self) -> tuple[str, ...]: ...

    # lines 198-219
    def cal(self) -> Self:
        input_dict, symbols, timestamp = self.config.dataset.to_kunquant(
            data_columns=self.config.data_columns
        )
        # 随便拿一个确定时间
        # [time, stocks]
        num_time = next(iter(input_dict.values())).shape[0]

        if self._lib is None:
            self._lib = self._make()

        modu = self._lib.getModule(f"{self.__class__.__name__}")  # type: ignore

        executor = kr.createMultiThreadExecutor(self.config.njobs)
        with Timer(f" {self.__class__.__name__}: cal"):
            out_dict = kr.runGraph(executor, modu, input_dict, 0, num_time)

        self._lib = None

        self._to_xarray_dataset(out_dict, timestamp, symbols)

        return self

    # lines 221-245
    def cal_stream(
        self, data: dict[str, np.ndarray], timestamp: int, symbols: list[str]
    ) -> Self:
        if self._stream_context is None:
            self.init_stream()

        for name in self.config.data_columns:
            self._stream_context.pushData(
                self._buffer_name_to_id[name], data[name]
            )

        self._stream_context.run()

        out_dict = {}
        for factor in self.config.factor_names:
            alpha = self._stream_context.getCurrentBuffer(
                self._buffer_name_to_id[factor]
            )[:]
            out_dict[factor] = np.expand_dims(alpha, axis=0)

        self._to_xarray_dataset(
            out_dict, np.array([timestamp]), np.array(symbols)
        )

        return self

    # lines 247-282
    def _make(self):
        with Timer(f" {self.__class__.__name__}: make"):
            return cfake.compileit(
                [
                    (
                        f"{self.__class__.__name__}",
                        self._get_factor_func(),
                        KunCompilerConfig(
                            input_layout="TS",
                            output_layout="TS",
                        ),
                    )
                ],
                f"{self.__class__.__name__}",
                cfake.CppCompilerConfig(),
            )

    def _make_stream(self):
        with Timer(f"{self.__class__.__name__}: make stream"):
            return cfake.compileit(
                [
                    (
                        f"{self.__class__.__name__}_stream",
                        self._get_factor_func(),
                        KunCompilerConfig(
                            blocking_len=8,
                            partition_factor=8,
                            input_layout="STREAM",
                            output_layout="STREAM",
                            options={"opt_reduce": False, "fast_log": True},
                        ),
                    )
                ],
                f"{self.__class__.__name__}_stream",
                cfake.CppCompilerConfig(),
            )
```

> `_make()` and `_make_stream()` both call `self._get_factor_func()` — the compiled module is named after `self.__class__.__name__`, so `Alpha158Stock` and `Alpha158SpotKline` get **separate** compiled artifacts automatically. No collision risk from adding the Stock variant.

---

## 3. Exact Current Source — `base/config.py` (the dataclasses that change)

```python
# lines 1-8
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Literal

from enums.data import Frequency, Market

if TYPE_CHECKING:
    from .data import Dataset
    from .factor import FactorKunQuant
```

```python
# lines 66-83 -- FactorConfig, EXACT current field order
@dataclass
class FactorConfig:
    window: int
    dataset: "Dataset"
    mode: Literal["stream", "batch"]
    data_columns: list
    file_path: str | None = None
    factor_names: list | None = None
    start_date: str | None = None
    end_date: str | None = None
    symbols: list | None = None
    njobs: int = 128
    kwargs: dict | None = None

    name: str | None = None

    def to_dict(self):
        return asdict(self)
```

`DLConfig` / `MLConfig` type hints to widen (lines 89–90, 124–125), verbatim:

```python
@dataclass
class DLConfig:
    # 数据相关
    factors: list["FactorKunQuant"]
    labels: list["FactorKunQuant"]
    model_save_dir: str
    factor_data_strategy: Literal["read", "cal"]
    label_data_strategy: Literal["read", "cal"]
    ...
```

```python
@dataclass
class MLConfig:
    # 数据相关
    factors: list["FactorKunQuant"]
    labels: list["FactorKunQuant"]
    model_save_dir: str
    factor_data_strategy: Literal["read", "cal"]
    label_data_strategy: Literal["read", "cal"]
    ...
```

**Construction-site audit (proves `kw_only=True` is safe):** every `FactorConfig(...)` construction in the repo is 100% keyword-argument:
- `config/__init__.py:127` (`alpha101_config`), `:154` (`alpha158_config`), `:184` (`spot_label_config`)
- `test.py:27` (exploratory script)
- `utils/module.py:18` — `FactorConfig(**config)`, dict-splat, also keyword-only

Zero positional constructions found. `@dataclass(kw_only=True)` is a zero-behavior-change edit.

**`asdict()` hazard:** `to_dict()` calls `dataclasses.asdict()`, which recurses into `dataset: "Dataset"` — a non-dataclass, so it is passed through by reference. `Factor.get_config()` (§2a) then overwrites `cfg["dataset"]` with `self.config.dataset.get_config()`. This behavior must be preserved identically on `BaseFactorConfig`; keep `to_dict()` defined once on the base, not duplicated on each subclass.

---

## 4. Exact Current Source — `dataset/stock.py:_to_kunquant()` (lines 48–70)

```python
    def _to_kunquant(
        self, data: xr.Dataset, data_columns: tuple
    ) -> tuple[dict, np.ndarray, np.ndarray]:
        with Timer(f"{self.__class__.__name__}: to kunquant"):
            data = data.drop_vars(["open", "high", "low", "close", "volume"])
            data = data.rename(
                {
                    "adjOpen": "open",
                    "adjHigh": "high",
                    "adjLow": "low",
                    "adjClose": "close",
                    "adjVolume": "volume",
                }
            )
            data = data.sortby(["timestamp", "symbol"])
            timestamp = data["timestamp"].values
            symbols = data["symbol"].values
            input_dict = {}
            for col in data_columns:
                input_dict[col] = np.ascontiguousarray(
                    data[col].to_numpy().astype(np.float32)
                )  # [time, symbol]
            return input_dict, symbols, timestamp
```

**The analog to mirror — `dataset/spot.py:_to_kunquant()` (lines 117–139), which already supplies `amount`:**

```python
    def _to_kunquant(
        self, data: xr.Dataset, data_columns: tuple
    ) -> tuple[dict, np.ndarray, np.ndarray]:
        with Timer(f"{self.__class__.__name__}: to kunquant"):
            data = data.rename(
                {
                    "Quote asset volume": "amount",
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                }
            )
            data = data.sortby(["timestamp", "symbol"])
            timestamp = data["timestamp"].values
            symbols = data["symbol"].values
            input_dict = {}
            for col in data_columns:
                input_dict[col] = np.ascontiguousarray(
                    data[col].to_numpy().astype(np.float32)
                )  # [time, symbol]
            return input_dict, symbols, timestamp
```

The two differ ONLY in the rename block. The D-02 `amount = volume * close` synthesis goes into the Stock version's rename block region (after `.rename(...)`, before `.sortby(...)` or immediately after — both work; RESEARCH.md proposes after `sortby`). Note the Stock version's `data["volume"]`/`data["close"]` are the **adj-renamed** vars at that point, so the proxy is adjusted-dollar-volume — consistent with the rest of the adjusted-price pipeline.

The `to_kunquant()` public wrapper in `base/data.py` (lines 151–160) is unchanged:

```python
    def to_kunquant(
        self, data_columns: tuple[str, ...]
    ) -> tuple[dict, np.ndarray, np.ndarray]:
        data = self.read().get_xarray_dataset()
        return self._to_kunquant(data, data_columns)
```

---

## 5. Exact Current Source — the factor classes being mirrored/fixed

### `factor/alpha158.py:Alpha158SpotKline` — the template `Alpha158Stock` copies

Full class already quoted in RESEARCH.md Pattern 3; the on-disk source (lines 13–85) is identical to what RESEARCH.md shows. Key structural facts the planner needs:
- It defines **five** methods beyond the ABC contract: `_get_factor_names`, `_get_func_names`, `_factor_names_stream`, `_get_func_stream`, `_get_factor_func` (a one-line delegate to `_get_func_stream`).
- `_get_func_names()` calls `Alpha158.AllData(...)` **outside** any `Builder()` context; `_get_func_stream()` calls it again **inside** `with builder:`. This double-build is intentional (names first, ops second) — replicate it exactly in `Alpha158Stock`, do not "optimize" it.
- `_get_func_stream()` declares `close/low/high/vopen/amount/vol` `Input(...)` nodes inside the builder that it then **never uses** (it calls `self._get_func_names()` instead). This is dead-but-load-bearing-looking code in the original; `Alpha158Stock` should mirror it for structural parity, or the planner should consciously decide to drop it in both classes.

### `factor/alpha101.py:Alpha101Stock` (lines 54–90) — the broken class to fix

```python
class Alpha101Stock(FactorKunQuant):
    def __init__(self, factor_config: FactorConfig):
        super().__init__(factor_config)

    def _get_factor_func(self) -> Function:
        factor_names = self.get_factor_names()
        builder = Builder()
        with builder:
            close = Input("close")
            low = Input("low")
            high = Input("high")
            vopen = Input("open")
            vol = Input("volume")
            all_data = Alpha101.AllData(
                low=low,
                high=high,
                close=close,
                open=vopen,
                volume=vol,
            )
            for alpha in Alpha101.all_alpha:
                if alpha.__name__ in factor_names:
                    Output(
                        alpha(all_data),
                        alpha.__name__,
                    )
        return Function(builder.ops)
```

Two deltas vs. the working `Alpha101SpotKline` (lines 13–51):
1. **Missing `amount = Input("amount")` and `amount=amount` in `AllData(...)`** → the verified `RuntimeError: Bad inputs, given <class 'NoneType'>`.
2. **Missing `WindowedZScore(..., self.config.window)` wrapper** around `alpha(all_data)` — `Alpha101SpotKline` normalizes, `Alpha101Stock` does not. RESEARCH.md's fix section only mentions (1); the planner should explicitly decide whether (2) is also an unintentional divergence (it makes the two markets' factor values non-comparable in scale) or a deliberate choice, and record that decision.

---

## 6. Import-Direction Map (no cycle introduced)

Current runtime import edges among the touched modules:

```
enums/*, utils/*                 (leaf)
        ▲
base/config.py ── TYPE_CHECKING only ──▶ base/data.py, base/factor.py   [no runtime edge]
        ▲
        │ (runtime)
base/factor.py ──▶ base/config.py, dataset/backend.py, enums/constant, utils/timer
base/data.py   ──▶ base/config.py, base/backend.py, dataset/cleaning
        ▲
factor/alpha101.py, factor/alpha158.py, label/spot.py ──▶ base/factor.py, base/config.py
        ▲
config/__init__.py ──▶ base/config.py, dataset/spot.py, dataset/backend.py
        ▲
base/model.py ──▶ base/config.py   (never imports base/factor.py)
```

Proposed additions and their verdict:

| New edge | Cycle? | Why |
|---|---|---|
| `base/factor_polars.py` → `base/factor.py` (`Factor`) | No | one-directional, `base/factor.py` gains no back-import |
| `base/factor_polars.py` → `base/config.py` (`PolarsFactorConfig`) | No | same shape as the existing `base/factor.py → base/config.py` runtime edge |
| `base/config.py` → `base/factor.py` (`Factor`) | No | stays inside the existing `if TYPE_CHECKING:` block — change `from .factor import FactorKunQuant` to `from .factor import Factor`; no runtime import is created |
| `factor/momentum.py` → `base/factor_polars.py` | No | mirrors `factor/alpha101.py → base/factor.py` exactly |
| `config/__init__.py` → `dataset/stock.py` (for the new Stock factories) | No | `config/__init__.py` already imports `dataset/spot.py`; `dataset/stock.py` does not import `config/` |

**Verified: the proposed hierarchy introduces zero import cycles.** The one thing that would create a cycle — importing `Factor` at runtime in `base/config.py` — is already prevented by the pre-existing `TYPE_CHECKING` guard, which must be preserved.

**Purity-test constraint (`tests/test_extensibility_contract.py`):** that test asserts `base/factor.py`, `base/model.py`, `base/backend.py` contain none of the substrings `SpotKlineDataset`, `StockDataset`, `crypto_spot`, `us_equity`. The refactored `base/factor.py` must stay clean; the planner should also add `base/factor_polars.py` to that test's `CORE_LAYER_FILES` tuple, since it is a new core-layer file with the same purity obligation.

---

## 7. Interchangeability Contract — exact `base/model.py` call surface

These are the ONLY methods `BaseModel` invokes on a factor/label object (verified by reading `base/model.py:115–201`). Every one must exist on `Factor` for D-03 to hold:

| Call site (`base/model.py`) | Expression | Must live on `Factor` as |
|---|---|---|
| `_reset_factors_config` L121-126 | `factor.config.start_date = ...`, `factor.config.end_date = ...`, `factor._reset_dataset_config()` | concrete |
| `_reset_labels_config` L129-135 | same on `label` | concrete |
| `_get_labels_batch` L146,148 | `label.cal().get_labels()` / `label.read().get_labels()` | `cal()` abstract, `read()`/`get_labels()` concrete |
| `_get_features_batch` L163,165 | `factor.cal().get_features()` / `factor.read().get_features()` | `cal()` abstract, `read()`/`get_features()` concrete |
| `get_factor_names` L187 | `factor._get_factor_names()` (**private**, not the public wrapper) | abstract |
| `get_label_names` L194 | `label._get_factor_names()` | abstract |
| `get_config` L200-201 | `factor.get_config()` / `label.get_config()` | concrete |

Both `cal()` (L146/L163) and `read()` (L148/L165) are dispatched through `match self.config.factor_data_strategy` — no `isinstance` branching anywhere. **`base/model.py` requires zero edits**; that is the phase's interchangeability proof, and a test asserting "no `isinstance`/`FactorKunQuant` reference in `base/model.py`" is a cheap regression lock (mirrors the existing purity test's grep style).

Note `get_factor_names()` returning `self.config.factor_names` (a `list`, per the dataclass annotation) while the abstract `_get_factor_names()` is annotated `-> tuple[str, ...]` — a pre-existing type inconsistency. `BaseModel.get_factor_names()` uses `chain.from_iterable`, which accepts either. Don't try to fix this in-flight.

---

## 8. Conventions Observed in the Factor Layer

**Class/file naming**
- Factor modules: lowercase, named after the factor family — `factor/alpha101.py`, `factor/alpha158.py`, `label/spot.py`. New: `factor/momentum.py`.
- Classes: `<FactorFamily><DatasetFlavor>` — `Alpha101SpotKline`, `Alpha101Stock`, `Alpha158SpotKline`. New: `Alpha158Stock` (NOT `Alpha158StockKline`).
- ABCs: `Factor` / `FactorKunQuant` / `FactorPolars` — backend suffix, no `Base` prefix (contrast `BaseModel` in the model layer; the factor layer does not use that prefix).

**Constructor signature** — every concrete factor uses the parameter name `factor_config`, not `config`:
```python
def __init__(self, factor_config: FactorConfig):
    super().__init__(factor_config)
```
The ABC's own `__init__` uses `config`. Preserve both.

**Method-name prefixing** — `_get_x()` = subclass hook taking already-materialized data; `get_x()` = public wrapper that sources the data and delegates. Follow this for `_get_factor_lazyframe()`.

**The `_get_labels` refusal idiom** — every feature-only factor class ends with:
```python
    def _get_labels(self, data: xr.Dataset) -> NoReturn:
        raise RuntimeError(f"{__class__.__name__} does not support get_label()")

    def _get_features(self, data: xr.Dataset) -> xr.Dataset:
        return data
```
`from typing import NoReturn` is imported for exactly this. (`__class__.__name__` — the implicit-closure form, not `self.__class__.__name__` — is what's on disk in all three classes; keep consistent even though `self.class_name` would be cleaner.)

**Timing** — every expensive operation is wrapped in `with Timer(f"{self.__class__.__name__}: <verb>"):` from `utils/timer.py`. `FactorPolars.cal()` should wrap its `.collect()` the same way. Note the inconsistent leading space in some f-strings (`f" {self.__class__.__name__}: cal"`) — cosmetic, don't chase.

**Fluent returns** — `read()`, `save()`, `cal()`, `cal_stream()`, `init_stream()` all `return self` typed `-> Self` (`from typing import Self`, Python 3.13). Mandatory for `FactorPolars.cal()` since `base/model.py` chains `.cal().get_features()`.

**Comments** — inline explanatory comments and docstrings in the `base/` layer are Chinese; newer Phase-2 code (`dataset/stock.py`, `tests/`) uses English with decision-ID references (e.g. `# D-02: ...`, `# 02-CONTEXT.md D-12`). For new Phase-3 code, follow the newer English + decision-ID convention; for hoisted code, preserve the original Chinese verbatim.

**Config factories** (`config/__init__.py`) — all keyword-only, all paths derived from `_data_root()` / `_market_data_root()` / `_market_downloads_root()`, never hardcoded absolutes. New factories must use these helpers. Factor zarr outputs go to `_data_root() / "data" / "factor" / "<name>.zarr"`.

**Tests** — `tests/test_<subject>.py`, module-level private `_helper()` builders, `tmp_path` fixture, one behavior per test with a docstring naming the requirement/criterion it proves (`tests/test_stock_dataset.py`, `tests/test_extensibility_contract.py` are the models to follow).

---

## 9. Analog Map for Each New File

| New file / class | Copy structure from | Copy idioms from | Do NOT copy |
|---|---|---|---|
| `base/factor.py:Factor(ABC)` | `base/data.py:Dataset` (ABC with config-property lifecycle, `read`/`save`, backend attribute, `@abstractmethod` tail) | `base/factor.py:FactorKunQuant` verbatim for §2a methods | Any `self.config.mode` / `njobs` / `data_columns` reference — those move to `FactorConfig` |
| `base/factor_polars.py:FactorPolars` | `base/factor.py:FactorKunQuant` (shape: `__init__` → hooks → `cal()`) | `dataset/backend.py:PlBackend` + `dataset/stock.py:_raw_data_to_xr()` for the `.collect().to_pandas().set_index([...]).to_xarray()` boundary conversion | streaming methods, `_make*`, `Function`/KunQuant imports |
| `factor/momentum.py:Momentum` | `factor/alpha101.py:Alpha101SpotKline` (constructor, `_get_labels` refusal, `_get_features` passthrough) | `dataset/cleaning.py` for polars-expression style; `.over("symbol")` window idiom | KunQuant `Builder`/`Input`/`Output` |
| `factor/alpha158.py:Alpha158Stock` | `factor/alpha158.py:Alpha158SpotKline` — structural copy, all five methods | — | nothing; it is a deliberate near-duplicate per D-01 |
| `config/__init__.py:stock_alpha158_config()` etc. | `alpha158_config()` in the same file | `stock_kline_config()` for the `StockDataset` wiring | hardcoded paths |
| `tests/test_factor_*.py` | `tests/test_stock_dataset.py` (fixture helpers + tmp_path) and `tests/test_extensibility_contract.py` (purity/introspection assertions) | — | network or real-data dependencies |

---

## 10. Planner Watch-Items (concrete, from source inspection)

1. **`__init__` ordering** — `self.config = config` (setter, which calls `_get_factor_names()`) precedes `self.data_backend = XrBackend()`. Preserve.
2. **`mode` removal from the shared path** — `_auto_filter`, `num_symbols`, `symbols` all read `self.config.mode`; all three need the hoist-with-override split or `FactorPolars` breaks at runtime (`AttributeError`), not at type-check time.
3. **`Alpha101Stock` has two divergences** from its SpotKline sibling (missing `amount`, missing `WindowedZScore`). RESEARCH.md flags only the first. Decide on the second explicitly.
4. **`_get_func_stream()` in `Alpha158SpotKline` declares unused `Input(...)` nodes** — replicate or consciously drop in both classes; don't silently diverge.
5. **Add `base/factor_polars.py` to `tests/test_extensibility_contract.py:CORE_LAYER_FILES`** if the new module is created.
6. **`utils/module.py:18` hardcodes `FactorConfig(**config)`** — out of scope per RESEARCH Open Question #1, but a `Factor.config_cls` ClassVar declared now (even if unused) would cost one line and unblock Phase 4. Planner's call.
7. **`to_dict()`/`asdict()` recursion into `dataset`** — define `to_dict()` once on `BaseFactorConfig`; `get_config()`'s `cfg["dataset"]` overwrite depends on it.

---

*Pattern map for phase 03-factor-computation-kunquant-polars — 2026-09-05*
