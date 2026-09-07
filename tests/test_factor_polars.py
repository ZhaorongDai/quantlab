"""Polars factor-backend tests (FACTOR-03).

Scaffolded by 03-01 Task 2 (Nyquist Wave 0). The real content -- the
`FactorPolars` ABC (`base/factor_polars.py`), the `Momentum` example factor
(`factor/momentum.py`, D-08) and the D-04 laziness proof -- lands in **03-04**.

`test_spot_kline_lazyframe_exposes_raw_title_case_columns` is the original
infrastructure self-test: it locks in the raw column contract
`factor/momentum.py` is written against. `FactorPolars`
consumes `Dataset.get_lazyframe()`, which -- unlike the KunQuant path's
`_to_kunquant()` -- performs **no** rename, so a Polars factor over crypto
spot data sees Binance's raw Title-Case `Close`, not the lowercase `close`
KunQuant receives. Getting that backwards is the most likely way 03-04's
first draft fails.

03-04 has since landed, so the four contract proofs below now sit alongside
that scaffold test: the FACTOR-03/FACTOR-04 end-to-end computation, the D-05
dynamic-name proof, the D-04 runtime laziness proof and the D-07 no-streaming
assertion. `base.factor_polars`, `factor.momentum` and `PolarsFactorConfig`
all exist and are imported at module level; the collect-time import-safety
rule in `tests/conftest.py`'s docstring applies to that fixture module, not
to this one.
"""

from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl
import pytest
import xarray as xr

from base.config import DatasetConfig, PolarsFactorConfig
from base.factor_polars import FactorPolars
from dataset.spot import SpotKlineDataset
from factor.momentum import Momentum


class _ProbeCalled(Exception):
    """Raised by a patched `BaseDataset.head` to prove the probe ran.

    A test-local exception type rather than a built-in: it can only come from
    the patch below, so a test asserting on it cannot pass by accident on some
    unrelated failure that happens to raise the same class.
    """


def _momentum_config(
    dataset_config: DatasetConfig,
    tmp_path: Path,
    n: int = 5,
    factor_names: list | None = None,
) -> PolarsFactorConfig:
    """Build a `PolarsFactorConfig` for `Momentum` over a synthetic Zarr store.

    Deliberately does NOT use `config.momentum_config()`: that factory points
    at production data paths, whereas every test here runs against the
    `tmp_path`-scoped `spot_kline_zarr` fixture.

    `factor_names` defaults to `None` -- the normal case, in which the names
    are derived from the computation graph at config-assignment time. Passing
    a value exercises the explicit-pin channel instead.
    """
    return PolarsFactorConfig(
        window=n,
        dataset=SpotKlineDataset(dataset_config),
        file_path=str(tmp_path / "factors" / "momentum.zarr"),
        factor_names=factor_names,
        kwargs={"n": n},
    )


def test_spot_kline_lazyframe_exposes_raw_title_case_columns(
    spot_kline_zarr: Callable[..., DatasetConfig],
) -> None:
    """`SpotKlineDataset.get_lazyframe()` exposes the `[timestamp, symbol]`
    index columns plus Binance's RAW Title-Case OHLCV names -- the Polars
    backend's boundary contract (FACTOR-03 / D-04), distinct from the
    lowercase names `_to_kunquant()` renames to for KunQuant.
    """
    dataset_config = spot_kline_zarr()
    lazyframe = SpotKlineDataset(dataset_config).read().get_lazyframe()

    names = lazyframe.collect_schema().names()

    assert "timestamp" in names
    assert "symbol" in names
    assert "Close" in names


def test_momentum_cal_returns_xarray_dataset_with_only_factor_columns(
    spot_kline_zarr: Callable[..., DatasetConfig], tmp_path: Path
) -> None:
    """FACTOR-03 / FACTOR-04 / D-06: `Momentum.cal().get_features()` returns an
    `xr.Dataset` over `[timestamp, symbol]` whose data_vars are EXACTLY the
    computed factor columns.

    The `data_vars == ["momentum_5"]` half is also the D-04 leakage check: a
    `_get_factor_lazyframe` that forgot its final `select(...)` would persist
    `Close`/`Volume` into the factor store as if they were factors.
    """
    config = _momentum_config(spot_kline_zarr(), tmp_path, n=5)

    result = Momentum(config).cal().get_features()

    assert isinstance(result, xr.Dataset)
    assert sorted(result.data_vars) == ["momentum_5"]
    assert set(result.sizes) == {"timestamp", "symbol"}
    assert int(np.isfinite(result["momentum_5"].values).sum()) > 0


def test_factor_names_resolve_dynamically_from_the_lazyframe_schema(
    spot_kline_zarr: Callable[..., DatasetConfig], tmp_path: Path
) -> None:
    """D-05: Polars factor names come from the computation GRAPH's own schema,
    never from a declaration -- and they are known from construction onward.

    The assertions run BEFORE `cal()` first, and that ordering is the point.
    Names are derived at config-assignment time through a bounded probe read
    (03-VERIFICATION.md Gap 1), so a bare-constructed factor already reports
    them; there is no state in which a `FactorPolars` cannot answer what it
    computes. Repeating the same two assertions after `cal()` keeps the
    original coverage: computing must not change the answer.

    Deriving from the graph rather than from the factor store on disk is the
    decided behaviour, not an implementation accident. A store written under
    `n=5` read back through a config now saying `n=60` yields `momentum_60`
    and fails loudly at lookup, instead of silently reporting the stale name
    the data happens to carry.
    """
    config = _momentum_config(spot_kline_zarr(), tmp_path, n=5)
    factor = Momentum(config)

    assert factor.get_factor_names() == ("momentum_5",)
    assert factor.num_factors == 1

    factor.cal()

    assert factor.get_factor_names() == ("momentum_5",)
    assert factor.num_factors == 1


def test_an_explicit_factor_names_pin_is_not_overwritten_at_construction(
    spot_kline_zarr: Callable[..., DatasetConfig],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit `config.factor_names` wins, and skips the probe entirely.

    `base/factor.py:_maybe_resolve_factor_names` owns the two channels:
    explicit pin, else derive. `FactorPolars` supplies only the derivation, so
    the pin comes free from the base class -- this test is what proves that
    "free" is real rather than assumed.

    Two-sided by design, and the second half is what makes the first half
    meaningful. With the bounded read patched to raise, the PINNED factor
    constructs (the derivation was never reached) while the UNPINNED one
    raises (the derivation is precisely what the pin skips). Asserting only
    the first half would pass just as well against an implementation that
    probed and then discarded the result.

    Note that `cal()` still overwrites `config.factor_names` from the
    collected schema. That is pre-existing, unchanged behaviour and is
    deliberately not asserted here -- this test is about construction.
    """
    dataset_config = spot_kline_zarr()

    def _forbidden_head(self, n: int):
        raise _ProbeCalled(
            "BaseDataset.head() was called; an explicit factor_names pin must "
            "short-circuit the derivation before any probe read"
        )

    monkeypatch.setattr("base.data.BaseDataset.head", _forbidden_head)

    pinned = Momentum(
        _momentum_config(
            dataset_config, tmp_path, n=5, factor_names=["pinned_name"]
        )
    )

    assert list(pinned.get_factor_names()) == ["pinned_name"]
    assert pinned.num_factors == 1

    with pytest.raises(_ProbeCalled):
        Momentum(_momentum_config(dataset_config, tmp_path, n=5))


def test_get_factor_lazyframe_stays_lazy_until_cal(
    spot_kline_zarr: Callable[..., DatasetConfig],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """D-04: nothing inside `_get_factor_lazyframe` materializes -- `cal()` is
    what triggers computation.

    Proven at runtime rather than by reading source: `pl.LazyFrame.collect` is
    replaced with a stub that raises, so any eager call inside the hook is an
    immediate hard failure instead of a silent per-call performance cliff
    (T-03-04-03). Schema inspection via `collect_schema()` must still work
    under the stub -- reading names is not materializing.

    The input frame is hand-built here, with no `Dataset` and no store on
    disk. That is itself the demonstration of why the hook takes `lf` as a
    parameter: a factor's logic is unit-testable in isolation.
    """
    config = _momentum_config(spot_kline_zarr(), tmp_path, n=5)
    factor = Momentum(config)

    def _forbidden_collect(self, *args, **kwargs):
        raise AssertionError(
            "_get_factor_lazyframe() materialized its result -- the Polars "
            "factor contract (D-04) requires it to stay lazy until cal()"
        )

    monkeypatch.setattr(pl.LazyFrame, "collect", _forbidden_collect)

    frame = pl.LazyFrame(
        {
            "timestamp": list(range(6)) * 2,
            "symbol": ["AAA"] * 6 + ["BBB"] * 6,
            "Close": [float(i) + 1.0 for i in range(12)],
        }
    )

    result = factor._get_factor_lazyframe(frame)

    assert isinstance(result, pl.LazyFrame)
    assert not isinstance(result, pl.DataFrame)
    assert result.collect_schema().names() == [
        "timestamp",
        "symbol",
        "momentum_5",
    ]


def test_polars_backend_exposes_no_streaming_surface() -> None:
    """D-07: the Polars backend is batch-only BY DECISION, not by omission.

    `FactorPolars` -- and therefore every subclass of it -- carries no
    streaming or compiled-graph member. Those live exclusively on the KunQuant
    backend, which is what keeps the shared `Factor` base free of a dormant
    streaming surface no Polars factor could ever implement.
    """
    streaming_members = ("cal_stream", "init_stream", "_make", "_make_stream")

    for member in streaming_members:
        assert not hasattr(FactorPolars, member), (
            f"FactorPolars must not expose {member!r} (D-07)"
        )
        assert not hasattr(Momentum, member), (
            f"Momentum must not inherit {member!r} (D-07)"
        )
