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

from quantlab.base.config import DatasetConfig, PolarsFactorConfig
from quantlab.base.factor import FactorPolars
from quantlab.dataset.spot import SpotKlineDataset
from quantlab.factor.momentum import Momentum


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
    window: int | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> PolarsFactorConfig:
    """Build a `PolarsFactorConfig` for `Momentum` over a synthetic Zarr store.

    Every test here runs against the `tmp_path`-scoped `spot_kline_zarr`
    fixture rather than production data paths.

    `factor_names` defaults to `None` -- the normal case, in which the names
    are derived from the computation graph at config-assignment time. Passing
    a value exercises the explicit-pin channel instead.

    `window` defaults to `n`, and `start_date`/`end_date` to `None`, so every
    pre-existing caller is byte-identical. Pass them to separate the LOOKBACK
    the factor asks `_reset_dataset_config()` to widen the dataset by from the
    horizon the factor computes over -- the two are the same number by
    default here, which is precisely why no existing test could tell whether
    the widening reached the dataset at all.
    """
    return PolarsFactorConfig(
        window=n if window is None else window,
        dataset=SpotKlineDataset(dataset_config),
        file_path=str(tmp_path / "factors" / "momentum.zarr"),
        factor_names=factor_names,
        start_date=start_date,
        end_date=end_date,
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

    monkeypatch.setattr("quantlab.base.data.BaseDataset.head", _forbidden_head)

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


def test_a_dated_dataset_config_keeps_the_factor_lookback_window(
    spot_kline_zarr: Callable[..., DatasetConfig], tmp_path: Path
) -> None:
    """RV-01: a factor whose DATASET config carries dates computes over the
    WIDENED window `_reset_dataset_config()` asked for, not the narrow
    pre-widening window the construction-time name probe used to leave behind.

    The mechanism, because the assertion is meaningless without it. The probe
    that derives the factor names fires from inside the `Factor.config` setter
    (`base/factor.py`), BEFORE `_reset_dataset_config()` widens the dataset's
    `start_date` by the factor's `window` days. While that probe went through
    `BaseDataset.read()`, it ran `_filter()`, which narrows
    `data_backend.data` IN PLACE via `filter_by_date` -- and `filter_by_date`
    can only ever narrow. `XrBackend.read()`'s cache early-return then made
    the narrowing PERMANENT: `cal()`'s own `read()` got the already-truncated
    object back instead of re-opening the store. So the factor computed over
    the 29 requested timestamps with no lookback at all, and its first `n`
    rows per symbol came out NaN with nothing raised anywhere.

    Hence both assertions read the dataset the factor ACTUALLY COMPUTED OVER,
    never what its config claims. The expected count is the literal 49 and is
    deliberately NOT derived from `config.dataset.config.start_date`: that
    value is written by the very code under test, so a derived expectation
    passes just as happily under the bug.

    Store: 120 daily bars from 2024-01-01. Requested: 2024-02-01..2024-02-29
    (29 bars). Lookback: `window=20`, so the dataset must widen back to
    2024-01-12 -- 20 + 29 = 49 bars.
    """
    dataset_config = spot_kline_zarr(
        periods=120, start_date="2024-02-01", end_date="2024-02-29"
    )
    config = _momentum_config(
        dataset_config,
        tmp_path,
        n=5,
        window=20,
        start_date="2024-02-01",
        end_date="2024-02-29",
    )

    factor = Momentum(config).cal()

    computed_over = factor.config.dataset.get_xarray_dataset()

    assert computed_over.sizes["timestamp"] == 49, (
        f"the factor computed over {computed_over.sizes['timestamp']} "
        "timestamps; it must be 49 -- the 29 requested bars plus the 20 days "
        "of lookback _reset_dataset_config() widened the dataset by. A count "
        "of 29 is RV-01: the construction-time name probe filtered the "
        "shared dataset down to the requested window before the widening "
        "ever happened, and XrBackend.read()'s cache made it stick."
    )

    momentum = factor.get_features()["momentum_5"]
    nan_count = int(np.isnan(momentum.values).sum())

    assert nan_count == 0, (
        f"momentum_5 carries {nan_count} NaN over the requested window; with "
        "20 days of lookback preserved, every one of the 29 requested bars "
        "has 5 prior bars to shift against. Under RV-01 the first 5 "
        "timestamps per symbol are NaN (~17% of the panel) because the "
        "lookback was silently dropped."
    )
