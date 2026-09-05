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


def _momentum_config(
    dataset_config: DatasetConfig, tmp_path: Path, n: int = 5
) -> PolarsFactorConfig:
    """Build a `PolarsFactorConfig` for `Momentum` over a synthetic Zarr store.

    Deliberately does NOT use `config.momentum_config()`: that factory points
    at production data paths, whereas every test here runs against the
    `tmp_path`-scoped `spot_kline_zarr` fixture.
    """
    return PolarsFactorConfig(
        window=n,
        dataset=SpotKlineDataset(dataset_config),
        file_path=str(tmp_path / "factors" / "momentum.zarr"),
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
    """D-05: Polars factor names come from the computed frame's own schema,
    never from a declaration.

    The `RuntimeError` half is the documented precondition (03-RESEARCH.md
    Pitfall 4): `_get_factor_names()` -- and therefore `num_factors` -- is only
    valid after `cal()`/`read()` has populated `config.factor_names`. That is a
    deliberate, narrow gap, not a bug: `base/model.py` always calls
    `.cal()`/`.read()` before asking a factor for its names.
    """
    config = _momentum_config(spot_kline_zarr(), tmp_path, n=5)
    factor = Momentum(config)

    with pytest.raises(RuntimeError, match="cal"):
        factor._get_factor_names()

    factor.cal()

    assert factor.get_factor_names() == ("momentum_5",)
    assert factor.num_factors == 1


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
