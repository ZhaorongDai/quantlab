"""The bounded-read contract on the `DataBackend` interface.

`head(n)` is the bounded twin of the already-public `get_lazyframe()`: it
answers "what does this store look like?" without materializing it. It exists
because `base/factor_polars.py` must derive its factor names from the
computation graph at config-assignment time, and doing that through one
concrete backend's private path (`xr.open_zarr(...).isel(...)`) would use a
backend-specific escape hatch to fix a backend-dependence gap.

Every behavioural test below drives BOTH concrete backends from one shared
fixture, with no per-backend branch: that uniformity is the contract. The
fourth test is the reason `head` is an `@abstractmethod` rather than an
optional `limit=` keyword on `get_lazyframe()` -- ABC enforcement makes a
future backend impossible to construct without one, whereas a keyword is
satisfied by plain inheritance and only fails at whichever call site happens
to pass it.

**The non-mutation test carries the hazard.** `filter_by_date` and
`filter_by_symbol` on `XrBackend` mutate `self.data` IN PLACE, so a `head`
implementation copied from them would silently truncate the dataset object
every consumer shares -- after which `cal()` computes over a handful of rows
forever and nothing downstream can tell.
"""

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import polars as pl
import pytest
import xarray as xr

from base.backend import DataBackend
from dataset.backend import PlBackend, XrBackend

#: 10 timestamps x 3 symbols = 30 long-format rows.
_PERIODS = 10
_SYMBOLS = ("AAA", "BBB", "CCC")
_TOTAL_ROWS = _PERIODS * len(_SYMBOLS)


def _synthetic_dataset() -> xr.Dataset:
    """A small OHLCV-shaped `xr.Dataset` over `[timestamp, symbol]`."""
    rng = np.random.default_rng(0)
    shape = (_PERIODS, len(_SYMBOLS))
    return xr.Dataset(
        {
            "Close": (["timestamp", "symbol"], rng.random(shape) + 1.0),
            "Volume": (["timestamp", "symbol"], rng.random(shape) * 100.0),
        },
        coords={
            "timestamp": pd.date_range("2024-01-01", periods=_PERIODS),
            "symbol": list(_SYMBOLS),
        },
    )


@pytest.fixture
def backends(tmp_path: Path) -> list[DataBackend]:
    """One `XrBackend` and one `PlBackend` over the SAME 30 long-format rows.

    The parquet is written from the xarray store's own long-format frame, so
    the two backends carry identical column names and dtypes and every test
    below can treat them uniformly.
    """
    dataset = _synthetic_dataset()
    frame = pl.from_pandas(dataset.to_dataframe().reset_index())

    parquet_path = tmp_path / "rows.parquet"
    frame.write_parquet(parquet_path)

    return [
        XrBackend().to_internal(dataset),
        PlBackend().read(str(parquet_path)),
    ]


def test_head_returns_at_most_n_rows(backends: list[DataBackend]) -> None:
    """`head(4)` over a 30-row store yields at most 4 rows on both backends."""
    for backend in backends:
        height = backend.head(4).collect().height

        assert 0 < height <= 4, (
            f"{backend!r}.head(4) returned {height} rows out of "
            f"{_TOTAL_ROWS}; a bounded read must not materialize the store"
        )


def test_head_preserves_the_get_lazyframe_schema(
    backends: list[DataBackend],
) -> None:
    """`head(n)` carries the same column names AND dtypes `get_lazyframe()`
    returns.

    The dtype half is not decoration: `base/factor_polars.py` runs a real
    factor expression over this probe, and an integer division or a `.str.`
    operation derives a different result -- or fails outright -- under the
    wrong dtype. The names half catches an `XrBackend` implementation that
    forgets `reset_index()` and silently drops `timestamp`/`symbol`.
    """
    for backend in backends:
        assert (
            backend.head(2).collect_schema()
            == backend.get_lazyframe().collect_schema()
        ), f"{backend!r}.head(2) does not preserve the get_lazyframe() schema"


def test_head_does_not_mutate_backend_state(
    backends: list[DataBackend],
) -> None:
    """A probe read leaves the backing store untouched.

    `filter_by_date`/`filter_by_symbol` on `XrBackend` DO mutate `self.data`
    in place. An implementation copied from them would truncate the dataset
    object shared with `cal()` down to the probe size, and every later
    computation would silently run over those few rows.
    """
    for backend in backends:
        backend.head(2).collect()

        assert backend.get_lazyframe().collect().height == _TOTAL_ROWS, (
            f"{backend!r}.head(2) mutated the backing store; the probe must "
            "return a derived frame and leave self.data alone"
        )


def test_head_is_an_interface_obligation_not_a_convenience() -> None:
    """A `DataBackend` subclass without `head` cannot be instantiated.

    Declared abstract on purpose. A future backend that forgot the bounded
    read would otherwise inherit a silent gap that only surfaces at the one
    call site that needs it -- this repo's recorded "gate whose flag was
    always true where it was read" shape.
    """
    assert "head" in DataBackend.__abstractmethods__

    class BackendWithoutHead(DataBackend):
        def get_xarray_dataset(self, indexes: list[str]) -> xr.Dataset: ...

        def get_lazyframe(self) -> pl.LazyFrame: ...

        def read(self, path: str, **kwargs): ...

        def write(self, path: str, **kwargs): ...

        def to_internal(self, data): ...

        def filter_by_date(self, col: str, start_date: str, end_date: str): ...

        def filter_by_symbol(self, col: str, symbols: tuple[str, ...]): ...

    with pytest.raises(TypeError, match="head"):
        BackendWithoutHead()
