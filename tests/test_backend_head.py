"""The bounded-read contract on the `DataBackend` interface.

`head(path, n)` is the bounded twin of the already-public `get_lazyframe()`:
it answers "what does this store look like?" without materializing it. It
exists because `base/factor_polars.py` must derive its factor names from the
computation graph at config-assignment time, and doing that through one
concrete backend's private path (`xr.open_zarr(...).isel(...)`) would use a
backend-specific escape hatch to fix a backend-dependence gap.

**The store PATH is threaded in, and that is the RV-01 fix.** `head` used to
read `self.data`, so the only way for a caller to reach the store was to call
`read()` first. `BaseDataset.read()` runs `_filter()`, which narrows
`data_backend.data` IN PLACE via `filter_by_date`, and `XrBackend.read()`'s
cache early-return makes that narrowing SURVIVE every later read. Because the
`FactorPolars` name probe fires from the `Factor.config` setter -- before
`_reset_dataset_config()` widens the window by the factor's `window` days, and
`filter_by_date` can only narrow -- a probe that went through `read()` silently
dropped the factor's entire lookback (RV-01: 29 timestamps instead of 49, and
a 17%-NaN factor column, with nothing raised). A reader who does not know that
will "simplify" these implementations back to `self.data`; that is the change
`test_head_opens_the_store_without_a_prior_read` exists to catch.

Every behavioural test below drives BOTH concrete backends from one shared
fixture, with no per-backend branch: that uniformity is the contract. The
abstractness test is the reason `head` is an `@abstractmethod` rather than an
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

import numpy as np
import pandas as pd
import polars as pl
import pytest
import xarray as xr

from quantlab.base.backend import DataBackend
from quantlab.backend import PlBackend, XrBackend

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
def stores(tmp_path: Path) -> dict[str, str]:
    """The SAME 30 long-format rows written to a zarr store and a parquet.

    The parquet is written from the xarray store's own long-format frame, so
    the two backends carry identical column names and dtypes and every test
    below can treat them uniformly.
    """
    dataset = _synthetic_dataset()
    frame = pl.from_pandas(dataset.to_dataframe().reset_index())

    zarr_path = tmp_path / "rows.zarr"
    dataset.to_zarr(zarr_path, mode="w")

    parquet_path = tmp_path / "rows.parquet"
    frame.write_parquet(parquet_path)

    return {"zarr": str(zarr_path), "parquet": str(parquet_path)}


@pytest.fixture
def backends(stores: dict[str, str]) -> list[tuple[DataBackend, str]]:
    """`(backend, path)` pairs, one per concrete backend.

    Both backends are built through `read()` -- the `XrBackend` deliberately
    NOT through `to_internal(dataset)`. `head` decodes the store from disk, so
    an in-memory `to_internal` twin would have the schema-equality test
    compare an on-disk decode against an in-memory object, which can diverge
    on datetime resolution and would fail for a reason that has nothing to do
    with the bounded read.
    """
    return [
        (XrBackend().read(stores["zarr"]), stores["zarr"]),
        (PlBackend().read(stores["parquet"]), stores["parquet"]),
    ]


def test_head_returns_at_most_n_rows(
    backends: list[tuple[DataBackend, str]],
) -> None:
    """`head(path, 4)` over a 30-row store yields at most 4 rows on both
    backends."""
    for backend, path in backends:
        height = backend.head(path, 4).collect().height

        assert 0 < height <= 4, (
            f"{backend!r}.head(path, 4) returned {height} rows out of "
            f"{_TOTAL_ROWS}; a bounded read must not materialize the store"
        )


def test_head_preserves_the_get_lazyframe_schema(
    backends: list[tuple[DataBackend, str]],
) -> None:
    """`head(path, n)` carries the same column names AND dtypes
    `get_lazyframe()` returns.

    The dtype half is not decoration: `base/factor_polars.py` runs a real
    factor expression over this probe, and an integer division or a `.str.`
    operation derives a different result -- or fails outright -- under the
    wrong dtype. The names half catches an `XrBackend` implementation that
    forgets `reset_index()` and silently drops `timestamp`/`symbol`.
    """
    for backend, path in backends:
        assert (
            backend.head(path, 2).collect_schema()
            == backend.get_lazyframe().collect_schema()
        ), (
            f"{backend!r}.head(path, 2) does not preserve the "
            "get_lazyframe() schema"
        )


def test_head_does_not_mutate_backend_state(
    backends: list[tuple[DataBackend, str]],
) -> None:
    """A probe read leaves the backing store untouched.

    `filter_by_date`/`filter_by_symbol` on `XrBackend` DO mutate `self.data`
    in place. An implementation copied from them would truncate the dataset
    object shared with `cal()` down to the probe size, and every later
    computation would silently run over those few rows.
    """
    for backend, path in backends:
        backend.head(path, 2).collect()

        assert backend.get_lazyframe().collect().height == _TOTAL_ROWS, (
            f"{backend!r}.head(path, 2) mutated the backing store; the probe "
            "must return a derived frame and leave self.data alone"
        )


def test_head_opens_the_store_without_a_prior_read(
    stores: dict[str, str],
) -> None:
    """A FRESH backend -- never `read()`, never `to_internal()` -- answers
    `head(path, 2)` with real rows.

    The decisive mechanism test for the RV-01 fix, and the one that cannot
    pass while `head` reads `self.data`: on these backends `self.data` raises
    `AttributeError("Please cal 'read' or 'to_internal' first.")`, so an
    implementation that reaches for it fails here rather than returning a
    truncated answer.

    Reaching the store through the caller's prior `read()` is exactly what
    made the `FactorPolars` probe narrow its dataset's date window in place
    (RV-01). Threading the path in removes the dependence entirely: the probe
    no longer needs anybody to have read anything.
    """
    fresh: list[tuple[DataBackend, str]] = [
        (XrBackend(), stores["zarr"]),
        (PlBackend(), stores["parquet"]),
    ]

    for backend, path in fresh:
        with pytest.raises(AttributeError):
            backend.data

        height = backend.head(path, 2).collect().height

        assert 0 < height <= 2, (
            f"a freshly-constructed {backend!r} returned {height} rows from "
            "head(path, 2); the bounded read must open the store itself "
            "rather than depend on an earlier caller having read it"
        )


def test_head_raises_immediately_on_an_absent_store(
    tmp_path: Path,
) -> None:
    """`head(missing_path, 2)` raises `FileNotFoundError` on BOTH backends,
    at the call rather than at `.collect()` time.

    Deliberate, and recorded as D-3 of the RV-01 fix plan: `head` carries
    `read()`'s own `Path(path).exists()` guard and its exact message. Without
    it a missing zarr directory surfaces as an obscure xarray engine-guess
    error, and `PlBackend`'s `scan_parquet` fails LATER, at collect time, far
    from the construction that was actually wrong. RV-02 -- that a factor can
    no longer be constructed before its dataset store exists -- stays filed in
    `03-VERIFICATION.md` and is not closed here; this test locks the error
    that decision preserves.
    """
    missing = str(tmp_path / "definitely-not-here")
    fresh: list[DataBackend] = [XrBackend(), PlBackend()]

    for backend in fresh:
        with pytest.raises(FileNotFoundError):
            backend.head(missing, 2)


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
