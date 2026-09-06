"""Tests for the interval-to-daily-panel densification (DATA-05, D-04).

Every test in this file runs OFFLINE. The only route to the network in this
code path is `acquisition.universe.requests.get`, which the shared
`mock_universe_fetchers` fixture in `tests/conftest.py` replaces with a
URL-keyed fake that raises `AssertionError` on any unexpected URL. The
densification tests do not even need that: they drive module-local
`IndexConstituentDataset` subclasses whose `_build_intervals()` returns a
hand-written frame and which therefore perform no I/O at all.
"""

import numpy as np
import xarray as xr

from base.config import ConstituentDatasetConfig
from dataset.constituent import SP500ConstituentDataset

_COVERAGE_START = "1976-07-01"


def _make_config(tmp_path, **overrides) -> ConstituentDatasetConfig:
    """Module-local config constructor (03.1-PATTERNS.md §6: shared mocks live
    in conftest, per-module config constructors stay module-local)."""
    params = dict(
        zarr_file_path=str(tmp_path / "us_equity" / "sp500_constituent.zarr"),
        cache_dir=str(tmp_path / "reference" / "_cache"),
    )
    params.update(overrides)
    return ConstituentDatasetConfig(**params)  # type: ignore[arg-type]


def test_sp500_panel_round_trips_through_zarr(mock_universe_fetchers, tmp_path):
    """DATA-05 / D-04: the S&P 500 membership panel survives a full
    build -> Zarr write -> fresh-instance read cycle as a boolean
    `(timestamp, symbol)` grid."""
    cfg = _make_config(tmp_path)
    SP500ConstituentDataset(cfg).from_raw_data().save()

    reloaded = (
        SP500ConstituentDataset(_make_config(tmp_path))
        .read()
        .get_xarray_dataset()
    )

    assert isinstance(reloaded, xr.Dataset)
    assert tuple(reloaded["is_member"].dims) == ("timestamp", "symbol")
    assert set(reloaded.data_vars) == {"is_member"}
    assert reloaded["is_member"].dtype == np.dtype(bool)

    symbols = set(reloaded["symbol"].values.tolist())
    # `ADDED1` is the fixture's still-open member; `ZZZZ` was removed in 1985
    # and must still be a column, not an absent symbol.
    assert "ADDED1" in symbols
    assert "ZZZZ" in symbols
