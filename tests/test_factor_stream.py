"""Streaming KunQuant factor-computation tests (FACTOR-02).

Scaffolded by 03-01 Task 2 (Nyquist Wave 0). The real content -- the
aarch64 SIMD block-width fix (BUG-02) and the incremental `cal_stream()`
replay smoke test -- lands in **03-05**.

The single test below is deliberately an infrastructure self-test rather than
a placeholder: it proves the shared `spot_kline_zarr` fixture delivers exactly
the panel shape the FACTOR-02 replay test will consume (8 symbols), so
`uv run pytest tests/test_factor_stream.py -q` is a meaningful command today
rather than an exit-code-5 "no tests ran".

Import-safety rule (tests/conftest.py module docstring): nothing here may
import a module that does not exist yet at module level.
"""

from typing import Callable

from base.config import DatasetConfig
from dataset.spot import SpotKlineDataset


def test_stream_fixture_provides_eight_symbols(
    spot_kline_zarr: Callable[..., DatasetConfig],
) -> None:
    """The default `spot_kline_zarr()` panel is 8 symbols wide and at least 40
    timestamps deep -- the shape 03-05's streaming replay test requires
    (8 symbols matches KunQuant's stream `blocking_len`/`partition_factor`
    of 8; the depth must exceed the largest rolling factor window).
    """
    dataset_config = spot_kline_zarr()
    data = SpotKlineDataset(dataset_config).read().get_xarray_dataset()

    assert data.sizes["symbol"] == 8
    assert data.sizes["timestamp"] >= 40
