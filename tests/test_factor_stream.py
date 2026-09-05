"""Streaming KunQuant factor-computation tests (FACTOR-02).

Scaffolded by 03-01 Task 2 (Nyquist Wave 0); filled in by **03-05**, which
landed the aarch64 SIMD block-width fix (BUG-02) and the two tests below.

**These are the first executions of `init_stream()`/`cal_stream()` anywhere in
this repository's history.** A full-repo grep finds exactly two non-definition
call sites, both in `backtest/test_strategy.py` and both commented out. Nothing
had ever exercised graph construction (`_make_stream`), buffer wiring
(`queryBufferHandle`) or the run loop (`pushData`/`run`/`getCurrentBuffer`) --
so this file is the streaming interface's only proof of life. Treat a failure
here as a real regression in the interface, not as a flaky test.

The first test is the 03-01 infrastructure self-test, kept unchanged: it proves
the shared `spot_kline_zarr` fixture delivers exactly the panel shape the
FACTOR-02 replay test consumes (8 symbols).

Import-safety rule (tests/conftest.py module docstring): nothing here may
import a module that does not exist yet at module level.
"""

from pathlib import Path
from typing import Callable

import numpy as np
import xarray as xr

from base.config import DatasetConfig, FactorConfig
from dataset.spot import SpotKlineDataset
from factor.alpha158 import Alpha158SpotKline

# The `data_columns`/`factor_names` pairing below is LOAD-BEARING, not
# arbitrary. `init_stream()` calls `queryBufferHandle` for every name in
# `config.data_columns` AND every name in `config.factor_names`, but KunQuant
# prunes any declared input that no reachable `Output(...)` consumes. Measured
# during 03-05 planning: with these three factor names the compiled stream
# module exposes handles for `close`, `open`, `volume` and the three factors,
# but NOT for `high`, `low` or `amount` -- so a wider `data_columns` list makes
# `init_stream()` raise `RuntimeError: Cannot find the buffer name`. That is a
# property of the existing code (the production Alpha158 config uses the full
# factor set, which consumes all six inputs), not a defect this test works
# around.
_DATA_COLUMNS = ["open", "close", "volume"]
_FACTOR_NAMES = ["KMID", "VOLUME0", "STD5"]


def test_stream_fixture_provides_eight_symbols(
    spot_kline_zarr: Callable[..., DatasetConfig],
) -> None:
    """The default `spot_kline_zarr()` panel is 8 symbols wide and at least 40
    timestamps deep -- the shape the streaming replay test below requires
    (8 symbols matches the SIMD block width KunQuant selects for float on
    x86_64, and its stream `partition_factor` of 8; the depth must exceed the
    largest rolling factor window).
    """
    dataset_config = spot_kline_zarr()
    data = SpotKlineDataset(dataset_config).read().get_xarray_dataset()

    assert data.sizes["symbol"] == 8
    assert data.sizes["timestamp"] >= 40


def _stream_factor(
    spot_kline_zarr: Callable[..., DatasetConfig],
    tmp_path: Path,
    periods: int = 60,
) -> Alpha158SpotKline:
    """Build a stream-mode `Alpha158SpotKline` over a synthetic 8-symbol store.

    `init_stream()` sizes its `StreamContext` from `self.num_symbols`, which
    `FactorKunQuant` resolves in stream mode as
    `len(self.config.dataset.config.symbols)` -- so the `DatasetConfig` must
    carry a non-None `symbols` tuple. It is set BEFORE the `SpotKlineDataset`
    is constructed because `base/data.py:Dataset.config`'s setter calls
    `_reset_symbols()` (which reads the Zarr) whenever `symbols` is not None;
    the fixture guarantees the store is already on disk.
    """
    dataset_config = spot_kline_zarr(periods=periods, seed=0)
    dataset_config.symbols = tuple(f"S{i}USDT" for i in range(8))

    return Alpha158SpotKline(
        FactorConfig(
            window=10,
            dataset=SpotKlineDataset(dataset_config),
            mode="stream",
            data_columns=_DATA_COLUMNS,
            factor_names=_FACTOR_NAMES,
            file_path=str(tmp_path / "factors" / "stream.zarr"),
            njobs=4,
        )
    )


def test_init_stream_binds_a_buffer_handle_for_every_declared_name(
    spot_kline_zarr: Callable[..., DatasetConfig], tmp_path: Path
) -> None:
    """`init_stream()` populates a buffer handle for every `data_columns` name
    and every `factor_names` name.

    The first execution of `init_stream()` in this repository. It is also the
    assertion that catches the `RuntimeError: Cannot find the buffer name`
    trap directly: KunQuant prunes declared inputs that no selected output
    consumes, so a `data_columns` list wider than the chosen `factor_names`
    actually need makes this method raise rather than silently under-bind.
    """
    factor = _stream_factor(spot_kline_zarr, tmp_path, periods=30)

    factor.init_stream()

    handles = factor._buffer_name_to_id
    for name in _DATA_COLUMNS + _FACTOR_NAMES:
        assert name in handles, (
            f"init_stream() bound no buffer handle for '{name}'"
        )
    assert len(handles) == len(_DATA_COLUMNS) + len(_FACTOR_NAMES)


def test_cal_stream_replay_produces_incremental_factor_updates(
    spot_kline_zarr: Callable[..., DatasetConfig], tmp_path: Path
) -> None:
    """FACTOR-02 / ROADMAP Phase 3 Success Criterion 2: replaying a full
    historical panel through `cal_stream()` one bar at a time completes without
    error and yields genuinely incremental `(1, num_symbols)` factor updates.

    The first execution of `cal_stream()` in this repository -- every prior
    call site (`backtest/test_strategy.py`) is commented out, so before this
    test nothing had ever proven the streaming path runs at all.

    The per-bar dicts come from the real dataset adapter
    (`Dataset.to_kunquant()`) rather than a hand-rolled Binance-to-KunQuant
    rename, and from the SAME `Dataset` instance the factor holds, so the
    symbol ordering matches the stream context's.

    `get_features()` reflects only the MOST RECENT bar -- each `cal_stream()`
    call replaces the backend's data -- so the two `KMID` snapshots the
    incremental-update assertion compares are captured inside the loop.
    """
    factor = _stream_factor(spot_kline_zarr, tmp_path, periods=60)

    input_dict, symbols, timestamps = factor.config.dataset.to_kunquant(
        tuple(_DATA_COLUMNS)
    )
    symbol_list = list(symbols)
    num_steps = len(timestamps)
    assert num_steps == 60

    previous_kmid = None
    last_kmid = None

    for step in range(num_steps):
        bar = {
            column: np.ascontiguousarray(
                input_dict[column][step].astype(np.float32)
            )
            for column in _DATA_COLUMNS
        }
        # Nothing may raise at any step -- an exception here IS the failure.
        factor.cal_stream(bar, int(step), symbol_list)

        if step >= num_steps - 2:
            previous_kmid = last_kmid
            last_kmid = factor.get_features()["KMID"].values.copy()

    result = factor.get_features()

    assert isinstance(result, xr.Dataset)
    assert dict(result.sizes) == {"timestamp": 1, "symbol": 8}
    assert sorted(result.data_vars) == ["KMID", "STD5", "VOLUME0"]
    assert np.isfinite(result["KMID"].to_numpy()).sum() > 0

    # T-03-05-02: a stream that returned a stale or constant buffer would look
    # "successful" while producing garbage. Two consecutive snapshots must
    # differ for the updates to be genuinely incremental.
    assert previous_kmid is not None and last_kmid is not None
    assert not np.array_equal(previous_kmid, last_kmid), (
        "successive cal_stream() calls produced identical KMID values -- the "
        "stream is not updating incrementally"
    )
