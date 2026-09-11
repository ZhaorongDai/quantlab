"""`quantlab.acquisition.registry.convert()` -- the registry-level raw->Zarr
entry point (DATA-07, phase 03.5).

The requirement these tests exist for is stated as a CALL SITE, not as a
behaviour: an in-process caller holding only a `SourceDescriptor` and a
`DatasetConfig` converts an already-persisted raw parquet tier into a Zarr
store, without naming `StockDataset`, `TiingoAcquisition` or any `ingest_*.py`
anywhere. Every test below therefore reaches the conversion through
`convert(DESCRIPTOR, config)` and never through the Dataset class -- a test
that constructed the dataset itself would pass while the requirement failed.

Offline, all of it. No credential is read, no network call is made, and every
path is under `tmp_path`. The raw tier is a hive-partitioned parquet tree
written by the shared `hive_raw_tree` fixture, in the shape
`StockDataset._scan_raw` asserts: `Path(raw_data_dir_path).name == vendor`.
"""

from pathlib import Path
from typing import Callable

import pytest
import xarray as xr

from quantlab.base.config import DatasetConfig
from quantlab.base.data import ConversionResult
from quantlab.acquisition.registry import DataSourceRegistry, convert

#: Two calendar years, a handful of observed days in each, so a `year` window
#: is cheap and there are exactly TWO planned windows to write, skip and count.
#: Sparse across symbols on purpose (`B` trades only in 2024): a window that
#: derived its own symbol axis would come back with one column instead of the
#: pinned two, which is the misalignment `from_raw_data_chunked`'s pinned axis
#: exists to prevent -- and which `pinned_symbols` in the result reports.
_YEARS = (2023, 2024)
_DAYS_PER_YEAR = ("01-04", "06-15", "12-28")
_EXPECTED_WINDOWS = len(_YEARS)
_EXPECTED_SYMBOLS = 2


def _raw_rows(stock_pqt_row: Callable[..., dict]) -> list[dict]:
    rows = []
    for year in _YEARS:
        for day in _DAYS_PER_YEAR:
            date_str = f"{year}-{day}"
            rows.append(stock_pqt_row(date_str, "A", close=100.0))
            if year == 2024:
                rows.append(stock_pqt_row(date_str, "B", close=100.0))
    return rows


@pytest.fixture
def tiingo_raw_tier(
    stock_pqt_row: Callable[..., dict],
    hive_raw_tree: Callable[..., Path],
    tmp_path: Path,
) -> Callable[..., DatasetConfig]:
    """Factory. `tiingo_raw_tier()` writes the two-year raw parquet panel once
    and returns a `DatasetConfig` a caller could hand straight to `convert()`.

    Calling it twice returns configs pointing at the SAME raw tree and the SAME
    Zarr path, which is what makes the idempotency probe a genuine second run
    over the first run's store rather than a fresh one.
    """
    raw_dir = tmp_path / "raw"
    hive_raw_tree(raw_dir, "tiingo", _raw_rows(stock_pqt_row), batch_key="panel")

    def _build(store_name: str = "out.zarr") -> DatasetConfig:
        return DatasetConfig(
            raw_data_dir_path=str(raw_dir / "tiingo"),
            zarr_file_path=str(tmp_path / store_name),
            catalog_path=str(tmp_path / "catalog"),
            market="us_equity",
            frequency="1d",
            vendor="tiingo",
        )

    return _build


def test_convert_writes_a_real_zarr_store_and_returns_a_conversion_result(
    tiingo_raw_tier: Callable[..., DatasetConfig],
) -> None:
    """DATA-07's tracer: descriptor + config in, Zarr store on disk out.

    The assertions are deliberately split between the STORE and the RESULT.
    Asserting only the returned object would pass against a `convert()` that
    computed counts and wrote nothing; asserting only the store would pass
    against one that wrote but reported nothing a caller could render (D-04).
    Both halves are the requirement.
    """
    descriptor = DataSourceRegistry.get("tiingo")
    config = tiingo_raw_tier()

    result = convert(descriptor, config)

    # -- the store is real --------------------------------------------------
    assert Path(config.zarr_file_path).exists()
    stored = xr.open_zarr(config.zarr_file_path)
    assert set(stored.dims) >= {"timestamp", "symbol"}
    assert stored.sizes["symbol"] == _EXPECTED_SYMBOLS
    assert stored.sizes["timestamp"] == len(_YEARS) * len(_DAYS_PER_YEAR)

    # -- the outcome is renderable without reading that store ---------------
    assert isinstance(result, ConversionResult)
    assert result.zarr_path == config.zarr_file_path
    assert Path(result.ledger_path).exists()
    assert result.granularity == "year"
    assert result.pinned_symbols == _EXPECTED_SYMBOLS
    assert result.windows_planned == _EXPECTED_WINDOWS
    assert result.windows_written == _EXPECTED_WINDOWS
    assert result.windows_skipped == 0
    assert result.rows_written == len(_YEARS) * len(_DAYS_PER_YEAR)
    assert result.peak_window_bytes is not None and result.peak_window_bytes > 0
    assert result.resumed is False
    assert result.cancelled is False
    # D-11: `convert()` runs no guard and invents no estimate. `None` here is
    # the honest answer when the caller offered none, not a missing value.
    assert result.predicted_peak_bytes is None


def test_a_second_convert_over_the_same_window_writes_nothing_twice(
    tiingo_raw_tier: Callable[..., DatasetConfig],
) -> None:
    """DATA-07's idempotency probe row, as an assertion rather than as prose.

    A second call over the same config must not re-append a single window --
    re-appending would duplicate every timestamp in the store, which is the
    silent corruption the chunk ledger exists to make impossible. The proof is
    that the second run's SKIPPED count equals the first run's WRITTEN count
    and its own written count is zero.
    """
    descriptor = DataSourceRegistry.get("tiingo")
    first = convert(descriptor, tiingo_raw_tier())

    second = convert(descriptor, tiingo_raw_tier())

    assert second.windows_written == 0
    assert second.windows_skipped == first.windows_written
    assert second.resumed is True
    assert second.rows_written == 0
    # Nothing was materialised, so there is no observed peak to report.
    assert second.peak_window_bytes is None

    # The store itself is unchanged -- the counts above would also be produced
    # by a run that skipped the ledger check and appended anyway if the ledger
    # were the only thing consulted.
    stored = xr.open_zarr(first.zarr_path)
    assert stored.sizes["timestamp"] == len(_YEARS) * len(_DAYS_PER_YEAR)
    assert stored.sizes["symbol"] == _EXPECTED_SYMBOLS


def test_predicted_peak_is_echoed_back_untouched(
    tiingo_raw_tier: Callable[..., DatasetConfig],
) -> None:
    """`predicted_peak_bytes` is the CALLER'S number, carried not computed.

    D-11 leaves the RAM guard at the call site, so the entry point has no
    roster category to size against. Echoing the caller's own estimate into
    the result is what lets a report put prediction beside outcome without
    `convert()` pretending to be self-protecting.
    """
    result = convert(
        DataSourceRegistry.get("tiingo"),
        tiingo_raw_tier(),
        predicted_peak_bytes=123_456,
    )

    assert result.predicted_peak_bytes == 123_456
    # The echo must not disturb anything that was measured.
    assert result.windows_written == _EXPECTED_WINDOWS


def test_run_still_returns_an_acquisition_result_and_converts_nothing() -> None:
    """SC-2's surviving first half: conversion is a SEPARATE function.

    Pinned structurally rather than by running an acquisition (which would
    need a credential): `run` and `convert` are two module-level functions
    with two different return annotations, and `run` grew no conversion
    keyword. A `mode=`/`to_zarr=` flag appearing on `run` is exactly the
    regression 03.4 D-14 forbids, and it would be invisible to every
    behavioural test in this file.
    """
    import inspect

    from quantlab.acquisition import registry
    from quantlab.base.acquisition import AcquisitionResult

    assert registry.run.__annotations__["return"] is AcquisitionResult
    assert registry.convert.__annotations__["return"] is ConversionResult

    run_params = set(inspect.signature(registry.run).parameters)
    assert not run_params & {"mode", "to_zarr", "convert", "granularity"}


# ---------------------------------------------------------------------------
# The SECOND vendor, and the three ways a capability lookup can refuse.
#
# Every refusal below is `ValueError` and every assertion is on the MESSAGE,
# not merely on the type. Asserting the type alone would pass against a single
# generic raise, which is exactly the shape these three arms replace: a caller
# who cannot tell "you asked for a combination nobody serves" from "you asked
# for one nobody can convert yet" has to read this module's source to act.
# ---------------------------------------------------------------------------


@pytest.fixture
def alpaca_raw_tier(
    stock_pqt_row: Callable[..., dict],
    hive_raw_tree: Callable[..., Path],
    tmp_path: Path,
) -> Callable[..., DatasetConfig]:
    """The same two-year panel under an `alpaca`-terminated raw root.

    `StockDataset._scan_raw` asserts `Path(raw_data_dir_path).name == vendor`
    and then asserts the literal `vendor` column agrees, so the vendor token
    has to be threaded through both the path and the config -- which is also
    the reason this cannot reuse `tiingo_raw_tier` with a renamed config.
    """
    raw_dir = tmp_path / "raw"
    hive_raw_tree(raw_dir, "alpaca", _raw_rows(stock_pqt_row), batch_key="panel")

    def _build(frequency: str = "1d", store_name: str = "alpaca.zarr") -> DatasetConfig:
        return DatasetConfig(
            raw_data_dir_path=str(raw_dir / "alpaca"),
            zarr_file_path=str(tmp_path / store_name),
            catalog_path=str(tmp_path / "catalog"),
            market="us_equity",
            frequency=frequency,  # type: ignore[arg-type]
            vendor="alpaca",
        )

    return _build


def test_convert_reaches_the_second_vendor_through_the_same_entry_point(
    alpaca_raw_tier: Callable[..., DatasetConfig],
) -> None:
    """The tracer's sideways expansion: a SECOND descriptor, same call shape.

    `StockDataset` answering for both vendors is the D-01 claim made concrete
    -- capabilities are keyed by `(market, frequency, data_type)`, which is the
    axis Dataset subclasses divide on, so one class on three rows across two
    vendors is one correct answer to three questions rather than a duplicated
    decision.
    """
    result = convert(DataSourceRegistry.get("alpaca"), alpaca_raw_tier())

    assert isinstance(result, ConversionResult)
    assert result.windows_written == _EXPECTED_WINDOWS
    assert result.pinned_symbols == _EXPECTED_SYMBOLS
    assert Path(result.zarr_path).exists()


def test_tick_is_refused_because_no_capability_carries_a_conversion_target(
    alpaca_raw_tier: Callable[..., DatasetConfig],
) -> None:
    """SC-7: the tick refusal names phase 03.3 and the irregular event axis.

    Asserted on the SUBSTRINGS rather than on `ValueError` alone, deliberately.
    The type alone is satisfied by any raise at all -- including the
    placeholder this arm replaced -- so a message assertion is the only thing
    that distinguishes "refused for the right reason" from "blew up".

    And no store is created: a refusal that had already written a
    partially-flattened panel would leave behind exactly the
    plausible-looking-but-wrong artefact the refusal exists to prevent.
    """
    config = alpaca_raw_tier(frequency="tick", store_name="tick.zarr")

    with pytest.raises(ValueError) as excinfo:
        convert(DataSourceRegistry.get("alpaca"), config)

    message = str(excinfo.value)
    assert "03.3" in message
    assert "irregular event" in message
    assert not Path(config.zarr_file_path).exists()


def test_the_tick_lookup_MATCHES_and_the_refusal_is_the_absent_target() -> None:
    """The refusal is DATA, not a failed lookup (D-01).

    This is the assertion that stops the SC-7 arm from being re-implemented as
    `if frequency == "tick"` one day: `capabilities_for` must still RETURN the
    tick rows -- the vendor genuinely serves them, and the registry must keep
    advertising what it serves -- while every one of them carries
    `dataset_cls is None`. Phase 03.3 lands tick conversion by FILLING those
    two fields; nothing in `convert()` has to change.
    """
    descriptor = DataSourceRegistry.get("alpaca")

    tick_rows = descriptor.capabilities_for("us_equity", "tick")

    assert len(tick_rows) == 2
    assert {row.data_type for row in tick_rows} == {"quotes", "trades"}
    assert all(row.dataset_cls is None for row in tick_rows)
    # ...and the lookup did not fail: `supports()` still says yes.
    assert descriptor.supports("us_equity", "tick")


def test_an_unserved_combination_names_the_request_and_what_is_served(
    tiingo_raw_tier: Callable[..., DatasetConfig],
) -> None:
    """NO MATCH: the message names the requested tuple AND the served ones.

    A caller who mistyped one element learns which from the same message,
    rather than getting a bare "not supported" about a registry they cannot
    see -- the same courtesy `DataSourceRegistry.get()` already extends for a
    mistyped vendor token.
    """
    descriptor = DataSourceRegistry.get("tiingo")
    config = tiingo_raw_tier()
    config.frequency = "1m"  # type: ignore[assignment]

    with pytest.raises(ValueError) as excinfo:
        convert(descriptor, config)

    message = str(excinfo.value)
    assert "1m" in message
    assert "us_equity" in message
    # What it DOES serve, enumerated from `capabilities` rather than restated.
    assert "'1d'" in message or '"1d"' in message


def test_two_matching_targets_without_a_data_type_refuse_rather_than_guess(
    isolated_registry,
    tiingo_raw_tier: Callable[..., DatasetConfig],
) -> None:
    """AMBIGUOUS: two rows, two targets, no `data_type` -> refuse.

    This is the case D-02's whole argument is about. No shipped vendor has this
    shape today, which is precisely why it needs a test rather than a comment:
    the day one does, `convert()` must refuse instead of letting capability
    ORDER decide what gets converted.

    Built through `isolated_registry`, which restores
    `DataSourceRegistry.SOURCES` by rebinding the saved tuple on teardown, so
    the synthetic descriptor cannot leak into a later test.
    """
    from quantlab.acquisition import tiingo as tiingo_module
    from quantlab.acquisition.registry import (
        Capability,
        SourceDescriptor,
        register_source,
    )
    from quantlab.dataset.spot import SpotKlineDataset
    from quantlab.dataset.stock import StockDataset

    register_source(
        SourceDescriptor(
            vendor="twotarget",  # type: ignore[arg-type]
            display_name="Two Target Vendor",
            acquisition_cls=tiingo_module.TiingoAcquisition,
            config_factory=lambda **kw: None,  # type: ignore[arg-type,return-value]
            capabilities=(
                Capability(
                    market="us_equity",
                    frequency="1d",
                    data_type="bars",
                    dataset_cls=StockDataset,
                ),
                Capability(
                    market="us_equity",
                    frequency="1d",
                    data_type="klines",
                    dataset_cls=SpotKlineDataset,
                ),
            ),
            required_env=(),
        )
    )
    descriptor = isolated_registry.get("twotarget")

    with pytest.raises(ValueError) as excinfo:
        convert(descriptor, tiingo_raw_tier(store_name="ambiguous.zarr"))

    message = str(excinfo.value)
    assert "data_type" in message
    assert "bars" in message and "klines" in message

    # Supplying the discriminator resolves it -- the refusal is about the
    # MISSING argument, not about the vendor being unusable.
    assert descriptor.capabilities_for("us_equity", "1d", "bars")[
        0
    ].dataset_cls is StockDataset


def test_alpaca_rows_carry_targets_on_bars_and_none_on_tick() -> None:
    """The four Alpaca rows, pinned as DATA.

    `dataset_cls` on exactly the two `bars` rows and `None` on exactly the two
    `tick` rows is the whole SC-7 mechanism; a runtime check on the rows is
    what keeps a future edit from filling a tick row before Phase 03.3 has
    landed a Dataset that can express the axis.
    """
    from quantlab.dataset.stock import StockDataset

    capabilities = DataSourceRegistry.get("alpaca").capabilities

    bars = [c for c in capabilities if c.data_type == "bars"]
    tick = [c for c in capabilities if c.frequency == "tick"]

    assert len(bars) == 2
    assert all(c.dataset_cls is StockDataset for c in bars)
    assert len(tick) == 2
    assert all(c.dataset_cls is None for c in tick)
