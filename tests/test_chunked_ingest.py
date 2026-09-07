"""Tests for the chunked densify-and-append ingestion path (260906-13w).

The full `us_all` window (15,424 symbols x ~5,215 trading days) densifies to a
~7.2 GiB float64 grid that `StockDataset._raw_data_to_xr()` holds alongside the
~29.6M-row frame and conversion scratch -- an OOM on a 16 GiB machine. This
module pins the fix: densify and append ONE time window at a time, so peak RAM
scales with the WINDOW rather than the range (D-01).

Every test here is offline. No credential, no network, no real store beyond
`tmp_path`.

Two properties are load-bearing and are asserted structurally rather than by
sampling RSS:

- the symbol axis is computed ONCE over the whole range and every window is
  materialised on it (D-02) -- so `sizes["symbol"]` is the FULL pinned length
  in every window while `sizes["timestamp"]` is strictly shorter than the full
  axis;
- `_raw_data_to_xr()` is called ZERO times during a chunked run -- calling it
  even once would reintroduce the whole-range allocation the chunking exists
  to avoid.
"""

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from loguru import logger

from base.chunking import ChunkLedger, TimeChunkPlanner
from base.config import BaseDatasetConfig, DatasetConfig
from base.data import BaseDataset
from dataset.backend import XrBackend
from dataset.stock import StockDataset

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

#: Three calendar years of observed timestamps. Deliberately sparse *within*
#: each year (a handful of days, not a full session calendar) so a window is
#: cheap, and deliberately sparse *across symbols* so the pinned-axis reindex
#: is genuinely exercised: `B` trades only in 2023 and `C` only in 2024, so a
#: window that derived its own symbol axis would produce 1-2 columns instead of
#: the pinned 3 and the append would silently misalign.
_YEARS = (2022, 2023, 2024)
_DAYS_PER_YEAR = ("01-04", "06-15", "12-28")


def _raw_rows(stock_pqt_row: Callable[..., dict]) -> list[dict]:
    rows = []
    for year in _YEARS:
        for day in _DAYS_PER_YEAR:
            date_str = f"{year}-{day}"
            # Constant close on purpose: `flag_anomalies` flags a >50%
            # single-step jump, and a jump landing ON a chunk boundary is the
            # one documented value difference between the chunked and
            # unchunked paths. Keeping the series flat lets this module assert
            # the two stores are equal for EVERY variable, `anomaly_flag`
            # included, rather than having to carve out an exception.
            rows.append(stock_pqt_row(date_str, "A", close=100.0))
            if year == 2023:
                rows.append(stock_pqt_row(date_str, "B", close=100.0))
            if year == 2024:
                rows.append(stock_pqt_row(date_str, "C", close=100.0))
    return rows


@pytest.fixture
def three_year_stock_config(
    stock_pqt_row: Callable[..., dict],
    hive_raw_tree: Callable[..., Path],
    tmp_path: Path,
) -> Callable[..., DatasetConfig]:
    """Factory. `three_year_stock_config(store_name)` writes the 3-year raw
    parquet panel once and returns a `DatasetConfig` pointing at a fresh Zarr
    path, so one test can build both a chunked and an unchunked store from the
    SAME raw input.

    03.2 D-08/D-11 changed the raw tier's SHAPE, not this module's claims: raw
    parquet now lives in a hive-partitioned tree under a vendor-terminated root
    with a literal `vendor` column, so the panel is written via `hive_raw_tree`
    rather than as one flat `raw/all/data_1.pqt`. Every assertion in this file
    is untouched -- which is exactly the proof that the hive rework left
    `_scan_raw`'s output column set and the chunked/unchunked equivalence
    alone.
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


class _SpyStockDataset(StockDataset):
    """Records which densification seam was used, and the SHAPE of every
    window -- the memory-bound property stated deterministically.
    """

    def __init__(self, config: DatasetConfig):
        self.whole_range_calls = 0
        self.window_calls: list[tuple] = []
        self.window_sizes: list[dict] = []
        super().__init__(config)

    def _raw_data_to_xr(self) -> xr.Dataset:
        self.whole_range_calls += 1
        return super()._raw_data_to_xr()

    def _raw_data_to_xr_window(
        self, start_date, end_date, symbols: Optional[list[str]] = None
    ) -> xr.Dataset:
        window = super()._raw_data_to_xr_window(start_date, end_date, symbols)
        self.window_calls.append((pd.Timestamp(start_date), pd.Timestamp(end_date)))
        self.window_sizes.append(dict(window.sizes))
        return window


class _FailsOnSecondWindow(_SpyStockDataset):
    """Crashes inside window 2 of 3, the way a real interrupted run does."""

    def _raw_data_to_xr_window(
        self, start_date, end_date, symbols: Optional[list[str]] = None
    ) -> xr.Dataset:
        if len(self.window_calls) == 1:
            raise RuntimeError("simulated crash inside window 2")
        return super()._raw_data_to_xr_window(start_date, end_date, symbols)


def _captured_warnings():
    """Attach a temporary in-memory loguru sink.

    loguru does not propagate to stdlib `logging`, so pytest's `caplog` sees
    nothing (same constraint recorded in tests/test_universe.py and
    tests/test_constituent_panel.py).
    """
    messages: list[str] = []
    sink_id = logger.add(messages.append, level="WARNING", format="{message}")
    return messages, sink_id


def _panel(path: str) -> xr.Dataset:
    return xr.open_zarr(path).load()


# ---------------------------------------------------------------------------
# TimeChunkPlanner
# ---------------------------------------------------------------------------


def test_plan_from_timestamps_returns_one_window_per_observed_year() -> None:
    """Every window edge is an OBSERVED timestamp, never a calendar boundary
    with no row behind it. Handing naive calendar edges to a densifier would
    fabricate rows for days on which nothing traded.
    """
    axis = pd.to_datetime(
        [f"{year}-{day}" for year in _YEARS for day in _DAYS_PER_YEAR]
    )

    windows = TimeChunkPlanner("year").plan_from_timestamps(axis)

    assert len(windows) == 3
    assert [w[0].year for w in windows] == list(_YEARS)
    for start, end in windows:
        assert start in set(axis)
        assert end in set(axis)
        assert start <= end
    # Time-ordered and non-overlapping.
    for earlier, later in zip(windows, windows[1:]):
        assert earlier[1] < later[0]
    # The first window opens on the first observed day of 2022, not 2022-01-01.
    assert windows[0][0] == pd.Timestamp("2022-01-04")


def test_plan_from_timestamps_is_granularity_sensitive() -> None:
    axis = pd.to_datetime([f"2022-{day}" for day in _DAYS_PER_YEAR])

    assert len(TimeChunkPlanner("year").plan_from_timestamps(axis)) == 1
    assert len(TimeChunkPlanner("quarter").plan_from_timestamps(axis)) == 3
    assert len(TimeChunkPlanner("month").plan_from_timestamps(axis)) == 3


def test_plan_from_timestamps_rejects_an_empty_axis() -> None:
    with pytest.raises(ValueError) as excinfo:
        TimeChunkPlanner("year").plan_from_timestamps([])

    assert "TimeChunkPlanner" in str(excinfo.value)


def test_unknown_granularity_lists_the_accepted_values() -> None:
    with pytest.raises(ValueError) as excinfo:
        TimeChunkPlanner("fortnight")

    message = str(excinfo.value)
    assert "fortnight" in message
    for accepted in ("year", "quarter", "month"):
        assert accepted in message


def test_plan_calendar_and_plan_from_timestamps_share_one_period_rule() -> None:
    """`plan_calendar` is SIZING ONLY -- it runs before the download, when no
    timestamp axis exists. It must nonetheless agree with
    `plan_from_timestamps` about what "a year" is, or the guard would size a
    different set of windows than the writer materialises.
    """
    planner = TimeChunkPlanner("year")
    calendar = planner.plan_calendar("2022-01-04", "2024-12-28")
    observed = planner.plan_from_timestamps(
        pd.to_datetime([f"{year}-{day}" for year in _YEARS for day in _DAYS_PER_YEAR])
    )

    assert len(calendar) == len(observed) == 3
    # ISO strings, clipped to the requested range at both outer edges.
    assert calendar[0] == ("2022-01-04", "2022-12-31")
    assert calendar[-1] == ("2024-01-01", "2024-12-28")
    assert all(isinstance(edge, str) for window in calendar for edge in window)

    # And the shared period rule holds across granularities.
    assert len(TimeChunkPlanner("month").plan_calendar("2022-01-01", "2022-03-31")) == 3
    assert (
        len(TimeChunkPlanner("quarter").plan_calendar("2022-01-01", "2022-12-31")) == 4
    )


def test_plan_calendar_covers_the_full_backfill_range() -> None:
    """Plan verification step 2: 2006..2026 inclusive is 21 yearly windows."""
    windows = TimeChunkPlanner("year").plan_calendar("2006-01-01", "2026-09-06")

    assert len(windows) == 21


# ---------------------------------------------------------------------------
# XrBackend.append
# ---------------------------------------------------------------------------


def _small_panel(dates: list[str], symbols: list[str], offset: float) -> xr.Dataset:
    values = np.arange(len(dates) * len(symbols), dtype=float).reshape(
        len(dates), len(symbols)
    ) + offset
    return xr.Dataset(
        {"close": (["timestamp", "symbol"], values)},
        coords={"timestamp": pd.to_datetime(dates), "symbol": symbols},
    )


def test_append_creates_the_store_then_extends_the_timestamp_dimension(
    tmp_path: Path,
) -> None:
    path = str(tmp_path / "nested" / "appended.zarr")

    XrBackend().to_internal(
        _small_panel(["2022-01-04", "2022-06-15"], ["A", "B"], 0.0)
    ).append(path)
    XrBackend().to_internal(
        _small_panel(["2023-01-04", "2023-06-15", "2023-12-28"], ["A", "B"], 100.0)
    ).append(path)

    store = _panel(path)
    assert store.sizes["timestamp"] == 5
    assert store.sizes["symbol"] == 2
    assert store["symbol"].values.tolist() == ["A", "B"]
    assert store["close"].values[0].tolist() == [0.0, 1.0]
    assert store["close"].values[-1].tolist() == [104.0, 105.0]


def test_append_refuses_a_changed_symbol_axis(tmp_path: Path) -> None:
    """Zarr's own `mode="a"` append SILENTLY overwrites a non-append
    coordinate, so a window that derived its own symbol axis would corrupt the
    store with no error at all. The backend must raise instead (T-13w-01).
    """
    path = str(tmp_path / "appended.zarr")
    XrBackend().to_internal(
        _small_panel(["2022-01-04"], ["A", "B"], 0.0)
    ).append(path)

    with pytest.raises(ValueError) as excinfo:
        XrBackend().to_internal(
            _small_panel(["2023-01-04"], ["A", "Z"], 100.0)
        ).append(path)

    message = str(excinfo.value)
    assert "symbol" in message


def test_plain_append_still_refuses_a_labels_differ_axis_of_the_same_length(
    tmp_path: Path,
) -> None:
    """The count-unchanged case: one delisting plus one new listing leaves the
    symbol COUNT identical while the labels differ. Raw
    `to_zarr(mode="a", append_dim=...)` succeeds silently here and re-attributes
    every stored row -- measured 2026-09-06 as `rows 0-4 were written for XYZ
    but are now labelled: ARM`. A length check alone would not catch it.

    This is the regression guard for 260906-x2s's hard constraint: the widen
    (`XrBackend.widen_and_append`, pinned in
    `tests/test_symbol_axis_widening.py`) is a separately-named OPT-IN that
    satisfies this guard by construction. Plain `append()` must keep refusing
    for a caller who did not opt in.

    RED under: any relaxation of `_assert_append_compatible` -- comparing only
    `len(incoming) != len(stored)`, or short-circuiting when the counts match.
    """
    path = str(tmp_path / "labels_differ.zarr")
    XrBackend().to_internal(
        _small_panel(["2022-01-04"], ["A", "XYZ"], 0.0)
    ).append(path)

    with pytest.raises(ValueError) as excinfo:
        XrBackend().to_internal(
            _small_panel(["2023-01-04"], ["A", "ARM"], 100.0)
        ).append(path)

    message = str(excinfo.value)
    assert "symbol" in message
    # The store still holds its original labels and its original history.
    store = _panel(path)
    assert store["symbol"].values.tolist() == ["A", "XYZ"]
    assert store.sizes["timestamp"] == 1


def test_append_refuses_a_changed_dtype(tmp_path: Path) -> None:
    """An int64 store silently casts an appended float NaN to 0 -- a real
    value fabricated out of a missing one, with no error. Refuse instead.
    """
    path = str(tmp_path / "appended.zarr")
    first = xr.Dataset(
        {"volume": (["timestamp", "symbol"], np.array([[1, 2]], dtype="int64"))},
        coords={"timestamp": pd.to_datetime(["2022-01-04"]), "symbol": ["A", "B"]},
    )
    XrBackend().to_internal(first).append(path)

    second = xr.Dataset(
        {"volume": (["timestamp", "symbol"], np.array([[np.nan, 2.0]]))},
        coords={"timestamp": pd.to_datetime(["2023-01-04"]), "symbol": ["A", "B"]},
    )
    with pytest.raises(ValueError) as excinfo:
        XrBackend().to_internal(second).append(path)

    message = str(excinfo.value)
    assert "volume" in message
    assert "dtype" in message.lower()


# ---------------------------------------------------------------------------
# StockDataset windowed densification
# ---------------------------------------------------------------------------


def test_windowed_densification_returns_exactly_the_pinned_symbol_axis(
    three_year_stock_config: Callable[..., DatasetConfig],
) -> None:
    """`B` never trades in 2022 and `C` never trades before 2024, yet the 2022
    window must still carry all three columns -- all-NaN for the absent ones,
    which is precisely the value the whole-range densification already
    produces for an untraded cell.
    """
    dataset = StockDataset(three_year_stock_config())
    pinned, timestamps = dataset._raw_axes_in_range()

    assert pinned == ["A", "B", "C"]
    assert len(timestamps) == 9

    window = dataset._raw_data_to_xr_window(
        "2022-01-01", "2022-12-31", symbols=pinned
    )

    assert window["symbol"].values.tolist() == pinned
    assert window.sizes["timestamp"] == 3
    assert np.isnan(window["adjClose"].sel(symbol="B").values).all()
    assert np.isnan(window["adjClose"].sel(symbol="C").values).all()
    assert (window["adjClose"].sel(symbol="A").values == 100.0).all()


def test_whole_range_densification_equals_the_union_of_its_windows(
    three_year_stock_config: Callable[..., DatasetConfig],
) -> None:
    """The `_raw_data_to_xr()` refactor is behaviour-preserving: the full-range
    panel is exactly what concatenating the per-year windows produces.
    """
    dataset = StockDataset(three_year_stock_config())
    pinned, timestamps = dataset._raw_axes_in_range()

    whole = dataset._raw_data_to_xr()
    windows = [
        dataset._raw_data_to_xr_window(start, end, symbols=pinned)
        for start, end in TimeChunkPlanner("year").plan_from_timestamps(timestamps)
    ]
    stitched = xr.concat(windows, dim="timestamp")

    assert whole["symbol"].values.tolist() == pinned
    assert whole["timestamp"].values.tolist() == stitched["timestamp"].values.tolist()
    np.testing.assert_allclose(
        whole["adjClose"].values.astype(float),
        stitched["adjClose"].values.astype(float),
    )


# ---------------------------------------------------------------------------
# BaseDataset.from_raw_data_chunked
# ---------------------------------------------------------------------------


def test_chunked_run_densifies_each_window_once_and_never_the_whole_range(
    three_year_stock_config: Callable[..., DatasetConfig],
) -> None:
    """The memory-shape constraint, stated deterministically: every window is
    strictly shorter than the full timestamp axis while carrying the FULL
    pinned symbol axis, and the unbounded whole-range densifier is never
    reached.
    """
    dataset = _SpyStockDataset(three_year_stock_config())

    dataset.from_raw_data_chunked(granularity="year")

    assert dataset.whole_range_calls == 0
    assert len(dataset.window_calls) == 3
    for sizes in dataset.window_sizes:
        assert sizes["timestamp"] < 9
        assert sizes["symbol"] == 3


def test_chunked_store_matches_the_unchunked_store(
    three_year_stock_config: Callable[..., DatasetConfig],
) -> None:
    chunked_config = three_year_stock_config("chunked.zarr")
    unchunked_config = three_year_stock_config("unchunked.zarr")

    StockDataset(chunked_config).from_raw_data_chunked(granularity="year")
    StockDataset(unchunked_config).from_raw_data().save()

    chunked = _panel(chunked_config.zarr_file_path)
    unchunked = _panel(unchunked_config.zarr_file_path)

    assert (
        chunked["timestamp"].values.tolist()
        == unchunked["timestamp"].values.tolist()
    )
    assert chunked["symbol"].values.tolist() == unchunked["symbol"].values.tolist()
    assert set(chunked.data_vars) == set(unchunked.data_vars)
    for name in unchunked.data_vars:
        left = chunked[name].values
        right = unchunked[name].values
        if left.dtype == bool or right.dtype == bool:
            assert np.array_equal(left, right), name
        else:
            np.testing.assert_allclose(
                left.astype(float), right.astype(float), err_msg=name
            )


def test_chunked_run_warns_about_the_cleaning_boundaries(
    three_year_stock_config: Callable[..., DatasetConfig],
) -> None:
    """Cleaning per window means `flag_anomalies` has no prior sample at each
    chunk's opening timestamp. That is a bounded, documented consequence -- it
    must be announced, not silent.
    """
    messages, sink_id = _captured_warnings()
    try:
        StockDataset(three_year_stock_config()).from_raw_data_chunked(
            granularity="year"
        )
    finally:
        logger.remove(sink_id)

    boundary_warnings = [m for m in messages if "boundary" in m.lower()]
    assert boundary_warnings, messages
    # Three windows -> two interior boundaries (the very first timestamp of
    # the whole axis has no prior sample in the unchunked path either).
    assert "2" in boundary_warnings[0]


class _UnboundedDataset(BaseDataset):
    """A dataset kind whose raw source cannot push a date filter down.

    It implements only the one abstract member and inherits BOTH default
    seams, which is exactly the shape `BaseDataset`'s
    correct-but-unbounded defaults exist to serve: it must still work, and
    it must be told that chunking is bounding its write and not its
    densification.
    """

    def __init__(self, config: BaseDatasetConfig, panel: xr.Dataset):
        self._panel = panel
        super().__init__(config)

    def _raw_data_to_xr(self) -> xr.Dataset:
        return self._panel.copy(deep=True)


def _ohlcv_panel() -> xr.Dataset:
    dates = pd.to_datetime(
        [f"{year}-{day}" for year in _YEARS for day in _DAYS_PER_YEAR]
    )
    values = np.full((len(dates), 2), 100.0)
    return xr.Dataset(
        {
            name: (["timestamp", "symbol"], values.copy())
            for name in ("open", "high", "low", "close", "volume")
        },
        coords={"timestamp": dates, "symbol": ["A", "B"]},
    )


def test_default_windowed_seam_warns_that_chunking_bounds_only_the_write(
    tmp_path: Path,
) -> None:
    """A class that has NOT overridden `_raw_data_to_xr_window` still works --
    the default is correct -- but chunking then bounds the WRITE and not the
    DENSIFY, and the memory win is absent. Say so.
    """
    config = BaseDatasetConfig(zarr_file_path=str(tmp_path / "unbounded.zarr"))

    messages, sink_id = _captured_warnings()
    try:
        _UnboundedDataset(config, _ohlcv_panel()).from_raw_data_chunked()
    finally:
        logger.remove(sink_id)

    assert any("not been overridden" in m for m in messages), messages
    # The default is correct, not merely tolerated: the store is complete.
    assert _panel(config.zarr_file_path).sizes["timestamp"] == 9


def test_second_run_against_a_complete_store_densifies_zero_windows(
    three_year_stock_config: Callable[..., DatasetConfig],
) -> None:
    config = three_year_stock_config()
    StockDataset(config).from_raw_data_chunked(granularity="year")
    first_length = _panel(config.zarr_file_path).sizes["timestamp"]

    resumed = _SpyStockDataset(config)
    resumed.from_raw_data_chunked(granularity="year")

    assert resumed.window_calls == []
    assert resumed.whole_range_calls == 0
    assert _panel(config.zarr_file_path).sizes["timestamp"] == first_length


def test_a_crash_in_window_two_resumes_at_window_two(
    three_year_stock_config: Callable[..., DatasetConfig],
) -> None:
    config = three_year_stock_config()

    with pytest.raises(RuntimeError):
        _FailsOnSecondWindow(config).from_raw_data_chunked(granularity="year")

    ledger = ChunkLedger(ChunkLedger.default_path(config.zarr_file_path))
    assert len(ledger.windows) == 1
    assert _panel(config.zarr_file_path).sizes["timestamp"] == 3

    resumed = _SpyStockDataset(config)
    resumed.from_raw_data_chunked(granularity="year")

    assert len(resumed.window_calls) == 2
    assert [start.year for start, _ in resumed.window_calls] == [2023, 2024]
    assert _panel(config.zarr_file_path).sizes["timestamp"] == 9


def test_the_ledger_lives_beside_the_store_not_inside_it(
    three_year_stock_config: Callable[..., DatasetConfig],
) -> None:
    """A `mode="w"` rewrite of the store deletes the store DIRECTORY, so a
    ledger written inside it would vanish exactly when it is needed.
    """
    config = three_year_stock_config()
    StockDataset(config).from_raw_data_chunked(granularity="year")

    ledger_path = Path(ChunkLedger.default_path(config.zarr_file_path))
    assert ledger_path.exists()
    assert ledger_path.parent == Path(config.zarr_file_path).parent
    assert not str(ledger_path).startswith(config.zarr_file_path + "/")

    ledger = ChunkLedger(str(ledger_path))
    assert len(ledger.windows) == 3
    assert sum(w["rows"] for w in ledger.windows) == 9
    assert ledger.symbol_count == 3
    assert ledger.symbol_fingerprint == ChunkLedger.fingerprint(["A", "B", "C"])


# ---------------------------------------------------------------------------
# Ledger integrity on resume (260906-13w Task 2, D-04 / T-13w-02)
#
# The ledger and the store are two independent records of the same truth,
# written at two different instants. A resume trusts NEITHER alone: it
# cross-checks them and raises on disagreement, because an append is
# irreversible and cannot be validated after the fact from the store alone.
# ---------------------------------------------------------------------------


def test_resume_with_a_changed_roster_raises_naming_both_symbol_counts(
    three_year_stock_config: Callable[..., DatasetConfig],
    stock_pqt_row: Callable[..., dict],
    hive_raw_tree: Callable[..., Path],
) -> None:
    config = three_year_stock_config()
    StockDataset(config).from_raw_data_chunked(granularity="year")

    # A roster refresh between two runs: a fourth symbol appears, so the
    # pinned axis is no longer the axis the store was written on. Written as a
    # second shard under the SAME vendor root -- a new symbol arrives as a new
    # batch, not as a new vendor.
    hive_raw_tree(
        Path(config.raw_data_dir_path).parent,
        "tiingo",
        [stock_pqt_row("2024-06-15", "D", close=100.0)],
        batch_key="extra",
    )

    with pytest.raises(ValueError) as excinfo:
        StockDataset(config).from_raw_data_chunked(granularity="year")

    message = str(excinfo.value)
    assert "3" in message and "4" in message  # both symbol counts
    assert "roster" in message.lower()


def test_resume_with_a_store_ledger_tail_mismatch_raises_naming_both_dates(
    three_year_stock_config: Callable[..., DatasetConfig],
) -> None:
    """A crash BETWEEN a successful `to_zarr` and the ledger write leaves the
    store one window ahead. Re-running would duplicate that window; refuse.
    """
    import json

    config = three_year_stock_config()
    StockDataset(config).from_raw_data_chunked(granularity="year")

    ledger_path = ChunkLedger.default_path(config.zarr_file_path)
    with open(ledger_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["windows"] = payload["windows"][:-1]
    with open(ledger_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)

    with pytest.raises(ValueError) as excinfo:
        StockDataset(config).from_raw_data_chunked(granularity="year")

    message = str(excinfo.value)
    assert "2024-12-28" in message  # the store's tail
    assert "2023-12-28" in message  # the ledger's last recorded end


def test_a_store_with_no_ledger_raises_rather_than_appending_blind(
    three_year_stock_config: Callable[..., DatasetConfig],
) -> None:
    config = three_year_stock_config()
    StockDataset(config).from_raw_data_chunked(granularity="year")
    Path(ChunkLedger.default_path(config.zarr_file_path)).unlink()

    with pytest.raises(ValueError) as excinfo:
        StockDataset(config).from_raw_data_chunked(granularity="year")

    message = str(excinfo.value)
    assert "ledger" in message.lower()


def test_an_absent_store_with_an_empty_ledger_is_the_normal_first_run(
    three_year_stock_config: Callable[..., DatasetConfig],
) -> None:
    config = three_year_stock_config()
    assert not Path(config.zarr_file_path).exists()

    StockDataset(config).from_raw_data_chunked(granularity="year")

    assert _panel(config.zarr_file_path).sizes["timestamp"] == 9
