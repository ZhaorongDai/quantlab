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

import inspect
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from loguru import logger

from quantlab.base.chunking import ChunkLedger, TimeChunkPlanner
from quantlab.base.config import BaseDatasetConfig, DatasetConfig
from quantlab.base.data import BaseDataset
from quantlab.base.progress import CancelToken, ProgressEvent, ProgressReporter
from quantlab.dataset.backend import XrBackend
from quantlab.dataset.stock import StockDataset

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


#: The consequence clause the refusal message must state, asserted IDENTICALLY
#: by all three overlap-refusal tests below. The three fixtures break the time
#: axis in provably different ways -- measured 2026-09-07 on the unguarded
#: tree, a partial overlap gives is_unique False / is_monotonic_increasing
#: False, a window starting exactly on the stored end gives is_unique False
#: with is_monotonic_increasing still TRUE, and a window ending before the
#: stored start gives is_unique TRUE with is_monotonic_increasing False. The
#: ONE property all three break is that the axis is no longer STRICTLY
#: increasing, so that is the only honest wording. Asserting the same clause in
#: all three places is what stops a message true of a single shape from
#: surviving by being checked only where it happens to hold.
_OVERLAP_CONSEQUENCE = (
    "no longer STRICTLY increasing -- duplicate labels, out-of-order labels, "
    "or both"
)


def test_append_refuses_a_window_overlapping_the_stored_timestamps(
    tmp_path: Path,
) -> None:
    """`_assert_append_compatible` guarded every dimension EXCEPT the one being
    appended along, so nothing compared an incoming window's timestamps against
    what the store already held. Measured 2026-09-07 on the unguarded tree: a
    store on 2022-01-04..2022-01-06 taking a 2022-01-05..2022-01-07 window
    returns cleanly and comes back holding
    `[01-04, 01-05, 01-06, 01-05, 01-06, 01-07]` -- is_unique False,
    is_monotonic_increasing False. The damage then surfaces far from its cause:
    `.sel(timestamp=slice(...))` raises `KeyError: 'Value based partial slicing
    on non-monotonic DatetimeIndexes with non-existing keys is not allowed.'`,
    a `to_xarray` round-trip raises `cannot convert a DataFrame with a
    non-unique MultiIndex into xarray`, and a point `.sel()` on a duplicated
    date quietly returns TWO rows where the caller expects one.

    The incoming window rides the SAME symbol axis and the same dtypes, so
    neither sibling guard can fire and this test cannot be green for their
    reason.

    RED under: removing the append-dim check from
    `quantlab/dataset/backend.py::XrBackend._assert_append_compatible`
    (mutation M1).
    """
    path = str(tmp_path / "overlap.zarr")
    XrBackend().to_internal(
        _small_panel(["2022-01-04", "2022-01-05", "2022-01-06"], ["A", "B"], 0.0)
    ).append(path)
    before = _panel(path)["close"].values.copy()

    with pytest.raises(ValueError) as excinfo:
        XrBackend().to_internal(
            _small_panel(
                ["2022-01-05", "2022-01-06", "2022-01-07"], ["A", "B"], 100.0
            )
        ).append(path)

    message = str(excinfo.value)
    assert path in message
    assert "timestamp" in message
    assert "2022-01-06T00:00:00" in message  # the stored end
    assert "2022-01-05T00:00:00" in message  # the incoming start
    assert 'save(mode="w")' in message
    assert _OVERLAP_CONSEQUENCE in message

    # Nothing was written: the store is bit-identical to before the refusal.
    store = _panel(path)
    index = pd.DatetimeIndex(store["timestamp"].values)
    assert len(index) == 3
    assert index.is_unique
    assert index.is_monotonic_increasing
    assert store["close"].values.tolist() == before.tolist()


def test_append_allows_a_gap_between_the_stored_end_and_the_incoming_start(
    tmp_path: Path,
) -> None:
    """Gaps are deliberately NOT guarded (D-01, decided 2026-09-07). Only
    OVERLAP is refused. A discontinuous time axis is a legitimate shape this
    layer takes no position on -- the storage layer cannot tell a deliberately
    sparse range from a missing one -- so a window starting strictly after the
    stored end appends normally, whatever the distance.

    This test exists so a later reader cannot "finish the job" by extending
    the guard to contiguity: doing so reddens here, which is the point.

    RED under: extending the append-dim check in
    `quantlab/dataset/backend.py::XrBackend._assert_append_compatible` to also
    refuse a gap (mutation M5).
    """
    path = str(tmp_path / "gap.zarr")
    XrBackend().to_internal(
        _small_panel(["2022-01-04", "2022-01-05"], ["A", "B"], 0.0)
    ).append(path)

    XrBackend().to_internal(
        _small_panel(["2023-06-01"], ["A", "B"], 100.0)
    ).append(path)

    store = _panel(path)
    index = pd.DatetimeIndex(store["timestamp"].values)
    assert len(index) == 3
    assert index.is_unique
    assert index.is_monotonic_increasing
    assert store["close"].values[-1].tolist() == [100.0, 101.0]


def test_append_refuses_a_window_that_ends_before_the_stored_start(
    tmp_path: Path,
) -> None:
    """The backwards window: it ends BEFORE the store even begins, so it
    shares no label with the store at all. Measured 2026-09-07 on the unguarded
    tree -- store 2022-06-01/2022-06-02 taking a 2022-01-04 window comes back
    `[06-01, 06-02, 01-04]` with **is_unique TRUE** (nothing repeats) and
    **is_monotonic_increasing False**, and the same
    `.sel(timestamp=slice(...))` `KeyError: 'Value based partial slicing on
    non-monotonic DatetimeIndexes with non-existing keys is not allowed.'`
    follows.

    That is_unique TRUE measurement is why this fixture matters beyond
    coverage: it FALSIFIES any refusal message whose stated consequence is
    "duplicate labels". Nothing is duplicated here, and a reader who cannot
    see the measurement has no way to check the claim -- which is why the same
    `_OVERLAP_CONSEQUENCE` clause is asserted here as in the two other refusal
    tests.

    Refusing this shape is the direct consequence of comparing the incoming
    START against the stored END, which is the comparison the message
    describes.

    RED under: comparing the incoming start against the stored MINIMUM instead
    of its maximum (mutation M3 family).
    """
    path = str(tmp_path / "backwards.zarr")
    XrBackend().to_internal(
        _small_panel(["2022-06-01", "2022-06-02"], ["A", "B"], 0.0)
    ).append(path)
    before = _panel(path)["close"].values.copy()

    with pytest.raises(ValueError) as excinfo:
        XrBackend().to_internal(
            _small_panel(["2022-01-04"], ["A", "B"], 100.0)
        ).append(path)

    message = str(excinfo.value)
    assert path in message
    assert "timestamp" in message
    assert "2022-06-02T00:00:00" in message  # the stored end
    assert "2022-01-04T00:00:00" in message  # the incoming start
    assert 'save(mode="w")' in message
    assert _OVERLAP_CONSEQUENCE in message

    store = _panel(path)
    index = pd.DatetimeIndex(store["timestamp"].values)
    assert len(index) == 2
    assert index.is_unique
    assert index.is_monotonic_increasing
    assert store["close"].values.tolist() == before.tolist()


def test_append_refuses_a_window_starting_exactly_on_the_stored_end(
    tmp_path: Path,
) -> None:
    """The boundary case, and the ONLY test that reddens when the comparison
    is relaxed from `<=` to `<`. Record that plainly: a mechanism covered by
    exactly one test is a weaker guarantee than it looks, so this test must
    not be deleted as redundant with the partial-overlap one.

    Measured 2026-09-07 on the unguarded tree, this shape runs OPPOSITE to the
    ends-before fixture next door: store 2022-01-04..2022-01-06 taking a
    2022-01-06..2022-01-08 window comes back `[01-04, 01-05, 01-06, 01-06,
    01-07, 01-08]` with **is_unique False** but **is_monotonic_increasing
    still TRUE**. So the two fixtures break the axis in opposite ways -- one
    duplicates without disordering, the other disorders without duplicating --
    and only a consequence true of BOTH survives the shared
    `_OVERLAP_CONSEQUENCE` assertion the three refusal tests share.

    RED under: relaxing the comparison to strictly-less-than (mutation M2).
    """
    path = str(tmp_path / "boundary.zarr")
    XrBackend().to_internal(
        _small_panel(["2022-01-04", "2022-01-05", "2022-01-06"], ["A", "B"], 0.0)
    ).append(path)
    before = _panel(path)["close"].values.copy()

    with pytest.raises(ValueError) as excinfo:
        XrBackend().to_internal(
            _small_panel(
                ["2022-01-06", "2022-01-07", "2022-01-08"], ["A", "B"], 100.0
            )
        ).append(path)

    message = str(excinfo.value)
    assert path in message
    assert "timestamp" in message
    assert "2022-01-06T00:00:00" in message  # both the stored end AND the
    # incoming start -- that coincidence IS this fixture
    assert 'save(mode="w")' in message
    assert _OVERLAP_CONSEQUENCE in message

    store = _panel(path)
    index = pd.DatetimeIndex(store["timestamp"].values)
    assert len(index) == 3
    assert index.is_unique
    assert index.is_monotonic_increasing
    assert store["close"].values.tolist() == before.tolist()


def test_append_skips_the_overlap_check_without_an_append_dim_coordinate(
    tmp_path: Path,
) -> None:
    """A store can legitimately carry NO coordinate on the append dimension.
    Measured 2026-09-07: a panel with a `timestamp` DIM but no `timestamp`
    COORD writes and re-appends cleanly (n=2 then n=4), and it did so before
    this guard existed.

    The new check therefore skips that case exactly the way the non-append dim
    loop above it already skips a coordinate absent on either side. Without
    the skip there is nothing to take a `.min()` of and a working path becomes
    a crash -- a guard that over-refuses gets deleted rather than obeyed
    (T-uac-05).

    RED under: dropping the `append_dim in self.data.coords and append_dim in
    existing.coords` presence check from
    `quantlab/dataset/backend.py::XrBackend._assert_append_compatible`.
    """

    def _no_coord_panel(n: int, offset: float) -> xr.Dataset:
        return xr.Dataset(
            {
                "close": (
                    ["timestamp", "symbol"],
                    np.arange(n * 2, dtype=float).reshape(n, 2) + offset,
                )
            },
            coords={"symbol": ["A", "B"]},
        )

    path = str(tmp_path / "no_time_coord.zarr")
    XrBackend().to_internal(_no_coord_panel(2, 0.0)).append(path)
    XrBackend().to_internal(_no_coord_panel(2, 100.0)).append(path)

    store = _panel(path)
    assert store.sizes["timestamp"] == 4
    assert "timestamp" not in store.coords
    assert store["close"].values[-1].tolist() == [102.0, 103.0]


def test_append_offers_no_overwrite_escape_hatch() -> None:
    """The DECLARED half of D-02 -- and only that half. Read the next test
    with this one; neither is sufficient alone.

    D-02 (decided 2026-09-07): the refusal is unconditional, with no overwrite
    opt-out now or later. Recomputing an already-stored range is
    `save(mode="w")`'s job -- replace the store -- while `append()` extends
    it. A flag on `append()` would blur exactly the boundary those two methods
    are separate in order to keep sharp.

    **This assertion is a structural PROXY that does NOT span the property it
    stands for, and that is measured rather than suspected.** On 2026-09-07 a
    hatch popped from `**kwargs` INSIDE the method body -- `if
    kwargs.pop("force", False): ... return self`, inserted ahead of the guard
    call -- made the existing symbol-axis guard's `ValueError` disappear
    entirely while `inspect.signature` still returned this exact, byte-
    identical parameter tuple. So on its own this test stays green through
    precisely the drift D-02 exists to catch. The behavioural half lives in
    `test_append_refuses_an_overlapping_window_carrying_an_unrecognised_kwarg`
    directly below.

    RED under: adding a NAMED parameter that lets a caller past the refusal
    (`force=`, `overwrite=`, `mode=`). Deliberately NOT red under a smuggled
    one -- that is the next test's job, and mutation M6 demonstrates the split.
    """
    parameters = tuple(inspect.signature(XrBackend.append).parameters)
    assert parameters == ("self", "path", "append_dim", "kwargs")


def test_append_refuses_an_overlapping_window_carrying_an_unrecognised_kwarg(
    tmp_path: Path,
) -> None:
    """The BEHAVIOURAL half of D-02, and the half that spans the realistic
    drift. Read it with `test_append_offers_no_overwrite_escape_hatch`
    directly above, whose signature assertion measurably does NOT cover this.

    The mechanism, which is the reason this passes: `XrBackend.append` calls
    `_assert_append_compatible(path, append_dim)` BEFORE it touches `kwargs`
    at all -- before the `kwargs.pop("encoding", None)` and before any
    `to_zarr` -- so no keyword can be consumed ahead of the refusal, whatever
    it is named.

    Not green today by accident: measured 2026-09-07 on the unguarded tree
    this same call raised `TypeError: Dataset.to_zarr() got an unexpected
    keyword argument 'force'` from deep inside the write, not a `ValueError`
    from the guard. With the guard in, the `ValueError` must arrive from
    `_assert_append_compatible`, ahead of `to_zarr`, and the store must be
    untouched.

    RED under: any hatch consumed from `**kwargs` before the guard call --
    exactly mutation M6, under which the signature test above stays GREEN.
    """
    path = str(tmp_path / "smuggled.zarr")
    XrBackend().to_internal(
        _small_panel(["2022-01-04", "2022-01-05", "2022-01-06"], ["A", "B"], 0.0)
    ).append(path)
    before = _panel(path)["close"].values.copy()

    with pytest.raises(ValueError) as excinfo:
        XrBackend().to_internal(
            _small_panel(
                ["2022-01-05", "2022-01-06", "2022-01-07"], ["A", "B"], 100.0
            )
        ).append(path, "timestamp", force=True)

    assert _OVERLAP_CONSEQUENCE in str(excinfo.value)

    store = _panel(path)
    index = pd.DatetimeIndex(store["timestamp"].values)
    assert len(index) == 3
    assert index.is_unique
    assert index.is_monotonic_increasing
    assert store["close"].values.tolist() == before.tolist()


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
    correct-but-unbounded defaults exist to serve: a non-market dataset for
    which densify-then-slice is the intended behaviour, not a degradation.

    Note it subclasses `BaseDataset`, NOT `MarketDataset`. That distinction
    became load-bearing in 03.5 D-08: a `MarketDataset` with no windowed
    densify no longer reaches this code path at all, because it cannot be
    constructed.
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


def test_default_windowed_seam_is_silent_for_a_non_market_dataset(
    tmp_path: Path,
) -> None:
    """A NON-MARKET class that has not overridden `_raw_data_to_xr_window`
    still works, and is no longer warned at about it (03.5 D-08).

    This test previously asserted the opposite -- that a run emitted a
    "not been overridden" warning. That warning was deleted with the
    warn-and-degrade block it guarded, because it was answering the wrong
    question in both directions. For a MARKET dataset the answer is now
    structural: `MarketDataset` declares the seam abstract, so a source with
    no windowed densify raises `TypeError` at construction rather than
    degrading silently and logging about it mid-backfill. For a dataset kind
    like this one, densify-then-slice IS the intended behaviour and there is
    nothing to warn about.

    What survives unchanged is the part that always mattered: the default is
    correct, so the store is complete.
    """
    config = BaseDatasetConfig(zarr_file_path=str(tmp_path / "unbounded.zarr"))

    messages, sink_id = _captured_warnings()
    try:
        _UnboundedDataset(config, _ohlcv_panel()).from_raw_data_chunked()
    finally:
        logger.remove(sink_id)

    assert not [m for m in messages if "_raw_data_to_xr_window" in m], messages
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


# ---------------------------------------------------------------------------
# New-listing reconciliation (260906-x2s Task 2)
#
# A periodic refresh HALTS on the first new listing: the pinned whole-range
# symbol axis no longer matches the store's, and `assert_consistent` refuses.
# That refusal is correct and stays the DEFAULT. These tests pin the two
# explicit opt-ins past it and, critically, the fact that they differ
# OBSERVABLY -- `rebuild` recovers the new listing's real history from raw,
# `widen` leaves it NaN. Neither test is redundant with the other.
# ---------------------------------------------------------------------------


class _GrowingRoster:
    """Two-stage raw tree: a panel over `{A, B}`, then a new listing `C` whose
    raw rows span timestamps ALREADY in the store.

    Spanning the existing timestamps is the whole point. If `C` only traded
    after the store's last row, `rebuild` and `widen` would produce the same
    thing and the pair of tests below could not tell them apart.
    """

    def __init__(self, raw_dir: Path, tmp_path: Path, row, hive) -> None:
        self._raw_dir = raw_dir
        self._tmp_path = tmp_path
        self._row = row
        self._hive = hive

    def config(self, store_name: str = "growing.zarr") -> DatasetConfig:
        return DatasetConfig(
            raw_data_dir_path=str(self._raw_dir / "tiingo"),
            zarr_file_path=str(self._tmp_path / store_name),
            catalog_path=str(self._tmp_path / "catalog"),
            market="us_equity",
            frequency="1d",
            vendor="tiingo",
        )

    def write_initial(self) -> None:
        rows = [
            self._row(f"{year}-{day}", symbol, close=close)
            for year in _YEARS
            for day in _DAYS_PER_YEAR
            for symbol, close in (("A", 100.0), ("B", 200.0))
        ]
        self._hive(self._raw_dir, "tiingo", rows, batch_key="panel")

    def write_new_listing(self) -> None:
        rows = [
            self._row(f"{year}-{day}", "C", close=300.0)
            for year in _YEARS
            for day in _DAYS_PER_YEAR
        ]
        self._hive(self._raw_dir, "tiingo", rows, batch_key="newlisting")


@pytest.fixture
def growing_roster(
    stock_pqt_row: Callable[..., dict],
    hive_raw_tree: Callable[..., Path],
    tmp_path: Path,
) -> _GrowingRoster:
    return _GrowingRoster(
        tmp_path / "growing", tmp_path, stock_pqt_row, hive_raw_tree
    )


def _built_over_ab(growing_roster: _GrowingRoster) -> DatasetConfig:
    """A complete store over `{A, B}`, with `C`'s raw rows then added."""
    config = growing_roster.config()
    growing_roster.write_initial()
    StockDataset(config).from_raw_data_chunked(granularity="year")
    assert _panel(config.zarr_file_path)["symbol"].values.tolist() == ["A", "B"]
    growing_roster.write_new_listing()
    return config


def test_the_default_still_refuses_a_roster_change(
    growing_roster: _GrowingRoster,
) -> None:
    """`refuse` is the DEFAULT and is byte-identical to the behaviour before
    260906-x2s: the `ChunkLedger` roster error still raises and the store is
    untouched.

    RED under: changing the default to `rebuild` or `widen`, which would let a
    single unqualified call silently rewrite or NaN-backfill a live
    multi-hour store.
    """
    config = _built_over_ab(growing_roster)
    before = _panel(config.zarr_file_path)

    with pytest.raises(ValueError) as excinfo:
        StockDataset(config).from_raw_data_chunked(granularity="year")

    message = str(excinfo.value)
    assert "roster" in message.lower()
    xr.testing.assert_identical(_panel(config.zarr_file_path), before)


def test_rebuild_redensifies_every_window_onto_the_new_union(
    growing_roster: _GrowingRoster,
) -> None:
    """Direction 2. Every window is re-densified from raw, so the store ends
    up exactly as a from-scratch chunked run over the same raw tree would
    build it -- and `C`'s pre-existing timestamps therefore carry its REAL
    values, not a backfill.

    RED under: routing `rebuild` to the `widen` branch (C's history comes back
    NaN), or appending onto the existing store instead of rebuilding it.
    """
    config = _built_over_ab(growing_roster)

    StockDataset(config).from_raw_data_chunked(
        granularity="year", on_new_listing="rebuild"
    )

    rebuilt = _panel(config.zarr_file_path)
    assert rebuilt["symbol"].values.tolist() == ["A", "B", "C"]
    # C's REAL history, recovered from raw across the whole range.
    assert (rebuilt["adjClose"].sel(symbol="C").values == 300.0).all()
    assert not np.isnan(rebuilt["adjClose"].sel(symbol="C").values).any()

    scratch_config = growing_roster.config("scratch.zarr")
    StockDataset(scratch_config).from_raw_data_chunked(granularity="year")
    xr.testing.assert_identical(rebuilt, _panel(scratch_config.zarr_file_path))


def test_widen_keeps_history_and_backfills_the_new_listing_with_nan(
    growing_roster: _GrowingRoster,
) -> None:
    """Direction 1 at this layer. The store is KEPT -- every pre-existing
    symbol's history is bit-identical -- and the new listing's whole historical
    block is NaN even though raw rows for it exist, because a widen does not
    re-read raw.

    This test and `test_rebuild_redensifies_every_window_onto_the_new_union`
    are the pair that makes the two directions observably different; neither is
    redundant.

    RED under: routing `widen` to `rebuild` (C comes back at 300.0), or
    skipping `_widen_fill_values` (the widen refuses on the bool
    `anomaly_flag`).
    """
    config = _built_over_ab(growing_roster)
    before = _panel(config.zarr_file_path)

    StockDataset(config).from_raw_data_chunked(
        granularity="year", on_new_listing="widen"
    )

    after = _panel(config.zarr_file_path)
    assert after["symbol"].values.tolist() == ["A", "B", "C"]
    assert after.sizes["timestamp"] == before.sizes["timestamp"]

    for symbol in ("A", "B"):
        for name in before.data_vars:
            np.testing.assert_array_equal(
                after[name].sel(symbol=symbol).values,
                before[name].sel(symbol=symbol).values,
                err_msg=f"{name}/{symbol}",
            )
    assert np.isnan(after["adjClose"].sel(symbol="C").values).all()
    # The bool flag keeps its dtype through the widen -- `False` is the honest
    # value for a symbol that was not trading.
    assert after["anomaly_flag"].dtype == np.dtype("bool")
    assert (~after["anomaly_flag"].sel(symbol="C").values).all()


def test_a_widen_rebases_the_ledger_so_the_next_run_resumes(
    growing_roster: _GrowingRoster,
) -> None:
    """`ChunkLedger.assert_consistent` fingerprints the pinned symbol list
    ORDER-SENSITIVELY. A widen changes the axis, so without a rebase in the
    same operation the widened store is UN-RESUMABLE: the very next run raises
    the roster-refresh error it just got past.

    RED under: omitting the `ChunkLedger.rebase` call.
    """
    config = _built_over_ab(growing_roster)
    StockDataset(config).from_raw_data_chunked(
        granularity="year", on_new_listing="widen"
    )

    ledger = ChunkLedger(ChunkLedger.default_path(config.zarr_file_path))
    assert ledger.symbol_count == 3
    assert ledger.symbol_fingerprint == ChunkLedger.fingerprint(["A", "B", "C"])
    # The rebase re-fingerprints the AXIS; it must not touch the record of
    # which windows are already written.
    assert len(ledger.windows) == 3

    resumed = _SpyStockDataset(config)
    resumed.from_raw_data_chunked(granularity="year")  # default: refuse

    assert resumed.window_calls == []
    assert resumed.whole_range_calls == 0


def test_a_failed_rebuild_restores_the_original_store(
    growing_roster: _GrowingRoster,
) -> None:
    """A rebuild that DELETES first has no way back from a failure halfway
    through a multi-hour re-densify. The store and the ledger are renamed
    aside and restored on any exception.

    RED under: deleting the store (or the ledger) before the rebuild instead
    of renaming it aside.
    """
    config = _built_over_ab(growing_roster)
    before = _panel(config.zarr_file_path)
    ledger_path = ChunkLedger.default_path(config.zarr_file_path)
    with open(ledger_path, "r", encoding="utf-8") as handle:
        ledger_before = handle.read()

    class _FailsDuringRebuild(StockDataset):
        def _raw_data_to_xr_window(self, start_date, end_date, symbols=None):
            raise RuntimeError("simulated crash inside the rebuild")

    with pytest.raises(RuntimeError):
        _FailsDuringRebuild(config).from_raw_data_chunked(
            granularity="year", on_new_listing="rebuild"
        )

    restored = _panel(config.zarr_file_path)
    assert restored["symbol"].values.tolist() == ["A", "B"]
    xr.testing.assert_identical(restored, before)
    with open(ledger_path, "r", encoding="utf-8") as handle:
        assert handle.read() == ledger_before
    assert not Path(f"{config.zarr_file_path}.superseded.tmp").exists()
    assert not Path(f"{ledger_path}.superseded.tmp").exists()


class _CancelAfterNWindows(ProgressReporter):
    """Drives the token from the loop's OWN `window_written` events.

    Cancelling from a second thread would make "after exactly N windows" a
    race the test cannot win on a fast machine; cancelling on the Nth
    `window_written` lands the token deterministically between that window's
    ledger record and the next window's top-of-loop check.
    """

    def __init__(self, token: CancelToken, after: int) -> None:
        self._token = token
        self._after = after
        self.written = 0

    def emit(self, event: ProgressEvent) -> None:
        if event.kind == "window_written":
            self.written += 1
            if self.written >= self._after:
                self._token.cancel()


def test_a_cancelled_rebuild_restores_the_original_store(
    growing_roster: _GrowingRoster,
) -> None:
    """A CANCEL is the third exit from the loop, and it is a halfway exit --
    not a landing.

    The `break` on cancel leaves the loop through the `else:` (success) arm,
    which is where the superseded copies are discarded. Left alone, a
    cancelled rebuild therefore `rmtree`s the ONLY complete record of what the
    store held, keeping a truncated partial in its place at the path every
    reader resolves. That is silent, unrecoverable data loss on the one
    strategy that exists BECAUSE raw can no longer reconstruct the old store
    (symbols dropped from the roster have no raw rows in the current window).

    RED under: routing the cancel exit to `_discard_rebuild_asides` -- i.e.
    the `else:` arm not telling a cancel apart from a completed rebuild.
    """
    config = _built_over_ab(growing_roster)
    before = _panel(config.zarr_file_path)
    ledger_path = ChunkLedger.default_path(config.zarr_file_path)
    with open(ledger_path, "r", encoding="utf-8") as handle:
        ledger_before = handle.read()

    token = CancelToken()
    dataset = StockDataset(config)
    dataset.from_raw_data_chunked(
        granularity="year",
        on_new_listing="rebuild",
        reporter=_CancelAfterNWindows(token, after=1),
        cancel=token,
    )

    result = dataset.last_chunk_result
    assert result.cancelled is True
    assert result.windows_written < result.windows_planned
    # The result says the appended windows did not survive, so a renderer is
    # not left describing work that no longer exists on disk.
    assert result.rebuild_rolled_back is True

    # The pre-rebuild store is still REACHABLE, and unchanged.
    restored = _panel(config.zarr_file_path)
    assert restored["symbol"].values.tolist() == ["A", "B"]
    xr.testing.assert_identical(restored, before)
    with open(ledger_path, "r", encoding="utf-8") as handle:
        assert handle.read() == ledger_before
    # Restored by RENAME, so no aside is left behind to pay for twice.
    aside = BaseDataset.SUPERSEDED_SUFFIX
    assert not Path(f"{config.zarr_file_path}{aside}").exists()
    assert not Path(f"{ledger_path}{aside}").exists()


def test_a_rebuild_cancelled_before_the_first_window_keeps_everything(
    growing_roster: _GrowingRoster,
) -> None:
    """The degenerate case, which is the WORSE one: a token already cancelled
    when the loop starts wrote nothing at all, so discarding the asides
    removed the store, its ledger AND the backup and put nothing in their
    place -- leaving `ConversionResult.zarr_path` naming a directory that no
    longer exists.

    RED under: the same `else:`-arm bug as the test above, with zero windows
    written to soften it.
    """
    config = _built_over_ab(growing_roster)
    before = _panel(config.zarr_file_path)
    ledger_path = ChunkLedger.default_path(config.zarr_file_path)

    token = CancelToken()
    token.cancel()
    dataset = StockDataset(config)
    dataset.from_raw_data_chunked(
        granularity="year", on_new_listing="rebuild", cancel=token
    )

    result = dataset.last_chunk_result
    assert result.cancelled is True
    assert result.windows_written == 0
    assert result.rebuild_rolled_back is True

    assert Path(result.zarr_path).exists()
    assert Path(ledger_path).exists()
    xr.testing.assert_identical(_panel(config.zarr_file_path), before)
    aside = BaseDataset.SUPERSEDED_SUFFIX
    assert not Path(f"{config.zarr_file_path}{aside}").exists()


def test_a_completed_rebuild_still_discards_the_superseded_copies(
    growing_roster: _GrowingRoster,
) -> None:
    """The other side of the cancel split: a rebuild that RAN TO THE END is a
    landing, so the superseded copies are dropped rather than kept forever at
    twice the store's size.

    RED under: over-correcting the fix into "never discard on any exit".
    """
    config = _built_over_ab(growing_roster)
    ledger_path = ChunkLedger.default_path(config.zarr_file_path)

    dataset = StockDataset(config)
    dataset.from_raw_data_chunked(
        granularity="year", on_new_listing="rebuild"
    )

    result = dataset.last_chunk_result
    assert result.cancelled is False
    assert result.rebuild_rolled_back is False
    assert _panel(config.zarr_file_path)["symbol"].values.tolist() == [
        "A",
        "B",
        "C",
    ]
    aside = BaseDataset.SUPERSEDED_SUFFIX
    assert not Path(f"{config.zarr_file_path}{aside}").exists()
    assert not Path(f"{ledger_path}{aside}").exists()


def test_an_unknown_strategy_lists_the_accepted_values(
    three_year_stock_config: Callable[..., DatasetConfig],
) -> None:
    """Mirrors `test_unknown_granularity_lists_the_accepted_values`. Validated
    UP FRONT, before any window runs.

    RED under: accepting an arbitrary string and silently falling through to
    `refuse` -- a typo'd `--on-new-listing rebiuld` would then halt with the
    roster error and look like the flag simply did not work.
    """
    config = three_year_stock_config()

    with pytest.raises(ValueError) as excinfo:
        StockDataset(config).from_raw_data_chunked(on_new_listing="rebiuld")

    message = str(excinfo.value)
    assert "rebiuld" in message
    for accepted in BaseDataset.NEW_LISTING_STRATEGIES:
        assert accepted in message
    assert not Path(config.zarr_file_path).exists()


# ---------------------------------------------------------------------------
# The data_vars axis on the chunked path (260908-0f4 Task 3)
#
# The per-window write goes through `XrBackend.widen_and_append` -- the ONE
# composed three-axis entry point -- rather than through plain `append`. A
# vendor that adds a column between two periodic refreshes used to halt the
# whole run; it now reconciles, with every inherited refusal intact because
# the composed method's CLOSING call is the unchanged `append`.
#
# The swap has one observable store-side residue and it is asserted rather
# than assumed away: the widens COMMIT before that closing append, so a
# rename-shaped window (one variable gone, one new) still HALTS -- no window
# written, no history truncated -- but leaves the store carrying the new name
# backfilled all-NaN over its own extent, where plain `append` refused the
# identical shape with zero store mutation. Measured live both ways.
# ---------------------------------------------------------------------------

#: A fourth year, appended to raw between the two runs, so the second run has
#: a window to actually write. Without it every window is ledger-skipped and
#: no variable reconciliation is ever reached.
_LATER_YEAR = 2025


class _ExtendableRaw:
    """Raw tree over a FIXED roster `{A, B}` whose date range can be extended
    between two chunked runs.

    The roster is deliberately fixed: this section is about the data-variable
    axis, and a symbol-axis drift would drag `on_new_listing` into every
    assertion.
    """

    def __init__(self, raw_dir: Path, tmp_path: Path, row, hive) -> None:
        self._raw_dir = raw_dir
        self._tmp_path = tmp_path
        self._row = row
        self._hive = hive

    def _write(self, years, batch_key: str) -> None:
        rows = [
            self._row(f"{year}-{day}", symbol, close=close)
            for year in years
            for day in _DAYS_PER_YEAR
            for symbol, close in (("A", 100.0), ("B", 200.0))
        ]
        self._hive(self._raw_dir, "tiingo", rows, batch_key=batch_key)

    def write_initial(self) -> None:
        self._write(_YEARS, "vardrift-base")

    def extend(self) -> None:
        self._write((_LATER_YEAR,), "vardrift-later")

    def config(self, store_name: str = "vardrift.zarr") -> DatasetConfig:
        return DatasetConfig(
            raw_data_dir_path=str(self._raw_dir / "tiingo"),
            zarr_file_path=str(self._tmp_path / store_name),
            catalog_path=str(self._tmp_path / "catalog"),
            market="us_equity",
            frequency="1d",
            vendor="tiingo",
        )


@pytest.fixture
def extendable_raw(
    stock_pqt_row: Callable[..., dict],
    hive_raw_tree: Callable[..., Path],
    tmp_path: Path,
) -> _ExtendableRaw:
    return _ExtendableRaw(
        tmp_path / "vardrift", tmp_path, stock_pqt_row, hive_raw_tree
    )


def _store_over_three_years(extendable_raw: _ExtendableRaw) -> DatasetConfig:
    """A complete chunked store over `{A, B}`, with a fourth year then added
    to raw so the next run has exactly one window to write.
    """
    config = extendable_raw.config()
    extendable_raw.write_initial()
    StockDataset(config).from_raw_data_chunked(granularity="year")
    assert _panel(config.zarr_file_path).sizes["timestamp"] == len(_YEARS) * len(
        _DAYS_PER_YEAR
    )
    extendable_raw.extend()
    return config


class _AddsAVariable(StockDataset):
    """A vendor schema change as it actually reaches this layer: one NEW data
    variable in every window.

    Built in `_clean` rather than by writing a mismatched parquet shard on
    purpose -- `_scan_raw` leaves polars' `extra_columns`/`missing_columns` at
    their RAISING defaults, so a mixed-schema raw tier is refused by the
    scanner long before the store is reached. Testing it that way would test
    the scanner, not the reconciliation.
    """

    def _clean(self, data: xr.Dataset) -> xr.Dataset:
        data = super()._clean(data)
        return data.assign(newvar=data["adjClose"] * 2.0)


class _RenamesAVariable(StockDataset):
    """The dangerous shape: one variable GONE and one new -- a vendor column
    rename. The missing half must still be refused unconditionally.
    """

    def _clean(self, data: xr.Dataset) -> xr.Dataset:
        data = super()._clean(data)
        data = data.assign(adjCloseV2=data["adjClose"])
        return data.drop_vars("adjClose")


class _AddsABooleanVariable(StockDataset):
    """A NON-float new variable, plus the fill its dataset declares for it.

    `True` in every window it carries, so the `False` the widen materialises
    over the store's pre-existing extent is distinguishable from the window's
    own data rather than coinciding with it.
    """

    def _clean(self, data: xr.Dataset) -> xr.Dataset:
        data = super()._clean(data)
        return data.assign(
            halted=xr.full_like(data["adjClose"], True, dtype=bool)
        )

    def _widen_fill_values(self) -> dict:
        return {**super()._widen_fill_values(), "halted": False}


def test_a_window_carrying_a_new_variable_is_reconciled_rather_than_refused(
    extendable_raw: _ExtendableRaw,
) -> None:
    """The gap 260907-vyr closed on the factor path, closed here. A vendor
    that adds a column between two periodic refreshes used to halt the whole
    run with `XrBackend.append: ... the incoming panel carries data
    variable(s) [...] that the store does not hold`.

    RED under: reverting the per-window write to plain `append` (M1).
    """
    config = _store_over_three_years(extendable_raw)

    _AddsAVariable(config).from_raw_data_chunked(granularity="year")

    after = _panel(config.zarr_file_path)
    assert "newvar" in after.data_vars
    assert after.sizes["timestamp"] == (len(_YEARS) + 1) * len(_DAYS_PER_YEAR)


def test_the_new_variable_is_backfilled_and_the_stored_ones_are_untouched(
    extendable_raw: _ExtendableRaw,
) -> None:
    """Reconciling the variable axis must not disturb the data already on
    disk: the new name is materialised all-NaN over the store's PRE-EXISTING
    extent and carries real values only where the window supplied them, while
    every stored variable keeps its values bit-for-bit.

    RED under: reverting to plain `append` (M1), or backfilling with something
    other than NaN over the historical block.
    """
    config = _store_over_three_years(extendable_raw)
    before = _panel(config.zarr_file_path)
    stored_rows = before.sizes["timestamp"]

    _AddsAVariable(config).from_raw_data_chunked(granularity="year")

    after = _panel(config.zarr_file_path)
    assert np.isnan(after["newvar"].values[:stored_rows]).all()
    assert not np.isnan(after["newvar"].values[stored_rows:]).any()
    for name in before.data_vars:
        np.testing.assert_array_equal(
            after[name].values[:stored_rows],
            before[name].values,
            err_msg=name,
        )


def test_a_window_missing_a_stored_variable_is_still_refused(
    extendable_raw: _ExtendableRaw,
) -> None:
    """The superset rule is intact, so a column RENAME still HALTS: no window
    is written and no history is truncated. That is the part that matters and
    it is fully preserved.

    The store's POST-REFUSAL state is asserted too, because that state is what
    the swap CHANGES. `widen_and_append` commits its widens before the closing
    `append()` refuses, so the store afterwards carries the stored name AND
    the newly-introduced one, the latter all-NaN over the store's own extent;
    plain `append()` refused the identical shape leaving the variable set
    untouched. Measured live, both sides. This test LOCKS that accepted side
    effect rather than discovering it -- the append dimension does not grow
    and every stored value stays bit-identical, which is why it is accepted.

    A drop-only fixture could not carry this: with nothing to widen, both
    write paths leave the store identical and the assertion would be a
    restatement rather than a lock.

    RED under: reverting to plain `append` (M1) -- the refusal happens either
    way, but the post-refusal variable set does not.
    """
    config = _store_over_three_years(extendable_raw)
    before = _panel(config.zarr_file_path)

    with pytest.raises(ValueError) as excinfo:
        _RenamesAVariable(config).from_raw_data_chunked(granularity="year")

    assert "adjClose" in str(excinfo.value)

    after = _panel(config.zarr_file_path)
    assert sorted(after.data_vars) == sorted(
        list(before.data_vars) + ["adjCloseV2"]
    )
    assert after.sizes["timestamp"] == before.sizes["timestamp"]
    np.testing.assert_array_equal(
        after["adjClose"].values, before["adjClose"].values
    )
    assert np.isnan(after["adjCloseV2"].values).all()


def test_a_non_float_new_variable_is_widened_with_the_declared_fill(
    extendable_raw: _ExtendableRaw,
) -> None:
    """`_widen_fill_values()` reaches BOTH widened axes of the chunked path.
    NaN materialised into a boolean array becomes `True` -- a fabricated
    history rather than an absent one -- so the widen refuses a non-float
    variable outright unless the dataset declares its fill.

    RED under: not forwarding `_widen_fill_values()` to the per-window write
    (the widen refuses on `halted`), or forwarding it to only one axis.
    """
    config = _store_over_three_years(extendable_raw)
    stored_rows = _panel(config.zarr_file_path).sizes["timestamp"]

    _AddsABooleanVariable(config).from_raw_data_chunked(granularity="year")

    after = _panel(config.zarr_file_path)
    assert after["halted"].dtype == np.dtype("bool")
    # `False` over the store's own history -- the DECLARED fill, not a NaN
    # coerced into `True`.
    assert (~after["halted"].values[:stored_rows]).all()
    # `True` where the window actually supplied it, so the two are genuinely
    # distinguishable.
    assert after["halted"].values[stored_rows:].all()


def test_agreeing_axes_still_go_through_the_unchanged_append_guard(
    extendable_raw: _ExtendableRaw,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The overwhelmingly common case -- symbol axis and variable set both
    agreeing -- must be preserved BY CONSTRUCTION, not by a second write path.
    `widen_and_append` delegates straight to the identical `append`, so the
    overlap and dtype refusals still live exactly where they did.

    Asserted structurally: the closing `append` is reached once per written
    window. A refactor that wrote the window directly from inside the composed
    method would remove the guard from this path entirely.

    RED under: writing the window with `to_zarr(mode="a")` from inside the
    composed method, or introducing a conditional second write path.
    """
    config = _store_over_three_years(extendable_raw)
    scratch = extendable_raw.config("scratch.zarr")

    appended: list[str] = []
    original = XrBackend.append

    def _recording_append(self, path, append_dim="timestamp", **kwargs):
        appended.append(path)
        return original(self, path, append_dim, **kwargs)

    monkeypatch.setattr(XrBackend, "append", _recording_append)

    StockDataset(config).from_raw_data_chunked(granularity="year")

    # Exactly one window was outstanding, and the unchanged guard ran for it.
    assert appended == [config.zarr_file_path]

    monkeypatch.undo()
    StockDataset(scratch).from_raw_data_chunked(granularity="year")
    xr.testing.assert_identical(
        _panel(config.zarr_file_path), _panel(scratch.zarr_file_path)
    )
