"""A run records the data it reads at the dataset seam.

A ``DataRecorder`` is a context a run opens. Every read through
``BaseDataset.panel`` and ``Factor.read`` while it is open is logged as a
request; when it closes, each distinct request is hashed once and, when an
expected record is given, compared with it by digest alone. Outside a recorder
a read costs nothing extra.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from loguru import logger

import quantlab.utils.fingerprint as fingerprint
from quantlab.base.config import PolarsFactorConfig
from quantlab.dataset.memory import FrameDataset
from quantlab.dataset.merged import MergedDataset
from quantlab.dataset.stock import StockDataset
from quantlab.utils.fingerprint import DataRecorder, active_recorder
from tests.backtest_fixtures import PastReturnFactor, write_price_store


@pytest.fixture
def warnings_logged():
    """Collect the loguru warnings emitted during the test."""
    messages: list[str] = []
    handler_id = logger.add(messages.append, level="WARNING", format="{message}")
    yield messages
    logger.remove(handler_id)


@pytest.fixture
def hashes(monkeypatch):
    """Count the calls to ``dataset_fingerprint``."""
    calls = []
    original = fingerprint.dataset_fingerprint

    def counting(ds, variables):
        calls.append(sorted(variables))
        return original(ds, variables)

    monkeypatch.setattr(fingerprint, "dataset_fingerprint", counting)
    return calls


@pytest.fixture
def prices(tmp_path):
    """A ``StockDataset`` over a 60-bar business-day store of six symbols."""
    return StockDataset(write_price_store(tmp_path / "a"))


def _frame(symbols=("AAA", "BBB"), scale=1.0):
    """Return a ten-business-day panel with ``close`` scaled by ``scale``."""
    timestamps = pd.bdate_range("2024-01-01", periods=10)
    values = np.arange(len(timestamps) * len(symbols), dtype=float).reshape(
        len(timestamps), len(symbols)
    )
    return xr.Dataset(
        {"close": (["timestamp", "symbol"], values * scale),
         "volume": (["timestamp", "symbol"], values + 100.0)},
        coords={"timestamp": timestamps, "symbol": list(symbols)},
    )


def test_each_distinct_request_is_logged_and_hashed_once(prices, hashes):
    with DataRecorder(keys=[(prices, "price_dataset")]) as recorder:
        prices.panel("2024-01-02", "2024-01-31")
        prices.panel("2024-01-02", "2024-01-31")
        prices.panel("2024-01-02", "2024-02-15")
        assert hashes == []  # nothing is hashed before the run ends

    entries = recorder.records["price_dataset"]
    assert [e["request"]["end"] for e in entries] == ["2024-01-31T00:00:00", "2024-02-15T00:00:00"]
    assert len(hashes) == 2
    assert entries[0]["n_timestamps"] == 22
    assert entries[0]["request"]["variables"] is None
    assert "adjClose" in entries[0]["variables"]  # no variables: every variable


def test_two_spellings_of_the_same_bars_are_one_request(prices, hashes):
    """The request is the bars read, not the dates as a caller spelled them."""
    with DataRecorder(keys=[(prices, "price_dataset")]) as recorder:
        prices.panel("2023-12-30", "2024-01-31")  # a weekend before the first bar
        prices.panel(pd.Timestamp("2024-01-01"), "2024-01-31 23:00")

    (entry,) = recorder.records["price_dataset"]
    assert entry["request"]["start"] == "2024-01-01T00:00:00"
    assert len(hashes) == 1


def test_outside_a_recorder_a_read_records_and_hashes_nothing(prices, hashes):
    prices.panel("2024-01-02", "2024-01-31")

    assert active_recorder() is None
    assert hashes == []


def test_nested_recorders_log_to_the_innermost_only(tmp_path, prices):
    other = StockDataset(write_price_store(tmp_path / "b"))
    with DataRecorder(keys=[(prices, "outer")]) as outer:
        prices.panel("2024-01-02", "2024-01-31")
        with DataRecorder(keys=[(other, "inner")]) as inner:
            assert active_recorder() is inner
            other.panel("2024-01-02", "2024-01-31")
        assert active_recorder() is outer

    assert list(outer.records) == ["outer"]
    assert list(inner.records) == ["inner"]
    assert active_recorder() is None


def test_variables_narrow_the_panel_and_the_hash(prices, hashes):
    panel = prices.panel("2024-01-02", "2024-01-31", variables=["adjOpen", "adjClose"])
    assert sorted(panel.data_vars) == ["adjClose", "adjOpen"]

    with DataRecorder(keys=[(prices, "price_dataset")]) as recorder:
        prices.panel("2024-01-02", "2024-01-31", variables=["adjClose", "adjOpen"])

    (entry,) = recorder.records["price_dataset"]
    assert entry["variables"] == ["adjClose", "adjOpen"]
    assert entry["request"]["variables"] == ["adjClose", "adjOpen"]
    assert hashes == [["adjClose", "adjOpen"]]


def test_symbols_are_part_of_the_request(prices):
    with DataRecorder(keys=[(prices, "p")]) as recorder:
        prices.panel("2024-01-02", "2024-01-31", symbols=["BBB", "AAA"])
        prices.panel("2024-01-02", "2024-01-31")

    entries = recorder.records["p"]
    assert [e["request"]["symbols"] for e in entries] == [["BBB", "AAA"], None]
    assert [e["n_symbols"] for e in entries] == [2, 6]


def test_a_merged_dataset_records_its_leaves_not_itself():
    left = FrameDataset(_frame(("AAA", "BBB")))
    right = FrameDataset(_frame(("CCC",)))
    merged = MergedDataset([left, right])

    with DataRecorder(keys=[(merged, "m"), (left, "m.0"), (right, "m.1")]) as recorder:
        panel = merged.panel("2024-01-02", "2024-01-05", variables=["close"])

    assert list(panel.data_vars) == ["close"]
    assert sorted(recorder.records) == ["m.0", "m.1"]
    assert recorder.records["m.0"][0]["variables"] == ["close"]


def test_an_in_memory_dataset_hashes_its_held_panel():
    held = FrameDataset(_frame())
    with DataRecorder(keys=[(held, "held")]) as recorder:
        held.panel("2024-01-02", "2024-01-05")

    expected = fingerprint.dataset_fingerprint(
        _frame().sel(timestamp=slice("2024-01-02", "2024-01-05")), ["close", "volume"]
    )
    assert recorder.records["held"][0]["digest"] == expected["digest"]


def test_an_unmapped_dataset_falls_back_to_its_class_and_store(prices):
    with DataRecorder() as recorder:
        prices.panel("2024-01-02", "2024-01-05")

    assert list(recorder.records) == [f"StockDataset:{prices.store_path}"]


def test_a_dataset_given_two_keys_takes_the_first(prices):
    with DataRecorder(keys=[(prices, "first"), (prices, "second")]) as recorder:
        prices.panel("2024-01-02", "2024-01-05")

    assert list(recorder.records) == ["first"]


def test_a_factor_store_read_is_recorded_under_the_factor_key(tmp_path, hashes):
    factor = PastReturnFactor(
        PolarsFactorConfig(
            warmup_bars=5,
            dataset=StockDataset(write_price_store(tmp_path)),
            kwargs={"n": 3},
            file_path=str(tmp_path / "factor.zarr"),
        )
    )
    factor.build("2024-01-10", "2024-02-20")
    hashes.clear()

    with DataRecorder(keys=[(factor, "model.factors.0")]) as recorder:
        factor.read("2024-01-15", "2024-02-01")

    (entry,) = recorder.records["model.factors.0"]
    assert entry["variables"] == ["past_ret_3"]
    assert entry["request"] == {
        "start": "2024-01-15T00:00:00", "end": "2024-02-01T00:00:00", "symbols": None,
        "variables": None,
    }
    assert len(hashes) == 1


def _record(dataset, *ranges):
    """Return the records of reading ``ranges`` of ``dataset`` under key ``data``."""
    with DataRecorder(keys=[(dataset, "data")]) as recorder:
        for start, end in ranges:
            dataset.panel(start, end)
    return recorder.records


def _compare(dataset, expected, *ranges, keys=None):
    """Read ``ranges`` of ``dataset`` inside a recorder expecting ``expected``."""
    with DataRecorder(keys=keys or [(dataset, "data")], expected=expected) as recorder:
        for start, end in ranges:
            dataset.panel(start, end)
    return recorder


def test_an_unchanged_record_is_silent(warnings_logged):
    expected = _record(FrameDataset(_frame()), ("2024-01-02", "2024-01-05"))

    _compare(FrameDataset(_frame()), expected, ("2024-01-02", "2024-01-05"))

    assert warnings_logged == []


def test_a_changed_digest_warns_and_never_raises(warnings_logged):
    expected = _record(FrameDataset(_frame()), ("2024-01-02", "2024-01-05"))

    _compare(FrameDataset(_frame(scale=2.0)), expected, ("2024-01-02", "2024-01-05"))

    (message,) = warnings_logged
    assert "data fingerprint mismatch" in message
    assert "'data'" in message and "digest differs" in message
    assert "2024-01-02" in message and "4 timestamps x 2 symbols" in message


def test_only_the_digest_decides(warnings_logged):
    expected = _record(FrameDataset(_frame()), ("2024-01-02", "2024-01-05"))
    expected["data"][0]["n_timestamps"] = 99  # explanation only, never compared

    _compare(FrameDataset(_frame()), expected, ("2024-01-02", "2024-01-05"))

    assert warnings_logged == []


def test_a_missing_and_an_extra_request_warn(warnings_logged):
    expected = _record(FrameDataset(_frame()), ("2024-01-02", "2024-01-05"))

    _compare(FrameDataset(_frame()), expected, ("2024-01-02", "2024-01-08"))

    assert len(warnings_logged) == 2
    assert any("not read by this run" in m and "2024-01-05" in m for m in warnings_logged)
    assert any("absent from the expected record" in m and "2024-01-08" in m
               for m in warnings_logged)


def test_a_missing_and_an_extra_key_warn(warnings_logged):
    expected = _record(FrameDataset(_frame()), ("2024-01-02", "2024-01-05"))
    held = FrameDataset(_frame())

    _compare(held, expected, ("2024-01-02", "2024-01-05"), keys=[(held, "renamed")])

    assert len(warnings_logged) == 2
    assert any("'data'" in m and "not read by this run" in m for m in warnings_logged)
    assert any("'renamed'" in m and "absent from the expected record" in m
               for m in warnings_logged)


def test_the_failure_path_compares_partially_and_the_error_propagates(warnings_logged):
    expected = _record(
        FrameDataset(_frame()), ("2024-01-02", "2024-01-05"), ("2024-01-02", "2024-01-12")
    )
    changed = FrameDataset(_frame(scale=2.0))

    with pytest.raises(RuntimeError, match="boom"):
        with DataRecorder(keys=[(changed, "data")], expected=expected) as recorder:
            changed.panel("2024-01-02", "2024-01-05")
            raise RuntimeError("boom")

    (message,) = warnings_logged  # the request not read yet is not reported
    assert "digest differs" in message
    assert "comparison is PARTIAL" in message
    assert len(recorder.records["data"]) == 1


def test_a_failing_diagnostic_never_replaces_the_error(warnings_logged, monkeypatch):
    held = FrameDataset(_frame())

    def broken(ds, variables):
        raise OSError("disk gone")

    monkeypatch.setattr(fingerprint, "dataset_fingerprint", broken)
    with pytest.raises(RuntimeError, match="boom"):
        with DataRecorder(keys=[(held, "data")], expected={}):
            held.panel("2024-01-02", "2024-01-05")
            raise RuntimeError("boom")

    assert any("OSError" in m for m in warnings_logged)


def test_a_merge_input_reads_its_own_name_of_a_shared_variable(spot_kline_zarr):
    from quantlab.dataset.spot import SpotKlineDataset

    spot = SpotKlineDataset(spot_kline_zarr())
    held = FrameDataset(_frame(("ZZZ1", "ZZZ2")))
    merged = MergedDataset([spot, held])

    with DataRecorder(keys=[(spot, "spot"), (held, "held")]) as recorder:
        panel = merged.panel("2024-01-02", "2024-01-05", variables=["close"])

    assert list(panel.data_vars) == ["close"]
    assert recorder.records["spot"][0]["variables"] == ["Close"]
    assert recorder.records["held"][0]["variables"] == ["close"]


def test_a_resample_view_records_the_source_store_it_read(tmp_path):
    from quantlab.base.config import DatasetConfig
    from quantlab.dataset.spot import SpotKlineDataset

    timestamps = pd.date_range("2024-01-01", periods=5 * 24 * 60, freq="min")
    store = tmp_path / "klines.zarr"
    xr.Dataset(
        {"Close": (["timestamp", "symbol"], np.arange(len(timestamps), dtype=float)[:, None])},
        coords={"timestamp": timestamps, "symbol": ["AUSDT"]},
    ).to_zarr(store, mode="w")
    minute = SpotKlineDataset(DatasetConfig(
        raw_data_dir_path=str(tmp_path / "raw"), zarr_file_path=str(store),
        market="crypto_spot", frequency="1m",
    ))
    daily = minute.resample("1d", "last")

    with DataRecorder() as recorder:
        daily.panel("2024-01-02", "2024-01-03")

    (key,) = recorder.records
    assert key == f"SpotKlineDataset:{store}"  # the source store, not the sibling
    (entry,) = recorder.records[key]
    assert entry["n_timestamps"] > 2  # source minute bars, not resampled days


def test_a_failing_diagnostic_after_a_successful_run_only_warns(warnings_logged, monkeypatch):
    held = FrameDataset(_frame())

    def broken(ds, variables):
        raise OSError("disk gone")

    monkeypatch.setattr(fingerprint, "dataset_fingerprint", broken)
    with DataRecorder(keys=[(held, "data")]):
        held.panel("2024-01-02", "2024-01-05")

    (message,) = warnings_logged
    assert "OSError" in message and "continuing" in message


def test_a_merge_input_holding_none_of_the_variables_is_not_read():
    left = FrameDataset(_frame(("AAA",)))
    right = FrameDataset(_frame(("BBB",)).rename({"close": "bid"}))
    merged = MergedDataset([left, right])

    with DataRecorder(keys=[(left, "left"), (right, "right")]) as recorder:
        panel = merged.panel("2024-01-02", "2024-01-05", variables=["bid"])

    assert list(panel.data_vars) == ["bid"]
    assert list(recorder.records) == ["right"]
    with pytest.raises(KeyError, match="no input holds"):
        merged.panel("2024-01-02", "2024-01-05", variables=["ask"])


def test_a_kunquant_factor_reads_only_its_data_columns(spot_kline_zarr):
    from tests.test_factor_merge import MaDeviation
    from quantlab.base.config import FactorConfig
    from quantlab.dataset.spot import SpotKlineDataset

    spot = SpotKlineDataset(spot_kline_zarr())
    factor = MaDeviation(FactorConfig(
        dataset=spot, mode="batch", data_columns=("close",), warmup_bars=5, njobs=1,
    ))

    with DataRecorder(keys=[(spot, "dataset")]) as recorder:
        factor.compute("2024-01-10", "2024-01-20")

    (entry,) = recorder.records["dataset"]
    assert entry["request"]["variables"] == ["Close"]  # the store's own name


def test_unrecorded_reads_reach_no_recorder_and_are_not_hashed(prices, hashes):
    from quantlab.utils.fingerprint import unrecorded

    with DataRecorder(keys=[(prices, "p")]) as outer:
        with unrecorded():
            assert active_recorder() is None
            prices.panel("2024-01-02", "2024-01-31")
            with DataRecorder(keys=[(prices, "inner")]) as inner:
                prices.panel("2024-01-02", "2024-01-05")
        assert active_recorder() is outer

    assert outer.records == {}
    assert list(inner.records) == ["inner"]
    assert len(hashes) == 1
