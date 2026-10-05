"""`quantlab.api.forward_returns`: a caller's frame in, a forward-return label frame out.

Tested only through the public function (ADR 0011): the frames and errors a caller sees.
Numerical correctness is parity with the library path, ``Return`` and ``BinaryReturn``
computed on the equivalent Zarr-backed dataset. The library labels fix ``delay`` at 1, so
another delay is checked against the library label shifted by the extra bars.
"""

import numpy as np
import pandas as pd
import polars as pl
import pytest
import xarray as xr

from conftest import compute_all
from quantlab.factor.config import FactorConfig
from quantlab.dataset.stock import StockDataset
from quantlab.label.predefined.fret import BinaryReturn, Return
from tests.backtest_fixtures import write_price_store

import quantlab.api as qa

ADJUSTED = ("adjOpen", "adjHigh", "adjLow", "adjClose", "adjVolume")
CANONICAL = ("open", "high", "low", "close", "volume")


def _frame(tmp_path) -> tuple[pd.DataFrame, StockDataset]:
    """A canonical long frame and the Zarr-backed dataset holding the same bars."""
    dataset = StockDataset(write_price_store(tmp_path))
    store = xr.open_zarr(dataset.config.zarr_file_path).load()
    frame = (
        store[list(ADJUSTED)]
        .rename(dict(zip(ADJUSTED, CANONICAL)))
        .to_dataframe()
        .reset_index()
    )
    return frame, dataset


def _library_label(cls, dataset: StockDataset, span: int) -> xr.Dataset:
    return compute_all(
        cls(
            FactorConfig(
                warmup_bars=0,
                dataset=dataset,
                mode="batch",
                data_columns=("adjOpen",),
                kwargs={"n_forward_periods": span},
                njobs=4,
            )
        )
    )


def _forward(prices: np.ndarray, span: int, delay: int) -> np.ndarray:
    """``prices[t + delay + span] / prices[t + delay] - 1``, NaN past the last bar.

    The label is computed in float32, so it is compared with an absolute tolerance.
    """
    out = np.full_like(prices, np.nan)
    ahead = delay + span
    out[: len(prices) - ahead] = prices[ahead:] / prices[delay : len(prices) - span] - 1
    return out


def _wide(frame: pd.DataFrame, column: str) -> np.ndarray:
    return frame.pivot(index="timestamp", columns="symbol", values=column).to_numpy()


# --------------------------------------------------------------------------- parity


@pytest.mark.parametrize("span", [1, 3, 5])
@pytest.mark.parametrize(
    "binary, cls, name",
    [(False, Return, "ret_{}"), (True, BinaryReturn, "ret_binary_{}")],
    ids=["return", "binary"],
)
def test_matches_the_library_label_on_a_zarr_dataset(tmp_path, span, binary, cls, name):
    frame, dataset = _frame(tmp_path)

    result = qa.forward_returns(frame, span=span, binary=binary, as_xarray=True)

    expected = _library_label(cls, dataset, span)
    assert isinstance(result, xr.Dataset)
    assert list(result.data_vars) == [name.format(span)]
    xr.testing.assert_equal(result, expected)
    # Only the last delay + span bars have no later bars to read.
    assert result[name.format(span)].isel(timestamp=slice(-(span + 1), None)).isnull().all()
    assert result[name.format(span)].isel(timestamp=slice(None, -(span + 1))).notnull().all()


@pytest.mark.parametrize("span, delay", [(1, 2), (3, 2), (2, 4)])
@pytest.mark.parametrize("binary, cls", [(False, Return), (True, BinaryReturn)])
def test_a_longer_delay_is_the_library_label_shifted_earlier(tmp_path, span, delay, binary, cls):
    frame, dataset = _frame(tmp_path)

    result = qa.forward_returns(frame, span=span, delay=delay, binary=binary, as_xarray=True)

    expected = _library_label(cls, dataset, span).shift(timestamp=-(delay - 1))
    xr.testing.assert_equal(result, expected)


@pytest.mark.parametrize("span", [1, 3])
def test_delay_zero_starts_at_the_signal_bar(tmp_path, span):
    frame, dataset = _frame(tmp_path)

    result = qa.forward_returns(frame, span=span, delay=0, as_xarray=True)

    # One bar earlier than the library label, which starts at the next bar ...
    library = _library_label(Return, dataset, span).shift(timestamp=1)
    later = slice(1, None)
    xr.testing.assert_equal(result.isel(timestamp=later), library.isel(timestamp=later))
    # ... and the first bar, which the library label has no bar before to shift from.
    opens = _wide(frame, "open")
    np.testing.assert_allclose(
        result[f"ret_{span}"].values[0], opens[span] / opens[0] - 1, atol=1e-6
    )


# --------------------------------------------------------------------------- price column


def test_price_close_reads_the_close_column(tmp_path):
    frame, _ = _frame(tmp_path)

    by_close = qa.forward_returns(frame, price="close", span=2, delay=1)
    by_open = qa.forward_returns(frame, price="open", span=2, delay=1)

    np.testing.assert_allclose(
        _wide(by_close, "ret_2"), _forward(_wide(frame, "close"), 2, 1), atol=1e-6
    )
    np.testing.assert_allclose(
        _wide(by_open, "ret_2"), _forward(_wide(frame, "open"), 2, 1), atol=1e-6
    )


def test_a_frame_holding_only_the_price_column_is_enough(tmp_path):
    frame, _ = _frame(tmp_path)

    only_close = frame[["timestamp", "symbol", "close"]]

    pd.testing.assert_frame_equal(
        qa.forward_returns(only_close, price="close"),
        qa.forward_returns(frame, price="close"),
    )


def test_columns_maps_caller_names_before_the_price_is_chosen(tmp_path):
    frame, _ = _frame(tmp_path)
    renamed = frame.rename(columns={"timestamp": "date", "symbol": "ticker", "open": "Open"})

    result = qa.forward_returns(
        renamed, columns={"date": "timestamp", "ticker": "symbol", "Open": "open"}
    )

    pd.testing.assert_frame_equal(result, qa.forward_returns(frame))


@pytest.mark.parametrize("library", ["pandas", "polars"])
def test_a_missing_price_column_raises_naming_the_available_columns(tmp_path, library):
    frame, _ = _frame(tmp_path)
    frame = frame.drop(columns="open")
    if library == "polars":
        frame = pl.from_pandas(frame)

    with pytest.raises(ValueError) as info:
        qa.forward_returns(frame)
    message = str(info.value)
    assert "'open'" in message
    for present in ("close", "high", "low", "volume"):
        assert repr(present) in message
    assert "price='close' to use" in message
    assert "columns=" in message


@pytest.mark.parametrize("library", ["pandas", "polars"])
def test_without_close_the_first_present_column_is_suggested(tmp_path, library):
    frame, _ = _frame(tmp_path)
    frame = frame[["timestamp", "symbol", "high", "low"]]
    if library == "polars":
        frame = pl.from_pandas(frame)

    with pytest.raises(ValueError, match=r"price='high' to use"):
        qa.forward_returns(frame, price="close")


def test_the_suggestion_uses_the_names_after_columns_renames(tmp_path):
    frame, _ = _frame(tmp_path)
    renamed = frame.drop(columns="open").rename(columns={"close": "Close"})

    with pytest.raises(ValueError, match=r"price='close' to use"):
        qa.forward_returns(renamed, columns={"Close": "close"})


def test_a_multiindexed_frame_is_checked_on_its_columns(tmp_path):
    frame, _ = _frame(tmp_path)
    indexed = frame.drop(columns="open").set_index(["timestamp", "symbol"])

    with pytest.raises(ValueError, match=r"'open'.*price='close' to use"):
        qa.forward_returns(indexed)


def test_an_unknown_price_column_raises_naming_the_available_columns(tmp_path):
    frame, _ = _frame(tmp_path)

    with pytest.raises(ValueError, match=r"'vwap'.*'close'"):
        qa.forward_returns(frame, price="vwap")


# --------------------------------------------------------------------------- return types


def test_pandas_in_gives_a_long_pandas_frame_out(tmp_path):
    frame, _ = _frame(tmp_path)
    panel = qa.forward_returns(frame, span=3, as_xarray=True)

    result = qa.forward_returns(frame, span=3)

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["timestamp", "symbol", "ret_3"]
    assert len(result) == panel.sizes["timestamp"] * panel.sizes["symbol"]
    back = result.set_index(["timestamp", "symbol"]).to_xarray()
    xr.testing.assert_equal(back, panel)


@pytest.mark.parametrize("binary", [False, True])
def test_polars_in_gives_polars_out_with_the_same_values(tmp_path, binary):
    frame, _ = _frame(tmp_path)

    from_pandas = qa.forward_returns(frame, span=2, binary=binary)
    from_polars = qa.forward_returns(pl.from_pandas(frame), span=2, binary=binary)

    assert isinstance(from_polars, pl.DataFrame)
    pd.testing.assert_frame_equal(from_polars.to_pandas(), from_pandas, check_dtype=False)


def test_ragged_data_is_densified_with_nan(tmp_path):
    frame, _ = _frame(tmp_path)
    first = frame["timestamp"].min()
    ragged = frame[~((frame["timestamp"] == first) & (frame["symbol"] == "AAA"))]

    result = qa.forward_returns(ragged, span=1, delay=0, as_xarray=True)

    assert result.sizes["timestamp"] == frame["timestamp"].nunique()
    assert np.isnan(result["ret_1"].sel(symbol="AAA").values[0])
    assert not np.isnan(result["ret_1"].sel(symbol="BBB").values[0])


# --------------------------------------------------------------------------- arguments


@pytest.mark.parametrize(
    "kwargs, error, match",
    [
        ({"span": 0}, ValueError, r"span must be at least 1, got 0"),
        ({"span": 1.5}, TypeError, r"span must be an int, got 1\.5"),
        ({"span": True}, TypeError, r"span must be an int, got True"),
        ({"delay": -1}, ValueError, r"delay must be at least 0, got -1"),
        ({"delay": "1"}, TypeError, r"delay must be an int, got '1'"),
        ({"price": 3}, TypeError, r"price must be a column name, got 3"),
        ({"binary": 1}, TypeError, r"binary must be True or False, got 1"),
    ],
)
def test_invalid_arguments_raise_before_any_work(kwargs, error, match):
    # The frame is not even a DataFrame: the arguments are checked first.
    with pytest.raises(error, match=match):
        qa.forward_returns(None, **kwargs)


def test_a_frame_of_another_type_raises():
    with pytest.raises(TypeError, match="pandas or polars"):
        qa.forward_returns([1, 2, 3])
