"""`quantlab.api.compute_factors`: a caller's frame in, a factor frame out.

Tested only through the public function (ADR 0011): the frames and errors a caller sees.
Numerical correctness is parity with the library path, the same factor class computed on
the equivalent Zarr-backed dataset. Input rules (column mapping, MultiIndex, duplicates,
ragged data, time zones, symbol types) run through a small Polars factor defined here, so
they need no KunQuant compile.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest
import xarray as xr

from conftest import compute_all
from quantlab.factor.config import FactorConfig, PolarsFactorConfig
from quantlab.dataset.spot import SpotKlineDataset
from quantlab.dataset.stock import StockDataset
from quantlab.factor.polars import FactorPolars
from tests.backtest_fixtures import write_price_store

import quantlab.api as qa

REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ("open", "high", "low", "close", "volume")
ADJUSTED = ("adjOpen", "adjHigh", "adjLow", "adjClose", "adjVolume")
BINANCE = ("Open", "High", "Low", "Close", "Volume", "Quote asset volume")


class CloseChange(FactorPolars):
    """One-bar close change per symbol, reading the canonical ``close`` column."""

    def _get_factor_lazyframe(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        close = pl.col("close")
        return (
            lf.sort(["symbol", "timestamp"])
            .with_columns((close / close.shift(1).over("symbol") - 1.0).alias("chg"))
            .select(["timestamp", "symbol", "chg"])
        )


def _equity_frame(tmp_path) -> tuple[pd.DataFrame, StockDataset]:
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


def _crypto_frame(spot_kline_zarr) -> tuple[pd.DataFrame, SpotKlineDataset]:
    dataset = SpotKlineDataset(spot_kline_zarr(periods=70))
    store = xr.open_zarr(dataset.config.zarr_file_path).load()
    frame = (
        store[list(BINANCE)]
        .rename(dict(zip(BINANCE, CANONICAL + ("amount",))))
        .to_dataframe()
        .reset_index()
    )
    return frame, dataset


def _library_factor(cls, dataset, data_columns) -> xr.Dataset:
    return compute_all(
        cls(
            FactorConfig(
                warmup_bars=0,
                dataset=dataset,
                mode="batch",
                data_columns=data_columns,
                njobs=4,
            )
        )
    )


def _small_frame() -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=4, freq="D")
    rows = []
    for i, ts in enumerate(timestamps):
        rows.append({"timestamp": ts, "symbol": "AAA", "close": 10.0 + i})
        rows.append({"timestamp": ts, "symbol": "BBB", "close": 20.0 * (1 + i)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- parity


@pytest.mark.parametrize(
    "name, class_path",
    [
        ("alpha158", "quantlab.factor.predefined.alpha158.Alpha158Stock"),
        ("alpha101", "quantlab.factor.predefined.alpha101.Alpha101Stock"),
    ],
)
def test_equity_short_names_match_the_stock_classes_on_a_zarr_dataset(tmp_path, name, class_path):
    from quantlab.core.component import get_cls_from_path

    frame, dataset = _equity_frame(tmp_path)

    result = qa.compute_factors(frame, name, as_xarray=True)

    expected = _library_factor(get_cls_from_path(class_path), dataset, ADJUSTED)
    assert isinstance(result, xr.Dataset)
    xr.testing.assert_equal(result, expected)
    # The warm-up bars of a rolling window are NaN, not dropped.
    if name == "alpha158":
        assert result["STD5"].isel(timestamp=slice(0, 4)).isnull().all()
        assert result["STD5"].isel(timestamp=slice(4, None)).notnull().any()


@pytest.mark.parametrize(
    "name, class_path",
    [
        ("alpha158_crypto", "quantlab.factor.predefined.alpha158.Alpha158SpotKline"),
        ("alpha101_crypto", "quantlab.factor.predefined.alpha101.Alpha101SpotKline"),
    ],
)
def test_crypto_short_names_match_the_spot_kline_classes(spot_kline_zarr, name, class_path):
    from quantlab.core.component import get_cls_from_path

    frame, dataset = _crypto_frame(spot_kline_zarr)

    result = qa.compute_factors(frame, name, as_xarray=True)

    expected = _library_factor(
        get_cls_from_path(class_path), dataset, CANONICAL + ("amount",)
    )
    xr.testing.assert_equal(result, expected)


def test_pandas_in_gives_a_long_pandas_frame_out(tmp_path):
    frame, _ = _equity_frame(tmp_path)
    panel = qa.compute_factors(frame, "alpha158", as_xarray=True)

    result = qa.compute_factors(frame, "alpha158")

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns[:2]) == ["timestamp", "symbol"]
    assert set(result.columns[2:]) == set(panel.data_vars)
    assert len(result) == panel.sizes["timestamp"] * panel.sizes["symbol"]
    back = result.set_index(["timestamp", "symbol"]).to_xarray()
    xr.testing.assert_equal(back[list(panel.data_vars)], panel)


def test_polars_in_gives_polars_out_with_the_same_values(tmp_path):
    frame, _ = _equity_frame(tmp_path)

    from_pandas = qa.compute_factors(frame, "alpha158")
    from_polars = qa.compute_factors(pl.from_pandas(frame), "alpha158")

    assert isinstance(from_polars, pl.DataFrame)
    pd.testing.assert_frame_equal(from_polars.to_pandas(), from_pandas, check_dtype=False)


def test_a_catalog_class_passed_directly_equals_its_short_name(tmp_path):
    from quantlab.factor.predefined.alpha158 import Alpha158Stock

    frame, _ = _equity_frame(tmp_path)

    by_class = qa.compute_factors(frame, Alpha158Stock, as_xarray=True)

    xr.testing.assert_equal(by_class, qa.compute_factors(frame, "alpha158", as_xarray=True))


def test_a_subclass_of_a_catalog_class_reads_its_entry_columns(tmp_path):
    from quantlab.factor.predefined.alpha158 import Alpha158Stock

    class MyAlpha158(Alpha158Stock):
        pass

    frame, _ = _equity_frame(tmp_path)

    result = qa.compute_factors(frame, MyAlpha158, as_xarray=True)

    xr.testing.assert_equal(result, qa.compute_factors(frame, "alpha158", as_xarray=True))


def test_any_factor_subclass_reads_the_canonical_columns():
    frame = _small_frame()

    result = qa.compute_factors(frame, CloseChange)

    wide = result.pivot(index="timestamp", columns="symbol", values="chg")
    assert np.isnan(wide.iloc[0]).all()
    np.testing.assert_allclose(wide["AAA"].iloc[1:], [11 / 10 - 1, 12 / 11 - 1, 13 / 12 - 1])
    np.testing.assert_allclose(wide["BBB"].iloc[1:], [1.0, 0.5, 1 / 3])


# --------------------------------------------------------------------------- input rules


def test_columns_maps_caller_names_onto_canonical_ones():
    frame = _small_frame()
    renamed = frame.rename(columns={"timestamp": "date", "symbol": "ticker", "close": "Close"})

    result = qa.compute_factors(
        renamed,
        CloseChange,
        columns={"date": "timestamp", "ticker": "symbol", "Close": "close"},
    )

    pd.testing.assert_frame_equal(result, qa.compute_factors(frame, CloseChange))


def test_a_mapping_naming_an_absent_column_raises():
    with pytest.raises(ValueError, match="'Close'"):
        qa.compute_factors(_small_frame(), CloseChange, columns={"Close": "close"})


def test_a_timestamp_symbol_multiindex_is_reset():
    frame = _small_frame()

    result = qa.compute_factors(frame.set_index(["timestamp", "symbol"]), CloseChange)

    pd.testing.assert_frame_equal(result, qa.compute_factors(frame, CloseChange))


@pytest.mark.parametrize("library", ["pandas", "polars"])
def test_duplicate_rows_raise_listing_the_first_duplicates(library):
    frame = pd.concat([_small_frame(), _small_frame().iloc[[3]]], ignore_index=True)
    if library == "polars":
        frame = pl.from_pandas(frame)

    with pytest.raises(ValueError, match=r"duplicate.*2024-01-02.*BBB"):
        qa.compute_factors(frame, CloseChange)


def test_ragged_data_is_densified_with_nan():
    frame = _small_frame()
    ragged = frame.drop(index=[2]).reset_index(drop=True)  # AAA on 2024-01-02

    result = qa.compute_factors(ragged, CloseChange, as_xarray=True)

    assert dict(result.sizes) == {"timestamp": 4, "symbol": 2}
    aaa = result["chg"].sel(symbol="AAA").values
    assert np.isnan(aaa[:3]).all()  # no bar on day 2, so days 2 and 3 have no change
    np.testing.assert_allclose(aaa[3], 13 / 12 - 1)


@pytest.mark.parametrize("library", ["pandas", "polars"])
def test_tz_aware_timestamps_become_naive_utc(library):
    frame = _small_frame()
    frame["timestamp"] = frame["timestamp"].dt.tz_localize("America/New_York")
    expected_times = frame["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None).unique()
    if library == "polars":
        frame = pl.from_pandas(frame)

    result = qa.compute_factors(frame, CloseChange)

    times = result["timestamp"].to_pandas() if library == "polars" else result["timestamp"]
    assert times.dt.tz is None
    np.testing.assert_array_equal(np.unique(times.to_numpy()), np.sort(expected_times))


@pytest.mark.parametrize("library", ["pandas", "polars"])
def test_integer_symbols_become_strings(library):
    frame = _small_frame()
    frame["symbol"] = frame["symbol"].map({"AAA": 10107, "BBB": 7000})
    if library == "polars":
        frame = pl.from_pandas(frame)

    result = qa.compute_factors(frame, CloseChange)

    symbols = result["symbol"].to_list()
    assert all(isinstance(s, str) for s in symbols)
    assert sorted(set(symbols)) == ["10107", "7000"]


@pytest.mark.parametrize("library", ["pandas", "polars"])
@pytest.mark.parametrize(
    "timestamp, symbol",
    [(pd.Timestamp("2024-01-02"), None), (pd.NaT, "AAA")],
    ids=["null-symbol", "null-timestamp"],
)
def test_a_row_without_timestamp_or_symbol_raises(library, timestamp, symbol):
    frame = pd.concat(
        [_small_frame(), pd.DataFrame([{"timestamp": timestamp, "symbol": symbol, "close": 3.0}])],
        ignore_index=True,
    )
    if library == "polars":
        frame = pl.from_pandas(frame)

    with pytest.raises(ValueError, match=r"1 row\(s\) with a missing timestamp or symbol.*row 8"):
        qa.compute_factors(frame, CloseChange)


def test_a_timestamp_index_asks_for_reset_index():
    frame = _small_frame().set_index("timestamp")

    with pytest.raises(ValueError, match=r"reset_index\(\)"):
        qa.compute_factors(frame, CloseChange)


# --------------------------------------------------------------------------- errors


def test_an_unknown_short_name_raises_listing_the_valid_names():
    with pytest.raises(ValueError) as info:
        qa.compute_factors(_small_frame(), "alpha191")
    message = str(info.value)
    for name in ("alpha101", "alpha158", "alpha101_crypto", "alpha158_crypto"):
        assert repr(name) in message
    for absent in ("LiteratureAlpha", "ResidualMomentumFF3", "MarketFeatures"):
        assert absent not in message


def test_a_missing_required_column_raises_naming_it(tmp_path):
    frame, _ = _equity_frame(tmp_path)
    with pytest.raises(ValueError, match="'amount'"):
        qa.compute_factors(frame, "alpha158_crypto")
    with pytest.raises(ValueError, match="'volume'"):
        qa.compute_factors(frame.drop(columns="volume"), "alpha158")


def test_a_factor_that_is_not_a_factor_class_raises():
    with pytest.raises(TypeError, match="Factor"):
        qa.compute_factors(_small_frame(), 42)


def test_a_frame_of_another_type_raises():
    with pytest.raises(TypeError, match="pandas or polars"):
        qa.compute_factors([1, 2, 3], CloseChange)


# --------------------------------------------------------------------------- import cost


def test_importing_the_api_loads_no_heavy_dependency():
    code = (
        "import sys\n"
        "import quantlab.api\n"
        "print(sorted(m for m in ('KunQuant', 'vectorbt', 'plotly') if m in sys.modules))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO_ROOT
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "[]"
