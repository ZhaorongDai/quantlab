"""Resampling a dataset or factor panel onto coarser bars.

A *resample* aggregates a ``(timestamp, symbol)`` panel onto a coarser
regular time grid, one aggregation method per variable. ``resample()`` on a
dataset or factor returns a copy whose config carries ``resample_freq`` and
``resample_how``; the copy shares no memory with its source, reads its own
store beside the source store when one was saved, and otherwise resamples the
source on read. The backend does the grouping; the dataset decides how bars
are cut (``_resample_labels``), and a factor cuts them the way its dataset
does.
"""

import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest
import xarray as xr

from quantlab.backend import PlBackend, XrBackend
from quantlab.base.config import DatasetConfig, FactorConfig, PolarsFactorConfig
from quantlab.dataset.spot import SpotKlineDataset
from quantlab.factor.alpha158 import Alpha158SpotKline
from quantlab.factor.momentum import Momentum
from quantlab.utils.module import load_dataset_from_config, load_factor_from_config
from quantlab.utils.resample import session_labels

SYMBOLS = ["AAAUSDT", "BBBUSDT"]
BARS_PER_DAY = 6
DAYS = 3


def _minute_panel(symbols: list[str] = SYMBOLS) -> xr.Dataset:
    """Three days of six minute bars each, labelled at their open time.

    ``Close`` counts bars from 1.0 so every aggregate is easy to predict;
    ``Volume`` is 1.0 everywhere so a daily ``sum`` is the bar count.
    """
    timestamps = pd.DatetimeIndex(
        np.concatenate(
            [
                pd.date_range(f"2024-01-0{day} 00:00", periods=BARS_PER_DAY, freq="min").values
                for day in range(1, DAYS + 1)
            ]
        )
    )
    n = len(timestamps)
    scale = 10.0 ** np.arange(len(symbols))
    close = np.arange(1.0, n + 1.0)[:, None] * scale[None, :]
    volume = np.ones((n, len(symbols)))
    return xr.Dataset(
        {
            "Open": (["timestamp", "symbol"], close - 0.5),
            "High": (["timestamp", "symbol"], close + 1.0),
            "Low": (["timestamp", "symbol"], close - 1.0),
            "Close": (["timestamp", "symbol"], close),
            "Volume": (["timestamp", "symbol"], volume),
            "Quote asset volume": (["timestamp", "symbol"], volume * close),
        },
        coords={"timestamp": timestamps, "symbol": symbols},
    )


OHLCV_HOW = {
    "Open": "first",
    "High": "max",
    "Low": "min",
    "Close": "last",
    "Volume": "sum",
    "Quote asset volume": "sum",
}


def _write_minute_store(tmp_path: Path, symbols: list[str]) -> DatasetConfig:
    spot_dir = tmp_path / "spot"
    spot_dir.mkdir(exist_ok=True)
    zarr_path = spot_dir / "klines.zarr"
    _minute_panel(symbols).to_zarr(zarr_path, mode="w")
    return DatasetConfig(
        raw_data_dir_path=str(spot_dir / "raw"),
        zarr_file_path=str(zarr_path),
        market="crypto_spot",
        frequency="1m",
    )


@pytest.fixture
def minute_config(tmp_path: Path) -> DatasetConfig:
    """A ``SpotKlineDataset`` config over a two-symbol synthetic minute store."""
    return _write_minute_store(tmp_path, SYMBOLS)


@pytest.fixture
def minute_config_8(tmp_path: Path) -> DatasetConfig:
    """The same store with eight symbols, the lane count KunQuant needs."""
    return _write_minute_store(tmp_path, [f"S{i}USDT" for i in range(8)])


# -- backend --------------------------------------------------------------------


def _labels(timestamps) -> pd.Series:
    index = pd.DatetimeIndex(timestamps)
    return pd.Series(index.floor("D").values, index=index)


def test_xr_backend_resample_applies_one_method_per_variable():
    panel = _minute_panel()
    panel["Close"][BARS_PER_DAY - 1, 0] = np.nan  # last bar of day one, first symbol
    backend = XrBackend().to_internal(panel)

    backend.resample(_labels(panel["timestamp"].values), OHLCV_HOW)
    daily = backend.data

    assert tuple(daily["Close"].dims) == ("timestamp", "symbol")
    assert daily.sizes == {"timestamp": DAYS, "symbol": 2}
    assert daily["timestamp"].dtype == np.dtype("datetime64[ns]")
    assert daily["timestamp"].values[0] == np.datetime64("2024-01-01")
    # `last` skips the NaN and takes bar 5; the second symbol keeps bar 6.
    assert daily["Close"].values[0].tolist() == [5.0, 60.0]
    assert daily["Open"].values[0].tolist() == [0.5, 9.5]
    assert daily["High"].values[0].tolist() == [7.0, 61.0]
    assert daily["Low"].values[0].tolist() == [0.0, 9.0]
    assert daily["Volume"].values.tolist() == [[6.0, 6.0]] * DAYS


def test_xr_backend_resample_count_and_mean():
    panel = _minute_panel()[["Close"]]
    panel["Close"][0, 0] = np.nan
    backend = XrBackend().to_internal(panel)
    backend.resample(_labels(panel["timestamp"].values), {"Close": "count"})
    assert backend.data["Close"].values[0].tolist() == [5, 6]

    backend = XrBackend().to_internal(_minute_panel()[["Close"]])
    backend.resample(_labels(panel["timestamp"].values), {"Close": "mean"})
    assert backend.data["Close"].values[0].tolist() == [3.5, 35.0]


def test_xr_backend_resample_refuses_a_missing_label_or_method():
    panel = _minute_panel()[["Close"]]
    backend = XrBackend().to_internal(panel)
    labels = _labels(panel["timestamp"].values).iloc[:-1]
    with pytest.raises(ValueError, match="has no label"):
        backend.resample(labels, {"Close": "last"})
    with pytest.raises(ValueError, match="no method for"):
        backend.resample(_labels(panel["timestamp"].values), {})


def test_pl_backend_resample_matches_the_xarray_result():
    panel = _minute_panel()
    labels = _labels(panel["timestamp"].values)
    expected = XrBackend().to_internal(panel).resample(labels, OHLCV_HOW).data

    table = PlBackend().to_internal(
        pl.from_pandas(panel.to_dataframe().reset_index()).lazy()
    )
    got = table.resample(labels, OHLCV_HOW).get_xarray_dataset(["timestamp", "symbol"])

    for name in OHLCV_HOW:
        np.testing.assert_allclose(got[name].values, expected[name].values)


# -- dataset --------------------------------------------------------------------


def test_resample_returns_an_independent_copy(minute_config: DatasetConfig):
    minute = SpotKlineDataset(minute_config).read()

    daily = minute.resample("1d", OHLCV_HOW)

    assert daily is not minute
    assert daily.config is not minute.config
    assert daily.data_backend is not minute.data_backend
    assert daily.config.resample_freq == "1d"
    assert daily.config.resample_how == OHLCV_HOW
    assert minute.config.resample_freq is None
    assert minute.get_xarray_dataset().sizes["timestamp"] == DAYS * BARS_PER_DAY

    panel = daily.get_xarray_dataset()
    assert panel.sizes == {"timestamp": DAYS, "symbol": 2}
    assert panel["Close"].values[:, 0].tolist() == [6.0, 12.0, 18.0]
    assert daily.time_interval == np.timedelta64(1, "D")
    # No memory is shared with the source panel.
    assert not np.shares_memory(
        panel["Close"].values, minute.get_xarray_dataset()["Close"].values
    )
    # Nor is the config shared.
    daily.config.resample_how = "last"
    assert minute.config.resample_how is None


def test_resample_on_an_empty_dataset_resamples_on_read(minute_config: DatasetConfig):
    daily = SpotKlineDataset(minute_config).resample("1d", "last")
    with pytest.raises(AttributeError):
        daily.get_xarray_dataset()

    panel = daily.read().get_xarray_dataset()
    assert panel.sizes["timestamp"] == DAYS
    assert panel["Close"].values[:, 1].tolist() == [60.0, 120.0, 180.0]
    # A second read reuses the held panel instead of resampling twice.
    assert daily.read().get_xarray_dataset().sizes["timestamp"] == DAYS
    assert daily.read(overwrite=True).get_xarray_dataset().sizes["timestamp"] == DAYS
    assert daily.get_lazyframe().collect().height == DAYS * 2


def test_read_narrows_dates_before_resampling(minute_config: DatasetConfig):
    config = dataclasses.replace(minute_config, start_date="2024-01-02")
    daily = SpotKlineDataset(config).resample("1d", "last").read()
    assert daily.get_xarray_dataset()["timestamp"].values.astype("datetime64[D]").tolist() == [
        pd.Timestamp("2024-01-02").date(),
        pd.Timestamp("2024-01-03").date(),
    ]


def test_resampled_dataset_saves_beside_the_source_and_reads_it_back(
    minute_config: DatasetConfig,
):
    daily = SpotKlineDataset(minute_config).read().resample("1d", OHLCV_HOW)
    expected_path = str(Path(minute_config.zarr_file_path).with_name("klines_resample_1d.zarr"))
    assert daily.store_path == expected_path
    assert SpotKlineDataset(minute_config).store_path == minute_config.zarr_file_path

    daily.save()

    assert Path(expected_path).is_dir()
    stored = xr.open_zarr(expected_path)
    assert stored.sizes["timestamp"] == DAYS
    # The source store is untouched.
    assert xr.open_zarr(minute_config.zarr_file_path).sizes["timestamp"] == DAYS * BARS_PER_DAY

    # A dataset built with the resample fields reads the saved store.
    config = dataclasses.replace(minute_config, resample_freq="1d", resample_how=OHLCV_HOW)
    reader = SpotKlineDataset(config).read()
    assert reader.get_xarray_dataset()["Close"].values[:, 0].tolist() == [6.0, 12.0, 18.0]
    assert reader.head(2).collect().height == 2


def test_resampled_dataset_refuses_to_build_from_raw_files(minute_config: DatasetConfig):
    daily = SpotKlineDataset(minute_config).resample("1d", "last")
    with pytest.raises(ValueError, match="from_raw_data"):
        daily.from_raw_data()
    with pytest.raises(ValueError, match="from_raw_data_chunked"):
        daily.update()


def test_resample_validates_freq_how_and_coverage(minute_config: DatasetConfig):
    minute = SpotKlineDataset(minute_config).read()
    with pytest.raises(ValueError, match="resample_freq '2d' is not one of"):
        minute.resample("2d", "last")
    with pytest.raises(ValueError, match="unknown method"):
        minute.resample("1d", "median")
    with pytest.raises(ValueError, match="does not name"):
        minute.resample("1d", {"Close": "last"})
    with pytest.raises(ValueError, match="which the panel does not have"):
        minute.resample("1d", {**OHLCV_HOW, "Adj": "last"})
    with pytest.raises(ValueError, match="is not coarser"):
        minute.resample("1m", "last")
    with pytest.raises(ValueError, match="needs resample_how"):
        SpotKlineDataset(dataclasses.replace(minute_config, resample_freq="1d"))
    with pytest.raises(ValueError, match="resample_freq is None"):
        SpotKlineDataset(dataclasses.replace(minute_config, resample_how="last"))


def test_resampled_dataset_config_round_trips(minute_config: DatasetConfig):
    daily = SpotKlineDataset(minute_config).resample("1d", OHLCV_HOW)
    saved = json.loads(json.dumps(daily.get_config()))
    assert saved["resample_freq"] == "1d"
    assert saved["resample_how"] == OHLCV_HOW

    rebuilt = load_dataset_from_config(saved)
    assert rebuilt.config.resample_freq == "1d"
    assert rebuilt.read().get_xarray_dataset().sizes["timestamp"] == DAYS


def test_resample_chains_onto_a_coarser_grid(minute_config: DatasetConfig):
    minute = SpotKlineDataset(minute_config).read()
    five = minute.resample("5m", OHLCV_HOW)
    assert five.get_xarray_dataset().sizes["timestamp"] == 2 * DAYS
    daily = five.resample("1d", OHLCV_HOW)
    assert daily.get_xarray_dataset()["Volume"].values[:, 0].tolist() == [6.0] * DAYS


# -- factor ---------------------------------------------------------------------


def _momentum(minute_config: DatasetConfig, tmp_path: Path) -> Momentum:
    return Momentum(
        PolarsFactorConfig(
            window=1,
            dataset=SpotKlineDataset(minute_config),
            file_path=str(tmp_path / "factors" / "momentum.zarr"),
            kwargs={"n": 1},
        )
    )


def test_factor_resample_aggregates_the_computed_panel(
    minute_config: DatasetConfig, tmp_path: Path
):
    minute = _momentum(minute_config, tmp_path).cal()
    minute_panel = minute.get_features()

    daily = minute.resample("1d", "last")

    assert daily.config.dataset is not minute.config.dataset
    assert daily.config.dataset.config.resample_freq is None  # computed on minute bars
    assert minute.get_features().sizes["timestamp"] == DAYS * BARS_PER_DAY
    panel = daily.get_features()
    assert panel.sizes == {"timestamp": DAYS, "symbol": 2}
    last_bar_of_each_day = minute_panel["momentum_1"].values[BARS_PER_DAY - 1 :: BARS_PER_DAY]
    np.testing.assert_allclose(panel["momentum_1"].values, last_bar_of_each_day)


def test_factor_cal_on_a_resampled_copy_resamples_its_output(
    minute_config: DatasetConfig, tmp_path: Path
):
    daily = _momentum(minute_config, tmp_path).resample("1d", {"momentum_1": "mean"})
    panel = daily.cal().get_features()
    assert panel.sizes["timestamp"] == DAYS
    assert np.isfinite(panel["momentum_1"].values[1:]).all()


def test_factor_resample_saves_beside_the_source_and_reads_it_back(
    minute_config: DatasetConfig, tmp_path: Path
):
    minute = _momentum(minute_config, tmp_path).cal().save(mode="w")
    daily = minute.resample("1d", "last")
    assert daily.store_path == str(tmp_path / "factors" / "momentum_resample_1d.zarr")

    daily.save(mode="w")
    assert xr.open_zarr(daily.store_path).sizes["timestamp"] == DAYS
    assert xr.open_zarr(minute.store_path).sizes["timestamp"] == DAYS * BARS_PER_DAY

    # Without a saved store the resampled copy reads the source and resamples.
    fresh = _momentum(minute_config, tmp_path).resample("1d", "last")
    Path(daily.store_path).rename(tmp_path / "aside.zarr")
    assert fresh.read().get_features().sizes["timestamp"] == DAYS
    # With one, it reads the saved store.
    (tmp_path / "aside.zarr").rename(daily.store_path)
    assert _momentum(minute_config, tmp_path).resample("1d", "last").read().get_features().sizes[
        "timestamp"
    ] == DAYS

    with pytest.raises(ValueError, match="update"):
        daily.update()


def test_factor_resample_config_round_trips(minute_config: DatasetConfig, tmp_path: Path):
    daily = _momentum(minute_config, tmp_path).resample("1d", "last")
    saved = json.loads(json.dumps(daily.get_config()))
    assert saved["resample_freq"] == "1d"
    assert saved["dataset"]["resample_freq"] is None

    rebuilt = load_factor_from_config(saved)
    assert rebuilt.config.resample_freq == "1d"
    assert rebuilt.cal().get_features().sizes["timestamp"] == DAYS


def test_kunquant_factor_resample_and_stream_refusal(
    minute_config_8: DatasetConfig, tmp_path: Path
):
    factor = Alpha158SpotKline(
        FactorConfig(
            window=1,
            dataset=SpotKlineDataset(minute_config_8),
            mode="batch",
            data_columns=["open", "close", "volume"],
            factor_names=["KMID", "VOLUME0"],
            file_path=str(tmp_path / "factors" / "alpha158.zarr"),
            njobs=2,
        )
    )
    daily = factor.cal().resample("1d", {"KMID": "mean", "VOLUME0": "last"})
    assert daily._lib is None
    assert daily.get_features().sizes == {"timestamp": DAYS, "symbol": 8}
    assert factor.get_features().sizes["timestamp"] == DAYS * BARS_PER_DAY
    with pytest.raises(ValueError, match="init_stream"):
        daily.init_stream()


# -- session-based labels -------------------------------------------------------


def test_session_labels_cut_by_session_and_label_daily_bars_at_midnight():
    sessions = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-24").date(), pd.Timestamp("2024-01-25").date()],
            "open": pd.to_datetime(["2024-01-24 14:30", "2024-01-25 14:30"]),
            "close": pd.to_datetime(["2024-01-24 21:00", "2024-01-25 21:00"]),
        }
    )
    bars = pd.to_datetime(
        ["2024-01-24 14:31", "2024-01-24 15:00", "2024-01-24 21:00", "2024-01-25 14:31"]
    ).values

    daily = session_labels(bars, "1d", sessions, "Demo")
    assert daily.astype("datetime64[D]").astype(str).tolist() == [
        "2024-01-24", "2024-01-24", "2024-01-24", "2024-01-25",
    ]

    hourly = session_labels(bars, "1h", sessions, "Demo")
    assert pd.DatetimeIndex(hourly).strftime("%Y-%m-%d %H:%M").tolist() == [
        "2024-01-24 15:30", "2024-01-24 15:30", "2024-01-24 21:00", "2024-01-25 15:30",
    ]

    with pytest.raises(ValueError, match="falls in no trading session"):
        session_labels(pd.to_datetime(["2024-01-24 14:30"]).values, "1d", sessions, "Demo")


def test_nbbo_dataset_labels_bars_by_xnys_session(tmp_path: Path):
    from quantlab.base.config import NbboDatasetConfig
    from quantlab.dataset.nbbo import NbboPanelDataset

    config = NbboDatasetConfig(
        raw_data_dir_path=str(tmp_path / "downloads/us_equity/tick/wrds_taq/wrds"),
        zarr_file_path=str(tmp_path / "nbbo_1m.zarr"),
        reference_dir=str(tmp_path / "_reference"),
        session_start="04:00",
        session_end="20:00",
    )
    ds = NbboPanelDataset(config)
    # Friday 2024-01-26 with an extended close at 20:00 ET = 01:00 UTC Saturday.
    bars = pd.to_datetime(["2024-01-26 09:01", "2024-01-27 01:00"]).values
    labels = ds._resample_labels(bars, "1d")
    assert labels.astype("datetime64[D]").astype(str).tolist() == ["2024-01-26", "2024-01-26"]
