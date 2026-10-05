"""``Volatility`` is a ``Forward`` label of span-scale open-to-open volatility.

At bar t it is the sample standard deviation of the one-bar open-to-open
returns ``adjOpen[k] / adjOpen[k - 1] - 1`` for ``k`` in ``t + 2 .. t + n + 1``,
times ``sqrt(n)``: the same opens ``t + 1 .. t + n + 1`` that ``Return`` of
the same span reads, so the two pair up.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from quantlab.base.config import FactorConfig, ModelConfig
from quantlab.dataset.config import DatasetConfig
from quantlab.dataset.stock import StockDataset
from quantlab.label.forward import Forward
from quantlab.label.predefined.fret import Return, Volatility, _TrailingOpenVolatility
from quantlab.model.predefined.xgb import XGBoostRegressor
from quantlab.core.component import rebuild


def _config(dataset_config: DatasetConfig, tmp_path: Path, n: int, name: str = "vol") -> FactorConfig:
    return FactorConfig(
        warmup_bars=n + 1,
        dataset=StockDataset(dataset_config),
        mode="batch",
        data_columns=("adjOpen",),
        kwargs={"n_forward_periods": n},
        file_path=str(tmp_path / "label" / f"{name}_{n}.zarr"),
        njobs=2,
    )


@pytest.fixture
def dataset_config(stock_zarr):
    """60 daily bars from 2024-01-01 to 2024-02-29."""
    return stock_zarr(periods=60)


def _opens(dataset_config: DatasetConfig) -> np.ndarray:
    return (
        xr.open_zarr(dataset_config.zarr_file_path)["adjOpen"]
        .transpose("timestamp", "symbol")
        .values.astype(float)
    )


def _hand_volatility(opens: np.ndarray, n: int) -> np.ndarray:
    """``std(adjOpen[k] / adjOpen[k - 1] - 1, k = t+2..t+n+1, ddof=1) * sqrt(n)``."""
    one_bar = np.full_like(opens, np.nan)
    one_bar[1:] = opens[1:] / opens[:-1] - 1
    out = np.full_like(opens, np.nan)
    for t in range(len(opens)):
        window = one_bar[t + 2 : t + n + 2]
        if len(window) == n:
            out[t] = window.std(axis=0, ddof=1) * np.sqrt(n)
    return out


def _values(panel: xr.Dataset, name: str) -> np.ndarray:
    return panel[name].transpose("timestamp", "symbol").values


@pytest.mark.parametrize("n", [2, 3, 5, 10])
def test_volatility_matches_a_hand_computation_from_open_prices(dataset_config, tmp_path, n):
    label = Volatility(_config(dataset_config, tmp_path, n))

    got = _values(label.compute("2024-01-01", "2024-02-29"), f"vol_{n}")

    expected = _hand_volatility(_opens(dataset_config), n)
    # KunQuant computes in float32.
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-6)
    assert np.isnan(got[-(n + 1) :]).all()
    assert not np.isnan(got[: -(n + 1)]).any()


def test_volatility_shares_return_conventions(dataset_config, tmp_path):
    vol = Volatility(_config(dataset_config, tmp_path, 5))
    ret = Return(_config(dataset_config, tmp_path, 5, name="ret"))

    assert isinstance(vol, Forward)
    assert vol.lookahead_bars() == ret.lookahead_bars() == 6
    assert vol.span_bars() == ret.span_bars() == 5
    assert vol.get_factor_names() == ("vol_5",)


def test_volatility_refuses_a_span_below_two(dataset_config, tmp_path):
    with pytest.raises(ValueError, match="n_forward_periods"):
        Volatility(_config(dataset_config, tmp_path, 1))


def test_volatility_is_nan_where_the_window_misses_an_open(stock_zarr, tmp_path):
    config = stock_zarr(symbols=["AAPL", "MSFT"], periods=40)
    panel = xr.open_zarr(config.zarr_file_path).load()
    opens = panel["adjOpen"].transpose("timestamp", "symbol").values.copy()
    opens[:10, 1] = np.nan  # MSFT lists on bar 10
    panel["adjOpen"] = (("timestamp", "symbol"), opens)
    panel.to_zarr(config.zarr_file_path, mode="w")
    n = 3

    got = _values(Volatility(_config(config, tmp_path, n)).compute("2024-01-01", "2024-02-09"), f"vol_{n}")

    expected = _hand_volatility(opens.astype(float), n)
    np.testing.assert_array_equal(np.isnan(got), np.isnan(expected))
    assert np.isnan(got[:9, 1]).all()
    np.testing.assert_allclose(got[~np.isnan(expected)], expected[~np.isnan(expected)], rtol=1e-4)


def test_volatility_rebuilds_from_its_factor_config(dataset_config, tmp_path):
    label = Volatility(_config(dataset_config, tmp_path, 3))

    config = json.loads(json.dumps(label.get_config()))

    assert config["name"] == "quantlab.label.predefined.fret.Volatility"
    assert rebuild(config) == label


def test_a_model_trains_on_a_volatility_label(dataset_config, tmp_path, monkeypatch):
    # The trailing volatility is the natural feature of a volatility model.
    trailing = _TrailingOpenVolatility(_config(dataset_config, tmp_path, 5, name="trailing"))
    model = XGBoostRegressor(
        ModelConfig(
            factors=[trailing],
            labels=[Volatility(_config(dataset_config, tmp_path, 3))],
            model_save_dir=str(tmp_path / "models"),
            factor_data_strategy="cal",
            label_data_strategy="cal",
            start_date="2024-01-10",
            end_date="2024-02-20",
            train_start="2024-01-10",
            train_end="2024-02-05",
            test_start="2024-02-06",
            test_end="2024-02-20",
            hyperparameters={"num_boost_round": 5},
        )
    )

    model.collect()
    model.train()

    assert list(model.get_label_names()) == ["vol_3"]
    prediction = model.predict_window("2024-02-06", "2024-02-20")
    assert np.isfinite(prediction["vol_3"].values).any()
