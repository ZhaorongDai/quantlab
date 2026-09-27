"""The SpotKline alphas take their z-score window from ``kwargs["zscore_window"]``.

The window is independent of ``warmup_bars``: the warm-up has to cover the
alpha's own lookback plus the z-score window, and with enough of it
``compute(start, end)`` is fully normalized from the first requested bar.
"""

import warnings
from pathlib import Path

import numpy as np
import pytest

from quantlab.base.config import FactorConfig
from quantlab.dataset.spot import SpotKlineDataset
from quantlab.factor.alpha101 import Alpha101SpotKline
from quantlab.factor.alpha158 import Alpha158SpotKline


def _std5(dataset, tmp_path: Path, warmup_bars: int, **kwargs) -> Alpha158SpotKline:
    return Alpha158SpotKline(
        FactorConfig(
            warmup_bars=warmup_bars,
            dataset=dataset,
            mode="batch",
            data_columns=("close",),
            factor_names=("STD5",),
            kwargs=kwargs or None,
            file_path=str(tmp_path / "std5.zarr"),
            njobs=2,
        )
    )


def _whole_history(factor):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return factor.compute("2024-01-01", "2024-12-31")


@pytest.fixture
def dataset(spot_kline_zarr):
    return SpotKlineDataset(spot_kline_zarr(periods=60))


def test_a_sufficient_warm_up_normalizes_the_first_requested_bar(dataset, tmp_path):
    # STD5 needs 4 earlier bars, a 5-bar z-score 4 more.
    factor = _std5(dataset, tmp_path, warmup_bars=8, zscore_window=5)

    panel = factor.compute("2024-01-20", "2024-01-31")

    assert not np.isnan(panel["STD5"].isel(timestamp=0).values).any()
    np.testing.assert_allclose(
        panel["STD5"].values,
        _whole_history(factor)["STD5"].sel(timestamp=slice("2024-01-20", "2024-01-31")).values,
        atol=1e-4,
    )


def test_the_z_score_window_does_not_follow_warmup_bars(dataset, tmp_path):
    short = _std5(dataset, tmp_path / "a", warmup_bars=8, zscore_window=5)
    long = _std5(dataset, tmp_path / "b", warmup_bars=30, zscore_window=5)

    np.testing.assert_allclose(
        short.compute("2024-02-01", "2024-02-20")["STD5"].values,
        long.compute("2024-02-01", "2024-02-20")["STD5"].values,
        atol=1e-4,
    )


def test_the_z_score_window_defaults_independently_of_warmup_bars(dataset, tmp_path):
    factor = _std5(dataset, tmp_path, warmup_bars=30)

    assert factor.zscore_window == Alpha158SpotKline.DEFAULT_ZSCORE_WINDOW
    assert _std5(dataset, tmp_path, 30, zscore_window=7).zscore_window == 7


@pytest.mark.parametrize("cls", [Alpha101SpotKline, Alpha158SpotKline])
@pytest.mark.parametrize("bad", [0, -3, 2.5, "10"])
def test_a_z_score_window_that_is_not_a_positive_integer_is_refused(
    cls, bad, dataset, tmp_path
):
    with pytest.raises(ValueError, match="zscore_window"):
        cls(
            FactorConfig(
                warmup_bars=10,
                dataset=dataset,
                mode="batch",
                data_columns=("close",),
                factor_names=("alpha001",) if cls is Alpha101SpotKline else ("STD5",),
                kwargs={"zscore_window": bad},
                file_path=str(tmp_path / "f.zarr"),
            )
        )
