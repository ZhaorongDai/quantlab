"""``one_bar_returns``: the one formula for ``price[t] / price[t - 1] - 1``.

Its rule: NaN on the first bar and wherever either price is missing, the
input's float precision kept, and on a data array the time axis is
``timestamp`` wherever it sits. The call sites' own tests lock that their
outputs did not change.
"""

import numpy as np
import pandas as pd
import xarray as xr

from quantlab.utils.returns import one_bar_returns


def test_the_first_bar_and_a_missing_price_give_nan():
    prices = np.array([[10.0, 20.0], [11.0, np.nan], [12.1, 22.0]])

    returns = one_bar_returns(prices)

    np.testing.assert_allclose(
        returns, [[np.nan, np.nan], [0.1, np.nan], [0.1, np.nan]], equal_nan=True
    )


def test_a_single_series_is_one_column():
    np.testing.assert_allclose(
        one_bar_returns(np.array([4.0, 5.0, 4.0])), [np.nan, 0.25, -0.2], equal_nan=True
    )


def test_the_float_precision_is_kept():
    assert one_bar_returns(np.array([1.0, 2.0], dtype=np.float32)).dtype == np.float32
    assert one_bar_returns(np.array([1, 2])).dtype == np.float64


def test_a_data_array_keeps_its_axes_and_runs_along_timestamp():
    timestamps = pd.bdate_range("2024-01-01", periods=3)
    prices = xr.DataArray(
        [[10.0, 11.0, 12.1], [20.0, 25.0, 20.0]],
        dims=("symbol", "timestamp"),
        coords={"symbol": ["A", "B"], "timestamp": timestamps},
        attrs={"units": "USD"},
    )

    returns = one_bar_returns(prices)

    assert returns.dims == ("symbol", "timestamp")
    xr.testing.assert_identical(returns.symbol, prices.symbol)
    xr.testing.assert_identical(returns, prices / prices.shift(timestamp=1) - 1.0)
