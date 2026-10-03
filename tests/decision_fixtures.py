"""Hand-built price datasets that drive ``DecisionInputs`` from bare panels in the portfolio tests.

``MaskedDataset`` holds constant prices (every order fills, nothing drifts) and
answers ``tradable_bars`` with a given mask, so a test chooses which symbols are
tradable at each bar independently of the prices. ``decide_panel`` runs a rule
over a prediction panel through ``DecisionInputs.weights`` on such a dataset.
"""

import numpy as np
import pandas as pd
import xarray as xr

from quantlab.dataset.memory import FrameDataset
from quantlab.portfolio.decision_inputs import DecisionInputs, rebalance_mask

PRICE = 100.0


class MaskedDataset(FrameDataset):
    """Prices of ``PRICE`` on every bar and symbol of ``tradable``; tradable where it is True."""

    def __init__(self, tradable: xr.DataArray):
        """Hold constant ``open``/``close`` prices on the mask's labels."""
        tradable = tradable.transpose("timestamp", "symbol").astype(bool)
        price = xr.full_like(tradable, PRICE, dtype=np.float64)
        super().__init__(xr.Dataset({"open": price, "close": price}))
        self.mask = tradable

    def tradable_bars(self, prices, fill_column):
        """The given mask on the panel's labels; False off it."""
        return self.mask.reindex(
            timestamp=prices.timestamp.values, symbol=prices.symbol.values, fill_value=False
        )


def decide_panel(rule, predictions: xr.Dataset, tradable=None, rebalance=None) -> xr.Dataset:
    """Return ``DecisionInputs.weights`` of ``rule`` over ``predictions`` on a ``MaskedDataset``.

    ``tradable`` defaults to every symbol everywhere. ``rebalance`` is a mask
    of the bars to decide: all True (every bar, the last included: a padding
    bar is appended and cut off again) or ``rebalance_mask(n, p)`` for some
    ``p``; default all True.
    """
    predictions = predictions.transpose("timestamp", "symbol")
    n_bars = predictions.sizes["timestamp"]
    if tradable is None:
        tradable = xr.ones_like(predictions[list(predictions.data_vars)[0]], dtype=bool)
    rebalance = np.ones(n_bars, dtype=bool) if rebalance is None else np.asarray(rebalance, dtype=bool)
    pad = bool(rebalance.all())
    if pad:
        periods = 1
        after = pd.DatetimeIndex(predictions.timestamp.values)[-1] + pd.offsets.BDay(1)
        predictions = xr.concat(
            [predictions, xr.full_like(predictions.isel(timestamp=[-1]), np.nan).assign_coords(timestamp=[after])],
            dim="timestamp",
        )
        tradable = xr.concat(
            [tradable, xr.zeros_like(tradable.isel(timestamp=[-1])).assign_coords(timestamp=[after])],
            dim="timestamp",
        )
    else:
        periods = next(
            (p for p in range(1, n_bars + 1) if np.array_equal(rebalance_mask(n_bars, p), rebalance)),
            None,
        )
        if periods is None:
            raise ValueError(f"no rebalance period gives the mask {rebalance.tolist()}")
    out = DecisionInputs(
        MaskedDataset(tradable),
        rule,
        fill_column="open",
        valuation_column="close",
        rebalance_periods=periods,
        anchor=predictions.timestamp.values[0],
    ).weights(predictions)
    return out.isel(timestamp=slice(0, n_bars)) if pad else out
