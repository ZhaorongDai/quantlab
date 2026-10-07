"""Bad prints: a causal rule and a dataset view masking them (#223).

A bad print is a bar whose adjusted close moves more than ``jump`` times (up
or down) from the symbol's last priced bar of the ``lookback`` bars before,
on a volume below ``volume_ratio`` times the mean volume of those bars. The
rule reads nothing after the bar, so a live feed and a backtest flag the same
bars. ``BadPrintMaskedDataset`` merges datasets like ``MergedDataset`` and
sets the price variables of a flagged bar to NaN, so returns into and out of
it are missing.
"""

import dataclasses

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.dataset._support.cleaning import bad_print_mask
from quantlab.dataset.bad_prints import BadPrintMaskedDataset
from quantlab.dataset.memory import FrameDataset
from quantlab.utils.returns import one_bar_returns

T = 40
DAYS = pd.bdate_range("2024-01-01", periods=T)
#: Symbol positions: a bad print, a real jump on heavy volume, a split, a quiet one.
BAD, JUMP, SPLIT, QUIET = range(4)
SPIKE_BAR = 25


def _arrays():
    """Return ``(adjusted close, raw close, adjusted volume)`` of four symbols."""
    rng = np.random.default_rng(223)
    adj = 10.0 * np.exp(np.cumsum(rng.normal(0, 0.01, size=(T, 4)), axis=0))
    volume = rng.uniform(1_000, 2_000, size=(T, 4))
    # One bar at a hundredth of the price on ordinary volume, back the next bar.
    adj[SPIKE_BAR, BAD] = adj[SPIKE_BAR - 1, BAD] / 100
    # A forty-fold rise that stays, on a hundred times the volume.
    adj[SPIKE_BAR:, JUMP] *= 40
    volume[SPIKE_BAR, JUMP] *= 100
    raw = adj.copy()
    # A 1:20 reverse split: the raw close jumps, the adjusted one does not.
    raw[SPIKE_BAR:, SPLIT] *= 20
    return adj, raw, volume


def _panel():
    adj, raw, volume = _arrays()
    coords = {"timestamp": DAYS, "symbol": ["BAD", "JUMP", "SPLIT", "QUIET"]}
    dims = ("timestamp", "symbol")
    return xr.Dataset(
        {
            "adjClose": (dims, adj),
            "close": (dims, raw),
            "adjVolume": (dims, volume),
            "volume": (dims, volume),
        },
        coords=coords,
    )


def _caps():
    adj, _, _ = _arrays()
    coords = {"timestamp": DAYS, "symbol": ["BAD", "JUMP", "SPLIT", "QUIET"]}
    return xr.Dataset({"marketcap": (("timestamp", "symbol"), adj * 1e6)}, coords=coords)


def _view(**params):
    return BadPrintMaskedDataset([FrameDataset(_panel()), FrameDataset(_caps())], **params)


# -- the rule ----------------------------------------------------------------


def test_the_rule_flags_a_low_volume_spike_and_nothing_else():
    adj, _, volume = _arrays()
    flags = bad_print_mask(adj, volume)
    assert flags[SPIKE_BAR, BAD]
    # The bar back is a 100-fold move from the bad print on ordinary volume.
    assert flags[SPIKE_BAR + 1, BAD]
    assert not flags[:, JUMP].any(), "a jump on heavy volume is real"
    assert not flags[:, SPLIT].any(), "a split leaves the adjusted close smooth"
    assert not flags[:, QUIET].any()
    assert flags.sum() == 2


def test_the_rule_reads_nothing_after_the_bar():
    adj, _, volume = _arrays()
    whole = bad_print_mask(adj, volume)
    for end in (SPIKE_BAR, SPIKE_BAR + 1, T - 5):
        cut = bad_print_mask(adj[: end + 1], volume[: end + 1])
        np.testing.assert_array_equal(cut, whole[: end + 1])


def test_the_rule_reads_only_its_lookback():
    """Rows from ``t - lookback`` on decide row ``t``: any window gives the same flags."""
    adj, _, volume = _arrays()
    whole = bad_print_mask(adj, volume, lookback=10)
    tail = bad_print_mask(adj[SPIKE_BAR - 10 :], volume[SPIKE_BAR - 10 :], lookback=10)
    np.testing.assert_array_equal(tail[10:], whole[SPIKE_BAR:])


def test_a_gap_compares_with_the_last_priced_bar_of_the_lookback():
    adj, _, volume = _arrays()
    adj[SPIKE_BAR - 3 : SPIKE_BAR, BAD] = np.nan
    assert bad_print_mask(adj, volume, lookback=5)[SPIKE_BAR, BAD]
    # No priced bar in the lookback: nothing to compare with, no flag.
    assert not bad_print_mask(adj, volume, lookback=3)[SPIKE_BAR, BAD]


def test_a_jump_with_no_volume_history_is_flagged():
    adj, _, volume = _arrays()
    volume[SPIKE_BAR - 20 : SPIKE_BAR, BAD] = 0.0
    volume[SPIKE_BAR, BAD] = 500.0
    assert bad_print_mask(adj, volume)[SPIKE_BAR, BAD]


def test_the_thresholds_are_parameters():
    adj, _, volume = _arrays()
    # At 200 times the mean volume, the heavy-volume jump is low volume too.
    assert bad_print_mask(adj, volume, volume_ratio=200.0)[SPIKE_BAR, JUMP]
    # A 100-fold move is below a 1000-fold jump.
    assert not bad_print_mask(adj, volume, jump=1000.0).any()


@pytest.mark.parametrize(
    "params, field",
    [({"jump": 1.0}, "jump"), ({"volume_ratio": 0.0}, "volume_ratio"), ({"lookback": 0}, "lookback")],
)
def test_the_rule_refuses_invalid_parameters(params, field):
    adj, _, volume = _arrays()
    with pytest.raises(ValueError, match=field):
        bad_print_mask(adj, volume, **params)


# -- the dataset view ----------------------------------------------------------


def test_the_view_masks_the_price_variables_of_a_flagged_bar():
    panel = _view().panel(DAYS[0], DAYS[-1])
    plain = _panel().reindex(symbol=panel["symbol"].values)
    for name in ("adjClose", "close", "marketcap"):
        assert np.isnan(panel[name].sel(symbol="BAD", timestamp=DAYS[SPIKE_BAR]).item()), name
        assert np.isnan(panel[name].sel(symbol="BAD", timestamp=DAYS[SPIKE_BAR + 1]).item()), name
    # Volume stays; every other cell is the input's.
    xr.testing.assert_equal(panel["volume"], plain["volume"])
    unflagged = panel.drop_sel(timestamp=DAYS[SPIKE_BAR : SPIKE_BAR + 2])
    xr.testing.assert_equal(
        unflagged["adjClose"], plain["adjClose"].drop_sel(timestamp=DAYS[SPIKE_BAR : SPIKE_BAR + 2])
    )
    xr.testing.assert_equal(panel["adjClose"].sel(symbol="JUMP"), plain["adjClose"].sel(symbol="JUMP"))


def test_returns_into_and_out_of_a_bad_print_are_missing():
    returns = one_bar_returns(_view().panel(DAYS[0], DAYS[-1])["adjClose"].sel(symbol="BAD").values)
    assert np.isnan(returns[SPIKE_BAR : SPIKE_BAR + 3]).all()
    assert np.isfinite(returns[SPIKE_BAR + 3 :]).all()
    assert np.nanmax(np.abs(returns)) < 0.1


def test_a_window_reads_the_lookback_before_it():
    """The bar after the spike flags the same whether or not the spike is in the window."""
    view = _view()
    whole = view.panel(DAYS[0], DAYS[-1])
    for start in (SPIKE_BAR, SPIKE_BAR + 1, SPIKE_BAR + 2):
        part = view.panel(DAYS[start], DAYS[-1])
        xr.testing.assert_identical(part, whole.sel(timestamp=slice(DAYS[start], None)))


def test_the_view_keeps_the_requested_variables_and_symbols():
    panel = _view().panel(DAYS[0], DAYS[-1], symbols=["BAD", "QUIET"], variables=["adjClose"])
    assert list(panel.data_vars) == ["adjClose"]
    assert panel["symbol"].values.tolist() == ["BAD", "QUIET"]
    assert np.isnan(panel["adjClose"].sel(symbol="BAD", timestamp=DAYS[SPIKE_BAR]).item())


def test_the_flags_are_readable():
    flags = _view().bad_prints(DAYS[0], DAYS[-1])
    assert flags.dims == ("timestamp", "symbol")
    assert int(flags.sum()) == 2
    assert bool(flags.sel(symbol="BAD", timestamp=DAYS[SPIKE_BAR]))


def test_the_view_is_built_from_its_config_and_copies_with_its_parameters():
    view = _view(jump=4.0, volume_ratio=10.0, lookback=15)
    copied = view.copy()
    assert (copied.config.jump, copied.config.volume_ratio, copied.config.lookback) == (4.0, 10.0, 15)
    assert copied.datasets[0] is not view.datasets[0]
    assert view.get_config()["volume_ratio"] == 10.0
    again = BadPrintMaskedDataset(dataclasses.replace(view.config))
    xr.testing.assert_identical(again.panel(DAYS[0], DAYS[-1]), view.panel(DAYS[0], DAYS[-1]))


def test_the_view_refuses_invalid_parameters():
    with pytest.raises(ValueError, match="jump"):
        _view(jump=0.5)
    with pytest.raises(ValueError, match="lookback"):
        _view(lookback=0)


def test_the_view_holds_no_store():
    with pytest.raises(ValueError, match="view"):
        _view().save()


def test_the_view_rebuilds_from_its_saved_config(tmp_path):
    from quantlab.core.component import rebuild
    from quantlab.dataset.config import FrameDatasetConfig

    inputs = []
    for name, panel in (("prices", _panel()), ("caps", _caps())):
        path = tmp_path / f"{name}.zarr"
        FrameDataset(panel).to_zarr(path)
        inputs.append(FrameDataset(FrameDatasetConfig(zarr_file_path=str(path))))
    view = BadPrintMaskedDataset(inputs, jump=4.0, lookback=15)
    rebuilt = rebuild(view.get_config())
    assert type(rebuilt) is BadPrintMaskedDataset
    assert (rebuilt.config.jump, rebuilt.config.lookback) == (4.0, 15)
    xr.testing.assert_identical(rebuilt.panel(DAYS[0], DAYS[-1]), view.panel(DAYS[0], DAYS[-1]))
