"""The us3000 universe's library components.

- ``EstuConstituentDataset``: membership = a BarraStyle store's ``estu``,
  one interval per run of member bars, the panel ending on the store's last
  bar and following it when the store is extended.
- ``MemberReturn``: ``Return`` blanked where the symbol is not a member at t,
  rebuilt from its config alone.

Every value is SYNTHETIC.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from quantlab.core.component import rebuild
from quantlab.dataset.config import ConstituentDatasetConfig, DatasetConfig
from quantlab.dataset.estu import EstuConstituentDataset
from quantlab.dataset.stock import StockDataset
from quantlab.factor.config import FactorConfig
from quantlab.label.predefined.fret import Return
from quantlab.label.predefined.member_return import MemberReturn


def _estu_store(path, days, mask, symbols=(101, 202)):
    """A BarraStyle-shaped store with only ``estu`` (1.0 member, NaN not)."""
    values = np.where(np.asarray(mask, bool), 1.0, np.nan)  # SYNTHETIC
    xr.Dataset(
        {"estu": (("timestamp", "symbol"), values)},
        coords={"timestamp": pd.DatetimeIndex(days), "symbol": np.asarray(symbols, np.int64)},
    ).to_zarr(path, mode="w")


def _members(tmp_path, store):
    return EstuConstituentDataset(ConstituentDatasetConfig(
        zarr_file_path=str(tmp_path / "membership.zarr"),
        cache_dir=str(tmp_path),
        start_date="2024-01-01",
        kwargs={"barra_store": str(store)},
    ))


DAYS = ["2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09", "2024-01-10"]  # Thu..Wed


def test_membership_is_estu_with_runs_spanning_the_weekend_and_ending_on_the_last_bar(tmp_path):
    store = tmp_path / "barra.zarr"
    # 101 in throughout; 202 in Thu-Fri, out Mon, in again Tue-Wed.
    _estu_store(store, DAYS, [[1, 1], [1, 1], [1, 0], [1, 1], [1, 1]])  # SYNTHETIC

    panel = _members(tmp_path, store).from_raw_data().get_xarray_dataset()["is_member"]

    assert str(panel["timestamp"].values[-1])[:10] == "2024-01-10"
    member = panel.sel(timestamp=["2024-01-06", "2024-01-08", "2024-01-09"])
    assert member.sel(symbol=101).values.tolist() == [True, True, True]
    assert member.sel(symbol=202).values.tolist() == [False, False, True]


def test_membership_follows_the_store_when_it_is_extended(tmp_path):
    store = tmp_path / "barra.zarr"
    _estu_store(store, DAYS[:3], [[1, 1], [1, 1], [1, 1]])  # SYNTHETIC
    members = _members(tmp_path, store)
    members.update()
    _estu_store(store, DAYS, [[1, 1], [1, 1], [1, 1], [1, 0], [1, 0]])  # SYNTHETIC

    panel = _members(tmp_path, store).update().panel("2024-01-08", "2024-01-10")["is_member"]

    assert panel.sel(symbol=202).values.tolist() == [True, False, False]
    assert panel.sel(symbol=101).values.tolist() == [True, True, True]


def test_a_member_return_is_the_return_blanked_outside_membership_and_rebuilds(stock_zarr, tmp_path):
    dataset_config: DatasetConfig = stock_zarr(periods=30)
    prices = xr.open_zarr(dataset_config.zarr_file_path).load()
    first = prices["symbol"].values[0]
    # The first symbol is a member only on the first 10 bars; the others always.
    member = xr.ones_like(prices["adjOpen"], dtype=bool)
    member.loc[{"symbol": first}] = np.arange(prices.sizes["timestamp"]) < 10
    prices.where(member).to_zarr(tmp_path / "members.zarr", mode="w")
    config = FactorConfig(
        warmup_bars=3, dataset=StockDataset(dataset_config), mode="batch",
        data_columns=("adjOpen",), kwargs={"n_forward_periods": 1, "members_store": str(tmp_path / "members.zarr")},
        file_path=str(tmp_path / "label" / "ret_1.zarr"), njobs=2,
    )
    label = MemberReturn(config)
    plain = Return(FactorConfig(**{**config.__dict__, "file_path": str(tmp_path / "plain.zarr")}))

    got = label.compute("2024-01-01", "2024-01-20")["ret_1"]
    expected = plain.compute("2024-01-01", "2024-01-20")["ret_1"]

    stamps = got["timestamp"].values
    inside = stamps < prices["timestamp"].values[10]
    np.testing.assert_array_equal(got.sel(symbol=first).values[inside], expected.sel(symbol=first).values[inside])
    assert got.sel(symbol=first).isel(timestamp=~inside).isnull().all()
    others = [s for s in got["symbol"].values if s != first]
    xr.testing.assert_equal(got.sel(symbol=others), expected.sel(symbol=others))
    rebuilt = rebuild(label.get_config())
    assert type(rebuilt) is MemberReturn
    assert rebuilt.config.factor.config.kwargs["members_store"] == str(tmp_path / "members.zarr")
