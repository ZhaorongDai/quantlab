"""A shared store's folder records the component that writes it (``quantlab.core.store_folder``).

A store lives in ``<category>/<group>/<stem>/<stem>.zarr``; beside it, ``component.json``
holds the component's ``get_config()`` so the daily data update can rebuild the dataset or
factor without the recipe that defined it, and ``README.md`` says what the store is.
"""

import json

import numpy as np
import pytest
import xarray as xr

from quantlab.core.store_folder import COMPONENT_FILE, load_component, save_component
from quantlab.dataset.memory import FrameDataset
from quantlab.factor.config import PolarsFactorConfig
from tests.backtest_fixtures import PastReturnFactor, write_price_store
from quantlab.dataset.stock import StockDataset


@pytest.fixture
def factor(tmp_path):
    return PastReturnFactor(PolarsFactorConfig(
        warmup_bars=5, dataset=StockDataset(write_price_store(tmp_path / "prices")), kwargs={"n": 3},
        file_path=str(tmp_path / "factors" / "past_ret_3" / "past_ret_3.zarr"),
    ))


def test_a_saved_component_rebuilds_equal(factor, tmp_path):
    folder = tmp_path / "factors" / "past_ret_3"

    save_component(factor, folder)

    assert (folder / COMPONENT_FILE).is_file()
    rebuilt = load_component(folder)
    assert type(rebuilt) is type(factor) and rebuilt == factor


def test_the_readme_is_written_when_given_and_kept_otherwise(factor, tmp_path):
    folder = tmp_path / "factors" / "past_ret_3"

    save_component(factor, folder, readme="# past_ret_3\nThree-bar past return.")
    save_component(factor, folder)

    assert (folder / "README.md").read_text() == "# past_ret_3\nThree-bar past return.\n"


def test_a_folder_without_a_component_names_the_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match=r"component\.json"):
        load_component(tmp_path / "nowhere")


def test_a_component_held_in_memory_is_refused(tmp_path):
    panel = xr.Dataset(
        {"close": (("timestamp", "symbol"), np.ones((2, 1)))},
        coords={"timestamp": np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[ns]"), "symbol": ["A"]},
    )

    with pytest.raises(ValueError, match="in memory"):
        save_component(FrameDataset(panel), tmp_path / "frame")
    assert not (tmp_path / "frame" / COMPONENT_FILE).exists()
