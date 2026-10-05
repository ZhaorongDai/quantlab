"""A model takes labels only in ``labels`` and never as features.

A label is anything with ``lookahead_bars()``, such as ``Forward`` and
``Return``: it reads bars after t, so as a feature it would leak the future,
and a factor without it would be an unshifted target.
"""

from pathlib import Path

import pytest

from quantlab.base.config import ModelConfig
from quantlab.factor.config import FactorConfig, PolarsFactorConfig
from quantlab.label.config import ForwardConfig
from quantlab.dataset.config import DatasetConfig
from quantlab.dataset.spot import SpotKlineDataset
from quantlab.dataset.stock import StockDataset
from quantlab.factor.predefined.momentum import Momentum
from quantlab.label.forward import Forward
from quantlab.label.predefined.fret import Return
from quantlab.model.predefined.xgb import XGBoostRegressor


def _momentum(dataset_config: DatasetConfig) -> Momentum:
    return Momentum(
        PolarsFactorConfig(
            warmup_bars=5, dataset=SpotKlineDataset(dataset_config), kwargs={"n": 5}
        )
    )


def _return(dataset_config: DatasetConfig, tmp_path: Path) -> Return:
    return Return(
        FactorConfig(
            warmup_bars=0,
            dataset=StockDataset(dataset_config),
            mode="batch",
            data_columns=("adjOpen",),
            kwargs={"n_forward_periods": 1},
            file_path=str(tmp_path / "ret.zarr"),
            njobs=2,
        )
    )


def _model(tmp_path: Path, factors, labels) -> XGBoostRegressor:
    return XGBoostRegressor(
        ModelConfig(
            factors=factors,
            labels=labels,
            model_save_dir=str(tmp_path / "models"),
            factor_data_strategy="cal",
            label_data_strategy="cal",
        )
    )


@pytest.fixture
def dataset_config(stock_zarr):
    return stock_zarr(periods=60)


@pytest.fixture
def spot_config(spot_kline_zarr):
    return spot_kline_zarr(periods=60)


def test_a_model_takes_factors_as_features_and_labels_as_labels(dataset_config, spot_config, tmp_path):
    label = Forward(ForwardConfig(factor=_momentum(spot_config), span=5))

    model = _model(tmp_path, [_momentum(spot_config)], [label, _return(dataset_config, tmp_path)])

    assert len(model.config.labels) == 2


def test_a_model_refuses_a_label_among_its_factors(dataset_config, spot_config, tmp_path):
    label = _return(dataset_config, tmp_path)

    with pytest.raises(TypeError, match=r"factors\[1\] is the label Return"):
        _model(tmp_path, [_momentum(spot_config), label], [label])


def test_a_model_refuses_a_factor_among_its_labels(dataset_config, spot_config, tmp_path):
    factor = _momentum(spot_config)

    with pytest.raises(TypeError, match=r"labels\[0\] is Momentum, which is not a label"):
        _model(tmp_path, [factor], [factor])
