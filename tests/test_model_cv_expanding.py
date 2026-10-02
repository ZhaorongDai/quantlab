"""Expanding-window walk-forward CV: ``train_cv(..., expanding=True)``.

Every fold trains from the first fold's start up to its test segment. The
test segments, the fold count, the purge and the ``cv_folds.json`` format are
those of the sliding mode, so the two modes compare on the same bars.

The factor and label values are the bar index, so the rows a head receives
in ``_fit_model`` name the bars it was fitted on. Everything is synthetic,
CPU-only and offline.
"""

import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.backtest.predefined.us_equity import USEquityCrossectionSelectStockVectorBt
from quantlab.base.config import CrossSectionBacktestConfig, ModelConfig, TopNConfig
from quantlab.model.library_model import LibraryModel
from quantlab.portfolio.predefined.top_n import TopNConstructor
from tests.backtest_fixtures import make_model, make_stock_dataset, write_price_store
from tests.label_stubs import StubLabel

N_TIMES = 20
SYMBOLS = ["S0", "S1"]
TIMES = np.datetime64("2024-01-01") + np.arange(N_TIMES).astype("timedelta64[D]")
#: 20 bars, train_periods 10: test segments of 2 bars, 5 folds.
TRAIN_PERIODS = 10
N_FOLDS = 5
DATE_KEYS = ("train_start", "train_end", "test_start", "test_end")


def date(i):
    return np.datetime_as_string(TIMES[i], unit="D")


class Panel:
    """A factor or label stand-in whose values are the bar index."""

    def __init__(self, name):
        self.name = name
        index = np.repeat(np.arange(N_TIMES, dtype="float32")[:, None], len(SYMBOLS), 1)
        self._ds = xr.Dataset(
            {name: (("timestamp", "symbol"), index)},
            coords={"timestamp": TIMES, "symbol": SYMBOLS},
        )

    def _get_factor_names(self):
        return [self.name]

    def compute(self, start, end):
        return self._ds.sel(timestamp=slice(start, end))

    read = compute

    def get_config(self):
        return {"name": "Panel", "factor_names": [self.name]}


def bars(rows) -> list[int]:
    return sorted({int(v) for v in np.asarray(rows)[..., 0].ravel()})


#: ``(fold train_start, train bars, validation bars)`` per fit, in fit order.
FITTED: list[tuple[str, list[int], list[int] | None]] = []


@pytest.fixture(autouse=True)
def _reset_fitted():
    FITTED.clear()
    yield
    FITTED.clear()


class RecordingHead(LibraryModel):
    def _init_model(self, num_features, num_labels, hyperparameters):
        return {"num_labels": num_labels}

    def _fit_model(self, train_rows, val_rows):
        FITTED.append(
            (
                self.config.train_start,
                bars(train_rows.y),
                None if val_rows is None else bars(val_rows.y),
            )
        )

    def _forward(self, x):
        return np.repeat(x[..., :1], self.model["num_labels"], axis=-1)


def _model(tmp_path, name, lookahead=0, val_size=0.0):
    return RecordingHead(
        ModelConfig(
            factors=[Panel("f")],
            labels=[StubLabel(Panel("y"), lookahead=lookahead)],
            model_save_dir=str(tmp_path / name),
            factor_data_strategy="cal",
            label_data_strategy="cal",
            start_date=date(0),
            end_date=date(N_TIMES - 1),
            val_size=val_size,
        )
    )


def _manifest(tmp_path, name) -> dict:
    (manifest,) = sorted((tmp_path / name).rglob("cv_folds.json"))
    return json.loads(manifest.read_text())


def _days(record) -> tuple:
    return tuple(np.datetime64(record[k], "D") for k in DATE_KEYS)


def test_every_expanding_fold_starts_where_the_first_fold_starts(tmp_path):
    model = _model(tmp_path, "expanding")
    model.collect()

    results = model.train_cv(train_periods=TRAIN_PERIODS, expanding=True)

    assert len(results) == N_FOLDS
    assert {np.datetime64(r["train_start"], "D") for r in results} == {TIMES[0]}
    # Fold i trains on bars 0 .. 2i + 9 and tests on 2i + 10 .. 2i + 11.
    assert [_days(r) for r in results] == [
        (TIMES[0], TIMES[2 * i + 9], TIMES[2 * i + 10], TIMES[2 * i + 11])
        for i in range(N_FOLDS)
    ]
    assert [fitted for _, fitted, _ in FITTED] == [
        list(range(0, 2 * i + 10)) for i in range(N_FOLDS)
    ]


def test_expanding_and_sliding_test_on_the_same_bars(tmp_path):
    sliding = _model(tmp_path, "sliding")
    sliding.collect()
    sliding_results = sliding.train_cv(train_periods=TRAIN_PERIODS)
    expanding = _model(tmp_path, "expanding")
    expanding.collect()
    expanding_results = expanding.train_cv(train_periods=TRAIN_PERIODS, expanding=True)

    def tests(results):
        return [(r["fold"], r["test_start"], r["test_end"]) for r in results]

    assert tests(expanding_results) == tests(sliding_results)
    assert [r["train_end"] for r in expanding_results] == [
        r["train_end"] for r in sliding_results
    ]
    # The sliding run keeps a fixed-length window; the expanding one does not.
    assert len({r["train_start"] for r in sliding_results}) == N_FOLDS


def test_the_purge_ends_every_expanding_window_lookahead_bars_before_its_test(tmp_path):
    # L = 2: fold i fits on 0 .. 2i + 7, and the manifest records that end.
    model = _model(tmp_path, "models", lookahead=2)
    model.collect()
    model.train_cv(train_periods=TRAIN_PERIODS, expanding=True)

    assert [fitted for _, fitted, _ in FITTED] == [
        list(range(0, 2 * i + 8)) for i in range(N_FOLDS)
    ]
    manifest = _manifest(tmp_path, "models")
    assert manifest["format_version"] == 2
    assert [_days(f) for f in manifest["folds"]] == [
        (TIMES[0], TIMES[2 * i + 7], TIMES[2 * i + 10], TIMES[2 * i + 11])
        for i in range(N_FOLDS)
    ]


def test_the_validation_segment_grows_with_the_expanding_window(tmp_path):
    # val_size 0.2 of fold i's 10 + 2i bar window: its first
    # int(0.8 * (10 + 2i)) bars train and the rest validate.
    model = _model(tmp_path, "models", val_size=0.2)
    model.collect()
    model.train_cv(train_periods=TRAIN_PERIODS, expanding=True)

    for i, (_, train, val) in enumerate(FITTED):
        window = 2 * i + 10
        split = int(window * 0.8)
        assert train == list(range(0, split))
        assert val == list(range(split, window))
    assert [len(val) for _, _, val in FITTED] == [2, 3, 3, 4, 4]


# --------------------------------------------------------------------------
# An expanding run's manifest replays with run_cv
# --------------------------------------------------------------------------

N_BARS = 80
BT_TRAIN_PERIODS = 30
BT_TEST_PERIODS = 6
BT_N_FOLDS = 8
HORIZON = 2


def _day(ts) -> str:
    return pd.Timestamp(str(ts)).strftime("%Y-%m-%d")


def _model_dates(bars_) -> dict:
    return dict(
        start_date=_day(bars_[0]),
        end_date=_day(bars_[N_BARS - 1]),
        train_start=_day(bars_[0]),
        train_end=_day(bars_[BT_TRAIN_PERIODS - 1]),
        test_start=_day(bars_[BT_TRAIN_PERIODS]),
        test_end=_day(bars_[N_BARS - 1]),
    )


def test_an_expanding_manifest_replays_with_run_cv(tmp_path):
    dataset_config = write_price_store(tmp_path, n_bars=N_BARS)
    stamps = xr.open_zarr(dataset_config.zarr_file_path).timestamp.values
    model = make_model(
        tmp_path / "train", dataset_config, n_forward_periods=HORIZON, **_model_dates(stamps)
    )
    model.collect()
    model.train_cv(train_periods=BT_TRAIN_PERIODS, expanding=True)
    (manifest_path,) = sorted((tmp_path / "train" / "models").rglob("cv_folds.json"))
    manifest = json.loads(manifest_path.read_text())
    assert len(manifest["folds"]) == BT_N_FOLDS
    assert {_day(f["train_start"]) for f in manifest["folds"]} == {_day(stamps[0])}

    backtester = USEquityCrossectionSelectStockVectorBt(
        CrossSectionBacktestConfig(
            price_dataset=make_stock_dataset(dataset_config),
            model=make_model(
                tmp_path / "backtest",
                dataset_config,
                n_forward_periods=HORIZON,
                **_model_dates(stamps),
            ),
            model_mode="load",
            cv_project_dir=str(manifest_path.parent),
            start_date=_day(stamps[BT_TRAIN_PERIODS]),
            end_date=_day(stamps[BT_TRAIN_PERIODS + BT_N_FOLDS * BT_TEST_PERIODS - 1]),
            output_dir=str(tmp_path / "runs"),
            rebalance_periods=2,
            constructor=TopNConstructor(TopNConfig(direction="long_only", top_n=2)),
            fees=0.0,
            slippage=0.0,
            init_cash=1_000_000.0,
        )
    )
    result = backtester.run_cv()

    assert [record["fold"] for record in result.folds] == list(range(BT_N_FOLDS))
    for record in result.folds:
        first = BT_TRAIN_PERIODS + record["fold"] * BT_TEST_PERIODS
        np.testing.assert_array_equal(
            record["weights"].timestamp.values.astype("datetime64[ns]"),
            stamps[first : first + BT_TEST_PERIODS].astype("datetime64[ns]"),
        )
        assert record["checkpoint"] == manifest["folds"][record["fold"]]["checkpoint"]
        assert record["metrics"]["in_sample_range"] is None
        assert tuple(record["metrics"]["training_window"])[0] == _day(stamps[0])
