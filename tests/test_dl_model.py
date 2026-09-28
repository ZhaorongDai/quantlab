"""`DLModel` trains on one cross-section per step (issue #39, ADR 0006).

A deep head maps `[S_t, N, F]` to `[S_t, L]`: the symbols with a finite
feature at a bar, each with its last N bars. The base class builds the
windows, drops NaN labels from the loss (never from the input), transforms
the target per bar, calls the head's stop hooks and reports the shared
first-label metrics on the raw label.

What turns this file red:

- a symbol that joins, leaves, or was never seen in training gets no
  prediction, or an absent one gets a finite prediction;
- a NaN label is trained on as a value, or its symbol is dropped from the
  cross-section the other symbols see;
- a window row before a symbol's history is not zero, or clipping is ignored;
- the model's warm-up does not make a short request predict like a long one;
- a target transform is not per bar, or metrics see the transformed target;
- the stop hooks are not called per fit and per epoch, or the threshold
  helper stops at the wrong epoch;
- a hook default is missing, or a hook the head overrides is not the one used;
- a DL `train()` writes no `metrics.json`.

Everything is synthetic, CPU-only and offline.
"""

import json
from pathlib import Path

import KunQuant.ops as op
import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr
from KunQuant.Op import Builder, Input, Output
from KunQuant.Stage import Function

from quantlab.base.config import DLConfig, FactorConfig, ForwardConfig
from quantlab.base.data import InsufficientHistoryError
from quantlab.base.factor import FactorKunQuant
from quantlab.base.model import BaseModel, DLModel
from quantlab.dataset.spot import SpotKlineDataset
from quantlab.dl_model.training import (
    CrossSectionBatch,
    TrainLossThreshold,
    cs_rank_norm,
    cs_zscore,
    drop_extreme,
    masked_mse,
)
from quantlab.label.forward import Forward
from quantlab.utils.metrics import regression_panel_metrics
from tests.dl_heads import MeanContextHead, RecordingHead
from tests.label_stubs import StubLabel

N_TIMES = 40
TIMES = pd.date_range("2024-01-01", periods=N_TIMES, freq="D").values
SYMBOLS = [f"S{i}" for i in range(6)]


def _day(i: int) -> str:
    return str(np.datetime_as_string(TIMES[i], unit="D"))


class FakeRecorder:
    """Records what the model writes to a W&B run."""

    def __init__(self, name: str):
        self.name = name
        self.summary: dict = {}
        self.logged: list[dict] = []

    def log(self, data, step=None):
        self.logged.append(dict(data))

    def finish(self):
        pass


@pytest.fixture
def recorders(monkeypatch) -> list[FakeRecorder]:
    created: list[FakeRecorder] = []

    def fake_init_wandb(self, project_name, experiment_name):
        created.append(FakeRecorder(experiment_name))
        self._wandb_recorder = created[-1]

    monkeypatch.setattr(BaseModel, "_init_wandb", fake_init_wandb)
    return created


class Calendar:
    """The one dataset method a model's warm-up calls: `bar_before`."""

    def __init__(self, times):
        self.times = pd.DatetimeIndex(times)

    def bar_before(self, date, n):
        position = int(self.times.searchsorted(pd.Timestamp(date), side="left"))
        if n == 0:
            return pd.Timestamp(date)
        if position < n:
            raise InsufficientHistoryError("short", available=position, requested=n)
        return self.times[position - n]


class Config:
    def __init__(self, dataset):
        self.dataset = dataset


class Panel:
    """A factor stand-in over a fixed `[T, S]` panel on the `TIMES` calendar."""

    def __init__(self, arrays: dict[str, np.ndarray], symbols=SYMBOLS):
        self._ds = xr.Dataset(
            {name: (("timestamp", "symbol"), values) for name, values in arrays.items()},
            coords={"timestamp": TIMES[: len(next(iter(arrays.values())))],
                    "symbol": list(symbols)},
        )
        self.config = Config(Calendar(self._ds.timestamp.values))

    def _get_factor_names(self):
        return list(self._ds.data_vars)

    def compute(self, start, end):
        return self._ds.sel(timestamp=slice(start, end))

    read = compute

    def get_config(self):
        return {"name": "Panel", "factor_names": self._get_factor_names()}


def _features(seed=0, n_symbols=len(SYMBOLS)) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {name: rng.standard_normal((N_TIMES, n_symbols)) for name in ("f_a", "f_b")}


def _label_of(features: dict[str, np.ndarray], seed=1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return features["f_a"] + 0.1 * rng.standard_normal(features["f_a"].shape)


def _model(tmp_path: Path, features, label, *, cls=MeanContextHead, symbols=SYMBOLS,
           name="ckpt", **overrides):
    hp = overrides.pop("hyperparameters", {})
    kwargs = dict(
        factors=[Panel(features, symbols)],
        labels=[StubLabel(Panel({"ret": label}, symbols))],
        model_save_dir=str(tmp_path / name),
        factor_data_strategy="cal",
        label_data_strategy="cal",
        start_date=_day(0),
        end_date=_day(N_TIMES - 1),
        train_start=_day(0),
        train_end=_day(29),
        test_start=_day(30),
        test_end=_day(N_TIMES - 1),
        epochs=3,
        lr=1e-2,
        hyperparameters=hp,
    )
    kwargs.update(overrides)
    return cls(DLConfig(**kwargs)).collect()


def _weights(model) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in model.model.state_dict().items()}


def _same_weights(a, b) -> bool:
    return all(torch.equal(a[k], b[k]) for k in a)


def _feature_panel(features, symbols=SYMBOLS) -> xr.Dataset:
    return Panel(features, symbols)._ds


# ---------------------------------------------------------------------------
# The cross-section changes between bars
# ---------------------------------------------------------------------------


def test_a_changing_cross_section_trains_and_every_present_symbol_is_predicted(
    tmp_path, recorders
):
    features = _features()
    for values in features.values():
        values[:20, 4] = np.nan  # S4 joins at bar 20
        values[25:, 5] = np.nan  # S5 leaves at bar 25
    label = _label_of(features)
    model = _model(tmp_path, features, label)
    model.train()

    # A symbol never seen in training, present on the last five bars only.
    wider = {k: np.concatenate([v, np.full((N_TIMES, 1), np.nan)], axis=1)
             for k, v in _features(seed=5).items()}
    for k, v in features.items():
        wider[k][:, :6] = v
        wider[k][35:, 6] = np.random.default_rng(9).standard_normal(5)
    out = model.predict_panel(_feature_panel(wider, SYMBOLS + ["NEW"]))

    present = np.isfinite(np.stack(list(wider.values()), axis=-1)).any(axis=-1)
    predicted = np.isfinite(out["ret"].sel(symbol=SYMBOLS + ["NEW"]).values)
    np.testing.assert_array_equal(predicted, present)
    assert predicted[35:, 6].all()


def test_a_missing_label_adds_no_loss_but_its_symbol_is_context(tmp_path, recorders):
    features = _features()
    label = _label_of(features)
    label[:, 2] = np.nan  # S2 never has a label

    with_context = _model(tmp_path, features, label, name="a", val_size=0.0)
    with_context.train()

    no_features = {k: v.copy() for k, v in features.items()}
    for values in no_features.values():
        values[:, 2] = np.nan
    without = _model(tmp_path, no_features, label, name="b", val_size=0.0)
    without.train()

    zero = label.copy()
    zero[:, 2] = 0.0
    as_zero = _model(tmp_path, features, zero, name="c", val_size=0.0)
    as_zero.train()

    context = _weights(with_context)
    assert all(torch.isfinite(v).all() for v in context.values())
    # S2's features reach the other symbols during training ...
    assert not _same_weights(context, _weights(without))
    # ... and its NaN label is not trained on as a zero return.
    assert not _same_weights(context, _weights(as_zero))

    # At prediction too, S2's features change S0's output.
    panel = _feature_panel(features)
    moved = panel.copy(deep=True)
    moved["f_a"].values[:, 2] += 5.0
    before = with_context.predict_panel(panel)["ret"].values[:, 0]
    after = with_context.predict_panel(moved)["ret"].values[:, 0]
    assert not np.allclose(before, after)


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("clip, expected", [(True, 3.0), (False, 10.0)])
def test_window_rows_before_a_symbols_history_are_zero_and_values_clip(
    tmp_path, recorders, clip, expected
):
    features = {"f_a": np.ones((N_TIMES, 2)), "f_b": np.ones((N_TIMES, 2))}
    features["f_a"][:5, 1] = np.nan
    features["f_b"][:5, 1] = np.nan
    features["f_a"][5:, 1] = 10.0
    label = np.random.default_rng(0).standard_normal((N_TIMES, 2))
    model = _model(tmp_path, features, label, cls=RecordingHead, symbols=["S0", "S1"],
                   epochs=1, hyperparameters={"window_bars": 3, "clip": clip})
    model.train()
    model.model.inputs.clear()

    model.predict_panel(_feature_panel(features, ["S0", "S1"]))

    inputs = model.model.inputs  # one call per bar, in bar order
    assert len(inputs) == N_TIMES
    first = inputs[0]  # bar 0: only S0, two rows before the panel starts
    assert first.shape == (1, 3, 2)
    np.testing.assert_array_equal(first[0, :2].numpy(), 0.0)
    np.testing.assert_array_equal(first[0, 2].numpy(), 1.0)
    joined = inputs[5]  # bar 5: S1's first bar
    assert joined.shape == (2, 3, 2)
    np.testing.assert_array_equal(joined[1, :2].numpy(), 0.0)
    np.testing.assert_array_equal(joined[1, 2].numpy(), [expected, 1.0])


# ---------------------------------------------------------------------------
# Warm-up, on a real factor over a real dataset calendar
# ---------------------------------------------------------------------------


class MaDeviation(FactorKunQuant):
    def _get_factor_names(self):
        return ("ma_dev_5",)

    def _get_factor_func(self):
        builder = Builder()
        with builder:
            close = Input("close")
            Output(op.SubConst(op.Div(close, op.WindowedAvg(close, 5)), 1.0), "ma_dev_5")
        return Function(builder.ops)


class OneBarReturn(FactorKunQuant):
    def _get_factor_names(self):
        return ("ret_1",)

    def _get_factor_func(self):
        builder = Builder()
        with builder:
            close = Input("close")
            Output(op.SubConst(op.Div(close, op.BackRef(close, 1)), 1.0), "ret_1")
        return Function(builder.ops)


def _kun(cls, dataset, tmp_path, name):
    return cls(FactorConfig(
        warmup_bars=5, dataset=dataset, mode="batch", data_columns=("close",),
        file_path=str(tmp_path / "factors" / f"{name}.zarr"), njobs=2,
    ))


@pytest.fixture
def warm_model(spot_kline_zarr, tmp_path, recorders):
    dataset = SpotKlineDataset(spot_kline_zarr(periods=60))
    factor = _kun(MaDeviation, dataset, tmp_path, "ma_dev")
    label = Forward(ForwardConfig(factor=_kun(OneBarReturn, dataset, tmp_path, "ret"),
                                  span=1, delay=0))

    def build(start="2024-01-20", strategy="cal"):
        return MeanContextHead(DLConfig(
            factors=[factor], labels=[label], model_save_dir=str(tmp_path / "ckpt"),
            factor_data_strategy=strategy, label_data_strategy=strategy,
            start_date=start, end_date="2024-02-25",
            train_start=start, train_end="2024-02-10",
            test_start="2024-02-11", test_end="2024-02-25",
            epochs=2, hyperparameters={"window_bars": 5},
        ))

    build.factor, build.label = factor, label
    return build


def test_training_collect_starts_the_features_the_warm_up_earlier(warm_model):
    model = warm_model().collect()

    panel = model.data_backend.get_xarray_dataset()
    assert str(panel.timestamp.values[0])[:10] == "2024-01-16"  # 4 bars before
    first_label = panel["ret_1"].sel(timestamp="2024-01-20")
    assert np.isfinite(first_label).all()
    assert np.isnan(panel["ret_1"].sel(timestamp="2024-01-16")).all()


def test_the_warm_up_makes_a_short_request_predict_like_a_long_panel(warm_model):
    trained = warm_model().collect()
    checkpoint = trained.train()

    short = warm_model(start="2024-02-12").load(checkpoint).collect()
    panel = short.data_backend.get_xarray_dataset()
    assert str(panel.timestamp.values[0])[:10] == "2024-02-08"

    first = dict(timestamp="2024-02-12")
    features = short.get_factor_names()
    expected = trained.predict_panel(trained.data_backend.get_xarray_dataset()[features])
    got = short.predict_panel(panel[features])
    xr.testing.assert_allclose(got["ret_1"].sel(**first), expected["ret_1"].sel(**first))
    # Without the warm-up bars the first window is short and the answer differs.
    cut = short.predict_panel(panel[features].sel(timestamp=slice("2024-02-12", None)))
    assert not np.allclose(cut["ret_1"].sel(**first), expected["ret_1"].sel(**first))


def test_train_cv_runs_a_windowed_head_and_folds_start_after_the_warm_up(warm_model):
    model = warm_model().collect()

    results = model.train_cv(train_periods=15)

    assert len(results) == 7
    assert results[0]["train_start"][:10] == "2024-01-20"
    for result in results:
        assert Path(result["checkpoint"]).is_file()
        assert np.isfinite(result["test_mse"])


def test_a_short_history_before_the_start_warns(warm_model):
    with pytest.warns(UserWarning, match="4 warm-up bar"):
        warm_model(start="2024-01-02").collect()


def test_a_read_store_shorter_than_the_warm_up_warns_and_starts_at_the_store(warm_model):
    warm_model.factor.build("2024-01-18", "2024-02-25")
    warm_model.label.build("2024-01-18", "2024-02-25")

    with pytest.warns(UserWarning, match="store starts at 2024-01-18"):
        model = warm_model(strategy="read").collect()

    assert str(model.data_backend.get_xarray_dataset().timestamp.values[0])[:10] == "2024-01-18"


# ---------------------------------------------------------------------------
# Target transforms
# ---------------------------------------------------------------------------


def test_rank_is_qlib_csranknorm_and_zscore_is_the_sample_zscore():
    y = np.array([[0.4, 1.0], [np.nan, 2.0], [0.1, 2.0], [0.4, np.nan], [0.9, 7.0]])
    pct = pd.DataFrame(y).rank(pct=True).to_numpy()
    np.testing.assert_allclose(cs_rank_norm(torch.tensor(y)).numpy(), (pct - 0.5) * 3.46)

    frame = pd.DataFrame(y)
    np.testing.assert_allclose(
        cs_zscore(torch.tensor(y)).numpy(), ((frame - frame.mean()) / frame.std()).to_numpy()
    )


def test_drop_extreme_keeps_nan_labels_and_drops_both_tails():
    y = torch.tensor(
        [[3.0], [np.nan], [1.0], [9.0], [5.0], [7.0], [np.nan], [2.0], [8.0], [4.0], [6.0]]
    )
    assert drop_extreme(y, 0.1).all()  # 9 finite -> 0 each tail
    keep = drop_extreme(y, 0.25)  # 2 each tail
    assert not keep[[2, 7, 3, 8]].any()
    assert keep[[0, 1, 4, 5, 6, 9, 10]].all()


@pytest.mark.parametrize("kind", ["rank", "zscore"])
def test_transforms_are_per_bar_so_a_per_bar_rescale_trains_the_same_model(
    tmp_path, recorders, kind
):
    features = _features()
    label = _label_of(features)
    rng = np.random.default_rng(3)
    scale = rng.uniform(0.5, 50.0, size=(N_TIMES, 1))
    shift = rng.uniform(-5.0, 5.0, size=(N_TIMES, 1))
    hp = {"transform": kind}

    plain = _model(tmp_path, features, label, name="plain", hyperparameters=hp)
    plain.train()
    rescaled = _model(tmp_path, features, label * scale + shift, name="rescaled",
                      hyperparameters=hp)
    rescaled.train()

    for key, value in _weights(plain).items():
        torch.testing.assert_close(value, _weights(rescaled)[key], rtol=1e-4, atol=1e-5)


def test_drop_extreme_removes_symbols_from_the_training_cross_section_only(
    tmp_path, recorders
):
    features = _features(n_symbols=10)
    label = _label_of(features)
    symbols = [f"S{i}" for i in range(10)]
    model = _model(tmp_path, features, label, cls=RecordingHead, symbols=symbols,
                   epochs=1, val_size=0.0,
                   hyperparameters={"transform": ("zscore", 0.1)})
    model.train()

    sizes = [x.shape[0] for x in model.model.inputs]
    assert sizes[:30] == [8] * 30  # 30 training bars, one extreme dropped per tail
    assert set(sizes[30:]) == {10}  # evaluation keeps the whole cross-section


def test_metrics_use_the_raw_label_not_the_transformed_target(tmp_path, recorders):
    features = _features()
    label = _label_of(features) * 100.0  # far from the z-scored target's scale
    model = _model(tmp_path, features, label)
    checkpoint = model.train()

    metrics = json.loads((checkpoint.parent / "metrics.json").read_text())
    pred = model.predict_panel(_feature_panel(features))["ret"].values[30:]
    expected = regression_panel_metrics(pred, label[30:])
    for key, value in expected.items():
        assert metrics[f"test_{key}"] == pytest.approx(float(value), rel=1e-5)


# ---------------------------------------------------------------------------
# Stopping rules
# ---------------------------------------------------------------------------


def test_train_loss_threshold_stops_at_the_threshold_or_the_cap():
    rule = TrainLossThreshold(threshold=1.0, max_epochs=10)
    assert [rule.update(x) for x in (2.0, 1.5, 0.9)] == [False, False, True]

    rule = TrainLossThreshold(threshold=0.0, max_epochs=3)
    assert [rule.update(1.0) for _ in range(3)] == [False, False, True]


@pytest.mark.parametrize(
    "stopping, epochs",
    [
        (("threshold", float("inf"), 40), 1),
        (("threshold", -1.0, 3), 3),
        (("threshold", -1.0, 40), 5),  # config.epochs caps
    ],
)
def test_a_fit_runs_the_epochs_the_heads_stop_hook_and_config_allow(
    tmp_path, recorders, stopping, epochs
):
    model = _model(tmp_path, _features(), _label_of(_features()), epochs=5,
                   hyperparameters={"stopping": stopping})
    model.train()

    (run,) = recorders
    assert len(run.logged) == epochs


class DefaultStoppingHead(MeanContextHead):
    """Uses DLModel's own stop hooks, recording when each is called."""

    def _on_fit_start(self):
        self.calls = ["start"]
        self.initial = {k: v.detach().clone() for k, v in self.model.state_dict().items()}

    def _should_stop(self, epoch, train_loss, val_loss):
        self.calls.append(("epoch", epoch, val_loss is not None))
        return super(MeanContextHead, self)._should_stop(epoch, train_loss, val_loss)

    def _on_fit_end(self):
        self.calls.append("end")
        self.last = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        super(MeanContextHead, self)._on_fit_end()


def test_the_default_hooks_run_every_epoch_and_keep_the_last_weights(tmp_path, recorders):
    features = _features()
    model = _model(tmp_path, features, _label_of(features), cls=DefaultStoppingHead, epochs=4)
    model.train()

    assert model.calls == ["start"] + [("epoch", e, True) for e in range(4)] + ["end"]
    assert _same_weights(model.last, _weights(model))
    assert not _same_weights(model.initial, _weights(model))


def test_a_fit_with_val_loss_patience_keeps_its_best_validation_epoch(tmp_path, recorders):
    features = _features()
    model = _model(tmp_path, features, _label_of(features), epochs=15, lr=0.3,
                   hyperparameters={"stopping": ("patience", 3)})
    checkpoint = model.train()

    (run,) = recorders
    best = min(entry["val_loss"] for entry in run.logged)
    metrics = json.loads((checkpoint.parent / "metrics.json").read_text())
    assert metrics["val_loss"] == pytest.approx(best, rel=1e-6)


# ---------------------------------------------------------------------------
# Files and reload
# ---------------------------------------------------------------------------


def test_a_dl_train_writes_metrics_json_and_reloads_to_the_same_predictions(
    tmp_path, recorders
):
    features = _features()
    label = _label_of(features)
    model = _model(tmp_path, features, label)
    checkpoint = model.train()

    metrics = json.loads((checkpoint.parent / "metrics.json").read_text())
    keys = ("loss", "mse", "rmse", "mae", "r2", "ic", "rank_ic")
    assert set(metrics) == {f"{s}_{k}" for s in ("train", "val", "test") for k in keys}
    assert metrics == recorders[0].summary

    fresh = MeanContextHead(model.config).load(checkpoint)  # no collect() needed
    xr.testing.assert_allclose(
        fresh.predict_panel(_feature_panel(features)),
        model.predict_panel(_feature_panel(features)),
    )


# ---------------------------------------------------------------------------
# Heads choose their own optimizer and loss
# ---------------------------------------------------------------------------


class FrozenOptimizerHead(MeanContextHead):
    """An optimizer with a zero learning rate: training must not move a weight."""

    def _init_model(self, num_features, num_labels, hyperparameters):
        net = super()._init_model(num_features, num_labels, hyperparameters)
        self.initial = {k: v.detach().clone() for k, v in net.state_dict().items()}
        return net

    def _init_optim(self, model):
        return torch.optim.SGD(model.parameters(), lr=0.0)


def test_a_head_can_supply_its_own_optimizer(tmp_path, recorders):
    features = _features()
    model = _model(tmp_path, features, _label_of(features), cls=FrozenOptimizerHead,
                   hyperparameters={"stopping": ("threshold", -1.0, 3)})
    model.train()

    assert _same_weights(model.initial, _weights(model))


class ConstantStepHead(MeanContextHead):
    """Steps that never update and report a loss of 1.0: the head, not the
    base class, decides what one step does and what loss it reports."""

    def _train_one_batch(self, epoch, batch):
        return torch.tensor(1.0)

    def _val_one_batch(self, epoch, batch):
        return torch.tensor(1.0)


def test_a_head_decides_what_a_step_does_and_reports(tmp_path, recorders):
    features = _features()
    model = _model(tmp_path, features, _label_of(features), cls=ConstantStepHead,
                   hyperparameters={"stopping": ("threshold", -1.0, 3)})

    checkpoint = model.train()

    (run,) = recorders
    assert [entry["train_loss"] for entry in run.logged] == [1.0, 1.0, 1.0]
    assert [entry["val_loss"] for entry in run.logged] == [1.0, 1.0, 1.0]
    metrics = json.loads((checkpoint.parent / "metrics.json").read_text())
    assert metrics["test_loss"] == 1.0


class HookRecordingHead(MeanContextHead):
    """Counts `_test_one_batch` calls and the rows each receives."""

    def _init_model(self, num_features, num_labels, hyperparameters):
        self.test_calls = []
        return super()._init_model(num_features, num_labels, hyperparameters)

    def _test_one_batch(self, epoch, batch):
        self.test_calls.append((epoch, batch.x.shape[0], batch.y.shape[0]))


def test_test_hook_runs_on_every_test_bar_after_each_epoch(tmp_path, recorders):
    features = _features()
    model = _model(tmp_path, features, _label_of(features), cls=HookRecordingHead,
                   hyperparameters={"stopping": ("threshold", -1.0, 2)})
    model.train()

    assert [epoch for epoch, _, _ in model.test_calls] == [0] * 10 + [1] * 10
    assert all(rows == len(SYMBOLS) and labels == rows for _, rows, labels in model.test_calls)


# ---------------------------------------------------------------------------
# The smallest head, and the optional hooks
# ---------------------------------------------------------------------------


class MinimalHead(DLModel):
    """Only what a head must write: a window, a network and a loss."""

    window_bars = 2

    def _init_model(self, num_features, num_labels, hyperparameters):
        return torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(2 * num_features, num_labels))

    def _loss(self, output, batch):
        return masked_mse(output, batch.y, batch.mask)


def test_a_head_with_only_a_window_a_network_and_a_loss_trains_and_predicts(
    tmp_path, recorders
):
    features = _features()
    label = _label_of(features)
    model = _model(tmp_path, features, label, cls=MinimalHead, epochs=3)
    checkpoint = model.train()

    (run,) = recorders
    assert len(run.logged) == 3  # no early stop by default
    assert run.logged[-1]["train_loss"] < run.logged[0]["train_loss"]
    out = model.predict_panel(_feature_panel(features))
    assert np.isfinite(out["ret"].values).all()
    assert (checkpoint.parent / "metrics.json").is_file()


def test_the_loss_hook_sees_a_masked_zero_filled_target_and_the_bar(tmp_path, recorders):
    features = _features()
    label = _label_of(features)
    label[:, 2] = np.nan
    seen: list[CrossSectionBatch] = []

    class Spy(MinimalHead):
        def _loss(self, output, batch):
            seen.append(batch)
            return super()._loss(output, batch)

    _model(tmp_path, features, label, cls=Spy, epochs=1, val_size=0.0).train()

    batch = seen[0]
    row = list(batch.symbols).index("S2")
    assert not batch.mask[row].any() and batch.y[row].eq(0).all()
    assert torch.isnan(batch.y_raw[row]).all()
    assert torch.isfinite(batch.y).all()
    assert list(batch.symbols) == SYMBOLS
    assert batch.timestamp in TIMES[:30]
    assert batch.x.shape == (len(SYMBOLS), 2, 2)


class TupleNet(torch.nn.Module):
    def __init__(self, num_features, num_labels):
        super().__init__()
        self.pred = torch.nn.Linear(num_features, num_labels)
        self.aux = torch.nn.Linear(num_features, 1)

    def forward(self, x):
        return self.pred(x[:, -1]), self.aux(x[:, -1])


class AuxOutputHead(MinimalHead):
    """A network with an auxiliary output: the loss uses both, prediction one."""

    def _init_model(self, num_features, num_labels, hyperparameters):
        return TupleNet(num_features, num_labels)

    def _loss(self, output, batch):
        pred, aux = output
        return masked_mse(pred, batch.y, batch.mask) + 0.1 * aux.pow(2).mean()

    def _forward(self, x):
        return self.model(x)[0]


def test_forward_maps_a_multi_output_network_to_the_prediction(tmp_path, recorders):
    features = _features()
    model = _model(tmp_path, features, _label_of(features), cls=AuxOutputHead, epochs=2)
    model.train()

    out = model.predict_panel(_feature_panel(features))
    assert np.isfinite(out["ret"].values).all()


class ScaledFeatureHead(RecordingHead):
    def _transform_feature(self, x):
        return torch.nan_to_num(x, nan=-1.0) * 2.0


def test_transform_feature_replaces_the_default_in_training_and_prediction(
    tmp_path, recorders
):
    features = {"f_a": np.ones((N_TIMES, 2)), "f_b": np.full((N_TIMES, 2), 10.0)}
    label = np.random.default_rng(0).standard_normal((N_TIMES, 2))
    model = _model(tmp_path, features, label, cls=ScaledFeatureHead, symbols=["S0", "S1"],
                   epochs=1, hyperparameters={"window_bars": 2})
    model.train()

    first = model.model.inputs[0]
    assert set(first[:, -1].reshape(-1).tolist()) == {2.0, 20.0}  # no clip to 3
    model.model.inputs.clear()
    model.predict_panel(_feature_panel(features, ["S0", "S1"]))
    np.testing.assert_array_equal(model.model.inputs[0][:, 0].numpy(), -2.0)  # before history


def test_a_transform_feature_that_leaves_nan_is_refused_naming_the_head(tmp_path, recorders):
    class LeavesNaN(MinimalHead):
        def _transform_feature(self, x):
            return x

    with pytest.raises(ValueError, match="LeavesNaN._transform_feature"):
        _model(tmp_path, _features(), _label_of(_features()), cls=LeavesNaN, epochs=1).train()
