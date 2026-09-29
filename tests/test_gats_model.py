"""`GATsRegressor` reproduces Qlib's GATs on the cross-section (issue #40).

The network is checked against Qlib's own `GATModel`, vendored verbatim in
`tests/qlib_gats_reference.py`: with the weights copied across, both give
the same output for the same input. The head's defaults follow Qlib's
Alpha158 benchmark config, and its stopping rule is Qlib's: keep the epoch
with the lowest validation loss, stop after `early_stop` epochs without a
strict improvement.

What turns this file red:
- the attention, the residual or the head drifts from Qlib's `GATModel`;
- the output depends on the order of the symbols in a bar;
- a one-symbol bar fails, or the output is not `[S_t, L]`;
- a default differs from Qlib's benchmark (window 20, LSTM 64x2, dropout
  0.7, lr 1e-4, 200 epochs, patience 10, rank target);
- training keeps the last epoch instead of the best validation epoch, or
  stops at the wrong epoch;
- `train` / `load` / `predict_panel` do not round-trip.
"""

import numpy as np
import pytest
import torch

from quantlab.model.gats import GATsNet, GATsRegressor
from quantlab.utils.torch_training import cs_rank_norm
from tests.qlib_gats_reference import GATModel
from tests.test_torch_model import _feature_panel, _features, _label_of, _model

SMALL = {"window_bars": 3, "hidden_size": 8, "num_layers": 1, "dropout": 0.0}


@pytest.fixture(autouse=True)
def _offline_wandb(monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("WANDB_SILENT", "true")


# ---------------------------------------------------------------------------
# The network against Qlib's GATModel
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("base_model", ["LSTM", "GRU"])
def test_the_network_matches_qlibs_gatmodel_with_the_same_weights(base_model):
    torch.manual_seed(0)
    reference = GATModel(d_feat=5, hidden_size=16, num_layers=2, dropout=0.0,
                         base_model=base_model).eval()
    ours = GATsNet(num_features=5, num_labels=1, hidden_size=16, num_layers=2,
                   dropout=0.0, base_model=base_model).eval()
    ours.load_state_dict(reference.state_dict())
    x = torch.randn(9, 7, 5)
    with torch.no_grad():
        torch.testing.assert_close(ours(x)[:, 0], reference(x), atol=1e-6, rtol=1e-5)


def test_permuting_the_symbols_permutes_the_output():
    torch.manual_seed(1)
    net = GATsNet(num_features=4, num_labels=2, hidden_size=8, num_layers=2).eval()
    x = torch.randn(11, 6, 4)
    perm = torch.randperm(11)
    with torch.no_grad():
        torch.testing.assert_close(net(x[perm]), net(x)[perm])


def test_a_one_symbol_bar_gives_one_row_of_every_label():
    net = GATsNet(num_features=4, num_labels=3, hidden_size=8, num_layers=1).eval()
    with torch.no_grad():
        assert net(torch.randn(1, 5, 4)).shape == (1, 3)


def test_an_unknown_base_model_is_refused():
    with pytest.raises(ValueError, match="base_model"):
        GATsNet(num_features=4, num_labels=1, base_model="Transformer")


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_the_defaults_are_qlibs_alpha158_benchmark(tmp_path):
    features = _features()
    head = _model(tmp_path, features, _label_of(features), cls=GATsRegressor)
    for key in ("epochs", "lr"):  # the test helper sets both; the defaults are wanted
        head.config.hyperparameters.pop(key)
    assert head.window_bars == 20
    assert head.epochs == 200
    assert head.early_stop == 10
    net = head._init_model(num_features=2, num_labels=1, hyperparameters={})
    assert isinstance(net.rnn, torch.nn.LSTM)
    assert (net.rnn.hidden_size, net.rnn.num_layers, net.rnn.dropout) == (64, 2, 0.7)
    assert head._init_optim(net).param_groups[0]["lr"] == 1e-4


def test_the_default_target_is_qlibs_cross_sectional_rank(tmp_path):
    features = _features()
    head = _model(tmp_path, features, _label_of(features), cls=GATsRegressor,
                  hyperparameters=SMALL)
    y = torch.tensor([[0.3], [float("nan")], [0.1], [0.2]])
    target, keep = head._transform_target(y, training=True)
    torch.testing.assert_close(target, cs_rank_norm(y), equal_nan=True)
    assert keep is None


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


class RecordingGATs(GATsRegressor):
    """Snapshots the weights and the validation loss after every epoch."""

    def _on_fit_start(self):
        super()._on_fit_start()
        self.history = []

    def _should_stop(self, epoch, train_loss, val_loss):
        weights = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        self.history.append((val_loss, weights))
        return super()._should_stop(epoch, train_loss, val_loss)


def test_training_keeps_the_best_validation_epoch_and_stops_after_the_patience(tmp_path):
    features = _features()
    head = _model(tmp_path, features, _label_of(features), cls=RecordingGATs,
                  hyperparameters={**SMALL, "early_stop": 2, "lr": 5e-2}, epochs=40)
    head.train()

    losses = [loss for loss, _ in head.history]
    best = int(np.argmin(losses))  # the first epoch with the lowest loss
    assert len(losses) == min(best + 1 + 2, 40)
    kept = {k: v.cpu() for k, v in head.model.state_dict().items()}
    assert all(torch.equal(kept[k], head.history[best][1][k]) for k in kept)


def test_train_load_and_predict_panel_round_trip(tmp_path):
    features = _features()
    label = _label_of(features)
    trained = _model(tmp_path, features, label, cls=GATsRegressor,
                     hyperparameters=SMALL, epochs=3)
    checkpoint = trained.train()
    assert checkpoint.name == "GATsRegressor_total.pth"

    fresh = _model(tmp_path, features, label, cls=GATsRegressor,
                   hyperparameters=SMALL, epochs=3, name="fresh")
    fresh.load(checkpoint)
    panel = _feature_panel(features)
    a = trained.predict_panel(panel)["ret"].values
    b = fresh.predict_panel(panel)["ret"].values
    assert np.isfinite(a[SMALL["window_bars"]:]).all()
    np.testing.assert_array_equal(a, b)
