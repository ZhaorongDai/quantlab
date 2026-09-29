"""`MASTERRegressor` reproduces MASTER on the cross-section (issue #43).

The network is checked against the official MASTER network, vendored
verbatim in `tests/master_reference.py`: with the weights copied across,
both give the same output for the same input. The market gate reads the
factors named by the `gate_features` hyperparameter at the last bar and
rescales the other factors; the head's defaults follow the official code
and paper.

What turns this file red:
- the gate, the positional encoding, either attention, the temporal
  aggregation or the decoder drifts from the official network;
- the gate weights do not sum to the number of gated features, or ignore
  the market features;
- the output depends on the order of the symbols in a bar;
- a gate feature missing from the factors is not refused, by name;
- a default differs from the official one (window 8, D 256, 4 and 2 heads,
  dropout 0.5, beta 5, lr 1e-5, 40 epochs, train-loss threshold 0.95,
  z-score target with 2.5% tails dropped in training);
- `train` / `load` / `predict_panel` do not round-trip.
"""

import numpy as np
import pytest
import torch

from quantlab.model.master import MASTERNet, MASTERRegressor
from quantlab.utils.torch_training import cs_zscore, drop_extreme
from tests.master_reference import MASTER
from tests.test_torch_model import _feature_panel, _features, _label_of, _model

SMALL = {"window_bars": 3, "d_model": 8, "t_nhead": 2, "s_nhead": 2, "dropout": 0.0,
         "gate_features": ["f_b"]}


@pytest.fixture(autouse=True)
def _offline_wandb(monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("WANDB_SILENT", "true")


def _head(tmp_path, **hyperparameters):
    features = _features()
    return _model(tmp_path, features, _label_of(features), cls=MASTERRegressor,
                  hyperparameters={**SMALL, **hyperparameters})


# ---------------------------------------------------------------------------
# The network against the official MASTER
# ---------------------------------------------------------------------------


def test_the_network_matches_the_official_master_with_the_same_weights():
    torch.manual_seed(0)
    stock, market = 6, 3
    reference = MASTER(d_feat=stock, d_model=16, t_nhead=4, s_nhead=2,
                       T_dropout_rate=0.0, S_dropout_rate=0.0,
                       gate_input_start_index=stock, gate_input_end_index=stock + market,
                       beta=5.0).eval()
    ours = MASTERNet(num_features=stock + market, num_labels=1,
                     gate_columns=list(range(stock, stock + market)),
                     d_model=16, t_nhead=4, s_nhead=2, dropout=0.0, beta=5.0).eval()
    ours.load_state_dict(reference.state_dict())
    x = torch.randn(10, 8, stock + market)
    with torch.no_grad():
        torch.testing.assert_close(ours(x)[:, 0], reference(x), atol=1e-5, rtol=1e-5)


def test_the_gate_columns_may_sit_anywhere_among_the_features():
    torch.manual_seed(2)
    trailing = MASTERNet(num_features=5, num_labels=1, gate_columns=[3, 4], d_model=8,
                         t_nhead=2, s_nhead=2, dropout=0.0).eval()
    leading = MASTERNet(num_features=5, num_labels=1, gate_columns=[0, 1], d_model=8,
                        t_nhead=2, s_nhead=2, dropout=0.0).eval()
    leading.load_state_dict(trailing.state_dict())
    x = torch.randn(4, 3, 5)
    moved = torch.cat([x[..., 3:], x[..., :3]], dim=-1)  # the same data, gate first
    with torch.no_grad():
        torch.testing.assert_close(leading(moved), trailing(x))


def test_the_gate_weights_sum_to_the_gated_features_and_follow_the_market():
    torch.manual_seed(3)
    net = MASTERNet(num_features=7, num_labels=1, gate_columns=[5, 6], d_model=8,
                    t_nhead=2, s_nhead=2).eval()
    market = torch.randn(4, 2)
    weights = net.gate(market)
    assert weights.shape == (4, 5)
    torch.testing.assert_close(weights.sum(dim=-1), torch.full((4,), 5.0))
    shifted = market.clone()
    shifted[:, 0] += 1.0
    assert not torch.allclose(net.gate(shifted), weights)


def test_permuting_the_symbols_permutes_the_output():
    torch.manual_seed(1)
    net = MASTERNet(num_features=5, num_labels=2, gate_columns=[4], d_model=8,
                    t_nhead=2, s_nhead=2).eval()
    x = torch.randn(9, 4, 5)
    perm = torch.randperm(9)
    with torch.no_grad():
        torch.testing.assert_close(net(x[perm]), net(x)[perm])


def test_a_one_symbol_bar_gives_one_row_of_every_label():
    net = MASTERNet(num_features=3, num_labels=2, gate_columns=[2], d_model=8,
                    t_nhead=2, s_nhead=2).eval()
    with torch.no_grad():
        assert net(torch.randn(1, 4, 3)).shape == (1, 2)


def test_a_model_width_the_heads_do_not_divide_is_refused():
    with pytest.raises(ValueError, match="d_model"):
        MASTERNet(num_features=3, num_labels=1, gate_columns=[2], d_model=10, t_nhead=4)


# ---------------------------------------------------------------------------
# gate_features
# ---------------------------------------------------------------------------


def test_gate_features_missing_from_the_factors_are_refused_by_name(tmp_path):
    with pytest.raises(ValueError, match="spy_ret_mean_5"):
        _head(tmp_path, gate_features=["f_b", "spy_ret_mean_5"])


def test_a_head_without_gate_features_is_refused(tmp_path):
    with pytest.raises(ValueError, match="gate_features"):
        _head(tmp_path, gate_features=[])


def test_gate_features_cannot_be_every_factor(tmp_path):
    with pytest.raises(ValueError, match="gate_features"):
        _head(tmp_path, gate_features=["f_a", "f_b"])


# ---------------------------------------------------------------------------
# Defaults, target and stopping
# ---------------------------------------------------------------------------


def test_the_defaults_are_the_official_ones(tmp_path):
    head = _head(tmp_path)
    for key in ("window_bars", "d_model", "t_nhead", "s_nhead", "dropout", "epochs", "lr"):
        head.config.hyperparameters.pop(key, None)
    assert head.window_bars == 8
    assert head.epochs == 40
    net = head._init_model(num_features=2, num_labels=1, hyperparameters={})
    assert net.d_model == 256 and net.beta == 5.0
    temporal, stock = net.layers[2], net.layers[3]  # the official layer order
    assert temporal.nhead == 4 and stock.nhead == 2
    assert temporal.ffn[2].p == 0.5 and stock.ffn[2].p == 0.5
    assert head._init_optim(net).param_groups[0]["lr"] == 1e-5


def test_the_training_target_drops_both_tails_then_z_scores(tmp_path):
    head = _head(tmp_path)
    y = torch.linspace(-1.0, 1.0, 80)[:, None] ** 3
    target, keep = head._transform_target(y, training=True)
    expected_keep = drop_extreme(y, 0.025)
    assert torch.equal(keep, expected_keep) and int((~keep).sum()) == 4
    torch.testing.assert_close(target, cs_zscore(y[keep]))
    evaluation, kept = head._transform_target(y, training=False)
    assert kept is None
    torch.testing.assert_close(evaluation, cs_zscore(y))


def test_training_stops_once_the_train_loss_reaches_the_threshold(tmp_path):
    class Counting(MASTERRegressor):
        def _should_stop(self, epoch, train_loss, val_loss):
            self.epochs_run = epoch + 1
            return super()._should_stop(epoch, train_loss, val_loss)

    features = _features()
    reached = _model(tmp_path, features, _label_of(features), cls=Counting, name="reached",
                     hyperparameters={**SMALL, "train_loss_threshold": 100.0}, epochs=10)
    reached.train()
    assert reached.epochs_run == 1
    never = _model(tmp_path, features, _label_of(features), cls=Counting, name="never",
                   hyperparameters={**SMALL, "train_loss_threshold": -1.0}, epochs=3)
    never.train()
    assert never.epochs_run == 3


def test_train_load_and_predict_panel_round_trip(tmp_path):
    features = _features()
    label = _label_of(features)
    trained = _model(tmp_path, features, label, cls=MASTERRegressor,
                     hyperparameters=SMALL, epochs=3)
    checkpoint = trained.train()
    assert checkpoint.name == "MASTERRegressor_total.pth"

    fresh = _model(tmp_path, features, label, cls=MASTERRegressor,
                   hyperparameters=SMALL, epochs=3, name="fresh")
    fresh.load(checkpoint)
    panel = _feature_panel(features)
    a = trained.predict_panel(panel)["ret"].values
    b = fresh.predict_panel(panel)["ret"].values
    assert np.isfinite(a[SMALL["window_bars"]:]).all()
    np.testing.assert_array_equal(a, b)
