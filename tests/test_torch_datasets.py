"""The shipped torch datasets, tested directly on hand-built training panels.

`CrossSectionDataset` is the default `TorchModel._dataset`: one item per bar,
every present symbol with its last N bars. What turns this file red:

- an item's `x`, `mask`, `y`, `y_raw` or `where` has the wrong shape or
  points at the wrong cells;
- evaluation misses a present symbol, or training includes a bar without a
  valid target;
- a window row before the panel's first bar is not NaN;
- a sample for bar t changes when features after t change (lookahead).

Everything is synthetic, CPU-only and offline.
"""

import numpy as np
import pytest
import torch

from quantlab.torch_model.data import Batch, CrossSectionDataset, TrainingPanel

T, S, F, L = 6, 4, 2, 1


def _panel(seed=0) -> TrainingPanel:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((T, S, F)).astype("float32")
    x[:2, 3] = np.nan  # S3 absent on bars 0 and 1
    x[4, :] = np.nan  # bar 4 has no present symbol
    y = rng.standard_normal((T, S, L)).astype("float32")
    panel = TrainingPanel.from_arrays(
        x, timestamps=np.arange(T), symbols=np.array(["A", "B", "C", "D"]), y_raw=y
    )
    panel.mask[:, :2] = panel.present[:, :2]  # only A and B have a valid target
    panel.mask[3] = False  # bar 3 has none
    panel.target[panel.mask] = panel.y_raw[panel.mask]
    return panel


def test_panel_marks_present_cells_and_starts_with_no_target():
    panel = TrainingPanel.from_arrays(
        np.array([[[1.0], [np.nan]]]), timestamps=np.arange(1), symbols=np.arange(2),
        y_raw=np.array([[[0.5], [0.2]]]),
    )
    assert panel.present.tolist() == [[True, False]]
    assert not panel.mask.any() and (panel.target == 0).all()


def test_an_item_is_one_bar_with_every_present_symbol_and_its_where():
    panel = _panel()
    dataset = CrossSectionDataset(panel, bars=[1, 2], window_bars=3, training=False)
    item = dataset[1]  # bar 2: all four symbols present

    assert isinstance(item, Batch)
    assert item.x.shape == (4, 3, F)
    assert item.mask.shape == (4,) and item.y.shape == item.y_raw.shape == (4, L)
    assert item.where[0].tolist() == [2] * 4 and item.where[1].tolist() == [0, 1, 2, 3]
    torch.testing.assert_close(item.x[:, -1], panel.x[2])
    torch.testing.assert_close(item.x[:, 0], panel.x[0], equal_nan=True)
    assert item.mask.tolist() == [True, True, False, False]


def test_window_rows_before_the_panel_are_nan():
    item = CrossSectionDataset(_panel(), bars=[0], window_bars=3, training=False)[0]
    assert torch.isnan(item.x[:, :2]).all() and torch.isfinite(item.x[:, 2]).all()
    assert item.where[1].tolist() == [0, 1, 2]  # S3 is absent on bar 0


def test_evaluation_covers_every_present_cell_and_training_only_valid_targets():
    panel = _panel()
    evaluation = CrossSectionDataset(panel, bars=range(T), window_bars=2, training=False)
    covered = {
        (int(t), int(s)) for item in evaluation for t, s in zip(*item.where)
    }
    assert covered == {tuple(c) for c in torch.nonzero(panel.present).tolist()}
    assert [int(item.where[0][0]) for item in evaluation] == [0, 1, 2, 3, 5]  # bar 4 empty

    training = CrossSectionDataset(panel, bars=range(T), window_bars=2, training=True)
    assert [int(item.where[0][0]) for item in training] == [0, 1, 2, 5]  # bar 3 has no target
    assert all(item.mask.any() for item in training)


def test_changing_features_after_a_bar_never_changes_its_sample():
    before = _panel()
    after = _panel()
    after.x[3:] += 100.0  # everything from bar 3 on
    for training in (True, False):
        old = CrossSectionDataset(before, bars=[0, 1, 2], window_bars=3, training=training)
        new = CrossSectionDataset(after, bars=[0, 1, 2], window_bars=3, training=training)
        for a, b in zip(old, new):
            torch.testing.assert_close(a.x, b.x, equal_nan=True)
            assert torch.equal(a.where[1], b.where[1])


def test_window_bars_below_one_is_refused():
    with pytest.raises(ValueError, match="window_bars"):
        CrossSectionDataset(_panel(), bars=[0], window_bars=0, training=False)
