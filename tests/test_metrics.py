"""Tests for `quantlab/utils/metrics.py` (quick task 260914-lno).

The metrics module is the one place the model layer turns a `[T, S]`
prediction panel into numbers written to W&B and returned from `train_cv`, so
every definition is locked by a hand-computed case:

- the four error metrics count only JOINTLY finite cells;
- cross-sectional IC is a per-timestamp Pearson averaged over timestamps, and
  skips rows with <2 joint-valid symbols or zero variance (exactly), returning
  NaN when every row is skipped;
- RankIC applies the joint mask BEFORE ranking and averages ties;
- both IC functions agree with a row-by-row scipy `pearsonr`/`spearmanr`
  reference on a random NaN-laden panel;
- both IC functions stay vectorised (an AST lock: no loop or comprehension
  node in their bodies);
- nothing here emits a RuntimeWarning -- the whole module runs under
  `error::RuntimeWarning`, so an all-NaN `nanmean` sneaking back in turns red.
"""

import ast
import inspect
import textwrap
import warnings

import numpy as np
import pytest
from scipy.stats import pearsonr, spearmanr

from quantlab.utils import metrics
from quantlab.utils.metrics import (
    cross_sectional_ic,
    cross_sectional_rank_ic,
    mae,
    mse,
    r2,
    regression_panel_metrics,
    rmse,
)

pytestmark = pytest.mark.filterwarnings("error::RuntimeWarning")

NAN = np.nan


# --------------------------------------------------------------------------
# Error metrics
# --------------------------------------------------------------------------


def test_error_metrics_use_only_jointly_finite_cells():
    """`[1, 2, nan]` vs `[1, 4, 5]`: only the first two cells count, so the
    diffs are 0 and -2. Turns red if a NaN cell leaks into a sum or the
    denominator counts masked cells."""
    pred, target = [1.0, 2.0, NAN], [1.0, 4.0, 5.0]
    assert mse(pred, target) == 2.0
    assert mae(pred, target) == 1.0
    assert rmse(pred, target) == pytest.approx(np.sqrt(2.0))
    assert r2(pred, target) == pytest.approx(1 - 4 / 4.5)


def test_error_metrics_mask_infinities_on_either_side():
    """An inf on EITHER side removes the cell -- the mask is joint, not
    target-only."""
    assert mse([1.0, np.inf, 3.0], [1.0, 2.0, 5.0]) == 2.0
    assert mse([1.0, 2.0, 3.0], [1.0, -np.inf, 5.0]) == 2.0


def test_error_metrics_are_nan_without_valid_cells():
    """No jointly finite cell -> NaN, silently (the module runs under
    error::RuntimeWarning)."""
    for fn in (mse, rmse, mae, r2):
        assert np.isnan(fn([NAN, 1.0], [1.0, NAN]))


def test_r2_is_nan_when_undefined():
    """R² needs >=2 valid cells and a non-constant target."""
    assert np.isnan(r2([1.0], [2.0]))
    assert np.isnan(r2([1.0, 2.0, 3.0], [5.0, 5.0, 5.0]))


def test_shape_mismatch_raises():
    """Silently broadcasting a [T, S] prediction against a [T, 1] target
    would produce plausible numbers; it must raise instead."""
    with pytest.raises(ValueError, match="same shape"):
        mse(np.zeros((3, 2)), np.zeros((3, 1)))
    with pytest.raises(ValueError, match="same shape"):
        cross_sectional_ic(np.zeros((3, 2)), np.zeros((3, 1)))


# --------------------------------------------------------------------------
# Cross-sectional IC
# --------------------------------------------------------------------------


def test_ic_averages_per_timestamp_correlations():
    """Row 0 correlates +1, row 1 correlates -1: a per-row Pearson averaged
    over rows is 0. A pooled correlation over all cells would give 0 too on
    this panel, so the next test pins the per-row part independently."""
    pred = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    target = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
    assert cross_sectional_ic(pred, target) == pytest.approx(0.0, abs=1e-12)


def test_ic_is_per_row_not_pooled():
    """Two rows with identical within-row ordering but wildly different
    levels: per-row IC is exactly 1; a pooled correlation would not be."""
    pred = np.array([[1.0, 2.0, 3.0], [101.0, 102.0, 103.0]])
    target = np.array([[3.0, 4.0, 5.0], [-10.0, -9.0, -8.0]])
    assert cross_sectional_ic(pred, target) == pytest.approx(1.0)


def test_ic_uses_the_joint_mask():
    """Only the pairs (1, 2) and (2, 4) are jointly valid -> IC 1.0."""
    pred = np.array([[1.0, 2.0, 3.0, NAN]])
    target = np.array([[2.0, 4.0, NAN, 7.0]])
    assert cross_sectional_ic(pred, target) == pytest.approx(1.0)


def test_ic_skips_thin_and_constant_rows():
    """Row 0 is the only usable row (IC 1). Row 1 has one valid pair, row 2
    has a constant prediction, row 3 a constant target: all three are
    skipped rather than dragging the mean toward 0 or producing NaN."""
    pred = np.array(
        [
            [1.0, 2.0, 3.0],
            [1.0, NAN, NAN],
            [5.0, 5.0, 5.0],
            [1.0, 2.0, 3.0],
        ]
    )
    target = np.array(
        [
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
            [7.0, 7.0, 7.0],
        ]
    )
    assert cross_sectional_ic(pred, target) == pytest.approx(1.0)


def test_ic_is_nan_when_every_row_is_skipped():
    pred = np.array([[1.0, NAN], [2.0, 2.0]])
    target = np.array([[1.0, 2.0], [1.0, 3.0]])
    assert np.isnan(cross_sectional_ic(pred, target))
    assert np.isnan(cross_sectional_ic(np.full((2, 3), NAN), np.full((2, 3), NAN)))


def test_ic_rejects_non_2d_input():
    with pytest.raises(ValueError, match="2-D"):
        cross_sectional_ic(np.zeros(3), np.zeros(3))
    with pytest.raises(ValueError, match="2-D"):
        cross_sectional_rank_ic(np.zeros((2, 2, 1)), np.zeros((2, 2, 1)))


# --------------------------------------------------------------------------
# Cross-sectional RankIC
# --------------------------------------------------------------------------


def test_rank_ic_is_monotone_invariant():
    """A monotone but non-linear relation has RankIC exactly 1."""
    assert cross_sectional_rank_ic([[1.0, 10.0, 100.0]], [[1.0, 2.0, 3.0]]) == pytest.approx(1.0)


def test_rank_ic_averages_ties():
    """Target ranks with a tie are [1.5, 1.5, 3] -> Pearson 1.5/sqrt(3)."""
    assert cross_sectional_rank_ic([[1.0, 2.0, 3.0]], [[1.0, 1.0, 2.0]]) == pytest.approx(
        0.8660254, abs=1e-7
    )


def test_rank_ic_masks_before_ranking():
    """The NaN target cell must also vanish from the PREDICTION ranks.

    Masked first: pred ranks [1, 2, 3], target ranks [3, 2, 1] -> -1.0.
    Ranked first: pred ranks [1, 2, 4] vs [3, 2, 1] -> about -0.98, so the
    order of operations is what this assertion reads.
    """
    assert cross_sectional_rank_ic([[1.0, 2.0, 3.0, 4.0]], [[4.0, 3.0, NAN, 1.0]]) == pytest.approx(
        -1.0
    )


# --------------------------------------------------------------------------
# Agreement with a row-by-row scipy reference
# --------------------------------------------------------------------------


def _reference(pred, target, stat):
    values = []
    for p_row, t_row in zip(pred, target):
        ok = np.isfinite(p_row) & np.isfinite(t_row)
        if ok.sum() < 3:
            continue
        values.append(stat(p_row[ok], t_row[ok])[0])
    return float(np.mean(values))


def test_ic_and_rank_ic_match_scipy_row_by_row():
    """Random 50 x 12 panel with ~20% NaN cells on each side. Rows are built
    to keep >=3 joint-valid, non-constant symbols, so the reference and the
    vectorised version average over exactly the same rows. Ties are injected
    into the target so RankIC exercises average ranking."""
    rng = np.random.default_rng(7)
    T, S = 50, 12
    pred = rng.standard_normal((T, S))
    target = 0.4 * pred + rng.standard_normal((T, S))
    target[:, :3] = np.round(target[:, :3])
    pred[rng.random((T, S)) < 0.2] = NAN
    target[rng.random((T, S)) < 0.2] = NAN
    pred[:, :3] = np.where(np.isnan(pred[:, :3]), rng.standard_normal((T, 3)), pred[:, :3])
    target[:, 3:6] = np.where(
        np.isnan(target[:, 3:6]), rng.standard_normal((T, 3)), target[:, 3:6]
    )
    pred[:, 3:6] = np.where(np.isnan(pred[:, 3:6]), rng.standard_normal((T, 3)), pred[:, 3:6])
    joint = np.isfinite(pred) & np.isfinite(target)
    assert joint.sum(axis=1).min() >= 3

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref_ic = _reference(pred, target, pearsonr)
        ref_rank_ic = _reference(pred, target, spearmanr)

    assert cross_sectional_ic(pred, target) == pytest.approx(ref_ic, abs=1e-10)
    assert cross_sectional_rank_ic(pred, target) == pytest.approx(ref_rank_ic, abs=1e-10)


def test_regression_panel_metrics_keys_and_values():
    rng = np.random.default_rng(3)
    pred = rng.standard_normal((20, 6))
    target = pred + 0.1 * rng.standard_normal((20, 6))
    out = regression_panel_metrics(pred, target)
    assert list(out) == ["mse", "rmse", "mae", "r2", "ic", "rank_ic"]
    assert out["mse"] == mse(pred, target)
    assert out["ic"] == cross_sectional_ic(pred, target)
    assert out["rank_ic"] == cross_sectional_rank_ic(pred, target)
    assert all(isinstance(v, float) for v in out.values())


# --------------------------------------------------------------------------
# Structural lock: the IC functions stay vectorised
# --------------------------------------------------------------------------


@pytest.mark.parametrize("fn", [cross_sectional_ic, cross_sectional_rank_ic])
def test_ic_functions_have_no_python_loops(fn):
    """A Python row loop over a 10k-timestamp panel is orders of magnitude
    slower and is exactly what a "readable" rewrite reintroduces. Turns red
    on any for/while loop or comprehension inside the function body."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    loops = (ast.For, ast.AsyncFor, ast.While, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
    found = [type(node).__name__ for node in ast.walk(tree) if isinstance(node, loops)]
    assert found == [], f"{fn.__name__} contains loop nodes: {found}"
    assert fn.__module__ == metrics.__name__
