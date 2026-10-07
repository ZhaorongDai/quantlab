"""Factor attribution in the backtest report and the tracker summary (#212, ADR 0026).

With a risk model, a run's ``report.html`` gains a "Factor attribution"
section drawn from its ``metrics.json`` block and ``factor_attribution.zarr``
only (#215): six tiles, then return and risk side by side (by part as bars
from the zero axis, over time, styles, industries), the style exposure
heatmap and each part's return against its realized risk, every tile and
chart explained in its ``title``. Factors are named by the store's ``label``
coordinate, or by their names without one.
Without a risk model no such tab appears (the page bytes are locked by
``tests.test_backtest_report_lock``). The tracker summary holds the scalar
``factor_attribution`` entries beside the whole / in-sample / out-of-sample
blocks.

The runs are the stub-risk-model backtests of
``tests.test_backtest_factor_attribution`` (``run_weights()``, whole only) and
``tests.test_backtest_factor_attribution_runs`` (``run()``, three segments).
Everything is synthetic, CPU-only and offline.
"""

import json
import re

import numpy as np
import pytest

from quantlab.backtest.predefined.us_equity import USEquityCrossectionSelectStockVectorBt
from quantlab.runs.backtest_run import BacktestRun
from quantlab.runs.factor_attribution_report import factor_attribution_section
from tests.backtest_fixtures import make_model, train_checkpoint
from tests.test_backtest_factor_attribution import (
    _backtester,
    _weights,
    model,  # noqa: F401 (module fixture)
    planted,  # noqa: F401 (module fixture)
    run as weights_run,  # noqa: F401 (module fixture)
)
from tests.test_backtest_factor_attribution_runs import (
    _config,
    _day,
    cv,  # noqa: F401 (module fixture)
    risk_model,  # noqa: F401 (module fixture)
    store,  # noqa: F401 (module fixture)
)
from tests.tracking_fixtures import RecordingTracker

TAB = ">Factor attribution</button>"
GROUP_CURVES = ("country", "industry", "style", "specific", "uncovered", "risk_free", "trading", "total")


@pytest.fixture(scope="module")
def tracked(store, risk_model):  # noqa: F811
    """A tracked load-mode ``run()`` with a risk model over both segments, and its tracker."""
    root, dataset_config = store
    dates = dict(
        start_date=_day(0), end_date=_day(29), train_start=_day(0), train_end=_day(24),
        test_start=_day(25), test_end=_day(29),
    )
    checkpoint = train_checkpoint(make_model(root / "report_train", dataset_config, **dates))
    tracker = RecordingTracker()
    result = USEquityCrossectionSelectStockVectorBt(_config(
        dataset_config,
        make_model(root / "report_backtest", dataset_config, **dates),
        risk_model,
        root / "report_runs",
        checkpoint=str(checkpoint),
        start_date=_day(20),
        end_date=_day(55),
        tracker=tracker,
    )).run()
    return result, tracker


def _page(result) -> str:
    return BacktestRun.open(result.run_dir).report()


def _pane(page: str, label: str) -> str:
    """The html of the tab whose button reads ``label``."""
    buttons = page.split('<button class="tab')[1:]
    index = next(i for i, b in enumerate(buttons) if f">{label}</button>" in b)
    start = page.index(f'id="tab{index}">')
    end = page.find('<div class="pane', start)
    return page[start:] if end < 0 else page[start:end]


TRACES = ("return_by_part", "risk_by_part", "cumulative_total", "style_exposure", "style_contribution",
          "style_forecast_risk", "style_realized_risk", "industry_return", "industry_risk",
          "style_exposure_heatmap", "Realized (63-bar)")
CARDS = ("Return by part", "Risk by part", "Return over time", "Risk over time",
         "Styles: exposure and return", "Styles: risk", "Industries: return", "Industries: risk",
         "Style exposure over time", "Return against realized risk")
TILES = ("Log growth / yr", "From factors", "Risk-free + trading", "Forecast vol", "Realized vol", "Coverage")


def test_a_run_with_a_risk_model_has_the_factor_attribution_section(tracked):
    result, _ = tracked
    page = _page(result)
    assert TAB in page
    pane = _pane(page, "Factor attribution")
    for curve in GROUP_CURVES[:-1]:
        assert f'"name":"cumulative_{curve}"' in pane, curve
    for trace in TRACES:
        assert f'"name":"{trace}"' in pane, trace
    for card in CARDS:
        assert re.search(rf'<h3 title="[^"]+">{re.escape(card)} ', pane), card
    for tile in TILES:
        assert re.search(rf'<div class="tile" title="[^"]+"><div class="kl">{re.escape(tile)}</div>', pane), tile
    # Every bar of the "by part" charts starts from the zero axis (no waterfall).
    assert '"type":"waterfall"' not in pane


def test_the_section_shows_the_headline_segment(tracked):
    result, _ = tracked
    pane = _pane(_page(result), "Factor attribution")
    out_of_sample = result.metrics["factor_attribution"]["out_of_sample"]
    total = out_of_sample["annualized_log_return"]["total"]
    assert f"{100 * total:+.2f}%" in pane
    assert f"{total:+.1%}" in pane  # the Total bar's label
    assert "out-of-sample" in pane


def test_run_weights_section_summarizes_the_whole_window(weights_run):  # noqa: F811
    pane = _pane(_page(weights_run), "Factor attribution")
    total = weights_run.metrics["factor_attribution"]["whole"]["annualized_log_return"]["total"]
    assert f"{100 * total:+.2f}%" in pane
    assert "whole window" in pane


def test_run_cv_section_draws_the_stitched_attribution(cv):  # noqa: F811
    pane = _pane(_page(cv), "Factor attribution")
    stitched = cv.metrics["stitched"]["factor_attribution"]
    segment = stitched["out_of_sample"] or stitched["whole"]
    assert f"{100 * segment['annualized_log_return']['total']:+.2f}%" in pane
    assert '"name":"cumulative_total"' in pane


def test_factors_are_named_by_the_store_label_and_by_their_name_without_one(weights_run):  # noqa: F811
    run = BacktestRun.open(weights_run.run_dir)
    block = run.metrics()["factor_attribution"]
    stored = run.factor_attribution()
    named = stored.assign_coords(label=("factor", [f"Name of {f}" for f in stored["factor"].values]))
    with_labels = factor_attribution_section(block, named, out_of_sample=False)
    assert "Name of industry_1" in with_labels and "Name of style" in with_labels
    without = factor_attribution_section(block, stored.drop_vars("label"), out_of_sample=False)
    assert "Name of" not in without and "industry_1" in without


def test_a_model_without_country_or_industries_draws_only_the_groups_it_has(weights_run):  # noqa: F811
    run = BacktestRun.open(weights_run.run_dir)
    block, stored = run.metrics()["factor_attribution"], run.factor_attribution()
    styles_only = stored.assign_coords(group=("factor", ["style"] * stored.sizes["factor"]))
    page = factor_attribution_section(block, styles_only, out_of_sample=False)
    assert "Industries: return" not in page and "cumulative_country" not in page
    assert '"name":"cumulative_style"' in page


def test_a_value_that_could_not_be_computed_is_a_dash_not_a_zero(weights_run):  # noqa: F811
    run = BacktestRun.open(weights_run.run_dir)
    block, stored = run.metrics()["factor_attribution"], run.factor_attribution()
    whole = dict(block["whole"])
    whole["ex_post_risk"] = {
        "volatility": None,
        "term_contribution": {term: None for term in whole["ex_post_risk"]["term_contribution"]},
        "factor_contribution": {name: None for name in whole["ex_post_risk"]["factor_contribution"]},
        "group_contribution": {name: None for name in whole["ex_post_risk"]["group_contribution"]},
    }
    page = factor_attribution_section({"whole": whole}, stored, out_of_sample=False)
    assert '<div class="kl">Realized vol</div><div class="kv">—</div>' in page
    assert '"\\u2014"' in page  # the realized bars' labels, in plotly's JSON


def test_a_block_without_a_segment_is_a_note(weights_run):  # noqa: F811
    stored = BacktestRun.open(weights_run.run_dir).factor_attribution()
    page = factor_attribution_section({"whole": None}, stored, out_of_sample=False)
    assert page.startswith('<p class="note">') and "plotly" not in page


def test_without_a_risk_model_there_is_no_tab(planted, tmp_path):  # noqa: F811
    plain = _backtester(planted[0], output_dir=str(tmp_path)).run_weights(_weights())
    page = _page(plain)
    assert "Factor attribution" not in page


def test_the_tracker_summary_holds_the_scalar_factor_attribution(tracked):
    result, tracker = tracked
    (run,) = tracker.runs
    block = result.metrics["factor_attribution"]
    for segment in ("whole", "in_sample", "out_of_sample"):
        key = f"factor_attribution/{segment}/annualized_log_return/total"
        assert run.summary[key] == pytest.approx(block[segment]["annualized_log_return"]["total"])
        key = f"factor_attribution/{segment}/ex_ante_risk/volatility/total"
        assert run.summary[key] == pytest.approx(block[segment]["ex_ante_risk"]["volatility"]["total"])
    assert "factor_attribution/whole/coverage/mean_covered_weight" in run.summary
    for key, value in run.summary.items():
        assert isinstance(value, (int, float)) and not isinstance(value, bool), key
        assert np.isfinite(value), key
    json.dumps(run.summary, allow_nan=False)
