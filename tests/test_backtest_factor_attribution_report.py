"""Factor attribution in the backtest report and the tracker summary (#212, ADR 0026).

With a risk model, a run's ``report.html`` gains a "Factor attribution" tab
drawn from its ``metrics.json`` block and ``factor_attribution.zarr`` only:
the cumulative log contribution curves by group (Country, Industry, Style,
Specific, Uncovered, Risk-free, Trading and their total), the per-style
contribution bars by segment, the style exposures (segment means and time
series), the industry top/bottom table, the ex-ante risk split over time with
its group and factor tables, the ex-post risk table and the coverage series.
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

import numpy as np
import pytest

from quantlab.backtest.predefined.us_equity import USEquityCrossectionSelectStockVectorBt
from quantlab.runs.backtest_run import BacktestRun
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


def test_a_run_with_a_risk_model_has_the_factor_attribution_tab(tracked):
    result, _ = tracked
    page = _page(result)
    assert TAB in page
    pane = _pane(page, "Factor attribution")
    for curve in GROUP_CURVES:
        assert f'"name":"factor_attribution_{curve}"' in pane, curve
    for trace in ("style_contribution_whole", "style_contribution_in_sample",
                  "style_contribution_out_of_sample", "style_mean_exposure_whole",
                  "style_exposure_style", "ex_ante_total", "ex_ante_factor",
                  "ex_ante_specific", "covered_weight"):
        assert f'"name":"{trace}"' in pane, trace
    for heading in ("Factor attribution (annualised log growth)", "Top and bottom industries",
                    "Ex-ante risk by group", "Ex-ante risk by factor", "Ex-post risk contribution",
                    "Coverage"):
        assert f"<h2>{heading}" in pane, heading
    for row in (">Country</th>", ">Industry</th>", ">Style</th>", ">Specific</th>", ">of which Country</th>",
                ">Uncovered</th>", ">Risk-free</th>", ">Trading</th>", ">Total</th>"):
        assert row in pane, row


def test_the_tab_shows_the_metrics_numbers(tracked):
    result, _ = tracked
    pane = _pane(_page(result), "Factor attribution")
    block = result.metrics["factor_attribution"]
    for segment in ("whole", "in_sample", "out_of_sample"):
        total = block[segment]["annualized_log_return"]["total"]
        assert f"{100 * total:+,.2f}%" in pane, segment
    top = block["out_of_sample"]["industries"]["top"][0]["factor"]
    assert f">{top}</th>" in pane


def test_run_weights_tab_has_the_whole_segment_only(weights_run):  # noqa: F811
    pane = _pane(_page(weights_run), "Factor attribution")
    assert '"name":"style_contribution_whole"' in pane
    assert "style_contribution_in_sample" not in pane
    assert "style_contribution_out_of_sample" not in pane


def test_run_cv_tab_draws_the_stitched_attribution(cv):  # noqa: F811
    pane = _pane(_page(cv), "Factor attribution")
    total = cv.metrics["stitched"]["factor_attribution"]["whole"]["annualized_log_return"]["total"]
    assert f"{100 * total:+,.2f}%" in pane
    assert '"name":"factor_attribution_total"' in pane


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
