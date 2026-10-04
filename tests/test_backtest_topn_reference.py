"""Top-n selection through the portfolio layer reproduces the pre-migration backtests.

`tests/topn_reference.npz` holds the weights and equity of a long-only and a
long-short top-n backtest. The weights were captured with
`CrossSectionTopNSelector` before the rule moved into `TopNConstructor` (#76).
The equity was recaptured when a delisted holding started to be settled at its
last valuation instead of sold at its last open (#86): it is bit-identical to
the #76 capture on every bar before CCC's settlement bar, and differs from
there on. The weights are unchanged by tradability at t and locked positions
(#87): CCC is never held when its prices stop, so nothing is ever locked here.
Re-running the same scenario through the backtester's `constructor` must give
bit-identical weights, and equity equal to a relative 1e-12: the reference was
captured on macOS arm64, and x86_64 Linux rounds the simulation's last digit
differently on a few bars (a relative 2e-16). A backtester rebuilt from the
run's `config.json` must re-run it identically, bit for bit, on one machine.

Everything is synthetic, CPU-only and offline.
"""


import numpy as np
import pytest

from quantlab.backtest.predefined.us_equity import USEquityCrossectionSelectStockVectorBt
from quantlab.base.config import CrossSectionBacktestConfig, TopNConfig
from quantlab.portfolio.predefined.top_n import TopNConstructor
from quantlab.runs.backtest_run import BacktestRun
from tests.backtest_fixtures import make_stock_dataset
from tests.topn_reference import BACKTEST, CASES, REFERENCE, scenario


def _backtester(tmp_path, case, *, output_dir=None):
    direction, top_n, _ = CASES[case]
    dataset_config, model, start, end = scenario(tmp_path, case)
    return USEquityCrossectionSelectStockVectorBt(
        CrossSectionBacktestConfig(
            price_dataset=make_stock_dataset(dataset_config),
            model=model,
            model_mode="train",
            start_date=start,
            end_date=end,
            output_dir=output_dir,
            constructor=TopNConstructor(TopNConfig(direction=direction, top_n=top_n)),
            **BACKTEST,
        )
    )


@pytest.mark.parametrize("case", sorted(CASES))
def test_top_n_backtests_reproduce_the_pre_migration_reference(tmp_path, case):
    reference = np.load(REFERENCE)

    result = _backtester(tmp_path, case).run()

    weights = result.weights["weight"].transpose("timestamp", "symbol")
    np.testing.assert_array_equal(weights.timestamp.values, reference[f"{case}_timestamps"])
    np.testing.assert_array_equal(weights.symbol.values.astype(str), reference[f"{case}_symbols"])
    np.testing.assert_array_equal(weights.values, reference[f"{case}_weights"])
    # Not bit for bit: the last digit of the equity differs across CPU architectures.
    np.testing.assert_allclose(
        result.simulation.value.values, reference[f"{case}_value"], rtol=1e-12, atol=0
    )


def test_the_constructor_round_trips_through_the_run_and_reruns_identically(tmp_path):
    backtester = _backtester(tmp_path, "long_short", output_dir=str(tmp_path / "runs"))
    original = backtester.run()
    saved = backtester.get_config()

    assert saved["constructor"] == {
        "direction": "long_short",
        "top_n": 2,
        "score_label": None,
        "name": "quantlab.portfolio.predefined.top_n.TopNConstructor",
    }
    assert not {"direction", "top_n", "score_label"} & set(saved)

    rebuilt = BacktestRun.open(original.run_dir).rebuild_backtester()
    assert rebuilt.config.constructor == TopNConstructor(
        TopNConfig(direction="long_short", top_n=2)
    )
    again = rebuilt.run()

    np.testing.assert_array_equal(again.weights["weight"].values, original.weights["weight"].values)
    np.testing.assert_array_equal(again.simulation.value.values, original.simulation.value.values)


def test_a_constructor_that_is_not_a_portfolio_constructor_is_refused(tmp_path):
    dataset_config, model, start, end = scenario(tmp_path, "long_only")
    with pytest.raises(TypeError, match="PortfolioConstructor"):
        USEquityCrossectionSelectStockVectorBt(
            CrossSectionBacktestConfig(
                price_dataset=make_stock_dataset(dataset_config),
                model=model,
                model_mode="train",
                start_date=start,
                end_date=end,
                output_dir=None,
                constructor={"direction": "long_only", "top_n": 3},
                **BACKTEST,
            )
        )
