"""Factor attribution of a backtest on ``run_weights()`` (#208, #210, #211, ADR 0026).

A backtest config takes an optional ``risk_model``. With one, each bar's NAV
return is split into factor, specific, uncovered, risk-free and trading terms
over the holdings the engine held at the close of the bar before. What is
locked here, on a stub ``FactorRiskModel`` whose regression store is written
from planted factor and specific returns:

- the terms sum to the NAV return at every bar, and the factor contribution is
  the start-of-bar holdings (the Execution rules' ``replay``, independent of the
  engine) times the exposures of t-1 times the stored factor returns of t;
- a rejected order keeps a holding and a delisting settlement closes it, as
  the engine held them;
- a held symbol without exposures lands in ``uncovered`` with its own return;
  a thin-industry NaN factor return contributes 0;
- the log contributions add up to log NAV at every bar; the metrics report
  annualized log growth per term and per factor, and the coverage;
- ``factor_attribution.zarr`` in the run directory, and a rebuild from
  ``config.json`` reproduces the attribution; without a risk model nothing
  changes;
- three refusals: stores not covering the window, a different bar interval,
  no held symbol covered;
- ex ante (#210), from the planted estimate row of t-1: the forecast
  variance of the covered book over the factors with a covariance, whose
  x-sigma-rho contributions and specific part sum to the forecast volatility;
  ex post, each term's ``cov(c, r) / sigma(r)``, summing to the realized
  volatility;
- groups (#211): country, industry and style contributions summing to the
  factor term, the styles' mean exposures and the industry table, and the
  risk contributions per group.

The pure function's own formulas are covered through these entry points.
Everything is synthetic, CPU-only and offline.
"""

import dataclasses

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.backtest.config import WeightsBacktestConfig
from quantlab.backtest.predefined.weights import WeightsVectorBt
from quantlab.dataset.memory import FrameDataset
from quantlab.execution.rules import ExecutionSettings, replay
from quantlab.factor.base import Factor
from quantlab.factor.config import BaseFactorConfig
from quantlab.risk.attribution import LOW_COVERAGE, TERMS
from quantlab.risk.base import FactorRiskModel
from quantlab.risk.config import FactorRiskConfig
from quantlab.runs.backtest_run import BacktestRun

BARS = pd.bdate_range("2024-01-01", periods=16)
T = BARS.size
SYMBOLS = ["A", "B", "C", "D", "E", "F"]
FACTORS = ("market", "industry_1", "industry_2", "style")
INDUSTRY = np.array([1.0, 1.0, 2.0, 2.0, 2.0, np.nan])  # F has no industry: uncovered
THIN_BAR = 7  # industry_2 has no factor return on this bar
NO_COVARIANCE_BAR = 4  # industry_2 has no forecast covariance in this estimate row
GROUPS = {"market": "country", "industry_1": "industry", "industry_2": "industry", "style": "style"}
DELIST_BAR = 9  # D's last price; settled on the next bar
REJECT_BAR = 5  # B has no open here, so the order filling here is rejected
FEES, SLIPPAGE, INIT_CASH = 0.001, 0.0005, 1000.0


class Exposures(Factor):
    """The exposures: its dataset's variables, unchanged."""

    config_cls = BaseFactorConfig

    def _get_factor_names(self):
        return ("industry", "style")

    def _compute_panel(self, inputs):
        return inputs


@dataclasses.dataclass(kw_only=True, frozen=True)
class StubRiskConfig(FactorRiskConfig):
    """``FactorRiskConfig`` and the Zarr stores of the planted regression and estimate rows."""

    planted: str
    planted_estimate: str


class StubRiskModel(FactorRiskModel):
    """Market, two industries and one style; its store rows are the planted ones."""

    config_cls = StubRiskConfig

    @property
    def factor_names(self):
        return FACTORS

    @property
    def exposure_names(self):
        return ("industry", "style")

    def factor_groups(self):
        return dict(GROUPS)

    def exposure_matrix(self, exposures):
        industry = exposures["industry"].values
        style = exposures["style"].values
        matrix = np.column_stack(
            [np.ones_like(style), industry == 1.0, industry == 2.0, style]
        ).astype(float)
        return matrix, np.isfinite(industry) & np.isfinite(style)

    @property
    def regression_warmup_bars(self):
        return 0

    @property
    def estimate_warmup_bars(self):
        return 0

    def _compute_regression(self, start, end):
        rows = xr.open_zarr(self.config.planted).load()
        return rows.sel(timestamp=slice(pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta("1D")))

    def _compute_estimate(self, start, end):
        rows = xr.open_zarr(self.config.planted_estimate).load()
        return rows.sel(timestamp=slice(pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta("1D")))


def _plant(seed: int = 208):
    """Return the price panel, the exposures panel and the stored regression and estimate rows."""
    rng = np.random.default_rng(seed)
    n = len(SYMBOLS)
    style = rng.normal(size=(T, n))
    risk_free = 1e-4 + 1e-6 * np.arange(T)
    factor_return = np.full((T, len(FACTORS)), np.nan)
    specific = np.full((T, n), np.nan)
    close = np.empty((T, n))
    close[0] = rng.uniform(20, 50, size=n)
    stored_factor = factor_return.copy()
    for t in range(1, T):
        f = np.array([rng.normal(0, 0.01), *rng.normal(0, 0.005, 2), rng.normal(0, 0.003)])
        exposures = np.column_stack(
            [np.ones(n), INDUSTRY == 1.0, INDUSTRY == 2.0, style[t - 1]]
        )
        s = rng.normal(0, 0.01, size=n)
        excess = exposures @ f + s
        excess[-1] = rng.normal(0, 0.02)  # F: no exposures, its own return
        close[t] = close[t - 1] * (1.0 + excess + risk_free[t - 1])
        factor_return[t] = f
        stored_factor[t] = f
        specific[t] = s
        if t == THIN_BAR:
            # The thin industry has no factor return; its members' specific
            # returns keep that part, as the regression store's would.
            stored_factor[t, 2] = np.nan
            specific[t] = np.where(INDUSTRY == 2.0, s + f[2], s)
    specific[:, -1] = np.nan
    close[DELIST_BAR + 1:, 3] = np.nan
    specific[DELIST_BAR + 1:, 3] = np.nan
    open_ = np.vstack([close[:1], close[:-1] * (1.0 + rng.normal(0, 0.002, size=(T - 1, n)))])
    open_[np.isnan(close)] = np.nan
    open_[REJECT_BAR, 1] = np.nan

    coords = {"timestamp": BARS, "symbol": SYMBOLS}
    panel = ("timestamp", "symbol")
    prices = xr.Dataset(
        {
            "open": (panel, open_),
            "close": (panel, close),
            "marketcap": (panel, np.full((T, n), 1e9)),
            "risk_free": (panel, np.repeat(risk_free[:, None], n, axis=1)),
        },
        coords=coords,
    )
    exposures = xr.Dataset(
        {"industry": (panel, np.repeat(INDUSTRY[None, :], T, axis=0)), "style": (panel, style)},
        coords=coords,
    )
    regression = xr.Dataset(
        {
            "factor_return": (("timestamp", "factor"), stored_factor),
            "specific_return": (panel, specific),
        },
        coords={**coords, "factor": list(FACTORS)},
    )
    return prices, exposures, regression, _estimate(rng, BARS, SYMBOLS)


def _estimate(rng, bars, symbols) -> xr.Dataset:
    """Random positive-definite factor covariances and specific risks, one row per bar.

    Rows 0 and 1 have no covariance at all (bar 2, the first holding
    something, is forecast from row 1), row ``NO_COVARIANCE_BAR`` none for
    industry_2; the last symbol (uncovered in ``_plant``) has no specific risk.
    """
    k = len(FACTORS)
    roots = rng.normal(0, 0.01, size=(len(bars), k, k))
    covariance = roots @ roots.transpose(0, 2, 1) + 1e-5 * np.eye(k)
    covariance[:2] = np.nan
    covariance[NO_COVARIANCE_BAR, 2, :] = np.nan
    covariance[NO_COVARIANCE_BAR, :, 2] = np.nan
    specific_risk = rng.uniform(0.005, 0.02, size=(len(bars), len(symbols)))
    specific_risk[:, -1] = np.nan
    return xr.Dataset(
        {
            "factor_covariance": (("timestamp", "factor_i", "factor_j"), covariance),
            "specific_risk": (("timestamp", "symbol"), specific_risk),
        },
        coords={"timestamp": bars, "symbol": symbols, "factor_i": list(FACTORS), "factor_j": list(FACTORS)},
    )


def _weights() -> xr.DataArray:
    """Long-short targets every second bar; D is left alone from bar 8 so it is settled."""
    rng = np.random.default_rng(7)
    rows = np.full((T, len(SYMBOLS)), np.nan)
    for t in range(0, T - 1, 2):
        row = rng.uniform(-1.0, 1.0, size=len(SYMBOLS))
        row = 0.95 * row / np.abs(row).sum()
        if t >= DELIST_BAR - 1:
            row[3] = np.nan
            others = ~np.isnan(row)
            row[others] = 0.6 * row[others] / np.abs(row[others]).sum()
        rows[t] = row
    rows[0, 3] = 0.25  # D held from the start, so its delisting is settled
    rows[0] = rows[0] * 0.95 / np.abs(rows[0]).sum()
    return xr.DataArray(rows, dims=("timestamp", "symbol"), coords={"timestamp": BARS, "symbol": SYMBOLS})


def _risk_model(
    root, prices, exposures, regression, estimate, *, start=BARS[0], estimate_start=None, name="risk"
) -> StubRiskModel:
    planted = root / f"{name}_planted.zarr"
    planted_estimate = root / f"{name}_planted_estimate.zarr"
    regression.to_zarr(planted, mode="w")
    estimate.to_zarr(planted_estimate, mode="w")
    model = StubRiskModel(StubRiskConfig(
        exposures=Exposures(BaseFactorConfig(warmup_bars=0, dataset=FrameDataset(exposures))),
        dataset=FrameDataset(prices),
        exposure_data_strategy="cal",
        price_column="close",
        regression_path=str(root / f"{name}_regression.zarr"),
        estimate_path=str(root / f"{name}_estimate.zarr"),
        planted=str(planted),
        planted_estimate=str(planted_estimate),
    ))
    model.regression.build(start, regression["timestamp"].values[-1])
    model.estimate.build(estimate_start or start, estimate["timestamp"].values[-1])
    return model


def _backtester(prices, risk_model=None, output_dir=None) -> WeightsVectorBt:
    return WeightsVectorBt(WeightsBacktestConfig(
        price_dataset=FrameDataset(prices),
        start_date=str(BARS[0].date()),
        end_date=str(BARS[-1].date()),
        output_dir=output_dir,
        rebalance_periods=1,
        fees=FEES,
        slippage=SLIPPAGE,
        init_cash=INIT_CASH,
        fill_price_column="open",
        valuation_price_column="close",
        trading_days_per_year=252,
        session_minutes_per_day=390,
        risk_model=risk_model,
    ))


@pytest.fixture(scope="module")
def planted():
    return _plant()


@pytest.fixture(scope="module")
def model(planted, tmp_path_factory):
    return _risk_model(tmp_path_factory.mktemp("risk"), *planted)


@pytest.fixture(scope="module")
def run(planted, model, tmp_path_factory):
    prices = planted[0]
    output = tmp_path_factory.mktemp("runs")
    return _backtester(prices, model, output_dir=str(output)).run_weights(_weights())


def _start_of_bar_holdings(prices) -> np.ndarray:
    """The holdings at the close of each bar before, from the Execution rules' replay."""
    open_ = prices["open"].values
    close = prices["close"].values
    delisted = FrameDataset(prices).delisting_bars(prices, "close").values
    book = replay(
        _weights().values, open_, close, delisted,
        ExecutionSettings(fees=FEES, slippage=SLIPPAGE), init_cash=INIT_CASH,
    )
    valued = np.nan_to_num(pd.DataFrame(close).ffill().to_numpy())
    worth = book.shares * valued
    at_close = worth / (book.cash + worth.sum(axis=1))[:, None]
    holdings = np.zeros_like(at_close)
    holdings[1:] = at_close[:-1]
    return holdings


def test_the_terms_sum_to_the_nav_return_at_every_bar(run):
    attribution = run.simulation.factor_attribution
    assert tuple(attribution["term"].values) == TERMS
    np.testing.assert_allclose(
        attribution["contribution"].sum("term").values, run.simulation.returns.values, atol=1e-14
    )
    np.testing.assert_allclose(
        attribution["contribution"].sel(term="factor").values,
        attribution["factor_contribution"].sum("factor").values,
        atol=1e-14,
    )


def test_the_factor_contribution_is_held_exposure_times_factor_return(planted, run):
    prices, exposures, regression, _ = planted
    holdings = _start_of_bar_holdings(prices)
    factor_return = np.nan_to_num(regression["factor_return"].values)
    specific = regression["specific_return"].values
    expected = np.zeros((T, len(FACTORS)))
    for t in range(1, T):
        covered = (holdings[t] != 0) & np.isfinite(INDUSTRY) & np.isfinite(specific[t])
        x = np.column_stack(
            [np.ones(len(SYMBOLS)), INDUSTRY == 1.0, INDUSTRY == 2.0, exposures["style"].values[t - 1]]
        )
        expected[t] = (holdings[t][covered] @ x[covered]) * factor_return[t]
    np.testing.assert_allclose(
        run.simulation.factor_attribution["factor_contribution"].values, expected, atol=1e-12
    )


def test_rejected_orders_and_settlements_shape_the_attributed_holdings(planted, run):
    prices, _, regression, _ = planted
    simulation = run.simulation
    assert [r["axis_symbol"] for r in simulation.rejected_orders] == ["B"]
    assert [r["axis_symbol"] for r in simulation.settlements] == ["D"]
    holdings = _start_of_bar_holdings(prices)
    # B kept its pre-rejection holding through the rejected fill.
    assert holdings[REJECT_BAR + 1, 1] != 0.0
    # D is held up to its settlement and attributed nothing afterwards.
    assert holdings[DELIST_BAR + 1, 3] != 0.0
    assert (holdings[DELIST_BAR + 2:, 3] == 0.0).all()
    specific = regression["specific_return"].values
    attributed = simulation.factor_attribution["contribution"].sel(term="specific").values
    for t in range(1, T):
        covered = (holdings[t] != 0) & np.isfinite(INDUSTRY) & np.isfinite(specific[t])
        assert attributed[t] == pytest.approx(holdings[t][covered] @ specific[t][covered], abs=1e-12)


def test_held_symbols_without_exposures_are_uncovered(planted, run):
    prices = planted[0]
    holdings = _start_of_bar_holdings(prices)
    close = pd.DataFrame(prices["close"].values).ffill().to_numpy()
    own = np.vstack([np.zeros((1, len(SYMBOLS))), close[1:] / close[:-1] - 1.0])
    uncovered = run.simulation.factor_attribution["contribution"].sel(term="uncovered").values
    # F has no industry; D, settled at its last valuation, has no specific return
    # on its settlement bar and earns nothing there.
    np.testing.assert_allclose(uncovered[1:], (holdings[:, 5] * own[:, 5])[1:], atol=1e-12)
    covered = run.simulation.factor_attribution["covered_weight"].values
    assert np.nanmin(covered) < 1.0


def test_a_thin_industry_without_factor_return_contributes_zero(planted, run):
    prices = planted[0]
    holdings = _start_of_bar_holdings(prices)
    assert np.abs(holdings[THIN_BAR][INDUSTRY == 2.0]).sum() > 0
    industry_2 = run.simulation.factor_attribution["factor_contribution"].sel(factor="industry_2")
    assert float(industry_2.isel(timestamp=THIN_BAR)) == 0.0
    assert float(np.abs(industry_2).sum()) > 0


def test_log_contributions_add_up_to_log_nav(run):
    attribution = run.simulation.factor_attribution
    value = run.simulation.value.values
    cumulative = attribution["log_contribution"].sum("term").cumsum("timestamp").values
    np.testing.assert_allclose(cumulative, np.log(value / value[0]), atol=1e-12)
    np.testing.assert_allclose(
        attribution["factor_log_contribution"].sum("factor").values,
        attribution["log_contribution"].sel(term="factor").values,
        atol=1e-14,
    )


def test_the_metrics_report_annualized_log_growth_and_coverage(run):
    block = run.metrics["factor_attribution"]
    assert sorted(block) == ["whole"]
    whole = block["whole"]
    growth = whole["annualized_log_return"]
    value = run.simulation.value.values
    years = T / 252
    assert growth["total"] == pytest.approx(np.log(value[-1] / value[0]) / years, rel=1e-10)
    assert sum(growth[term] for term in TERMS) == pytest.approx(growth["total"], rel=1e-10)
    assert sorted(whole["factor_annualized_log_return"]) == sorted(FACTORS)
    assert sum(whole["factor_annualized_log_return"].values()) == pytest.approx(growth["factor"], rel=1e-10)
    coverage = whole["coverage"]
    assert 0.0 < coverage["min_covered_weight"] <= coverage["mean_covered_weight"] < 1.0
    assert (coverage["note"] is None) == (coverage["mean_covered_weight"] >= LOW_COVERAGE)


def test_the_run_directory_holds_the_attribution_and_rebuilds_it(run):
    opened = BacktestRun.open(run.run_dir)
    stored = opened.factor_attribution()
    xr.testing.assert_allclose(stored, run.simulation.factor_attribution)
    assert opened.recipe()["risk_model"]["name"].endswith("StubRiskModel")
    assert opened.metrics()["factor_attribution"]["whole"]["annualized_log_return"]["total"] == (
        pytest.approx(run.metrics["factor_attribution"]["whole"]["annualized_log_return"]["total"])
    )
    again = opened.rebuild_backtester(output_dir=None).run_weights(_weights())
    xr.testing.assert_allclose(
        again.simulation.factor_attribution, run.simulation.factor_attribution
    )
    assert again.metrics["factor_attribution"] == run.metrics["factor_attribution"]


def test_without_a_risk_model_nothing_changes(planted, run, tmp_path):
    prices = planted[0]
    plain = _backtester(prices, output_dir=str(tmp_path)).run_weights(_weights())
    assert "factor_attribution" not in plain.metrics
    assert plain.simulation.factor_attribution is None
    assert BacktestRun.open(plain.run_dir).factor_attribution() is None
    with_model = {k: v for k, v in run.metrics.items() if k != "factor_attribution"}
    assert sorted(with_model) == sorted(plain.metrics)
    assert with_model["whole"] == plain.metrics["whole"]


def test_risk_stores_not_covering_the_window_are_refused(planted, tmp_path):
    late = _risk_model(tmp_path, *planted, start=BARS[3], name="late")
    prices = planted[0]
    with pytest.raises(ValueError, match="Extend it with extend"):
        _backtester(prices, late).run_weights(_weights())


def test_a_risk_model_on_another_bar_interval_is_refused(planted, tmp_path):
    prices, exposures, regression, estimate = planted
    weekly = regression.isel(timestamp=slice(0, None, 5))
    model = _risk_model(tmp_path, prices, exposures, weekly, estimate, name="weekly")
    with pytest.raises(ValueError, match="bar interval"):
        _backtester(prices, model).run_weights(_weights())


def test_a_risk_model_covering_no_held_symbol_is_refused(planted, tmp_path):
    prices, exposures, regression, estimate = planted
    renamed = [str(10_000 + j) for j in range(len(SYMBOLS))]
    model = _risk_model(
        tmp_path,
        prices.assign_coords(symbol=renamed),
        exposures.assign_coords(symbol=renamed),
        regression.assign_coords(symbol=renamed),
        estimate.assign_coords(symbol=renamed),
        name="renamed",
    )
    with pytest.raises(ValueError, match="same symbol axis"):
        _backtester(prices, model).run_weights(_weights())


# --------------------------------------------------------------------------
# Risk attribution (#210)
# --------------------------------------------------------------------------


def _ex_ante_by_hand(planted) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forecast factor and specific variance and per-factor x-sigma-rho of each bar, from row t-1.

    The forecast's coverage (``FactorRiskModel.forecast``): a held symbol
    with every exposure and a specific risk at t-1, not exposed to a factor
    without a covariance there.
    """
    prices, exposures, regression, estimate = planted
    holdings = _start_of_bar_holdings(prices)
    covariance = estimate["factor_covariance"].values
    specific_risk = estimate["specific_risk"].values
    factor_variance, specific_variance = np.zeros(T), np.zeros(T)
    contribution = np.full((T, len(FACTORS)), np.nan)
    for t in range(1, T):
        x_all = np.column_stack(
            [np.ones(len(SYMBOLS)), INDUSTRY == 1.0, INDUSTRY == 2.0, exposures["style"].values[t - 1]]
        )
        kept = np.isfinite(np.diag(covariance[t - 1]))
        covered = (
            (holdings[t] != 0) & np.isfinite(INDUSTRY) & np.isfinite(specific_risk[t - 1])
            & ~(x_all[:, ~kept] != 0).any(axis=1)
        )
        x = np.where(kept, holdings[t][covered] @ x_all[covered], 0.0)
        fx = np.zeros(len(FACTORS))
        fx[kept] = covariance[t - 1][np.ix_(kept, kept)] @ x[kept]
        factor_variance[t] = x @ fx
        specific_variance[t] = (holdings[t][covered] ** 2) @ (specific_risk[t - 1][covered] ** 2)
        sigma = np.sqrt(factor_variance[t] + specific_variance[t])
        if sigma > 0:
            contribution[t] = x * fx / sigma
    return factor_variance, specific_variance, contribution


def test_the_ex_ante_forecast_of_bar_t_is_the_estimate_row_of_t_minus_1(planted, run):
    factor_variance, specific_variance, contribution = _ex_ante_by_hand(planted)
    attribution = run.simulation.factor_attribution
    np.testing.assert_allclose(attribution["ex_ante_factor_variance"].values, factor_variance, atol=1e-16)
    np.testing.assert_allclose(attribution["ex_ante_specific_variance"].values, specific_variance, atol=1e-16)
    np.testing.assert_allclose(attribution["factor_risk_contribution"].values, contribution, atol=1e-14)
    # Bar 1 holds nothing; row 1 has no covariance, so bar 2's forecast covers no holding.
    assert factor_variance[1] == specific_variance[1] == 0.0
    assert factor_variance[2] == specific_variance[2] == 0.0
    assert np.isnan(attribution["factor_risk_contribution"].values[2]).all()
    # industry_2 has no covariance in row NO_COVARIANCE_BAR: it and its holdings
    # (C, D, E) add no risk to the next bar.
    assert contribution[NO_COVARIANCE_BAR + 1, 2] == 0.0
    assert np.abs(contribution[NO_COVARIANCE_BAR + 2:, 2]).sum() > 0


def test_ex_ante_contributions_sum_to_the_forecast_volatility(run):
    attribution = run.simulation.factor_attribution
    sigma = np.sqrt(
        attribution["ex_ante_factor_variance"] + attribution["ex_ante_specific_variance"]
    ).values
    parts = (
        attribution["factor_risk_contribution"].sum("factor")
        + attribution["specific_risk_contribution"]
    ).values
    forecast = sigma > 0
    # Row 1 has no covariance: bar 2's forecast covers no holding.
    assert not forecast[2] and forecast[3:].all()
    np.testing.assert_allclose(parts[forecast], sigma[forecast], rtol=1e-12)
    # A bar without a forecast (nothing held) has no contributions.
    assert np.isnan(parts[~forecast]).all()


def test_ex_ante_leaves_out_the_uncovered_holdings(planted, run):
    prices = planted[0]
    holdings = _start_of_bar_holdings(prices)
    # F (no exposures) is held, uncovered, and adds nothing to the forecast.
    assert (holdings[1:, 5] != 0).any()
    _, specific_variance, _ = _ex_ante_by_hand(planted)
    np.testing.assert_allclose(
        run.simulation.factor_attribution["ex_ante_specific_variance"].values, specific_variance
    )


def test_the_metrics_report_ex_ante_and_ex_post_risk(run):
    whole = run.metrics["factor_attribution"]["whole"]
    attribution = run.simulation.factor_attribution
    scale = np.sqrt(252)
    ex_ante = whole["ex_ante_risk"]
    sigma = np.sqrt(attribution["ex_ante_factor_variance"] + attribution["ex_ante_specific_variance"])
    forecast = sigma.values > 0  # the means are over the bars with a forecast
    assert ex_ante["volatility"]["total"] == pytest.approx(float(sigma[forecast].mean()) * scale)
    assert ex_ante["volatility"]["factor"] == pytest.approx(
        float(np.sqrt(attribution["ex_ante_factor_variance"][forecast]).mean()) * scale
    )
    assert ex_ante["contribution"]["factor"] + ex_ante["contribution"]["specific"] == pytest.approx(
        ex_ante["volatility"]["total"]
    )
    assert sum(ex_ante["factor_contribution"].values()) == pytest.approx(ex_ante["contribution"]["factor"])

    ex_post = whole["ex_post_risk"]
    returns = run.simulation.returns.values
    assert ex_post["volatility"] == pytest.approx(float(np.std(returns, ddof=1)) * scale)
    assert sorted(ex_post["term_contribution"]) == sorted(TERMS)
    assert sum(ex_post["term_contribution"].values()) == pytest.approx(ex_post["volatility"])
    assert sum(ex_post["factor_contribution"].values()) == pytest.approx(
        ex_post["term_contribution"]["factor"]
    )
    factor_term = attribution["contribution"].sel(term="factor").values
    assert ex_post["term_contribution"]["factor"] == pytest.approx(
        np.cov(factor_term, returns)[0, 1] / np.std(returns, ddof=1) * scale
    )


def test_the_run_directory_holds_the_per_bar_risk(run):
    stored = BacktestRun.open(run.run_dir).factor_attribution()
    for name in ("ex_ante_factor_variance", "ex_ante_specific_variance", "specific_risk_contribution"):
        assert stored[name].dims == ("timestamp",), name
    assert stored["factor_risk_contribution"].dims == ("timestamp", "factor")


def test_an_estimate_store_not_covering_the_window_is_refused(planted, tmp_path):
    late = _risk_model(tmp_path, *planted, estimate_start=BARS[3], name="late_estimate")
    with pytest.raises(ValueError, match="estimate.read.*Extend it with extend"):
        _backtester(planted[0], late).run_weights(_weights())


# --------------------------------------------------------------------------
# Factor groups (#211)
# --------------------------------------------------------------------------


def test_group_contributions_sum_to_the_factor_term(run):
    whole = run.metrics["factor_attribution"]["whole"]
    groups = whole["group_annualized_log_return"]
    assert list(groups) == ["country", "industry", "style"]
    assert sum(groups.values()) == pytest.approx(whole["annualized_log_return"]["factor"], rel=1e-10)
    factors = whole["factor_annualized_log_return"]
    assert groups["industry"] == pytest.approx(factors["industry_1"] + factors["industry_2"])
    assert groups["country"] == pytest.approx(factors["market"])


def test_style_exposures_and_the_industry_table(run):
    whole = run.metrics["factor_attribution"]["whole"]
    attribution = run.simulation.factor_attribution
    holding = attribution["gross_weight"].values > 0
    exposure = attribution["exposure"].sel(factor="style").values[holding]
    assert whole["style_mean_exposure"] == {"style": pytest.approx(float(exposure.mean()))}
    factors = whole["factor_annualized_log_return"]
    ranked = sorted(["industry_1", "industry_2"], key=factors.get, reverse=True)
    top = whole["industries"]["top"]
    assert [row["factor"] for row in top] == ranked
    assert whole["industries"]["bottom"] == []
    net = attribution["exposure"].sel(factor=ranked[0]).values[holding]
    assert top[0] == {
        "factor": ranked[0],
        "annualized_log_return": pytest.approx(factors[ranked[0]]),
        "mean_exposure": pytest.approx(float(net.mean())),
    }


def test_the_run_directory_holds_the_exposures_with_their_groups(run):
    stored = BacktestRun.open(run.run_dir).factor_attribution()
    assert stored["exposure"].dims == ("timestamp", "factor")
    assert dict(zip(stored["factor"].values, stored["group"].values)) == GROUPS
    # Labels default to the factor names (#215).
    assert list(stored["label"].values) == list(FACTORS)


def test_risk_contributions_are_reported_per_group(run):
    whole = run.metrics["factor_attribution"]["whole"]
    for block in (whole["ex_ante_risk"], whole["ex_post_risk"]):
        factors = block["factor_contribution"]
        assert block["group_contribution"] == {
            "country": pytest.approx(factors["market"]),
            "industry": pytest.approx(factors["industry_1"] + factors["industry_2"]),
            "style": pytest.approx(factors["style"]),
        }
