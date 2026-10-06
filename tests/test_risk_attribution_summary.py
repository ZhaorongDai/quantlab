"""``attribution_summary`` on a hand-built attribution (#209, #210, #211).

The backtest entry points lock the summary on a four-factor model; what they
cannot reach is locked here on a dataset built by hand: the industry table
of a model with many industries (top and bottom, never overlapping) and a
segment's bars picked by a mask, whose terms reconcile to the segment's NAV
log growth.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.risk.attribution import TERMS, TOP_INDUSTRIES, attribution_summary

N_INDUSTRIES = 12
FACTORS = ["country", *(f"industry_{j}" for j in range(N_INDUSTRIES)), "style_a", "style_b"]
GROUPS = ["country", *["industry"] * N_INDUSTRIES, "style", "style"]
BARS = pd.bdate_range("2024-01-01", periods=40)


def _attribution(seed: int = 211) -> xr.Dataset:
    rng = np.random.default_rng(seed)
    t, k = BARS.size, len(FACTORS)
    factor = rng.normal(0, 0.002, size=(t, k))
    contribution = np.column_stack([factor.sum(axis=1), rng.normal(0, 0.005, size=(t, len(TERMS) - 1))])
    returns = contribution.sum(axis=1)
    link = (np.log1p(returns) / returns)[:, None]
    forecast = rng.uniform(0.0, 0.001, size=(t, k))
    return xr.Dataset(
        {
            "contribution": (("timestamp", "term"), contribution),
            "log_contribution": (("timestamp", "term"), contribution * link),
            "factor_contribution": (("timestamp", "factor"), factor),
            "factor_log_contribution": (("timestamp", "factor"), factor * link),
            "return": ("timestamp", returns),
            "exposure": (("timestamp", "factor"), rng.normal(size=(t, k))),
            "ex_ante_factor_variance": ("timestamp", rng.uniform(1e-5, 1e-4, size=t)),
            "ex_ante_specific_variance": ("timestamp", rng.uniform(1e-5, 1e-4, size=t)),
            "factor_risk_contribution": (("timestamp", "factor"), forecast),
            "specific_risk_contribution": ("timestamp", rng.uniform(0.0, 0.01, size=t)),
            "covered_weight": ("timestamp", rng.uniform(0.8, 1.0, size=t)),
            "gross_weight": ("timestamp", np.ones(t)),
        },
        coords={"timestamp": BARS, "term": list(TERMS), "factor": FACTORS, "group": ("factor", GROUPS)},
    )


def test_the_industry_table_lists_the_top_and_bottom_industries_once():
    summary = attribution_summary(_attribution(), bars_per_year=252)
    growth = summary["factor_annualized_log_return"]
    ranked = sorted((f for f in FACTORS if f.startswith("industry_")), key=growth.get, reverse=True)
    table = summary["industries"]
    assert [row["factor"] for row in table["top"]] == ranked[:TOP_INDUSTRIES]
    assert [row["factor"] for row in table["bottom"]] == ranked[::-1][:TOP_INDUSTRIES]
    assert not {row["factor"] for row in table["top"]} & {row["factor"] for row in table["bottom"]}


def test_a_short_industry_list_is_not_repeated_at_the_bottom():
    attribution = _attribution().isel(factor=[0, 1, 2, 3, 4, 5, 6, 7, -1])  # seven industries
    table = attribution_summary(attribution, bars_per_year=252)["industries"]
    assert len(table["top"]) == TOP_INDUSTRIES and len(table["bottom"]) == 2


def test_a_segment_reconciles_to_its_own_nav_log_growth():
    attribution = _attribution()
    bars = np.zeros(BARS.size, dtype=bool)
    bars[5:17] = True
    summary = attribution_summary(attribution, bars_per_year=252, segment=bars)
    returns = attribution["return"].values[bars]
    years = bars.sum() / 252
    assert summary["annualized_log_return"]["total"] == pytest.approx(np.log1p(returns).sum() / years)
    growth = summary["annualized_log_return"]
    assert sum(growth[term] for term in TERMS) == pytest.approx(growth["total"])
    ex_post = summary["ex_post_risk"]
    assert ex_post["volatility"] == pytest.approx(np.std(returns, ddof=1) * np.sqrt(252))
    assert sum(ex_post["term_contribution"].values()) == pytest.approx(ex_post["volatility"])
    groups = summary["group_annualized_log_return"]
    assert sum(groups.values()) == pytest.approx(growth["factor"])
    assert sorted(summary["style_mean_exposure"]) == ["style_a", "style_b"]


def test_a_single_bar_has_no_realized_volatility():
    bars = np.zeros(BARS.size, dtype=bool)
    bars[3] = True
    ex_post = attribution_summary(_attribution(), bars_per_year=252, segment=bars)["ex_post_risk"]
    assert ex_post["volatility"] is None
    assert set(ex_post["group_contribution"].values()) == {None}
