"""StockFit's as-reported facts, derived metrics, share classes, scores and trends.

Seam 1 of the StockFit spec (the raw-tier refresh entry, HTTP replaced by
``tests.stockfit_fixtures.FakeVendor``) lands the endpoints; seam 2 builds
each panel from the raw tier on disk. The recorded bodies are VERBATIM; a
test that needs what the recordings lack (a re-pull with a changed value, a
statement under the query the refresh sends) edits the vendor's bodies first
and marks the edit SYNTHETIC.

The vendor's clock pulls on Monday 2026-11-16 at 09:00 New York time, so the
raw tier is complete through 2026-11-15.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tests.stockfit_fixtures import (
    GOOGL,
    KEY,
    PLUG,
    FakeClock,
    FakeVendor,
    filing_row,
    fixture,
)

#: Each new endpoint and the recorded body it lands for GOOGL.
GOOGL_BODIES = {
    "growth_quarter": "growth_googl_quarter",
    "key_metrics_quarter": "key_metrics_googl_quarter",
    "eps_history_quarter": "eps_history_googl_quarter",
    "dividend_history_quarter": "dividend_history_googl_quarter",
    "scores": "scores_googl",
    "trends": "trends_googl",
    "share_classes": "share_classes_googl",
}

INCOME = "income_statement_quarter"


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("STOCKFIT_API_KEY", KEY)


def _serve(vendor, endpoint: str, cik: int, body) -> None:
    """Make the vendor answer the query the refresh sends for ``endpoint`` with ``body``."""
    from quantlab.acquisition.stockfit.endpoints import ENDPOINTS
    from tests.stockfit_fixtures import _identity

    spec = ENDPOINTS[endpoint]
    vendor.bodies[(spec.path, _identity(spec.query(str(cik))))] = body


def _googl_vendor(clock=None) -> FakeVendor:
    """The recorded GOOGL bodies, each served under the query the refresh sends.

    Re-keying is SYNTHETIC (the recordings were made with other ``limit`` and
    ``splitAdjust`` values; the bodies are VERBATIM). GOOGL's quarterly income
    statement is the recording made with ``includeSourceDetails``.
    """
    vendor = FakeVendor(clock or FakeClock())
    for endpoint, name in GOOGL_BODIES.items():
        _serve(vendor, endpoint, GOOGL, fixture(name))
    _serve(vendor, INCOME, GOOGL, fixture("income_statement_googl_quarter_source_details"))
    return vendor


def _refresh(tmp_path, vendor, endpoints, ciks):
    from quantlab.acquisition.stockfit.refresh import refresh_raw_tier

    client = vendor.client()
    result = refresh_raw_tier(tmp_path / "downloads", endpoints=endpoints, ciks=ciks, client=client)
    assert not result.failed, result.summary()
    return tmp_path / "downloads" / "stockfit"


def _on(panel, name, day, cik):
    return panel[name].sel(timestamp=pd.Timestamp(day), symbol=cik).item()


# -- seam 1: the refresh lands the endpoints ---------------------------------------


def test_a_refresh_lands_each_companys_as_reported_derived_and_share_class_pulls(env, tmp_path):
    from quantlab.dataset.stockfit import raw

    vendor = _googl_vendor()
    _serve(vendor, "as_reported_quarter", PLUG, fixture("as_reported_plug_quarter"))
    _serve(vendor, "as_reported_annual", PLUG, fixture("as_reported_plug_annual"))
    root = _refresh(tmp_path, vendor, ("as_reported_quarter", "as_reported_annual", *GOOGL_BODIES), (PLUG, GOOGL))

    for endpoint, name in GOOGL_BODIES.items():
        assert raw.read_pull(root, endpoint, str(GOOGL)) == fixture(name), endpoint
    assert raw.read_pull(root, "as_reported_quarter", str(PLUG)) == fixture("as_reported_plug_quarter")
    assert raw.read_pull(root, "as_reported_annual", str(PLUG)) == fixture("as_reported_plug_annual")
    eps = vendor.requested("/api/earnings/eps-history")
    assert eps and all(str(params["splitAdjust"]).lower() == "false" for params in eps)  # as filed, never re-split


def test_a_new_10q_re_pulls_the_filers_as_reported_derived_and_share_class_pulls(env, tmp_path):
    vendor = _googl_vendor()
    endpoints = ("as_reported_quarter", *GOOGL_BODIES)
    _refresh(tmp_path, vendor, endpoints, (GOOGL,))
    vendor.clock.advance(days=1)
    vendor.lists["/api/filings/latest"] = [filing_row("10-Q", GOOGL, vendor.clock.now().date().isoformat())]
    vendor.calls.clear()
    _refresh(tmp_path, vendor, ("filings_latest", *endpoints), (GOOGL,))

    paths = {url.split("/v1")[-1] for url, _, _ in vendor.calls}
    assert {"/api/financials/as-reported", "/api/financials/growth", "/api/ownership/share-classes",
            "/api/financials/scores", "/api/earnings/trends"} <= paths


# -- seam 2: the as-reported panel ---------------------------------------------------


def _as_reported_vendor() -> FakeVendor:
    """PLUG's recorded quarterly as-reported facts, and a second company using one us-gaap concept.

    The second company's body (served for GOOGL's CIK) is SYNTHETIC.
    """
    vendor = FakeVendor(FakeClock())
    _serve(vendor, "as_reported_quarter", PLUG, fixture("as_reported_plug_quarter"))
    _serve(vendor, "as_reported_quarter", GOOGL, [{  # SYNTHETIC
        "period": "2026-09-30", "fiscalYear": 2026, "fiscalPeriod": "Q3",
        "dateFiled": "2026-10-29", "acceptedAt": "2026-10-29T00:29:06Z",
        "facts": {"us-gaap:NetIncomeLoss": 1.0, "goog:OtherBets": 2.0}, "sources": {},
    }])
    return vendor


def _as_reported_panel(tmp_path, root, concepts, start="2026-08-01"):
    from quantlab.dataset.config import StockFitAsReportedConfig
    from quantlab.dataset.stockfit.as_reported import StockFitAsReportedDataset

    config = StockFitAsReportedConfig(
        zarr_file_path=str(tmp_path / "as_reported.zarr"), raw_data_dir_path=str(root),
        concepts=concepts, start_date=start,
    )
    return StockFitAsReportedDataset(config).update().panel(start, "2026-12-31").load()


def test_the_concept_list_is_regenerated_from_the_raw_tier_without_filer_extensions(env, tmp_path):
    from quantlab.dataset.stockfit.as_reported import as_reported_concepts

    root = _refresh(tmp_path, _as_reported_vendor(), ("as_reported_quarter",), (PLUG, GOOGL))

    assert as_reported_concepts(root, min_share=0.75) == ("us-gaap:NetIncomeLoss",)
    every = as_reported_concepts(root, min_share=0.5)
    assert "us-gaap:CostOfRevenue" in every and all(c.startswith("us-gaap:") for c in every)


def test_an_as_reported_value_shows_from_its_filing_flagged_as_possibly_restated(env, tmp_path):
    from quantlab.dataset.stockfit.as_reported import as_reported_concepts

    root = _refresh(tmp_path, _as_reported_vendor(), ("as_reported_quarter",), (PLUG, GOOGL))
    panel = _as_reported_panel(tmp_path, root, as_reported_concepts(root, min_share=0.75))

    q2 = fixture("as_reported_plug_quarter")[1]
    assert _on(panel, "NetIncomeLoss", "2026-11-09", PLUG) == q2["facts"]["us-gaap:NetIncomeLoss"]
    assert _on(panel, "NetIncomeLoss", "2026-11-10", PLUG) == 99_766_000  # accepted 17:01 on 11-09
    assert _on(panel, "period_end", "2026-11-10", PLUG) == pd.Timestamp("2026-09-30").value
    assert _on(panel, "may_include_restatement", "2026-11-10", PLUG) == 1.0
    assert pd.isna(_on(panel, "NetIncomeLoss", "2026-08-10", PLUG))  # Q2 accepted 17:06 on 08-10


def test_an_as_reported_period_ending_after_its_filing_date_never_becomes_the_period_shown(env, tmp_path):
    vendor = FakeVendor(FakeClock())
    _serve(vendor, "as_reported_quarter", GOOGL, [  # SYNTHETIC
        {"period": "2026-06-30", "fiscalYear": 2026, "fiscalPeriod": "Q2", "dateFiled": "2026-07-23",
         "facts": {"us-gaap:NetIncomeLoss": 1.0}, "sources": {}},
        {"period": "2029-03-31", "fiscalYear": 2029, "fiscalPeriod": "Q1", "dateFiled": "2026-07-23",
         "facts": {"us-gaap:NetIncomeLoss": 9.0}, "sources": {}},
        {"period": "2026-09-30", "fiscalYear": 2026, "fiscalPeriod": "Q3", "dateFiled": "2026-10-29",
         "facts": {"us-gaap:NetIncomeLoss": 2.0}, "sources": {}},
    ])
    root = _refresh(tmp_path, vendor, ("as_reported_quarter",), (GOOGL,))
    panel = _as_reported_panel(tmp_path, root, ("us-gaap:NetIncomeLoss",), start="2026-07-01")

    assert _on(panel, "NetIncomeLoss", "2026-07-23", GOOGL) == 1.0
    assert _on(panel, "NetIncomeLoss", "2026-10-29", GOOGL) == 2.0


def test_a_value_a_re_pull_changes_shows_from_the_pull_day_unflagged(env, tmp_path):
    vendor = _as_reported_vendor()
    root = _refresh(tmp_path, vendor, ("as_reported_quarter",), (PLUG,))
    restated = fixture("as_reported_plug_quarter")
    restated[0]["facts"]["us-gaap:NetIncomeLoss"] = 100_000_000  # SYNTHETIC silent change
    _serve(vendor, "as_reported_quarter", PLUG, restated)
    vendor.clock.advance(days=2)  # 2026-11-18, 09:00 in New York; a filing (SYNTHETIC) drives the re-pull
    vendor.lists["/api/filings/latest"] = [filing_row("10-Q", PLUG, "2026-11-17")]
    _refresh(tmp_path, vendor, ("filings_latest", "as_reported_quarter"), (PLUG,))
    vendor.clock.advance(days=1)  # a second company's first pull: the raw tier is complete through 2026-11-18
    _refresh(tmp_path, vendor, ("filings_latest", "as_reported_quarter"), (PLUG, GOOGL))
    panel = _as_reported_panel(tmp_path, root, ("us-gaap:NetIncomeLoss",), start="2026-07-01")

    assert _on(panel, "NetIncomeLoss", "2026-11-17", PLUG) == 99_766_000
    assert _on(panel, "may_include_restatement", "2026-11-17", PLUG) == 1.0
    assert _on(panel, "NetIncomeLoss", "2026-11-18", PLUG) == 100_000_000
    assert _on(panel, "may_include_restatement", "2026-11-18", PLUG) == 0.0
    assert _on(panel, "release_date", "2026-11-18", PLUG) == pd.Timestamp("2026-11-09").value


# -- seam 2: derived metrics and share classes at the statements' filing dates -------
#
# GOOGL's recorded quarterly income statement holds two quarters: 2026-09-30,
# accepted 2026-10-29T00:29:06Z (20:29 on 10-28 in New York, after the close),
# and 2026-06-30, accepted 2026-07-23T01:15:54Z (21:15 on 07-22).


def _derived_panel(tmp_path, table, start="2026-07-01"):
    from quantlab.dataset.config import StockFitDerivedConfig
    from quantlab.dataset.stockfit.derived import StockFitDerivedDataset

    root = _refresh(tmp_path, _googl_vendor(), (INCOME, *GOOGL_BODIES), (GOOGL,))
    config = StockFitDerivedConfig(
        zarr_file_path=str(tmp_path / f"{table}.zarr"), raw_data_dir_path=str(root), table=table, start_date=start,
    )
    dataset = StockFitDerivedDataset(config).update()
    return dataset, dataset.panel(start, "2026-12-31").load()


def test_a_derived_row_shows_from_the_filing_of_the_same_statement_period(env, tmp_path):
    _, panel = _derived_panel(tmp_path, "growth")

    assert _on(panel, "revenueGrowth", "2026-10-28", GOOGL) == 0.0901
    assert _on(panel, "revenueGrowth", "2026-10-29", GOOGL) == 0.065
    assert _on(panel, "release_date", "2026-10-29", GOOGL) == pd.Timestamp("2026-10-29").value
    assert _on(panel, "period_end", "2026-10-29", GOOGL) == pd.Timestamp("2026-09-30").value
    assert _on(panel, "may_include_restatement", "2026-10-29", GOOGL) == 1.0
    assert pd.isna(_on(panel, "revenueGrowth", "2026-07-22", GOOGL))


def test_a_derived_row_with_no_matching_statement_period_is_dropped_and_counted(env, tmp_path):
    dataset, panel = _derived_panel(tmp_path, "growth", start="2025-12-01")

    assert dataset.build_report.unmatched == 2  # 2026-03-31 and 2025-12-31: no statement period
    assert dataset.build_report.rows == 2
    assert panel["revenueGrowth"].sel(symbol=GOOGL, timestamp=slice("2025-12-01", "2026-07-22")).isnull().all()


def test_a_derived_row_borrows_the_filing_of_a_statement_period_dated_only_by_its_sources(env, tmp_path):
    from quantlab.dataset.config import StockFitDerivedConfig
    from quantlab.dataset.stockfit.derived import StockFitDerivedDataset

    vendor = _googl_vendor()
    income = fixture("income_statement_googl_quarter_source_details")
    income[0].pop("dateFiled")  # SYNTHETIC: the 2026-09-30 quarter without a filing of its own
    income[0].pop("acceptedAt")
    _serve(vendor, INCOME, GOOGL, income)
    root = _refresh(tmp_path, vendor, (INCOME, "growth_quarter"), (GOOGL,))
    config = StockFitDerivedConfig(
        zarr_file_path=str(tmp_path / "growth.zarr"), raw_data_dir_path=str(root), table="growth",
        start_date="2026-07-01",
    )
    dataset = StockFitDerivedDataset(config).update()
    panel = dataset.panel("2026-07-01", "2026-12-31").load()

    # Its source, the 10-Q accepted 2026-10-29T00:29:06Z (the evening before in New York), dates it.
    assert _on(panel, "revenueGrowth", "2026-10-29", GOOGL) == 0.065
    assert _on(panel, "release_date", "2026-10-29", GOOGL) == pd.Timestamp("2026-10-29").value


@pytest.mark.parametrize(
    ("table", "variable", "expected"),
    [
        ("key_metrics", "roe", 4.016300096583611),
        ("eps_history", "epsDiluted", 2.12),
        ("share_classes", "class_count", 3.0),
        ("share_classes", "shares_outstanding", 5_888_000_000 + 5_578_000_000 + 828_000_000),
        ("share_classes", "shares_authorized", 180_000_000_000 + 60_000_000_000 + 60_000_000_000),
    ],
)
def test_each_derived_table_shows_its_values_at_the_borrowed_filing_date(env, tmp_path, table, variable, expected):
    _, panel = _derived_panel(tmp_path, table)

    assert _on(panel, variable, "2026-10-29", GOOGL) == expected
    assert pd.isna(_on(panel, variable, "2026-07-22", GOOGL))


def test_a_company_paying_no_dividend_shows_its_period_with_no_dividend(env, tmp_path):
    _, panel = _derived_panel(tmp_path, "dividend_history")

    assert _on(panel, "period_end", "2026-10-29", GOOGL) == pd.Timestamp("2026-09-30").value
    assert pd.isna(_on(panel, "dividendPerShare", "2026-10-29", GOOGL))  # served as null


# -- seam 2: scores and trends, accumulated from snapshots -----------------------------


def _snapshot_panels(tmp_path):
    """Three daily refreshes of GOOGL's scores and trends; the second serves a changed F-score (SYNTHETIC).

    Pulled on 2026-11-16, 11-18 and 11-19 at 09:00 in New York, each re-pull
    driven by a 10-Q in the recent-filings feed (SYNTHETIC rows).
    """
    from quantlab.dataset.config import StockFitScoresTrendsConfig
    from quantlab.dataset.stockfit.scores_trends import StockFitScoresTrendsDataset

    vendor = _googl_vendor()
    endpoints = ("filings_latest", "scores", "trends")
    root = _refresh(tmp_path, vendor, endpoints, (GOOGL,))
    changed = fixture("scores_googl")
    changed["piotroskiFScore"] = 7  # SYNTHETIC recomputation
    _serve(vendor, "scores", GOOGL, changed)
    vendor.clock.advance(days=2)
    vendor.lists["/api/filings/latest"] = [filing_row("10-Q", GOOGL, "2026-11-17")]
    _refresh(tmp_path, vendor, endpoints, (GOOGL,))
    vendor.clock.advance(days=1)
    vendor.lists["/api/filings/latest"].append(filing_row("10-Q", GOOGL, "2026-11-18"))
    _refresh(tmp_path, vendor, endpoints, (GOOGL,))
    return {
        table: StockFitScoresTrendsDataset(StockFitScoresTrendsConfig(
            zarr_file_path=str(tmp_path / f"{table}.zarr"), raw_data_dir_path=str(root), table=table,
            start_date="2026-11-01",
        )).update().panel("2026-11-01", "2026-12-31").load()
        for table in ("scores", "trends")
    }


def test_scores_and_trends_show_nothing_before_their_first_snapshot(env, tmp_path):
    panels = _snapshot_panels(tmp_path)

    assert pd.isna(_on(panels["scores"], "piotroskiFScore", "2026-11-13", GOOGL))
    assert _on(panels["scores"], "piotroskiFScore", "2026-11-16", GOOGL) == 6
    assert _on(panels["scores"], "altmanZScore", "2026-11-16", GOOGL) == 3.92
    assert _on(panels["scores"], "altmanZone", "2026-11-16", GOOGL) == 2.0  # safe
    assert _on(panels["scores"], "risingRoa", "2026-11-16", GOOGL) == 0.0
    assert _on(panels["scores"], "period_end", "2026-11-16", GOOGL) == pd.Timestamp("2025-12-31").value
    assert pd.isna(_on(panels["trends"], "cagr_revenue3yr", "2026-11-13", GOOGL))
    assert _on(panels["trends"], "cagr_revenue3yr", "2026-11-16", GOOGL) == 12.5
    assert _on(panels["trends"], "margins_gross_direction", "2026-11-16", GOOGL) == 1.0  # expanding
    assert _on(panels["trends"], "quality_roe", "2026-11-16", GOOGL) == 31.8


def test_a_changed_snapshot_shows_from_the_day_of_its_pull(env, tmp_path):
    scores = _snapshot_panels(tmp_path)["scores"]

    assert _on(scores, "piotroskiFScore", "2026-11-17", GOOGL) == 6
    assert _on(scores, "piotroskiFScore", "2026-11-18", GOOGL) == 7
    assert "may_include_restatement" not in scores
