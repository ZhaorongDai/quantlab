"""The Sharadar SFP fund panel: ETFs on the permaticker axis, usable as a benchmark.

SFP converts through the same path as SEP, adjusted prices included; only the
table and its TICKERS rows differ. Every row is invented (`# SYNTHETIC`); the
raw tier is written by the real client through a faked transport.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.sharadar_fixtures import (
    ACTIONS_COLUMNS,
    SEP_COLUMNS,
    TICKERS_COLUMNS,
    FakeTransport,
    action_row,
    bulk_routes,
    csv_text,
    sep_row,
    tickers_row,
)


@pytest.fixture(autouse=True)
def _api_key(monkeypatch):
    monkeypatch.setenv("SHARADAR_API_KEY", "synthetic-key")  # SYNTHETIC


def _pull(download_dir, sfp_rows, tickers_rows, actions_rows=()):
    """Pull SFP, TICKERS and ACTIONS through the client; return the vendor root."""
    from quantlab.acquisition.sharadar.client import SharadarClient

    transport = FakeTransport(
        bulk_routes(
            {
                # SFP has SEP's columns, verbatim (schema/funds, 2026-08-18).
                "funds": csv_text(SEP_COLUMNS, sfp_rows),
                "tickers": csv_text(TICKERS_COLUMNS, tickers_rows),
                "actions": csv_text(ACTIONS_COLUMNS, list(actions_rows)),
            }
        )
    )
    client = SharadarClient(transport=transport, sleep=lambda seconds: None)
    for code in ("sfp", "tickers", "actions"):
        client.bulk_table(code, download_dir)
    return download_dir / "sharadar"


def _fund_config(tmp_path, vendor_root, **fields):
    from quantlab.dataset.config import SharadarDatasetConfig

    return SharadarDatasetConfig(
        zarr_file_path=str(tmp_path / "sharadar_sfp_1d.zarr"),
        raw_data_dir_path=str(vendor_root),
        table="sfp",
        **fields,
    )


def _build(config):
    from quantlab.dataset.sharadar.stock import SharadarStockDataset

    SharadarStockDataset(config).from_raw_data().save()
    return SharadarStockDataset(config)


DAYS = ["2024-01-02", "2024-01-03", "2024-01-04"]
FUND_TICKERS = [
    tickers_row("SFP", 118691, "SPY", category="ETF"),  # SYNTHETIC
    tickers_row("SFP", 777, "CEFX", category="CEF"),  # SYNTHETIC
    # A stock of the same ticker is another security.
    tickers_row("SEP", 999, "SPY", category="Domestic Common Stock"),  # SYNTHETIC
]


def test_the_sfp_table_is_pulled_to_its_own_raw_directory(tmp_path):
    root = _pull(tmp_path / "downloads", [sep_row("SPY", DAYS[0], 470.0)], FUND_TICKERS)  # SYNTHETIC
    assert (root / "sfp" / "sfp.parquet").is_file()


def test_a_fund_panel_keeps_every_fund_category_by_default(tmp_path):
    rows = [sep_row(t, DAYS[0], 10.0) for t in ("SPY", "CEFX")]  # SYNTHETIC
    root = _pull(tmp_path / "downloads", rows, FUND_TICKERS)
    ds = _build(_fund_config(tmp_path, root))
    # The stock default (domestic common stock) would drop every fund.
    assert ds.config.category_filter is None
    assert ds.panel(DAYS[0], DAYS[-1]).symbol.values.tolist() == [777, 118691]


def test_the_stock_default_is_still_domestic_common_stock(tmp_path):
    from quantlab.dataset.config import SharadarDatasetConfig
    from quantlab.dataset.sharadar.stock import SharadarStockDataset

    ds = SharadarStockDataset(
        SharadarDatasetConfig(
            zarr_file_path=str(tmp_path / "s.zarr"), raw_data_dir_path=str(tmp_path)
        )
    )
    assert ds.config.category_filter == (
        "Domestic Common Stock",
        "Domestic Common Stock Primary Class",
        "Domestic Common Stock Secondary Class",
    )


def test_a_fund_category_filter_can_be_named(tmp_path):
    rows = [sep_row(t, DAYS[0], 10.0) for t in ("SPY", "CEFX")]  # SYNTHETIC
    root = _pull(tmp_path / "downloads", rows, FUND_TICKERS)
    panel = _build(_fund_config(tmp_path, root, category_filter=("ETF",))).panel(DAYS[0], DAYS[-1])
    assert panel.symbol.values.tolist() == [118691]


def test_fund_prices_are_adjusted_from_actions(tmp_path):
    rows = [
        sep_row("SPY", DAYS[0], 100.0),  # SYNTHETIC
        sep_row("SPY", DAYS[1], 98.0),  # SYNTHETIC
        sep_row("SPY", DAYS[2], 99.0),  # SYNTHETIC
    ]
    actions = [action_row(DAYS[1], "dividend", "SPY", 2.0)]  # SYNTHETIC
    root = _pull(tmp_path / "downloads", rows, FUND_TICKERS, actions)
    panel = _build(_fund_config(tmp_path, root)).panel(DAYS[0], DAYS[-1]).sel(symbol=118691)
    assert panel["divCash"].values.tolist() == [0.0, 2.0, 0.0]
    np.testing.assert_allclose(panel["adjClose"].values, [100.0, 100.0, 100.0 * 99.0 / 98.0])


def test_an_etf_benchmark_config_selects_one_fund():
    from quantlab.dataset.config import SharadarDatasetConfig

    config = SharadarDatasetConfig.etf_benchmark(
        permaticker=118691,
        zarr_file_path="/data/zarrs/sharadar_spy_1d.zarr",
        raw_data_dir_path="/data/downloads/sharadar",
    )
    assert (config.table, config.permatickers) == ("sfp", (118691,))


def test_a_backtest_takes_an_sfp_fund_as_its_benchmark(tmp_path):
    from quantlab.backtest.predefined.us_equity import (
        USEquityCrossectionSelectStockVectorBt,
    )
    from quantlab.dataset.config import SharadarDatasetConfig
    from quantlab.dataset.sharadar.stock import SharadarStockDataset
    from tests.test_backtest_benchmark import N_BARS, _config

    bars = pd.bdate_range("2024-01-01", periods=N_BARS)
    rows = [
        sep_row("SPY", day.strftime("%Y-%m-%d"), 400.0 + i)  # SYNTHETIC
        for i, day in enumerate(bars)
    ]
    root = _pull(tmp_path / "downloads", rows, FUND_TICKERS)
    benchmark = SharadarDatasetConfig.etf_benchmark(
        permaticker=118691,
        zarr_file_path=str(tmp_path / "sharadar_spy_1d.zarr"),
        raw_data_dir_path=str(root),
    )
    SharadarStockDataset(benchmark).from_raw_data().save()

    config = _config(tmp_path, benchmark_dataset=SharadarStockDataset(benchmark))
    run = USEquityCrossectionSelectStockVectorBt(config).run()
    assert str(run.metrics["benchmark"]["axis_symbol"]) == "118691"
    assert "Total Return [%]" in run.metrics["benchmark"]["whole"]
    assert run.metrics["relative"]["whole"] is not None
