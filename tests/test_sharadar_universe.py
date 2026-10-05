"""The Sharadar universe: S&P 500 membership and the `category` filter.

The raw tier is written by the real client from a faked transport
(`tests/sharadar_fixtures.FakeTransport`). SP500 semantics follow the vendor's
data dictionary: an `added` or `removed` row's date is the effective
membership date, so a stock is a member from its `added` date and is no
longer one on its `removed` date.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tests.sharadar_fixtures import (
    ACTIONS_COLUMNS,
    SEP_COLUMNS,
    SP500_COLUMNS,
    TICKERS_COLUMNS,
    FakeTransport,
    bulk_routes,
    csv_text,
    sep_row,
    sp500_row,
    tickers_row,
)

DOMESTIC = "Domestic Common Stock"  # VERBATIM category value


@pytest.fixture(autouse=True)
def _api_key(monkeypatch):
    monkeypatch.setenv("SHARADAR_API_KEY", "synthetic-key")  # SYNTHETIC


def _pull(download_dir, sep_rows, tickers_rows, sp500_rows=None):
    """Pull SEP, TICKERS, an empty ACTIONS and (when given) SP500; return the vendor root."""
    from quantlab.acquisition.sharadar.client import SharadarClient

    tables = {
        "stocks": csv_text(SEP_COLUMNS, sep_rows),
        "tickers": csv_text(TICKERS_COLUMNS, tickers_rows),
        "actions": csv_text(ACTIONS_COLUMNS, []),
    }
    if sp500_rows is not None:
        tables["sp500"] = csv_text(SP500_COLUMNS, sp500_rows)
    client = SharadarClient(transport=FakeTransport(bulk_routes(tables)), sleep=lambda s: None)
    client.bulk_table("sep", download_dir)
    client.bulk_table("tickers", download_dir)
    client.bulk_table("actions", download_dir)
    if sp500_rows is not None:
        client.bulk_table("sp500", download_dir)
    return download_dir / "sharadar"


def _membership(tmp_path, vendor_root, **fields):
    from quantlab.dataset.config import ConstituentDatasetConfig
    from quantlab.dataset.sharadar.membership import SharadarSP500ConstituentDataset

    config = ConstituentDatasetConfig(
        zarr_file_path=str(tmp_path / "sharadar_sp500_membership.zarr"),
        cache_dir=str(vendor_root),
        **fields,
    )
    return SharadarSP500ConstituentDataset(config)


def _stock(tmp_path, vendor_root, **fields):
    from quantlab.dataset.config import SharadarDatasetConfig
    from quantlab.dataset.sharadar.stock import SharadarStockDataset

    ds = SharadarStockDataset(
        SharadarDatasetConfig(
            zarr_file_path=str(tmp_path / "sharadar_sep_1d.zarr"),
            raw_data_dir_path=str(vendor_root),
            **fields,
        )
    )
    return ds.from_raw_data().get_xarray_dataset()


def _is_member(panel, symbol, day):
    return bool(panel["is_member"].sel(symbol=symbol, timestamp=pd.Timestamp(day)))


# The three tickers every membership test shares.
TICKERS = [
    tickers_row("SEP", 101, "AAA"),  # SYNTHETIC
    tickers_row("SEP", 202, "BBB"),  # SYNTHETIC
    tickers_row("SEP", 303, "CCC"),  # SYNTHETIC
]
PRICES = [sep_row(t, "2024-01-02", 10.0) for t in ("AAA", "BBB", "CCC")]  # SYNTHETIC


def test_a_stock_enters_and_leaves_on_the_effective_dates(tmp_path):
    sp500 = [
        # AAA was a member before the table starts and is removed on 2024-03-04.
        sp500_row("2024-03-04", "removed", "AAA", contraticker="BBB"),  # SYNTHETIC
        # BBB replaces it on the same effective date.
        sp500_row("2024-03-04", "added", "BBB", contraticker="AAA"),  # SYNTHETIC
        sp500_row("2024-06-28", "current", "BBB"),  # SYNTHETIC
    ]
    root = _pull(tmp_path / "downloads", PRICES, TICKERS, sp500)
    panel = _membership(tmp_path, root, start_date="2024-01-01").from_raw_data().get_xarray_dataset()

    assert panel.symbol.values.tolist() == [101, 202]
    assert _is_member(panel, 101, "2024-03-03")
    assert not _is_member(panel, 101, "2024-03-04")
    assert not _is_member(panel, 202, "2024-03-03")
    assert _is_member(panel, 202, "2024-03-04")
    # The panel ends on the table's last date, not today.
    assert pd.Timestamp(panel.timestamp.values[-1]) == pd.Timestamp("2024-06-28")
    assert _is_member(panel, 202, "2024-06-28")


def test_a_member_with_no_change_is_a_member_throughout(tmp_path):
    sp500 = [
        sp500_row("2024-03-31", "historical", "CCC"),  # SYNTHETIC
        sp500_row("2024-06-28", "current", "CCC"),  # SYNTHETIC
    ]
    root = _pull(tmp_path / "downloads", PRICES, TICKERS, sp500)
    panel = _membership(tmp_path, root, start_date="2024-01-01").from_raw_data().get_xarray_dataset()
    assert panel["is_member"].sel(symbol=303).values.all()


def test_a_member_that_left_and_rejoined_has_a_gap(tmp_path):
    sp500 = [
        sp500_row("2024-02-01", "removed", "AAA"),  # SYNTHETIC
        sp500_row("2024-04-01", "added", "AAA"),  # SYNTHETIC
        sp500_row("2024-06-28", "current", "AAA"),  # SYNTHETIC
    ]
    root = _pull(tmp_path / "downloads", PRICES, TICKERS, sp500)
    panel = _membership(tmp_path, root, start_date="2024-01-01").from_raw_data().get_xarray_dataset()
    assert _is_member(panel, 101, "2024-01-31")
    assert not _is_member(panel, 101, "2024-03-01")
    assert _is_member(panel, 101, "2024-04-01")


def test_a_member_ticker_without_a_permaticker_is_refused(tmp_path):
    sp500 = [sp500_row("2024-06-28", "current", "ZZZ")]  # SYNTHETIC
    root = _pull(tmp_path / "downloads", PRICES, TICKERS, sp500)
    with pytest.raises(ValueError, match="ZZZ"):
        _membership(tmp_path, root, start_date="2024-01-01").from_raw_data()


# -- the category filter -------------------------------------------------------

MARKET_TICKERS = [
    tickers_row("SEP", 101, "AAA", category=DOMESTIC),  # SYNTHETIC
    tickers_row("SEP", 202, "BBB", category="Domestic Common Stock Primary Class"),  # SYNTHETIC
    tickers_row("SEP", 303, "CCC", category="Domestic Common Stock Secondary Class"),  # SYNTHETIC
    tickers_row("SEP", 404, "ADRX", category="ADR Common Stock"),  # SYNTHETIC
    tickers_row("SEP", 505, "PRFX", category="Domestic Preferred Stock"),  # SYNTHETIC
    tickers_row("SEP", 606, "WRNT", category="Domestic Warrant"),  # SYNTHETIC
]
MARKET_PRICES = [
    sep_row(t, "2024-01-02", 10.0)  # SYNTHETIC
    for t in ("AAA", "BBB", "CCC", "ADRX", "PRFX", "WRNT")
]


def test_the_default_market_universe_is_domestic_common_stock(tmp_path):
    root = _pull(tmp_path / "downloads", MARKET_PRICES, MARKET_TICKERS)
    panel = _stock(tmp_path, root)
    # ADRs, preferreds and warrants are absent; every domestic common class stays.
    assert panel.symbol.values.tolist() == [101, 202, 303]


def test_the_category_filter_can_be_widened_or_turned_off(tmp_path):
    root = _pull(tmp_path / "downloads", MARKET_PRICES, MARKET_TICKERS)
    adr = _stock(tmp_path, root, category_filter=(DOMESTIC, "ADR Common Stock"))
    assert adr.symbol.values.tolist() == [101, 404]
    everything = _stock(tmp_path, root, category_filter=None)
    assert everything.symbol.values.tolist() == [101, 202, 303, 404, 505, 606]


def test_named_permatickers_are_never_filtered(tmp_path):
    root = _pull(tmp_path / "downloads", MARKET_PRICES, MARKET_TICKERS)
    panel = _stock(tmp_path, root, permatickers=(404, 505))
    assert panel.symbol.values.tolist() == [404, 505]


def test_an_sp500_roster_keeps_every_member_whatever_its_category(tmp_path):
    sp500 = [
        # An ADR member (rare, but the roster must keep it) and a domestic one.
        sp500_row("2024-06-28", "current", "ADRX"),  # SYNTHETIC
        sp500_row("2024-06-28", "current", "AAA"),  # SYNTHETIC
        # A member that left before the conversion window starts.
        sp500_row("2023-06-01", "removed", "BBB"),  # SYNTHETIC
    ]
    root = _pull(tmp_path / "downloads", MARKET_PRICES, MARKET_TICKERS, sp500)
    panel = _stock(tmp_path, root, start_date="2024-01-01", roster_universe="sp500")
    # Every permaticker that was a member inside the window, all its bars.
    assert panel.symbol.values.tolist() == [101, 404]


def test_an_unknown_roster_universe_is_refused(tmp_path):
    from quantlab.dataset.config import SharadarDatasetConfig
    from quantlab.dataset.sharadar.stock import SharadarStockDataset

    with pytest.raises(ValueError, match="nasdaq100"):
        SharadarStockDataset(
            SharadarDatasetConfig(
                zarr_file_path=str(tmp_path / "x.zarr"),
                raw_data_dir_path=str(tmp_path),
                roster_universe="nasdaq100",
            )
        )


def test_an_empty_category_filter_is_refused(tmp_path):
    from quantlab.dataset.config import SharadarDatasetConfig
    from quantlab.dataset.sharadar.stock import SharadarStockDataset

    with pytest.raises(ValueError, match="category_filter"):
        SharadarStockDataset(
            SharadarDatasetConfig(
                zarr_file_path=str(tmp_path / "x.zarr"),
                raw_data_dir_path=str(tmp_path),
                category_filter=(),
            )
        )
