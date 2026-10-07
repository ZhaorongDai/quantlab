"""Leaf tests of the report's Holdings tab (#225).

``write_backtest_report`` gets the holdings and the names to show them by as
plain data; ``report_holdings_inputs`` builds those names from a label
function. These tests read the JSON the tab embeds: the targets of the last
rebalance before each bar, the holdings, the cash, dust folded into "Other",
and the ticker in use on each bar, the symbol id when none is known. Without
holdings the tab and its data are absent. Synthetic, offline, milliseconds.
"""

import json
import re

import numpy as np
import pandas as pd
import xarray as xr

from quantlab.runs.backtest_report import (
    DUST_THRESHOLD,
    report_holdings_inputs,
    write_backtest_report,
)

BARS = pd.bdate_range("2024-01-01", periods=5)
SYMBOLS = ["10001", "10002", "10003"]

_DATA = re.compile(r'<script type="application/json" id="holdings-data">(.*?)</script>', re.S)


def _value() -> xr.DataArray:
    return xr.DataArray(
        [100.0, 101.0, 99.0, 102.0, 103.0], dims=("timestamp",), coords={"timestamp": BARS}
    )


def _panel(rows) -> xr.DataArray:
    return xr.DataArray(
        np.asarray(rows, dtype=float),
        dims=("timestamp", "symbol"),
        coords={"timestamp": BARS, "symbol": SYMBOLS},
    )


#: Rebalance on bars 0 and 2; bar 2 keeps 10003 (NaN) at its last target.
WEIGHTS = _panel([
    [0.6, 0.4 - DUST_THRESHOLD / 2, DUST_THRESHOLD / 2],
    [np.nan] * 3,
    [0.0, 0.5, np.nan],
    [np.nan] * 3,
    [np.nan] * 3,
])
HOLDINGS = _panel([
    [0.0, 0.0, 0.0],
    [0.59, 0.39, 0.00004],
    [0.61, 0.38, 0.00005],
    [0.0, 0.49, 0.00005],
    [0.0, 0.52, 0.00006],
])


def _page(tmp_path, **kwargs) -> str:
    path = tmp_path / "report.html"
    write_backtest_report(
        _value(), path, in_sample_range=None, notes=[], title="run",
        metrics={"whole": {"Total Return [%]": 3.0, "Annualized Return [%]": 12.5,
                           "Max Drawdown [%]": 1.98}},
        weights=WEIGHTS, **kwargs,
    )
    return path.read_text(encoding="utf-8")


def _data(page: str) -> dict:
    return json.loads(_DATA.search(page).group(1))


def test_without_holdings_the_page_has_no_holdings_tab(tmp_path):
    page = _page(tmp_path)
    assert ">Holdings</button>" not in page
    assert "holdings-data" not in page


def test_each_bar_shows_the_targets_of_the_last_rebalance_before_it(tmp_path):
    data = _data(_page(tmp_path, holdings=HOLDINGS))
    days = data["days"]
    assert [day["d"] for day in days] == [str(bar.date()) for bar in BARS]
    assert [day["r"] for day in days] == [None, "2024-01-01", "2024-01-01", "2024-01-03", "2024-01-03"]

    def rows(day):
        return {data["names"][k][2]: (t, h) for k, t, h in day["h"]}

    assert rows(days[0]) == {}
    assert rows(days[1]) == {"10001": (0.6, 0.59), "10002": (0.4 - DUST_THRESHOLD / 2, 0.39)}
    # Bar 4 trades 10001 out: a target of 0 and nothing held drops it.
    assert rows(days[3]) == {"10002": (0.5, 0.49)}
    assert days[1]["cash"] == 1.0 - HOLDINGS.values[1].sum()


def test_dust_targets_are_folded_into_other(tmp_path):
    days = _data(_page(tmp_path, holdings=HOLDINGS))["days"]
    # 10003's target of half the threshold is dust on every bar it is held,
    # kept at its last target by the NaN of the second rebalance.
    assert days[1]["other"] == [1, DUST_THRESHOLD / 2, 0.00004]
    assert days[4]["other"] == [1, DUST_THRESHOLD / 2, 0.00006]
    assert days[0]["other"] == [0, 0, 0]


def test_the_summary_figures_are_the_headline_metrics(tmp_path):
    summary = dict(_data(_page(tmp_path, holdings=HOLDINGS))["summary"])
    assert summary["Total return"] == "3.00%"
    assert summary["Annualised return"] == "12.50%"
    assert summary["Max drawdown"] == "-1.98%"


def test_symbols_without_names_are_labelled_by_their_id(tmp_path):
    names = _data(_page(tmp_path, holdings=HOLDINGS))["names"]
    assert sorted(names) == [["10001", "", "10001"], ["10002", "", "10002"]]


def test_a_renamed_symbol_shows_the_ticker_in_use_on_each_bar(tmp_path):
    def label(symbols, day):
        return [
            ("NEW", "New Co") if s == "10002" and str(day) >= "2024-01-03" else (f"T{s}", None)
            for s in symbols
        ]

    inputs = report_holdings_inputs(HOLDINGS, WEIGHTS, label=label)
    assert inputs["holding_names"]["10002"] == [
        ("2024-01-02", "T10002", ""), ("2024-01-03", "NEW", "New Co"),
    ]
    data = _data(_page(tmp_path, **inputs))

    def tickers(day):
        return {data["names"][k][2]: data["names"][k][0] for k, _, _ in day["h"]}

    assert tickers(data["days"][1])["10002"] == "T10002"
    assert tickers(data["days"][2])["10002"] == "NEW"


def test_report_holdings_inputs_without_a_label_names_nothing():
    inputs = report_holdings_inputs(HOLDINGS, WEIGHTS)
    assert inputs["holding_names"] is None
    assert inputs["holdings"] is HOLDINGS


def test_a_name_cannot_close_the_data_script(tmp_path):
    names = {"10001": [("2024-01-01", "</script><b>x", "A & B")]}
    page = _page(tmp_path, holdings=HOLDINGS, holding_names=names)
    data = _data(page)
    assert ["</script><b>x", "A & B", "10001"] in data["names"]
