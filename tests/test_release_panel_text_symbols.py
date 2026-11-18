"""Release and filing panels keyed by text symbols, through `quantlab.dataset._support.point_in_time`.

Not every vendor keys its securities by an integer: a 13F holding is keyed
by its Composite FIGI and a fund by its series (``<cik>-<series id>``). The
point-in-time rule, the grid and the release and filing panels take such a
text axis as they take an integer one. Every row is `# SYNTHETIC`.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import polars as pl

from quantlab.dataset._support.point_in_time import (
    FilingPanelDataset,
    ReleasePanelDataset,
    ReleaseSpec,
    TradingCalendar,
    as_of,
)
from quantlab.dataset.config import RawDatasetConfig

DAYS = list(pd.bdate_range("2024-01-02", "2024-01-12"))
FIGI_A, FIGI_B = "BBG000AAAAA1", "BBG000BBBBB2"  # SYNTHETIC
SERIES = "0000000001-S000000001"  # SYNTHETIC


def _calendar(days=DAYS, start="2024-01-01", end="2024-12-31") -> TradingCalendar:
    series = pl.Series("timestamp", list(days), dtype=pl.Datetime("ns"))
    return TradingCalendar(series, through=series.max().date(), start=start, end=end)


def _placed(rows, calendar=None) -> pl.DataFrame:
    frame = pl.DataFrame(
        {"symbol": [r[0] for r in rows], "date": [date.fromisoformat(r[1]) for r in rows], "value": [r[2] for r in rows]},
        schema={"symbol": pl.String, "date": pl.Date, "value": pl.Float64},
    )
    return (calendar or _calendar()).place(frame).sort("available")


ROWS = [(FIGI_A, "2024-01-03", 1.0), (SERIES, "2024-01-06", 5.0), (FIGI_B, "2024-01-10", 7.0)]  # SYNTHETIC


def test_the_as_of_rule_shows_a_text_symbols_latest_row_from_its_available_day():
    panel = as_of(_placed(ROWS), _calendar().days, [FIGI_A, SERIES], ReleaseSpec(variables={"value": {}}, stamp=None))

    assert panel["symbol"].values.tolist() == [FIGI_A, SERIES]
    series = panel["value"].sel(symbol=SERIES).to_series().dropna()
    assert series.index.min() == pd.Timestamp("2024-01-08")  # Saturday's row shows on Monday


class _Releases(ReleasePanelDataset):
    config_cls = RawDatasetConfig
    ROWS: list = ROWS
    CALENDAR_DAYS: list = DAYS

    def _trading_calendar(self) -> TradingCalendar:
        return _calendar(self.CALENDAR_DAYS, self.config.start_date, self.config.end_date)

    def _released(self) -> pl.DataFrame:
        return _placed(self.ROWS, self._trading_calendar())

    def _spec(self) -> ReleaseSpec:
        return ReleaseSpec(variables={"value": {}}, stamp=None)

    def _nothing_shown(self) -> str:
        return "no row is shown"


def _releases(path, days=DAYS, base=_Releases):
    cls = type("Releases", (base,), {"CALENDAR_DAYS": days})
    return cls(RawDatasetConfig(zarr_file_path=str(path / "releases.zarr"), start_date="2024-01-02"))


def test_a_release_panel_of_text_symbols_is_built_and_widened_by_an_update(tmp_path):
    _releases(tmp_path / "updated", DAYS[:5]).update()
    _releases(tmp_path / "updated").update()
    updated = _releases(tmp_path / "updated").panel("2024-01-01", "2024-12-31").load()
    _releases(tmp_path / "rebuilt").update()
    rebuilt = _releases(tmp_path / "rebuilt").panel("2024-01-01", "2024-12-31").load()

    assert sorted(updated["symbol"].values.tolist()) == sorted([FIGI_A, FIGI_B, SERIES])
    np.testing.assert_array_equal(
        updated["value"].sel(symbol=rebuilt["symbol"].values).values, rebuilt["value"].values
    )
    assert updated["value"].sel(symbol=FIGI_B).to_series().dropna().index.min() == pd.Timestamp("2024-01-10")


class _Filings(_Releases, FilingPanelDataset):
    def _released(self) -> pl.DataFrame:
        return _placed(self.ROWS, self._trading_calendar()).select("symbol", "available", "value")

    def _empty(self) -> dict[str, object]:
        return {"value": 0.0}


def test_a_filing_panel_of_text_symbols_shows_a_filing_on_its_own_day_only(tmp_path):
    _releases(tmp_path, base=_Filings).update()
    panel = _releases(tmp_path, base=_Filings).panel("2024-01-01", "2024-12-31").load()

    shown = panel["value"].sel(symbol=SERIES).to_series()
    assert shown[shown != 0.0].to_dict() == {pd.Timestamp("2024-01-08"): 5.0}
