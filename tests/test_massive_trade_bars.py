"""The Massive Trade bar dataset (#238): raw trade files to a store on the permaticker axis.

Seam 1 of spec #237: fixture raw files on disk (trade ``csv.gz`` days, a
condition table, Sharadar TICKERS and ACTIONS) converted into a store, and
assertions on the panel. What is locked here:

- the session is XNYS's regular session, half days included (2024-11-29
  closes at 13:00 ET), bars right-closed and labelled at their end, a
  trade after the early close outside every bar;
- each raw ``(date, ticker)`` maps to its permaticker as traded that day:
  a renamed security is one column, a reused ticker is two, a fund maps
  through SFP, and a ticker that maps to nothing is counted and dropped;
- a cell without an eligible trade has NaN prices and zero volume and
  ``n_trades``, a permaticker without a trade that day too;
- the condition rules and the correction filter reach the store, and the
  per-day statistics land in the sidecar.

Every value is SYNTHETIC.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

from tests.massive_fixtures import trade_line, write_conditions, write_trades
from tests.sharadar_fixtures import action_row, tickers_row

FULL_DAY = date(2024, 11, 27)  # 09:30-16:00 ET = 14:30-21:00 UTC
HALF_DAY = date(2024, 11, 29)  # 09:30-13:00 ET = 14:30-18:00 UTC


def _write_sharadar(root: Path) -> None:
    """TICKERS and ACTIONS: 101 AAA; 202 OLD -> NEW on 11-29; 303 RRR -> RRS on 11-28, 404 RRR from 11-28; fund 505 FFF."""
    from quantlab.dataset.sharadar.tables import TABLES, bulk_file, write_bulk_pull

    def write(code: str, rows: list[dict]) -> None:
        schema = TABLES[code].schema
        frame = pl.DataFrame(
            {
                name: [
                    date.fromisoformat(r[name]) if dtype == pl.Date and r.get(name) else r.get(name) for r in rows
                ]
                for name, dtype in schema.items()
            },
            schema=schema,
        )
        path = bulk_file(root, code)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(path)
        write_bulk_pull(root, code, datetime(2024, 12, 2, 20, tzinfo=UTC))

    def change(day: str, new: str, old: str) -> dict:
        row = action_row(day, "tickerchangefrom", new, None)
        row.update(contraticker=old, contraname=f"{old} CORP")  # SYNTHETIC
        return row

    write(
        "tickers",
        [
            tickers_row("SEP", 101, "AAA"),  # SYNTHETIC
            tickers_row("SEP", 202, "NEW"),  # SYNTHETIC
            tickers_row("SEP", 303, "RRS"),  # SYNTHETIC
            tickers_row("SEP", 404, "RRR", firstpricedate="2024-11-28"),  # SYNTHETIC
            tickers_row("SFP", 505, "FFF", category="ETF"),  # SYNTHETIC
        ],
    )
    write("actions", [change("2024-11-29", "NEW", "OLD"), change("2024-11-28", "RRS", "RRR")])  # SYNTHETIC


def _utc(day: date, clock: str) -> datetime:
    return datetime.fromisoformat(f"{day.isoformat()}T{clock}")


def _raw(tmp_path: Path) -> tuple[Path, Path]:
    massive = tmp_path / "downloads" / "massive"
    sharadar = tmp_path / "downloads" / "sharadar"
    _write_sharadar(sharadar)
    write_conditions(massive, datetime(2024, 12, 2, 12, tzinfo=UTC))
    write_trades(
        massive,
        FULL_DAY,
        [
            trade_line("AAA", _utc(FULL_DAY, "14:30:30"), 10.0, 100, sequence=1),  # SYNTHETIC
            trade_line("OLD", _utc(FULL_DAY, "14:31:30"), 20.0, 200, sequence=2),  # SYNTHETIC
            trade_line("RRR", _utc(FULL_DAY, "14:31:40"), 30.0, 300, sequence=3),  # SYNTHETIC
            trade_line("ZZZ", _utc(FULL_DAY, "14:31:50"), 1.0, 1, sequence=4),  # SYNTHETIC
            trade_line("AAA", _utc(FULL_DAY, "21:00:00"), 11.0, 100, sequence=5),  # SYNTHETIC
        ],
    )
    write_trades(
        massive,
        HALF_DAY,
        [
            trade_line("AAA", _utc(HALF_DAY, "14:30:10"), 10.0, 100, conditions="14,41", sequence=1),  # SYNTHETIC
            trade_line("AAA", _utc(HALF_DAY, "14:30:20"), 15.0, 7, conditions="37", sequence=2),  # SYNTHETIC
            trade_line("AAA", _utc(HALF_DAY, "14:30:30"), 12.0, 50, correction=1, sequence=3),  # SYNTHETIC
            trade_line("AAA", _utc(HALF_DAY, "14:31:00"), 11.0, 20, sequence=4),  # SYNTHETIC
            trade_line("AAA", _utc(HALF_DAY, "17:59:30"), 13.0, 30, sequence=5),  # SYNTHETIC
            trade_line("AAA", _utc(HALF_DAY, "18:30:00"), 14.0, 40, sequence=6),  # SYNTHETIC
            trade_line("FFF", _utc(HALF_DAY, "14:35:00"), 50.0, 5, sequence=7),  # SYNTHETIC
            trade_line("NEW", _utc(HALF_DAY, "14:31:30"), 21.0, 210, sequence=8),  # SYNTHETIC
            trade_line("RRR", _utc(HALF_DAY, "14:31:40"), 40.0, 400, sequence=9),  # SYNTHETIC
            trade_line("RRS", _utc(HALF_DAY, "14:31:50"), 31.0, 310, sequence=10),  # SYNTHETIC
            trade_line("ZZZ", _utc(HALF_DAY, "14:31:50"), 1.0, 1, sequence=11),  # SYNTHETIC
            trade_line("ZZZ", _utc(HALF_DAY, "14:32:50"), 1.0, 1, sequence=12),  # SYNTHETIC
        ],
    )
    return massive, sharadar


def _dataset(tmp_path: Path, start=FULL_DAY, end=HALF_DAY, **overrides):
    from quantlab.dataset.config import MassiveTradeBarsDatasetConfig
    from quantlab.dataset.massive.trade_bars import MassiveTradeBarDataset

    massive, sharadar = _raw(tmp_path)
    config = MassiveTradeBarsDatasetConfig(
        zarr_file_path=str(tmp_path / "zarrs" / "massive_trade_bars_1m.zarr"),
        raw_data_dir_path=str(massive),
        sharadar_dir=str(sharadar),
        start_date=start.isoformat(),
        end_date=end.isoformat(),
        **overrides,
    )
    return MassiveTradeBarDataset(config)


@pytest.fixture
def converted(tmp_path):
    ds = _dataset(tmp_path)
    ds.from_raw_data_chunked(granularity="day")
    return ds, ds.panel("2024-11-27", "2024-11-30")


def _cell(panel, symbol: int, moment: datetime) -> dict:
    return {name: float(panel[name].sel(timestamp=moment, symbol=symbol)) for name in panel.data_vars}


def test_the_store_is_on_the_permaticker_axis_with_the_trade_bar_variables(converted):
    _, panel = converted
    assert panel.symbol.values.tolist() == [101, 202, 303, 404, 505]
    assert panel.symbol.dtype == np.int64
    assert set(panel.data_vars) == {"open", "high", "low", "close", "volume", "n_trades"}
    assert all(panel[name].dtype == np.float64 for name in panel.data_vars)


def test_the_regular_session_on_xnys_with_its_half_day(converted):
    _, panel = converted
    labels = pd.DatetimeIndex(panel.timestamp.values)
    full = labels[labels.normalize() == pd.Timestamp(FULL_DAY)]
    half = labels[labels.normalize() == pd.Timestamp(HALF_DAY)]
    assert (len(full), full[0], full[-1]) == (390, pd.Timestamp("2024-11-27 14:31"), pd.Timestamp("2024-11-27 21:00"))
    assert (len(half), half[0], half[-1]) == (210, pd.Timestamp("2024-11-29 14:31"), pd.Timestamp("2024-11-29 18:00"))


def test_a_trade_at_the_close_is_in_the_last_bar_and_one_after_the_early_close_in_none(converted):
    _, panel = converted
    assert _cell(panel, 101, _utc(FULL_DAY, "21:00"))["close"] == 11.0
    assert _cell(panel, 101, _utc(HALF_DAY, "18:00"))["close"] == 13.0
    assert float(panel["volume"].sel(symbol=101).sel(timestamp=slice("2024-11-29", "2024-11-30")).sum()) == 100 + 7 + 20 + 30


def test_conditions_and_corrections_reach_the_store(converted):
    _, panel = converted
    first = _cell(panel, 101, _utc(HALF_DAY, "14:31"))
    # 10.0 counts for all; the odd lot (15.0) for volume only; the corrected 12.0 not at all;
    # 11.0 exactly on the boundary is in this bar.
    assert (first["open"], first["high"], first["low"], first["close"]) == (10.0, 11.0, 10.0, 11.0)
    assert (first["volume"], first["n_trades"]) == (127.0, 3.0)


def test_an_empty_bar_has_nan_prices_and_zero_volume(converted):
    _, panel = converted
    empty = _cell(panel, 101, _utc(HALF_DAY, "14:40"))
    assert all(np.isnan(empty[k]) for k in ("open", "high", "low", "close"))
    assert (empty["volume"], empty["n_trades"]) == (0.0, 0.0)
    # The fund traded only on the half day: zero volume all of the full day.
    full_day = panel.sel(symbol=505, timestamp=slice("2024-11-27", "2024-11-27T23:59"))
    assert float(full_day["volume"].sum()) == 0.0 and bool(np.isnan(full_day["close"]).all())


def test_a_renamed_security_is_one_column(converted):
    _, panel = converted
    assert _cell(panel, 202, _utc(FULL_DAY, "14:32"))["close"] == 20.0  # as OLD
    assert _cell(panel, 202, _utc(HALF_DAY, "14:32"))["close"] == 21.0  # as NEW


def test_a_reused_ticker_maps_to_its_holder_of_the_day(converted):
    _, panel = converted
    assert _cell(panel, 303, _utc(FULL_DAY, "14:32"))["close"] == 30.0  # RRR, then 303's
    assert np.isnan(_cell(panel, 404, _utc(FULL_DAY, "14:32"))["close"])
    assert _cell(panel, 404, _utc(HALF_DAY, "14:32"))["close"] == 40.0  # RRR, now 404's
    assert _cell(panel, 303, _utc(HALF_DAY, "14:32"))["close"] == 31.0  # as RRS


def test_a_fund_maps_through_sfp(converted):
    _, panel = converted
    assert _cell(panel, 505, _utc(HALF_DAY, "14:35"))["close"] == 50.0


def test_an_unmapped_ticker_is_counted_and_dropped(converted):
    import json

    ds, panel = converted
    assert 0 not in panel.symbol.values
    stats = json.loads(ds.stats_path.read_text())
    half = stats["days"]["2024-11-29"]
    assert half["unmapped"] == {"ZZZ": {"trades": 2, "reason": "no permaticker"}}
    assert (half["unmapped_tickers"], half["unmapped_trades"]) == (1, 2)
    assert (half["trades_in"], half["dropped_correction"], half["outside_session"]) == (12, 1, 1)
    assert stats["days"]["2024-11-27"]["unmapped"]["ZZZ"]["trades"] == 1
    assert stats["settings"] == {"bar_interval": "1m", "session_start": "09:30", "session_end": "16:00"}


def test_a_one_day_store(tmp_path):
    ds = _dataset(tmp_path, start=HALF_DAY, end=HALF_DAY)
    ds.from_raw_data_chunked(granularity="day")
    panel = ds.panel("2024-11-29", "2024-11-30")
    assert panel.sizes == {"timestamp": 210, "symbol": 5}
    assert Path(ds.config.zarr_file_path).name == "massive_trade_bars_1m.zarr"


def test_symbols_are_refused(tmp_path):
    with pytest.raises(ValueError, match="permaticker"):
        _dataset(tmp_path, symbols=("AAA",))
