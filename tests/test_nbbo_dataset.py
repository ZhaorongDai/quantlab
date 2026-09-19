"""WRDS TAQ `complete_nbbo` -> raw tick shards -> 1m NBBO Zarr panel (phase 03.9).

Offline, all of it: `FakeWrdsSession` serves the rows, the autouse
`_forbid_wrds_network` tripwire in `tests/conftest.py` makes any real
connection attempt fail (D-28), and every path is under `tmp_path`.
"""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

from tests.wrds_fixtures import taq_row

DAY = date(2024, 1, 24)

#: AAPL on 2024-01-24 (EST, UTC-5), in PHYSICAL order. r0 is the real L6 row.
TRACER_ROWS = [
    # r0: pre-market, one-sided (ask NULL) -- the real LIVE-CHECK-1 L6 row.
    taq_row("04:00:00.005984", "180", "100", None, None, nano=226),
    # r1: the seed -- the last record at or before the 09:30 open.
    taq_row("09:29:59.500000", 194.00, 200, 194.02, 300, nano=0),
    taq_row("09:30:30.000000", 194.01, 100, 194.04, 100, nano=0),  # r2
    # r3: exactly on the 09:31 edge -> belongs to the bar labelled 09:31.
    taq_row("09:31:00.000000", 194.02, 100, 194.03, 200, nano=0),
    taq_row("09:32:15.000000", 194.00, 400, 194.04, 100, nano=500),  # r4
    # r5: exactly on the close -> belongs to the bar labelled 16:00.
    taq_row("16:00:00.000000", 193.90, 100, 193.95, 100, nano=0),
    # r6: after-hours -> kept in raw, contributes to no bar.
    taq_row("16:05:00.000000", 193.80, 100, 193.85, 100, nano=0),
]


def _approx(value):
    return pytest.approx(value, abs=1e-9)


def test_tracer_one_wrds_symbol_day_lands_raw_and_resamples_to_a_zarr_panel(
    mock_wrds_session, tmp_path
):
    import quantlab.config as config
    from quantlab.acquisition import registry
    from quantlab.acquisition.wrds_taq import WRDS_SOURCE, WrdsTaqNbboAcquisition
    from quantlab.base.config import NbboDatasetConfig
    from quantlab.dataset.cleaning import NBBO_PANEL_VARIABLES
    from quantlab.dataset.nbbo import NbboPanelDataset

    mock_wrds_session.trading_days_by_year = {2024: [DAY]}
    mock_wrds_session.rows = {(DAY, "AAPL"): list(TRACER_ROWS)}

    config.set_data_root(tmp_path)
    cfg = WrdsTaqNbboAcquisition.build_config(
        ("AAPL",), start_date="2024-01-24", end_date="2024-01-24"
    )
    registry.run(WRDS_SOURCE, cfg)

    # -- raw: every record of the day, one row each, in arrival order -------
    raw_root = Path(cfg.raw_data_dir_path)
    assert raw_root.name == "wrds"
    shard_dirs = sorted({path.parent for path in raw_root.rglob("*.pqt")})
    assert shard_dirs == [
        raw_root / "data_type=nbbo" / "date=2024-01-24" / "symbol=AAPL"
    ]
    raw = pl.concat(
        [pl.read_parquet(path) for path in sorted(shard_dirs[0].glob("*.pqt"))]
    )
    assert raw.height == 7
    assert raw["wrds_row_ord"].to_list() == list(range(7))
    # Via pandas: a polars scalar comes back as a microsecond `datetime`.
    assert raw["timestamp"].to_pandas().iloc[4] == pd.Timestamp(
        "2024-01-24T14:32:15.000000500"
    )
    assert raw["timestamp"].dtype == pl.Datetime("ns")
    assert raw["best_ask"][0] is None
    assert set(raw.columns) == set(WrdsTaqNbboAcquisition.RAW_COLUMNS) - {"symbol"}
    assert (
        Path(cfg.watermark_path) / "nbbo" / "AAPL.json"
    ).exists(), "no watermark under _watermarks/wrds/nbbo/"
    assert Path(cfg.watermark_path).parts[-2:] == ("_watermarks", "wrds")

    # -- conversion through the registry -----------------------------------
    dataset_config = NbboDatasetConfig(
        zarr_file_path=str(tmp_path / "nbbo_1m.zarr"),
        raw_data_dir_path=cfg.raw_data_dir_path,
        catalog_path=str(tmp_path / "catalog"),
        start_date="2024-01-24",
        end_date="2024-01-24",
        symbols=("AAPL",),
        bar_interval="1m",
    )
    registry.convert(
        WRDS_SOURCE, dataset_config, data_type="nbbo", granularity="day"
    )

    panel = (
        NbboPanelDataset(
            NbboDatasetConfig(
                zarr_file_path=str(tmp_path / "nbbo_1m.zarr"),
                raw_data_dir_path=cfg.raw_data_dir_path,
                catalog_path=str(tmp_path / "catalog"),
                start_date="2024-01-24",
                end_date="2024-01-24",
                symbols=("AAPL",),
                bar_interval="1m",
            )
        )
        .read()
        .get_xarray_dataset()
    )

    timestamps = pd.DatetimeIndex(panel["timestamp"].values)
    assert len(timestamps) == 390
    assert timestamps[0] == pd.Timestamp("2024-01-24T14:31:00")
    assert timestamps[-1] == pd.Timestamp("2024-01-24T21:00:00")
    assert [str(s) for s in panel["symbol"].values] == ["AAPL"]
    assert set(panel.data_vars) == set(NBBO_PANEL_VARIABLES)
    for name in NBBO_PANEL_VARIABLES:
        assert panel[name].dtype == np.float64, name

    def bar(label: str) -> dict[str, float]:
        row = panel.sel(timestamp=pd.Timestamp(label), symbol="AAPL")
        return {name: float(row[name].values) for name in NBBO_PANEL_VARIABLES}

    # 09:31 ET: r2 and r3; snapshot r3 (on the edge); seed r1 for 30 s then r2.
    b = bar("2024-01-24T14:31:00")
    assert b["n_updates"] == 2
    assert b["bid"] == _approx(194.02)
    assert b["ask"] == _approx(194.03)
    assert b["bid_size"] == 100
    assert b["ask_size"] == 200
    assert b["mid"] == _approx(194.025)
    assert b["spread"] == _approx(0.01)
    assert b["spread_bps"] == _approx(1e4 * 0.01 / 194.025)
    assert b["imbalance"] == _approx(-1 / 3)
    assert b["tw_spread"] == _approx(0.025)
    assert b["tw_bid_size"] == _approx(150)
    assert b["tw_ask_size"] == _approx(200)
    assert b["n_ambiguous_ties"] == 0

    # 09:32 ET: no update, r3 carried forward.
    b = bar("2024-01-24T14:32:00")
    assert b["n_updates"] == 0
    assert b["bid"] == _approx(194.02)
    assert b["tw_spread"] == _approx(0.01)

    # 09:33 ET: r4 at 09:32:15.000000500.
    b = bar("2024-01-24T14:33:00")
    before = 15.0000005
    after = 60.0 - before
    assert b["n_updates"] == 1
    assert b["spread"] == _approx(0.04)
    assert b["imbalance"] == _approx(0.6)
    assert b["tw_spread"] == _approx((0.01 * before + 0.04 * after) / 60.0)

    # 16:00 ET: r5 exactly on the close; r6 (16:05) contributes to no bar.
    b = bar("2024-01-24T21:00:00")
    assert b["n_updates"] == 1
    assert b["bid"] == _approx(193.90)
    assert b["ask"] == _approx(193.95)


# ---------------------------------------------------------------------------
# Plan 03.9-06: XNYS session edges, windows and the filter policy.
# ---------------------------------------------------------------------------


def _acquire(tmp_path, rows: dict, *, symbols=("AAPL",)):
    """Land `rows` (`{(day, symbol): [taq_row, ...]}`) in raw through the real
    acquisition path and return the acquisition config."""
    import quantlab.config as config
    from quantlab.acquisition import registry
    from quantlab.acquisition.wrds_taq import WRDS_SOURCE, WrdsTaqNbboAcquisition
    from tests.wrds_fixtures import FakeWrdsSession

    days = sorted({day for day, _ in rows})
    by_year: dict[int, list[date]] = {}
    for day in days:
        by_year.setdefault(day.year, []).append(day)
    FakeWrdsSession.trading_days_by_year = by_year
    FakeWrdsSession.rows = {key: list(value) for key, value in rows.items()}

    config.set_data_root(tmp_path)
    cfg = WrdsTaqNbboAcquisition.build_config(
        tuple(symbols),
        start_date=days[0].isoformat(),
        end_date=days[-1].isoformat(),
    )
    registry.run(WRDS_SOURCE, cfg)
    return cfg


def _dataset_config(tmp_path, acq_cfg, start, end, *, name="nbbo.zarr", **kwargs):
    from quantlab.base.config import NbboDatasetConfig

    symbols = kwargs.pop("symbols", ("AAPL",))
    return NbboDatasetConfig(
        zarr_file_path=str(tmp_path / name),
        raw_data_dir_path=acq_cfg.raw_data_dir_path,
        catalog_path=str(tmp_path / "catalog"),
        start_date=start,
        end_date=end,
        symbols=symbols,
        **kwargs,
    )


def _panel(dataset_config):
    from quantlab.dataset.nbbo import NbboPanelDataset

    return NbboPanelDataset(dataset_config).from_raw_data().get_xarray_dataset()


def _bar(panel, label: str, symbol: str = "AAPL") -> dict[str, float]:
    from quantlab.dataset.cleaning import NBBO_PANEL_VARIABLES

    row = panel.sel(timestamp=pd.Timestamp(label), symbol=symbol)
    return {name: float(row[name].values) for name in NBBO_PANEL_VARIABLES}


HALF_DAY = date(2024, 11, 29)
HALF_DAY_ROWS = [
    taq_row("09:00:00.000000", 230.00, 100, 230.10, 100, nano=0, day="2024-11-29"),
    taq_row("12:59:30.000000", 231.00, 200, 231.20, 300, nano=0, day="2024-11-29"),
    # After the 13:00 early close: post-close state.
    taq_row("14:00:00.000000", 232.00, 100, 232.50, 100, nano=0, day="2024-11-29"),
]


def test_half_day_rth_panel_ends_at_the_early_close(mock_wrds_session, tmp_path):
    acq = _acquire(tmp_path, {(HALF_DAY, "AAPL"): HALF_DAY_ROWS})
    panel = _panel(_dataset_config(tmp_path, acq, "2024-11-29", "2024-11-29"))

    timestamps = pd.DatetimeIndex(panel["timestamp"].values)
    assert len(timestamps) == 210
    assert timestamps[0] == pd.Timestamp("2024-11-29T14:31:00")
    assert timestamps[-1] == pd.Timestamp("2024-11-29T18:00:00")

    last = _bar(panel, "2024-11-29T18:00:00")
    assert last["n_updates"] == 1
    assert last["bid"] == _approx(231.00)
    assert last["ask"] == _approx(231.20)
    # The 14:00 ET record changes no bar: no bar anywhere carries its prices.
    assert not np.any(panel["bid"].values == 232.00)


def test_dst_day_rth_panel_starts_at_13_31_utc(mock_wrds_session, tmp_path):
    day = date(2024, 3, 11)
    acq = _acquire(
        tmp_path,
        {
            (day, "AAPL"): [
                taq_row("09:29:00.000000", 170.0, 100, 170.1, 100, nano=0, day="2024-03-11"),
                taq_row("10:00:00.000000", 170.2, 100, 170.3, 100, nano=0, day="2024-03-11"),
            ]
        },
    )
    panel = _panel(_dataset_config(tmp_path, acq, "2024-03-11", "2024-03-11"))
    timestamps = pd.DatetimeIndex(panel["timestamp"].values)
    assert len(timestamps) == 390
    assert timestamps[0] == pd.Timestamp("2024-03-11T13:31:00")
    assert timestamps[-1] == pd.Timestamp("2024-03-11T20:00:00")


def test_non_session_raw_date_fails_the_conversion_naming_it(
    mock_wrds_session, tmp_path
):
    from quantlab.dataset.nbbo import NbboPanelDataset

    thanksgiving = date(2024, 11, 28)
    acq = _acquire(
        tmp_path,
        {
            (thanksgiving, "AAPL"): [
                taq_row("10:00:00.000000", 230.0, 100, 230.1, 100, nano=0, day="2024-11-28"),
            ],
            (HALF_DAY, "AAPL"): HALF_DAY_ROWS,
        },
    )
    cfg = _dataset_config(tmp_path, acq, "2024-11-28", "2024-11-29")
    with pytest.raises(ValueError, match="2024-11-28"):
        NbboPanelDataset(cfg).from_raw_data_chunked(granularity="day")
    assert not Path(cfg.zarr_file_path).exists()


def test_session_window_inside_rth_and_out_of_range_edges(
    mock_wrds_session, tmp_path
):
    from quantlab.dataset.nbbo import NbboPanelDataset

    acq = _acquire(tmp_path, {(DAY, "AAPL"): list(TRACER_ROWS)})
    panel = _panel(
        _dataset_config(
            tmp_path, acq, "2024-01-24", "2024-01-24",
            session_start="10:00", session_end="15:30",
        )
    )
    timestamps = pd.DatetimeIndex(panel["timestamp"].values)
    assert len(timestamps) == 330
    assert timestamps[0] == pd.Timestamp("2024-01-24T15:01:00")
    assert timestamps[-1] == pd.Timestamp("2024-01-24T20:30:00")

    for edges in ({"session_start": "03:59"}, {"session_end": "20:01"}):
        with pytest.raises(ValueError):
            NbboPanelDataset(
                _dataset_config(tmp_path, acq, "2024-01-24", "2024-01-24", **edges)
            )


def test_extended_window_covers_pre_and_after_hours(mock_wrds_session, tmp_path):
    rows = [
        # The real L6 one-sided row (ask NULL).
        taq_row("04:00:00.005984", "180", "100", None, None, nano=226),
        taq_row("07:15:00.000000", 190.00, 100, 190.10, 100, nano=0),
        taq_row("17:30:00.000000", 195.00, 100, 195.20, 100, nano=0),
    ]
    acq = _acquire(tmp_path, {(DAY, "AAPL"): rows})
    panel = _panel(
        _dataset_config(
            tmp_path, acq, "2024-01-24", "2024-01-24",
            session_start="04:00", session_end="20:00",
        )
    )
    timestamps = pd.DatetimeIndex(panel["timestamp"].values)
    assert len(timestamps) == 960
    assert timestamps[0] == pd.Timestamp("2024-01-24T09:01:00")
    assert timestamps[-1] == pd.Timestamp("2024-01-25T01:00:00")

    first = _bar(panel, "2024-01-24T09:01:00")
    assert first["bid"] == _approx(180)
    assert np.isnan(first["ask"])
    assert np.isnan(first["spread"])
    assert first["n_updates"] == 1

    pre = _bar(panel, "2024-01-24T12:15:00")
    assert pre["n_updates"] == 1
    assert pre["bid"] == _approx(190.00)
    carried = _bar(panel, "2024-01-24T14:31:00")
    assert carried["n_updates"] == 0
    assert carried["bid"] == _approx(190.00)

    post = _bar(panel, "2024-01-24T22:30:00")
    assert post["n_updates"] == 1
    assert post["bid"] == _approx(195.00)
    end = _bar(panel, "2024-01-25T01:00:00")
    assert end["n_updates"] == 0
    assert end["bid"] == _approx(195.00)


def test_extended_window_is_seeded_from_before_its_start(
    mock_wrds_session, tmp_path
):
    rows = [
        taq_row("03:59:59.000000", 185.00, 100, 185.10, 100, nano=0),
        taq_row("05:00:30.000000", 186.00, 100, 186.10, 100, nano=0),
    ]
    acq = _acquire(tmp_path, {(DAY, "AAPL"): rows})
    panel = _panel(
        _dataset_config(
            tmp_path, acq, "2024-01-24", "2024-01-24",
            session_start="04:00", session_end="20:00",
        )
    )
    seeded = panel.sel(
        timestamp=slice(
            pd.Timestamp("2024-01-24T09:01"), pd.Timestamp("2024-01-24T10:00")
        )
    )
    assert seeded.sizes["timestamp"] == 60
    assert np.allclose(seeded["bid"].values, 185.00)
    assert np.all(seeded["n_updates"].values == 0)
    assert _bar(panel, "2024-01-24T10:01:00")["bid"] == _approx(186.00)


def test_extended_window_on_a_half_day_is_not_truncated(
    mock_wrds_session, tmp_path
):
    acq = _acquire(tmp_path, {(HALF_DAY, "AAPL"): HALF_DAY_ROWS})
    extended = _panel(
        _dataset_config(
            tmp_path, acq, "2024-11-29", "2024-11-29",
            session_start="04:00", session_end="20:00",
        )
    )
    timestamps = pd.DatetimeIndex(extended["timestamp"].values)
    assert len(timestamps) == 960
    assert timestamps[-1] == pd.Timestamp("2024-11-30T01:00:00")
    post = _bar(extended, "2024-11-29T19:00:00")
    assert post["n_updates"] == 1
    assert post["bid"] == _approx(232.00)

    rth = _panel(_dataset_config(tmp_path, acq, "2024-11-29", "2024-11-29"))
    rth_ts = pd.DatetimeIndex(rth["timestamp"].values)
    assert len(rth_ts) == 210
    assert rth_ts[-1] == pd.Timestamp("2024-11-29T18:00:00")
    assert not np.any(rth["bid"].values == 232.00)


def test_filter_policy_from_config_reaches_the_resampler(
    mock_wrds_session, tmp_path
):
    rows = [
        taq_row("09:29:00.000000", 194.00, 100, 194.02, 100, nano=0),
        # Crossed: bid > ask.
        taq_row("09:30:30.000000", 194.10, 100, 194.05, 100, nano=0),
    ]
    acq = _acquire(tmp_path, {(DAY, "AAPL"): rows})

    kept = _panel(
        _dataset_config(tmp_path, acq, "2024-01-24", "2024-01-24", drop_crossed=False)
    )
    b = _bar(kept, "2024-01-24T14:31:00")
    assert b["bid"] == _approx(194.10)
    assert b["ask"] == _approx(194.05)
    assert b["n_updates"] == 1

    dropped = _panel(_dataset_config(tmp_path, acq, "2024-01-24", "2024-01-24"))
    b = _bar(dropped, "2024-01-24T14:31:00")
    assert b["bid"] == _approx(194.00)
    assert b["ask"] == _approx(194.02)
    assert b["n_updates"] == 0
