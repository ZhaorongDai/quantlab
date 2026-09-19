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
