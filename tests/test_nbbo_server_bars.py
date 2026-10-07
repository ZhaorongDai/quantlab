"""NBBO bars resampled on the WRDS server (ADR 0027, #219): download, raw tier, panel.

Offline, all of it. `FakeWrdsSession.copy_nbbo_bars_csv` answers a bar
statement with what the server is meant to return, computed by the reference
`NbboResampler` from the same stored records the tick download reads; the
SQL itself is accepted against `NbboResampler` on WRDS (#221). The tests here
therefore lock the client side: the acquisition, the raw tier and the
conversion, against the tick path on the same records.
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from conftest import WHOLE_STORE
from loguru import logger

from tests.wrds_fixtures import fake_connect, render_composed, taq_row

AAPL = 14593
BRK_B = 83443
MSFT = 10107

DAY = date(2024, 1, 24)
HALF_DAY = date(2024, 11, 29)
#: Before 2018: no `time_m_nano`, so ties at a microsecond follow scan order.
OLD_DAY = date(2016, 3, 1)

#: The snapshot variables and counts the server computes in this slice, plus
#: the four the conversion derives from the snapshot.
COMPARED = (
    "bid", "ask", "bid_size", "ask_size", "mid", "spread", "spread_bps",
    "imbalance", "n_updates",
)
#: Not computed by the server yet: NaN in a server-bar panel.
NOT_YET = ("tw_spread", "tw_bid_size", "tw_ask_size", "n_ambiguous_ties")


def _rows() -> dict:
    """Records over three sessions and three tickers, with every edge the
    resampler handles: seed before the open, records on bar edges and on the
    close, after-close records, ties at one instant, a crossed quote (dropped
    by the default policy), one-sided and both-null quotes, a NaN price, a
    half day and a pre-2018 day."""
    day, half, old = DAY.isoformat(), HALF_DAY.isoformat(), OLD_DAY.isoformat()
    return {
        (DAY, "AAPL"): [
            taq_row("04:00:00.005984", "180", "100", None, None, nano=226, day=day),
            taq_row("09:29:59.500000", 194.00, 200, 194.02, 300, nano=0, day=day),
            taq_row("09:30:30.000000", 194.01, 100, 194.04, 100, nano=0, day=day),
            taq_row("09:31:00.000000", 194.02, 100, 194.03, 200, nano=0, day=day),
            # A tie: two records at one instant, the second wins.
            taq_row("09:32:15.000000", 194.00, 400, 194.04, 100, nano=500, day=day),
            taq_row("09:32:15.000000", 194.01, 300, 194.05, 100, nano=500, day=day),
            # Crossed: dropped, the previous quote stays in force.
            taq_row("09:40:00.000000", 194.10, 100, 194.00, 100, nano=0, day=day),
            # One-sided, then both sides null, then a NaN bid.
            taq_row("10:00:00.000000", None, None, 194.20, 100, nano=0, day=day),
            taq_row("10:05:00.000000", None, None, None, None, nano=0, day=day),
            taq_row("10:10:00.000000", "NaN", 100, 194.30, 200, nano=0, day=day),
            taq_row("16:00:00.000000", 193.90, 100, 193.95, 100, nano=0, day=day),
            taq_row("16:05:00.000000", 193.80, 100, 193.85, 100, nano=0, day=day),
        ],
        (DAY, "BRK.B"): [
            taq_row("09:45:00.000000", 380.00, 100, 380.20, 100, nano=0, day=day,
                    root="BRK", suffix="B"),
            taq_row("12:00:00.000001", 381.00, 300, 381.10, 100, nano=999, day=day,
                    root="BRK", suffix="B"),
        ],
        # Every record of MSFT that day is crossed: the ticker still gets its
        # bars, all without a quote and with no update.
        (DAY, "MSFT"): [
            taq_row("10:00:00.000000", 401.0, 100, 400.0, 100, nano=0, day=day,
                    root="MSFT"),
        ],
        (HALF_DAY, "AAPL"): [
            taq_row("09:00:00.000000", 230.00, 100, 230.10, 100, nano=0, day=half),
            taq_row("12:59:30.000000", 231.00, 200, 231.20, 300, nano=0, day=half),
            taq_row("14:00:00.000000", 232.00, 100, 232.50, 100, nano=0, day=half),
        ],
        (OLD_DAY, "AAPL"): [
            taq_row("09:29:00.000000", 100.00, 100, 100.02, 100, day=old),
            # Three records sharing a microsecond: scan order decides.
            taq_row("09:35:00.000123", 100.01, 100, 100.03, 100, day=old),
            taq_row("09:35:00.000123", 100.02, 200, 100.04, 100, day=old),
            taq_row("09:35:00.000123", 100.00, 300, 100.05, 100, day=old),
            taq_row("15:59:59.999999", 100.10, 100, 100.12, 400, day=old),
        ],
    }


def _reference(tmp_path) -> Path:
    """The CRSP reference tables, with MSFT added to the fixture rows."""
    from tests.crsp_fixtures import (
        _default_reference_rows,
        secinfo_row,
        write_reference_tables,
    )

    directory = tmp_path / "_reference"
    if not (directory / "manifest.json").exists():
        rows = _default_reference_rows()
        rows["crsp_a_stock.stksecurityinfohist"] = list(
            rows["crsp_a_stock.stksecurityinfohist"]
        ) + [
            secinfo_row(
                MSFT, "1986-03-13", "2025-12-31", "MSFT", "MSFT", None,
                securitybegdt="1986-03-13", securityenddt="2025-12-31",
            )
        ]
        write_reference_tables(directory, rows)
    return directory


def _serve(fake, rows: dict) -> tuple[list[str], str, str]:
    """Load `rows` into the fake server; return its tickers and window."""
    days = sorted({day for day, _ in rows})
    by_year: dict[int, list[date]] = {}
    for day in days:
        by_year.setdefault(day.year, []).append(day)
    fake.trading_days_by_year = by_year
    fake.rows = {key: list(value) for key, value in rows.items()}
    tickers = sorted({ticker for _, ticker in rows})
    return tickers, days[0].isoformat(), days[-1].isoformat()


def _download_bars(tmp_path, tickers, start, end, **settings):
    """Download server bars into `tmp_path` and return the acquisition config."""
    import quantlab.config as config
    from quantlab.acquisition import registry
    from quantlab.acquisition.wrds import WRDS_SOURCE
    from quantlab.acquisition.wrds.taq_bars import WrdsTaqNbboBarsAcquisition

    config.set_data_root(tmp_path)
    cfg = WrdsTaqNbboBarsAcquisition.build_config(
        tuple(tickers), start_date=start, end_date=end, **settings
    )
    registry.run(WRDS_SOURCE, cfg)
    return cfg


def _bars_config(tmp_path, acq_cfg, start, end, **fields):
    from quantlab.dataset.config import NbboBarsDatasetConfig

    fields.setdefault("permnos", (str(AAPL), str(MSFT), str(BRK_B)))
    return NbboBarsDatasetConfig(
        zarr_file_path=str(tmp_path / "bars.zarr"),
        raw_data_dir_path=acq_cfg.raw_data_dir_path,
        reference_dir=str(_reference(tmp_path)),
        start_date=start,
        end_date=end,
        **fields,
    )


# -- the highest seam: the same panel as the tick path ---------------------------


def test_server_bars_convert_to_the_tick_panel_on_the_same_records(
    mock_wrds_session, tmp_path
):
    import quantlab.config as config
    from quantlab.acquisition import registry
    from quantlab.acquisition.wrds import WRDS_SOURCE
    from quantlab.acquisition.wrds.taq import WrdsTaqNbboAcquisition
    from quantlab.dataset.config import NbboDatasetConfig
    from quantlab.dataset.nbbo import NbboPanelDataset
    from quantlab.dataset.nbbo.bars import NbboBarsDataset

    tickers, start, end = _serve(mock_wrds_session, _rows())
    permnos = (str(AAPL), str(MSFT), str(BRK_B))

    # The tick path: every record downloaded, resampled locally.
    config.set_data_root(tmp_path / "ticks")
    tick_acq = WrdsTaqNbboAcquisition.build_config(
        tuple(tickers), start_date=start, end_date=end
    )
    registry.run(WRDS_SOURCE, tick_acq)
    tick_cfg = NbboDatasetConfig(
        zarr_file_path=str(tmp_path / "ticks.zarr"),
        raw_data_dir_path=tick_acq.raw_data_dir_path,
        reference_dir=str(_reference(tmp_path)),
        start_date=start,
        end_date=end,
        permnos=permnos,
    )
    registry.convert(WRDS_SOURCE, tick_cfg, data_type="nbbo", granularity="day")

    # The server path: bars downloaded, converted.
    bars_acq = _download_bars(tmp_path / "bars", tickers, start, end)
    bars_cfg = _bars_config(tmp_path, bars_acq, start, end, permnos=permnos)
    registry.convert(WRDS_SOURCE, bars_cfg, granularity="day")

    ticks = NbboPanelDataset(tick_cfg).panel(*WHOLE_STORE)
    bars = NbboBarsDataset(bars_cfg).panel(*WHOLE_STORE)

    np.testing.assert_array_equal(ticks["timestamp"].values, bars["timestamp"].values)
    np.testing.assert_array_equal(ticks["symbol"].values, bars["symbol"].values)
    assert set(bars.data_vars) == set(ticks.data_vars)
    for name in COMPARED:
        np.testing.assert_array_equal(
            bars[name].values, ticks[name].values, err_msg=name
        )
    for name in NOT_YET:
        assert np.isnan(bars[name].values).all(), name

    # The fixture exercises what it claims to: quotes, quiet bars, no quote.
    assert np.nansum(bars["n_updates"].values) > 10
    assert (bars["n_updates"].values == 0).any()
    assert np.isnan(bars["bid"].values).any()

    # The same sidecars beside the store.
    tick_ds, bar_ds = NbboPanelDataset(tick_cfg), NbboBarsDataset(bars_cfg)
    assert json.loads(bar_ds.ticker_sidecar_path().read_text()) == json.loads(
        tick_ds.ticker_sidecar_path().read_text()
    )
    tick_stats = json.loads(Path(tick_ds.filter_stats_path).read_text())
    bar_stats = json.loads(Path(bar_ds.filter_stats_path).read_text())
    assert bar_stats["config"] == tick_stats["config"]
    assert sorted(bar_stats["by_session"]) == sorted(tick_stats["by_session"])
    assert bar_stats["unmapped"] == tick_stats["unmapped"]


# -- the raw tier ------------------------------------------------------------------


def test_raw_tier_layout_watermarks_and_recorded_settings(mock_wrds_session, tmp_path):
    from quantlab.acquisition.wrds.taq_bars import WrdsTaqNbboBarsAcquisition

    rows = {key: value for key, value in _rows().items() if key[0] == DAY}
    tickers, start, end = _serve(mock_wrds_session, rows)
    cfg = _download_bars(tmp_path, tickers, start, end)

    vendor_root = Path(cfg.raw_data_dir_path)
    root = vendor_root / "nbbo_bars"
    assert sorted(path.name for path in vendor_root.iterdir()) == ["nbbo_bars"]
    shard_dirs = sorted({path.parent for path in root.rglob("*.pqt")})
    assert shard_dirs == [
        root / "date=2024-01-24" / f"symbol={ticker}" for ticker in ("AAPL", "BRK.B", "MSFT")
    ]
    shard = pl.read_parquet(next(shard_dirs[0].glob("*.pqt")))
    assert shard.columns == [
        name for name in WrdsTaqNbboBarsAcquisition.RAW_COLUMNS if name != "symbol"
    ]
    assert shard.height == 390
    assert shard["timestamp"].dtype == pl.Datetime("ns")
    assert shard["timestamp"][0].isoformat() == "2024-01-24T14:31:00"
    assert shard["timestamp"][-1].isoformat() == "2024-01-24T21:00:00"

    watermarks = Path(cfg.watermark_path) / "nbbo_bars"
    assert sorted(
        path.name for path in watermarks.glob("*.json") if not path.name.startswith("_")
    ) == ["AAPL.json", "BRK.B.json", "MSFT.json"]
    assert not list(root.rglob("*.json")), "nothing but parquet under the raw root"
    assert json.loads((watermarks / "_request.json").read_text()) == {
        "bar_interval": "1m",
        "session_start": "09:30",
        "session_end": "16:00",
        "drop_crossed": True,
        "drop_locked": False,
        "drop_nonpositive_price": True,
        "keep_qu_cond": None,
    }

    # One statement per trading day and batch, always restricted by the pairs.
    (call,) = mock_wrds_session.bars_calls
    assert call["day"] == DAY
    assert '"taqm_2024"."complete_nbbo_20240124"' in call["sql"]
    assert "sym_root = ANY(ARRAY['AAPL', 'BRK', 'MSFT'])" in call["sql"]
    assert mock_wrds_session.copy_calls == []


def test_interrupted_download_resumes_from_the_recorded_pages(
    mock_wrds_session, tmp_path
):
    rows = {
        (DAY, "AAPL"): _rows()[(DAY, "AAPL")],
        (date(2024, 1, 25), "AAPL"): [
            taq_row("10:00:00.000000", 195.0, 100, 195.1, 100, nano=0, day="2024-01-25"),
        ],
    }
    tickers, start, end = _serve(mock_wrds_session, rows)
    mock_wrds_session.bars_raise_on = {1: ValueError("connection dropped")}

    cfg = _download_bars(tmp_path, tickers, start, end)
    assert [call["day"] for call in mock_wrds_session.bars_calls] == [
        DAY, date(2024, 1, 25),
    ]
    root = Path(cfg.raw_data_dir_path) / "nbbo_bars"
    assert [path.name for path in sorted(root.glob("date=*"))] == ["date=2024-01-24"]

    mock_wrds_session.bars_raise_on = {}
    mock_wrds_session.bars_calls = []
    _download_bars(tmp_path, tickers, start, end)
    # Only the day that failed is fetched again.
    assert [call["day"] for call in mock_wrds_session.bars_calls] == [date(2024, 1, 25)]
    assert [path.name for path in sorted(root.glob("date=*"))] == [
        "date=2024-01-24", "date=2024-01-25",
    ]


def test_a_page_short_of_the_server_count_fails_its_batch(mock_wrds_session, tmp_path):
    import quantlab.config as config
    from quantlab.acquisition import registry
    from quantlab.acquisition.wrds import WRDS_SOURCE
    from quantlab.acquisition.wrds.taq_bars import WrdsTaqNbboBarsAcquisition

    tickers, start, end = _serve(
        mock_wrds_session, {(DAY, "AAPL"): _rows()[(DAY, "AAPL")]}
    )
    mock_wrds_session.bars_short_by = {DAY: 3}
    config.set_data_root(tmp_path)
    cfg = WrdsTaqNbboBarsAcquisition.build_config(
        tuple(tickers), start_date=start, end_date=end
    )
    result = registry.run(WRDS_SOURCE, cfg)
    assert "AAPL" in result.failures
    assert "server counted [390]" in result.failures["AAPL"]
    assert not list((Path(cfg.raw_data_dir_path) / "nbbo_bars").rglob("*.pqt"))

    mock_wrds_session.bars_short_by = {}
    result = registry.run(WRDS_SOURCE, cfg)
    assert result.failures == {}
    assert len(list((Path(cfg.raw_data_dir_path) / "nbbo_bars").rglob("*.pqt"))) == 1


def test_a_run_with_other_settings_is_refused_before_any_query(
    mock_wrds_session, tmp_path
):
    tickers, start, end = _serve(
        mock_wrds_session, {(DAY, "AAPL"): _rows()[(DAY, "AAPL")]}
    )
    _download_bars(tmp_path, tickers, start, end)
    mock_wrds_session.bars_calls = []

    with pytest.raises(ValueError, match="session_end") as refusal:
        _download_bars(
            tmp_path, ["MSFT"], start, end, session_end="15:00", drop_locked=True
        )
    assert "('16:00', '15:00')" in str(refusal.value)
    assert "drop_locked" in str(refusal.value)
    assert mock_wrds_session.bars_calls == []
    # The same settings, spelled differently, are the same request.
    _download_bars(tmp_path, tickers, start, end, session_start="09:30:00")


def test_each_page_logs_day_batch_rows_and_seconds(mock_wrds_session, tmp_path):
    tickers, start, end = _serve(
        mock_wrds_session,
        {key: value for key, value in _rows().items() if key[0] == DAY},
    )
    messages: list[str] = []
    handler = logger.add(lambda message: messages.append(str(message)), level="INFO")
    try:
        _download_bars(tmp_path, tickers, start, end)
    finally:
        logger.remove(handler)
    (line,) = [message for message in messages if "bar row(s)" in message]
    assert "2024-01-24 batch of 3 symbol(s) AAPL..MSFT: 1170 bar row(s) in" in line
    assert line.rstrip().endswith("s")


def test_the_registry_resolves_the_bars_capability():
    from quantlab.acquisition.wrds import WRDS_SOURCE
    from quantlab.acquisition.wrds.taq import WrdsTaqNbboAcquisition
    from quantlab.acquisition.wrds.taq_bars import WrdsTaqNbboBarsAcquisition
    from quantlab.dataset.nbbo import NbboPanelDataset
    from quantlab.dataset.nbbo.bars import NbboBarsDataset

    (bars,) = WRDS_SOURCE.capabilities_for("us_equity", "1m", "nbbo_bars")
    assert bars.acquisition_cls is WrdsTaqNbboBarsAcquisition
    assert bars.config_factory == WrdsTaqNbboBarsAcquisition.build_config
    assert bars.dataset_cls is NbboBarsDataset
    (ticks,) = WRDS_SOURCE.capabilities_for("us_equity", "tick", "nbbo")
    assert ticks.acquisition_cls is WrdsTaqNbboAcquisition
    assert ticks.dataset_cls is NbboPanelDataset


def test_build_config_and_construction_refuse_bad_settings(mock_wrds_session, tmp_path):
    import quantlab.config as config
    from quantlab.acquisition.wrds.taq_bars import WrdsTaqNbboBarsAcquisition

    config.set_data_root(tmp_path)
    with pytest.raises(ValueError, match="keyword"):
        WrdsTaqNbboBarsAcquisition.build_config(("AAPL",), kwargs={"session_end": "15:00"})
    with pytest.raises(ValueError, match="nbbo_bars"):
        WrdsTaqNbboBarsAcquisition.build_config(("AAPL",), kwargs={"data_type": "nbbo"})
    with pytest.raises(ValueError, match="outside the extended window"):
        WrdsTaqNbboBarsAcquisition.build_config(("AAPL",), session_start="21:00")
    cfg = WrdsTaqNbboBarsAcquisition.build_config(("AAPL",))
    bad = replace(cfg, kwargs={**cfg.kwargs, "session_end": "21:00"})
    with pytest.raises(ValueError, match="outside the extended window"):
        WrdsTaqNbboBarsAcquisition(bad)
    assert Path(cfg.raw_data_dir_path).parts[-4:] == ("us_equity", "1m", "wrds_taq", "wrds")


# -- the conversion's own refusals -----------------------------------------------


def test_conversion_refuses_settings_that_differ_from_the_raw_tier(
    mock_wrds_session, tmp_path
):
    from quantlab.dataset.nbbo.bars import NbboBarsDataset

    tickers, start, end = _serve(
        mock_wrds_session, {(DAY, "AAPL"): _rows()[(DAY, "AAPL")]}
    )
    acq = _download_bars(tmp_path, tickers, start, end)
    cfg = _bars_config(tmp_path, acq, start, end, drop_locked=True)
    with pytest.raises(ValueError, match="drop_locked"):
        NbboBarsDataset(cfg).from_raw_data_chunked(granularity="day")
    assert not Path(cfg.zarr_file_path).exists()

    (Path(acq.watermark_path) / "nbbo_bars" / "_request.json").unlink()
    with pytest.raises(ValueError, match="no NBBO bar settings are recorded"):
        NbboBarsDataset(replace(cfg, drop_locked=False)).from_raw_data_chunked(
            granularity="day"
        )


def test_conversion_refuses_two_tickers_resolving_to_one_permno(
    mock_wrds_session, tmp_path
):
    from quantlab.dataset.nbbo.bars import NbboBarsDataset

    tickers, start, end = _serve(
        mock_wrds_session,
        {
            (DAY, "AAPL"): _rows()[(DAY, "AAPL")],
            (DAY, "BRK.B"): _rows()[(DAY, "BRK.B")],
        },
    )
    acq = _download_bars(tmp_path, tickers, start, end)
    # A second raw ticker directory that resolves to AAPL's PERMNO on DAY:
    # the plain root of AAPL's own ticker under another name is not in CRSP,
    # so plant the clash in the symbology instead.
    from tests.crsp_fixtures import (
        _default_reference_rows,
        secinfo_row,
        write_reference_tables,
    )

    reference = tmp_path / "_clash_reference"
    rows = _default_reference_rows()
    rows["crsp_a_stock.stksecurityinfohist"] = [
        row
        for row in rows["crsp_a_stock.stksecurityinfohist"]
        if row["permno"] != str(BRK_B)
    ] + [
        secinfo_row(AAPL, "2024-01-01", "2024-12-31", "BRK", "BRKB", "B"),
    ]
    write_reference_tables(reference, rows)
    cfg = replace(
        _bars_config(tmp_path, acq, start, end, permnos=(str(AAPL),)),
        reference_dir=str(reference),
    )
    with pytest.raises(ValueError, match=r"2024-01-24 PERMNO 14593: \['AAPL', 'BRK.B'\]"):
        NbboBarsDataset(cfg).from_raw_data_chunked(granularity="day")


# -- the statement -----------------------------------------------------------------


def _query(policy=None, *, day=DAY, pairs=(("AAPL", None), ("BRK", "B")), has_nano=True):
    from quantlab.acquisition.wrds.nbbo_bars_sql import NbboBarsQuery
    from quantlab.dataset.nbbo.bars import NbboBarsRequest
    from quantlab.dataset.nbbo.resample import NbboFilterPolicy

    request = NbboBarsRequest("1m", policy=policy or NbboFilterPolicy())
    session = request.session(request.calendar(), day)
    return NbboBarsQuery(day, pairs, session, request, has_nano)


def test_statement_is_composed_and_restricted_by_the_pairs():
    from psycopg2 import sql

    statement = _query().statement()
    assert isinstance(statement, sql.Composed)
    text = render_composed(statement)
    assert text.startswith("COPY (") and text.endswith(
        "TO STDOUT WITH (FORMAT csv, HEADER true)"
    )
    assert 'FROM "taqm_2024"."complete_nbbo_20240124"' in text
    assert (
        "WHERE sym_root = ANY(ARRAY['AAPL', 'BRK']) AND "
        "(sym_root, coalesce(sym_suffix, '')) IN (('AAPL', ''), ('BRK', 'B'))"
    ) in text
    # Session bounds (09:30 and 16:00 ET, as nanoseconds since midnight) and
    # the bar length are literals; 390 one-minute bars.
    assert "t_ns <= 34200000000000" in text
    assert "t_ns <= 57600000000000" in text
    assert "+ 60000000000 - 1) / 60000000000" in text
    assert "generate_series(0, 390)" in text
    assert "row_number() OVER () AS ord" in text
    assert "coalesce(time_m_nano, 0)" in text
    assert "time_m_nano" not in render_composed(_query(day=OLD_DAY, has_nano=False).statement())


def test_statement_half_day_bounds_come_from_the_calendar():
    text = render_composed(_query(day=HALF_DAY).statement())
    assert "t_ns <= 46800000000000" in text  # 13:00 ET
    assert "generate_series(0, 210)" in text


def test_statement_filter_rules_appear_only_when_enabled():
    from quantlab.dataset.nbbo.resample import NbboFilterPolicy

    default = render_composed(_query().statement())
    assert "coalesce(bid <= 0, false)" in default
    assert "coalesce(bid > ask, false)" in default
    assert "bid = ask" not in default
    assert "qu_cond::text" not in default

    everything = render_composed(
        _query(NbboFilterPolicy(drop_locked=True, keep_qu_cond=("R", "C"))).statement()
    )
    assert "coalesce(bid = ask, false)" in everything
    assert "qu_cond::text = ANY(ARRAY['R', 'C']::text[])" in everything

    nothing = render_composed(
        _query(
            NbboFilterPolicy(drop_crossed=False, drop_nonpositive_price=False)
        ).statement()
    )
    assert "NOT (false)" in nothing
    assert "bid <= 0" not in nothing and "bid > ask" not in nothing


def test_statement_values_reach_the_query_only_as_literals():
    from psycopg2 import sql

    hostile = "X'; DROP TABLE t; --"
    statement = _query(pairs=((hostile, None),)).statement()
    literals = []

    def walk(node):
        if isinstance(node, sql.Composed):
            for part in node.seq:
                walk(part)
        elif isinstance(node, sql.Literal):
            literals.append(node.wrapped)
        elif isinstance(node, sql.SQL):
            assert hostile not in node.string

    walk(statement)
    assert [hostile] in literals


def test_statement_refuses_an_empty_batch():
    with pytest.raises(ValueError, match="whole complete_nbbo table"):
        _query(pairs=())


def test_real_session_runs_the_statement_through_copy(monkeypatch):
    from quantlab.acquisition.wrds.taq import WrdsSession

    monkeypatch.setenv("WRDS_USERNAME", "test-wrds-user-not-real")
    connections: list = []
    monkeypatch.setattr("psycopg2.connect", fake_connect(connections))
    monkeypatch.setattr(WrdsSession, "_assert_pgpass_entry", lambda self: None)
    session = WrdsSession("test-wrds-user-not-real")
    query = _query()
    connections_payload = b"sym_root,sym_suffix\n"

    def payload(**kwargs):
        connection = fake_connect(connections)(**kwargs)
        connection.copy_payload = connections_payload
        return connection

    monkeypatch.setattr("psycopg2.connect", payload)
    assert session.copy_nbbo_bars_csv(query) == connections_payload
    ((text, params),) = connections[0].executed
    assert text == render_composed(query.statement())
    assert params is None
