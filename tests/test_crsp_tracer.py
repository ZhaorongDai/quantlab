"""The phase 03.10 TRACER: one PERMNO-month becomes a drop-in CRSP panel.

ONE path, end to end, entirely offline:

    FakeCrspSession (AAPL's real 2020-08 `crsp_a_stock.dsf_v2` rows)
      -> registry.run(WRDS_SOURCE, cfg)        # the new crsp_daily capability
      -> PERMNO-keyed raw shards under month=2020-08/
      -> registry.convert(WRDS_SOURCE, CrspDatasetConfig, ...)
      -> a [timestamp, symbol] Zarr panel whose symbol axis is ['AAPL'] and
         whose total-return adjClose matches a hand calculation

The window is chosen so the arithmetic is checkable by hand: AAPL's 4:1 split
took effect on 2020-08-31, and 2020-08-07 is a dividend ex-date. Both are
VERBATIM live rows (`03.10-LIVE-CHECK-2.json` key `L4_1`), so the assertions
below are about CRSP's real numbers, not about a shape someone invented.

Every quantlab import is INSIDE a test body. That is not style: the tracer is
written before the modules it names exist, and a module-scope import would
turn the RED run into a collection error -- zero tests discovered, which
proves nothing about the behaviour (TDD gate #3770).
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The PERMNO the whole tracer travels on: Apple Inc.
AAPL_PERMNO = "14593"


def test_tracer_one_permno_month_lands_raw_and_converts_to_a_drop_in_panel(
    mock_crsp_session, tmp_path
):
    """One PERMNO-month: fake WRDS -> raw PERMNO shards -> drop-in panel.

    Asserted in three parts, because a tracer that only checked the last one
    could pass with the raw tier keyed on anything at all:

    1. **The pull.** The capability resolves, the roster succeeds, exactly one
       `month=2020-08` shard directory holds the four rows, every shard
       carries `RAW_COLUMNS`, the `symbol` column is the PERMNO string, a
       watermark lands per PERMNO, and ONE session was opened.
    2. **The SQL.** The single CRSP COPY names `"crsp_a_stock"."dsf_v2"`,
       carries the PERMNO array and the date BETWEEN, and contains no
       `ORDER BY` / `GROUP BY` / `DISTINCT` -- the D-03 prohibition, asserted
       on the rendered text rather than on the builder's arguments.
    3. **The panel.** The ticker is derived at CONVERSION time (so the raw
       tier never had to know it), and the total-return `adjClose` reproduces
       the split and the dividend exactly.
    """
    import xarray as xr

    from quantlab.acquisition import registry
    from quantlab.acquisition.wrds import WRDS_SOURCE
    from quantlab.acquisition.wrds_crsp import WrdsCrspDailyAcquisition
    from quantlab.base.config import CrspDatasetConfig
    from tests.crsp_fixtures import (
        AAPL_AUG_2020_ROWS,
        FakeCrspSession,
        run_crsp_pull,
        write_reference_tables,
    )
    from tests.wrds_fixtures import FakeWrdsSession

    FakeCrspSession.daily_rows = list(AAPL_AUG_2020_ROWS)

    # -- 1. the pull -------------------------------------------------------
    cfg, result = run_crsp_pull(
        tmp_path, [AAPL_PERMNO], start_date="2020-08-01", end_date="2020-08-31"
    )

    assert result.succeeded == (AAPL_PERMNO,), result
    assert result.failures == {}, result.failures

    raw_root = Path(cfg.raw_data_dir_path)
    assert raw_root.name == "wrds", raw_root
    assert str(raw_root).endswith(
        str(Path("downloads") / "us_equity" / "1d" / "wrds_crsp" / "wrds")
    ), raw_root

    partitions = sorted(p.name for p in raw_root.iterdir() if p.is_dir())
    assert partitions == ["month=2020-08"], partitions

    import polars as pl

    shards = sorted(raw_root.rglob("*.pqt"))
    assert shards, "the pull wrote no raw shard"
    frames = [pl.read_parquet(shard) for shard in shards]
    for frame in frames:
        # `month` is the hive key; `_write_shard` drops it from the file.
        assert tuple(frame.columns) == tuple(
            name
            for name in WrdsCrspDailyAcquisition.RAW_COLUMNS
            if name != "month"
        ), frame.columns
    raw = pl.concat(frames)
    assert raw.height == 4, raw
    assert raw["symbol"].to_list() == [AAPL_PERMNO] * 4
    assert raw.schema["permno"] == pl.Int64
    assert raw.schema["timestamp"] == pl.Datetime("us")
    assert [str(value) for value in raw["timestamp"].to_list()] == [
        "2020-08-06 00:00:00",
        "2020-08-07 00:00:00",
        "2020-08-28 00:00:00",
        "2020-08-31 00:00:00",
    ]

    watermark = Path(cfg.watermark_path) / f"{AAPL_PERMNO}.json"
    assert watermark.exists(), sorted(Path(cfg.watermark_path).rglob("*"))

    # -- 2. the SQL --------------------------------------------------------
    assert len(FakeCrspSession.crsp_copy_calls) == 1, FakeCrspSession.crsp_copy_calls
    sql_text = FakeCrspSession.crsp_copy_calls[0]["sql"]
    assert '"crsp_a_stock"."dsf_v2"' in sql_text, sql_text
    assert f"ANY(ARRAY[{AAPL_PERMNO}])" in sql_text, sql_text
    assert "BETWEEN '2020-08-01' AND '2020-08-31'" in sql_text, sql_text
    for forbidden in ("ORDER BY", "GROUP BY", "DISTINCT"):
        assert forbidden not in sql_text.upper(), (forbidden, sql_text)

    assert FakeWrdsSession.connections == 1, FakeWrdsSession.connections

    # -- 3. the panel ------------------------------------------------------
    reference_dir = WrdsCrspDailyAcquisition.reference_dir_for(cfg)
    assert Path(reference_dir).parent == raw_root.parent, reference_dir
    write_reference_tables(reference_dir)

    dataset_config = CrspDatasetConfig(
        zarr_file_path=str(tmp_path / "crsp.zarr"),
        raw_data_dir_path=cfg.raw_data_dir_path,
        catalog_path=str(tmp_path / "catalog"),
        reference_dir=str(reference_dir),
        start_date="2020-08-01",
        end_date="2020-08-31",
    )
    registry.convert(
        WRDS_SOURCE,
        dataset_config,
        data_type="crsp_daily",
        granularity="year",
    )

    panel = xr.open_zarr(dataset_config.zarr_file_path)
    assert [str(value)[:10] for value in panel["timestamp"].values] == [
        "2020-08-06",
        "2020-08-07",
        "2020-08-28",
        "2020-08-31",
    ]
    assert [str(value) for value in panel["symbol"].values] == ["AAPL"]

    expected_variables = {
        "open", "high", "low", "close", "volume",
        "adjOpen", "adjHigh", "adjLow", "adjClose", "adjVolume",
        "divCash", "splitFactor", "permno", "permco",
    }
    assert expected_variables <= set(panel.data_vars), sorted(panel.data_vars)
    for name in sorted(expected_variables):
        assert str(panel[name].dtype) == "float64", (name, panel[name].dtype)

    def at(variable: str, day: str) -> float:
        return float(panel[variable].sel(timestamp=day, symbol="AAPL").values)

    # The split day is the anchor: its adjusted close IS its close.
    assert at("close", "2020-08-31") == pytest.approx(129.04)
    assert at("adjClose", "2020-08-31") == pytest.approx(129.04)

    # Total return, not price: the day before the split is one 3.3912% step
    # below it on the adjusted series, even though the raw prices differ 4x.
    assert at("adjClose", "2020-08-28") / at("adjClose", "2020-08-31") == (
        pytest.approx(1 / 1.033912, rel=1e-9)
    )
    # The dividend ex-date: `dlyret` (-2.2695%) not `dlyretx` (-2.4495%).
    assert at("adjClose", "2020-08-07") / at("adjClose", "2020-08-06") == (
        pytest.approx(0.977305, rel=1e-9)
    )

    # The price factor moves by exactly the 4:1 split across 08-28 -> 08-31.
    factor_28 = at("adjClose", "2020-08-28") / at("close", "2020-08-28")
    factor_31 = at("adjClose", "2020-08-31") / at("close", "2020-08-31")
    assert factor_28 / factor_31 == pytest.approx(0.25, abs=1e-5)

    # Volume rides `dlycumfacshr`, which is 4 before the split and 1 after.
    assert at("adjVolume", "2020-08-28") == pytest.approx(4 * 1_000_000)
    assert at("adjVolume", "2020-08-31") == pytest.approx(1_000_000)

    assert at("divCash", "2020-08-07") == pytest.approx(0.82)
    assert at("divCash", "2020-08-06") == pytest.approx(0.0)
    assert at("splitFactor", "2020-08-31") == pytest.approx(4.0)
    assert at("splitFactor", "2020-08-28") == pytest.approx(1.0)

    for day in ("2020-08-06", "2020-08-07", "2020-08-28", "2020-08-31"):
        assert at("permno", day) == pytest.approx(14593.0)
