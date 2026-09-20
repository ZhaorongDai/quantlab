"""`scripts/ingest_wrds_crsp.py`: the WRDS CRSP Stock v2 daily ingest CLI (03.10-10).

Two groups, the shape `tests/test_ingest_wrds_taq.py` established in 03.9-07:

- end-to-end runs of the REAL script through `runpy.run_path` against
  `FakeCrspSession` (the `mock_crsp_session` fixture patches
  `quantlab.acquisition.wrds_taq.WrdsSession`, which every WRDS provider and
  this script reach at run time), so no test can open a WRDS connection;
- structural (AST/source) locks on the script: the guard's placement relative
  to the probe and the pull, the reference-pull ordering, the force flag, no
  credential argument, no vendor class, no `quantlab/config` factory, one
  session per run.

**Every `quantlab` import lives inside a test or a helper body**, and so does
every read of the script. This module is written BEFORE the script exists, and
a module-scope import (or a module-scope `SCRIPT.read_text()`) would turn the
RED run into a COLLECTION error -- zero tests discovered, which proves nothing
about the behaviour (TDD gate #3770). The convention is plan 02's, continued.

The structural helpers are re-implemented here rather than imported from
`tests/test_ingest_wrds_taq.py`: that module is another plan's test file, and
importing its private helpers would couple two locks that must be able to move
independently.
"""

from __future__ import annotations

import ast
import runpy
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "ingest_wrds_crsp.py"

#: The live CRSP annual product end every fixture row is written against
#: (`03.10-LIVE-CHECK-2.json` key `L9_1`), restated so a silent change to
#: `FakeCrspSession.product_end` fails here rather than passing quietly.
PRODUCT_END = "2025-12-31"

#: AAPL's PERMNO and the August 2020 window `crsp_fixtures.AAPL_AUG_2020_ROWS`
#: covers (four trading days).
AAPL_PERMNO = "14593"
AUG_2020 = ["--start-date", "2020-08-01", "--end-date", "2020-08-31"]

#: QQQ's PERMNO (`CrspDatasetConfig.QQQ_PERMNO`), restated for the same reason.
QQQ_PERMNO = "86755"


# ---------------------------------------------------------------------------
# Driving the real script
# ---------------------------------------------------------------------------


def _run_script(monkeypatch, args: list[str]) -> int:
    """Run the real script as `__main__`; return its exit code (0 on return)."""
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), *args])
    try:
        runpy.run_path(str(SCRIPT), run_name="__main__")
    except SystemExit as exc:
        code = exc.code
        if code is None:
            return 0
        if isinstance(code, int):
            return code
        print(code, file=sys.stderr)
        return 1
    return 0


def _permno_args(tmp_path, *extra: str) -> list[str]:
    return [
        "--permnos", AAPL_PERMNO, *AUG_2020, "--data-dir", str(tmp_path), *extra
    ]


def _universe_args(tmp_path, universe: str, *extra: str) -> list[str]:
    return [
        "--universe", universe, *AUG_2020, "--data-dir", str(tmp_path), *extra
    ]


def _daily_copies(session) -> list[str]:
    """The rendered SQL of every `dsf_v2` COPY this run issued."""
    return [call["sql"] for call in session.crsp_copy_calls]


def _reference_dir(tmp_path) -> Path:
    return (
        tmp_path / "downloads" / "us_equity" / "1d" / "wrds_crsp" / "_reference"
    )


def _equity_store(tmp_path, name: str = "custom") -> Path:
    return tmp_path / "data" / "us_equity" / "1d" / f"wrds_crsp_{name}_1d.zarr"


# ---------------------------------------------------------------------------
# Task 2: end-to-end against the fake session
# ---------------------------------------------------------------------------


def test_end_to_end_permnos_pull_and_convert_uses_one_connection(
    mock_crsp_session, tmp_path, monkeypatch, capsys
):
    """One command: entitle, clip-check, references, roster, price, pull, convert.

    The connection count is the assertion that matters most (T-03.10-34): the
    probe, the reference pull, the daily pull and the conversion must share ONE
    `WrdsSession`, because every extra connection can push a Duo prompt.
    """
    import xarray as xr

    code = _run_script(monkeypatch, _permno_args(tmp_path, "--to-zarr"))
    out = capsys.readouterr()
    assert code == 0, out.err

    # The estimate is printed with the CRSP bucket unit (task 1).
    assert "Pre-flight WRDS volume estimate" in out.out
    assert "year buckets:" in out.out
    assert "trading days:" not in out.out
    assert "succeeded" in out.out

    raw_root = tmp_path / "downloads" / "us_equity" / "1d" / "wrds_crsp" / "wrds"
    assert list(raw_root.rglob("*.pqt")), f"no raw shard under {raw_root}"
    assert (_reference_dir(tmp_path) / "manifest.json").exists()

    store = _equity_store(tmp_path)
    assert str(store) in out.out
    for suffix in (
        ".crsp_adjustment.json",
        ".crsp_filter_report.json",
        ".crsp_symbology_report.json",
    ):
        assert f"{store}{suffix}" in out.out, suffix

    panel = xr.open_zarr(store).load()
    assert [str(value) for value in panel["symbol"].values] == ["AAPL"]

    assert mock_crsp_session.connections == 1
    assert mock_crsp_session.instance is None, "close_shared was not called"


def test_without_to_zarr_the_universe_run_stops_at_raw(
    mock_crsp_session, tmp_path, monkeypatch, capsys
):
    """`--universe crsp_sp500` resolves its roster from the reference tier."""
    code = _run_script(monkeypatch, _universe_args(tmp_path, "crsp_sp500"))
    out = capsys.readouterr()
    assert code == 0, out.err

    assert "Skipping Zarr conversion" in out.out
    assert not (tmp_path / "data" / "us_equity" / "1d").exists()

    # The roster is CRSP's own dsp500list_v2 membership: AAPL alone here.
    assert f"{AAPL_PERMNO}" in out.out
    for sql in _daily_copies(mock_crsp_session):
        assert f"'{AAPL_PERMNO}'" in sql or f"{AAPL_PERMNO}" in sql
    assert (_reference_dir(tmp_path) / "dsp500list_v2.parquet").exists()


def test_the_nasdaq100_universe_resolves_the_alphabet_permnos(
    mock_crsp_session, tmp_path, monkeypatch, capsys
):
    """`comp_nasdaq100` pulls idxcst_his + ccmxpf_lnkhist and links both classes."""
    code = _run_script(
        monkeypatch,
        [
            "--universe", "comp_nasdaq100",
            "--start-date", "2015-01-01", "--end-date", "2015-12-31",
            "--data-dir", str(tmp_path),
        ],
    )
    out = capsys.readouterr()
    assert code == 0, out.err

    reference = _reference_dir(tmp_path)
    assert (reference / "idxcst_his.parquet").exists()
    assert (reference / "ccmxpf_lnkhist.parquet").exists()
    # GOOGL (90319) and GOOG (14542), numerically sorted.
    assert "14542" in out.out and "90319" in out.out


def test_an_unlinked_nasdaq100_spell_refuses_naming_the_flag(
    mock_crsp_session, tmp_path, monkeypatch, capsys
):
    _plant_unlinked_ndx_spell(mock_crsp_session)
    code = _run_script(
        monkeypatch,
        [
            "--universe", "comp_nasdaq100",
            "--start-date", "2015-01-01", "--end-date", "2015-12-31",
            "--data-dir", str(tmp_path),
        ],
    )
    out = capsys.readouterr()
    assert code != 0
    assert "--allow-unlinked-ndx" in out.err
    assert _daily_copies(mock_crsp_session) == []


def test_allow_unlinked_ndx_proceeds_with_the_linked_members(
    mock_crsp_session, tmp_path, monkeypatch, capsys
):
    _plant_unlinked_ndx_spell(mock_crsp_session)
    code = _run_script(
        monkeypatch,
        [
            "--universe", "comp_nasdaq100",
            "--start-date", "2015-01-01", "--end-date", "2015-12-31",
            "--data-dir", str(tmp_path), "--allow-unlinked-ndx",
        ],
    )
    out = capsys.readouterr()
    assert code == 0, out.err
    assert "14542" in out.out and "90319" in out.out


def _plant_unlinked_ndx_spell(session) -> None:
    """A Nasdaq-100 spell whose `(gvkey, iid)` has no LC/LU/LS link.

    SYNTHETIC: `iid = '90C'` exists in the live `ccmxpf_lnkhist` rows only as
    an `NR` ("no research") row with a NULL `lpermno`, so a membership spell
    for it is a member with no security.
    """
    rows = list(session.reference_rows["comp.idxcst_his"])
    rows.append(
        {
            "gvkey": "160329",
            "iid": "90C",  # SYNTHETIC spell over a live NR-only link
            "gvkeyx": "000208",
            "from": "2015-01-01",
            "thru": None,
        }
    )
    session.reference_rows = dict(session.reference_rows)
    session.reference_rows["comp.idxcst_his"] = rows


def test_an_end_date_past_the_product_end_is_clipped_and_printed(
    mock_crsp_session, tmp_path, monkeypatch, capsys
):
    """The clip is stated verbatim, never silent (T-03.10-37)."""
    code = _run_script(
        monkeypatch,
        [
            "--permnos", AAPL_PERMNO,
            "--start-date", "2020-08-01", "--end-date", "2026-06-30",
            "--data-dir", str(tmp_path),
        ],
    )
    out = capsys.readouterr()
    assert code == 0, out.err
    assert (
        f"clipped end 2026-06-30 -> {PRODUCT_END} "
        f"(crsp_a_stock annual product end)"
    ) in out.out

    copies = _daily_copies(mock_crsp_session)
    assert copies, "no daily COPY issued"
    assert f"'{PRODUCT_END}'" in copies[-1]


def test_a_start_past_the_product_end_refuses_with_no_daily_copy(
    mock_crsp_session, tmp_path, monkeypatch, capsys
):
    code = _run_script(
        monkeypatch,
        [
            "--permnos", AAPL_PERMNO,
            "--start-date", "2026-01-05", "--end-date", "2026-06-30",
            "--data-dir", str(tmp_path),
        ],
    )
    out = capsys.readouterr()
    assert code != 0
    assert PRODUCT_END in out.err
    assert _daily_copies(mock_crsp_session) == []


def test_the_volume_guard_refuses_before_any_daily_copy(
    mock_crsp_session, tmp_path, monkeypatch, capsys
):
    from quantlab.acquisition.sql_volume import SqlVolumeGuard

    monkeypatch.setattr(SqlVolumeGuard, "MAX_RAW_ROWS", 1)
    code = _run_script(monkeypatch, _permno_args(tmp_path))
    out = capsys.readouterr()
    assert code != 0
    assert "--force-volume" in out.err
    assert _daily_copies(mock_crsp_session) == []


def test_force_volume_proceeds_and_says_so(
    mock_crsp_session, tmp_path, monkeypatch, capsys
):
    from quantlab.acquisition.sql_volume import SqlVolumeGuard

    monkeypatch.setattr(SqlVolumeGuard, "MAX_RAW_ROWS", 1)
    code = _run_script(monkeypatch, _permno_args(tmp_path, "--force-volume"))
    out = capsys.readouterr()
    assert code == 0, out.err
    assert "--force-volume:    ON" in out.out
    assert _daily_copies(mock_crsp_session)


def test_an_unentitled_index_schema_refuses_before_any_copy(
    mock_crsp_session, tmp_path, monkeypatch, capsys
):
    """`--universe crsp_sp500` needs crsp_a_indexes, and says so first."""
    mock_crsp_session.usable_schemas = {"crsp_a_stock"}
    code = _run_script(monkeypatch, _universe_args(tmp_path, "crsp_sp500"))
    out = capsys.readouterr()
    assert code != 0
    assert "crsp_a_indexes" in out.err
    assert _daily_copies(mock_crsp_session) == []
    assert not _reference_dir(tmp_path).exists()


def test_a_missing_username_is_refused_by_name(
    mock_crsp_session, tmp_path, monkeypatch, capsys
):
    monkeypatch.delenv("WRDS_USERNAME", raising=False)
    code = _run_script(monkeypatch, _permno_args(tmp_path))
    out = capsys.readouterr()
    assert code != 0
    assert "WRDS_USERNAME" in out.err
    assert _daily_copies(mock_crsp_session) == []


def test_a_planted_credential_value_is_never_printed(
    mock_crsp_session, tmp_path, monkeypatch, capsys
):
    planted = "planted-wrds-user-9e21"
    monkeypatch.setenv("WRDS_USERNAME", planted)
    code = _run_script(monkeypatch, _permno_args(tmp_path, "--to-zarr"))
    out = capsys.readouterr()
    assert code == 0, out.err
    assert planted not in out.out
    assert planted not in out.err


def test_qqq_gets_its_own_benchmark_store(
    mock_crsp_session, tmp_path, monkeypatch, capsys
):
    """D-15: the ETF is a SEPARATE store, never one more equity column."""
    import xarray as xr

    code = _run_script(
        monkeypatch,
        [
            "--qqq", "--permnos", AAPL_PERMNO,
            "--start-date", "1999-01-01", "--end-date", PRODUCT_END,
            "--data-dir", str(tmp_path), "--to-zarr",
        ],
    )
    out = capsys.readouterr()
    assert code == 0, out.err

    equity = xr.open_zarr(_equity_store(tmp_path)).load()
    assert "QQQ" not in [str(value) for value in equity["symbol"].values]

    benchmark = tmp_path / "data" / "us_equity" / "1d" / "wrds_crsp_qqq_1d.zarr"
    assert str(benchmark) in out.out
    assert [
        str(value) for value in xr.open_zarr(benchmark).load()["symbol"].values
    ] == ["QQQ"]


def test_the_universe_conversion_also_writes_the_membership_panel(
    mock_crsp_session, tmp_path, monkeypatch, capsys
):
    import xarray as xr

    code = _run_script(
        monkeypatch, _universe_args(tmp_path, "crsp_sp500", "--to-zarr")
    )
    out = capsys.readouterr()
    assert code == 0, out.err

    membership = (
        tmp_path / "data" / "us_equity" / "1d" / "wrds_crsp_sp500_membership.zarr"
    )
    assert str(membership) in out.out
    panel = xr.open_zarr(membership).load()
    assert "is_member" in panel.data_vars
    assert "AAPL" in [str(value) for value in panel["symbol"].values]

    # The equity store carries the universe's own name, not "custom".
    assert _equity_store(tmp_path, "sp500").exists()


@pytest.mark.parametrize(
    "args, needle",
    [
        (AUG_2020, "--universe"),
        (["--permnos", "AAPL", *AUG_2020], "PERMNO"),
        (["--permnos", AAPL_PERMNO, "--start-date", "2020-08-01"], "--end-date"),
        (
            ["--permnos", AAPL_PERMNO, *AUG_2020, "--rows-per-symbol-day", "10"],
            "counted server-side",
        ),
        (["--universe", "sp500", *AUG_2020], "invalid choice"),
        (
            ["--permnos", AAPL_PERMNO, *AUG_2020, "--security-filter", "bogus"],
            "invalid choice",
        ),
    ],
)
def test_parser_refusals_cost_no_connection(
    mock_crsp_session, tmp_path, monkeypatch, capsys, args, needle
):
    code = _run_script(monkeypatch, [*args, "--data-dir", str(tmp_path)])
    out = capsys.readouterr()
    assert code == 2, out.out
    assert needle in out.err
    assert mock_crsp_session.connections == 0
