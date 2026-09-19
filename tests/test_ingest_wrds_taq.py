"""`scripts/ingest_wrds_taq.py`: the WRDS TAQ NBBO ingest CLI (plan 03.9-07).

Two groups:

- end-to-end runs of the REAL script through `runpy.run_path` against
  `FakeWrdsSession` (the `mock_wrds_session` fixture patches
  `quantlab.acquisition.wrds_taq.WrdsSession`, which the script imports by name
  at run time), so no test can reach WRDS;
- structural (AST/source) locks on the script: the guard's placement, the
  force flag, no credential argument, no vendor class, no config factory, one
  session per run.
"""

import ast
import runpy
import sys
from datetime import date
from pathlib import Path

import pytest

from tests.wrds_fixtures import taq_row

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "ingest_wrds_taq.py"

DAY1 = date(2024, 1, 24)
DAY2 = date(2024, 1, 25)
WINDOW = ["--start-date", "2024-01-24", "--end-date", "2024-01-25"]


def _rows() -> dict:
    """AAPL and BRK.B on both days, in physical order."""
    rows: dict = {}
    for day in (DAY1, DAY2):
        iso = day.isoformat()
        rows[(day, "AAPL")] = [
            taq_row("09:29:00.000000", 194.00, 100, 194.02, 100, nano=0, day=iso),
            taq_row("10:00:00.000000", 194.03, 100, 194.05, 200, nano=0, day=iso),
            taq_row("15:00:00.000000", 194.10, 100, 194.12, 100, nano=0, day=iso),
        ]
        rows[(day, "BRK.B")] = [
            taq_row(
                "09:25:00.000000", 380.00, 100, 380.20, 100,
                nano=0, day=iso, root="BRK", suffix="B",
            ),
            taq_row(
                "11:00:00.000000", 380.10, 100, 380.30, 100,
                nano=0, day=iso, root="BRK", suffix="B",
            ),
        ]
    return rows


@pytest.fixture
def wrds(mock_wrds_session):
    mock_wrds_session.trading_days_by_year = {2024: [DAY1, DAY2]}
    mock_wrds_session.rows = _rows()
    return mock_wrds_session


def _run_script(monkeypatch, args: list[str]) -> int:
    """Run the real script as `__main__`; return its exit code (0 on a normal
    return)."""
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), *args])
    try:
        runpy.run_path(str(SCRIPT), run_name="__main__")
    except SystemExit as exc:
        code = exc.code
        if code is None:
            return 0
        if isinstance(code, int):
            return code
        # `raise SystemExit("message")`: Python prints it and exits 1.
        print(code, file=sys.stderr)
        return 1
    return 0


def _base(tmp_path, *extra: str) -> list[str]:
    return ["--symbols", "AAPL,BRK.B", *WINDOW, "--data-dir", str(tmp_path), *extra]


# ---------------------------------------------------------------------------
# Task 1: end-to-end against the fake session
# ---------------------------------------------------------------------------


def test_end_to_end_pull_and_convert_uses_one_connection(
    wrds, tmp_path, monkeypatch, capsys
):
    import xarray as xr

    code = _run_script(
        monkeypatch, _base(tmp_path, "--to-zarr", "--bar-interval", "5m")
    )
    out = capsys.readouterr()
    assert code == 0, out.err

    assert "Pre-flight WRDS volume estimate" in out.out
    assert "succeeded" in out.out
    raw_root = (
        tmp_path / "downloads" / "us_equity" / "tick" / "wrds_taq" / "wrds"
        / "data_type=nbbo"
    )
    assert list(raw_root.rglob("*.pqt")), f"no raw shard under {raw_root}"

    store = (
        tmp_path / "data" / "us_equity" / "tick" / "wrds_nbbo_5m_0930-1600.zarr"
    )
    assert f"Zarr store written at: {store}" in out.out
    sidecar = f"{store}.nbbo_filter_stats.json"
    assert sidecar in out.out
    assert Path(sidecar).exists()

    panel = xr.open_zarr(store).load()
    assert panel.sizes["timestamp"] == 156
    assert [str(s) for s in panel["symbol"].values] == ["AAPL", "BRK.B"]

    # Probe, pull and conversion shared ONE session (D-20).
    assert wrds.connections == 1
    assert wrds.count_calls, "the volume probe issued no count"
    assert wrds.copy_calls, "the pull issued no COPY"
    assert wrds.instance is None, "close_shared was not called"


def test_without_to_zarr_the_run_stops_at_raw(wrds, tmp_path, monkeypatch, capsys):
    code = _run_script(monkeypatch, _base(tmp_path))
    out = capsys.readouterr()
    assert code == 0, out.err
    assert "Skipping Zarr conversion" in out.out
    assert not (tmp_path / "data" / "us_equity" / "tick").exists()


def test_volume_refusal_exits_before_any_copy(wrds, tmp_path, monkeypatch, capsys):
    wrds.count_adjust = {DAY2: 800_000_000}
    code = _run_script(monkeypatch, _base(tmp_path))
    out = capsys.readouterr()
    assert code != 0
    assert "--force-volume" in out.err
    assert "--end-date 2024-01-24" in out.err
    assert wrds.copy_calls == []
    assert wrds.connections == 1


def test_force_volume_proceeds_and_says_so(wrds, tmp_path, monkeypatch, capsys):
    wrds.count_adjust = {DAY2: 800_000_000}
    code = _run_script(monkeypatch, _base(tmp_path, "--force-volume"))
    out = capsys.readouterr()
    assert code == 0, out.err
    assert "--force-volume:    ON" in out.out
    assert wrds.copy_calls


def test_unentitled_year_refuses_before_any_count_or_copy(
    wrds, tmp_path, monkeypatch, capsys
):
    wrds.entitled_years = {2024}
    code = _run_script(
        monkeypatch,
        [
            "--symbols", "AAPL",
            "--start-date", "2012-01-03", "--end-date", "2012-01-04",
            "--data-dir", str(tmp_path),
        ],
    )
    out = capsys.readouterr()
    assert code != 0
    assert "taqm_2012" in out.err
    assert wrds.count_calls == []
    assert wrds.copy_calls == []


def test_missing_username_is_refused_by_name(wrds, tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("WRDS_USERNAME", raising=False)
    code = _run_script(monkeypatch, _base(tmp_path))
    out = capsys.readouterr()
    assert code != 0
    assert "WRDS_USERNAME" in out.err
    assert wrds.count_calls == [] and wrds.copy_calls == []


def test_username_value_is_never_printed(wrds, tmp_path, monkeypatch, capsys):
    planted = "planted-wrds-user-7f3a"
    monkeypatch.setenv("WRDS_USERNAME", planted)
    code = _run_script(monkeypatch, _base(tmp_path, "--to-zarr"))
    out = capsys.readouterr()
    assert code == 0, out.err
    assert planted not in out.out
    assert planted not in out.err


def test_extended_session_window_converts(wrds, tmp_path, monkeypatch, capsys):
    import xarray as xr

    code = _run_script(
        monkeypatch,
        _base(
            tmp_path, "--to-zarr", "--session-start", "04:00",
            "--session-end", "20:00", "--bar-interval", "30m",
        ),
    )
    out = capsys.readouterr()
    assert code == 0, out.err
    store = (
        tmp_path / "data" / "us_equity" / "tick" / "wrds_nbbo_30m_0400-2000.zarr"
    )
    panel = xr.open_zarr(store).load()
    assert panel.sizes["timestamp"] == 2 * 32


def test_session_window_outside_sip_range_is_refused_before_wrds(
    wrds, tmp_path, monkeypatch, capsys
):
    code = _run_script(
        monkeypatch, _base(tmp_path, "--to-zarr", "--session-start", "03:59")
    )
    out = capsys.readouterr()
    assert code == 2
    assert "04:00" in out.err and "20:00" in out.err
    assert wrds.connections == 0
    assert wrds.count_calls == [] and wrds.copy_calls == []


@pytest.mark.parametrize(
    "args, needle",
    [
        (["--universe", "nasdaq_all", *WINDOW], "invalid choice"),
        (
            ["--symbols", "AAPL", *WINDOW, "--rows-per-symbol-day", "100"],
            "server-side",
        ),
        (["--symbols", "AAPL", "--start-date", "2024-01-24"], "--end-date"),
        (["--symbols", "BRK-B", *WINDOW], "dot notation"),
        (["--symbols", "AAPL", "--universe", "sp500", *WINDOW], "Exactly one"),
    ],
)
def test_parser_refusals(wrds, tmp_path, monkeypatch, capsys, args, needle):
    code = _run_script(monkeypatch, [*args, "--data-dir", str(tmp_path)])
    out = capsys.readouterr()
    assert code == 2
    assert needle in out.err
    assert wrds.connections == 0


def test_universe_resolves_by_interval_overlap(wrds, tmp_path, monkeypatch, capsys):
    from quantlab.acquisition.universe import UniverseCatalog

    calls = []

    class _StubCatalog:
        def get_symbols_in_range(self, category, start, end):
            calls.append((category, start, end))
            return ["AAPL", "BRK.B"]

        def get_symbols_as_of(self, *args):  # pragma: no cover - must not run
            raise AssertionError("as-of resolution used for a window backfill")

    loaded = []

    def _load(cls, config):
        loaded.append(config)
        return _StubCatalog()

    monkeypatch.setattr(UniverseCatalog, "load", classmethod(_load))
    code = _run_script(
        monkeypatch, ["--universe", "sp500", *WINDOW, "--data-dir", str(tmp_path)]
    )
    out = capsys.readouterr()
    assert code == 0, out.err
    assert calls == [("sp500_constituent", "2024-01-24", "2024-01-25")]
    assert loaded[0].output_path == str(
        tmp_path / "data" / "reference" / "universe.parquet"
    )
    assert wrds.copy_calls
