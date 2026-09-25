"""`scripts/ingest_wrds_crsp_etf.py`: ETF daily rows by PERMNO, one store per ETF.

End-to-end runs of the real script through `runpy.run_path` against
`FakeCrspSession` (the `mock_crsp_session` fixture), so no test opens a WRDS
connection. Every `quantlab` import lives inside a test body, as in
`tests/test_ingest_wrds_crsp.py`.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "ingest_wrds_crsp_etf.py"

#: The live CRSP product end the fixture rows are written against.
PRODUCT_END = "2025-12-31"
#: `crsp_fixtures.QQQ_ROWS` carries three days: 1999-03-10, 2010-06-01 and
#: 2025-12-31, so the window has to be wide to reach them.
WIDE = ["--start-date", "1999-01-01", "--end-date", PRODUCT_END]

QQQ_PERMNO = "86755"
SPY_PERMNO = "84398"


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


def _store(tmp_path, name: str) -> Path:
    return tmp_path / "data" / "us_equity" / "1d" / f"wrds_crsp_{name}_1d.zarr"


def test_a_known_etf_is_pulled_by_permno_into_its_own_store(
    mock_crsp_session, tmp_path, monkeypatch, capsys
):
    import xarray as xr

    code = _run_script(
        monkeypatch, ["--etf", "qqq", *WIDE, "--data-dir", str(tmp_path)]
    )
    out = capsys.readouterr()
    assert code == 0, out.err

    store = _store(tmp_path, "qqq")
    assert str(store) in out.out
    panel = xr.open_zarr(store).load()
    assert [str(value) for value in panel["symbol"].values] == [QQQ_PERMNO]
    assert {"adjOpen", "adjClose"} <= set(panel.data_vars)
    assert mock_crsp_session.connections == 1
    assert all(QQQ_PERMNO in call["sql"] for call in mock_crsp_session.crsp_copy_calls)


def test_name_equals_permno_names_the_store(
    mock_crsp_session, tmp_path, monkeypatch, capsys
):
    import xarray as xr

    code = _run_script(
        monkeypatch,
        ["--etf", f"nasdaq_etf={QQQ_PERMNO}", *WIDE, "--data-dir", str(tmp_path)],
    )
    out = capsys.readouterr()
    assert code == 0, out.err
    panel = xr.open_zarr(_store(tmp_path, "nasdaq_etf")).load()
    assert [str(value) for value in panel["symbol"].values] == [QQQ_PERMNO]


def test_every_named_etf_is_in_the_download(
    mock_crsp_session, tmp_path, monkeypatch, capsys
):
    """SPY has no fixture rows, so only its place in the pull is checked; its
    missing store is reported by name and fails the run."""
    code = _run_script(
        monkeypatch, ["--etf", "spy,qqq", *WIDE, "--data-dir", str(tmp_path)]
    )
    out = capsys.readouterr()
    assert f"SPY (PERMNO {SPY_PERMNO})" in out.out
    assert f"QQQ (PERMNO {QQQ_PERMNO})" in out.out
    copied = " ".join(call["sql"] for call in mock_crsp_session.crsp_copy_calls)
    assert SPY_PERMNO in copied and QQQ_PERMNO in copied
    assert _store(tmp_path, "qqq").exists()
    assert code == 1
    assert "spy" in out.err.lower()


@pytest.mark.parametrize(
    "etf, needle",
    [
        ("iwm", "not a known ETF"),
        ("x=abc", "is not a PERMNO"),
        ("Bad-Name=123", "must be lowercase"),
        ("spy,spy", "twice"),
        (f"spy,alias={SPY_PERMNO}", "twice"),
        ("", "names no ETF"),
    ],
)
def test_bad_etf_arguments_cost_no_connection(
    mock_crsp_session, tmp_path, monkeypatch, capsys, etf, needle
):
    code = _run_script(
        monkeypatch, ["--etf", etf, *WIDE, "--data-dir", str(tmp_path)]
    )
    out = capsys.readouterr()
    assert code == 2, out.out
    assert needle in out.err
    assert mock_crsp_session.connections == 0


def test_no_credential_argument():
    source = SCRIPT.read_text(encoding="utf-8")
    for flag in ("--password", "--username", "--user", "--token"):
        assert f'"{flag}"' not in source
