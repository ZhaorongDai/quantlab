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


# ---------------------------------------------------------------------------
# Task 2: structural locks on the script
# ---------------------------------------------------------------------------


def _tree() -> ast.Module:
    return ast.parse(SCRIPT.read_text(encoding="utf-8"))


def _main_block(tree: ast.Module) -> ast.If:
    for node in tree.body:
        if (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "__name__"
        ):
            return node
    raise AssertionError(f"{SCRIPT} has no `if __name__ == '__main__':` block")


def _call_name(call: ast.Call) -> str | None:
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _call_linenos(node: ast.AST, name: str) -> list[int]:
    return sorted(
        call.lineno
        for call in ast.walk(node)
        if isinstance(call, ast.Call) and _call_name(call) == name
    )


def test_the_wrds_guard_is_called_by_name_in_main():
    lines = _call_linenos(_main_block(_tree()), "assert_acquisition_volume_fits")
    assert lines, (
        f"{SCRIPT.name}: no assert_acquisition_volume_fits(...) call in the "
        f"__main__ body; the SQL volume guard must run before the pull"
    )


def test_the_wrds_guard_precedes_run_and_any_acquisition_construction():
    main = _main_block(_tree())
    guard = _call_linenos(main, "assert_acquisition_volume_fits")
    probe = _call_linenos(main, "count_rows_by_day")
    runs = _call_linenos(main, "run")
    constructions = _call_linenos(main, "acquisition_cls")
    assert guard and probe and runs, (
        f"guard lines {guard}, probe lines {probe}, run lines {runs}: each "
        f"must be present in __main__"
    )
    assert max(probe) < min(guard), (
        f"the count probe (lines {probe}) must feed the guard (lines {guard})"
    )
    later = [line for line in runs + constructions if line <= max(guard)]
    assert not later, (
        f"the guard (lines {guard}) must precede every run(...) (lines {runs}) "
        f"and acquisition construction (lines {constructions}); offending: {later}"
    )


def test_force_volume_is_a_typed_flag_only():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "force=args.force_volume" in source
    assert "add_volume_guard_args" in source
    for forbidden in (
        "FORCE_VOLUME",
        "getenv",
        'environ.get("FORCE',
        "environ.get('FORCE",
    ):
        lines = [
            number
            for number, line in enumerate(source.splitlines(), start=1)
            if forbidden in line
        ]
        assert not lines, f"{forbidden!r} appears at lines {lines}"


def test_the_script_names_no_vendor_class_and_calls_config_factory_once():
    source = SCRIPT.read_text(encoding="utf-8")
    vendor_lines = [
        number
        for number, line in enumerate(source.splitlines(), start=1)
        if "WrdsTaqNbboAcquisition" in line
    ]
    assert not vendor_lines, f"vendor class named at lines {vendor_lines}"
    factory = _call_linenos(_tree(), "config_factory")
    assert len(factory) == 1, f"config_factory(...) called at lines {factory}"


def _quantlab_config_factories() -> set[str]:
    module = ast.parse(
        (REPO_ROOT / "quantlab" / "config" / "__init__.py").read_text(
            encoding="utf-8"
        )
    )
    return {
        node.name for node in module.body if isinstance(node, ast.FunctionDef)
    } - {"get_data_root", "set_data_root"}


def test_the_script_calls_no_quantlab_config_factory():
    factories = _quantlab_config_factories()
    assert "universe_config" in factories, "the factory scan found nothing"
    tree = _tree()
    called = sorted(
        (_call_name(call), call.lineno)
        for call in ast.walk(tree)
        if isinstance(call, ast.Call) and _call_name(call) in factories
    )
    imported = sorted(
        (alias.name, node.lineno)
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
        if alias.name in factories
    )
    assert not called and not imported, (
        f"quantlab/config factory calls {called} / imports {imported}; "
        f"construct the configs directly"
    )


def test_apply_data_dir_precedes_every_path_derivation():
    main = _main_block(_tree())
    apply = _call_linenos(main, "apply_data_dir")
    derivations = _call_linenos(main, "get_data_root") + _call_linenos(
        main, "config_factory"
    )
    assert apply, "apply_data_dir(args) is not called in __main__"
    assert derivations, "no get_data_root()/config_factory call found in __main__"
    early = [line for line in derivations if line <= min(apply)]
    assert not early, (
        f"apply_data_dir (line {min(apply)}) must precede every "
        f"get_data_root()/config_factory call; offending lines {early}"
    )


def test_no_credential_argument():
    offending = []
    for call in ast.walk(_tree()):
        if isinstance(call, ast.Call) and _call_name(call) == "add_argument":
            for arg in call.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    flag = arg.value.lower()
                    if any(
                        word in flag
                        for word in ("password", "passwd", "user", "pgpass")
                    ):
                        offending.append((arg.value, call.lineno))
    assert not offending, f"credential-shaped arguments: {offending}"


def test_one_session_per_run():
    tree = _tree()
    shared = [
        call.lineno
        for call in ast.walk(tree)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "shared"
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "WrdsSession"
    ]
    assert len(shared) == 1, f"WrdsSession.shared() called at lines {shared}"
    in_finally = [
        call.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Try)
        for stmt in node.finalbody
        for call in ast.walk(stmt)
        if isinstance(call, ast.Call) and _call_name(call) == "close_shared"
    ]
    assert in_finally, "WrdsSession.close_shared() is not called inside a finally"
