"""`scripts/ingest_wrds_crsp.py`: the WRDS CRSP Stock v2 daily ingest CLI (03.10-10).

Two groups, the shape `tests/test_ingest_wrds_taq.py` established in 03.9-07:

- end-to-end runs of the REAL script through `runpy.run_path` against
  `FakeCrspSession` (the `mock_crsp_session` fixture patches
  `quantlab.acquisition.wrds.taq.WrdsSession`, which every WRDS provider and
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
    # ONE sidecar path, not three. The symbology report was dropped when
    # 03.11-07 deleted the machinery that produced it, and the adjustment
    # anchor sidecar when 03.12-02 deleted the window-anchor machinery -- in
    # both cases the script stopped printing the path in the same commit, so
    # listing either here would assert a line no code can emit.
    for suffix in (".crsp_filter_report.json",):
        assert f"{store}{suffix}" in out.out, suffix

    panel = xr.open_zarr(store).load()
    # The axis is the int64 PERMNO (D-01), so the column is 14593, not "AAPL".
    assert [str(value) for value in panel["symbol"].values] == [AAPL_PERMNO]

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
    from quantlab.acquisition._support.sql_volume import SqlVolumeGuard

    monkeypatch.setattr(SqlVolumeGuard, "MAX_RAW_ROWS", 1)
    code = _run_script(monkeypatch, _permno_args(tmp_path))
    out = capsys.readouterr()
    assert code != 0
    assert "--force-volume" in out.err
    assert _daily_copies(mock_crsp_session) == []


def test_force_volume_proceeds_and_says_so(
    mock_crsp_session, tmp_path, monkeypatch, capsys
):
    from quantlab.acquisition._support.sql_volume import SqlVolumeGuard

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
    # On the PERMNO axis (D-01) the ETF's absence is spelled 86755, not "QQQ".
    assert QQQ_PERMNO not in [str(value) for value in equity["symbol"].values]

    benchmark = tmp_path / "data" / "us_equity" / "1d" / "wrds_crsp_qqq_1d.zarr"
    assert str(benchmark) in out.out
    assert [
        str(value) for value in xr.open_zarr(benchmark).load()["symbol"].values
    ] == [QQQ_PERMNO]


#: SPY's PERMNO (`quantlab.base.config.SPY_PERMNO`), restated for the same
#: reason as QQQ's.
SPY_PERMNO = "84398"


def test_benchmark_adds_the_sp500_etf_by_permno_to_the_roster(
    mock_crsp_session, tmp_path, monkeypatch, capsys
):
    """`--universe crsp_sp500 --benchmark` pulls SPY (84398) beside the members."""
    code = _run_script(
        monkeypatch, _universe_args(tmp_path, "crsp_sp500", "--benchmark")
    )
    out = capsys.readouterr()
    assert code == 0, out.err
    roster_line = next(
        line for line in out.out.splitlines() if line.startswith("Roster (")
    )
    assert SPY_PERMNO in roster_line and AAPL_PERMNO in roster_line
    assert QQQ_PERMNO not in roster_line


def test_benchmark_adds_the_nasdaq100_etf_by_permno_to_the_roster(
    mock_crsp_session, tmp_path, monkeypatch, capsys
):
    """`--universe comp_nasdaq100 --benchmark` pulls QQQ (86755), not SPY.

    Writing the ETF's own store is the loop `--qqq` also runs, locked by the
    QQQ store tests below.
    """
    code = _run_script(
        monkeypatch,
        [
            "--universe", "comp_nasdaq100", "--benchmark",
            "--start-date", "2015-01-01", "--end-date", "2015-12-31",
            "--data-dir", str(tmp_path),
        ],
    )
    out = capsys.readouterr()
    assert code == 0, out.err
    roster_line = next(
        line for line in out.out.splitlines() if line.startswith("Roster (")
    )
    assert QQQ_PERMNO in roster_line and "90319" in roster_line
    assert SPY_PERMNO not in roster_line


def test_etf_benchmark_configs_select_the_etf_alone():
    from quantlab.base.config import CrspDatasetConfig

    paths = dict(
        zarr_file_path="spy.zarr", raw_data_dir_path="raw/wrds", reference_dir="ref"
    )
    spy = CrspDatasetConfig.etf_benchmark(permno=SPY_PERMNO, **paths)
    assert (spy.permnos, spy.security_filter) == ((SPY_PERMNO,), "none")
    assert CrspDatasetConfig.qqq_benchmark(**paths) == CrspDatasetConfig.etf_benchmark(
        permno=QQQ_PERMNO, **paths
    )


#: The stable prefix the script prints when the equity roster holds no equity
#: PERMNO. Restated here so a reworded line fails this test rather than silently
#: turning the skip back into an unreported one (GAP-D).
SKIP_PREFIX = "Skipping the equity conversion:"


def _plant_stale_equity_store(tmp_path, *, end_date="2020-12-31"):
    """The trap the live `--qqq --to-zarr` run fell into (GAP-2 / GAP-D).

    A `wrds_crsp_custom_1d.zarr` left by an EARLIER run under a DIFFERENT
    window. The defect was that a QQQ-only run touched that store at all,
    because an empty equity roster fell back to the `custom` store name and
    converted the whole raw tier into it.

    The assertion this helper serves is "the planted files were left UNTOUCHED"
    -- not "something refused". It is deliberately independent of WHICH guard
    would have fired: the anchor-window gate that used to refuse here was
    deleted in 03.12-02, and this trap still has to hold, because the guard
    under test is the empty-roster skip, not the conversion's own defences.

    Returns `(store_path, store_entries)` so the caller can prove the planted
    directory is byte-for-byte the one it planted rather than merely present.
    """
    store = _equity_store(tmp_path)
    store.mkdir(parents=True, exist_ok=True)
    (store / "zarr.json").write_text('{"planted": true}', encoding="utf-8")
    return store, sorted(path.name for path in store.iterdir())


def test_qqq_alone_writes_only_the_benchmark_store_over_a_stale_custom_store(
    mock_crsp_session, tmp_path, monkeypatch, capsys
):
    """GAP-D: `--qqq` with no roster flag writes the BENCHMARK store and nothing else.

    This command has never produced `wrds_crsp_qqq_1d.zarr`. With `--qqq` and no
    `--permnos`/`--universe` the roster is exactly the QQQ PERMNO, so
    `equity_permnos` is the EMPTY tuple -- and the equity conversion ran anyway,
    under the fallback store name `custom`, against a store an earlier run had
    written under a different window. The conversion's own defences refused it
    and the run died BEFORE the QQQ block, which is why truths 08-T6 and 10-T6
    both claim a store that is not on disk.

    The stale store is PLANTED rather than assumed absent: "no equity store was
    written" is proved by the planted bytes being unchanged, so a conversion
    that ran and happened to fail cannot pass this test either.
    """
    import xarray as xr

    store, planted_entries = _plant_stale_equity_store(tmp_path)
    planted_bytes = (store / "zarr.json").read_bytes()

    code = _run_script(
        monkeypatch,
        [
            "--qqq",
            "--start-date", "1999-01-01", "--end-date", PRODUCT_END,
            "--data-dir", str(tmp_path), "--to-zarr",
        ],
    )
    out = capsys.readouterr()
    assert code == 0, out.err

    # The skip is REPORTED, not silent: an empty roster is an outcome the
    # operator has to be able to see in the log.
    assert SKIP_PREFIX in out.out, out.out

    # The benchmark store exists and holds exactly the one symbol -- the int64
    # PERMNO 86755 (D-01), which is what the axis carries now.
    benchmark = tmp_path / "data" / "us_equity" / "1d" / "wrds_crsp_qqq_1d.zarr"
    assert benchmark.exists(), out.out
    assert [
        str(value) for value in xr.open_zarr(benchmark).load()["symbol"].values
    ] == [QQQ_PERMNO]

    # The planted equity store was not written into -- byte-unchanged contents
    # and no new files in the directory.
    assert (store / "zarr.json").read_bytes() == planted_bytes
    assert sorted(path.name for path in store.iterdir()) == planted_entries


def test_a_universe_with_qqq_writes_all_three_stores(
    mock_crsp_session, tmp_path, monkeypatch, capsys
):
    """The MIRROR IMAGE of the test above: a NON-empty equity roster still converts.

    The GAP-D guard skips the equity path when its roster is empty. This asserts
    it is scoped to that condition -- with `--universe` supplying equity PERMNOs
    and `--qqq` supplying the benchmark, all three outputs are written and QQQ is
    still absent from the equity panel (D-15). Without this the guard could widen
    to "skip whenever --qqq is passed" and no test would notice.

    The WIDE window is not cosmetic: `crsp_fixtures.QQQ_ROWS` carries three days
    (1999-03-10, 2010-06-01, 2025-12-31), so the August-2020 window every other
    universe test uses leaves the benchmark conversion with an empty timestamp
    axis, which `TimeChunkPlanner` refuses by design.
    """
    import xarray as xr

    code = _run_script(
        monkeypatch,
        [
            "--universe", "crsp_sp500", "--qqq",
            "--start-date", "1999-01-01", "--end-date", PRODUCT_END,
            "--data-dir", str(tmp_path), "--to-zarr",
        ],
    )
    out = capsys.readouterr()
    assert code == 0, out.err
    assert SKIP_PREFIX not in out.out, out.out

    equity = xr.open_zarr(_equity_store(tmp_path, "sp500")).load()
    # D-15 on the PERMNO axis: the ETF's PERMNO is what must be absent.
    assert QQQ_PERMNO not in [str(value) for value in equity["symbol"].values]

    benchmark = tmp_path / "data" / "us_equity" / "1d" / "wrds_crsp_qqq_1d.zarr"
    assert [
        str(value) for value in xr.open_zarr(benchmark).load()["symbol"].values
    ] == [QQQ_PERMNO]

    membership = (
        tmp_path / "data" / "us_equity" / "1d" / "wrds_crsp_sp500_membership.zarr"
    )
    assert "is_member" in xr.open_zarr(membership).load().data_vars


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
    # The membership panel moved onto the PERMNO axis with the price panel
    # (03.11-05), so membership is asserted by PERMNO.
    assert AAPL_PERMNO in [str(value) for value in panel["symbol"].values]

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
        (["--benchmark", *AUG_2020], "--benchmark pulls the ETF"),
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


# ---------------------------------------------------------------------------
# Task 3: structural locks on the script
#
# These are the protections an end-to-end test CANNOT see. A run whose guard
# had moved BELOW the pull would still pass every test above -- the fake
# always admits -- while a real over-ceiling pull would have already moved
# bytes by the time it was refused. The ordering, the single session, the
# absent credential argument and the absent factory call are therefore
# asserted on the SOURCE, and every failure message names the line numbers it
# compared so the fix is obvious from the report alone.
#
# The helpers below are re-implemented rather than imported from
# `tests/test_ingest_wrds_taq.py`: that is another plan's test module, and
# importing its private helpers would tie two locks that must be free to move
# independently.
# ---------------------------------------------------------------------------


def _source() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def _tree() -> ast.Module:
    return ast.parse(_source())


def _main_body(tree: ast.Module) -> ast.If:
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


def test_the_crsp_guard_is_called_by_name_in_main():
    lines = _call_linenos(_main_body(_tree()), "assert_acquisition_volume_fits")
    assert lines, (
        f"{SCRIPT.name}: no assert_acquisition_volume_fits(...) call in the "
        f"__main__ body; the SQL volume guard must run before the pull"
    )


def test_the_crsp_guard_follows_the_probe_and_precedes_run():
    main = _main_body(_tree())
    guard = _call_linenos(main, "assert_acquisition_volume_fits")
    probe = _call_linenos(main, "count_rows_by_year")
    runs = _call_linenos(main, "run")
    constructions = _call_linenos(main, "acquisition_cls") + _call_linenos(
        main, "ACQ"
    )
    assert guard and probe and runs, (
        f"guard lines {guard}, probe lines {probe}, run lines {runs}: each "
        f"must be present in __main__"
    )
    assert max(probe) < min(guard), (
        f"the per-year count probe (lines {probe}) must feed the guard "
        f"(lines {guard})"
    )
    later = [line for line in runs + constructions if line <= max(guard)]
    assert not later, (
        f"the guard (lines {guard}) must precede every run(...) (lines {runs}) "
        f"and every acquisition construction (lines {constructions}); "
        f"offending: {later}"
    )


def test_the_reference_pull_precedes_the_roster_and_the_probe():
    """The roster CANNOT be resolved before the tables that answer it.

    `WrdsCrspDailyAcquisition._assert_permnos` refuses the whole run on a
    ticker roster (03.10-03), so a universe must be resolved to PERMNOs before
    it reaches the config -- and it can only be resolved from the reference
    tier. The probe then prices exactly that roster.
    """
    main = _main_body(_tree())
    pull = _call_linenos(main, "pull")
    roster = _call_linenos(main, "permnos_in_range")
    probe = _call_linenos(main, "count_rows_by_year")
    assert pull and roster and probe, (
        f"pull lines {pull}, permnos_in_range lines {roster}, "
        f"count_rows_by_year lines {probe}: each must be present in __main__"
    )
    assert max(pull) < min(roster), (
        f"the reference pull (lines {pull}) must precede the roster "
        f"resolution (lines {roster})"
    )
    assert max(roster) < min(probe), (
        f"the roster (lines {roster}) must precede the volume probe "
        f"(lines {probe}); the probe prices exactly the roster that will pull"
    )


def test_force_volume_is_a_typed_flag_only():
    source = _source()
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


def test_the_crsp_cli_names_no_acquisition_class_and_calls_one_factory():
    """The vendor class is RESOLVED, never named (D-12).

    The name is read off the registry rather than written here, so renaming
    the class cannot quietly turn this lock into a check for a string nothing
    uses any more.
    """
    from quantlab.registry import DataSourceRegistry

    acquisition_name = DataSourceRegistry.get("wrds").acquisition_cls_for(
        "us_equity", "1d", "crsp_daily"
    ).__name__
    source = _source()
    named = [
        number
        for number, line in enumerate(source.splitlines(), start=1)
        if acquisition_name in line
    ]
    assert not named, (
        f"{acquisition_name} is named at lines {named}; resolve it through "
        f"SOURCE.acquisition_cls_for(*CAPABILITY) instead"
    )
    assert "acquisition_cls_for" in source

    factory = _call_linenos(_tree(), "config_factory_for")
    assert len(factory) == 1, (
        f"config_factory_for(...) called at lines {factory}; the acquisition "
        f"config is built exactly once"
    )


def _quantlab_config_factories() -> set[str]:
    module = ast.parse(
        (REPO_ROOT / "quantlab" / "config" / "__init__.py").read_text(
            encoding="utf-8"
        )
    )
    return {
        node.name for node in module.body if isinstance(node, ast.FunctionDef)
    } - {"get_data_root", "set_data_root"}


def test_the_crsp_cli_calls_no_quantlab_config_factory():
    """D-12, and a standing user instruction: configs are constructed, never
    fetched from the hardcoded-absolute-path factories in quantlab/config."""
    factories = _quantlab_config_factories()
    assert "stock_kline_config" in factories, "the factory scan found nothing"
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
    main = _main_body(_tree())
    apply = _call_linenos(main, "apply_data_dir")
    derivations = _call_linenos(main, "get_data_root") + _call_linenos(
        main, "config_factory_for"
    )
    assert apply, "apply_data_dir(args) is not called in __main__"
    assert derivations, (
        "no get_data_root()/config_factory_for call found in __main__"
    )
    early = [line for line in derivations if line <= min(apply)]
    assert not early, (
        f"apply_data_dir (line {min(apply)}) must precede every "
        f"get_data_root()/config_factory_for call; offending lines {early}"
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
    assert len(shared) == 1, (
        f"WrdsSession.shared() called at lines {shared}; one WRDS connection "
        f"per run (D-20), because every extra one can push a Duo prompt"
    )
    in_finally = [
        call.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Try)
        for stmt in node.finalbody
        for call in ast.walk(stmt)
        if isinstance(call, ast.Call) and _call_name(call) == "close_shared"
    ]
    assert in_finally, "WrdsSession.close_shared() is not called inside a finally"
