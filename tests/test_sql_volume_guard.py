"""SqlVolumeGuard: the WRDS-shaped pre-flight volume guard (D-16, D-24).

The REST requests-per-minute model of `UniverseCatalog.assert_acquisition_volume_fits`
does not apply to a PostgreSQL pull. This guard prices a fetch from real per-day
`count(*)` row counts and refuses above a byte ceiling and a row ceiling unless
forced. Pure arithmetic: every test here runs with no network and no credential.
"""

import ast
import inspect
from pathlib import Path

import pytest

from quantlab.acquisition._support.sql_volume import SqlVolumeGuard

ROWS_BY_DAY = {"2024-01-24": 1_243_426, "2024-01-25": 1_100_000}
TOTAL_ROWS = 2_343_426
WINDOW = {"symbols": 1, "start_date": "2024-01-24", "end_date": "2024-01-25"}


def _guard(**kwargs) -> SqlVolumeGuard:
    return SqlVolumeGuard(kwargs or None)


def test_estimate_under_default_ceilings_is_admitted():
    guard = _guard()
    estimate = guard.assert_acquisition_volume_fits(ROWS_BY_DAY, **WINDOW)
    assert estimate["rows"] == TOTAL_ROWS
    assert estimate["raw_bytes"] == TOTAL_ROWS * SqlVolumeGuard.DEFAULT_BYTES_PER_ROW
    assert estimate["trading_days"] == 2
    assert estimate["crossed"] == []
    assert estimate["forced"] is False
    assert estimate["fitting_end_date"] == "2024-01-25"
    assert estimate["symbols"] == 1
    assert estimate["start_date"] == "2024-01-24"
    assert estimate["end_date"] == "2024-01-25"


def test_raw_bytes_ceiling_refuses_and_names_a_fitting_segment():
    raw_bytes = TOTAL_ROWS * SqlVolumeGuard.DEFAULT_BYTES_PER_ROW
    guard = _guard(max_raw_bytes=raw_bytes - 1)
    with pytest.raises(ValueError) as excinfo:
        guard.assert_acquisition_volume_fits(ROWS_BY_DAY, **WINDOW)
    message = str(excinfo.value)
    assert "raw-bytes" in message
    assert "raw-rows" not in message
    assert f"{raw_bytes / 1024**3:.2f} GiB" in message
    assert f"{(raw_bytes - 1) / 1024**3:.2f} GiB" in message
    assert "MAX_RAW_BYTES" in message and "max_raw_bytes" in message
    assert "--force-volume" in message
    assert "--end-date 2024-01-24" in message


def test_raw_rows_ceiling_refuses_independently():
    guard = _guard(max_raw_rows=TOTAL_ROWS - 1)
    with pytest.raises(ValueError) as excinfo:
        guard.assert_acquisition_volume_fits(ROWS_BY_DAY, **WINDOW)
    message = str(excinfo.value)
    assert "raw-rows" in message
    assert "raw-bytes" not in message
    assert f"{TOTAL_ROWS:,}" in message
    assert "MAX_RAW_ROWS" in message and "max_raw_rows" in message
    assert "--end-date 2024-01-24" in message


def test_both_ceilings_crossed_are_both_named():
    guard = _guard(max_raw_rows=TOTAL_ROWS - 1, max_raw_bytes=TOTAL_ROWS)
    with pytest.raises(ValueError) as excinfo:
        guard.assert_acquisition_volume_fits(ROWS_BY_DAY, **WINDOW)
    message = str(excinfo.value)
    assert "raw-bytes" in message and "raw-rows" in message


def test_force_skips_the_raise_and_never_the_arithmetic():
    guard = _guard(max_raw_rows=TOTAL_ROWS - 1, max_raw_bytes=TOTAL_ROWS)
    estimate = guard.assert_acquisition_volume_fits(
        ROWS_BY_DAY, **WINDOW, force=True
    )
    assert estimate["forced"] is True
    assert estimate["crossed"] == ["raw-bytes", "raw-rows"]
    assert estimate["rows"] == TOTAL_ROWS
    assert estimate["raw_bytes"] == TOTAL_ROWS * SqlVolumeGuard.DEFAULT_BYTES_PER_ROW


def test_force_on_an_admitted_estimate_is_not_marked_forced():
    estimate = _guard().assert_acquisition_volume_fits(
        ROWS_BY_DAY, **WINDOW, force=True
    )
    assert estimate["crossed"] == []
    assert estimate["forced"] is False


def test_no_day_fits_says_narrow_the_symbols_instead():
    guard = _guard(max_raw_rows=1_000_000)
    assert guard.estimate(ROWS_BY_DAY, **WINDOW)["fitting_end_date"] is None
    with pytest.raises(ValueError) as excinfo:
        guard.assert_acquisition_volume_fits(ROWS_BY_DAY, **WINDOW)
    message = str(excinfo.value)
    assert "no single day fits" in message
    assert "--symbols" in message
    assert "--end-date" not in message


def test_knobs_are_read_from_a_kwargs_mapping():
    guard = SqlVolumeGuard(
        {"max_raw_bytes": 10**12, "max_raw_rows": 10**9, "bytes_per_row": 16}
    )
    estimate = guard.estimate(ROWS_BY_DAY, **WINDOW)
    assert estimate["max_raw_bytes"] == 10**12
    assert estimate["max_raw_rows"] == 10**9
    assert estimate["bytes_per_row"] == 16
    assert estimate["raw_bytes"] == TOTAL_ROWS * 16


@pytest.mark.parametrize("key", ["max_raw_bytes", "max_raw_rows", "bytes_per_row"])
@pytest.mark.parametrize("value", [0, -1, "20", True, None])
def test_non_positive_or_non_numeric_knob_is_refused(key, value):
    with pytest.raises(ValueError, match=key):
        SqlVolumeGuard({key: value})


def test_empty_rows_by_day_is_admitted_with_zero_rows():
    estimate = _guard().assert_acquisition_volume_fits({}, **WINDOW)
    assert estimate["rows"] == 0
    assert estimate["raw_bytes"] == 0
    assert estimate["trading_days"] == 0
    assert estimate["crossed"] == []


def test_unsorted_day_keys_are_ordered_before_the_prefix_search():
    rows_by_day = {"2024-01-26": 10, "2024-01-24": 10, "2024-01-25": 10}
    guard = _guard(max_raw_rows=25)
    estimate = guard.estimate(
        rows_by_day, symbols=1, start_date="2024-01-24", end_date="2024-01-26"
    )
    assert estimate["crossed"] == ["raw-rows"]
    assert estimate["fitting_end_date"] == "2024-01-25"


def test_the_sql_guard_ceiling_is_pinned_to_the_universe_guard():
    from quantlab.universe import UniverseCatalog

    assert SqlVolumeGuard.MAX_RAW_BYTES == UniverseCatalog.MAX_RAW_BYTES
    assert SqlVolumeGuard.MAX_RAW_BYTES == 20 * 1024**3


def test_sql_volume_module_imports_no_acquisition_class():
    import sys

    import quantlab.acquisition._support.sql_volume as module

    tree = ast.parse(Path(inspect.getfile(module)).read_text())
    top_levels = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            top_levels.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "relative import in a leaf module"
            top_levels.add(node.module.split(".")[0])
    forbidden = {"quantlab", "psycopg2", "wrds", "requests", "sqlalchemy"}
    assert not top_levels & forbidden, top_levels
    assert top_levels <= set(sys.stdlib_module_names), top_levels


# ---------------------------------------------------------------------------
# print_sql_volume_estimate (quantlab/utils/cli.py)
# ---------------------------------------------------------------------------


def _captured(estimate, **kwargs) -> str:
    from quantlab.utils.cli import print_sql_volume_estimate

    lines: list[str] = []
    returned = print_sql_volume_estimate(estimate, print_fn=lines.append, **kwargs)
    assert returned is estimate
    return "\n".join(lines)


def test_print_admitted_estimate_shows_rows_and_gib():
    estimate = _guard().assert_acquisition_volume_fits(ROWS_BY_DAY, **WINDOW)
    output = _captured(estimate)
    assert "count(*)" in output
    assert f"{TOTAL_ROWS:,}" in output
    assert f"{estimate['raw_bytes'] / 1024**3:.2f} GiB" in output
    assert "2024-01-24 .. 2024-01-25" in output
    assert "assumption" in output.lower()
    assert "NOT enforced" not in output


def test_print_forced_estimate_names_the_overridden_ceiling():
    estimate = _guard(max_raw_bytes=1).assert_acquisition_volume_fits(
        ROWS_BY_DAY, **WINDOW, force=True
    )
    output = _captured(estimate, forced=True)
    assert "--force-volume" in output
    assert "raw-bytes" in output
    assert "NOT enforced" in output


def test_print_never_renders_a_credential(monkeypatch):
    planted = "planted-wrds-user-7f3c"
    monkeypatch.setenv("WRDS_USERNAME", planted)
    estimate = _guard(max_raw_bytes=1).assert_acquisition_volume_fits(
        ROWS_BY_DAY, **WINDOW, force=True
    )
    assert planted not in _captured(estimate, forced=True)
    assert planted not in _captured(_guard().estimate(ROWS_BY_DAY, **WINDOW))


# ---------------------------------------------------------------------------
# The bucket UNIT (03.10-10 task 1)
#
# The same guard prices two different pulls. TAQ counts per trading DAY,
# because a TAQ page IS a day table. CRSP counts per calendar-YEAR bucket
# (`CrspVolumeProbe.count_rows_by_year`, plan 03), because a CRSP page is a
# calendar year -- so a CRSP refusal that said "trading day(s)" would name a
# boundary no page has. The label is a keyword on the estimate, defaulting to
# the TAQ wording so every existing caller and every existing message is
# byte-identical.
# ---------------------------------------------------------------------------


def test_estimate_defaults_to_the_trading_day_unit():
    estimate = _guard().estimate(ROWS_BY_DAY, **WINDOW)
    assert estimate["unit"] == "trading day"
    # The dict KEY stays `trading_days` for every unit: it is the bucket
    # count, and renaming it would break the TAQ callers the label exists to
    # leave alone.
    assert estimate["trading_days"] == 2


def test_the_default_unit_leaves_the_taq_refusal_text_unchanged():
    guard = _guard(max_raw_rows=TOTAL_ROWS - 1)
    with pytest.raises(ValueError) as excinfo:
        guard.assert_acquisition_volume_fits(ROWS_BY_DAY, **WINDOW)
    assert "2 trading day(s)" in str(excinfo.value)


def test_a_year_bucket_unit_is_carried_into_the_estimate_and_the_refusal():
    guard = _guard(max_raw_rows=TOTAL_ROWS - 1)
    estimate = guard.estimate(ROWS_BY_DAY, **WINDOW, unit="year bucket")
    assert estimate["unit"] == "year bucket"
    assert estimate["trading_days"] == 2
    with pytest.raises(ValueError) as excinfo:
        guard.assert_acquisition_volume_fits(
            ROWS_BY_DAY, **WINDOW, unit="year bucket"
        )
    message = str(excinfo.value)
    assert "2 year bucket(s)" in message
    assert "trading day" not in message


def test_print_renders_the_year_bucket_unit_label():
    estimate = _guard().assert_acquisition_volume_fits(
        ROWS_BY_DAY, **WINDOW, unit="year bucket"
    )
    output = _captured(estimate)
    assert "  year buckets:" in output
    assert "trading days:" not in output
    # Same column as every other label in the block.
    line = next(l for l in output.splitlines() if "year buckets:" in l)
    assert line == f"  {'year buckets:':<19}{2:,}"


def test_print_without_a_unit_key_falls_back_to_trading_days():
    estimate = _guard().assert_acquisition_volume_fits(ROWS_BY_DAY, **WINDOW)
    estimate.pop("unit", None)
    output = _captured(estimate)
    assert "  trading days:" in output
