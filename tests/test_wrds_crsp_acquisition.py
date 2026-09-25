"""WRDS CRSP Stock v2 daily provider hardening (phase 03.10, plan 03).

Every test here is OFFLINE. The autouse `_forbid_wrds_network` tripwire in
`tests/conftest.py` makes `psycopg2.connect` raise in every test; a test that
exercises the REAL `WrdsSession` installs its own `FakeConnection` factory
with `monkeypatch.setattr("psycopg2.connect", ...)` inside its body, which is
the only sanctioned way past the tripwire (D-28). Nothing here can push Duo.

What plan 02's tracer proved was that ONE PERMNO-month travels the whole path.
What this module pins is everything the tracer did not have to survive:

- **the annual edge (D-01)** -- `crsp_a_stock` is the ANNUAL-update product, so
  its last day is a hard boundary. A window past it is refused (or explicitly
  clipped), never silently emptied, and the refusal costs zero queries;
- **the vintage stamp (D-01)** -- CRSP revises history between annual
  releases, so two vintages must never share one raw tier. The first run
  stamps the product end beside the raw and watermark roots; a later run that
  probes a different one is refused before any COPY;
- **the page contract (D-02/D-19)** -- one page is one calendar year, pinned
  to the same 50 columns and dtypes in every era, unique on
  `(permno, dlycaldt)`, owned by the requested PERMNOs, counted with its own
  WHERE, and resumable at the year that failed. Nothing is de-duplicated,
  filtered or null-filled;
- **the SQL shape (D-03)** -- every data pull is a bare COPY with no ORDER BY
  / GROUP BY / DISTINCT / LIMIT, and each count uses the COPY's own WHERE;
- **the volume probe (D-03/T-03.10-07)** -- the operator can price the pull
  per year, with the exact batching and WHERE the pull will use, before it
  starts.

The scenario rows below are SYNTHETIC and say so: they exist to exercise
multi-year paging, so they are regular and cheap rather than transcribed. The
VERBATIM live rows live in `tests/crsp_fixtures.py` and the tracer uses them;
this module deliberately does not edit that file, because sibling wave-3 plans
read it.
"""

from __future__ import annotations

import json
import os
import re
from datetime import date
from pathlib import Path

import polars as pl
import psycopg2
import pytest
from loguru import logger

from quantlab.acquisition.wrds import crsp, taq
from tests.crsp_fixtures import (
    DSF_V2_SERVER_COLUMNS,
    FakeCrspSession,
    dsf_row,
    run_crsp_pull,
)
from tests.wrds_fixtures import FakeWrdsSession, fake_connect, render_composed

#: The fake WRDS username `mock_crsp_session` plants, restated here for the
#: REAL-session tests, which do not use that fixture.
USER = "test-wrds-user-not-real"
PGPASS_LINE = f"wrds-pgdata.wharton.upenn.edu:9737:wrds:{USER}:SENTINEL-PW\n"

AAPL = "14593"
MSFT = "10107"  # SYNTHETIC use: a second PERMNO, for batching
LEHMAN = "80599"


# -- helpers -------------------------------------------------------------------


@pytest.fixture
def log_records():
    """Every loguru message emitted during the test, as text."""
    records: list[str] = []
    handler = logger.add(lambda message: records.append(str(message)), level=0)
    yield records
    logger.remove(handler)


def _config(tmp_path, permnos, start_date, end_date, kwargs=None):
    """An `AcquisitionConfig` for a CRSP pull rooted at `tmp_path`."""
    import quantlab.config as config

    config.set_data_root(Path(tmp_path))
    return crsp.WrdsCrspDailyAcquisition.build_config(
        symbols=tuple(str(permno) for permno in permnos),
        start_date=start_date,
        end_date=end_date,
        kwargs=kwargs,
    )


def _acquisition(tmp_path, permnos, start_date, end_date, kwargs=None):
    return crsp.WrdsCrspDailyAcquisition(
        _config(tmp_path, permnos, start_date, end_date, kwargs)
    )


def scenario_rows(
    permnos=(AAPL, MSFT), years=(2018, 2019, 2020), days=("03-01", "06-03", "09-04")
) -> list[dict]:
    """SYNTHETIC multi-year daily rows, three per PERMNO per year.

    Invented rather than transcribed, and the docstring says so on purpose:
    what these rows exercise is PAGING -- which year a page covers, which
    batch it belongs to, whether a failure resumes there -- and a regular grid
    makes the expected counts readable at a glance. Every VALUE inside a row
    still comes from `dsf_row`'s live-observed ordinary-day defaults.
    """
    return [
        dsf_row(permno, f"{year}-{day}", dlyprc="100.000000", dlyret="0.010000")
        for permno in permnos
        for year in years
        for day in days
    ]


_BETWEEN = re.compile(r"BETWEEN '([^']+)' AND '([^']+)'")


def between_bounds(text: str) -> tuple[str, str]:
    """The `BETWEEN low AND high` pair of a rendered statement."""
    match = _BETWEEN.search(text)
    assert match is not None, text
    return match.group(1), match.group(2)


def copy_where(text: str) -> str:
    """The WHERE segment of a rendered `COPY (...) TO STDOUT` statement."""
    assert " WHERE " in text, text
    return text.split(" WHERE ", 1)[1].rsplit(") TO STDOUT", 1)[0]


def count_where(text: str) -> str:
    """The WHERE segment of a rendered `SELECT count(*)` statement."""
    assert " WHERE " in text, text
    return text.split(" WHERE ", 1)[1]


# -- D-01: the annual product end ------------------------------------------------


def test_resolve_window_inside_the_product_end_is_returned_unchanged(
    mock_crsp_session,
):
    session = FakeCrspSession.shared()

    assert crsp.WrdsCrspDailyAcquisition.resolve_window(
        session, "2024-01-01", "2025-06-30", clip=False
    ) == (date(2024, 1, 1), date(2025, 6, 30), None)


def test_resolve_window_past_the_product_end_refuses_without_the_clip_knob(
    mock_crsp_session,
):
    session = FakeCrspSession.shared()

    with pytest.raises(crsp.CrspProductEndError) as excinfo:
        crsp.WrdsCrspDailyAcquisition.resolve_window(
            session, "2024-01-01", "2026-06-30", clip=False
        )

    message = str(excinfo.value)
    for token in ("2026-06-30", "2025-12-31", "crsp_a_stock", "clip_to_product_end"):
        assert token in message, (token, message)


def test_resolve_window_with_clip_lowers_the_end_to_the_product_end(
    mock_crsp_session,
):
    session = FakeCrspSession.shared()

    assert crsp.WrdsCrspDailyAcquisition.resolve_window(
        session, "2024-01-01", "2026-06-30", clip=True
    ) == (date(2024, 1, 1), date(2025, 12, 31), date(2025, 12, 31))


def test_resolve_window_start_past_the_product_end_refuses_even_with_clip(
    mock_crsp_session,
):
    """Clipping a window whose START is past the edge would produce an EMPTY
    range, which reads exactly like a roster with no members."""
    session = FakeCrspSession.shared()

    with pytest.raises(crsp.CrspProductEndError) as excinfo:
        crsp.WrdsCrspDailyAcquisition.resolve_window(
            session, "2026-01-05", "2026-06-30", clip=True
        )

    message = str(excinfo.value)
    for token in ("2026-01-05", "2025-12-31", "nothing was downloaded"):
        assert token in message, (token, message)
    assert "annual update" in message.lower(), message


def test_a_download_past_the_product_end_issues_no_query_at_all(
    mock_crsp_session, tmp_path
):
    acq = _acquisition(tmp_path, [AAPL], "2025-06-01", "2026-06-30")

    with pytest.raises(crsp.CrspProductEndError):
        acq.download()

    assert FakeCrspSession.crsp_copy_calls == [], FakeCrspSession.crsp_copy_calls
    assert FakeCrspSession.crsp_count_calls == [], FakeCrspSession.crsp_count_calls
    assert not Path(acq.config.raw_data_dir_path).exists()


def test_clip_to_product_end_clips_the_window_and_warns_naming_both_dates(
    mock_crsp_session, tmp_path, log_records
):
    acq = _acquisition(
        tmp_path, [AAPL], "2025-06-01", "2026-06-30", {"clip_to_product_end": True}
    )

    acq.download()

    assert acq.config.end_date == "2025-12-31", acq.config.end_date
    last_copy = FakeCrspSession.crsp_copy_calls[-1]["sql"]
    assert "AND '2025-12-31'" in last_copy, last_copy

    warnings = "\n".join(log_records)
    assert "2026-06-30" in warnings, warnings
    assert "2025-12-31" in warnings, warnings


# -- D-03: entitlement precedes every query ---------------------------------------


def test_entitlement_is_checked_before_the_product_end_probe(
    mock_crsp_session, tmp_path
):
    """An unusable schema must stop the run BEFORE the `max(dlycaldt)` probe:
    the probe is itself a query against the schema the account cannot read, so
    checking it second would report the entitlement failure as a broken
    session."""
    FakeCrspSession.usable_schemas = set()
    acq = _acquisition(tmp_path, [AAPL], "2020-08-01", "2020-08-31")

    with pytest.raises(taq.WrdsEntitlementError) as excinfo:
        acq.download()

    assert "crsp_a_stock" in str(excinfo.value)
    assert not any(
        "max(" in text.lower() for text in FakeCrspSession.fetch_calls
    ), FakeCrspSession.fetch_calls
    assert FakeCrspSession.crsp_copy_calls == [], FakeCrspSession.crsp_copy_calls


# -- D-01: the vintage stamp -------------------------------------------------------


def test_the_first_run_stamps_the_vintage_beside_the_raw_and_watermark_roots(
    mock_crsp_session, tmp_path
):
    """The stamp is a SIBLING of both roots, never inside either.

    Not inside the watermark root because `CoverageLedger.iter_watermark_symbols`
    reads every `*.json` there as a symbol, so `wrds.json` would become a
    phantom PERMNO; not inside the raw root because `StockDataset._scan_raw`
    walks every file below it.
    """
    cfg, result = run_crsp_pull(tmp_path, [AAPL], "2020-08-01", "2020-08-31")
    assert result.failures == {}, result.failures

    stamp = crsp.WrdsCrspDailyAcquisition.vintage_path_for(cfg)
    assert stamp == Path(cfg.raw_data_dir_path).parent / "_vintage" / "wrds.json"
    assert json.loads(stamp.read_text()) == {"product_end": "2025-12-31"}

    assert Path(cfg.raw_data_dir_path) not in stamp.parents, stamp
    assert Path(cfg.watermark_path) not in stamp.parents, stamp


def test_a_second_vintage_over_one_raw_tier_is_refused_before_any_copy(
    mock_crsp_session, tmp_path
):
    run_crsp_pull(tmp_path, [AAPL], "2020-08-01", "2020-08-31")

    FakeCrspSession.crsp_copy_calls = []
    FakeCrspSession.product_end = date(2026, 12, 31)

    with pytest.raises(crsp.CrspVintageError) as excinfo:
        run_crsp_pull(tmp_path, [AAPL], "2020-08-01", "2020-08-31")

    message = str(excinfo.value)
    for token in ("2025-12-31", "2026-12-31", "subdir", "delete"):
        assert token in message, (token, message)
    assert FakeCrspSession.crsp_copy_calls == [], FakeCrspSession.crsp_copy_calls


# -- D-20/D-03: one connection, one worker -----------------------------------------


def test_max_workers_other_than_one_is_refused_before_a_session_exists(
    mock_crsp_session, tmp_path
):
    with pytest.raises(ValueError, match="max_workers"):
        _acquisition(
            tmp_path, [AAPL], "2020-08-01", "2020-08-31", {"max_workers": 2}
        )

    assert FakeWrdsSession.connections == 0, FakeWrdsSession.connections


def test_a_session_error_aborts_the_run_and_the_next_run_repeats_that_page(
    mock_crsp_session, tmp_path
):
    """A dead session is a GLOBAL stop (D-21), not one PERMNO's fault: the
    failure manifest stays empty and the next run re-issues the same page."""
    FakeCrspSession.daily_rows = scenario_rows()
    FakeCrspSession.raise_on_copy = {
        0: taq.WrdsSessionError("the WRDS session failed")
    }

    _, result = run_crsp_pull(tmp_path, [AAPL], "2019-01-01", "2020-12-31")

    assert result.quota_aborted is True, result
    assert result.failures == {}, result.failures
    first_copy = FakeCrspSession.crsp_copy_calls[0]["sql"]

    FakeCrspSession.raise_on_copy = {}
    run_crsp_pull(tmp_path, [AAPL], "2019-01-01", "2020-12-31")

    assert FakeCrspSession.crsp_copy_calls[1]["sql"] == first_copy


# -- T-03.10-10: the driver error text ---------------------------------------------


def test_a_real_session_driver_error_is_scrubbed_of_the_username(
    monkeypatch, tmp_path
):
    """Asserted through the REAL `WrdsSession`, not the fake.

    The scrub lives in `WrdsSession._query`, so a CRSP call that reached the
    driver by any other route would keep the username in its message. This
    test is the evidence that `CrspQueries.product_end` goes through the
    scrubbing path.
    """
    from quantlab.acquisition.wrds.taq import WrdsSession, WrdsSessionError

    monkeypatch.setenv("WRDS_USERNAME", USER)
    for name in ("PGHOSTADDR", "PGSERVICE", "PGSERVICEFILE"):
        monkeypatch.delenv(name, raising=False)
    pgpass = tmp_path / "pgpass"
    pgpass.write_text(PGPASS_LINE)
    os.chmod(pgpass, 0o600)
    monkeypatch.setenv("PGPASSFILE", str(pgpass))

    connections: list = []
    monkeypatch.setattr("psycopg2.connect", fake_connect(connections))

    WrdsSession.close_shared()
    try:
        session = WrdsSession.shared()
        # Opens the connection, so `fail_with` can be planted on it.
        assert session.schema_usable("crsp_a_stock") is True

        connections[0].fail_with = psycopg2.OperationalError(
            f"FATAL: role {USER} is not permitted to log in"
        )
        with pytest.raises(WrdsSessionError) as excinfo:
            crsp.CrspQueries.product_end(session)
    finally:
        WrdsSession.close_shared()

    message = str(excinfo.value)
    assert "$WRDS_USERNAME" in message, message
    assert USER not in message, message


# -- D-02: one page is one calendar year -------------------------------------------


def test_year_pages_clips_each_calendar_year_to_the_window():
    """The ONE definition of a page's bounds, shared by the acquisition and
    the volume probe -- so the rows the operator is quoted are the rows the
    pull will move."""
    assert crsp.year_pages(date(2018, 6, 1), date(2020, 3, 31)) == [
        (date(2018, 6, 1), date(2018, 12, 31)),
        (date(2019, 1, 1), date(2019, 12, 31)),
        (date(2020, 1, 1), date(2020, 3, 31)),
    ]
    assert crsp.year_pages(date(2020, 3, 1), date(2020, 3, 31)) == [
        (date(2020, 3, 1), date(2020, 3, 31))
    ]
    assert crsp.year_pages(date(2020, 3, 31), date(2020, 3, 1)) == []


def test_the_window_is_paged_one_calendar_year_at_a_time_in_order(
    mock_crsp_session, tmp_path
):
    FakeCrspSession.daily_rows = scenario_rows(permnos=(AAPL,))

    run_crsp_pull(tmp_path, [AAPL], "2018-06-01", "2020-03-31")

    assert [
        between_bounds(call["sql"]) for call in FakeCrspSession.crsp_copy_calls
    ] == [
        ("2018-06-01", "2018-12-31"),
        ("2019-01-01", "2019-12-31"),
        ("2020-01-01", "2020-03-31"),
    ]


def test_every_page_lands_the_pinned_schema_and_leaves_a_null_return_null(
    mock_crsp_session, tmp_path
):
    """The SELECT is pinned, so every shard of every era carries the same 50
    columns with the same dtypes -- and an absent `dlyret` stays NULL rather
    than becoming a 0% return, which would be a silently wrong number."""
    acq_cls = crsp.WrdsCrspDailyAcquisition
    rows = scenario_rows(permnos=(AAPL,), years=(2019, 2020))
    # `dsf_row` leaves `dlyret` NULL unless asked; this row is the one the
    # assertion below reads.
    rows.append(dsf_row(AAPL, "2019-12-02", dlyprc="101.000000"))
    FakeCrspSession.daily_rows = rows

    cfg, result = run_crsp_pull(tmp_path, [AAPL], "2019-01-01", "2020-12-31")
    assert result.failures == {}, result.failures

    shards = sorted(Path(cfg.raw_data_dir_path).rglob("*.pqt"))
    assert len(shards) >= 2, shards
    for shard in shards:
        frame = pl.read_parquet(shard)
        assert tuple(frame.columns) == acq_cls.RAW_COLUMNS, frame.columns
        for name, dtype in acq_cls.RAW_SCHEMA.items():
            assert frame.schema[name] == dtype, (shard, name, frame.schema[name])

    combined = pl.concat([pl.read_parquet(shard) for shard in shards])
    assert combined.schema["permno"] == pl.Int64
    assert combined.schema["dlycaldt"] == pl.Date
    assert combined.schema["dlyprc"] == pl.Float64
    null_day = combined.filter(pl.col("dlycaldt") == date(2019, 12, 2))
    assert null_day.height == 1, null_day
    assert null_day["dlyret"].to_list() == [None], null_day


def test_a_failed_page_resumes_at_that_year_on_the_next_run(
    mock_crsp_session, tmp_path
):
    FakeCrspSession.daily_rows = scenario_rows(permnos=(AAPL,))
    FakeCrspSession.raise_on_copy = {1: RuntimeError("boom")}

    cfg, result = run_crsp_pull(tmp_path, [AAPL], "2018-06-01", "2020-03-31")

    assert AAPL in result.failures, result.failures
    shards = [str(path) for path in Path(cfg.raw_data_dir_path).rglob("*.pqt")]
    assert any("month=2018-" in path for path in shards), shards

    FakeCrspSession.raise_on_copy = {}
    run_crsp_pull(tmp_path, [AAPL], "2018-06-01", "2020-03-31")

    resumed = [
        between_bounds(call["sql"])
        for call in FakeCrspSession.crsp_copy_calls[2:]
    ]
    assert resumed[0] == ("2019-01-01", "2019-12-31"), resumed


# -- D-19: a malformed page is refused, never repaired -----------------------------


def test_a_duplicated_permno_and_day_fails_the_batch_and_writes_no_shard(
    mock_crsp_session, tmp_path
):
    """`dsf_v2` is unique on `(permno, dlycaldt)`. A duplicate means the table
    stopped being what D-19 verified, so the page is refused -- a silent dedup
    would drop one of two prices with nothing recording the choice."""
    rows = scenario_rows(permnos=(AAPL,), years=(2020,))
    rows.append(dsf_row(AAPL, "2020-08-31", dlyprc="129.040000"))
    rows.append(dsf_row(AAPL, "2020-08-31", dlyprc="129.050000"))
    FakeCrspSession.daily_rows = rows

    cfg, result = run_crsp_pull(tmp_path, [AAPL], "2020-01-01", "2020-12-31")

    message = result.failures[AAPL]
    assert AAPL in message, message
    assert "2020-08-31" in message, message
    assert sorted(Path(cfg.raw_data_dir_path).rglob("*.pqt")) == []


def test_a_stranger_permno_in_a_page_fails_the_batch(mock_crsp_session, tmp_path):
    FakeCrspSession.daily_rows = scenario_rows(permnos=(AAPL,), years=(2020,))
    FakeCrspSession.inject_rows = [dsf_row(99999, "2020-06-03")]

    _, result = run_crsp_pull(tmp_path, [AAPL], "2020-01-01", "2020-12-31")

    assert "99999" in result.failures[AAPL], result.failures


def test_a_row_dated_outside_the_page_year_fails_the_batch(
    mock_crsp_session, tmp_path
):
    """A row outside the page bounds would land under a `month=` partition the
    page does not own -- where the next run's deterministic overwrite would
    never reach it."""
    FakeCrspSession.daily_rows = scenario_rows(permnos=(AAPL,), years=(2020,))
    FakeCrspSession.inject_rows = [dsf_row(AAPL, "2021-01-04")]

    _, result = run_crsp_pull(tmp_path, [AAPL], "2020-01-01", "2020-12-31")

    assert "2021-01-04" in result.failures[AAPL], result.failures


def test_a_count_mismatch_fails_the_page_naming_both_row_counts(
    mock_crsp_session, tmp_path
):
    FakeCrspSession.daily_rows = scenario_rows(permnos=(AAPL,), years=(2020,))
    FakeCrspSession.count_adjust = {"2020-01-01": 1}

    _, result = run_crsp_pull(tmp_path, [AAPL], "2020-01-01", "2020-12-31")

    message = result.failures[AAPL]
    assert "3 row(s)" in message, message
    assert "reported 4" in message, message
    assert "2020" in message, message


def test_a_missing_pinned_column_fails_the_page_before_any_copy(
    mock_crsp_session, tmp_path
):
    FakeCrspSession.daily_rows = scenario_rows(permnos=(AAPL,), years=(2020,))
    FakeCrspSession.server_columns = {
        "crsp_a_stock.dsf_v2": tuple(
            name for name in DSF_V2_SERVER_COLUMNS if name != "dlycumfacshr"
        )
    }

    _, result = run_crsp_pull(tmp_path, [AAPL], "2020-01-01", "2020-12-31")

    message = result.failures[AAPL]
    assert "dlycumfacshr" in message, message
    assert "dsf_v2" in message, message
    assert FakeCrspSession.crsp_copy_calls == [], FakeCrspSession.crsp_copy_calls


# -- D-03: the SQL shape -----------------------------------------------------------


def test_every_crsp_statement_keeps_the_bare_copy_sql_shape(
    mock_crsp_session, tmp_path
):
    FakeCrspSession.daily_rows = scenario_rows(permnos=(AAPL, MSFT))

    run_crsp_pull(tmp_path, [AAPL, MSFT], "2018-06-01", "2020-03-31")

    copies = [call["sql"] for call in FakeCrspSession.crsp_copy_calls]
    counts = [call["sql"] for call in FakeCrspSession.crsp_count_calls]
    assert len(copies) == 3, copies
    assert len(counts) == 3, counts

    for text in copies:
        assert text.startswith("COPY (SELECT "), text
        assert 'FROM "crsp_a_stock"."dsf_v2" WHERE ' in text, text
    for text in copies + counts:
        for forbidden in ("ORDER BY", "GROUP BY", "DISTINCT", "LIMIT"):
            assert forbidden not in text.upper(), (forbidden, text)

    # The count a page is checked against must select EXACTLY the rows the
    # COPY does, or the check is theatre.
    assert [count_where(text) for text in counts] == [
        copy_where(text) for text in copies
    ]


def test_a_ticker_roster_is_refused_because_crsp_symbols_are_permnos(
    mock_crsp_session, tmp_path
):
    acq = _acquisition(tmp_path, [AAPL], "2020-01-01", "2020-12-31")

    with pytest.raises(ValueError) as excinfo:
        acq.download(["AAPL"])

    message = str(excinfo.value)
    assert "AAPL" in message, message
    assert "PERMNO" in message, message
    assert FakeCrspSession.fetch_calls == [], FakeCrspSession.fetch_calls
    assert FakeCrspSession.crsp_copy_calls == [], FakeCrspSession.crsp_copy_calls


def test_daily_where_refuses_an_empty_permno_list():
    """The one statement this class must never be able to build: a query over
    a 110-million-row table with no PERMNO predicate."""
    with pytest.raises(ValueError, match="PERMNO"):
        crsp.CrspQueries.daily_where([], date(2020, 1, 1), date(2020, 12, 31))


def test_the_daily_table_constant_is_the_one_line_d19_fallback_switch(
    mock_crsp_session, tmp_path, monkeypatch
):
    """D-19's fallback is `stkdlysecuritydata`, and switching to it must be a
    single constant edit -- so both the builders AND `_fetch_page` have to
    read `CrspQueries.DAILY_TABLE` at call time."""
    fallback = "stkdlysecuritydata"
    monkeypatch.setattr(crsp.CrspQueries, "DAILY_TABLE", fallback)
    FakeCrspSession.server_columns = {
        f"crsp_a_stock.{fallback}": DSF_V2_SERVER_COLUMNS
    }

    acq = _acquisition(tmp_path, [AAPL], "2020-01-01", "2020-12-31")
    frame, next_token = acq._fetch_page([AAPL], "2020-01-01", "2020-12-31")

    assert next_token is None
    assert frame.height == 0
    assert any(
        f'"crsp_a_stock"."{fallback}"' in text
        for text in FakeCrspSession.fetch_calls
    ), FakeCrspSession.fetch_calls

    rendered = render_composed(
        crsp.CrspQueries.copy_query(
            crsp.CrspQueries.STOCK_SCHEMA,
            crsp.CrspQueries.DAILY_TABLE,
            crsp.WrdsCrspDailyAcquisition.CRSP_COLUMNS,
            crsp.CrspQueries.daily_where(
                [AAPL], date(2020, 1, 1), date(2020, 12, 31)
            ),
        )
    )
    assert f'"crsp_a_stock"."{fallback}"' in rendered, rendered
