"""Pre-flight acquisition volume guard (03.2 SC-6, D-09).

A high-frequency request is easy to write and expensive to discover. `us_all`
quotes over a decade is not a slow download, it is a multi-terabyte one, and
the current failure mode is that it starts, runs for hours and fills the disk.
SC-6 requires the refusal to happen BEFORE the client is constructed and before
any request is issued, to name a concrete narrowing (fewer symbols, a shorter
window, a coarser frequency) rather than just saying no, and to be overridable
with an explicit `--force-volume`.

Two grounded constants already live on `acquisition/universe.py:UniverseCatalog`
and the new acquisition-volume estimator is built beside them, sharing their
arithmetic. This module pins both, so an edit to either surfaces HERE -- next to
the guard whose thresholds it silently shifts -- rather than only in the
dense-panel tests it was written for.

`tick` frequency without an explicit `rows_per_symbol_day` must REFUSE rather
than guess: an invented row count produces an invented budget, and the whole
point of the guard is that its number is defensible (house rule: unknown is
represented by absence and never guessed).

Every test here is offline and allocates nothing: the estimator is arithmetic
over roster size, window length and frequency. No network call, no credential,
no real data volume.

This file lands in 03.2-01 (Wave 0) carrying its constant self-test; 03.2-04
Tasks 1-2 fill in the estimator and refusal tests. It is deliberately NOT an
empty placeholder: a pytest file with zero collected tests exits 5 ("no tests
ran"), which a later task's automated command reads as green.
"""


def test_the_grounded_constants_the_new_volume_guard_is_built_beside():
    """Self-test: the two existing `UniverseCatalog` constants the acquisition
    volume estimator shares arithmetic with.

    `MAX_DENSE_PANEL_BYTES` is 4 GiB -- a RAM budget, not a disk one (a dense
    float64 panel is materialised in memory before it is written).
    `TRADING_DAYS_PER_YEAR` is 252, the figure every window-length estimate in
    this codebase multiplies through.

    `UniverseCatalog` is imported INSIDE the test body on purpose. A module-scope
    import of `acquisition.universe` here would make this file a new
    collection-time liability of the same kind `tests/conftest.py`'s docstring
    forbids.
    """
    from acquisition.universe import UniverseCatalog

    assert UniverseCatalog.MAX_DENSE_PANEL_BYTES == 4 * 1024**3, (
        "MAX_DENSE_PANEL_BYTES changed; the acquisition volume guard's budget "
        "arithmetic is derived from it -- update both together, deliberately"
    )
    assert UniverseCatalog.TRADING_DAYS_PER_YEAR == 252, (
        "TRADING_DAYS_PER_YEAR changed; every window-length estimate in the "
        "volume guard multiplies through it"
    )


# ---------------------------------------------------------------------------
# Synthetic catalogs at REAL scale (03.2-04 Tasks 1-2)
#
# The guard's whole claim is arithmetic: "the full-market minute backfill is
# ~597,000 requests / ~358 GB / ~50 h and the S&P-500-minute year is ~4,900 /
# ~3 GB / ~25 min" (03.2-RESEARCH.md Pattern 6, Volume Arithmetic). A 20-row
# toy fixture cannot exercise that claim -- it would only prove the estimator
# multiplies, and every default threshold would be untested because nothing
# could reach one.
#
# So these tests build a reference table at the MEASURED scale: 15,424 `us_all`
# symbols (260906-0iy D-01) shaped to the MEASURED 0.368 daily density
# (`MAX_DENSE_PANEL_BYTES`'s own rationale block), 500 S&P constituents, and two
# smaller rosters used to isolate one ceiling at a time. 17,524 rows of parquet
# costs milliseconds and no network call; what it buys is that every scenario
# below reproduces a row of RESEARCH's table rather than approximating one.
#
# Every symbol in a synthetic category is given the SAME span, anchored at the
# window start. `estimate_dense_panel` reduces to one clipped span per symbol
# and then SUMS the span days, so staggering the starts would change nothing
# about the result while adding clipping edge cases to get wrong.
# ---------------------------------------------------------------------------

#: The measured `us_all` roster size and its measured daily density, quoted from
#: `UniverseCatalog.MAX_DENSE_PANEL_BYTES`'s rationale block ("15,424 symbols x
#: ~5,215 trading days = 80.4M dense cells, of which only ~29.6M are real
#: observations (density 0.368)"). Reproducing BOTH is what makes the scenarios
#: below RESEARCH's arithmetic rather than a toy.
US_ALL_SYMBOLS = 15_424
US_ALL_DENSITY = 0.368

#: RESEARCH Pattern 6's full-market window.
FULL_WINDOW = ("2016-01-01", "2026-09-06")
#: One year, the window RESEARCH sizes the S&P-500 minute scenario over.
ONE_YEAR = ("2025-01-01", "2025-12-31")


def _rows(category: str, count: int, window, span_fraction: float = 1.0) -> list[dict]:
    """`count` synthetic catalog rows in `category`, each listed for
    `span_fraction` of `window`."""
    import datetime

    start, end = window
    window_days = (
        datetime.date.fromisoformat(end) - datetime.date.fromisoformat(start)
    ).days + 1
    span_days = max(round(span_fraction * window_days), 1)
    listed_from = datetime.date.fromisoformat(start)
    listed_to = (listed_from + datetime.timedelta(days=span_days - 1)).isoformat()
    return [
        {
            "symbol": f"{category.upper()[:4]}{index:05d}",
            "category": category,
            "start_date": start,
            "end_date": listed_to,
            "end_date_is_inferred": False,
        }
        for index in range(count)
    ]


def _catalog(tmp_path):
    """A `UniverseCatalog` loaded from a synthetic table at real scale.

    Built through `UniverseCatalog.load()` -- the same read path production
    uses -- rather than by reaching into `_backend`, so the estimator is
    exercised against a real parquet scan. No fetcher runs, so no network call
    is even possible.
    """
    import polars as pl

    from acquisition.universe import UniverseCatalog
    from base.config import UniverseConfig

    rows = (
        _rows("us_all", US_ALL_SYMBOLS, FULL_WINDOW, US_ALL_DENSITY)
        + _rows("sp500_constituent", 500, ONE_YEAR)
        # Sized in `test_each_of_the_three_ceilings_raises_independently` to sit
        # over the byte ceiling and under the other two.
        + _rows("nasdaq100_constituent", 600, ("2019-01-01", "2025-12-31"))
        # Sized to sit under all three by default, so `page_limit` and
        # `rate_limit_per_min` alone decide which ceiling it crosses.
        + _rows("nasdaq_all", 1_000, ("2024-01-01", "2024-12-31"))
    )
    path = tmp_path / "reference" / "universe.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(path)
    return UniverseCatalog.load(
        UniverseConfig(
            output_path=str(path), cache_dir=str(tmp_path / "reference" / "_cache")
        )
    )


def _no_network(monkeypatch) -> None:
    """Make ANY socket allocation raise.

    Stronger than patching `requests.get`: it fails on a connection opened
    through any library by any means, which is what "issues zero vendor
    requests" has to mean if the claim is to survive someone adding an
    httpx-based client later.
    """
    import socket

    def _forbidden(*args, **kwargs):
        raise AssertionError(
            "the volume guard opened a socket; it is a pre-flight estimate and "
            "must cost zero vendor requests"
        )

    monkeypatch.setattr(socket, "socket", _forbidden)
    monkeypatch.setattr(socket, "create_connection", _forbidden)


# ---------------------------------------------------------------------------
# Task 1 -- the estimator
# ---------------------------------------------------------------------------


def test_estimate_acquisition_volume_reports_rows_bytes_requests_and_wall_clock(
    tmp_path, monkeypatch
):
    """The estimate is the whole cost of a fetch, priced before it starts.

    Four quantities, because no one of them is sufficient: rows say nothing
    about disk, a row-count ceiling passes a tick request that fills the volume
    anyway, and neither says how many hours the user is committing to.
    """
    catalog = _catalog(tmp_path)
    _no_network(monkeypatch)

    est = catalog.estimate_acquisition_volume(
        "us_all", *FULL_WINDOW, frequency="1d", batch_size=1
    )

    for key in (
        "symbols",
        "trading_days",
        "rows",
        "raw_bytes",
        "requests",
        "wall_clock_hours",
    ):
        assert key in est, f"{key} missing from the estimate: {sorted(est)}"
    assert est["symbols"] == US_ALL_SYMBOLS
    assert est["trading_days"] > 0
    assert est["rows"] > 0
    assert est["raw_bytes"] == est["rows"] * type(catalog).BYTES_PER_RAW_ROW
    assert est["requests"] > 0
    assert est["wall_clock_hours"] > 0


def test_daily_rows_equal_the_dense_panel_estimator_observed_cells(tmp_path):
    """The two estimators share a roster, so they must agree about it.

    `estimate_dense_panel`'s `observed_cells` already encodes the measured
    0.368 density; a DENSE count would overstate a full-market daily backfill
    by ~2.7x. Pinning the equality means a change to that density surfaces in
    both estimators rather than silently in one.
    """
    catalog = _catalog(tmp_path)

    dense = catalog.estimate_dense_panel("us_all", *FULL_WINDOW)
    volume = catalog.estimate_acquisition_volume(
        "us_all", *FULL_WINDOW, frequency="1d", batch_size=100
    )

    assert volume["rows"] == dense["observed_cells"]
    assert volume["symbols"] == dense["symbols"]
    assert volume["trading_days"] == dense["trading_days"]
    # The synthetic table reproduces the MEASURED density, so the scenario is
    # RESEARCH's row rather than an approximation of it (~15.3M rows).
    assert 0.36 < dense["density"] < 0.38
    assert 15_000_000 < volume["rows"] < 15_600_000


def test_minute_rows_are_daily_rows_times_the_documented_session_bar_count(tmp_path):
    """`1m` is `1d` x `BARS_PER_DAY_BY_FREQUENCY["1m"]`, and that 390 must
    carry the assumption it encodes.

    390 is the regular 09:30-16:00 ET session. Extended hours would make it
    ~960 -- a 2.5x error in every minute estimate. A bare `390 = 390` in the
    source is indistinguishable from a measurement, so the `#:` block is
    asserted here, not merely requested in review.
    """
    import inspect
    import re

    from acquisition.universe import UniverseCatalog

    catalog = _catalog(tmp_path)

    daily = catalog.estimate_acquisition_volume(
        "us_all", *FULL_WINDOW, frequency="1d", batch_size=100
    )
    minute = catalog.estimate_acquisition_volume(
        "us_all", *FULL_WINDOW, frequency="1m", batch_size=100
    )

    assert UniverseCatalog.BARS_PER_DAY_BY_FREQUENCY["1d"] == 1
    assert UniverseCatalog.BARS_PER_DAY_BY_FREQUENCY["1m"] == 390
    assert minute["rows"] == daily["rows"] * 390

    source = inspect.getsource(UniverseCatalog)
    block = re.search(
        r"((?:^[ \t]*#:.*\n)+)[ \t]*BARS_PER_DAY_BY_FREQUENCY",
        source,
        flags=re.MULTILINE,
    )
    assert block, "BARS_PER_DAY_BY_FREQUENCY carries no `#:` rationale block"
    rationale = block.group(1)
    assert "09:30" in rationale and "16:00" in rationale, rationale
    assert "extended" in rationale.lower(), rationale


def test_request_count_respects_both_the_page_cap_and_the_batch_floor(tmp_path):
    """`requests` is bounded BELOW by two independent things.

    Pages, because a 10,000-row response cannot carry more; and batches,
    because a fetch issues at least one request per batch even when every row
    would fit on one page. Taking only the page term understates a wide, short
    daily fetch by orders of magnitude.
    """
    import math

    catalog = _catalog(tmp_path)

    est = catalog.estimate_acquisition_volume(
        "us_all", *FULL_WINDOW, frequency="1d", batch_size=100, page_limit=10_000
    )
    assert est["requests"] >= math.ceil(est["symbols"] / 100)
    assert est["requests"] >= math.ceil(est["rows"] / 10_000)

    # A single day: the rows fit on far fewer pages than there are batches, so
    # the batch floor is what binds.
    one_day = catalog.estimate_acquisition_volume(
        "us_all", "2024-03-01", "2024-03-01", frequency="1d", batch_size=100
    )
    assert one_day["requests"] == math.ceil(one_day["symbols"] / 100)

    # Bigger batches never cost MORE requests.
    wide = catalog.estimate_acquisition_volume(
        "us_all", "2024-03-01", "2024-03-01", frequency="1d", batch_size=200
    )
    assert wide["requests"] <= one_day["requests"]


def test_wall_clock_is_requests_over_the_rate_limit_and_the_paid_tier_is_50x(tmp_path):
    """The free-vs-paid tier difference expressed as a number.

    Historical depth, fields and the recency floor are identical between the
    tiers for a backfill; the rate limit is the whole difference, and 200 ->
    10,000 per minute turns ~50 hours into ~1.
    """
    from acquisition.universe import UniverseCatalog

    catalog = _catalog(tmp_path)

    free = catalog.estimate_acquisition_volume(
        "us_all", *FULL_WINDOW, frequency="1m", batch_size=100
    )
    paid = catalog.estimate_acquisition_volume(
        "us_all", *FULL_WINDOW, frequency="1m", batch_size=100,
        rate_limit_per_min=10_000,
    )

    assert UniverseCatalog.DEFAULT_RATE_LIMIT_PER_MIN == 200
    assert free["wall_clock_hours"] == free["requests"] / 200 / 60
    assert paid["wall_clock_hours"] == paid["requests"] / 10_000 / 60
    assert free["requests"] == paid["requests"]
    assert free["wall_clock_hours"] / paid["wall_clock_hours"] == 50
    # RESEARCH Pattern 6: ~50 hours on the Basic tier, ~60 minutes paid.
    assert 45 < free["wall_clock_hours"] < 55
    assert 0.8 < paid["wall_clock_hours"] < 1.2


def test_tick_without_rows_per_symbol_day_refuses_rather_than_guessing(tmp_path):
    """T-03.2-20. Tick volume is not derivable from a calendar.

    RESEARCH flags its own ~100k trades / 10-20x quotes per symbol-day as
    order-of-magnitude only and NOT measured against Alpaca (Assumption A4).
    Encoding that as a default would make the guard confidently wrong in
    exactly the regime it exists for -- so absence is represented as absence.
    """
    import pytest

    catalog = _catalog(tmp_path)

    with pytest.raises(ValueError) as excinfo:
        catalog.estimate_acquisition_volume(
            "sp500_constituent", *ONE_YEAR, frequency="tick", batch_size=1
        )

    message = str(excinfo.value)
    assert "rows_per_symbol_day" in message
    assert "tick" in message
    assert "guess" in message.lower() or "derive" in message.lower()

    supplied = catalog.estimate_acquisition_volume(
        "sp500_constituent",
        *ONE_YEAR,
        frequency="tick",
        batch_size=1,
        rows_per_symbol_day=100_000,
    )
    assert supplied["rows"] == (
        supplied["symbols"] * supplied["trading_days"] * 100_000
    )


def test_invalid_category_and_malformed_date_raise_through_the_shared_validators(
    tmp_path,
):
    """The new estimator SHARES `_validate_category` / `_validate_iso_date`
    rather than re-validating.

    Asserted by message equality against a sibling estimator, not by exception
    type: a private re-implementation would still raise `ValueError` while
    being free to drift.
    """
    import pytest

    catalog = _catalog(tmp_path)

    with pytest.raises(ValueError) as new_category:
        catalog.estimate_acquisition_volume(
            "us_al", *FULL_WINDOW, frequency="1d", batch_size=1
        )
    with pytest.raises(ValueError) as old_category:
        catalog.estimate_dense_panel("us_al", *FULL_WINDOW)
    assert str(new_category.value) == str(old_category.value)

    with pytest.raises(ValueError) as new_date:
        catalog.estimate_acquisition_volume(
            "us_all", "01/01/2016", "2026-09-06", frequency="1d", batch_size=1
        )
    with pytest.raises(ValueError) as old_date:
        catalog.estimate_dense_panel("us_all", "01/01/2016", "2026-09-06")
    assert str(new_date.value) == str(old_date.value)


def test_degenerate_knobs_refuse_instead_of_dividing_by_zero(tmp_path):
    """A guard that raises `ZeroDivisionError` on `--batch-size 0` has failed
    at being a guard.

    Every knob reaching this method comes off a CLI or `config.kwargs`, so each
    is validated where it is consumed rather than trusted.
    """
    import pytest

    catalog = _catalog(tmp_path)

    with pytest.raises(ValueError, match="batch_size"):
        catalog.estimate_acquisition_volume(
            "us_all", *FULL_WINDOW, frequency="1d", batch_size=0
        )
    with pytest.raises(ValueError, match="page_limit"):
        catalog.estimate_acquisition_volume(
            "us_all", *FULL_WINDOW, frequency="1d", batch_size=1, page_limit=0
        )
    with pytest.raises(ValueError, match="rate_limit_per_min"):
        catalog.estimate_acquisition_volume(
            "us_all", *FULL_WINDOW, frequency="1d", batch_size=1,
            rate_limit_per_min=0,
        )
    with pytest.raises(ValueError, match="frequency"):
        catalog.estimate_acquisition_volume(
            "us_all", *FULL_WINDOW, frequency="5m", batch_size=1
        )
