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
    from quantlab.acquisition.universe import UniverseCatalog

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

    from quantlab.acquisition.universe import UniverseCatalog
    from quantlab.base.config import UniverseConfig

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

    from quantlab.acquisition.universe import UniverseCatalog

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
    import pytest

    from quantlab.acquisition.universe import UniverseCatalog

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
    assert free["wall_clock_hours"] / paid["wall_clock_hours"] == pytest.approx(50)
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
    """The new estimator SHARES `_validate_category` / `_normalize_iso_date`
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


# ---------------------------------------------------------------------------
# Task 2 -- the refusal
#
# Every scenario below STATES the regime it is exercising before asserting the
# outcome: it calls the estimator first and asserts which ceilings the numbers
# do and do not cross. A test that only asserts `pytest.raises` would keep
# passing after the scenario drifted into a different regime, which is how a
# guard test comes to prove something other than what it is named for.
# ---------------------------------------------------------------------------

#: The three isolating scenarios, each shaped so exactly ONE ceiling is
#: crossed by the DEFAULT thresholds. Byte-only is a wide minute window; the
#: other two ride `page_limit` and `rate_limit_per_min`, which move requests
#: and wall clock independently of bytes.
BYTES_ONLY = dict(
    category="nasdaq100_constituent",
    start_date="2019-01-01",
    end_date="2025-12-31",
    frequency="1m",
    batch_size=100,
)
REQUESTS_ONLY = dict(
    category="nasdaq_all",
    start_date="2024-01-01",
    end_date="2024-12-31",
    frequency="1m",
    batch_size=100,
    # A smaller page cap multiplies requests without touching a single byte,
    # and the paid tier keeps the wall clock far under its ceiling.
    page_limit=1_000,
    rate_limit_per_min=10_000,
)
WALL_CLOCK_ONLY = dict(
    category="nasdaq_all",
    start_date="2024-01-01",
    end_date="2024-12-31",
    frequency="1m",
    batch_size=100,
    # Same request count as a normal fetch, throttled hard enough that hours
    # -- and only hours -- go over.
    rate_limit_per_min=20,
)


def _split(scenario: dict):
    """`(positional args, keyword args)` for one scenario dict."""
    scenario = dict(scenario)
    positional = (
        scenario.pop("category"),
        scenario.pop("start_date"),
        scenario.pop("end_date"),
    )
    return positional, scenario


def test_full_market_minute_backfill_is_refused_by_the_default_ceilings(
    tmp_path, monkeypatch
):
    """SC-6, and the ROADMAP scope fence made arithmetic.

    `--frequency 1m --universe us_all` over the decade is ~596,000 requests,
    ~333 GiB against ~120 GiB free, and ~50 hours on the free tier. Nothing
    stops it today; the refusal has to happen here, before a request, or it
    happens as a full disk three hours in.
    """
    import re

    import pytest

    from quantlab.acquisition.universe import UniverseCatalog

    catalog = _catalog(tmp_path)
    _no_network(monkeypatch)

    est = catalog.estimate_acquisition_volume(
        "us_all", *FULL_WINDOW, frequency="1m", batch_size=100
    )
    # RESEARCH Pattern 6's bold row, reproduced: ~597,000 requests / ~358 GB /
    # ~50 h. Asserted so the scenario cannot drift out of the regime it names.
    assert 590_000 < est["requests"] < 600_000
    assert est["raw_bytes"] > UniverseCatalog.MAX_RAW_BYTES
    assert est["requests"] > UniverseCatalog.MAX_ACQUISITION_REQUESTS
    assert est["wall_clock_hours"] > UniverseCatalog.MAX_ACQUISITION_WALL_CLOCK_HOURS

    with pytest.raises(ValueError) as excinfo:
        catalog.assert_acquisition_volume_fits(
            "us_all", *FULL_WINDOW, frequency="1m", batch_size=100
        )
    message = str(excinfo.value)

    assert f"{est['requests']:,}" in message, message  # the request count
    assert "GiB" in message, message  # the byte figure
    assert re.search(r"\d+(\.\d+)? h\b", message), message  # the wall clock
    assert "200 req/min" in message, message  # the rate limit assumed
    # ALL THREE crossings named, not merely the first. Found by mutation:
    # truncating the crossed list to its first entry left the whole suite
    # green, because the isolating scenarios each cross exactly one ceiling
    # and this is the only scenario that crosses more than one. A caller who
    # raises the single ceiling they were told about, only to hit the next one
    # on the retry, learns to distrust the message and reaches for `force`.
    for ceiling, constant in (
        ("raw-bytes", "MAX_RAW_BYTES"),
        ("request", "MAX_ACQUISITION_REQUESTS"),
        ("wall-clock", "MAX_ACQUISITION_WALL_CLOCK_HOURS"),
    ):
        assert f"{ceiling} ceiling" in message, (ceiling, message)
        assert constant in message, (constant, message)
    # A concrete narrowing computed from the estimate, with its own numbers --
    # not "narrow the window".
    assert re.search(r"<= [\d,]+ symbol", message), message
    assert re.search(r"<= [\d,]+ calendar day", message), message
    assert "~" in message
    # The override, named.
    assert "force" in message, message


def test_sp500_minute_for_one_year_is_admitted_and_returns_the_estimate(tmp_path):
    """The other half of the fence: every capability-scale fetch is ADMITTED.

    ~4,900 requests / ~3 GB / ~25 min. A guard that also refused this would be
    refusing the thing the phase exists to make possible, and would be deleted
    within a week.
    """
    catalog = _catalog(tmp_path)

    result = catalog.assert_acquisition_volume_fits(
        "sp500_constituent", *ONE_YEAR, frequency="1m", batch_size=100
    )

    assert result == catalog.estimate_acquisition_volume(
        "sp500_constituent", *ONE_YEAR, frequency="1m", batch_size=100
    )
    assert 4_500 < result["requests"] < 5_500
    assert 0.3 < result["wall_clock_hours"] < 0.5


def test_us_all_daily_backfill_the_phase_exists_to_support_is_admitted(tmp_path):
    """The single largest fetch this phase actually means to support --
    ~15.3M rows, ~1,530 requests, ~8 minutes, ~0.9 GB -- must not be blocked
    by its own guard.
    """
    catalog = _catalog(tmp_path)

    result = catalog.assert_acquisition_volume_fits(
        "us_all", *FULL_WINDOW, frequency="1d", batch_size=100
    )

    assert 1_400 < result["requests"] < 1_700
    assert result["wall_clock_hours"] < 0.2


def test_force_skips_the_raise_and_never_the_arithmetic(tmp_path):
    """The override is deliberate and visible: an explicit named parameter
    with a `False` default.

    It skips the RAISE, never the estimate -- a bypass that also skipped the
    arithmetic would leave a caller who forced through with no idea what they
    committed to (T-03.2-21).
    """
    catalog = _catalog(tmp_path)

    forced = catalog.assert_acquisition_volume_fits(
        "us_all", *FULL_WINDOW, frequency="1m", batch_size=100, force=True
    )

    assert forced == catalog.estimate_acquisition_volume(
        "us_all", *FULL_WINDOW, frequency="1m", batch_size=100
    )
    for scenario in (BYTES_ONLY, REQUESTS_ONLY, WALL_CLOCK_ONLY):
        positional, keywords = _split(scenario)
        assert catalog.assert_acquisition_volume_fits(
            *positional, force=True, **keywords
        ) == catalog.estimate_acquisition_volume(*positional, **keywords)


def test_each_of_the_three_ceilings_raises_independently(tmp_path):
    """Three ceilings, because any one alone lets a real scenario through.

    A row-count/request check passes a tick request that fills the volume; a
    byte check passes a slow, small, many-request fetch that runs overnight.
    Each scenario here crosses exactly ONE ceiling -- asserted on the estimate
    before the raise -- and the message must name which.
    """
    import pytest

    from quantlab.acquisition.universe import UniverseCatalog

    catalog = _catalog(tmp_path)
    ceilings = {
        "raw-bytes": ("raw_bytes", UniverseCatalog.MAX_RAW_BYTES, "MAX_RAW_BYTES"),
        "request": (
            "requests",
            UniverseCatalog.MAX_ACQUISITION_REQUESTS,
            "MAX_ACQUISITION_REQUESTS",
        ),
        "wall-clock": (
            "wall_clock_hours",
            UniverseCatalog.MAX_ACQUISITION_WALL_CLOCK_HOURS,
            "MAX_ACQUISITION_WALL_CLOCK_HOURS",
        ),
    }

    for crossed, scenario in (
        ("raw-bytes", BYTES_ONLY),
        ("request", REQUESTS_ONLY),
        ("wall-clock", WALL_CLOCK_ONLY),
    ):
        positional, keywords = _split(scenario)
        est = catalog.estimate_acquisition_volume(*positional, **keywords)
        for name, (key, ceiling, _constant) in ceilings.items():
            if name == crossed:
                assert est[key] > ceiling, (crossed, name, est)
            else:
                assert est[key] <= ceiling, (
                    f"{scenario} was meant to cross {crossed} alone but also "
                    f"crosses {name}: {est}"
                )

        with pytest.raises(ValueError) as excinfo:
            catalog.assert_acquisition_volume_fits(*positional, **keywords)
        message = str(excinfo.value)

        assert f"{crossed} ceiling" in message, message
        assert ceilings[crossed][2] in message, message
        for name, (_key, _ceiling, constant) in ceilings.items():
            if name == crossed:
                continue
            assert f"{name} ceiling" not in message, (name, message)
            assert constant not in message, (constant, message)


def test_every_threshold_is_overridable_by_keyword_from_config_kwargs(tmp_path):
    """CONTEXT.md Claude's Discretion: every knob must be reachable from
    `config.kwargs`, never a constructor argument no config file can reach.

    Each keyword defaults to `None` meaning "use the class constant", so one
    can be raised deliberately without disturbing the other two -- which is
    the deliberate path, as opposed to `force`.
    """
    import pytest

    catalog = _catalog(tmp_path)
    positional, keywords = _split(BYTES_ONLY)
    est = catalog.estimate_acquisition_volume(*positional, **keywords)

    # Raising ONLY the crossed ceiling admits the fetch...
    admitted = catalog.assert_acquisition_volume_fits(
        *positional, max_raw_bytes=est["raw_bytes"], **keywords
    )
    assert admitted == est

    # ...and each of the other two can be lowered into a refusal on its own.
    with pytest.raises(ValueError, match="request ceiling"):
        catalog.assert_acquisition_volume_fits(
            *positional,
            max_raw_bytes=est["raw_bytes"],
            max_requests=est["requests"] - 1,
            **keywords,
        )
    with pytest.raises(ValueError, match="wall-clock ceiling"):
        catalog.assert_acquisition_volume_fits(
            *positional,
            max_raw_bytes=est["raw_bytes"],
            max_wall_clock_hours=est["wall_clock_hours"] / 2,
            **keywords,
        )


#: Opens the structural arm's assertion message and appears in exactly one file
#: in the repository, so a failure can be attributed to THIS arm rather than to
#: a neighbouring one that would have failed anyway.
_RESOLVER_TOKEN = "FORBIDDEN-IMPORT-RESOLVED"

#: Fully qualified names of the acquisition base and every concrete vendor
#: client. Reaching any of them from the universe module means a client could
#: be constructed there.
_FORBIDDEN_ACQUISITION_MODULES = frozenset(
    {
        "quantlab.base.acquisition",
        "quantlab.acquisition.tiingo",
        "quantlab.acquisition.alpaca",
    }
)


def _resolved_imports(source_path, module_name: str) -> set[str]:
    """Every module `source_path` imports, as a fully qualified dotted name.

    Relative imports are resolved against `module_name`'s own package, which is
    the whole point: `from ..base.acquisition import X` and
    `from ..base import acquisition` name the same module as
    `import quantlab.base.acquisition` and must be seen as such. `from X import
    y` also contributes `X.y`, because `y` may itself be a submodule.
    """
    import ast

    package = module_name.rpartition(".")[0]
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package
                for _ in range(node.level - 1):
                    base = base.rpartition(".")[0]
            else:
                base = ""
            target = f"{base}.{node.module}" if base and node.module else (
                node.module or base
            )
            found.add(target)
            found.update(f"{target}.{alias.name}" for alias in node.names)
    return found


def test_the_guard_constructs_no_acquisition_client_and_needs_no_credentials(
    tmp_path, monkeypatch
):
    """SC-6's ordering claim: the refusal happens BEFORE the client exists.

    An estimate that costs a vendor request has defeated its own purpose, so
    the ordering is the acceptance criterion rather than an implementation
    detail. Proved three ways, because the weak version of this test -- "it
    did not happen to call out" -- would pass for a guard that merely runs
    first today:

    1. any socket allocation raises;
    2. every vendor credential is REMOVED from the environment, so a client
       constructed here would raise on its own missing-credential guard;
    3. structurally, `quantlab/acquisition/universe.py` imports no acquisition
       module and binds no `Acquisition` subclass -- so there is nothing here
       that COULD be constructed, whatever the call order.
    """
    import inspect
    from pathlib import Path

    import pytest

    import quantlab.acquisition.universe as universe_module
    from quantlab.base.acquisition import Acquisition

    catalog = _catalog(tmp_path)
    _no_network(monkeypatch)
    for credential in ("TIINGO_API_KEY", "APCA_API_KEY_ID", "APCA_API_SECRET_KEY"):
        monkeypatch.delenv(credential, raising=False)

    with pytest.raises(ValueError):
        catalog.assert_acquisition_volume_fits(
            "us_all", *FULL_WINDOW, frequency="1m", batch_size=100
        )
    assert catalog.assert_acquisition_volume_fits(
        "us_all", *FULL_WINDOW, frequency="1d", batch_size=100
    )["requests"] > 0

    # Structural arm. A substring scan over the source text was airtight only
    # while every package was a separate top-level root, because then an
    # acquisition module could only be named absolutely. All twelve packages
    # now share one parent, so a RELATIVE import reaches a client without
    # producing any absolute dotted literal to scan for -- and merely
    # re-prefixing the old literals makes it worse, since a prefixed literal
    # cannot match a relative spelling at all. Resolve the module's imports
    # with `ast` instead, relative levels included, and compare fully
    # qualified names.
    #
    # This arm stays AHEAD of the bound-clients scan below: pytest stops at the
    # first failing assertion, and two of the four reachable spellings would
    # also trip that scan, so whichever runs first is the one that reports.
    resolved = _resolved_imports(Path(inspect.getfile(universe_module)),
                                 universe_module.__name__)
    forbidden_hits = sorted(resolved & _FORBIDDEN_ACQUISITION_MODULES)
    assert not forbidden_hits, (
        f"{_RESOLVER_TOKEN}: quantlab/acquisition/universe.py imports "
        f"{forbidden_hits}; the volume guard must live where no acquisition "
        f"client can be constructed, whatever the call order"
    )
    bound_clients = [
        name
        for name, value in vars(universe_module).items()
        if isinstance(value, type) and issubclass(value, Acquisition)
    ]
    assert not bound_clients, bound_clients


def test_the_dense_panel_guards_are_siblings_not_replaced(tmp_path):
    """The RAM guard and the disk/request/wall-clock guard answer different
    questions and both must survive.

    `MAX_DENSE_PANEL_BYTES` bounds a dense `[timestamp, symbol]` panel in
    memory; this phase never densifies (D-18), so its constraints are disk,
    requests and hours. Collapsing either into the other would silently drop a
    real ceiling -- so all six members are pinned together here, next to the
    guard whose thresholds a careless merge would shift.
    """
    from quantlab.acquisition.universe import UniverseCatalog

    catalog = _catalog(tmp_path)

    for member in (
        "estimate_dense_panel",
        "assert_dense_panel_fits",
        "assert_chunked_panel_fits",
        "estimate_acquisition_volume",
        "assert_acquisition_volume_fits",
    ):
        assert hasattr(UniverseCatalog, member), member

    assert UniverseCatalog.MAX_DENSE_PANEL_BYTES == 4 * 1024**3
    assert UniverseCatalog.MAX_RAW_BYTES == 20 * 1024**3
    assert UniverseCatalog.MAX_ACQUISITION_REQUESTS == 50_000
    assert UniverseCatalog.MAX_ACQUISITION_WALL_CLOCK_HOURS == 4.0
    # Derived, and asserted as derived: 50,000 requests at the free tier's
    # 200/min is 4.17 h, rounded to the round number a user actually feels.
    # Pinned to the ROUNDING rather than to the exact quotient so the two
    # cannot silently decouple -- raising the request ceiling without the
    # wall-clock one has to be a visible choice.
    assert UniverseCatalog.MAX_ACQUISITION_WALL_CLOCK_HOURS == round(
        UniverseCatalog.MAX_ACQUISITION_REQUESTS
        / UniverseCatalog.DEFAULT_RATE_LIMIT_PER_MIN
        / 60
    )

    # The RAM guard still refuses the window it always refused, unchanged.
    assert catalog.assert_dense_panel_fits("us_all", "2024-01-01", "2024-01-31") is None


# ---------------------------------------------------------------------------
# 03.2-07 Task 3 -- the guard is WIRED, not merely correct
#
# The mechanism above was proved as a function. That is a different claim from
# "something calls it": through 03.2-04 and 03.2-05 this guard had NO call site
# at all, and every test in this file passed. So the tests below assert
# placement and reachability at the three entry points, by parsing their
# source. They construct no client, resolve no roster and issue no request.
# ---------------------------------------------------------------------------

INGEST_SCRIPTS = ("ingest_tiingo.py", "ingest_us_equity.py", "ingest_alpaca.py")

#: The guard's own name and the name of the helper that selects what prices the
#: fetch. A call site must name the FORMER -- the helper alone would be
#: indirection that a reader of the script cannot see the guard through.
_GUARD_NAMES = frozenset({"assert_acquisition_volume_fits"})


def _main_body(path: str):
    """The statements under `if __name__ == "__main__":`."""
    import ast

    tree = ast.parse(open(path).read())
    for node in tree.body:
        if isinstance(node, ast.If) and "__main__" in ast.unparse(node.test):
            return node.body
    raise AssertionError(f"{path} has no __main__ block")


def _call_linenos(statements, predicate) -> list[int]:
    import ast

    return sorted(
        node.lineno
        for statement in statements
        for node in ast.walk(statement)
        if isinstance(node, ast.Call) and predicate(node)
    )


def _is_guard_call(node) -> bool:
    import ast

    return isinstance(node.func, ast.Attribute) and node.func.attr in _GUARD_NAMES


def _is_acquisition_construction(node) -> bool:
    import ast

    return isinstance(node.func, ast.Name) and node.func.id.endswith("Acquisition")


def _fetch_site_linenos(body) -> list[int]:
    """Every line in a `__main__` block where a real fetch is set in motion.

    TWO forms are matched, because the three shells are mid-migration onto the
    data-source registry (03.4 D-15):

    - **direct construction** -- `acquisition = <Vendor>Acquisition(cfg)`, then
      `.download()`/`.refresh()` on that name. The BINDING is the fetch site,
      because constructing the client is where the credential is demanded.
    - **registry** -- `run(SOURCE, cfg, ...)`. No vendor class is named at all,
      so there is no `acquisition = ...` binding to find; the `run(...)` CALL
      is the fetch site, and it is still the first point a credential is
      demanded (`registry.run` constructs `descriptor.acquisition_cls` as its
      first statement).

    Matching only the first form is how this assertion would go VACUOUS: a
    shell converted to the registry has no binding, `fetching` comes back
    empty, and an ordering assertion over an empty set proves nothing. That is
    the exact failure this file's "the guard is WIRED, not merely correct"
    section exists to prevent, one migration later -- so the emptiness is
    asserted as an error at the call site rather than silently skipped.
    """
    import ast

    module = ast.Module(body=body, type_ignores=[])
    bindings = [
        statement.lineno
        for statement in ast.walk(module)
        if isinstance(statement, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "acquisition"
            for t in statement.targets
        )
    ]
    registry_runs = [
        node.lineno
        for node in ast.walk(module)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "run"
    ]
    return sorted(bindings + registry_runs)


def test_every_ingest_entry_point_calls_the_guard_by_name():
    """A guard wired into one of three doors is a guard that does not exist.

    Asserted against the guard's OWN name at each `__main__`, not against a
    shared wrapper: the ingest scripts share the pricing-view selection and the
    printing (D-14), but each names `assert_acquisition_volume_fits` itself, so
    a reader of any one script sees the refusal where the decision to fetch is
    made.
    """
    for path in INGEST_SCRIPTS:
        assert _call_linenos(_main_body(path), _is_guard_call), path


def test_the_guard_precedes_the_client_that_fetches_in_every_entry_point():
    """SC-6's ordering claim, made structural at the call site.

    The fetch site is either the client bound to `acquisition` and then sent
    `download()`/`refresh()`, or -- for a shell already reduced over the
    data-source registry (03.4 D-15) -- the `run(SOURCE, cfg)` call that
    constructs it. `_fetch_site_linenos` matches both and explains why.

    `ingest_us_equity.py` additionally constructs a client EARLIER, inside its
    `--stamp-legacy-watermarks` branch -- that path issues zero price requests
    and exits, so the guard sitting after it is correct, and this test states
    that exception by name rather than silently tolerating any construction it
    happens to find.
    """
    import ast

    for path in INGEST_SCRIPTS:
        body = _main_body(path)
        guard = min(_call_linenos(body, _is_guard_call))

        fetching = _fetch_site_linenos(body)
        assert fetching, (
            f"{path}: __main__ has neither an `acquisition = ...` binding nor "
            f"a `run(...)` call, so there is no fetch site to order the guard "
            f"against and this assertion is vacuous."
        )
        assert guard < min(fetching), (
            f"{path}: the volume guard runs at line {guard}, AFTER the client "
            f"is constructed at line {min(fetching)}. A guard that runs after "
            f"the client exists has already spent what it was meant to save."
        )

        for construction in _call_linenos(body, _is_acquisition_construction):
            if construction >= guard:
                continue
            enclosing = [
                ast.unparse(statement)
                for statement in body
                if statement.lineno <= construction
                <= (statement.end_lineno or statement.lineno)
            ]
            assert any("stamp_watermarks" in text for text in enclosing), (
                f"{path}: a client is constructed at line {construction}, "
                f"before the guard at line {guard}, and it is not the "
                f"zero-request watermark-stamping migration."
            )


def test_force_volume_is_an_explicit_flag_on_every_entry_point():
    """The override is a flag a user types. There is deliberately no
    environment variable and no config key that disables the guard wholesale
    (T-03.2-21), so a source scan for one must come up empty."""
    import argparse

    from quantlab.utils.cli import add_volume_guard_args

    parser = argparse.ArgumentParser()
    add_volume_guard_args(parser)
    assert parser.parse_args([]).force_volume is False
    assert parser.parse_args(["--force-volume"]).force_volume is True

    for path in INGEST_SCRIPTS:
        source = open(path).read()
        assert "force=args.force_volume" in source, path
        assert "add_volume_guard_args" in source, path
        for escape in ("FORCE_VOLUME", "getenv", "environ.get(\"FORCE"):
            assert escape not in source, (path, escape)


def test_the_flag_and_its_help_text_are_defined_once():
    """D-14. `--force-volume` is DEFINED in utils/cli.py and only called from
    the scripts; a per-script `add_argument("--force-volume", ...)` would be
    the third copy this phase exists to prevent."""
    import ast

    for path in INGEST_SCRIPTS:
        tree = ast.parse(open(path).read())
        registered = {
            node.args[0].value
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and node.args
            and isinstance(node.args[0], ast.Constant)
        }
        assert "--force-volume" not in registered, path
        assert "--rows-per-symbol-day" not in registered, path


def test_an_explicit_symbol_list_is_priced_rather_than_exempted(monkeypatch):
    """An explicit 15,424-symbol list is exactly as expensive as the same
    roster resolved from a category, and no catalog is loaded to price it.

    The count matters, not merely that something was priced: a view that
    ignored the list and priced ONE symbol would still 'run the guard'.
    """
    import argparse

    import pytest

    _no_network(monkeypatch)

    from quantlab.utils.cli import volume_pricing

    symbols = tuple(f"SYM{index:05d}" for index in range(US_ALL_SYMBOLS))
    args = argparse.Namespace(
        symbols=",".join(symbols),
        universe=None,
        as_of_date=None,
        start_date=FULL_WINDOW[0],
        end_date=FULL_WINDOW[1],
        limit=None,
        force_volume=False,
        rows_per_symbol_day=None,
    )

    # `catalog=None`: an explicit list needs no reference table, so this path
    # must work on a machine that has never built universe.parquet.
    pricing, category, start, end, assumed = volume_pricing(
        args, None, symbols=symbols
    )
    assert not assumed
    assert category == "(explicit --symbols list)"

    with pytest.raises(ValueError, match="Refusing to fetch"):
        pricing.assert_acquisition_volume_fits(
            category, start, end, frequency="1m", batch_size=100
        )

    admitted = pricing.assert_acquisition_volume_fits(
        category, start, end, frequency="1m", batch_size=100, force=True
    )
    assert admitted["symbols"] == US_ALL_SYMBOLS
    # Dense, not density-adjusted: a hand-named list carries none of the
    # roster's delisting structure, and assuming it does would UNDERSTATE the
    # fetch by ~2.7x -- the wrong direction for a guard.
    assert admitted["density"] == 1.0


def test_a_limited_run_is_priced_at_its_truncated_size(tmp_path, monkeypatch):
    """`--limit 10` fetches ten symbols, so the guard must price ten. Pricing
    the whole category there would refuse a smoke test that costs nothing."""
    import argparse

    _no_network(monkeypatch)

    from quantlab.utils.cli import volume_pricing

    args = argparse.Namespace(
        symbols=None,
        universe=None,
        category="us_all",
        as_of_date=None,
        start_date=FULL_WINDOW[0],
        end_date=FULL_WINDOW[1],
        limit=10,
        force_volume=False,
        rows_per_symbol_day=None,
    )
    catalog = _catalog(tmp_path)

    pricing, category, start, end, _assumed = volume_pricing(
        args, catalog, symbols=tuple(f"SYM{i:05d}" for i in range(10))
    )
    estimate = pricing.assert_acquisition_volume_fits(
        category, start, end, frequency="1d", batch_size=1
    )
    assert estimate["symbols"] == 10

    # The same window WITHOUT --limit prices the whole roster: the two must not
    # be able to produce the same number.
    args.limit = None
    whole, whole_category, w_start, w_end, _ = volume_pricing(
        args, catalog, symbols=()
    )
    assert whole is catalog
    assert whole.assert_acquisition_volume_fits(
        whole_category, w_start, w_end, frequency="1d", batch_size=1
    )["symbols"] == US_ALL_SYMBOLS


def test_an_absent_window_is_sized_against_a_STATED_assumption():
    """`ingest_tiingo.py` has no default window. Sizing needs one, so the
    helper supplies a named constant AND reports that it did -- an assumption a
    user is told about is not the silent default the house rule forbids."""
    import argparse

    from quantlab.utils.cli import UNBOUNDED_WINDOW_START, volume_pricing

    args = argparse.Namespace(
        symbols="AAPL,MSFT",
        universe=None,
        as_of_date=None,
        start_date=None,
        end_date=None,
        limit=None,
        force_volume=False,
        rows_per_symbol_day=None,
    )
    _pricing, _category, start, end, assumed = volume_pricing(
        args, None, symbols=("AAPL", "MSFT")
    )

    assert assumed is True
    assert start == UNBOUNDED_WINDOW_START
    # And it is NOT written back onto args: an assumption made to size a fetch
    # must not become the window that fetch actually requests.
    assert args.start_date is None and args.end_date is None
    assert end > start


def test_a_refusal_prints_no_estimate_and_carries_the_numbers_itself():
    """WR-07. The behaviour the old docstring and test name got backwards.

    In all three scripts the call is
    `print_volume_estimate(pricing.assert_acquisition_volume_fits(...), ...)`
    -- the guard is an ARGUMENT, so when it raises the printer is never
    invoked. Asserted twice, because either half alone is weak:

    1. structurally, that every call site really does nest the guard inside the
       printer (a future call site that separated them would change the
       answer);
    2. behaviourally, that the refusal message carries the arithmetic itself,
       which is WHY the nesting is acceptable rather than a gap.
    """
    import ast

    import pytest

    from quantlab.utils.cli import _explicit_symbol_catalog

    # 1. Structural: the guard is an argument of the printer, at every door.
    for path in INGEST_SCRIPTS:
        printers = [
            node
            for statement in _main_body(path)
            for node in ast.walk(statement)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "print_volume_estimate"
        ]
        assert printers, path
        for printer in printers:
            nested = [
                inner
                for inner in ast.walk(printer)
                if isinstance(inner, ast.Call) and _is_guard_call(inner)
            ]
            assert nested, (
                f"{path}: print_volume_estimate no longer wraps the guard. If "
                f"that was deliberate, the docstrings claiming the estimate is "
                f"printed only on admission must change with it."
            )

    # 2. Behavioural: a refusal's message carries the numbers itself.
    pricing = _explicit_symbol_catalog(15_000)
    with pytest.raises(ValueError) as excinfo:
        pricing.assert_acquisition_volume_fits(
            "(explicit)",
            "2016-01-01",
            "2026-01-01",
            frequency="1m",
            batch_size=100,
        )
    message = str(excinfo.value)
    for fragment in ("symbol(s)", "trading day(s)", "request(s)", "GiB", "h at"):
        assert fragment in message, f"{fragment!r} missing from: {message}"
    assert "ceiling" in message and "force=True" in message


def test_the_forced_line_distinguishes_overridden_from_clean():
    """What this test actually proves, now named for it (WR-07).

    It calls `print_volume_estimate` directly with a hand-built dict and never
    exercises a refusal -- so it could not fail for the reason its old name
    (`..._whether_or_not_it_refused`) asserted, and would have stayed green if
    the refusal path stopped printing anything at all. Which it already had:
    see `test_a_refusal_prints_no_estimate_and_carries_the_numbers_itself`.

    The real claim: a user who proceeds sees the numbers they proceeded with,
    and the `--force-volume` line separates 'under every ceiling' from 'over
    one and overridden', which the estimate alone cannot say.
    """
    from quantlab.utils.cli import print_volume_estimate

    lines: list[str] = []
    estimate = {
        "symbols": 500,
        "trading_days": 252,
        "rows": 126_000,
        "bars_per_day": 1,
        "density": 1.0,
        "raw_bytes": 7_560_000,
        "requests": 500,
        "batch_size": 1,
        "page_limit": 10_000,
        "wall_clock_hours": 0.04,
        "rate_limit_per_min": 200,
    }
    print_volume_estimate(
        estimate,
        category="sp500_constituent",
        start_date="2025-01-01",
        end_date="2025-12-31",
        forced=True,
        print_fn=lines.append,
    )
    rendered = "\n".join(lines)
    assert "sp500_constituent" in rendered
    assert "500" in rendered
    assert "--force-volume:    ON" in rendered

    lines.clear()
    print_volume_estimate(
        estimate,
        category="sp500_constituent",
        start_date="2025-01-01",
        end_date="2025-12-31",
        forced=False,
        print_fn=lines.append,
    )
    assert "--force-volume" not in "\n".join(lines)


#: The two RAM guards. Either one satisfies "a densification is bounded":
#: `assert_dense_panel_fits` bounds a whole-window `from_raw_data()`,
#: `assert_chunked_panel_fits` bounds the largest window of a
#: `from_raw_data_chunked()`. Neither is `assert_acquisition_volume_fits`,
#: which bounds disk/requests/wall-clock and cannot see RAM at all.
_RAM_GUARD_NAMES = frozenset(
    {"assert_dense_panel_fits", "assert_chunked_panel_fits"}
)


def test_every_entry_point_that_densifies_guards_the_dense_panels_ram():
    """CR-03. A densification with no RAM guard in front of it dies AFTER a
    successful multi-hour fetch.

    `assert_acquisition_volume_fits` bounds raw disk bytes, request count and
    wall clock -- its own docstring calls the dense-panel guards its "siblings,
    never a replacement". `ingest_alpaca.py` shipped with the new guard wired
    and NEITHER sibling, while adding a `--frequency 1m` front door: the volume
    guard's own admitted scenario (S&P-500 minute for one year, ~4,900 requests
    and ~3 GB) then reaches `from_raw_data()` and asks pandas for ~4 TB against
    a 4 GiB budget.

    Scoped by REACHABILITY rather than by script name, which is what the
    pre-existing chunked-guard test got wrong: it pinned itself to
    `ingest_us_equity.py`, so the second door's gap was invisible to it. Any
    entry point that calls `from_raw_data` / `from_raw_data_chunked` must also
    name a RAM guard, and must name it FIRST.
    """
    import ast

    densifying = 0
    for path in INGEST_SCRIPTS:
        body = _main_body(path)

        def _is_densify(node) -> bool:
            return isinstance(node.func, ast.Attribute) and node.func.attr in (
                "from_raw_data",
                "from_raw_data_chunked",
            )

        def _is_ram_guard(node) -> bool:
            return (
                isinstance(node.func, ast.Attribute)
                and node.func.attr in _RAM_GUARD_NAMES
            )

        densify_sites = _call_linenos(body, _is_densify)
        if not densify_sites:
            continue
        densifying += 1
        ram_guards = _call_linenos(body, _is_ram_guard)
        assert ram_guards, (
            f"{path} densifies at line(s) {densify_sites} with no RAM guard "
            f"in its __main__ body. assert_acquisition_volume_fits bounds "
            f"disk/requests/wall-clock and cannot see the dense panel's RAM."
        )
        assert min(ram_guards) < min(densify_sites), (
            f"{path}: the RAM guard at {ram_guards} must precede the "
            f"densification at {densify_sites} -- a guard that runs after the "
            f"allocation has already spent what it exists to save."
        )

    assert densifying >= 2, (
        "expected at least ingest_us_equity.py and ingest_alpaca.py to "
        "densify; if a door stopped densifying, say so here rather than "
        "letting this test silently cover nothing"
    )


def test_the_intraday_ram_guard_is_sized_on_the_timestamp_axis_not_the_day():
    """`assert_dense_panel_fits` sizes the TIMESTAMP axis, and at `1m` a
    session is 390 rows rather than 1.

    Left at the `bars_per_day=1` default, the guard would report ~10 GiB for a
    fetch that allocates ~4 TB and would admit the exact scenario it was added
    to refuse -- a guard that is confidently wrong in the one regime it exists
    for. Asserted on the arithmetic, so it cannot pass by the call site merely
    existing.
    """
    import pytest

    from quantlab.utils.cli import _explicit_symbol_catalog

    # 500 symbols over two calendar years -- an S&P-500-shaped minute window,
    # ~5.5 GiB dense against the 4 GiB budget, versus ~14 MB for the same
    # window at `1d`. (Note the review's own "~4 TB" figure for one year is an
    # arithmetic slip: 500 x 98,280 x 7 x 8 is ~2.75 GB, which is why this
    # asserts over a window that is unambiguously over rather than one sitting
    # on the edge of the budget.)
    window = ("2024-01-01", "2025-12-31")
    pricing = _explicit_symbol_catalog(500)
    daily = pricing.estimate_dense_panel("(explicit)", *window, bars_per_day=1)
    minute = pricing.estimate_dense_panel("(explicit)", *window, bars_per_day=390)
    assert minute["dense_bytes"] == daily["dense_bytes"] * 390
    assert minute["timestamps"] == daily["trading_days"] * 390

    # The daily window fits; the same window at minute resolution does not.
    pricing.assert_dense_panel_fits("(explicit)", *window, num_variables=7)
    with pytest.raises(ValueError, match="Refusing to densify"):
        pricing.assert_dense_panel_fits(
            "(explicit)", *window, num_variables=7, bars_per_day=390
        )


def test_the_chunked_panel_guard_keeps_both_of_its_call_sites():
    """The two guards are SIBLINGS. That one bounds RAM for a dense panel;
    this one bounds disk, request count and wall clock. Adding the second must
    not have quietly replaced the first."""
    body = _main_body("ingest_us_equity.py")

    def _is_chunked(node) -> bool:
        import ast

        return (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "assert_chunked_panel_fits"
        )

    import ast

    # Counted as CALLS in the parsed module, never as substrings: the module
    # docstring names the guard too, and a substring count would report a
    # deleted call site as present because prose mentioned it.
    tree = ast.parse(open("ingest_us_equity.py").read())
    call_sites = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "assert_chunked_panel_fits"
    ]
    assert len(call_sites) == 2, [node.lineno for node in call_sites]
    assert _call_linenos(body, _is_chunked), "the --to-zarr sizing guard is gone"
