"""Alpaca vendor round-trip against a mocked transport (03.2 SC-5, SC-2, D-12/D-15).

`acquisition/alpaca.py:AlpacaAcquisition` is the second vendor behind the shared
`Acquisition` base. Three things about it are easy to get wrong in ways nothing
else catches:

- **429 is `rate_limited`, never `quota`.** Tiingo's 429 means the hourly
  allocation is spent and the whole run must abort (260906-26o D-05); Alpaca's
  429 means "slow down" and must back off and retry. Classifying Alpaca's 429
  as `quota` would abort a healthy 15k-symbol run on its first burst; the
  converse would grind through 10,000 fast-failing requests, which is the
  incident that produced the Tiingo abort in the first place (SC-2).
- **Credentials come from `os.environ` in `__init__`, never from the config.**
  `alpaca-py` does NOT read them from the environment despite a docstring
  implying it does (verified by grep over the 0.44.0 sdist), so this project
  reads `APCA_API_KEY_ID` / `APCA_API_SECRET_KEY` itself and keeps them off
  every config surface and out of every log line and failure manifest.
- **`asof` is threaded explicitly, never defaulted.** Alpaca defaults it to the
  current day and maps each symbol to the entity holding it TODAY, so a
  delisted ticker silently returns the current occupant's history -- exactly
  the survivorship bias the point-in-time roster exists to remove
  (RESEARCH Pitfall 5).

Every test here is offline. The vendor transport is `mock_alpaca_client`, which
patches `acquisition.alpaca._AlpacaMarketDataClient` by dotted string. No test
sleeps for real, makes a network call, requires a real credential, or touches
any real data volume.

This file lands in 03.2-01 (Wave 0) carrying its fixture self-tests; 03.2-03 and
03.2-06 fill in the behavioural tests above. It is deliberately NOT an empty
placeholder: a pytest file with zero collected tests exits 5 ("no tests ran"),
which a later task's automated command reads as green.
"""

import os

#: The vendor's own bar field set. Single letters ON PURPOSE -- see the
#: `alpaca_bars_page` docstring in tests/conftest.py.
_VENDOR_BAR_FIELDS = {"t", "o", "h", "l", "c", "v", "n", "vw"}

#: The project's raw column names. If any of these ever appears in a fixture
#: envelope, the fixture has silently done the code-under-test's mapping job.
_PROJECT_COLUMNS = {
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "trade_count",
    "vwap",
    "symbol",
}


def test_mock_alpaca_client_supplies_obviously_fake_credentials(
    mock_alpaca_client,
):
    """Fixture self-test: both Alpaca env vars are set, and to values that
    could not be real credentials (T-03.2-11).

    Two failure modes this closes. If the fixture set no credentials, every
    Alpaca test would pass or fail depending on whether the developer running
    it happens to have real keys exported. If it read the ambient values
    instead of overwriting them, a real secret could be captured into a test
    artefact -- a log line, a failure manifest, a CI transcript.
    """
    key_id = os.environ["APCA_API_KEY_ID"]
    secret = os.environ["APCA_API_SECRET_KEY"]

    assert key_id and secret
    assert key_id != secret
    for name, value in (("APCA_API_KEY_ID", key_id), ("APCA_API_SECRET_KEY", secret)):
        assert "not-real" in value, (
            f"{name} must be an obviously fake value so a real exported "
            f"credential can never be picked up by a test; got {value!r}"
        )


def test_alpaca_bars_page_emits_the_vendor_field_names_not_the_project_ones(
    alpaca_bars_page,
):
    """Fixture self-test: the envelope carries `t/o/h/l/c/v/n/vw`, not
    `timestamp/open/high/low/close/volume/trade_count/vwap`.

    Mapping the vendor's letters onto the project's names is the single most
    likely place for `AlpacaAcquisition` to be quietly wrong (a swapped `o`/`c`
    reads as plausible data forever). If the fixture pre-mapped them, that
    mapping would never be exercised and the bug would be untestable.
    """
    envelope = alpaca_bars_page(
        {"AAPL": ["2024-01-02T00:00:00Z"]},
        next_page_token=None,
    )

    assert set(envelope) == {"bars", "next_page_token", "currency"}
    assert envelope["currency"] == "USD"
    assert envelope["next_page_token"] is None

    (bar,) = envelope["bars"]["AAPL"]
    assert set(bar) == _VENDOR_BAR_FIELDS, (
        f"expected the vendor's field set {sorted(_VENDOR_BAR_FIELDS)}, got "
        f"{sorted(bar)}"
    )
    assert not (set(bar) & _PROJECT_COLUMNS), (
        "the fixture must not pre-map vendor fields onto project column names"
    )
    assert bar["t"].endswith("Z"), "timestamps are RFC-3339 with a trailing Z"


def test_alpaca_bars_page_rejects_project_column_names(alpaca_bars_page):
    """Fixture self-test: passing a project column name raises rather than
    being written through. A fixture that accepted `close` would let a test be
    written against a shape the vendor never emits."""
    try:
        alpaca_bars_page({"AAPL": [{"t": "2024-01-02T00:00:00Z", "close": 1.0}]})
    except ValueError as exc:
        assert "close" in str(exc)
    else:  # pragma: no cover - the fixture is broken if we get here
        raise AssertionError("a non-vendor bar field was accepted silently")
