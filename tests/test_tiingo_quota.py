"""Global quota abort and opt-in wait/resume (260906-26o Task 2, D-05/D-06).

The observed incident, 2026-09-06: after 4,638 successes Tiingo returned an
over-allocation error. Per-symbol failure isolation -- the property that makes
one delisted ticker's 404 harmless -- then caught that error for every
remaining symbol and kept going at ~260 sym/s. Roughly 10,000 symbols were
burned as fast-failing requests that may themselves have deepened the lockout.

Quota exhaustion is a GLOBAL, RECOVERABLE condition, not one ticker's fault.
It must stop dispatch, stay OUT of the per-symbol failure manifest, and leave
every un-attempted symbol without a watermark so a re-run resumes it.

Every test here is offline. The failing vendor responses are built from real
`requests.Response` objects wrapped the way `tiingo/restclient.py:_request`
wraps them, so the exception CHAIN under test is the real one -- notably
`RestClientError` has no `.response` of its own and the status is only
reachable via `exc.args[0].response`. A hand-rolled stand-in would have let
the obvious-but-broken `getattr(exc, "response")` pass.

No test sleeps for real (the `_sleep` seam is substituted), makes a network
call, requires `TIINGO_API_KEY`, or touches any real data volume.
"""

import json
import os
from pathlib import Path

import pytest
import requests
from loguru import logger
from tiingo.restclient import RestClientError

from base.config import AcquisitionConfig

#: Deliberately far larger than anything that could be dispatched before the
#: abort trips. "Stopped early" and "ground through all of them" must not be
#: able to produce the same call count.
_MANY = tuple(f"SYM{i:04d}" for i in range(200))

_FIVE = ("AAPL", "MSFT", "GOOG", "AMZN", "META")

#: The observed body, verbatim.
_ALLOCATION_BODY = (
    "Error: You have run over your hourly request allocation. Contact us at "
    "support@tiingo.com to have these lifted."
)


def _make_config(
    tmp_path: Path,
    symbols: tuple[str, ...] = _FIVE,
    kwargs: dict | None = None,
) -> AcquisitionConfig:
    return AcquisitionConfig(
        market="us_equity",
        frequency="1d",
        raw_data_dir_path=str(tmp_path / "raw"),
        watermark_path=str(tmp_path / "watermark"),
        symbols=symbols,
        start_date="2024-01-01",
        end_date="2024-01-31",
        kwargs={"max_workers": 2, **(kwargs or {})},
    )


def _rest_client_error(status_code: int, body: str, reason: str = "Error"):
    """Build the exception `tiingo` actually raises.

    `tiingo/restclient.py:_request` calls `resp.raise_for_status()`, catches
    the `requests.exceptions.HTTPError`, and raises `RestClientError(e)`. So
    the status is reachable at `exc.args[0].response.status_code` and NOT at
    `exc.response`.
    """
    response = requests.Response()
    response.status_code = status_code
    response.reason = reason
    response.url = "https://api.tiingo.com/tiingo/daily/aaa/prices"
    response._content = body.encode()
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as error:
        return RestClientError(error)
    raise AssertionError(f"status {status_code} did not raise")  # pragma: no cover


def _captured(level: str = "WARNING"):
    """Attach a temporary in-memory loguru sink (loguru does not propagate to
    stdlib `logging`, so pytest's `caplog` sees nothing).
    """
    messages: list[str] = []
    sink_id = logger.add(messages.append, level=level, format="{message}")
    return messages, sink_id


class _Vendor:
    """Drives the patched client: raises `error` for every ticker while
    `exhausted` is True, and serves normal data otherwise.
    """

    def __init__(self, error, exhausted: bool = True):
        self.error = error
        self.exhausted = exhausted


def _install_vendor(mock_client, vendor: _Vendor) -> None:
    original = mock_client.get_ticker_price

    def maybe_failing(self, ticker, **kwargs):
        if vendor.exhausted:
            raise vendor.error
        return original(self, ticker, **kwargs)

    mock_client.get_ticker_price = maybe_failing


def _fail_one(mock_client, bad_symbol: str, error) -> None:
    original = mock_client.get_ticker_price

    def failing(self, ticker, **kwargs):
        if ticker == bad_symbol:
            raise error
        return original(self, ticker, **kwargs)

    mock_client.get_ticker_price = failing


def _acquisition_class(on_sleep=None):
    """A subclass substituting the `_sleep` seam, so wait/resume is asserted by
    COUNTING waits rather than by actually waiting an hour.
    """
    from acquisition.tiingo import ConcurrentTiingoAcquisition

    class _Recording(ConcurrentTiingoAcquisition):
        def __init__(self, config):
            super().__init__(config)
            self.sleeps: list[float] = []

        def _sleep(self, seconds: float) -> None:
            self.sleeps.append(seconds)
            if on_sleep is not None:
                on_sleep()

    return _Recording


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def test_a_429_is_classified_as_global_quota_exhaustion(
    mock_tiingo_client, tmp_path
):
    from acquisition.tiingo import ConcurrentTiingoAcquisition

    acq = ConcurrentTiingoAcquisition(_make_config(tmp_path))
    error = _rest_client_error(429, _ALLOCATION_BODY, "Too Many Requests")

    # The obvious one-liner does NOT work, which is why the helper walks args.
    assert not hasattr(error, "response")
    assert acq._is_quota_error(error)


def test_the_allocation_wording_alone_is_enough(mock_tiingo_client, tmp_path):
    """The textual signal must stand on its own, because a vendor that stops
    setting 429 must not silently turn this back into a 10,000-request burn.
    """
    from acquisition.tiingo import ConcurrentTiingoAcquisition

    acq = ConcurrentTiingoAcquisition(_make_config(tmp_path))
    # No status code anywhere -- only the wording.
    assert acq._is_quota_error(RuntimeError(_ALLOCATION_BODY))


def test_the_token_survives_the_vendor_rewording_the_period(
    mock_tiingo_client, tmp_path
):
    """`"request allocation"` rather than the full observed sentence: a
    full-string match would break the moment the vendor says "daily" instead
    of "hourly", or edits its support address.
    """
    from acquisition.tiingo import ConcurrentTiingoAcquisition

    acq = ConcurrentTiingoAcquisition(_make_config(tmp_path))
    assert acq._is_quota_error(
        RuntimeError("Error: You have run over your DAILY REQUEST ALLOCATION.")
    )


def test_a_404_is_not_quota_exhaustion(mock_tiingo_client, tmp_path):
    from acquisition.tiingo import ConcurrentTiingoAcquisition

    acq = ConcurrentTiingoAcquisition(_make_config(tmp_path))
    error = _rest_client_error(404, "Not found", "Not Found")
    assert not acq._is_quota_error(error)


def test_a_plan_restricted_403_is_not_quota_exhaustion(
    mock_tiingo_client, tmp_path
):
    """403 is deliberately EXCLUDED from the status set: Tiingo also returns
    it for a plan-restricted single ticker, which is a PER-SYMBOL condition.
    Treating it as global would let one restricted ticker abort a 15k-symbol
    run. A 403 whose body carries the wording is still caught textually, so
    the stricter status set costs nothing.
    """
    from acquisition.tiingo import ConcurrentTiingoAcquisition

    acq = ConcurrentTiingoAcquisition(_make_config(tmp_path))
    restricted = _rest_client_error(
        403, "Error: This ticker is not available on your plan.", "Forbidden"
    )
    assert not acq._is_quota_error(restricted)

    # ...but a 403 that IS about the allocation still trips it.
    assert acq._is_quota_error(
        _rest_client_error(403, _ALLOCATION_BODY, "Forbidden")
    )


def test_a_404_still_isolates_per_symbol_and_the_run_completes(
    mock_tiingo_client, tmp_path
):
    """The property this change must NOT break: one delisted ticker's 404
    still lands in the manifest and the other symbols still complete.
    """
    from acquisition.tiingo import ConcurrentTiingoAcquisition

    _fail_one(
        mock_tiingo_client, "GOOG", _rest_client_error(404, "Not found", "Not Found")
    )
    ConcurrentTiingoAcquisition(_make_config(tmp_path)).download()

    for symbol in ("AAPL", "MSFT", "AMZN", "META"):
        assert (tmp_path / "watermark" / f"{symbol}.json").exists()
    assert not (tmp_path / "watermark" / "GOOG.json").exists()

    failures = json.loads((tmp_path / "watermark" / "_failures.json").read_text())
    assert set(failures) == {"GOOG"}


# ---------------------------------------------------------------------------
# Global abort
# ---------------------------------------------------------------------------


def test_quota_exhaustion_stops_dispatch_far_short_of_the_pending_list(
    mock_tiingo_client, tmp_path
):
    """The observed incident, in miniature: ~10,000 symbols were burned as
    fast-failing requests after the allocation was gone.

    Asserted by COUNTING vendor calls against a 200-symbol pending list, so
    "stopped early" and "ground through everything" cannot look the same.
    """
    from acquisition.tiingo import ConcurrentTiingoAcquisition

    _install_vendor(
        mock_tiingo_client,
        _Vendor(_rest_client_error(429, _ALLOCATION_BODY, "Too Many Requests")),
    )

    config = _make_config(tmp_path, symbols=_MANY)
    ConcurrentTiingoAcquisition(config).download()

    # A handful of in-flight workers may already have been dispatched; the
    # point is that the other ~190 never reached the vendor at all.
    assert len(mock_tiingo_client.calls) < 20
    assert len(mock_tiingo_client.calls) < len(_MANY)


def test_un_attempted_symbols_get_no_watermark_so_a_re_run_resumes_them(
    mock_tiingo_client, tmp_path
):
    from acquisition.tiingo import ConcurrentTiingoAcquisition

    _install_vendor(
        mock_tiingo_client,
        _Vendor(_rest_client_error(429, _ALLOCATION_BODY, "Too Many Requests")),
    )
    config = _make_config(tmp_path, symbols=_MANY)
    ConcurrentTiingoAcquisition(config).download()

    written = list((tmp_path / "watermark").glob("SYM*.json"))
    assert written == []


def test_the_quota_condition_never_lands_in_the_failure_manifest(
    mock_tiingo_client, tmp_path
):
    """Recording a GLOBAL condition as one ticker's fault would defame a
    perfectly good symbol and make the manifest lie about what the last run
    did -- and the next run would "retry" a symbol that never failed.
    """
    from acquisition.tiingo import ConcurrentTiingoAcquisition

    _install_vendor(
        mock_tiingo_client,
        _Vendor(_rest_client_error(429, _ALLOCATION_BODY, "Too Many Requests")),
    )
    ConcurrentTiingoAcquisition(_make_config(tmp_path)).download()

    failures = json.loads((tmp_path / "watermark" / "_failures.json").read_text())
    assert failures == {}


def test_the_abort_reports_completed_remaining_and_watermark_preservation(
    mock_tiingo_client, tmp_path
):
    from acquisition.tiingo import ConcurrentTiingoAcquisition

    _install_vendor(
        mock_tiingo_client,
        _Vendor(_rest_client_error(429, _ALLOCATION_BODY, "Too Many Requests")),
    )

    messages, sink_id = _captured()
    try:
        ConcurrentTiingoAcquisition(_make_config(tmp_path)).download()
    finally:
        logger.remove(sink_id)

    text = "\n".join(messages).lower()
    assert "allocation" in text
    assert "remain" in text
    assert "watermark" in text


# ---------------------------------------------------------------------------
# Wait and resume (D-06)
# ---------------------------------------------------------------------------


def test_wait_for_quota_is_off_by_default_and_never_sleeps(
    mock_tiingo_client, tmp_path
):
    """OFF by default so no run silently holds an hourly window open."""
    _install_vendor(
        mock_tiingo_client,
        _Vendor(_rest_client_error(429, _ALLOCATION_BODY, "Too Many Requests")),
    )

    acq = _acquisition_class()(_make_config(tmp_path))
    acq.download()

    assert acq.sleeps == []


def test_wait_for_quota_sleeps_once_then_completes_the_remainder(
    mock_tiingo_client, tmp_path
):
    """The resume pass recomputes `pending` from the watermarks on disk, so
    the resume logic IS the skip logic -- no parallel bookkeeping to drift.
    """
    vendor = _Vendor(_rest_client_error(429, _ALLOCATION_BODY, "Too Many Requests"))
    _install_vendor(mock_tiingo_client, vendor)

    def recover():
        vendor.exhausted = False

    config = _make_config(
        tmp_path, kwargs={"wait_for_quota": True, "quota_wait_seconds": 7}
    )
    acq = _acquisition_class(on_sleep=recover)(config)
    acq.download()

    assert acq.sleeps == [7]
    for symbol in _FIVE:
        assert (tmp_path / "watermark" / f"{symbol}.json").exists()


def test_wait_for_quota_gives_up_after_the_configured_maximum(
    mock_tiingo_client, tmp_path
):
    """Bounded, because the vendor's reset semantics are not established and
    an unbounded loop against a lockout is the failure this change prevents.
    """
    _install_vendor(
        mock_tiingo_client,
        _Vendor(_rest_client_error(429, _ALLOCATION_BODY, "Too Many Requests")),
    )

    config = _make_config(
        tmp_path,
        kwargs={
            "wait_for_quota": True,
            "quota_wait_seconds": 1,
            "quota_max_waits": 2,
        },
    )
    acq = _acquisition_class()(config)
    acq.download()

    assert acq.sleeps == [1, 1]
    assert not list((tmp_path / "watermark").glob("AAPL.json"))


def test_the_defaults_are_the_documented_ones(mock_tiingo_client, tmp_path):
    from acquisition.tiingo import ConcurrentTiingoAcquisition

    assert ConcurrentTiingoAcquisition.QUOTA_STATUS_CODES == frozenset({429})
    assert ConcurrentTiingoAcquisition.DEFAULT_QUOTA_WAIT_SECONDS == 3600
    assert ConcurrentTiingoAcquisition.DEFAULT_QUOTA_MAX_WAITS == 3
    assert ConcurrentTiingoAcquisition.DEFAULT_WAIT_FOR_QUOTA is False


# ---------------------------------------------------------------------------
# Credential safety (T-26o-01)
# ---------------------------------------------------------------------------


def test_no_credential_appears_on_any_quota_path(
    mock_tiingo_client, tmp_path, capsys
):
    """This repo has already leaked one real Tiingo key. Every new path that
    captures or logs vendor text goes through `_scrub()`.
    """
    from acquisition.tiingo import ConcurrentTiingoAcquisition

    key = os.environ["TIINGO_API_KEY"]
    body = (
        f"{_ALLOCATION_BODY} (request was "
        f"https://api.tiingo.com/tiingo/daily/aaa/prices?token={key})"
    )
    _install_vendor(
        mock_tiingo_client,
        _Vendor(_rest_client_error(429, body, "Too Many Requests")),
    )

    messages, sink_id = _captured(level="DEBUG")
    try:
        ConcurrentTiingoAcquisition(_make_config(tmp_path)).download()
    finally:
        logger.remove(sink_id)

    assert key not in "\n".join(messages)
    assert key not in (tmp_path / "watermark" / "_failures.json").read_text()

    captured = capsys.readouterr()
    assert key not in captured.out
    assert key not in captured.err


def test_is_quota_error_scrubs_the_text_it_inspects(mock_tiingo_client, tmp_path):
    """Classification reads vendor text; anything it hands onward for logging
    must already be scrubbed. Proven by asserting the classifier still works
    on a body carrying the key -- i.e. scrubbing does not destroy the signal.
    """
    from acquisition.tiingo import ConcurrentTiingoAcquisition

    key = os.environ["TIINGO_API_KEY"]
    acq = ConcurrentTiingoAcquisition(_make_config(tmp_path))
    error = _rest_client_error(
        429, f"{_ALLOCATION_BODY} token={key}", "Too Many Requests"
    )
    assert acq._is_quota_error(error)


@pytest.mark.parametrize("status", [429, 403])
def test_quota_status_set_is_429_only(mock_tiingo_client, tmp_path, status):
    from acquisition.tiingo import ConcurrentTiingoAcquisition

    assert (status in ConcurrentTiingoAcquisition.QUOTA_STATUS_CODES) == (
        status == 429
    )
