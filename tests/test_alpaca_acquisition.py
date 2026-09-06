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

import json
import os
from pathlib import Path

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


# ---------------------------------------------------------------------------
# 03.2-02 Task 2 -- the TRACER. One symbol, one month, every layer this phase
# touches, end to end on the mocked transport.
# ---------------------------------------------------------------------------


def test_tracer_one_alpaca_daily_batch_lands_as_a_hive_shard_and_reads_back(
    mock_alpaca_client, alpaca_bars_page, acquisition_config, tmp_path
):
    """The phase's spine, proved on the thinnest real path.

    ONE Alpaca symbol over ONE month, fetched through the base class's batched
    primitive, landing as a hive-partitioned parquet shard under a
    vendor-namespaced root, with a page-ledger sidecar and a watermark sidecar
    written -- then read back through the reworked `StockDataset._scan_raw`
    with neither the `vendor` column nor the `month` hive key present.

    Four independent contracts are asserted in one test because they are one
    vertical: a shard that lands but cannot be read back is not a working
    spine, and a read that works against a hand-built tree proves nothing about
    the writer.

    No network call, no credential -- `mock_alpaca_client` patches
    `acquisition.alpaca._AlpacaMarketDataClient` and sets obviously-fake
    `APCA_*` values.
    """
    import polars as pl

    from acquisition.alpaca import AlpacaAcquisition
    from base.config import DatasetConfig
    from base.pageledger import PageLedger
    from dataset.stock import StockDataset

    # ONE single-symbol, single-page envelope: `next_page_token=None` makes it
    # the last page, so the batch completes in one request.
    mock_alpaca_client.pages = [
        alpaca_bars_page(
            {
                "AAPL": [
                    "2024-01-02T00:00:00Z",
                    "2024-01-03T00:00:00Z",
                    "2024-01-04T00:00:00Z",
                ]
            },
            next_page_token=None,
        )
    ]

    cfg = acquisition_config(vendor="alpaca", symbols=("AAPL",), frequency="1d")
    AlpacaAcquisition(cfg).download()

    # -- 1. the shard landed at the deterministic hive path -----------------
    batch_key = PageLedger.batch_key(
        "alpaca", "1d", cfg.start_date, cfg.end_date, ("AAPL",)
    )
    shard = (
        Path(cfg.raw_data_dir_path)
        / "month=2024-01"
        / f"part-{batch_key}-00000.pqt"
    )
    assert shard.exists(), sorted(
        str(p) for p in Path(cfg.raw_data_dir_path).rglob("*")
    )

    # The vendor path segment is the basename, which is what makes
    # `_scan_raw`'s isolation assertion expressible (D-11).
    assert Path(cfg.raw_data_dir_path).name == "alpaca"

    # -- 2. the page ledger recorded the batch and marked it complete -------
    ledger_path = Path(
        PageLedger.default_path(cfg.watermark_path, batch_key)
    )
    assert ledger_path.exists()
    assert ledger_path.name.endswith(".pages.json")
    payload = json.loads(ledger_path.read_text())
    assert payload["complete"] is True
    assert payload["vendor"] == "alpaca"
    assert [page["index"] for page in payload["pages"]] == [0]
    assert payload["pages"][0]["next_token"] is None
    assert payload["symbols_with_data"] == ["AAPL"]

    # The ledger lives BESIDE the raw tree, never inside it -- a `.json` under
    # the scan root would break `pl.scan_parquet` outright (D-19 contract 2).
    assert not str(ledger_path).startswith(cfg.raw_data_dir_path)

    # -- 3. the watermark sidecar was written -------------------------------
    watermark = Path(cfg.watermark_path) / "AAPL.json"
    assert watermark.exists()
    assert json.loads(watermark.read_text())["last_date"] == cfg.end_date

    # -- 4. StockDataset reads it back, with provenance asserted and dropped -
    ds_config = DatasetConfig(
        raw_data_dir_path=cfg.raw_data_dir_path,
        zarr_file_path=str(tmp_path / "tracer.zarr"),
        catalog_path=str(tmp_path / "catalog"),
        market="us_equity",
        frequency="1d",
        vendor="alpaca",
        start_date=cfg.start_date,
        end_date=cfg.end_date,
    )
    frame = StockDataset(ds_config)._scan_raw().collect()

    assert frame.height == 3
    assert set(frame.get_column("symbol").to_list()) == {"AAPL"}
    # Both the provenance column and the hive key are dropped, so the frame
    # handed to `_raw_data_to_xr_window` has exactly its pre-refactor shape.
    assert "vendor" not in frame.columns
    assert "month" not in frame.columns
    assert frame.columns == list(AlpacaAcquisition.RAW_COLUMNS[:2]) + list(
        AlpacaAcquisition.RAW_COLUMNS[3:]
    )
    # Naive datetimes, matching every other timestamp in this codebase.
    assert frame.schema["timestamp"] == pl.Datetime
    assert frame.schema["timestamp"].time_zone is None


def test_tracer_alpaca_request_pins_sort_asc_and_threads_asof_explicitly(
    mock_alpaca_client, alpaca_bars_page, acquisition_config
):
    """The two request parameters whose vendor defaults are actively harmful.

    `sort` must be pinned to `asc`: Alpaca sorts symbol-major then timestamp,
    and only with `asc` is that a total monotone order in which the ledger's
    "furthest position reached" means anything. A `desc` request inverts the
    resume semantics silently.

    `asof` must be threaded EXPLICITLY -- including as `None` -- and never left
    to the vendor's default of today, which maps each symbol onto whatever
    entity holds that ticker now and so returns the current occupant's history
    for a delisted ticker (RESEARCH Pitfall 5). That is precisely the
    survivorship bias the point-in-time roster exists to remove.

    `feed` is the opposite case: it is OMITTED when unset, because whether the
    free tier reaches historical SIP data is unresolved (D-12) and no in-code
    default may assert an answer.
    """
    from acquisition.alpaca import AlpacaAcquisition

    mock_alpaca_client.pages = [
        alpaca_bars_page({"AAPL": ["2024-01-02T00:00:00Z"]}, next_page_token=None)
    ]

    cfg = acquisition_config(vendor="alpaca", symbols=("AAPL",), frequency="1d")
    AlpacaAcquisition(cfg).download()

    (call,) = mock_alpaca_client.calls
    assert call["path"] == "/stocks/bars"
    assert call["symbols"] == "AAPL"
    assert call["timeframe"] == "1Day"
    assert call["sort"] == "asc"
    assert "asof" in call and call["asof"] is None
    assert "feed" not in call
    assert call["limit"] == 10_000
    assert call["adjustment"] == "raw"
    # No token on the first request of a fresh batch.
    assert "page_token" not in call


def test_tracer_no_credential_reaches_the_config_or_a_scrubbed_message(
    mock_alpaca_client, acquisition_config
):
    """D-15 / T-03.2-01 / T-03.2-02, asserted rather than assumed.

    `AcquisitionConfig.to_dict()` returns `asdict(self)` and lands in persisted
    configs and model-checkpoint metadata, so a credential on that surface is a
    credential committed to disk. And `_scrub` is the single choke point every
    captured vendor message passes through before it reaches a log line or the
    failure manifest.
    """
    from acquisition.alpaca import AlpacaAcquisition

    cfg = acquisition_config(vendor="alpaca", symbols=("AAPL",))
    acq = AlpacaAcquisition(cfg)

    key = os.environ["APCA_API_KEY_ID"]
    secret = os.environ["APCA_API_SECRET_KEY"]

    serialized = json.dumps(cfg.to_dict(), default=str)
    assert key not in serialized
    assert secret not in serialized

    message = f"boom while calling with {key} and {secret}"
    scrubbed = acq._scrub(message)
    assert key not in scrubbed
    assert secret not in scrubbed
    assert AlpacaAcquisition.REDACTION in scrubbed


def test_tracer_field_map_covers_the_whole_vendor_bar_and_maps_nothing_twice():
    """The vendor's single letters map onto the project's names one-to-one.

    A swapped `o`/`c` reads as plausible data forever, so the mapping is pinned
    by direct equality rather than by a round-trip that could agree with itself.
    """
    from acquisition.alpaca import AlpacaAcquisition

    assert AlpacaAcquisition.FIELD_MAP == {
        "t": "timestamp",
        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "v": "volume",
        "n": "trade_count",
        "vw": "vwap",
    }
    assert set(AlpacaAcquisition.FIELD_MAP) == _VENDOR_BAR_FIELDS
    # One vendor field per project column -- no two letters collapse onto one.
    assert len(set(AlpacaAcquisition.FIELD_MAP.values())) == len(
        AlpacaAcquisition.FIELD_MAP
    )


# ---------------------------------------------------------------------------
# 03.2-03 Task 3 -- the `_classify_error` policy seam (SC-2, D-02).
#
# The same HTTP 429 means opposite things to the two vendors, and getting it
# backwards is invisible until it costs a whole run:
#
# - reading Alpaca's 429 as `quota` aborts every Alpaca run within seconds of
#   starting, while logging a message about an allocation ceiling for a vendor
#   that publishes no allocation concept at all (RESEARCH Pitfall 1, T-03.2-16);
# - reading Tiingo's 429 as `rate_limited` grinds through ~10,000 fast-failing
#   requests, which is the observed 2026-09-06 incident.
#
# Both directions are asserted below, per vendor, so neither can regress into
# the other silently. The failing responses are built the way
# `tests/test_tiingo_quota.py` builds them -- from a real `requests.Response`
# raised through `raise_for_status()` -- so the exception CHAIN under test is
# the real one rather than a hand-rolled stand-in that would let a naive
# `getattr(exc, "response")` pass.
#
# Nothing here sleeps for real: `_sleep` is substituted and the waits are
# COUNTED.
# ---------------------------------------------------------------------------


def _http_error(status_code: int, body: str = "", reason: str = "Error"):
    """The exception a `requests`-based transport actually raises.

    `_AlpacaMarketDataClient.get_page` calls `response.raise_for_status()`, so
    what reaches `_classify_error` is a `requests.exceptions.HTTPError` whose
    `.response` carries the status.
    """
    import requests

    response = requests.Response()
    response.status_code = status_code
    response.reason = reason
    response.url = "https://data.alpaca.markets/v2/stocks/bars"
    response._content = body.encode()
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as error:
        return error
    raise AssertionError(f"status {status_code} did not raise")  # pragma: no cover


def _recording_alpaca(config, on_sleep=None):
    """An `AlpacaAcquisition` whose `_sleep` seam is substituted, so backoff is
    asserted by COUNTING waits rather than by waiting.
    """
    from acquisition.alpaca import AlpacaAcquisition

    class _Recording(AlpacaAcquisition):
        def __init__(self, cfg):
            super().__init__(cfg)
            self.sleeps: list[float] = []

        def _sleep(self, seconds: float) -> None:
            self.sleeps.append(seconds)
            if on_sleep is not None:
                on_sleep(self)

    return _Recording(config)


def test_an_alpaca_429_classifies_rate_limited_and_never_trips_the_global_abort(
    mock_alpaca_client, acquisition_config
):
    """T-03.2-16, the phase's single most expensive misclassification.

    Alpaca's 429 is a per-MINUTE ceiling (200/min on the free tier) that a
    healthy full-market run is EXPECTED to hit and that clears in under a
    minute. It must back off inside the worker and leave the shared abort
    Event alone -- setting it would stop the world over a condition that has
    already cleared by the time the log line is written.
    """
    from acquisition.alpaca import AlpacaAcquisition

    cfg = acquisition_config(vendor="alpaca", symbols=("AAPL",))
    acq = AlpacaAcquisition(cfg)

    assert acq._classify_error(_http_error(429, reason="Too Many Requests")) == (
        "rate_limited"
    )
    # Classification alone must not touch shared state.
    assert not acq._abort.is_set()

    # And the vendor has NO quota concept at all -- there is nothing on this
    # class that could ever classify a status as global.
    assert not hasattr(AlpacaAcquisition, "QUOTA_STATUS_CODES")
    assert AlpacaAcquisition.RATE_LIMIT_STATUS_CODES == frozenset({429})


def test_a_rate_limited_alpaca_batch_backs_off_retries_and_stays_out_of_the_manifest(
    mock_alpaca_client, acquisition_config
):
    """The whole `rate_limited` outcome, end to end.

    A 429 on the first request must: sleep through the substituted seam, retry
    the SAME batch, succeed, leave the global abort unset, write the watermark,
    and leave the failure manifest EMPTY. A rate limit that lands in the
    manifest would tell the next run to retry a symbol that never failed.
    """
    cfg = acquisition_config(vendor="alpaca", symbols=("AAPL",))

    mock_alpaca_client.raise_on = {0: _http_error(429, reason="Too Many Requests")}

    def clear_the_limit(_acq):
        # The per-minute window has rolled over by the time we wake up.
        mock_alpaca_client.raise_on = None

    acq = _recording_alpaca(cfg, on_sleep=clear_the_limit)
    acq.download()

    assert acq.sleeps, "a 429 must back off before retrying, not spin"
    assert not acq._abort.is_set(), (
        "a per-minute rate limit is not a global condition -- setting the "
        "abort Event stops every other batch in the run"
    )

    manifest = json.loads(
        (Path(cfg.watermark_path) / "_failures.json").read_text()
    )
    assert manifest == {}, manifest
    assert (Path(cfg.watermark_path) / "AAPL.json").exists(), (
        "the retry succeeded, so the watermark must be written"
    )


def test_the_rate_limit_backoff_is_bounded_and_degrades_to_failed(
    mock_alpaca_client, acquisition_config
):
    """T-03.2-17. An unbounded retry loop against a rate limit is a worse
    version of the problem the backoff exists to solve.

    After `rate_limit_max_retries` consecutive 429s the batch must degrade to
    `failed`, land in the manifest and get no watermark -- so the next run
    retries it rather than the current run spinning forever.
    """
    cfg = acquisition_config(
        vendor="alpaca",
        symbols=("AAPL",),
        kwargs={"rate_limit_backoff_seconds": 0.01, "rate_limit_max_retries": 2},
    )

    # Never clears.
    limited = _http_error(429, reason="Too Many Requests")
    mock_alpaca_client.raise_on = {index: limited for index in range(50)}

    acq = _recording_alpaca(cfg)
    acq.download()

    assert acq.sleeps == [0.01, 0.01], (
        f"exactly rate_limit_max_retries backoffs, then give up; got "
        f"{acq.sleeps}"
    )
    assert not acq._abort.is_set()

    manifest = json.loads(
        (Path(cfg.watermark_path) / "_failures.json").read_text()
    )
    assert set(manifest) == {"AAPL"}, manifest
    assert not (Path(cfg.watermark_path) / "AAPL.json").exists()


def test_an_alpaca_500_classifies_failed_and_isolates_to_its_batch(
    mock_alpaca_client, acquisition_config
):
    """The conservative default. Anything that is not a known transient is
    per-unit: it fails ONE batch, lands in the manifest and is retried next
    run, while every other batch completes.
    """
    from acquisition.alpaca import AlpacaAcquisition

    cfg = acquisition_config(
        vendor="alpaca",
        symbols=("AAPL", "MSFT"),
        # `raise_on` keys on the client's CALL INDEX, and the result generator
        # is `generator_unordered`, so a single worker is what ties call 0 to
        # the FIRST batch. Isolation itself does not depend on the worker
        # count -- the concurrent case is covered by the Tiingo 404 test --
        # but the assertion about WHICH symbol failed does.
        kwargs={"batch_size": 1, "max_workers": 1},
    )
    acq = AlpacaAcquisition(cfg)

    assert acq._classify_error(_http_error(500, reason="Server Error")) == "failed"

    # An exception carrying no reachable status is also `failed`, never a
    # crash: a classifier that raised would tear down the whole fan-out.
    assert acq._classify_error(RuntimeError("no status anywhere")) == "failed"
    assert acq._status_of(RuntimeError("no status anywhere")) is None

    # Batch 0 (AAPL) fails, batch 1 (MSFT) completes.
    mock_alpaca_client.raise_on = {0: _http_error(500, reason="Server Error")}
    acq.download()

    manifest = json.loads(
        (Path(cfg.watermark_path) / "_failures.json").read_text()
    )
    assert set(manifest) == {"AAPL"}, manifest
    assert not (Path(cfg.watermark_path) / "AAPL.json").exists()
    assert (Path(cfg.watermark_path) / "MSFT.json").exists()
