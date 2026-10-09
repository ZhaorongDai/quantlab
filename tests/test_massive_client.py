"""The Massive client (#238): one day's file from S3 and the condition table over REST.

Seam 3 of spec #237, with the S3 and HTTP layers faked. What is locked here:

- a day's file is asked for under Massive's key layout and lands, byte for
  byte, at the raw tier's path; a file already there whole is not fetched
  again;
- a file is recorded only when its size is the vendor's and its gzip decodes
  to the end: a short or corrupt download leaves nothing in the raw tier;
- "not published" (404), "not entitled" (403) and a transport error are told
  apart, and a transport error is retried after a back-off;
- the condition table is pulled with the key in a header, never in a URL,
  across pages, and stored as a snapshot the conversion reads;
- a missing credential fails before any transfer, naming the variables; the
  S3 secret defaults to the API key.

Every byte and value is SYNTHETIC.
"""

from __future__ import annotations

import json
from datetime import date

import pytest

from tests.massive_fixtures import CONDITION_RECORDS, TRADE_HEADER, gzip_text

DAY = date(2024, 11, 29)
KEY = "us_stocks_sip/trades_v1/2024/11/2024-11-29.csv.gz"
BODY = gzip_text([TRADE_HEADER, "AAA,,0,12,1,1,10.0,1,1,100,1,0,0"])  # SYNTHETIC

ENV = {
    "MASSIVE_API_KEY": "synthetic-api-key",  # SYNTHETIC
    "MASSIVE_S3_ACCESS_KEY_ID": "synthetic-access-id",  # SYNTHETIC
}


class FakeStore:
    """An object store answering from ``objects``; ``errors`` are raised once per key, in order."""

    def __init__(self, objects: dict[str, bytes], *, errors: dict[str, list[Exception]] | None = None,
                 short: bool = False):
        self.objects = objects
        self.errors = errors or {}
        self.short = short
        self.reads: list[str] = []

    def _raise(self, key: str) -> None:
        if self.errors.get(key):
            raise self.errors[key].pop(0)

    def size(self, key: str) -> int:
        self._raise(key)
        if key not in self.objects:
            from quantlab.acquisition.massive.client import MassiveNotPublishedError

            raise MassiveNotPublishedError(key)
        return len(self.objects[key])

    def read(self, key: str):
        self.reads.append(key)
        self._raise(key)
        body = self.objects[key]
        if self.short:
            body = body[: len(body) // 2]
        for start in range(0, len(body), 7):
            yield body[start : start + 7]


@pytest.fixture
def env(monkeypatch):
    for name in ("MASSIVE_API_KEY", "MASSIVE_S3_ACCESS_KEY_ID", "MASSIVE_S3_SECRET_ACCESS_KEY"):
        monkeypatch.delenv(name, raising=False)
    for name, value in ENV.items():
        monkeypatch.setenv(name, value)


def _client(store=None, http_get=None, sleeps=None):
    from quantlab.acquisition.massive.client import MassiveClient

    return MassiveClient(
        store=store,
        http_get=http_get,
        sleep=(sleeps.append if sleeps is not None else (lambda seconds: None)),
    )


def test_a_days_file_lands_at_the_raw_tier_path(env, tmp_path):
    from quantlab.dataset.massive.raw import raw_days, raw_file

    store = FakeStore({KEY: BODY})
    path = _client(store).download_day("trades", DAY, tmp_path)
    assert path == raw_file(tmp_path / "massive", "trades", DAY)
    assert path.read_bytes() == BODY
    assert store.reads == [KEY]
    assert raw_days(tmp_path / "massive", "trades") == [DAY]


def test_the_minute_aggregates_key(env, tmp_path):
    key = "us_stocks_sip/minute_aggs_v1/2024/11/2024-11-29.csv.gz"
    store = FakeStore({key: BODY})
    path = _client(store).download_day("minute_aggs", DAY, tmp_path)
    assert path.parts[-3:] == ("minute_aggs", "2024", "2024-11-29.csv.gz")


def test_a_file_already_whole_is_not_fetched_again(env, tmp_path):
    store = FakeStore({KEY: BODY})
    client = _client(store)
    client.download_day("trades", DAY, tmp_path)
    client.download_day("trades", DAY, tmp_path)
    assert store.reads == [KEY]


def test_a_short_download_leaves_nothing(env, tmp_path):
    from quantlab.acquisition.massive.client import MassiveTransportError
    from quantlab.dataset.massive.raw import raw_days

    with pytest.raises(MassiveTransportError, match="bytes"):
        _client(FakeStore({KEY: BODY}, short=True)).download_day("trades", DAY, tmp_path)
    assert raw_days(tmp_path / "massive", "trades") == []
    assert not list((tmp_path / "massive").rglob("*.part"))


def test_a_corrupt_gzip_leaves_nothing(env, tmp_path):
    from quantlab.acquisition.massive.client import MassiveTransportError
    from quantlab.dataset.massive.raw import raw_days

    corrupt = BODY[:-12] + b"\x00" * 12
    with pytest.raises(MassiveTransportError, match="gzip"):
        _client(FakeStore({KEY: corrupt})).download_day("trades", DAY, tmp_path)
    assert raw_days(tmp_path / "massive", "trades") == []


def test_not_published_and_not_entitled_are_told_apart(env, tmp_path):
    from quantlab.acquisition.massive.client import (
        MassiveEntitlementError,
        MassiveNotPublishedError,
    )

    with pytest.raises(MassiveNotPublishedError):
        _client(FakeStore({})).download_day("trades", date(2024, 11, 28), tmp_path)
    store = FakeStore({KEY: BODY}, errors={KEY: [MassiveEntitlementError("403")]})
    with pytest.raises(MassiveEntitlementError):
        _client(store).download_day("trades", DAY, tmp_path)


def test_a_transport_error_is_retried_after_a_back_off(env, tmp_path):
    from quantlab.acquisition.massive.client import MassiveTransportError

    sleeps: list[float] = []
    store = FakeStore({KEY: BODY}, errors={KEY: [MassiveTransportError("reset"), MassiveTransportError("reset")]})
    path = _client(store, sleeps=sleeps).download_day("trades", DAY, tmp_path)
    assert path.read_bytes() == BODY
    assert len(sleeps) == 2 and sleeps[1] > sleeps[0]


def test_the_condition_table_is_pulled_across_pages_with_the_key_in_a_header(env, tmp_path):
    from quantlab.acquisition.massive.client import CONDITIONS_URL, HttpResponse
    from quantlab.dataset.massive.raw import latest_conditions, read_conditions

    calls = []
    next_url = "https://api.massive.com/v3/reference/conditions?cursor=SYNTHETIC"

    def http_get(url, *, params, headers):
        calls.append((url, dict(params), dict(headers)))
        if url == CONDITIONS_URL:
            body = {"status": "OK", "results": CONDITION_RECORDS[:5], "next_url": next_url}
        else:
            body = {"status": "OK", "results": CONDITION_RECORDS[5:]}
        return HttpResponse(200, json.dumps(body).encode())

    path = _client(http_get=http_get).condition_table(tmp_path)
    assert path == latest_conditions(tmp_path / "massive")
    assert json.loads(path.read_text()) == CONDITION_RECORDS
    assert [url for url, _, _ in calls] == [CONDITIONS_URL, next_url]
    assert calls[0][1] == {"asset_class": "stocks", "limit": "1000"}
    assert all(headers == {"Authorization": "Bearer synthetic-api-key"} for _, _, headers in calls)
    assert all("synthetic-api-key" not in url + json.dumps(params) for url, params, _ in calls)
    assert 37 in read_conditions(path)["id"].to_list()


def test_a_refused_key_on_the_condition_table_says_so(env, tmp_path):
    from quantlab.acquisition.massive.client import (
        HttpResponse,
        MassiveEntitlementError,
    )

    def http_get(url, *, params, headers):
        return HttpResponse(401, b'{"status":"ERROR","error":"Unknown API Key"}')

    with pytest.raises(MassiveEntitlementError, match="Unknown API Key"):
        _client(http_get=http_get).condition_table(tmp_path)


def test_a_throttled_condition_pull_is_retried(env, tmp_path):
    from quantlab.acquisition.massive.client import HttpResponse

    replies = [HttpResponse(429, b""), HttpResponse(200, json.dumps({"results": CONDITION_RECORDS}).encode())]
    sleeps: list[float] = []
    _client(http_get=lambda url, *, params, headers: replies.pop(0), sleeps=sleeps).condition_table(tmp_path)
    assert len(sleeps) == 1


def test_a_missing_credential_fails_before_any_transfer_naming_the_variables(monkeypatch):
    from quantlab.acquisition.massive.client import (
        MassiveClient,
        MassiveCredentialError,
    )

    for name in ("MASSIVE_API_KEY", "MASSIVE_S3_ACCESS_KEY_ID", "MASSIVE_S3_SECRET_ACCESS_KEY"):
        monkeypatch.delenv(name, raising=False)
    store = FakeStore({KEY: BODY})
    with pytest.raises(MassiveCredentialError, match="MASSIVE_API_KEY.*MASSIVE_S3_ACCESS_KEY_ID"):
        MassiveClient(store=store)
    monkeypatch.setenv("MASSIVE_API_KEY", "synthetic-api-key")  # SYNTHETIC
    with pytest.raises(MassiveCredentialError, match="MASSIVE_S3_ACCESS_KEY_ID") as raised:
        MassiveClient(store=store)
    assert "MASSIVE_API_KEY" not in str(raised.value).split("set")[0]
    assert store.reads == []


def test_the_s3_secret_defaults_to_the_api_key(env, monkeypatch):
    from quantlab.acquisition.massive.client import MassiveCredentials

    credentials = MassiveCredentials.from_env()
    assert credentials.s3_secret_access_key == "synthetic-api-key"
    monkeypatch.setenv("MASSIVE_S3_SECRET_ACCESS_KEY", "synthetic-secret")  # SYNTHETIC
    assert MassiveCredentials.from_env().s3_secret_access_key == "synthetic-secret"
    assert "synthetic" not in repr(credentials)
