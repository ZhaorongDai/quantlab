"""The shapes of a Massive REST pull, through ``MassiveClient.pull`` and the ``http_get`` seam.

What is locked here: every shape that pages follows ``next_url`` with the key taken
out of it and keeps the key out of the logs, a condition table with a page lost
writes nothing, and an endpoint's config class cannot
state a combination its shape does not have. What each shape asks for and
keeps is locked in ``test_massive_rest.py``, ``test_massive_rest_raw_only.py``
and ``test_massive_client.py``.

Every record is SYNTHETIC.
"""

from __future__ import annotations

import json
from datetime import date

import pytest
from loguru import logger

KEY = "synthetic-api-key"  # SYNTHETIC
TODAY = date(2024, 3, 13)  # a Wednesday: yesterday is one XNYS session
HOST = "https://api.massive.com"


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("MASSIVE_API_KEY", KEY)
    monkeypatch.setenv("MASSIVE_S3_ACCESS_KEY_ID", "synthetic-access-id")


class TwoPages:
    """A vendor answering every first request with a next_url carrying the key, then a last page."""

    def __init__(self):
        self.urls: list[str] = []

    def __call__(self, url, *, params, headers):
        from quantlab.acquisition._support.http import HttpResponse

        self.urls.append(url)
        record = {"ticker": "AAA", "composite_figi": "FIGI1", "day": "2024-01-02"}  # SYNTHETIC
        page = {"results": [record]}
        if "cursor" not in url:
            page["next_url"] = f"{url}?cursor=2&apiKey={KEY}"
        return HttpResponse(200, json.dumps(page).encode())


def _client(http_get):
    from quantlab.acquisition.massive.client import MassiveClient

    return MassiveClient(store=object(), http_get=http_get, sleep=lambda seconds: None)


def _shapes():
    from quantlab.acquisition.massive.shapes import (
        AsOfEndpoint,
        ConditionTableEndpoint,
        DatedEndpoint,
        SnapshotEndpoint,
        WindowedEndpoint,
    )

    return (
        DatedEndpoint("dated", f"{HOST}/d", "day", date(2024, 1, 1)),
        WindowedEndpoint("windowed", f"{HOST}/w", "day", date(2024, 3, 1), window_days=31),
        SnapshotEndpoint("snapshot", f"{HOST}/s"),
        AsOfEndpoint("as_of", f"{HOST}/a", "date", date(2024, 3, 12)),
        ConditionTableEndpoint("conditions", f"{HOST}/c"),
    )


@pytest.mark.parametrize("index", range(5), ids=["dated", "windowed", "snapshot", "as_of", "conditions"])
def test_every_shape_follows_next_url_without_the_key_and_logs_no_key(env, tmp_path, index):
    endpoint = _shapes()[index]
    vendor = TwoPages()
    lines: list[str] = []
    sink = logger.add(lines.append, level="DEBUG")
    try:
        pull = _client(vendor).pull(endpoint, tmp_path, today=TODAY)
    finally:
        logger.remove(sink)

    assert vendor.urls == [endpoint.url, f"{endpoint.url}?cursor=2"]
    assert pull.pages == 2
    assert not any(KEY in line for line in lines)
    assert not any(KEY in path.read_text() for path in pull.files)


def test_the_condition_table_is_a_pull_of_one_file(env, tmp_path):
    from quantlab.acquisition.massive.shapes import ConditionTableEndpoint

    pull = _client(TwoPages()).pull(ConditionTableEndpoint("conditions", f"{HOST}/c"), tmp_path, today=TODAY)

    assert pull.rows == 2 and pull.through is None and len(pull.files) == 1
    assert len(json.loads(pull.files[0].read_text())) == 2


def test_the_condition_table_is_not_kept_by_a_refresh():
    from quantlab.acquisition.massive.rest import CONDITIONS, ENDPOINTS

    assert CONDITIONS.name not in ENDPOINTS


def test_a_dated_endpoint_cannot_state_a_window_or_a_snapshot_option():
    from quantlab.acquisition.massive.shapes import DatedEndpoint

    for option in ({"window_days": 31}, {"as_of_param": "date"}, {"id_field": "composite_figi"}):
        with pytest.raises(TypeError):
            DatedEndpoint("x", f"{HOST}/x", "day", date(2024, 1, 1), **option)


def test_a_snapshot_endpoint_has_no_date_field():
    from quantlab.acquisition.massive.shapes import SnapshotEndpoint

    with pytest.raises(TypeError):
        SnapshotEndpoint("x", f"{HOST}/x", date_field="day")


def test_a_windowed_endpoint_cannot_sort_or_take_a_timestamp():
    from quantlab.acquisition.massive.shapes import DatedSort, WindowedEndpoint

    for option in ({"sort": DatedSort.DATE}, {"timestamped": True}):
        with pytest.raises(TypeError):
            WindowedEndpoint("x", f"{HOST}/x", "day", date(2024, 1, 1), window_days=31, **option)


def test_a_window_is_a_positive_day_count():
    from quantlab.acquisition.massive.shapes import WindowedEndpoint

    with pytest.raises(ValueError, match="window_days"):
        WindowedEndpoint("x", f"{HOST}/x", "day", date(2024, 1, 1), window_days=0)


def test_an_id_endpoint_needs_id_in_its_url_and_an_as_of_source():
    from quantlab.acquisition.massive.shapes import AsOfEndpoint, IdsEndpoint, SnapshotEndpoint

    source = AsOfEndpoint("tickers", f"{HOST}/t", "date", date(2024, 1, 1))
    with pytest.raises(ValueError, match="{id}"):
        IdsEndpoint("x", f"{HOST}/x", source, key="composite_figi")
    with pytest.raises(ValueError, match="as-of"):
        IdsEndpoint("x", f"{HOST}/{{id}}", SnapshotEndpoint("s", f"{HOST}/s"), key="composite_figi")


def test_the_sort_is_stated_once_for_the_three_ways_massive_takes_it():
    from quantlab.acquisition.massive.shapes import DatedEndpoint, DatedSort

    def asked(sort):
        return DatedEndpoint("x", f"{HOST}/x", "day", date(2024, 1, 1), sort=sort).sort_params

    assert asked(DatedSort.DATE_THEN_TICKER) == {"sort": "day.asc,ticker.asc"}
    assert asked(DatedSort.DATE) == {"sort": "day.asc"}
    assert asked(DatedSort.ORDER_PARAM) == {"sort": "day", "order": "asc"}


@pytest.mark.parametrize("pages", [
    [{"results": None}],
    [{"results": [{"id": 1}], "next_url": f"{HOST}/c?cursor=2"}, {"results": None}],
    [{"results": [{"id": 1}], "next_url": f"{HOST}/c?cursor=2"}, {}],
], ids=["null first page", "null last page", "no results"])
def test_a_condition_table_page_without_records_writes_no_file(env, tmp_path, pages):
    from quantlab.acquisition._support.http import HttpResponse
    from quantlab.acquisition.massive.client import MassiveResponseError

    replies = iter(pages)

    def http_get(url, *, params, headers):
        return HttpResponse(200, json.dumps(next(replies)).encode())

    with pytest.raises(MassiveResponseError, match="conditions"):
        _client(http_get).condition_table(tmp_path)
    assert not list((tmp_path / "massive").rglob("*.json"))
