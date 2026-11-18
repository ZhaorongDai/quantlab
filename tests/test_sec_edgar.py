"""SEC EDGAR's quarterly filing indexes: the raw-tier refresh and the readers of what it wrote.

Seam 1, ``quantlab.acquisition.sec_edgar.refresh_raw_tier`` with HTTP replaced
by ``FakeSec``; what it wrote is read back through ``quantlab.dataset.sec_edgar``.
Index lines are SYNTHETIC; the header and the line layout are VERBATIM EDGAR.
"""

from __future__ import annotations

import gzip
from datetime import UTC, date, datetime, timedelta

import pytest

from quantlab.acquisition._support.http import HttpResponse

USER_AGENT = "Synthetic research synthetic@example.invalid"  # SYNTHETIC

#: The description EDGAR writes above the header (VERBATIM layout, shortened).
PREAMBLE = (
    "Description:           Master Index of EDGAR Dissemination Feed\n"
    "Last Data Received:    March 31, 2024\n"
    "Comments:              webmaster@sec.gov\n"
    "Anonymous FTP:         ftp://ftp.sec.gov/edgar/\n"
    "\n\n\n"
    "CIK|Company Name|Form Type|Date Filed|Filename\n"
    "--------------------------------------------------------------------------------\n"
)


def index(*lines: tuple[int, str, str, str, str]) -> bytes:
    """Return a master index of ``(cik, company, form, date filed, accession)`` lines."""
    return (PREAMBLE + "".join(
        f"{cik}|{company}|{form}|{filed}|edgar/data/{cik}/{accession}.txt\n"
        for cik, company, form, filed, accession in lines
    )).encode("latin-1")


class FakeClock:
    """A UTC clock that only moves when slept on or advanced."""

    def __init__(self, start: datetime = datetime(2024, 5, 10, 14, tzinfo=UTC)):
        self.at = start
        self.sleeps: list[float] = []

    def now(self) -> datetime:
        return self.at

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.at += timedelta(seconds=seconds)

    def advance(self, **delta) -> None:
        self.at += timedelta(**delta)


class FakeSec:
    """EDGAR's full-index directory: ``indexes[(year, quarter)]`` is served, 404 otherwise; ``replies`` first."""

    def __init__(self, clock: FakeClock):
        self.clock = clock
        self.indexes: dict[tuple[int, int], bytes] = {}
        self.fallback: bytes | None = None  # served for a quarter not in ``indexes``; None is a 404
        self.replies: list[HttpResponse] = []
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, url, *, params, headers):
        self.calls.append((url, dict(headers)))
        if headers.get("User-Agent") != USER_AGENT:
            return HttpResponse(403, b"Undeclared Automated Tool")
        if self.replies:
            return self.replies.pop(0)
        year, quarter = url.split("/full-index/")[1].split("/")[:2]
        body = self.indexes.get((int(year), int(quarter.removeprefix("QTR"))), self.fallback)
        return HttpResponse(404, b"Not Found") if body is None else HttpResponse(200, body)

    def client(self, **options):
        from quantlab.acquisition.sec_edgar import SecEdgarClient

        return SecEdgarClient(
            user_agent=USER_AGENT, http_get=self, sleep=self.clock.sleep, monotonic=lambda: self.clock.now().timestamp(),
            now=self.clock.now, **options,
        )

    def requested(self) -> list[str]:
        return [url.split("/full-index/")[1] for url, _ in self.calls]


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def sec(clock):
    return FakeSec(clock)


def _refresh(tmp_path, sec, *, start=date(2024, 1, 1), end=None):
    from quantlab.acquisition.sec_edgar import refresh_raw_tier

    return refresh_raw_tier(tmp_path, client=sec.client(), start=start, end=end)


A1, A2, B1 = "0000009001-24-000001", "0000009001-24-000002", "0000009002-24-000001"


def test_a_backfill_lands_every_quarter_and_the_reader_gives_each_13f_filings_date(tmp_path, sec):
    from quantlab.dataset.sec_edgar import complete_through, filing_dates

    sec.indexes[(2024, 1)] = index(
        (9001, "MANAGER A", "13F-HR", "2024-02-14", A1),
        (9001, "MANAGER A", "10-K", "2024-02-20", "0000009001-24-000009"),
    )
    sec.indexes[(2024, 2)] = index(
        (9001, "MANAGER A", "13F-HR/A", "2024-05-09", A2),
        (9002, "MANAGER B | CO", "13F-HR", "20240508", B1),  # a '|' in the name, a date without dashes
        (9003, "CO-FILER", "13F-HR", "2024-05-08", B1),  # one filing listed under two filers
    )
    result = _refresh(tmp_path, sec)

    assert result.summary() == "2024QTR1: published; 2024QTR2: published"
    assert sec.requested() == ["2024/QTR1/master.idx", "2024/QTR2/master.idx"]
    root = tmp_path / "sec_edgar"
    # The whole index is kept as served, compressed.
    assert gzip.decompress((root / "full_index" / "2024QTR1" / "master.idx.gz").read_bytes()) == sec.indexes[(2024, 1)]
    assert filing_dates(root).rows() == [
        (A1, date(2024, 2, 14)), (A2, date(2024, 5, 9)), (B1, date(2024, 5, 8)),
    ]
    # Pulled on 2024-05-10 (UTC): every filing through 2024-05-08 is in the indexes.
    assert complete_through(root) == date(2024, 5, 8)


def test_a_later_run_pulls_only_the_current_quarter_and_drops_an_unchanged_index(tmp_path, sec, clock):
    sec.indexes[(2024, 1)] = index((9001, "MANAGER A", "13F-HR", "2024-02-14", A1))
    sec.indexes[(2024, 2)] = index((9001, "MANAGER A", "13F-HR/A", "2024-05-09", A2))
    _refresh(tmp_path, sec)
    clock.advance(days=40)  # 2024-06-19: 2024 Q1 ended 80 days ago
    before = len(sec.calls)
    stamp = (tmp_path / "sec_edgar" / "full_index" / "2024QTR2" / "master.idx.gz").stat().st_mtime_ns

    result = _refresh(tmp_path, sec)

    assert result.summary() == "2024QTR2: unchanged"
    assert sec.requested()[before:] == ["2024/QTR2/master.idx"]
    assert (tmp_path / "sec_edgar" / "full_index" / "2024QTR2" / "master.idx.gz").stat().st_mtime_ns == stamp
    from quantlab.dataset.sec_edgar import complete_through, read_watermark

    assert read_watermark(tmp_path / "sec_edgar", 2024, 2) == date(2024, 6, 19)
    assert complete_through(tmp_path / "sec_edgar") == date(2024, 6, 17)


def test_the_quarter_just_ended_is_pulled_again_for_its_first_weeks(tmp_path, sec, clock):
    from quantlab.dataset.sec_edgar import complete_through, filing_dates

    clock.at = datetime(2024, 3, 31, 12, tzinfo=UTC)
    sec.indexes[(2024, 1)] = index((9001, "MANAGER A", "13F-HR", "2024-02-14", A1))
    _refresh(tmp_path, sec)
    assert complete_through(tmp_path / "sec_edgar") == date(2024, 3, 29)

    clock.at = datetime(2024, 4, 2, 12, tzinfo=UTC)
    sec.indexes[(2024, 2)] = index()
    _refresh(tmp_path, sec)
    assert complete_through(tmp_path / "sec_edgar") == date(2024, 3, 31)

    clock.at = datetime(2024, 4, 20, 12, tzinfo=UTC)  # Q1's index looks whole, and is pulled again all the same
    sec.indexes[(2024, 1)] = index(
        (9001, "MANAGER A", "13F-HR", "2024-02-14", A1),
        (9002, "MANAGER B", "13F-HR", "2024-03-29", B1),  # SYNTHETIC: indexed late
    )
    result = _refresh(tmp_path, sec)

    assert result.summary() == "2024QTR1: published; 2024QTR2: unchanged"
    assert B1 in filing_dates(tmp_path / "sec_edgar").get_column("accession").to_list()
    assert complete_through(tmp_path / "sec_edgar") == date(2024, 4, 18)

    clock.at = datetime(2024, 5, 1, 12, tzinfo=UTC)  # 31 days after Q1's end
    before = len(sec.calls)
    _refresh(tmp_path, sec)
    assert sec.requested()[before:] == ["2024/QTR2/master.idx"]


def test_a_quarter_missing_stops_what_the_indexes_are_complete_through(tmp_path, sec):
    from quantlab.dataset.sec_edgar import complete_through

    sec.indexes[(2023, 4)] = index()
    sec.indexes[(2024, 2)] = index()
    result = _refresh(tmp_path, sec, start=date(2023, 10, 1))

    assert result.summary() == "2023QTR4: published; 2024QTR1: not on EDGAR yet; 2024QTR2: published"
    assert complete_through(tmp_path / "sec_edgar") == date(2023, 12, 31)


def test_a_403_or_5xx_is_retried_after_a_back_off_and_requests_are_spaced(tmp_path, sec, clock):
    sec.indexes[(2024, 1)] = index()
    sec.indexes[(2024, 2)] = index()
    sec.replies = [HttpResponse(403, b"Request Rate Threshold Exceeded"), HttpResponse(503, b"busy")]
    result = _refresh(tmp_path, sec)

    assert not result.failed, result.summary()
    assert sec.requested() == ["2024/QTR1/master.idx"] * 3 + ["2024/QTR2/master.idx"]
    assert clock.sleeps[:2] == [2.0, 4.0]
    assert all(wait <= 0.1 for wait in clock.sleeps[2:])


def test_a_quarter_that_keeps_failing_fails_its_part_and_the_others_still_run(tmp_path, sec):
    from quantlab.acquisition.sec_edgar import refresh_raw_tier

    sec.indexes[(2024, 2)] = index()
    sec.replies = [HttpResponse(500, b"down")] * 3
    result = refresh_raw_tier(tmp_path, client=sec.client(retries=2), start=date(2024, 1, 1))

    assert result.failed == ("2024QTR1",)
    assert result.summary().endswith("2024QTR2: published")


def test_a_body_that_is_not_a_master_index_is_not_published(tmp_path, sec):
    sec.indexes[(2024, 1)] = b"<html>maintenance</html>"
    sec.indexes[(2024, 2)] = index()
    result = _refresh(tmp_path, sec)

    assert result.failed == ("2024QTR1",)
    assert not (tmp_path / "sec_edgar" / "full_index" / "2024QTR1").exists()


def test_without_a_user_agent_nothing_is_sent(tmp_path, monkeypatch, sec):
    from quantlab.acquisition.sec_edgar import SecEdgarCredentialError, refresh_raw_tier

    monkeypatch.delenv("SEC_EDGAR_USER_AGENT", raising=False)
    with pytest.raises(SecEdgarCredentialError, match="SEC_EDGAR_USER_AGENT not set"):
        refresh_raw_tier(tmp_path, start=date(2024, 1, 1))
    assert sec.calls == []


def test_the_user_agent_is_sent_and_kept_out_of_every_message(tmp_path, sec):
    from quantlab.acquisition.sec_edgar import SecEdgarResponseError

    sec.replies = [HttpResponse(418, USER_AGENT.encode())]
    with pytest.raises(SecEdgarResponseError) as raised:
        sec.client().get_index(2024, 1)
    assert sec.calls[0][1]["User-Agent"] == USER_AGENT
    assert USER_AGENT not in str(raised.value)
    assert USER_AGENT not in repr(sec.client())


def test_the_registry_refreshes_sec_edgar_through_its_descriptor(tmp_path, monkeypatch, sec):
    import quantlab.acquisition.registry  # noqa: F401
    from quantlab.acquisition.base import DataSourceRegistry
    from quantlab.acquisition.registry import refresh_raw_tier

    source = DataSourceRegistry.get("sec_edgar")
    assert source.required_env == ("SEC_EDGAR_USER_AGENT",)
    sec.indexes[(2024, 2)] = index()
    result = refresh_raw_tier(source, tmp_path, client=sec.client(), start=date(2024, 4, 1))
    assert result.summary() == "2024QTR2: published"
