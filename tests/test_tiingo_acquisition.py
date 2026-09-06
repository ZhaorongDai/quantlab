import json
import os
from pathlib import Path

import pytest

from base.config import AcquisitionConfig


def _make_config(tmp_path: Path) -> AcquisitionConfig:
    return AcquisitionConfig(
        market="us_equity",
        frequency="1d",
        raw_data_dir_path=str(tmp_path / "raw"),
        watermark_path=str(tmp_path / "watermark"),
        symbols=("AAPL",),
        start_date="2024-01-01",
        end_date="2024-01-31",
    )


def test_missing_api_key_raises_before_network_call(
    monkeypatch, mock_tiingo_client, tmp_path
):
    monkeypatch.delenv("TIINGO_API_KEY", raising=False)

    from acquisition.tiingo import TiingoAcquisition

    config = _make_config(tmp_path)
    with pytest.raises(RuntimeError, match="TIINGO_API_KEY"):
        TiingoAcquisition(config)

    assert mock_tiingo_client.calls == []


def test_download_writes_parquet_and_watermark(mock_tiingo_client, tmp_path):
    from acquisition.tiingo import TiingoAcquisition
    from enums.data import TiingoColumns

    config = _make_config(tmp_path)
    acq = TiingoAcquisition(config)
    acq.download(["AAPL"])

    data_file = tmp_path / "raw" / "AAPL" / "data.pqt"
    assert data_file.exists()

    assert len(mock_tiingo_client.calls) == 1
    call = mock_tiingo_client.calls[0]
    assert call["ticker"] == "AAPL"
    assert call["columns"] == TiingoColumns.EOD
    assert call["frequency"] == "daily"
    assert call["startDate"] == "2024-01-01"
    assert call["endDate"] == "2024-01-31"

    watermark_file = tmp_path / "watermark" / "AAPL.json"
    assert watermark_file.exists()
    with open(watermark_file) as f:
        assert json.load(f)["last_date"] == "2024-01-31"


def test_refresh_uses_watermark_not_config_start_date(
    mock_tiingo_client, tmp_path
):
    from acquisition.tiingo import TiingoAcquisition

    config = _make_config(tmp_path)
    acq = TiingoAcquisition(config)

    watermark_dir = tmp_path / "watermark"
    watermark_dir.mkdir(parents=True, exist_ok=True)
    with open(watermark_dir / "AAPL.json", "w") as f:
        json.dump({"last_date": "2024-01-15"}, f)

    acq.refresh(["AAPL"])

    assert len(mock_tiingo_client.calls) == 1
    call = mock_tiingo_client.calls[0]
    assert call["startDate"] == "2024-01-15"
    assert call["startDate"] != config.start_date


def test_credential_never_exposed_on_config_surface(mock_tiingo_client, tmp_path):
    from acquisition.tiingo import TiingoAcquisition

    config = _make_config(tmp_path)
    acq = TiingoAcquisition(config)

    # Check dict *keys*, not the full serialized string -- pytest's tmp_path
    # embeds the test function name in its path, which can coincidentally
    # contain one of these substrings and produce a false positive.
    keys = set(config.to_dict().keys())
    for forbidden in ("api_key", "credential", "token", "secret"):
        assert forbidden not in keys

    for attr_name, attr_value in vars(acq).items():
        if attr_name == "_client":
            continue
        assert "test-key-not-real" not in repr(attr_value)


# ---------------------------------------------------------------------------
# ConcurrentTiingoAcquisition (260906-0iy Task 2, D-03)
#
# A ~15k-symbol, multi-hour backfill makes resumability and per-symbol failure
# isolation mandatory: one delisted ticker returning a 404 must not abort the
# other 15,000, and a job killed at ticker 20,000 must resume near ticker
# 20,000 rather than re-downloading from the top.
#
# No test here makes a real network call or requires a real credential --
# every one drives the `mock_tiingo_client` fixture.
# ---------------------------------------------------------------------------

_FIVE = ("AAPL", "MSFT", "GOOG", "AMZN", "META")


def _make_concurrent_config(
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
        kwargs=kwargs if kwargs is not None else {"max_workers": 3},
    )


def test_concurrent_download_writes_every_parquet_and_watermark(
    mock_tiingo_client, tmp_path
):
    from acquisition.tiingo import ConcurrentTiingoAcquisition

    config = _make_concurrent_config(tmp_path)
    ConcurrentTiingoAcquisition(config).download()

    for symbol in _FIVE:
        assert (tmp_path / "raw" / symbol / "data.pqt").exists()
        assert (tmp_path / "watermark" / f"{symbol}.json").exists()

    assert len(mock_tiingo_client.calls) == len(_FIVE)
    assert {call["ticker"] for call in mock_tiingo_client.calls} == set(_FIVE)


def test_second_download_skips_symbols_already_at_the_watermark(
    mock_tiingo_client, tmp_path
):
    """D-03's resumability property, asserted by COUNTING VENDOR CALLS rather
    than by timing -- a job killed at ticker 20,000 and restarted must issue
    zero requests for the 20,000 already at the target watermark.
    """
    from acquisition.tiingo import ConcurrentTiingoAcquisition

    config = _make_concurrent_config(tmp_path)
    ConcurrentTiingoAcquisition(config).download()
    assert len(mock_tiingo_client.calls) == len(_FIVE)

    mock_tiingo_client.calls.clear()
    ConcurrentTiingoAcquisition(config).download()

    assert mock_tiingo_client.calls == []


def test_resume_false_re_fetches_symbols_already_at_the_watermark(
    mock_tiingo_client, tmp_path
):
    """The knob is read from `config.kwargs` -- the escape hatch
    `AcquisitionConfig` already documents -- so it stays config-driven rather
    than becoming a constructor argument no config file can reach.
    """
    from acquisition.tiingo import ConcurrentTiingoAcquisition

    ConcurrentTiingoAcquisition(_make_concurrent_config(tmp_path)).download()
    mock_tiingo_client.calls.clear()

    no_resume = _make_concurrent_config(
        tmp_path, kwargs={"max_workers": 3, "resume": False}
    )
    ConcurrentTiingoAcquisition(no_resume).download()

    assert len(mock_tiingo_client.calls) == len(_FIVE)


def _fail_one(mock_client, bad_symbol: str, message: str) -> None:
    """Make `mock_client.get_ticker_price` raise for exactly one ticker."""
    original = mock_client.get_ticker_price

    def failing(self, ticker, **kwargs):
        if ticker == bad_symbol:
            raise RuntimeError(message)
        return original(self, ticker, **kwargs)

    mock_client.get_ticker_price = failing


def test_one_symbol_failure_does_not_abort_the_others(
    mock_tiingo_client, tmp_path
):
    """One delisted ticker returning a 404 must not abort the other ~15,000.

    The failed symbol gets NO watermark, so the next run retries it.
    """
    from acquisition.tiingo import ConcurrentTiingoAcquisition

    _fail_one(mock_tiingo_client, "GOOG", "404 Not Found")

    config = _make_concurrent_config(tmp_path)
    result = ConcurrentTiingoAcquisition(config).download()

    assert result is not None  # the run returns normally, it does not raise

    for symbol in ("AAPL", "MSFT", "AMZN", "META"):
        assert (tmp_path / "raw" / symbol / "data.pqt").exists()
        assert (tmp_path / "watermark" / f"{symbol}.json").exists()

    # No watermark for the failure => the next run retries it.
    assert not (tmp_path / "watermark" / "GOOG.json").exists()

    manifest_path = tmp_path / "watermark" / "_failures.json"
    assert manifest_path.exists()
    with open(manifest_path) as f:
        failures = json.load(f)
    assert set(failures) == {"GOOG"}
    assert "404" in failures["GOOG"]


def test_failure_manifest_never_contains_the_api_key(
    mock_tiingo_client, tmp_path, capsys
):
    """T-0iy-01. This repo has already leaked one real Tiingo key, and an
    exception string is a path a credential travels that nobody audits: the
    vendor client embeds the token in the request URL it echoes back on an
    auth error.
    """
    from acquisition.tiingo import ConcurrentTiingoAcquisition

    key = os.environ["TIINGO_API_KEY"]
    _fail_one(
        mock_tiingo_client,
        "GOOG",
        f"401 Client Error for url: https://api.tiingo.com/tiingo/daily/goog/prices?token={key}",
    )

    config = _make_concurrent_config(tmp_path)
    ConcurrentTiingoAcquisition(config).download()

    manifest_text = (tmp_path / "watermark" / "_failures.json").read_text()
    assert key not in manifest_text
    assert "REDACTED" in manifest_text

    captured = capsys.readouterr()
    assert key not in captured.out
    assert key not in captured.err


def test_concurrent_refresh_starts_each_symbol_from_its_own_watermark(
    mock_tiingo_client, tmp_path
):
    from acquisition.tiingo import ConcurrentTiingoAcquisition

    config = _make_concurrent_config(tmp_path, symbols=("AAPL", "MSFT"))

    watermark_dir = tmp_path / "watermark"
    watermark_dir.mkdir(parents=True, exist_ok=True)
    with open(watermark_dir / "AAPL.json", "w") as f:
        json.dump({"last_date": "2024-01-15"}, f)
    with open(watermark_dir / "MSFT.json", "w") as f:
        json.dump({"last_date": "2024-01-20"}, f)

    ConcurrentTiingoAcquisition(config).refresh()

    starts = {
        call["ticker"]: call["startDate"] for call in mock_tiingo_client.calls
    }
    assert starts == {"AAPL": "2024-01-15", "MSFT": "2024-01-20"}
