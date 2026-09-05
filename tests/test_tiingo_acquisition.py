import json
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
