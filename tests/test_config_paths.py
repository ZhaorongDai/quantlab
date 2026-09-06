"""Regression tests for the `data/{market}/{frequency}/{name}.zarr` storage
path convention (02-CONTEXT.md D-01/D-02) that every config factory in
`config/__init__.py` must follow, plus the new `stock_kline_config()` /
`stock_acquisition_config()` factories (D-09).
"""

from config import (
    alpha101_config,
    nasdaq100_constituent_config,
    sp500_constituent_config,
    spot_kline_config,
    stock_acquisition_config,
    stock_kline_config,
)


def test_spot_kline_config_uses_market_frequency_path_convention() -> None:
    cfg = spot_kline_config()

    assert "data/crypto_spot/1d/" in cfg.zarr_file_path.replace("\\", "/")
    assert cfg.market == "crypto_spot"
    assert cfg.frequency == "1d"


def test_stock_kline_config_uses_market_frequency_path_convention() -> None:
    cfg = stock_kline_config()

    assert "data/us_equity/1d/" in cfg.zarr_file_path.replace("\\", "/")
    assert cfg.market == "us_equity"
    assert cfg.frequency == "1d"


def test_stock_acquisition_config_has_no_credential_field() -> None:
    cfg = stock_acquisition_config(symbols=("AAPL",))

    assert cfg.market == "us_equity"
    assert cfg.frequency == "1d"
    assert "api_key" not in cfg.to_dict()


def test_alpha101_config_still_constructs_successfully() -> None:
    # Regression: alpha101_config()/alpha158_config()/spot_label_config() call
    # spot_kline_config(symbols=symbols) with no market/frequency override —
    # this must keep resolving via the new defaults, not raise a TypeError.
    fc = alpha101_config()

    assert fc is not None


def test_sp500_constituent_config_uses_market_frequency_path_convention() -> None:
    cfg = sp500_constituent_config()

    assert "data/us_equity/1d/" in cfg.zarr_file_path.replace("\\", "/")
    # CONFLICT 4's user-visible consequence: a membership panel is never handed
    # a nautilus catalog destination, because it has no bar representation.
    assert "catalog_path" not in cfg.to_dict()


def test_nasdaq100_constituent_config_uses_market_frequency_path_convention() -> None:
    cfg = nasdaq100_constituent_config()

    assert "data/us_equity/1d/" in cfg.zarr_file_path.replace("\\", "/")
    assert cfg.zarr_file_path.replace("\\", "/").endswith(
        "nasdaq100_constituent.zarr"
    )
    # RESEARCH Finding 6 bullet 4: two indices, two stores. Their coverage
    # starts differ by ~31 years, so a shared store would imply 1976
    # Nasdaq-100 coverage that does not exist.
    assert (
        nasdaq100_constituent_config().zarr_file_path
        != sp500_constituent_config().zarr_file_path
    )
