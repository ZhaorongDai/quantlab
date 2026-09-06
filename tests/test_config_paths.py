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
    assert cfg.vendor == "tiingo"

    # `to_dict()` returns `asdict(self)`, and that dict lands in persisted
    # configs and in the JSON saved beside model checkpoints. A credential
    # field here is a credential committed to disk in an artefact nobody
    # audits -- which is how this repo leaked a real Tiingo key once already.
    # Every vendor's credentials are read from `os.environ` inside the client's
    # `__init__` (D-15, T-03.2-02).
    payload = cfg.to_dict()
    forbidden = {
        "api_key",
        "token",
        "secret",
        "tiingo_api_key",
        "apca_api_key_id",
        "apca_api_secret_key",
    }
    assert not (set(payload) & forbidden), sorted(set(payload) & forbidden)


def test_stock_config_factories_carry_the_alpaca_vendor_through_both_paths() -> None:
    """03.2 D-11: `vendor` is DERIVED into the paths by the factory rather than
    accepted pre-built, so the raw root and the watermark root cannot drift
    apart at a call site.
    """
    from pathlib import Path

    acq = stock_acquisition_config(symbols=("AAPL",), vendor="alpaca")
    ds = stock_kline_config(vendor="alpaca")

    assert acq.vendor == "alpaca"
    assert Path(acq.raw_data_dir_path).name == "alpaca"
    assert acq.raw_data_dir_path.replace("\\", "/").endswith(
        "downloads/us_equity/1d/nasdaq_data/alpaca"
    )
    assert acq.watermark_path.replace("\\", "/").endswith(
        "downloads/us_equity/1d/nasdaq_data/_watermarks/alpaca"
    )
    assert not acq.watermark_path.startswith(acq.raw_data_dir_path)

    # The dataset side records the same vendor AND terminates at it, which is
    # what makes `StockDataset._scan_raw`'s basename assertion expressible.
    assert ds.vendor == "alpaca"
    assert Path(ds.raw_data_dir_path).name == ds.vendor
    assert ds.raw_data_dir_path == acq.raw_data_dir_path

    # Two vendors resolve to SIBLING roots under one shared parent -- that is
    # the tree SC-7's isolation assertions are written against.
    tiingo = stock_acquisition_config(symbols=("AAPL",), vendor="tiingo")
    assert (
        Path(tiingo.raw_data_dir_path).parent
        == Path(acq.raw_data_dir_path).parent
    )
    assert tiingo.raw_data_dir_path != acq.raw_data_dir_path


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


def test_stock_config_defaults_are_byte_identical_without_the_new_arguments() -> None:
    """260906-0iy Task 3. The full-market roster needs its own raw-data
    subdirectory and Zarr store, but every existing call site passes neither
    argument -- so the defaults must reproduce today's paths exactly, not
    merely "something under data/us_equity/1d/".
    """
    acq = stock_acquisition_config(symbols=("AAPL",))
    ds = stock_kline_config()

    # 03.2 D-11/D-19: both paths now carry the vendor segment, and `"tiingo"`
    # is the documented incumbent -- every pre-03.2 caller of these factories
    # fetched from Tiingo, so the default names what is already on disk.
    assert acq.raw_data_dir_path.replace("\\", "/").endswith(
        "downloads/us_equity/1d/nasdaq_data/tiingo"
    )
    # The watermark root is a SIBLING of the raw root, never inside it: a
    # polars directory scan walks every file beneath the root it is given, so a
    # `.json` sidecar in the raw tree breaks `pl.scan_parquet` outright.
    assert acq.watermark_path.replace("\\", "/").endswith(
        "downloads/us_equity/1d/nasdaq_data/_watermarks/tiingo"
    )
    assert not acq.watermark_path.startswith(acq.raw_data_dir_path)
    assert ds.raw_data_dir_path == acq.raw_data_dir_path
    assert ds.zarr_file_path.replace("\\", "/").endswith(
        "data/us_equity/1d/stock.zarr"
    )


def test_stock_config_subdir_and_store_name_redirect_under_the_same_root() -> None:
    """D-04: `QUANTLAB_DATA_DIR` stays the ONLY path knob. The new arguments
    select a subdirectory/filename BENEATH the existing
    `data/{market}/{frequency}/` convention -- they are not a second root and
    they never hardcode a volume.
    """
    acq = stock_acquisition_config(symbols=("AAPL",), subdir="us_all")
    ds = stock_kline_config(subdir="us_all", store_name="us_all.zarr")

    assert acq.raw_data_dir_path.replace("\\", "/").endswith(
        "downloads/us_equity/1d/us_all/tiingo"
    )
    assert acq.watermark_path.replace("\\", "/").endswith(
        "downloads/us_equity/1d/us_all/_watermarks/tiingo"
    )
    assert ds.raw_data_dir_path == acq.raw_data_dir_path
    assert "data/us_equity/1d/" in ds.zarr_file_path.replace("\\", "/")
    assert ds.zarr_file_path.replace("\\", "/").endswith("us_all.zarr")
