"""Unit tests for dataset/spot.py:SpotKlineDataset and ingest_binance_spot.py
(Phase 2 Plan 05, CONTEXT.md D-04/D-05).

Tests 1-2 cover the dedup_raw_frame() insertion into
SpotKlineDataset._raw_data_to_xr() (D-05, Task 1).
Tests 3-4 cover ingest_binance_spot.py:_build_dataset_config()'s
--raw-data-dir override behavior (Task 2).
"""

import inspect
from argparse import Namespace
from typing import Callable

import numpy as np
import xarray as xr

from quantlab.base.config import DatasetConfig
from quantlab.base.data import BaseDataset
from quantlab.dataset.spot import SpotKlineDataset


def _make_config(raw_data_dir_path: str) -> DatasetConfig:
    return DatasetConfig(
        raw_data_dir_path=raw_data_dir_path,
        zarr_file_path=str(
            __import__("pathlib").Path(raw_data_dir_path).parent / "klines.zarr"
        ),
        catalog_path=raw_data_dir_path,
        market="crypto_spot",
        frequency="1d",
    )


def test_overlapping_monthly_csvs_dedup_before_to_xarray(
    write_binance_csv: Callable[..., "__import__('pathlib').Path"],
    binance_csv_rows: list[list],
    tmp_path,
) -> None:
    """Test 1: two monthly CSV files whose date ranges overlap by one day
    (the same (timestamp, symbol) row present in both files, simulating the
    real-world Binance monthly-CSV-near-month-boundary duplicate risk) no
    longer raise `ValueError: cannot convert a DataFrame with a non-unique
    MultiIndex into xarray` -- SpotKlineDataset.from_raw_data() succeeds and
    the resulting dataset has exactly one row per overlapping
    (timestamp, symbol) pair."""
    day1, day2 = binance_csv_rows[0], binance_csv_rows[1]
    day3 = list(day2)
    day3[0] = day2[0] + 86400000  # one day after day2, still ms epoch

    # File 1 (January): day1, day2. File 2 (February): day2 (overlap), day3.
    write_binance_csv(symbol="BTCUSDT", year_month="2024-01", rows=[day1, day2])
    write_binance_csv(symbol="BTCUSDT", year_month="2024-02", rows=[day2, day3])

    config = _make_config(str(tmp_path))
    dataset = SpotKlineDataset(config)

    dataset.from_raw_data()

    xr_data = dataset.get_xarray_dataset()
    # 3 unique timestamps (day1, day2, day3), 1 symbol -- the overlapping
    # day2 row collapsed from 2 duplicate rows into exactly 1.
    assert xr_data.sizes["timestamp"] == 3
    assert xr_data.sizes["symbol"] == 1


def test_non_duplicate_csvs_convert_with_unchanged_row_count(
    write_binance_csv: Callable[..., "__import__('pathlib').Path"],
    tmp_path,
) -> None:
    """Test 2: a SpotKlineDataset built from CSVs with no duplicate rows
    converts to an xr.Dataset with the expected [timestamp, symbol] dims and
    an unchanged row count (regression -- dedup insertion must not alter
    non-duplicate data)."""
    write_binance_csv(symbol="BTCUSDT", year_month="2024-01")

    config = _make_config(str(tmp_path))
    dataset = SpotKlineDataset(config)

    dataset.from_raw_data()

    xr_data = dataset.get_xarray_dataset()
    assert xr_data.sizes["timestamp"] == 2
    assert xr_data.sizes["symbol"] == 1
    assert set(["timestamp", "symbol"]).issubset(set(xr_data.dims))


def test_build_dataset_config_raw_data_dir_none_is_noop() -> None:
    """Test 3: _build_dataset_config(args) with raw_data_dir=None returns a
    DatasetConfig whose raw_data_dir_path equals spot_kline_config()'s own
    default -- the override is a no-op when not passed."""
    from quantlab.config import spot_kline_config
    from ingest_binance_spot import _build_dataset_config

    args = Namespace(
        symbols=None, start_date=None, end_date=None, raw_data_dir=None
    )

    config = _build_dataset_config(args)
    default_config = spot_kline_config()

    assert config.raw_data_dir_path == default_config.raw_data_dir_path


def test_build_dataset_config_raw_data_dir_override_applies() -> None:
    """Test 4: _build_dataset_config(args) with raw_data_dir set returns a
    DatasetConfig whose raw_data_dir_path equals that override exactly, with
    no other field (market, frequency, zarr_file_path, catalog_path)
    altered."""
    from quantlab.config import spot_kline_config
    from ingest_binance_spot import _build_dataset_config

    args = Namespace(
        symbols=None,
        start_date=None,
        end_date=None,
        raw_data_dir="/custom/existing/csvs",
    )

    config = _build_dataset_config(args)
    default_config = spot_kline_config()

    assert config.raw_data_dir_path == "/custom/existing/csvs"
    assert config.market == default_config.market
    assert config.frequency == default_config.frequency
    assert config.zarr_file_path == default_config.zarr_file_path
    assert config.catalog_path == default_config.catalog_path


# --------------------------------------------------------------------------
# utils/nautilus.py:get_crypto_currency -- spelling and signature
# --------------------------------------------------------------------------


def test_get_crypto_currency_is_spelled_correctly() -> None:
    """`utils/nautilus.py` defined `get_crypot_currency` -- "crypot" for
    "crypto" -- right next to a correctly-spelled `get_crypto_currency_pair`,
    and `dataset/spot.py` imported the typo.
    """
    import quantlab.utils.nautilus as nautilus

    assert hasattr(nautilus, "get_crypto_currency")
    assert not hasattr(nautilus, "get_crypot_currency"), (
        "the misspelling must not survive as an alias"
    )


def test_get_crypto_currency_takes_only_a_symbol() -> None:
    """The old signature accepted `name: Optional[str] = None` and the body
    never referenced it -- `Currency.from_str(code, strict=False)` has no
    `name` argument at all, so it could not have been honoured without
    switching to the `Currency(...)` constructor and inventing a precision and
    currency type per coin.

    Dropped rather than wired up. This test pins the decision so a future edit
    has to argue with it instead of quietly re-adding an ignored parameter.
    """
    from quantlab.utils.nautilus import get_crypto_currency

    params = list(inspect.signature(get_crypto_currency).parameters)
    assert params == ["symbol"], f"unexpected signature: {params}"


def test_get_crypto_currency_returns_the_currency_for_its_code() -> None:
    """Both call sites in `dataset/spot.py` split a pair into base and quote
    and ask for each half, so the round-trip through the code is the whole
    contract."""
    from nautilus_trader.model.objects import Currency

    from quantlab.utils.nautilus import get_crypto_currency

    btc = get_crypto_currency("BTC")
    usdt = get_crypto_currency("USDT")

    assert isinstance(btc, Currency)
    assert str(btc) == "BTC"
    assert str(usdt) == "USDT"


def test_spot_windowed_seam_reproduces_the_inherited_whole_range_densify(
    write_binance_csv: Callable[..., "__import__('pathlib').Path"],
    binance_csv_rows: list[list],
    tmp_path,
) -> None:
    """D-09: `SpotKlineDataset._raw_data_to_xr_window` is the INHERITED body,
    declared explicitly so the class still satisfies `MarketDataset`'s new
    abstract seam (D-08) -- not a rewrite.

    Behaviour-preservation is the whole contract, so it is asserted twice and
    from two directions: against `BaseDataset`'s own default called unbound on
    the same instance, and against the whole-range densify sliced to the same
    window by hand. A "helpful" rewrite that returns anything else -- a
    date-pushdown that drops an untraded symbol, say -- reddens this test.
    """
    day1, day2 = binance_csv_rows[0], binance_csv_rows[1]
    # BTCUSDT trades on both days; ETHUSDT only on day 1, so the day-2 window
    # carries a symbol with no row -- the all-NaN-column case.
    write_binance_csv(symbol="BTCUSDT", year_month="2024-01", rows=[day1, day2])
    write_binance_csv(symbol="ETHUSDT", year_month="2024-01", rows=[day1])

    dataset = SpotKlineDataset(_make_config(str(tmp_path)))
    start = end = "2024-01-02"

    actual = dataset._raw_data_to_xr_window(start, end)
    inherited = BaseDataset._raw_data_to_xr_window(dataset, start, end)
    expected = dataset._raw_data_to_xr().sel(timestamp=slice(start, end))

    xr.testing.assert_identical(actual, inherited)
    xr.testing.assert_identical(actual, expected)
    assert actual.sizes["timestamp"] == 1


def test_spot_windowed_seam_keeps_a_pinned_symbol_absent_from_the_window(
    write_binance_csv: Callable[..., "__import__('pathlib').Path"],
    binance_csv_rows: list[list],
    tmp_path,
) -> None:
    """A pinned symbol with no row in this window comes back as an all-NaN
    COLUMN rather than being dropped.

    `from_raw_data_chunked()` refuses any window whose symbol axis differs
    from the pinned whole-range axis, so dropping the column would not merely
    change a value -- it would make every chunked run over crypto spot raise.
    """
    day1, day2 = binance_csv_rows[0], binance_csv_rows[1]
    write_binance_csv(symbol="BTCUSDT", year_month="2024-01", rows=[day1, day2])
    write_binance_csv(symbol="ETHUSDT", year_month="2024-01", rows=[day1])

    dataset = SpotKlineDataset(_make_config(str(tmp_path)))
    start = end = "2024-01-02"
    # SOLUSDT has no row anywhere in the raw tier, ETHUSDT none in THIS window.
    pinned = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

    windowed = dataset._raw_data_to_xr_window(start, end, symbols=pinned)
    inherited = BaseDataset._raw_data_to_xr_window(
        dataset, start, end, symbols=pinned
    )

    xr.testing.assert_identical(windowed, inherited)
    assert [
        str(symbol) for symbol in windowed["symbol"].values.tolist()
    ] == pinned
    for absent in ("ETHUSDT", "SOLUSDT"):
        assert bool(
            np.isnan(windowed["Close"].sel(symbol=absent).values).all()
        ), f"{absent} should be an all-NaN column, not a dropped one"
    assert not bool(
        np.isnan(windowed["Close"].sel(symbol="BTCUSDT").values).any()
    )


def test_spot_windowed_seam_documents_that_it_is_not_memory_bounded() -> None:
    """D-09's scope honesty lives in the code, not only in the planning tree.

    The method exists to satisfy an abstract declaration; if its docstring
    stopped saying that the implementation is unbounded and that the
    `file_date_filter` pushdown is the pending mechanism, a reader would
    reasonably conclude chunked conversion bounds memory for crypto spot. It
    does not.
    """
    doc = SpotKlineDataset._raw_data_to_xr_window.__doc__ or ""

    assert "file_date_filter" in doc
    assert "non-memory-bounded" in doc
